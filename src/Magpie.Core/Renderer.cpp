#include "pch.h"
#include "Renderer.h"
#include "CommonSharedConstants.h"
#include "DesktopDuplicationFrameSource.h"
#include "DeviceResources.h"
#include "DirectXHelper.h"
#include "DwmSharedSurfaceFrameSource.h"
#include "EffectCompiler.h"
#include "EffectDrawer.h"
#include "EffectsProfiler.h"
#include "GDIFrameSource.h"
#include "GraphicsCaptureFrameSource.h"
#include "LocalizationService.h"
#include "Logger.h"
#include "OverlayDrawer.h"
#include "ScalingOptions.h"
#include "ScalingWindow.h"
#include "ScreenshotFilenameTemplateHelper.h"
#include "StrHelper.h"
#include "TextureHelper.h"
#include "Win32Helper.h"
#include "VoltaDLSSBridge.h"
#ifdef MP_USE_COMPSWAPCHAIN
#include "CompSwapchainPresenter.h"
#else
#include "AdaptivePresenter.h"
#endif
#include <d3dkmthk.h>
#include <dispatcherqueue.h>

namespace Magpie {

enum class TakeScreenshotResult {
	Success,
	InvalidDirectory,
	InvalidFilenameTemplate,
	InternalError
};

// å¤§å¤šæ•°æ—¶å€™ä¼šåœ¨æœ€åŽæ·»åŠ  Bicubic æ¥é™é‡‡æ ·æˆ–å‡é‡‡æ ·ï¼Œå› æ­¤ç¼“å­˜åœ¨å†…å­˜ä¸­
static EffectDesc bicubicDesc;

Renderer::Renderer() noexcept {}

Renderer::~Renderer() noexcept {
	_hKeyboardHook.reset();

	if (_backendThread.joinable()) {
		const HANDLE hThread = _backendThread.native_handle();

		if (!wil::handle_wait(hThread, 0)) {
			const DWORD threadId = GetThreadId(_backendThread.native_handle());

			// æŒç»­å°è¯•ç›´åˆ° _backendThread åˆ›å»ºäº†æ¶ˆæ¯é˜Ÿåˆ—
			while (!PostThreadMessage(threadId, WM_QUIT, 0, 0)) {
				if (wil::handle_wait(hThread, 1)) {
					break;
				}
			}
		}
		
		_backendThread.join();
	}
}

static void LogAdapter(IDXGIAdapter4* adapter) noexcept {
	DXGI_ADAPTER_DESC1 desc;
	adapter->GetDesc1(&desc);

	Logger::Get().Info(fmt::format("å½“å‰å›¾å½¢é€‚é…å™¨: \n\tVendorId: {:#x}\n\tDeviceId: {:#x}\n\tDescription: {}",
		desc.VendorId, desc.DeviceId, StrHelper::UTF16ToUTF8(desc.Description)));
}

static void SetGpuPriority() noexcept {
	// æ¥è‡ª https://github.com/obsproject/obs-studio/blob/16cb051a57bb357fe866252c1360ce2c38e2deec/libobs-d3d11/d3d11-subsystem.cpp#L429
	// ä¸ä½¿ç”¨ REALTIME ä¼˜å…ˆçº§ï¼Œå®ƒä¼šé€ æˆç³»ç»Ÿä¸ç¨³å®šï¼Œè€Œä¸”å¯èƒ½ä¼šå¯¼è‡´æºçª—å£å¡é¡¿ã€‚
	// OBS è¿˜è°ƒç”¨äº† SetGPUThreadPriorityï¼Œä½†è¿™ä¸ªæŽ¥å£ä¼¼ä¹Žæ— ç”¨ã€‚
	NTSTATUS status = D3DKMTSetProcessSchedulingPriorityClass(
		GetCurrentProcess(), D3DKMT_SCHEDULINGPRIORITYCLASS_HIGH);
	if (status != STATUS_SUCCESS) {
		Logger::Get().NTError("D3DKMTSetProcessSchedulingPriorityClass å¤±è´¥", status);
	}
}

ScalingError Renderer::Initialize(HWND hwndAttach, OverlayOptions& overlayOptions) noexcept {
	_backendThread = std::thread(&Renderer::_BackendThreadProc, this);

	if (!_frontendResources.Initialize(true)) {
		Logger::Get().Error("åˆå§‹åŒ–å‰ç«¯èµ„æºå¤±è´¥");
		return ScalingError::ScalingFailedGeneral;
	}

	LogAdapter(_frontendResources.GetGraphicsAdapter());

	// æ¯æ¬¡åˆ›å»º D3D è®¾å¤‡åŽå°è¯•æé«˜ GPU ä¼˜å…ˆçº§ï¼ŒOBS ä¹Ÿæ˜¯è¿™ä¹ˆåšçš„
	SetGpuPriority();

#ifdef MP_USE_COMPSWAPCHAIN
	_presenter = std::make_unique<CompSwapchainPresenter>();
	if (!_presenter->Initialize(hwndAttach, _frontendResources)) {
		Logger::Get().Error("åˆå§‹åŒ– CompSwapchainPresenter å¤±è´¥");
#else
	_presenter = std::make_unique<AdaptivePresenter>();
	if (!_presenter->Initialize(hwndAttach, _frontendResources)) {
		Logger::Get().Error("åˆå§‹åŒ– AdaptivePresenter å¤±è´¥");
#endif
		return ScalingError::ScalingFailedGeneral;
	}

	// ç­‰å¾…åŽç«¯åˆå§‹åŒ–å®Œæˆ
	_sharedTextureHandle.wait(NULL, std::memory_order_relaxed);
	const HANDLE sharedTextureHandle = _sharedTextureHandle.load(std::memory_order_acquire);
	if (sharedTextureHandle == INVALID_HANDLE_VALUE) {
		Logger::Get().Error("åŽç«¯åˆå§‹åŒ–å¤±è´¥");
		// ä¸€èˆ¬çš„é”™è¯¯ä¸ä¼šè®¾ç½® _backendInitError
		return _backendInitError == ScalingError::NoError ? ScalingError::ScalingFailedGeneral : _backendInitError;
	}

	// èŽ·å–å…±äº«çº¹ç†
	HRESULT hr = _frontendResources.GetD3DDevice()->OpenSharedResource(
		sharedTextureHandle, IID_PPV_ARGS(_frontendSharedTexture.put()));
	if (FAILED(hr)) {
		Logger::Get().ComError("OpenSharedResource å¤±è´¥", hr);
		return ScalingError::ScalingFailedGeneral;
	}

	_frontendSharedTextureMutex = _frontendSharedTexture.try_as<IDXGIKeyedMutex>();

	_UpdateDestRect();

	Logger::Get().Info(fmt::format("ç›®æ ‡çŸ©å½¢: {},{},{},{} ({}x{})",
		_destRect.left, _destRect.top, _destRect.right, _destRect.bottom,
		_destRect.right - _destRect.left, _destRect.bottom - _destRect.top));

	if (!_cursorDrawer.Initialize(_frontendResources)) {
		Logger::Get().ComError("åˆå§‹åŒ– CursorDrawer å¤±è´¥", hr);
		return ScalingError::ScalingFailedGeneral;
	}

	if (!_overlayDrawer.Initialize(_frontendResources, overlayOptions)) {
		Logger::Get().Error("åˆå§‹åŒ– OverlayDrawer å¤±è´¥");
		return ScalingError::ScalingFailedGeneral;
	}

	const ScalingOptions& options = ScalingWindow::Get().Options();
	if (!options.Is3DGameMode()) {
		_overlayDrawer.ToolbarState(options.IsWindowedMode() ?
			options.windowedInitialToolbarState : options.fullscreenInitialToolbarState);
	}

	_hKeyboardHook.reset(SetWindowsHookEx(WH_KEYBOARD_LL, _LowLevelKeyboardHook, NULL, 0));
	if (!_hKeyboardHook) {
		Logger::Get().Win32Warn("SetWindowsHookEx å¤±è´¥");
	}

	return ScalingError::NoError;
}

void Renderer::OnCursorVisibilityChanged(bool isVisible, bool onDestory) {
	_backendThreadDispatcher.TryEnqueue([this, isVisible, onDestory]() {
		if (_frameSource) {
			_frameSource->OnCursorVisibilityChanged(isVisible, onDestory);
		}
	});
}

void Renderer::MessageHandler(UINT msg, WPARAM wParam, LPARAM lParam) noexcept {
	if (!_overlayDrawer.AnyVisibleWindow()) {
		return;
	}

	_overlayDrawer.MessageHandler(msg, wParam, lParam);

	// æœ‰äº›é¼ æ ‡æ“ä½œéœ€è¦æ¸²æŸ“ ImGui å¤šæ¬¡ï¼Œè§ https://github.com/ocornut/imgui/issues/2268
	if (msg == WM_LBUTTONDOWN || msg == WM_RBUTTONDOWN || msg == WM_MOUSEWHEEL ||
		msg == WM_MOUSEHWHEEL || msg == WM_LBUTTONUP || msg == WM_RBUTTONUP) {
		_FrontendRender();
	}
}

void Renderer::StartProfile() noexcept {
	_backendThreadDispatcher.TryEnqueue([this] {
		uint32_t passCount = 0;
		for (const EffectDesc* desc : _activeEffectDescs) {
			passCount += (uint32_t)desc->passes.size();
		}
		_effectsProfiler.Start(_backendResources.GetD3DDevice(), passCount);
	});
}

void Renderer::StopProfile() noexcept {
	_backendThreadDispatcher.TryEnqueue([this] {
		_effectsProfiler.Stop();
	});
}

winrt::fire_and_forget Renderer::TakeScreenshot(
	uint32_t effectIdx,
	uint32_t passIdx,
	uint32_t outputIdx
) noexcept {
	assert(effectIdx < _activeEffectDescs.size());

	co_await _backendThreadDispatcher;

	// å¿…é¡»åœ¨åŽç«¯çº¿ç¨‹ä¿®æ”¹ _pendingScreenshotCount
	++_pendingScreenshotCount;

	// ç¡®ä¿å³ä½¿åç¨‹å‡ºäº†é—®é¢˜ä¹Ÿèƒ½æ¢å¤ _pendingScreenshotCount
	auto se = wil::scope_exit([&] {
		--_pendingScreenshotCount;
	});

	std::wstring screenshotFileName;
	TakeScreenshotResult result = (TakeScreenshotResult)co_await _TakeScreenshotImpl(
		effectIdx, passIdx, outputIdx, screenshotFileName);

	LocalizationService& ls = LocalizationService::Get();

	if (result == TakeScreenshotResult::Success) {
		winrt::hstring successMsg = ls.GetLocalizedString(L"Message_ScreenshotSaved");
		ScalingWindow::Get().ShowToast(
			fmt::format(fmt::runtime(std::wstring_view(successMsg)), screenshotFileName));
	} else {
		Logger::Get().Error("_TakeScreenshotImpl å¤±è´¥");

		const wchar_t* errorMsgs[] = {
			L"Message_ScreenshotFailed_InvalidDirectory",
			L"Message_ScreenshotFailed_InvalidFilenameTemplate",
			L"Message_ScreenshotFailed_InternalError"
		};
		ScalingWindow::Get().ShowToast(
			ls.GetLocalizedString(L"Message_ScreenshotFailed_Title"),
			ls.GetLocalizedString(errorMsgs[(size_t)result - 1])
		);
	}
}

void Renderer::_FrontendRender(bool waitForGpu) noexcept {
	winrt::com_ptr<ID3D11Texture2D> frameTex;
	winrt::com_ptr<ID3D11RenderTargetView> frameRtv;
	POINT drawOffset;
	if (!_presenter->BeginFrame(frameTex, frameRtv, drawOffset)) {
		return;
	}

	ID3D11DeviceContext4* d3dDC = _frontendResources.GetD3DDC();
	d3dDC->ClearState();

	// æ‰€æœ‰æ¸²æŸ“éƒ½ä½¿ç”¨ä¸‰è§’å½¢å¸¦æ‹“æ‰‘
	d3dDC->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

	const RECT& rendererRect = ScalingWindow::Get().RendererRect();
	if (_destRect != rendererRect) {
		// å­˜åœ¨é»‘è¾¹æ—¶åº”ä»¥é»‘è‰²å¡«å……èƒŒæ™¯ã€‚ä½¿ç”¨äº¤æ¢é“¾å‘ˆçŽ°æ—¶éœ€è¦è¿™ä¸ªæ“ä½œï¼Œå› ä¸ºæˆ‘ä»¬æŒ‡å®šäº† 
		// DXGI_SWAP_EFFECT_FLIP_DISCARDï¼ŒåŒæ—¶ä¹Ÿæ˜¯ä¸ºäº†å’Œ RTSS å…¼å®¹ã€‚ä½¿ç”¨ DirectComposition
		// å‘ˆçŽ°æ—¶ä¹Ÿéœ€è¦è¿™ä¸ªæ“ä½œï¼Œå› ä¸ºå¿…é¡»æ¸²æŸ“æ‰€æœ‰åƒç´ ã€‚
		// 
		// å½“æ¸²æŸ“åˆ° IDCompositionSurface ä¸Šæ—¶è¿™ä¸ªè°ƒç”¨å¹¶ä¸ç¬¦åˆæ ‡å‡†ï¼Œæ–‡æ¡£è¯´ä¸åº”è¯¥åœ¨æ›´æ–°çŸ©å½¢å¤–ç»˜
		// åˆ¶ã€‚ä¸è¿‡ Chromium ä¹Ÿæ˜¯è¿™ä¹ˆåšçš„ï¼Œè€Œä¸”å£°ç§°è¿™ä¸ªæ“ä½œåªä¼šå½±å“æ›´æ–°çŸ©å½¢ä¸­çš„åƒç´ ã€‚è§
		// https://github.com/chromium/chromium/blob/3653c48c3dc9ca9004f241a79238a1b3e0d0c633/gpu/command_buffer/service/shared_image/dcomp_surface_image_backing.cc#L63
		static constexpr FLOAT BLACK[4] = { 0.0f,0.0f,0.0f,1.0f };
		d3dDC->ClearRenderTargetView(frameRtv.get(), BLACK);
	}

	_lastAccessMutexKey = ++_sharedTextureMutexKey;
	HRESULT hr = _frontendSharedTextureMutex->AcquireSync(_lastAccessMutexKey - 1, INFINITE);
	if (FAILED(hr)) {
		Logger::Get().ComError("AcquireSync å¤±è´¥", hr);
		return;
	}

	{
		D3D11_TEXTURE2D_DESC desc;
		frameTex->GetDesc(&desc);
		if ((LONG)desc.Width == _destRect.right - _destRect.left
			&& (LONG)desc.Height == _destRect.bottom - _destRect.top) {
			d3dDC->CopyResource(frameTex.get(), _frontendSharedTexture.get());
		} else {
			d3dDC->CopySubresourceRegion(
				frameTex.get(),
				0,
				drawOffset.x + _destRect.left - rendererRect.left,
				drawOffset.y + _destRect.top - rendererRect.top,
				0,
				_frontendSharedTexture.get(),
				0,
				nullptr
			);
		}
	}

	_frontendSharedTextureMutex->ReleaseSync(_lastAccessMutexKey);

	// å åŠ å±‚å’Œå…‰æ ‡éƒ½ç»˜åˆ¶åˆ° back buffer
	{
		ID3D11RenderTargetView* t = frameRtv.get();
		d3dDC->OMSetRenderTargets(1, &t, nullptr);
	}

	// ç»˜åˆ¶å åŠ å±‚ã€‚ImGui è‡³å°‘æ¸²æŸ“ä¸¤éï¼Œå¦åˆ™ç»å¸¸æœ‰å¸ƒå±€é”™è¯¯
	_overlayDrawer.Draw(2, _stepTimer.GetFPS(), _effectsProfiler.GetTimings(), drawOffset);

	// ç»˜åˆ¶å…‰æ ‡
	_cursorDrawer.Draw(frameTex.get(), drawOffset);
	
	_presenter->EndFrame(waitForGpu);
}

bool Renderer::Render(bool force, bool waitForGpu) noexcept {
	if (!force && _lastAccessMutexKey == _sharedTextureMutexKey.load(std::memory_order_relaxed)) {
		if (_lastAccessMutexKey == 0) {
			// ç¬¬ä¸€å¸§å°šæœªå®Œæˆ
			return false;
		}

		if (!_cursorDrawer.NeedRedraw() && !_overlayDrawer.NeedRedraw(_stepTimer.GetFPS())) {
			return false;
		}
	}

	_FrontendRender(waitForGpu);
	return true;
}

bool Renderer::OnResize() noexcept {
	if (!_presenter->OnResize()) {
		Logger::Get().Error("æ›´æ”¹å‘ˆçŽ°å™¨å°ºå¯¸å¤±è´¥");
		return false;
	}

	_sharedTextureHandle.store(NULL, std::memory_order_relaxed);

	_backendThreadDispatcher.TryEnqueue([this]() {
		ID3D11Texture2D* outputTexture = _ResizeEffects();
		if (!outputTexture) {
			Logger::Get().Win32Error("_ResizeEffects å¤±è´¥");
			_sharedTextureHandle.store(INVALID_HANDLE_VALUE, std::memory_order_relaxed);
			_sharedTextureHandle.notify_one();
			return;
		}

		HANDLE sharedHandle = _CreateSharedTexture(outputTexture);
		if (!sharedHandle) {
			Logger::Get().Win32Error("_CreateSharedTexture å¤±è´¥");
			_sharedTextureHandle.store(INVALID_HANDLE_VALUE, std::memory_order_relaxed);
			_sharedTextureHandle.notify_one();
			return;
		}

		_sharedTextureMutexKey.store(0, std::memory_order_relaxed);

		// æ¸²æŸ“å®Œæˆå†é€šçŸ¥å‰ç«¯é˜²æ­¢é»‘å±ã€‚å‰ç«¯ä¼šè‡ªåŠ¨æ‰§è¡Œæ¸²æŸ“ï¼Œå› æ­¤æ— éœ€å‘é€ WM_FRONTEND_RENDER
		_BackendRender(outputTexture);

		_sharedTextureHandle.store(sharedHandle, std::memory_order_release);
		_sharedTextureHandle.notify_one();
	});

	// ç­‰å¾…åŽç«¯æ›´æ”¹åˆ†è¾¨çŽ‡å’Œæ¸²æŸ“
	_sharedTextureHandle.wait(NULL, std::memory_order_relaxed);
	// å°†ä¸‰ä¸ªæˆå‘˜åŒæ­¥åˆ°å‰ç«¯çº¿ç¨‹
	const HANDLE sharedTextureHandle = _sharedTextureHandle.load(std::memory_order_acquire);
	if (sharedTextureHandle == INVALID_HANDLE_VALUE) {
		return false;
	}

	// èŽ·å–å…±äº«çº¹ç†
	HRESULT hr = _frontendResources.GetD3DDevice()->OpenSharedResource(
		sharedTextureHandle, IID_PPV_ARGS(_frontendSharedTexture.put()));
	if (FAILED(hr)) {
		Logger::Get().ComError("OpenSharedResource å¤±è´¥", hr);
		return false;
	}

	_frontendSharedTextureMutex = _frontendSharedTexture.try_as<IDXGIKeyedMutex>();
	// å¿…é¡»é‡ç½® _lastAccessMutexKeyï¼Œç¡®ä¿ä¸ä¼šå’Œ _sharedTextureMutexKey åˆšå·§ç›¸åŒå¯¼è‡´æŽ¥ä¸‹æ¥çš„æ¸²æŸ“è¢«è·³è¿‡
	_lastAccessMutexKey = 0;

	_UpdateDestRect();
	return true;
}

void Renderer::OnEndResize() noexcept {
	bool shouldRedraw = false;
	_presenter->OnEndResize(shouldRedraw);

	if (shouldRedraw) {
		_FrontendRender();
	}
}

void Renderer::OnMove() noexcept {
	_UpdateDestRect();
}

void Renderer::SwitchToolbarState() noexcept {
	const ScalingWindow& scalingWindow = ScalingWindow::Get();
	LocalizationService& ls = LocalizationService::Get();

	if (scalingWindow.Options().Is3DGameMode()) {
		scalingWindow.ShowToast(ls.GetLocalizedString(L"Message_ToolbarIn3DGameMode"));
		return;
	}

	const ToolbarState newState = ToolbarState(
		((uint32_t)_overlayDrawer.ToolbarState() + 1) % (uint32_t)ToolbarState::COUNT);
	_overlayDrawer.ToolbarState(newState);

	// æ˜¾ç¤ºçŠ¶æ€åˆ‡æ¢æ¶ˆæ¯
	const wchar_t* stateResName = nullptr;
	if (newState == ToolbarState::Off) {
		stateResName = L"Home_Toolbar_InitialState_Off/Content";
	} else if (newState == ToolbarState::AlwaysShow) {
		stateResName = L"Home_Toolbar_InitialState_AlwaysShow/Content";
	} else {
		stateResName = L"Home_Toolbar_InitialState_AutoHide/Content";
	}

	winrt::hstring newStateMsg = ls.GetLocalizedString(L"Message_ToolbarNewState");
	scalingWindow.ShowToast(fmt::format(
		fmt::runtime(std::wstring_view(newStateMsg)),
		std::wstring_view(ls.GetLocalizedString(stateResName))
	));

	// ç«‹å³æ¸²æŸ“ä¸€å¸§
	_FrontendRender();
}

const RECT& Renderer::SrcRect() const noexcept {
	return ScalingWindow::Get().SrcTracker().SrcRect();
}

bool Renderer::_InitFrameSource() noexcept {
	switch (ScalingWindow::Get().Options().captureMethod) {
	case CaptureMethod::GraphicsCapture:
		_frameSource = std::make_unique<GraphicsCaptureFrameSource>();
		break;
	case CaptureMethod::DesktopDuplication:
		_frameSource = std::make_unique<DesktopDuplicationFrameSource>();
		break;
	case CaptureMethod::GDI:
		_frameSource = std::make_unique<GDIFrameSource>();
		break;
	case CaptureMethod::DwmSharedSurface:
		_frameSource = std::make_unique<DwmSharedSurfaceFrameSource>();
		break;
	default:
		Logger::Get().Error("æœªçŸ¥çš„æ•èŽ·æ¨¡å¼");
		return false;
	}

	Logger::Get().Info(StrHelper::Concat("å½“å‰æ•èŽ·æ¨¡å¼: ", _frameSource->Name()));

	if (!_frameSource->Initialize(_backendResources, _backendDescriptorStore)) {
		Logger::Get().Error("åˆå§‹åŒ– FrameSource å¤±è´¥");
		_backendInitError = ScalingError::CaptureFailed;
		return false;
	}

	// ç”±äºŽ DPI ç¼©æ”¾ï¼Œæ•èŽ·å°ºå¯¸å’Œè¾¹ç•ŒçŸ©å½¢å°ºå¯¸ä¸ä¸€å®šç›¸åŒ
	D3D11_TEXTURE2D_DESC desc;
	_frameSource->GetOutput()->GetDesc(&desc);
	Logger::Get().Info(fmt::format("æ•èŽ·å°ºå¯¸: {}x{}", desc.Width, desc.Height));

	return true;
}

static std::optional<EffectDesc> CompileEffect(
	const EffectOption& effectOption,
	bool noFP16,
	bool forceInlineParams = false
) noexcept {
	// æŒ‡å®šæ•ˆæžœå
	EffectDesc result{ .name = effectOption.name };

	EffectCompilerFlags compileFlag = EffectCompilerFlags::None;
	const ScalingOptions& scalingOptions = ScalingWindow::Get().Options();
	if (scalingOptions.IsEffectCacheDisabled()) {
		compileFlag |= EffectCompilerFlags::NoCache;
	}
	if (scalingOptions.IsSaveEffectSources()) {
		compileFlag |= EffectCompilerFlags::SaveSources;
	}
	if (scalingOptions.IsWarningsAreErrors()) {
		compileFlag |= EffectCompilerFlags::WarningsAreErrors;
	}
	if (scalingOptions.IsInlineParams() || forceInlineParams) {
		compileFlag |= EffectCompilerFlags::InlineParams;
	}
	if (noFP16) {
		compileFlag |= EffectCompilerFlags::NoFP16;
	}

	bool success = true;
	uint32_t duration = Measure([&]() {
		success = !EffectCompiler::Compile(result, compileFlag, &effectOption.parameters);
	});

	if (success) {
		Logger::Get().Info(fmt::format("ç¼–è¯‘ {}.hlsl ç”¨æ—¶ {} æ¯«ç§’",
			effectOption.name, duration / 1000.0f));
		return result;
	} else {
		Logger::Get().Error(StrHelper::Concat("ç¼–è¯‘ ",
			effectOption.name, ".hlsl å¤±è´¥"));
		return std::nullopt;
	}
}

ID3D11Texture2D* Renderer::_BuildEffects() noexcept {
	const ScalingOptions& options = ScalingWindow::Get().Options();
	const bool noFP16 = !_backendResources.IsFP16Supported() || options.IsFP16Disabled();

	const std::vector<EffectOption>& effects = options.effects;
	assert(!effects.empty());
	const uint32_t effectCount = (uint32_t)effects.size();

	// å¹¶è¡Œç¼–è¯‘æ‰€æœ‰æ•ˆæžœ
	_effectDescs.resize(effects.size());
	bool anyFailure = false;
	wil::srwlock writeLock;
	
	int duration = Measure([&]() {
		Win32Helper::RunParallel([&](uint32_t id) {
			std::optional<EffectDesc> desc = CompileEffect(effects[id], noFP16);

			auto lk = writeLock.lock_exclusive();
			if (desc) {
				_effectDescs[id] = std::move(*desc);
			} else {
				anyFailure = true;
			}
		}, effectCount);
	});

	if (anyFailure) {
		return nullptr;
	}

	if (effectCount > 1) {
		Logger::Get().Info(fmt::format("ç¼–è¯‘ç€è‰²å™¨æ€»è®¡ç”¨æ—¶ {} æ¯«ç§’", duration / 1000.0f));
	}

	_effectDrawers.resize(effectCount);

	ID3D11Texture2D* inOutTexture = _frameSource->GetOutput();
	for (uint32_t i = 0; i < effectCount; ++i) {
		if (!_effectDrawers[i].Initialize(
			_effectDescs[i],
			effects[i],
			_backendResources,
			_backendDescriptorStore,
			&inOutTexture
		)) {
			Logger::Get().Error(fmt::format("åˆå§‹åŒ–æ•ˆæžœ#{} ({}) å¤±è´¥", i, effects[i].name));
			return nullptr;
		}

		// é‡Šæ”¾ CSO å†…å­˜ï¼Œä¸å†éœ€è¦å®ƒä»¬
		for (EffectPassDesc& passDesc : _effectDescs[i].passes) {
			passDesc.cso = nullptr;
		}
	}
	
	if (_ShouldAppendBicubic(inOutTexture)) {
		if (!_AppendBicubic(&inOutTexture)) {
			Logger::Get().Error("_AppendBicubic å¤±è´¥");
			return nullptr;
		}
	}

	_UpdateActiveEffectDescs();

	// åˆå§‹åŒ–æ‰€æœ‰æ•ˆæžœå…±ç”¨çš„åŠ¨æ€å¸¸é‡ç¼“å†²åŒº
	for (const EffectDesc& effectDesc : _effectDescs) {
		if (effectDesc.flags & EffectFlags::UseDynamic) {
			D3D11_BUFFER_DESC bd{
				.ByteWidth = 16,	// åªç”¨ 4 ä¸ªå­—èŠ‚
				.Usage = D3D11_USAGE_DYNAMIC,
				.BindFlags = D3D11_BIND_CONSTANT_BUFFER,
				.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE
			};

			HRESULT hr = _backendResources.GetD3DDevice()->CreateBuffer(&bd, nullptr, _dynamicCB.put());
			if (FAILED(hr)) {
				Logger::Get().ComError("CreateBuffer å¤±è´¥", hr);
				return nullptr;
			}

			break;
		}
	}

	return inOutTexture;
}

void Renderer::_UpdateActiveEffectDescs() noexcept {
	const uint32_t effectCount = (uint32_t)_effectDescs.size();
	const uint32_t drawerCount = (uint32_t)_effectDrawers.size();

	_activeEffectDescs.resize(drawerCount);

	for (uint32_t i = 0; i < effectCount; ++i) {
		_activeEffectDescs[i] = &_effectDescs[i];
	}

	if (drawerCount > effectCount) {
		// å·²è¿½åŠ  Bicubic
		assert(drawerCount == effectCount + 1);
		_activeEffectDescs[effectCount] = &bicubicDesc;
	}
}

bool Renderer::_ShouldAppendBicubic(ID3D11Texture2D* outTexture) noexcept {
	const ScalingOptions& options = ScalingWindow::Get().Options();

	D3D11_TEXTURE2D_DESC texDesc;
	outTexture->GetDesc(&texDesc);
	const SIZE lastOutputSize = { (LONG)texDesc.Width, (LONG)texDesc.Height };
	const SIZE rendererSize = Win32Helper::GetSizeOfRect(ScalingWindow::Get().RendererRect());

	if (options.IsWindowedMode()) {
		// çª—å£æ¨¡å¼ç¼©æ”¾æ—¶ä½¿ç”¨ Bicubic æ”¾å¤§ã€‚Bicubic (B=0, C=0.5) çš„é”åˆ©åº¦å’Œ Lanczos ç›¸å·®æ— å‡ 
		return lastOutputSize != rendererSize;
	} else {
		// è¾“å‡ºå°ºå¯¸å¤§äºŽäº¤æ¢é“¾å°ºå¯¸åˆ™éœ€è¦é™é‡‡æ ·
		return lastOutputSize.cx > rendererSize.cx || lastOutputSize.cy > rendererSize.cy;
	}
}

bool Renderer::_AppendBicubic(ID3D11Texture2D** inOutTexture) noexcept {
	const ScalingOptions& options = ScalingWindow::Get().Options();

	const EffectOption bicubicOption{
		.name = "Bicubic",
		.parameters{
			{"paramB", 0.0f},
			{"paramC", 0.5f}
		},
		.scalingType = options.IsWindowedMode() ? ScalingType::Fill : ScalingType::Fit
	};

	if (bicubicDesc.name.empty()) {
		// å‚æ•°ä¸ä¼šæ”¹å˜ï¼Œå› æ­¤å¯ä»¥å†…è”
		std::optional<EffectDesc> desc = CompileEffect(bicubicOption, true, true);
		if (!desc) {
			Logger::Get().Error("ç¼–è¯‘é™é‡‡æ ·æ•ˆæžœå¤±è´¥");
			return false;
		}

		bicubicDesc = std::move(*desc);
	}

	EffectDrawer& bicubicDrawer = _effectDrawers.emplace_back();
	if (!bicubicDrawer.Initialize(
		bicubicDesc,
		bicubicOption,
		_backendResources,
		_backendDescriptorStore,
		inOutTexture
	)) {
		Logger::Get().Error("åˆå§‹åŒ–é™é‡‡æ ·æ•ˆæžœå¤±è´¥");
		return false;
	}

	return true;
}

ID3D11Texture2D* Renderer::_ResizeEffects() noexcept {
	const ScalingOptions& options = ScalingWindow::Get().Options();
	const std::vector<EffectOption>& effects = options.effects;
	assert(!effects.empty());
	const uint32_t effectCount = (uint32_t)effects.size();

	ID3D11Texture2D* inOutTexture = _frameSource->GetOutput();
	for (uint32_t i = 0; i < effectCount; ++i) {
		if (!_effectDrawers[i].ResizeTextures(
			_effectDescs[i],
			effects[i],
			_backendResources,
			&inOutTexture
		)) {
			Logger::Get().Error(fmt::format("æ›´æ”¹æ•ˆæžœ#{} ({}) å°ºå¯¸å¤±è´¥", i, effects[i].name));
			return nullptr;
		}
	}

	// å¤„ç†è¿½åŠ çš„ Bicubic
	bool changed = false;
	if (_ShouldAppendBicubic(inOutTexture)) {
		if (_effectDrawers.size() > effectCount) {
			const EffectOption bicubicOption{
				.name = "Bicubic",
				.parameters{
					{"paramB", 0.0f},
					{"paramC", 0.5f}
				},
				.scalingType = options.IsWindowedMode() ? ScalingType::Fill : ScalingType::Fit
			};

			if (!_effectDrawers.back().ResizeTextures(
				bicubicDesc,
				bicubicOption,
				_backendResources,
				&inOutTexture
			)) {
				Logger::Get().Error("æ›´æ”¹æ•ˆæžœ Bicubic å°ºå¯¸å¤±è´¥");
				return nullptr;
			}
		} else {
			_AppendBicubic(&inOutTexture);
			changed = true;
		}
	} else {
		if (_effectDrawers.size() > effectCount) {
			_effectDrawers.resize(effectCount);
			changed = true;
		}
	}

	if (changed) {
		_UpdateActiveEffectDescs();
		_overlayDrawer.UpdateAfterActiveEffectsChanged();

		if (_effectsProfiler.IsProfiling()) {
			uint32_t passCount = 0;
			for (const EffectDesc* desc : _activeEffectDescs) {
				passCount += (uint32_t)desc->passes.size();
			}
			_effectsProfiler.SetPassCount(_backendResources.GetD3DDevice(), passCount);
		}
	}

	return inOutTexture;
}

void Renderer::_UpdateDestRect() noexcept {
	const RECT& rendererRect = ScalingWindow::Get().RendererRect();
	OutputAlignment alignment = ScalingWindow::Get().Options().outputAlignment;

	LONG destWidth;
	LONG destHeight;
	{
		D3D11_TEXTURE2D_DESC desc;
		_frontendSharedTexture->GetDesc(&desc);
		destWidth = (LONG)desc.Width;
		destHeight = (LONG)desc.Height;
	}

	using enum OutputAlignment;

	if (alignment == LeftTop || alignment == Left || alignment == LeftBottom) {
		_destRect.left = 0;
		_destRect.right = destWidth;
	} else if (alignment == Top || alignment == Center || alignment == Bottom) {
		_destRect.left = (rendererRect.left + rendererRect.right - destWidth) / 2;
		_destRect.right = _destRect.left + destWidth;
	} else {
		_destRect.left = rendererRect.right - destWidth;
		_destRect.right = rendererRect.right;
	}

	if (alignment == LeftTop || alignment == Top || alignment == RightTop) {
		_destRect.top = 0;
		_destRect.bottom = destHeight;
	} else if (alignment == Left || alignment == Center || alignment == Right) {
		_destRect.top = (rendererRect.top + rendererRect.bottom - destHeight) / 2;
		_destRect.bottom = _destRect.top + destHeight;
	} else {
		_destRect.top = rendererRect.bottom - destHeight;
		_destRect.bottom = rendererRect.bottom;
	}

	assert(_destRect.left + destWidth == _destRect.right);
	assert(_destRect.top + destHeight == _destRect.bottom);
}

HANDLE Renderer::_CreateSharedTexture(ID3D11Texture2D* effectsOutput) noexcept {
	D3D11_TEXTURE2D_DESC desc;
	effectsOutput->GetDesc(&desc);
	SIZE textureSize = { (LONG)desc.Width, (LONG)desc.Height };

	// åˆ›å»ºå…±äº«çº¹ç†
	_backendSharedTexture = DirectXHelper::CreateTexture2D(
		_backendResources.GetD3DDevice(),
		DXGI_FORMAT_R8G8B8A8_UNORM,
		textureSize.cx,
		textureSize.cy,
		D3D11_BIND_SHADER_RESOURCE,
		D3D11_USAGE_DEFAULT,
		D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX
	);
	if (!_backendSharedTexture) {
		Logger::Get().Error("åˆ›å»º Texture2D å¤±è´¥");
		return NULL;
	}

	_backendSharedTextureMutex = _backendSharedTexture.try_as<IDXGIKeyedMutex>();

	winrt::com_ptr<IDXGIResource> sharedDxgiRes = _backendSharedTexture.try_as<IDXGIResource>();

	HANDLE sharedHandle = NULL;
	HRESULT hr = sharedDxgiRes->GetSharedHandle(&sharedHandle);
	if (FAILED(hr)) {
		Logger::Get().ComError("GetSharedHandle å¤±è´¥", hr);
		return NULL;
	}

	return sharedHandle;
}

void Renderer::_BackendThreadProc() noexcept {
#ifdef _DEBUG
	SetThreadDescription(GetCurrentThread(), L"Magpie-ç¼©æ”¾åŽç«¯çº¿ç¨‹");
#endif

	winrt::init_apartment(winrt::apartment_type::single_threaded);

	if (const HANDLE sharedHandle = _InitBackend()) {
		_sharedTextureHandle.store(sharedHandle, std::memory_order_release);
		_sharedTextureHandle.notify_one();
	} else {
		_frameSource.reset();
		// é€šçŸ¥å‰ç«¯åˆå§‹åŒ–å¤±è´¥
		_sharedTextureHandle.store(INVALID_HANDLE_VALUE, std::memory_order_release);
		_sharedTextureHandle.notify_one();

		// å³ä½¿å¤±è´¥ä¹Ÿè¦åˆ›å»ºæ¶ˆæ¯å¾ªçŽ¯ï¼Œå¦åˆ™å‰ç«¯çº¿ç¨‹å°†ä¸€ç›´ç­‰å¾…
		MSG msg;
		while (GetMessage(&msg, NULL, 0, 0)) {
			DispatchMessage(&msg);
		}
		return;
	}

	auto se = wil::scope_exit([this] {
		// ç­‰å¾…æˆªå›¾å®Œæˆ
		MSG msg;
		while (_pendingScreenshotCount != 0) {
			// å¿…é¡»å¤„ç†æ¶ˆæ¯é˜Ÿåˆ—ï¼Œå¦åˆ™åç¨‹æ— æ³•æ‰§è¡Œ
			WaitMessage();

			while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
				DispatchMessage(&msg);
			}
		}

		// ä¸èƒ½åœ¨å‰ç«¯çº¿ç¨‹é‡Šæ”¾
		GetVoltaDLSSBridge().Shutdown();
		_frameSource.reset();
	});

	StepTimerStatus stepTimerStatus = StepTimerStatus::WaitingForNewFrame;
	const bool waitMsgForNewFrame =
		_frameSource->WaitType() == FrameSourceWaitType::WaitForMessage;

	MSG msg;
	while (true) {
		bool fpsUpdated = false;
		stepTimerStatus = _stepTimer.WaitForNextFrame(
			waitMsgForNewFrame && stepTimerStatus != StepTimerStatus::WaitingForFPSLimiter,
			fpsUpdated
		);

		while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
			if (msg.message == WM_QUIT) {
				return;
			}

			DispatchMessage(&msg);
		}

		if (stepTimerStatus == StepTimerStatus::WaitingForFPSLimiter) {
			// æ–°å¸§æ¶ˆæ¯å¯èƒ½å·²è¢«å¤„ç†ï¼Œä¹‹åŽçš„ WaitForNextFrame ä¸è¦ç­‰å¾…æ¶ˆæ¯ï¼Œç›´åˆ°çŠ¶æ€å˜åŒ–
			continue;
		}

		switch (_frameSource->Update()) {
		case FrameSourceState::Waiting:
			if (stepTimerStatus != StepTimerStatus::ForceNewFrame) {
				if (fpsUpdated) {
					// FPS å˜åŒ–åˆ™è¦æ±‚å‰ç«¯é‡æ–°æ¸²æŸ“ä»¥æ›´æ–°å åŠ å±‚ï¼Œè°ƒæ•´å¤§å°æ—¶è¿™ä¸ªæ“ä½œååˆ†å¿…è¦
					PostMessage(ScalingWindow::Get().Handle(),
						CommonSharedConstants::WM_FRONTEND_RENDER, 0, 0);
				}
				break;
			}

			// å¼ºåˆ¶å¸§
			[[fallthrough]];
		case FrameSourceState::NewFrame:
			_BackendRender(_effectDrawers.back().GetOutputTexture());
			// é€šçŸ¥å‰ç«¯æ‰§è¡Œæ¸²æŸ“
			PostMessage(ScalingWindow::Get().Handle(),
				CommonSharedConstants::WM_FRONTEND_RENDER, 0, 0);
			break;
		case FrameSourceState::Error:
			// æ•èŽ·å‡ºé”™ï¼Œé€€å‡ºç¼©æ”¾
			ScalingWindow::Dispatcher().TryEnqueue([]() {
				ScalingWindow& scalingWindow = ScalingWindow::Get();
				scalingWindow.ShowError(ScalingError::CaptureFailed);
				scalingWindow.Stop();
			});

			while (GetMessage(&msg, NULL, 0, 0)) {
				DispatchMessage(&msg);
			}
			return;
		}
	}
}

HANDLE Renderer::_InitBackend() noexcept {
	// åˆ›å»º DispatcherQueue
	{
		winrt::Windows::System::DispatcherQueueController dqc{ nullptr };
		HRESULT hr = CreateDispatcherQueueController(
			DispatcherQueueOptions{
				.dwSize = sizeof(DispatcherQueueOptions),
				.threadType = DQTYPE_THREAD_CURRENT
			},
			(PDISPATCHERQUEUECONTROLLER*)winrt::put_abi(dqc)
		);
		if (FAILED(hr)) {
			Logger::Get().ComError("CreateDispatcherQueueController å¤±è´¥", hr);
			return NULL;
		}

		_backendThreadDispatcher = dqc.DispatcherQueue();
	}

	if (!_backendResources.Initialize(false)) {
		return NULL;
	}
	
	ID3D11Device5* d3dDevice = _backendResources.GetD3DDevice();
	_backendDescriptorStore.Initialize(d3dDevice);

	if (!_InitFrameSource()) {
		return NULL;
	}

	{
		std::optional<float> maxFrameRate;
		if (_frameSource->WaitType() == FrameSourceWaitType::NoWait) {
			// æŸäº›æ•èŽ·æ–¹å¼ä¸ä¼šé™åˆ¶æ•èŽ·å¸§çŽ‡ï¼Œå› æ­¤å°†æ•èŽ·å¸§çŽ‡é™åˆ¶ä¸ºå±å¹•åˆ·æ–°çŽ‡
			const HWND hwndSrc = ScalingWindow::Get().SrcTracker().Handle();
			if (HMONITOR hMon = MonitorFromWindow(hwndSrc, MONITOR_DEFAULTTONEAREST)) {
				MONITORINFOEX mi{ { sizeof(MONITORINFOEX) } };
				GetMonitorInfo(hMon, &mi);

				DEVMODE dm{ .dmSize = sizeof(DEVMODE) };
				EnumDisplaySettings(mi.szDevice, ENUM_CURRENT_SETTINGS, &dm);

				if (dm.dmDisplayFrequency > 0) {
					Logger::Get().Info(fmt::format("å±å¹•åˆ·æ–°çŽ‡: {}", dm.dmDisplayFrequency));
					maxFrameRate = float(dm.dmDisplayFrequency);
				}
			}
		}

		const ScalingOptions& options = ScalingWindow::Get().Options();
		if (options.maxFrameRate) {
			if (!maxFrameRate || *options.maxFrameRate < *maxFrameRate) {
				maxFrameRate = options.maxFrameRate;
			}
		}
		
		// æµ‹è¯•ç€è‰²å™¨æ€§èƒ½æ—¶æœ€å°å¸§çŽ‡åº”è®¾ä¸ºæ— é™å¤§ï¼Œä½†ç”±äºŽ /fp:fast ä¸‹æ— é™å¤§ä¸å¯é ï¼Œå› æ­¤æ”¹ä¸ºä½¿ç”¨ max()ï¼Œ
		// å’Œæ— é™å¤§æ•ˆæžœç›¸åŒã€‚
		const float minFrameRate = options.IsBenchmarkMode()
			? std::numeric_limits<float>::max() : options.minFrameRate;
		_stepTimer.Initialize(minFrameRate, maxFrameRate);
	}

	ID3D11Texture2D* outputTexture = _BuildEffects();
	if (!outputTexture) {
		return NULL;
	}

	HRESULT hr = d3dDevice->CreateFence(
		_fenceValue, D3D11_FENCE_FLAG_NONE, IID_PPV_ARGS(&_d3dFence));
	if (FAILED(hr)) {
		// GH#979
		// è¿™ä¸ªé”™è¯¯ä¼šåœ¨æŸäº›å¾ˆæ—§çš„æ˜¾å¡ä¸Šå‡ºçŽ°ï¼Œä¼¼ä¹Žæ˜¯é©±åŠ¨çš„ bugã€‚æ–‡æ¡£ä¸­æåˆ° ID3D11Device5::CreateFence 
		// å’Œ ID3D12Device::CreateFence ç­‰ä»·ï¼Œä½†æ”¯æŒ DX12 çš„æ˜¾å¡ä¹Ÿæœ‰å¤±è´¥çš„å¯èƒ½ï¼Œå¦‚ GH#1013
		Logger::Get().ComError("CreateFence å¤±è´¥", hr);
		_backendInitError = ScalingError::CreateFenceFailed;
		return NULL;
	}

	if (!_fenceEvent.try_create(wil::EventOptions::None, nullptr)) {
		Logger::Get().Win32Error("CreateEvent å¤±è´¥");
		return NULL;
	}

	HANDLE sharedHandle = _CreateSharedTexture(outputTexture);
	if (!sharedHandle) {
		Logger::Get().Error("_CreateSharedTexture å¤±è´¥");
		return NULL;
	}

	// æœ€åŽå¯åŠ¨æ•èŽ·ä»¥å°½å¯èƒ½æŽ¨è¿Ÿæ˜¾ç¤ºé»„è‰²è¾¹æ¡† (Win10) æˆ–ç¦ç”¨åœ†è§’ (Win11)
	if (!_frameSource->Start()) {
		Logger::Get().Error("å¯åŠ¨æ•èŽ·å¤±è´¥");
		return NULL;
	}
return sharedHandle;
}

void Renderer::_BackendRender(ID3D11Texture2D* effectsOutput) noexcept {
	_stepTimer.PrepareForRender();

	ID3D11DeviceContext4* d3dDC = _backendResources.GetD3DDC();
	d3dDC->ClearState();

	if (ID3D11Buffer* t = _dynamicCB.get()) {
		_UpdateDynamicConstants();
		d3dDC->CSSetConstantBuffers(1, 1, &t);
	}
    const std::vector<EffectOption>& effects =
        ScalingWindow::Get().Options().effects;

    const bool useVoltaDLSS =
        !effects.empty() &&
        effects[0].name == "VoltaDLSS";

    if (useVoltaDLSS) {
        if (!GetVoltaDLSSBridge().Initialize()) {
            Logger::Get().Error("VoltaDLSSWmma.dll initialization failed");
            return;
        }

        ID3D11Texture2D* captureTexture =
            _frameSource->GetOutput();

        // VoltaDLSS is the first stage of the effect chain.
        // Let CUDA write directly into the first drawer's output,
        // then let normal Magpie effects consume that texture.
        ID3D11Texture2D* voltaOutput =
            _effectDrawers[0].GetOutputTexture();

        float kernelMs = 0.0f;

        if (!GetVoltaDLSSBridge().Process(
                _backendResources.GetD3DDevice(),
                captureTexture,
                voltaOutput,
                &kernelMs)) {
            Logger::Get().Error("VoltaDLSS processing failed");
            return;
        }

        static bool loggedVolta = false;

        if (!loggedVolta) {
            D3D11_TEXTURE2D_DESC inputDesc{};
            D3D11_TEXTURE2D_DESC outputDesc{};

            captureTexture->GetDesc(&inputDesc);
            voltaOutput->GetDesc(&outputDesc);

            Logger::Get().Info(fmt::format(
                "VoltaDLSS v2 active: {}x{} -> {}x{}, CUDA kernel {:.3f} ms",
                inputDesc.Width,
                inputDesc.Height,
                outputDesc.Width,
                outputDesc.Height,
                kernelMs
            ));

            loggedVolta = true;
        }

        // Skip the VoltaDLSS HLSL drawer itself because CUDA already
        // produced its output. Run every later effect normally.
        _effectsProfiler.OnBeginEffects(d3dDC);

        for (uint32_t i = 1; i < _effectDrawers.size(); ++i) {
            _effectDrawers[i].Draw(_effectsProfiler);
        }

        _effectsProfiler.OnEndEffects(d3dDC);
    }
    else {
        _effectsProfiler.OnBeginEffects(d3dDC);

        for (const EffectDrawer& effectDrawer : _effectDrawers) {
            effectDrawer.Draw(_effectsProfiler);
        }

        _effectsProfiler.OnEndEffects(d3dDC);
    }

	HRESULT hr = d3dDC->Signal(_d3dFence.get(), ++_fenceValue);
	if (FAILED(hr)) {
		Logger::Get().ComError("Signal å¤±è´¥", hr);
		return;
	}

	hr = _d3dFence->SetEventOnCompletion(_fenceValue, _fenceEvent.get());
	if (FAILED(hr)) {
		Logger::Get().ComError("SetEventOnCompletion å¤±è´¥", hr);
		return;
	}

	d3dDC->Flush();

	// ç­‰å¾…æ¸²æŸ“å®Œæˆ
	_fenceEvent.wait();

	// æŸ¥è¯¢æ•ˆæžœçš„æ¸²æŸ“æ—¶é—´
	_effectsProfiler.QueryTimings(d3dDC);

	// æ¸²æŸ“å®ŒæˆåŽå†æ›´æ–° _sharedTextureMutexKeyï¼Œå¦åˆ™å‰ç«¯å¿…é¡»ç­‰å¾…ï¼Œé™ä½Žå…‰æ ‡æµç•…åº¦
	const uint64_t key = ++_sharedTextureMutexKey;
	hr = _backendSharedTextureMutex->AcquireSync(key - 1, INFINITE);
	if (FAILED(hr)) {
		Logger::Get().ComError("AcquireSync å¤±è´¥", hr);
		return;
	}

	d3dDC->CopyResource(_backendSharedTexture.get(), effectsOutput);

	_backendSharedTextureMutex->ReleaseSync(key);

	// æ ¹æ® https://learn.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-id3d11device-opensharedresourceï¼Œ
	// æ›´æ–°å…±äº«çº¹ç†åŽå¿…é¡»è°ƒç”¨ Flush
	d3dDC->Flush();
}

bool Renderer::_UpdateDynamicConstants() const noexcept {
	// cbuffer __CB2 : register(b1) { uint __frameCount; };

	ID3D11DeviceContext4* d3dDC = _backendResources.GetD3DDC();

	D3D11_MAPPED_SUBRESOURCE ms;
	HRESULT hr = d3dDC->Map(_dynamicCB.get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &ms);
	if (SUCCEEDED(hr)) {
		// é¿å…ä½¿ç”¨ *(uint32_t*)ms.pDataï¼Œè§
		// https://learn.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-id3d11devicecontext-map
		const uint32_t frameCount = _stepTimer.GetFrameCount();
		std::memcpy(ms.pData, &frameCount, 4);
		d3dDC->Unmap(_dynamicCB.get(), 0);
	} else {
		Logger::Get().ComError("Map å¤±è´¥", hr);
		return false;
	}

	return true;
}

static void AppendSuffixToScreenshotFilename(std::wstring& screenshotFileName) noexcept {
	const ScalingOptions& options = ScalingWindow::Get().Options();

	uint32_t suffixNum = 1;

	WIN32_FIND_DATA findData{};
	wil::unique_hfind hFind(FindFirstFileEx(
		StrHelper::Concat(options.screenshotsDir.native(), L"\\", screenshotFileName, L"*").c_str(),
		FindExInfoBasic, &findData, FindExSearchNameMatch, nullptr, FIND_FIRST_EX_LARGE_FETCH));
	if (hFind) {
		do {
			std::wstring_view fileName(findData.cFileName);

			// ä¸è€ƒè™‘æ‰©å±•å
			if (size_t dotPos = fileName.find_last_of(L'.'); dotPos != std::wstring_view::npos) {
				fileName.remove_suffix(fileName.size() - dotPos);
			}

			// æ–‡ä»¶åé‡å¤æ—¶æ ¼å¼ä¸º "{baseFileName} ({suffixNum})"

			// åŽŸå§‹æ–‡ä»¶åè§†ä¸º 1
			if (fileName.size() == screenshotFileName.size()) {
				assert(fileName == screenshotFileName);
				suffixNum = std::max(suffixNum, 2u);
				continue;
			}

			// åŽç¼€è‡³å°‘ 4 ä¸ªå­—ç¬¦
			if (fileName.size() < screenshotFileName.size() + 4) {
				continue;
			}

			fileName.remove_prefix(screenshotFileName.size());

			if (!fileName.starts_with(L" (") || !fileName.ends_with(L')')) {
				continue;
			}

			uint32_t curSuffixNum;
			std::string curSuffixNumStr = StrHelper::UTF16ToUTF8(
				std::wstring_view(fileName.data() + 2, fileName.size() - 3));
			const char* end = curSuffixNumStr.data() + curSuffixNumStr.size();

			std::from_chars_result result = std::from_chars(
				curSuffixNumStr.data(), end, curSuffixNum);
			if (result.ec != std::errc{} || result.ptr != end || curSuffixNum < 2) {
				continue;
			}

			suffixNum = std::max(suffixNum, curSuffixNum + 1);
		} while (FindNextFile(hFind.get(), &findData));
	}

	if (suffixNum > 1) {
		screenshotFileName += L" (";
		screenshotFileName += StrHelper::ToWString(suffixNum);
		screenshotFileName += L')';
	}
}

winrt::IAsyncOperation<int> Renderer::_TakeScreenshotImpl(
	uint32_t effectIdx,
	uint32_t passIdx,
	uint32_t outputIdx,
	std::wstring& screenshotFileName
) noexcept {
	// æˆªå›¾æµç¨‹:
	// 1. åŽç«¯çº¿ç¨‹å‘å‡ºæ¸²æŸ“å’Œå¤åˆ¶çº¹ç†çš„ GPU æŒ‡ä»¤
	// 2. è½¬åˆ°çº¿ç¨‹æ± ç­‰å¾… GPU å®Œæˆ
	// 3. è½¬åˆ°åŽç«¯çº¿ç¨‹å¤åˆ¶çº¹ç†æ•°æ®åˆ°å†…å­˜
	// 4. è½¬åˆ°çº¿ç¨‹æ± å†™å…¥å›¾ç‰‡

	ID3D11Device5* d3dDevice = _backendResources.GetD3DDevice();
	ID3D11DeviceContext4* d3dDC = _backendResources.GetD3DDC();
	ID3D11Texture2D* sourceTex;
	EffectIntermediateTextureFormat format;
	// æ•ˆæžœè¾“å‡ºä¿å­˜ä¸º pngï¼Œä¸­é—´ç»“æžœä¿å­˜ä¸º dds
	const wchar_t* imgFormat;

	if (passIdx == std::numeric_limits<uint32_t>::max()) {
		sourceTex = _effectDrawers[effectIdx].GetOutputTexture();
		format = _activeEffectDescs[effectIdx]->textures[1].format;
		imgFormat = L"png";
	} else {
		const std::vector<EffectPassDesc>& passes = _activeEffectDescs[effectIdx]->passes;
		const uint32_t passCount = (uint32_t)passes.size();

		const SmallVector<uint32_t>& outputs = passes[passIdx].outputs;
		// åªæœ‰ä¸€ä¸ªè¾“å‡ºæ—¶æ‰å…è®¸ä¸æä¾› outputIdx
		assert(outputIdx != std::numeric_limits<uint32_t>::max() || outputs.size() == 1);
		const uint32_t targetOutput =
			outputIdx == std::numeric_limits<uint32_t>::max() ? outputs[0] : outputs[outputIdx];

		sourceTex = _effectDrawers[effectIdx].GetTexture(targetOutput);
		format = _activeEffectDescs[effectIdx]->textures[targetOutput].format;
		imgFormat = targetOutput == 1 ? L"png" : L"dds";

		// æœ€åŽä¸€ä¸ªé€šé“çš„è¾“å‡ºå³ OUTPUT ä¸ä¼šè¢«è¦†ç›–ï¼Œå¯ä»¥ç›´æŽ¥ä½¿ç”¨ã€‚
		// å€’æ•°ç¬¬äºŒä¸ªé€šé“çš„è¾“å‡ºä¹Ÿä¸ä¼šè¢«è¦†ç›–ï¼Œå› ä¸ºæœ€åŽä¸€ä¸ªé€šé“åªä¼šå†™å…¥ OUTPUTã€‚
		// ä»Žå€’æ•°ç¬¬ä¸‰ä¸ªé€šé“å¼€å§‹éœ€è¦æ£€æŸ¥è¾“å‡ºæ˜¯å¦è¢«åŽé¢çš„é€šé“è¦†ç›–ã€‚
		if (passIdx + 3 <= passCount) {
			// æ£€æŸ¥ targetOutput æ˜¯å¦è¢«åŽé¢çš„é€šé“ä¿®æ”¹ 
			for (uint32_t i = passIdx + 1, end = passCount - 1; i < end; ++i) {
				const SmallVector<uint32_t>& curOutputs = passes[i].outputs;
				if (std::find(curOutputs.begin(), curOutputs.end(), targetOutput) != curOutputs.end()) {
					// è‹¥è¢«è¦†ç›–éœ€é‡æ–°æ¸²æŸ“
					d3dDC->ClearState();

					if (ID3D11Buffer* t = _dynamicCB.get()) {
						d3dDC->CSSetConstantBuffers(1, 1, &t);
					}

					_effectDrawers[effectIdx].DrawForExport(*_activeEffectDescs[effectIdx], passIdx);
					break;
				}
			}
		}
	}

	// åˆ›å»º staging çº¹ç†
	D3D11_TEXTURE2D_DESC desc;
	sourceTex->GetDesc(&desc);
	desc.Usage = D3D11_USAGE_STAGING;
	desc.BindFlags = 0;
	desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	desc.MiscFlags = 0;

	winrt::com_ptr<ID3D11Texture2D> stagingTex;
	HRESULT hr = d3dDevice->CreateTexture2D(&desc, nullptr, stagingTex.put());
	if (FAILED(hr)) {
		Logger::Get().ComError("CreateTexture2D å¤±è´¥", hr);
		co_return (int)TakeScreenshotResult::InternalError;
	}

	d3dDC->CopyResource(stagingTex.get(), sourceTex);
	
	// ä¸ºé¿å…æ··ä¹±ï¼Œä½¿ç”¨ç‹¬ç«‹çš„æ …æ 
	winrt::com_ptr<ID3D11Fence> localFence;
	wil::unique_event_nothrow localFenceEvent;

	hr = d3dDevice->CreateFence(
		0, D3D11_FENCE_FLAG_NONE, IID_PPV_ARGS(&localFence));
	if (FAILED(hr)) {
		Logger::Get().ComError("CreateFence å¤±è´¥", hr);
		co_return (int)TakeScreenshotResult::InternalError;
	}

	if (!localFenceEvent.try_create(wil::EventOptions::None, nullptr)) {
		Logger::Get().Win32Error("CreateEvent å¤±è´¥");
		co_return (int)TakeScreenshotResult::InternalError;
	}

	hr = d3dDC->Signal(localFence.get(), 1);
	if (FAILED(hr)) {
		Logger::Get().ComError("Signal å¤±è´¥", hr);
		co_return (int)TakeScreenshotResult::InternalError;
	}

	hr = localFence->SetEventOnCompletion(1, localFenceEvent.get());
	if (FAILED(hr)) {
		Logger::Get().ComError("SetEventOnCompletion å¤±è´¥", hr);
		co_return (int)TakeScreenshotResult::InternalError;
	}

	d3dDC->Flush();

	// è½¬åˆ°åŽå°ç­‰å¾… GPU ä»¥é˜²æ­¢å¡é¡¿
	co_await winrt::resume_background();
	localFenceEvent.wait();
	co_await _backendThreadDispatcher;

	// è¯»å–çº¹ç†æ•°æ®åˆ°å†…å­˜
	D3D11_MAPPED_SUBRESOURCE mapped;
	hr = d3dDC->Map(stagingTex.get(), 0, D3D11_MAP_READ, 0, &mapped);
	if (FAILED(hr)) {
		Logger::Get().ComError("Map å¤±è´¥", hr);
		co_return (int)TakeScreenshotResult::InternalError;
	}

	std::vector<uint8_t> pixelData(size_t(mapped.RowPitch) * desc.Height);
	std::memcpy(pixelData.data(), mapped.pData, pixelData.size());

	d3dDC->Unmap(stagingTex.get(), 0);

	co_await winrt::resume_background();

	// ç¡®ä¿æˆªå›¾ä¿å­˜ç›®å½•å­˜åœ¨
	const ScalingOptions& options = ScalingWindow::Get().Options();

	if (options.screenshotsDir.empty()) {
		co_return (int)TakeScreenshotResult::InvalidDirectory;
	}

	if (!Win32Helper::CreateDir(options.screenshotsDir.c_str(), true)) {
		Logger::Get().Error("CreateDir å¤±è´¥");
		co_return (int)TakeScreenshotResult::InvalidDirectory;
	}

	// è§£æžæ–‡ä»¶åæ¨¡æ¿
	{
		std::string fileNameUtf8;
		if (!ScreenshotFilenameTemplateHelper::Apply(
			options.screenshotFilenameTemplate,
			ScalingWindow::Get().SrcTracker().Handle(),
			fileNameUtf8
		)) {
			Logger::Get().Error("ScreenshotFilenameTemplateHelper::Apply å¤±è´¥");
			co_return (int)TakeScreenshotResult::InvalidFilenameTemplate;
		}

		screenshotFileName = StrHelper::UTF8ToUTF16(fileNameUtf8);
	}

	// åŠ é”ä»¥ä¿è¯åŽç¼€çš„å”¯ä¸€æ€§
	static wil::srwlock lock;
	{
		auto lk = lock.lock_exclusive();

		AppendSuffixToScreenshotFilename(screenshotFileName);

		screenshotFileName += L'.';
		screenshotFileName += imgFormat;

		std::filesystem::path screenshotPath = options.screenshotsDir / screenshotFileName;

		if (!TextureHelper::SaveTexture(screenshotPath.c_str(), desc.Width, desc.Height,
			format, pixelData, mapped.RowPitch)) {
			Logger::Get().Error("TextureHelper::SaveTexture å¤±è´¥");
			co_return (int)TakeScreenshotResult::InternalError;
		}
	}

	co_return (int)TakeScreenshotResult::Success;
}

// ç›‘å¬ PrintScreen å®žçŽ°æˆªå±æ—¶éšè—å…‰æ ‡
LRESULT CALLBACK Renderer::_LowLevelKeyboardHook(int nCode, WPARAM wParam, LPARAM lParam) {
	if (nCode != HC_ACTION || wParam != WM_KEYDOWN) {
		return CallNextHookEx(NULL, nCode, wParam, lParam);
	}

	KBDLLHOOKSTRUCT* info = (KBDLLHOOKSTRUCT*)lParam;
	if (info->vkCode == VK_SNAPSHOT) {
		// ä¸ºäº†ç¼©çŸ­é’©å­å¤„ç†æ—¶é—´ï¼Œå¼‚æ­¥æ‰§è¡Œæ‰€æœ‰é€»è¾‘
		ScalingWindow::Dispatcher().TryEnqueue([]() -> winrt::fire_and_forget {
			// æš‚æ—¶éšè—å…‰æ ‡
			Renderer& renderer = ScalingWindow::Get().Renderer();
			renderer._cursorDrawer.IsCursorVisible(false);
			renderer._FrontendRender();

			const uint32_t runId = ScalingWindow::RunId();

			winrt::DispatcherQueue dispatcher = ScalingWindow::Dispatcher();
			co_await 200ms;
			co_await dispatcher;

			if (ScalingWindow::RunId() == runId &&
				!renderer._cursorDrawer.IsCursorVisible()
			) {
				renderer._cursorDrawer.IsCursorVisible(true);
				renderer._FrontendRender();
			}
		});
	}

	return CallNextHookEx(NULL, nCode, wParam, lParam);
}

}


