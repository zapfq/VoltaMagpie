#pragma once

#include <Windows.h>
#include <d3d11.h>
#include <string>

namespace Magpie {

class VoltaDLSSBridge {
public:
	using CreateFn = void* (*)();

	using DestroyFn = void (*)(void*);

	using ProcessFn = bool (*)(
		void*,
		ID3D11Device*,
		ID3D11Texture2D*,
		ID3D11Texture2D*,
		float*
	);

	bool Initialize() noexcept {
		if (_module && _context && _process) {
			return true;
		}

		wchar_t exePath[32768]{};
		const DWORD length = GetModuleFileNameW(
			nullptr,
			exePath,
			static_cast<DWORD>(std::size(exePath))
		);

		if (length == 0 || length >= std::size(exePath)) {
			return false;
		}

		std::wstring dllPath(exePath, length);
		const size_t slash = dllPath.find_last_of(L"\\/");

		if (slash == std::wstring::npos) {
			return false;
		}

		dllPath.resize(slash + 1);
		dllPath += L"VoltaDLSSWmma.dll";

		_module = LoadLibraryW(dllPath.c_str());
		if (!_module) {
			return false;
		}

		_create = reinterpret_cast<CreateFn>(
			GetProcAddress(_module, "VoltaDLSS_Create")
		);

		_destroy = reinterpret_cast<DestroyFn>(
			GetProcAddress(_module, "VoltaDLSS_Destroy")
		);

		_process = reinterpret_cast<ProcessFn>(
			GetProcAddress(_module, "VoltaDLSS_Process")
		);

		if (!_create || !_destroy || !_process) {
			Shutdown();
			return false;
		}

		_context = _create();

		if (!_context) {
			Shutdown();
			return false;
		}

		return true;
	}

	void Shutdown() noexcept {
		if (_context && _destroy) {
			_destroy(_context);
		}

		_context = nullptr;
		_create = nullptr;
		_destroy = nullptr;
		_process = nullptr;

		if (_module) {
			FreeLibrary(_module);
		}

		_module = nullptr;
	}

	bool Process(
		ID3D11Device* device,
		ID3D11Texture2D* inputTexture,
		ID3D11Texture2D* outputTexture,
		float* kernelMs
	) noexcept {
		if (!_context ||
			!_process ||
			!device ||
			!inputTexture ||
			!outputTexture) {
			return false;
		}

		return _process(
			_context,
			device,
			inputTexture,
			outputTexture,
			kernelMs
		);
	}

private:
	HMODULE _module = nullptr;
	void* _context = nullptr;

	CreateFn _create = nullptr;
	DestroyFn _destroy = nullptr;
	ProcessFn _process = nullptr;
};

inline VoltaDLSSBridge& GetVoltaDLSSBridge() noexcept {
	static VoltaDLSSBridge bridge;
	return bridge;
}

}