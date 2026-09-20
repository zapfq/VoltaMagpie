#include "easu_tensor.h"

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

#include <d3d11.h>

#include <new>

struct VoltaDLSSContext
{
    EasuTensorPipeline pipeline{};
    ID3D11Device* device = nullptr;
};

extern "C"
{

__declspec(dllexport)
void* VoltaDLSS_Create()
{
    return new (std::nothrow) VoltaDLSSContext{};
}

__declspec(dllexport)
void VoltaDLSS_Destroy(void* handle)
{
    if (handle == nullptr)
    {
        return;
    }

    auto* context =
        static_cast<VoltaDLSSContext*>(handle);

    destroyEasuTensorPipeline(
        context->pipeline);

    delete context;
}

__declspec(dllexport)
bool VoltaDLSS_Process(
    void* handle,
    ID3D11Device* device,
    ID3D11Texture2D* inputTexture,
    ID3D11Texture2D* outputTexture,
    float* kernelMs)
{
    if (handle == nullptr ||
        device == nullptr ||
        inputTexture == nullptr ||
        outputTexture == nullptr)
    {
        return false;
    }

    auto* context =
        static_cast<VoltaDLSSContext*>(handle);

    if (context->device != device)
    {
        const cudaError_t err =
            cudaD3D11SetDirect3DDevice(device);

        if (err != cudaSuccess)
        {
            return false;
        }

        context->device = device;
    }

    D3D11_TEXTURE2D_DESC inputDesc{};
    inputTexture->GetDesc(&inputDesc);

    D3D11_TEXTURE2D_DESC outputDesc{};
    outputTexture->GetDesc(&outputDesc);

    if (inputDesc.Width == 0 ||
        inputDesc.Height == 0 ||
        outputDesc.Width == 0 ||
        outputDesc.Height == 0)
    {
        return false;
    }

    /*
        The proven v1 EASU Tensor implementation operates
        on four-byte BGRA/RGBA textures.

        For the first bridge build, require a four-byte format.
    */

    if (inputDesc.Format != DXGI_FORMAT_R8G8B8A8_UNORM &&
        inputDesc.Format != DXGI_FORMAT_B8G8R8A8_UNORM)
    {
        return false;
    }

    if (outputDesc.Format != DXGI_FORMAT_R8G8B8A8_UNORM &&
        outputDesc.Format != DXGI_FORMAT_B8G8R8A8_UNORM)
    {
        return false;
    }

    /*
        The v1 pipeline allocates its CUDA buffers according
        to the source/output dimensions and automatically
        re-registers D3D11 resources when their pointers change.
    */

    if (!initEasuTensorPipeline(
            context->pipeline,
            static_cast<int>(inputDesc.Width),
            static_cast<int>(inputDesc.Height),
            static_cast<int>(outputDesc.Width),
            static_cast<int>(outputDesc.Height)))
    {
        return false;
    }

    return upscaleEasuTensorD3D11(
        context->pipeline,
        inputTexture,
        outputTexture,
        kernelMs);
}

}