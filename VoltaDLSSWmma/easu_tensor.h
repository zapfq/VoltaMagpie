#pragma once

#include "scaler.h"

#include <d3d11.h>

struct EasuTensorData
{
    int fpX;
    int fpY;
    float weights[12];
};

#if defined(__CUDACC__)
__device__
EasuTensorData easuGetTensorData(
    const uint8_t* input,
    int inputWidth,
    int inputHeight,
    int outputX,
    int outputY,
    int outputWidth,
    int outputHeight);
#endif

struct EasuTensorPipeline
{
    int inputWidth = 0;
    int inputHeight = 0;
    int outputWidth = 0;
    int outputHeight = 0;

    void* dInput = nullptr;
    void* dOutput = nullptr;

    void* captureResource = nullptr;
    ID3D11Texture2D* registeredTexture = nullptr;

    void* outputResource = nullptr;
    ID3D11Texture2D* registeredOutputTexture = nullptr;

    bool initialized = false;
};

bool initEasuTensorPipeline(
    EasuTensorPipeline& pipeline,
    int inputWidth,
    int inputHeight,
    int outputWidth,
    int outputHeight);

bool attachEasuTensorCaptureTexture(
    EasuTensorPipeline& pipeline,
    ID3D11Texture2D* texture);

bool attachEasuTensorOutputTexture(
    EasuTensorPipeline& pipeline,
    ID3D11Texture2D* texture);

void detachEasuTensorCaptureTexture(
    EasuTensorPipeline& pipeline);

void detachEasuTensorOutputTexture(
    EasuTensorPipeline& pipeline);

void destroyEasuTensorPipeline(
    EasuTensorPipeline& pipeline);

Image upscaleEasuTensor(
    EasuTensorPipeline& pipeline,
    const Image& input,
    float* kernelMs);

bool upscaleEasuTensorD3D11(
    EasuTensorPipeline& pipeline,
    ID3D11Texture2D* captureTexture,
    ID3D11Texture2D* outputTexture,
    float* kernelMs);