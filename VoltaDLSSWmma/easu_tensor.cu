#include "easu_tensor.h"

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <cuda_fp16.h>

#include <mma.h>

#include <d3d11.h>

#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace nvcuda;

namespace
{

constexpr int kThreadsPerBlock = 32;
constexpr int kPixelsPerBlock = 16;
constexpr int kEasuTaps = 12;

#define CUDA_CHECK(call)                                                    \
    do                                                                      \
    {                                                                       \
        cudaError_t _err = (call);                                          \
        if (_err != cudaSuccess)                                            \
        {                                                                   \
            std::cerr                                                       \
                << "CUDA error: "                                               \
                << cudaGetErrorString(_err)                                 \
                << " at "                                                       \
                << __FILE__                                                  \
                << ":"                                                         \
                << __LINE__                                                 \
                << '\n';                                                    \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

struct D3D11InteropTimings
{
    double inputMapMs = 0.0;
    double inputCopyMs = 0.0;
    double inputUnmapMs = 0.0;

    double kernelCpuMs = 0.0;
    double kernelGpuMs = 0.0;

    double outputMapMs = 0.0;
    double outputCopyMs = 0.0;
    double outputUnmapMs = 0.0;

    double finalSynchronizeMs = 0.0;

    double totalMs = 0.0;
};

} // namespace


__global__
void easuTensorKernel(
    const uint8_t* __restrict__ input,
    int inputWidth,
    int inputHeight,
    uint8_t* __restrict__ output,
    int outputWidth,
    int outputHeight)
{
    const int lane =
        static_cast<int>(threadIdx.x);

    const int outputY =
        static_cast<int>(blockIdx.y);

    const int groupX =
        static_cast<int>(blockIdx.x) *
        kPixelsPerBlock;

    if (lane >= kThreadsPerBlock ||
        outputY >= outputHeight)
    {
        return;
    }

    /*
        One warp processes 16 output pixels.

        The WMMA path operates on 16x16 half matrices.
    */

    __shared__ half weights[16 * 16];
    __shared__ half samples[16 * 16];
    __shared__ float result[16 * 16];

    /*
        BGRA byte offset for every tap/pixel combination.
    */

    __shared__
    int sampleByteIndex[
        kEasuTaps * kPixelsPerBlock];

    // Cache each BGRA source pixel once.
    __shared__
    uchar4 samplePixels[
        kEasuTaps * kPixelsPerBlock];

    /*
        Clear shared memory.
    */

    for (int index = lane;
         index < 16 * 16;
         index += kThreadsPerBlock)
    {
        weights[index] =
            __float2half(0.0f);

        samples[index] =
            __float2half(0.0f);

        result[index] =
            0.0f;
    }

    for (int index = lane;
         index < kEasuTaps * kPixelsPerBlock;
         index += kThreadsPerBlock)
    {
        sampleByteIndex[index] =
            0;
    }

    __syncthreads();

    /*
        Generate EASU data and source sample addresses.

        easuGetTensorData() is defined in easu_reference.cu.
    */

    if (lane < kPixelsPerBlock)
    {
        const int outputX =
            groupX + lane;

        if (outputX < outputWidth)
        {
            const EasuTensorData data =
                easuGetTensorData(
                    input,
                    inputWidth,
                    inputHeight,
                    outputX,
                    outputY,
                    outputWidth,
                    outputHeight);

            #pragma unroll

            for (int tap = 0;
                 tap < kEasuTaps;
                 ++tap)
            {
                weights[
                    lane * 16 +
                    tap] =
                    __float2half(
                        data.weights[tap]);
            }

            const int offsetsX[kEasuTaps] =
            {
                 0,  1,
                -1,  0,
                 1, -1,
                 2,  2,
                 0,  1,
                 1,  0
            };

            const int offsetsY[kEasuTaps] =
            {
                -1, -1,
                 1,  1,
                 0,  0,
                 1,  0,
                 1,  1,
                 2,  2
            };

            #pragma unroll

            for (int tap = 0;
                 tap < kEasuTaps;
                 ++tap)
            {
                int sampleX =
                    data.fpX +
                    offsetsX[tap];

                int sampleY =
                    data.fpY +
                    offsetsY[tap];

                sampleX =
                    max(
                        0,
                        min(
                            inputWidth - 1,
                            sampleX));

                sampleY =
                    max(
                        0,
                        min(
                            inputHeight - 1,
                            sampleY));

                const size_t byteIndex =
                    (
                        static_cast<size_t>(
                            sampleY) *
                        static_cast<size_t>(
                            inputWidth) +
                        static_cast<size_t>(
                            sampleX)
                    ) *
                    4u;

                sampleByteIndex[
                    tap * kPixelsPerBlock +
                    lane] =
                    static_cast<int>(
                        byteIndex);

                samplePixels[
                    tap * kPixelsPerBlock +
                    lane] =
                    *reinterpret_cast<const uchar4*>(
                        input + byteIndex);
            }
        }
    }

    __syncthreads();

    using MatrixA =
        wmma::fragment<
            wmma::matrix_a,
            16,
            16,
            16,
            half,
            wmma::row_major>;

    MatrixA a;

    wmma::load_matrix_sync(
        a,
        weights,
        16);

    /*
        Process R, G and B independently.
    */

    #pragma unroll 3

    for (int channel = 0;
         channel < 3;
         ++channel)
    {
        if (lane < kPixelsPerBlock)
        {
            const int outputX =
                groupX + lane;

            if (outputX < outputWidth)
            {
                #pragma unroll
                for (int tap = 0;
                     tap < kEasuTaps;
                     ++tap)
                {
                    const uchar4 pixel =
                        samplePixels[
                            tap * kPixelsPerBlock +
                            lane];

                    const float value =
                        (
                            channel == 0
                                ? static_cast<float>(pixel.z)
                                : channel == 1
                                    ? static_cast<float>(pixel.y)
                                    : static_cast<float>(pixel.x)
                        ) /
                        255.0f;

                    samples[
                        tap * 16 +
                        lane] =
                        __float2half(
                            value);
                }
            }
        }

        __syncthreads();


        using MatrixB =
            wmma::fragment<
                wmma::matrix_b,
                16,
                16,
                16,
                half,
                wmma::row_major>;

        using Accumulator =
            wmma::fragment<
                wmma::accumulator,
                16,
                16,
                16,
                float>;

        MatrixB b;
        Accumulator c;

        wmma::fill_fragment(
            c,
            0.0f);


        wmma::load_matrix_sync(
            b,
            samples,
            16);

        wmma::mma_sync(
            c,
            a,
            b,
            c);

        wmma::store_matrix_sync(
            result,
            c,
            16,
            wmma::mem_row_major);

        __syncthreads();

        if (lane < kPixelsPerBlock)
        {
            const int outputX =
                groupX + lane;

            if (outputX < outputWidth)
            {
                float value =
                    result[
                        lane * 16 +
                        lane];

                const float s0 =
                    __half2float(
                        samples[
                            0 * 16 +
                            lane]);

                const float s1 =
                    __half2float(
                        samples[
                            1 * 16 +
                            lane]);

                const float s2 =
                    __half2float(
                        samples[
                            2 * 16 +
                            lane]);

                const float s3 =
                    __half2float(
                        samples[
                            3 * 16 +
                            lane]);

                const float minValue =
                    fminf(
                        fminf(
                            s0,
                            s1),
                        fminf(
                            s2,
                            s3));

                const float maxValue =
                    fmaxf(
                        fmaxf(
                            s0,
                            s1),
                        fmaxf(
                            s2,
                            s3));

                value =
                    fminf(
                        maxValue,
                        fmaxf(
                            minValue,
                            value));

                value =
                    fminf(
                        1.0f,
                        fmaxf(
                            0.0f,
                            value));

                const int outputChannel = channel;

                const size_t outputIndex =
                    (
                        static_cast<size_t>(
                            outputY) *
                        static_cast<size_t>(
                            outputWidth) +
                        static_cast<size_t>(
                            outputX)
                    ) *
                    4u +
                    static_cast<size_t>(
                        outputChannel);

                output[
                    outputIndex] =
                    static_cast<uint8_t>(
                        value * 255.0f +
                        0.5f);
            }
        }

        __syncthreads();
    }

    /*
        Alpha channel.
    */

    if (lane < kPixelsPerBlock)
    {
        const int outputX =
            groupX + lane;

        if (outputX < outputWidth)
        {
            const size_t outputIndex =
                (
                    static_cast<size_t>(
                        outputY) *
                    static_cast<size_t>(
                        outputWidth) +
                    static_cast<size_t>(
                        outputX)
                ) *
                4u +
                3u;

            output[
                outputIndex] =
                255;
        }
    }
}

__global__
void easuTensorKernelDirect(
    cudaSurfaceObject_t inputSurface,
    int inputWidth,
    int inputHeight,
    cudaSurfaceObject_t outputSurface,
    int outputWidth,
    int outputHeight)
{
    const int lane =
        static_cast<int>(threadIdx.x);

    const int outputY =
        static_cast<int>(blockIdx.y);

    const int groupX =
        static_cast<int>(blockIdx.x) *
        kPixelsPerBlock;

    if (lane >= kThreadsPerBlock ||
        outputY >= outputHeight)
    {
        return;
    }

    /*
        One warp processes 16 output pixels.

        The WMMA path operates on 16x16 half matrices.
    */

    __shared__ half weights[16 * 16];
    __shared__ half samples[16 * 16];
    __shared__ float result[16 * 16];



    // Cache each BGRA source pixel once.
    __shared__
    uchar4 samplePixels[
        kEasuTaps * kPixelsPerBlock];

    /*
        Clear shared memory.
    */

    for (int index = lane;
         index < 16 * 16;
         index += kThreadsPerBlock)
    {
        weights[index] =
            __float2half(0.0f);

        samples[index] =
            __float2half(0.0f);

        result[index] =
            0.0f;
    }


    __syncwarp();

    /*
        Generate EASU data and source sample addresses.

        easuGetTensorData() is defined in easu_reference.cu.
    */

    if (lane < kPixelsPerBlock)
    {
        const int outputX =
            groupX + lane;

        if (outputX < outputWidth)
        {
            const EasuTensorData data =
                easuGetTensorDataSurface(
                    inputSurface,
                    inputWidth,
                    inputHeight,
                    outputX,
                    outputY,
                    outputWidth,
                    outputHeight);

            #pragma unroll

            for (int tap = 0;
                 tap < kEasuTaps;
                 ++tap)
            {
                weights[
                    lane * 16 +
                    tap] =
                    __float2half(
                        data.weights[tap]);
            }

            const int offsetsX[kEasuTaps] =
            {
                 0,  1,
                -1,  0,
                 1, -1,
                 2,  2,
                 0,  1,
                 1,  0
            };

            const int offsetsY[kEasuTaps] =
            {
                -1, -1,
                 1,  1,
                 0,  0,
                 1,  0,
                 1,  1,
                 2,  2
            };

            #pragma unroll

            for (int tap = 0;
                 tap < kEasuTaps;
                 ++tap)
            {
                int sampleX =
                    data.fpX +
                    offsetsX[tap];

                int sampleY =
                    data.fpY +
                    offsetsY[tap];

                sampleX =
                    max(
                        0,
                        min(
                            inputWidth - 1,
                            sampleX));

                sampleY =
                    max(
                        0,
                        min(
                            inputHeight - 1,
                            sampleY));

                samplePixels[
                    tap * kPixelsPerBlock +
                    lane] =
                    surf2Dread<uchar4>(
                        inputSurface,
                        sampleX * 4,
                        sampleY);
            }
        }
    }

    __syncwarp();

    using MatrixA =
        wmma::fragment<
            wmma::matrix_a,
            16,
            16,
            16,
            half,
            wmma::row_major>;

    MatrixA a;

    wmma::load_matrix_sync(
        a,
        weights,
        16);

    /*
        Process R, G and B independently.
    */

    uchar4 outputPixel =
        make_uchar4(
            0,
            0,
            0,
            255);

    #pragma unroll 3

    for (int channel = 0;
         channel < 3;
         ++channel)
    {
        if (lane < kPixelsPerBlock)
        {
            const int outputX =
                groupX + lane;

            if (outputX < outputWidth)
            {
                #pragma unroll
                for (int tap = 0;
                     tap < kEasuTaps;
                     ++tap)
                {
                    const uchar4 pixel =
                        samplePixels[
                            tap * kPixelsPerBlock +
                            lane];

                    const float value =
                        (
                            channel == 0
                                ? static_cast<float>(pixel.z)
                                : channel == 1
                                    ? static_cast<float>(pixel.y)
                                    : static_cast<float>(pixel.x)
                        ) /
                        255.0f;

                    samples[
                        tap * 16 +
                        lane] =
                        __float2half(
                            value);
                }
            }
        }

        __syncwarp();


        using MatrixB =
            wmma::fragment<
                wmma::matrix_b,
                16,
                16,
                16,
                half,
                wmma::row_major>;

        using Accumulator =
            wmma::fragment<
                wmma::accumulator,
                16,
                16,
                16,
                float>;

        MatrixB b;
        Accumulator c;

        wmma::fill_fragment(
            c,
            0.0f);


        wmma::load_matrix_sync(
            b,
            samples,
            16);

        wmma::mma_sync(
            c,
            a,
            b,
            c);

        wmma::store_matrix_sync(
            result,
            c,
            16,
            wmma::mem_row_major);

        __syncwarp();

        if (lane < kPixelsPerBlock)
        {
            const int outputX =
                groupX + lane;

            if (outputX < outputWidth)
            {
                float value =
                    result[
                        lane * 16 +
                        lane];

                const float s0 =
                    __half2float(
                        samples[
                            0 * 16 +
                            lane]);

                const float s1 =
                    __half2float(
                        samples[
                            1 * 16 +
                            lane]);

                const float s2 =
                    __half2float(
                        samples[
                            2 * 16 +
                            lane]);

                const float s3 =
                    __half2float(
                        samples[
                            3 * 16 +
                            lane]);

                const float minValue =
                    fminf(
                        fminf(
                            s0,
                            s1),
                        fminf(
                            s2,
                            s3));

                const float maxValue =
                    fmaxf(
                        fmaxf(
                            s0,
                            s1),
                        fmaxf(
                            s2,
                            s3));

                value =
                    fminf(
                        maxValue,
                        fmaxf(
                            minValue,
                            value));

                value =
                    fminf(
                        1.0f,
                        fmaxf(
                            0.0f,
                            value));

                const uint8_t outValue =
                    static_cast<uint8_t>(
                        value * 255.0f +
                        0.5f);

                if (channel == 0)
                    outputPixel.x = outValue;
                else if (channel == 1)
                    outputPixel.y = outValue;
                else
                    outputPixel.z = outValue;
            }
        }

        __syncwarp();
    }

        /*
        Direct CUDA surface output.
    */

    if (lane < kPixelsPerBlock)
    {
        const int outputX =
            groupX + lane;

        if (outputX < outputWidth)
        {
            surf2Dwrite<uchar4>(
                outputPixel,
                outputSurface,
                outputX * 4,
                outputY);
        }
    }
}



static void freeCudaPipeline(
    EasuTensorPipeline& pipeline)
{
    detachEasuTensorCaptureTexture(
        pipeline);

    detachEasuTensorOutputTexture(
        pipeline);

    if (pipeline.dInput != nullptr)
    {
        CUDA_CHECK(
            cudaFree(
                pipeline.dInput));

        pipeline.dInput =
            nullptr;
    }

    if (pipeline.dOutput != nullptr)
    {
        CUDA_CHECK(
            cudaFree(
                pipeline.dOutput));

        pipeline.dOutput =
            nullptr;
    }

    pipeline.initialized =
        false;
}


bool initEasuTensorPipeline(
    EasuTensorPipeline& pipeline,
    int inputWidth,
    int inputHeight,
    int outputWidth,
    int outputHeight)
{
    if (pipeline.initialized &&
        pipeline.inputWidth == inputWidth &&
        pipeline.inputHeight == inputHeight &&
        pipeline.outputWidth == outputWidth &&
        pipeline.outputHeight == outputHeight)
    {
        return true;
    }

    if (pipeline.initialized)
    {
        freeCudaPipeline(
            pipeline);
    }

    pipeline.inputWidth =
        inputWidth;

    pipeline.inputHeight =
        inputHeight;

    pipeline.outputWidth =
        outputWidth;

    pipeline.outputHeight =
        outputHeight;

    const size_t inputBytes =
        static_cast<size_t>(
            inputWidth) *
        static_cast<size_t>(
            inputHeight) *
        4u;

    const size_t outputBytes =
        static_cast<size_t>(
            outputWidth) *
        static_cast<size_t>(
            outputHeight) *
        4u;

    cudaError_t err =
        cudaMalloc(
            &pipeline.dInput,
            inputBytes);

    if (err != cudaSuccess)
    {
        std::cerr
            << "cudaMalloc(dInput) failed: "
            << cudaGetErrorString(err)
            << '\n';

        pipeline.dInput =
            nullptr;

        return false;
    }

    err =
        cudaMalloc(
            &pipeline.dOutput,
            outputBytes);

    if (err != cudaSuccess)
    {
        std::cerr
            << "cudaMalloc(dOutput) failed: "
            << cudaGetErrorString(err)
            << '\n';

        CUDA_CHECK(
            cudaFree(
                pipeline.dInput));

        pipeline.dInput =
            nullptr;

        return false;
    }

    pipeline.captureResource =
        nullptr;

    pipeline.registeredTexture =
        nullptr;

    pipeline.outputResource =
        nullptr;

    pipeline.registeredOutputTexture =
        nullptr;

    pipeline.initialized =
        true;

    return true;
}


bool attachEasuTensorCaptureTexture(
    EasuTensorPipeline& pipeline,
    ID3D11Texture2D* texture)
{
    if (!pipeline.initialized ||
        texture == nullptr)
    {
        return false;
    }

    if (pipeline.registeredTexture ==
            texture &&
        pipeline.captureResource !=
            nullptr)
    {
        return true;
    }

    detachEasuTensorCaptureTexture(
        pipeline);

    cudaGraphicsResource_t resource =
        nullptr;

    cudaError_t err =
        cudaGraphicsD3D11RegisterResource(
            &resource,
            texture,
            cudaGraphicsRegisterFlagsSurfaceLoadStore);

    if (err != cudaSuccess)
    {
        std::cerr
            << "cudaGraphicsD3D11RegisterResource(input) failed: "
            << cudaGetErrorString(err)
            << '\n';

        return false;
    }

    err = cudaGraphicsResourceSetMapFlags(
        resource,
        cudaGraphicsMapFlagsReadOnly);

    if (err != cudaSuccess)
    {
        std::cerr
            << "cudaGraphicsResourceSetMapFlags(input) failed: "
            << cudaGetErrorString(err)
            << '\n';

        cudaGraphicsUnregisterResource(
            resource);

        return false;
    }

    pipeline.captureResource =
        static_cast<void*>(
            resource);

    pipeline.registeredTexture =
        texture;

    return true;
}


bool attachEasuTensorOutputTexture(
    EasuTensorPipeline& pipeline,
    ID3D11Texture2D* texture)
{
    if (!pipeline.initialized ||
        texture == nullptr)
    {
        return false;
    }

    if (pipeline.registeredOutputTexture ==
            texture &&
        pipeline.outputResource !=
            nullptr)
    {
        return true;
    }

    detachEasuTensorOutputTexture(
        pipeline);

    cudaGraphicsResource_t resource =
        nullptr;

    cudaError_t err =
        cudaGraphicsD3D11RegisterResource(
            &resource,
            texture,
            cudaGraphicsRegisterFlagsSurfaceLoadStore);

    if (err != cudaSuccess)
    {
        std::cerr
            << "cudaGraphicsD3D11RegisterResource(output) failed: "
            << cudaGetErrorString(err)
            << '\n';

        return false;
    }

    err = cudaGraphicsResourceSetMapFlags(
        resource,
        cudaGraphicsMapFlagsWriteDiscard);

    if (err != cudaSuccess)
    {
        std::cerr
            << "cudaGraphicsResourceSetMapFlags(output) failed: "
            << cudaGetErrorString(err)
            << '\n';

        cudaGraphicsUnregisterResource(
            resource);

        return false;
    }

    pipeline.outputResource =
        static_cast<void*>(
            resource);

    pipeline.registeredOutputTexture =
        texture;

    return true;
}


void detachEasuTensorCaptureTexture(
    EasuTensorPipeline& pipeline)
{
    if (pipeline.captureResource ==
        nullptr)
    {
        pipeline.registeredTexture =
            nullptr;

        return;
    }

    cudaGraphicsResource_t resource =
        static_cast<
            cudaGraphicsResource_t>(
                pipeline.captureResource);

    cudaError_t err =
        cudaGraphicsUnregisterResource(
            resource);

    if (err != cudaSuccess)
    {
        std::cerr
            << "cudaGraphicsUnregisterResource(input) failed: "
            << cudaGetErrorString(err)
            << '\n';
    }

    pipeline.captureResource =
        nullptr;

    pipeline.registeredTexture =
        nullptr;
}


void detachEasuTensorOutputTexture(
    EasuTensorPipeline& pipeline)
{
    if (pipeline.outputResource ==
        nullptr)
    {
        pipeline.registeredOutputTexture =
            nullptr;

        return;
    }

    cudaGraphicsResource_t resource =
        static_cast<
            cudaGraphicsResource_t>(
                pipeline.outputResource);

    cudaError_t err =
        cudaGraphicsUnregisterResource(
            resource);

    if (err != cudaSuccess)
    {
        std::cerr
            << "cudaGraphicsUnregisterResource(output) failed: "
            << cudaGetErrorString(err)
            << '\n';
    }

    pipeline.outputResource =
        nullptr;

    pipeline.registeredOutputTexture =
        nullptr;
}


void destroyEasuTensorPipeline(
    EasuTensorPipeline& pipeline)
{
    freeCudaPipeline(
        pipeline);

    pipeline.inputWidth =
        0;

    pipeline.inputHeight =
        0;

    pipeline.outputWidth =
        0;

    pipeline.outputHeight =
        0;

    pipeline.captureResource =
        nullptr;

    pipeline.registeredTexture =
        nullptr;

    pipeline.outputResource =
        nullptr;

    pipeline.registeredOutputTexture =
        nullptr;
}


static void launchEasuKernel(
    EasuTensorPipeline& pipeline)
{
    const dim3 blocks(
        static_cast<unsigned int>(
            (pipeline.outputWidth + 15) / 16),
        static_cast<unsigned int>(
            pipeline.outputHeight),
        1);

    const dim3 threads(
        kThreadsPerBlock,
        1,
        1);

    if (pipeline.directSurfaceActive)
    {
        easuTensorKernelDirect<<<
            blocks,
            threads
        >>>(
            pipeline.inputSurface,
            pipeline.inputWidth,
            pipeline.inputHeight,
            pipeline.outputSurface,
            pipeline.outputWidth,
            pipeline.outputHeight);
    }
    else
    {
        easuTensorKernel<<<
            blocks,
            threads
        >>>(
            static_cast<const uint8_t*>(
                pipeline.dInput),
            pipeline.inputWidth,
            pipeline.inputHeight,
            static_cast<uint8_t*>(
                pipeline.dOutput),
            pipeline.outputWidth,
            pipeline.outputHeight);
    }

    CUDA_CHECK(
        cudaGetLastError());
}


static void copyD3D11TextureToInput(
    EasuTensorPipeline& pipeline,
    D3D11InteropTimings& timings)
{
    cudaGraphicsResource_t resource =
        static_cast<
            cudaGraphicsResource_t>(
                pipeline.captureResource);

    auto start =
        std::chrono::steady_clock::now();

    CUDA_CHECK(
        cudaGraphicsMapResources(
            1,
            &resource,
            0));

    auto end =
        std::chrono::steady_clock::now();

    timings.inputMapMs =
        std::chrono::duration<double,
            std::milli>(
            end -
            start).count();

    cudaArray_t array =
        nullptr;

    start =
        std::chrono::steady_clock::now();

    CUDA_CHECK(
        cudaGraphicsSubResourceGetMappedArray(
            &array,
            resource,
            0,
            0));

    const size_t widthBytes =
        static_cast<size_t>(
            pipeline.inputWidth) *
        4u;

    CUDA_CHECK(
        cudaMemcpy2DFromArray(
            pipeline.dInput,
            widthBytes,
            array,
            0,
            0,
            widthBytes,
            static_cast<size_t>(
                pipeline.inputHeight),
            cudaMemcpyDeviceToDevice));

    end =
        std::chrono::steady_clock::now();

    timings.inputCopyMs =
        std::chrono::duration<double,
            std::milli>(
            end -
            start).count();

    start =
        std::chrono::steady_clock::now();

    CUDA_CHECK(
        cudaGraphicsUnmapResources(
            1,
            &resource,
            0));

    end =
        std::chrono::steady_clock::now();

    timings.inputUnmapMs =
        std::chrono::duration<double,
            std::milli>(
            end -
            start).count();
}


static void copyOutputToD3D11Texture(
    EasuTensorPipeline& pipeline,
    D3D11InteropTimings& timings)
{
    cudaGraphicsResource_t resource =
        static_cast<
            cudaGraphicsResource_t>(
                pipeline.outputResource);

    auto start =
        std::chrono::steady_clock::now();

    CUDA_CHECK(
        cudaGraphicsMapResources(
            1,
            &resource,
            0));

    auto end =
        std::chrono::steady_clock::now();

    timings.outputMapMs =
        std::chrono::duration<double,
            std::milli>(
            end -
            start).count();

    cudaArray_t array =
        nullptr;

    start =
        std::chrono::steady_clock::now();

    CUDA_CHECK(
        cudaGraphicsSubResourceGetMappedArray(
            &array,
            resource,
            0,
            0));

    const size_t widthBytes =
        static_cast<size_t>(
            pipeline.outputWidth) *
        4u;

    CUDA_CHECK(
        cudaMemcpy2DToArray(
            array,
            0,
            0,
            pipeline.dOutput,
            widthBytes,
            widthBytes,
            static_cast<size_t>(
                pipeline.outputHeight),
            cudaMemcpyDeviceToDevice));

    end =
        std::chrono::steady_clock::now();

    timings.outputCopyMs =
        std::chrono::duration<double,
            std::milli>(
            end -
            start).count();

    start =
        std::chrono::steady_clock::now();

    CUDA_CHECK(
        cudaGraphicsUnmapResources(
            1,
            &resource,
            0));

    end =
        std::chrono::steady_clock::now();

    timings.outputUnmapMs =
        std::chrono::duration<double,
            std::milli>(
            end -
            start).count();
}


static float runEasuOnce(
    EasuTensorPipeline& pipeline)
{
    cudaEvent_t start =
        nullptr;

    cudaEvent_t stop =
        nullptr;

    CUDA_CHECK(
        cudaEventCreate(
            &start));

    CUDA_CHECK(
        cudaEventCreate(
            &stop));

    CUDA_CHECK(
        cudaEventRecord(
            start,
            0));

    launchEasuKernel(
        pipeline);

    CUDA_CHECK(
        cudaEventRecord(
            stop,
            0));

    CUDA_CHECK(
        cudaEventSynchronize(
            stop));

    float elapsedMs =
        0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &elapsedMs,
            start,
            stop));

    CUDA_CHECK(
        cudaEventDestroy(
            start));

    CUDA_CHECK(
        cudaEventDestroy(
            stop));

    return elapsedMs;
}


static void runEasuTimed(
    EasuTensorPipeline& pipeline,
    float* kernelMs)
{
    constexpr int kWarmupIterations =
        5;

    constexpr int kIterations =
        30;

    for (int i = 0;
         i < kWarmupIterations;
         ++i)
    {
        launchEasuKernel(
            pipeline);
    }

    CUDA_CHECK(
        cudaDeviceSynchronize());

    cudaEvent_t start =
        nullptr;

    cudaEvent_t stop =
        nullptr;

    CUDA_CHECK(
        cudaEventCreate(
            &start));

    CUDA_CHECK(
        cudaEventCreate(
            &stop));

    CUDA_CHECK(
        cudaEventRecord(
            start,
            0));

    for (int i = 0;
         i < kIterations;
         ++i)
    {
        launchEasuKernel(
            pipeline);
    }

    CUDA_CHECK(
        cudaEventRecord(
            stop,
            0));

    CUDA_CHECK(
        cudaEventSynchronize(
            stop));

    float elapsedMs =
        0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &elapsedMs,
            start,
            stop));

    CUDA_CHECK(
        cudaEventDestroy(
            start));

    CUDA_CHECK(
        cudaEventDestroy(
            stop));

    if (kernelMs != nullptr)
    {
        *kernelMs =
            elapsedMs /
            static_cast<float>(
                kIterations);
    }
}


static Image copyOutputToHost(
    EasuTensorPipeline& pipeline)
{
    Image output{};

    output.width =
        pipeline.outputWidth;

    output.height =
        pipeline.outputHeight;

    output.pixels.resize(
        static_cast<size_t>(
            pipeline.outputWidth) *
        static_cast<size_t>(
            pipeline.outputHeight) *
        4u);

    CUDA_CHECK(
        cudaMemcpy(
            output.pixels.data(),
            pipeline.dOutput,
            output.pixels.size(),
            cudaMemcpyDeviceToHost));

    return output;
}


Image upscaleEasuTensor(
    EasuTensorPipeline& pipeline,
    const Image& input,
    float* kernelMs)
{
    if (input.width <= 0 ||
        input.height <= 0 ||
        input.pixels.empty())
    {
        return {};
    }

    if (!initEasuTensorPipeline(
            pipeline,
            input.width,
            input.height,
            pipeline.outputWidth,
            pipeline.outputHeight))
    {
        std::cerr
            << "Failed to initialize EASU Tensor pipeline.\n";

        return {};
    }

    CUDA_CHECK(
        cudaMemcpy(
            pipeline.dInput,
            input.pixels.data(),
            input.pixels.size(),
            cudaMemcpyHostToDevice));

    runEasuTimed(
        pipeline,
        kernelMs);

    return copyOutputToHost(
        pipeline);
}


bool upscaleEasuTensorD3D11(
    EasuTensorPipeline& pipeline,
    ID3D11Texture2D* captureTexture,
    ID3D11Texture2D* outputTexture,
    float* kernelMs)
{
    if (captureTexture == nullptr ||
        outputTexture == nullptr)
    {
        std::cerr
            << "upscaleEasuTensorD3D11: null texture.\n";

        return false;
    }

    if (!pipeline.initialized)
    {
        std::cerr
            << "upscaleEasuTensorD3D11: pipeline not initialized.\n";

        return false;
    }

    D3D11_TEXTURE2D_DESC inputDesc{};

    captureTexture->GetDesc(
        &inputDesc);

    if (static_cast<int>(
            inputDesc.Width) !=
            pipeline.inputWidth ||
        static_cast<int>(
            inputDesc.Height) !=
            pipeline.inputHeight)
    {
        std::cerr
            << "Input texture size mismatch: "
            << inputDesc.Width
            << "x"
            << inputDesc.Height
            << " expected "
            << pipeline.inputWidth
            << "x"
            << pipeline.inputHeight
            << '\n';

        return false;
    }

    D3D11_TEXTURE2D_DESC outputDesc{};

    outputTexture->GetDesc(
        &outputDesc);

    if (static_cast<int>(
            outputDesc.Width) !=
            pipeline.outputWidth ||
        static_cast<int>(
            outputDesc.Height) !=
            pipeline.outputHeight)
    {
        std::cerr
            << "Output texture size mismatch: "
            << outputDesc.Width
            << "x"
            << outputDesc.Height
            << " expected "
            << pipeline.outputWidth
            << "x"
            << pipeline.outputHeight
            << '\n';

        return false;
    }

    if (!attachEasuTensorCaptureTexture(
            pipeline,
            captureTexture))
    {
        return false;
    }

    if (!attachEasuTensorOutputTexture(
            pipeline,
            outputTexture))
    {
        return false;
    }

    D3D11InteropTimings timings{};

    const auto totalStart =
        std::chrono::steady_clock::now();

    /*
        ---------------------------------------------------------
        DIRECT D3D11 -> CUDA ARRAY PATH
        ---------------------------------------------------------
    */

    cudaGraphicsResource_t directResources[2] =
    {
        static_cast<cudaGraphicsResource_t>(
            pipeline.captureResource),

        static_cast<cudaGraphicsResource_t>(
            pipeline.outputResource)
    };

    const auto mapStart =
        std::chrono::steady_clock::now();

    CUDA_CHECK(
        cudaGraphicsMapResources(
            2,
            directResources,
            0));

    const auto mapEnd =
        std::chrono::steady_clock::now();

    timings.inputMapMs =
        std::chrono::duration<double,
            std::milli>(
            mapEnd -
            mapStart).count();

    timings.inputCopyMs =
        0.0;

    timings.inputUnmapMs =
        0.0;

    timings.outputMapMs =
        0.0;

    timings.outputCopyMs =
        0.0;

    cudaArray_t inputArray =
        nullptr;

    cudaArray_t outputArray =
        nullptr;

    CUDA_CHECK(
        cudaGraphicsSubResourceGetMappedArray(
            &inputArray,
            directResources[0],
            0,
            0));

    CUDA_CHECK(
        cudaGraphicsSubResourceGetMappedArray(
            &outputArray,
            directResources[1],
            0,
            0));

    cudaResourceDesc inputSurfaceDesc{};

    inputSurfaceDesc.resType =
        cudaResourceTypeArray;

    inputSurfaceDesc.res.array.array =
        inputArray;

    cudaResourceDesc outputSurfaceDesc{};

    outputSurfaceDesc.resType =
        cudaResourceTypeArray;

    outputSurfaceDesc.res.array.array =
        outputArray;

    CUDA_CHECK(
        cudaCreateSurfaceObject(
            &pipeline.inputSurface,
            &inputSurfaceDesc));

    CUDA_CHECK(
        cudaCreateSurfaceObject(
            &pipeline.outputSurface,
            &outputSurfaceDesc));

    pipeline.directSurfaceActive =
        true;

    /*
        ---------------------------------------------------------
        KERNEL
        ---------------------------------------------------------
    */

    const auto kernelCpuStart =
        std::chrono::steady_clock::now();

    const float measuredKernelMs =
        runEasuOnce(
            pipeline);

    const auto kernelCpuEnd =
        std::chrono::steady_clock::now();

    timings.kernelCpuMs =
        std::chrono::duration<double,
            std::milli>(
            kernelCpuEnd -
            kernelCpuStart).count();

    timings.kernelGpuMs =
        measuredKernelMs;

    /*
        ---------------------------------------------------------
        SURFACE OBJECT CLEANUP
        ---------------------------------------------------------
    */

    CUDA_CHECK(
        cudaDestroySurfaceObject(
            pipeline.inputSurface));

    CUDA_CHECK(
        cudaDestroySurfaceObject(
            pipeline.outputSurface));

    pipeline.inputSurface =
        0;

    pipeline.outputSurface =
        0;

    pipeline.directSurfaceActive =
        false;

    /*
        ---------------------------------------------------------
        UNMAP
        ---------------------------------------------------------
    */

    const auto unmapStart =
        std::chrono::steady_clock::now();

    CUDA_CHECK(
        cudaGraphicsUnmapResources(
            2,
            directResources,
            0));

    const auto unmapEnd =
        std::chrono::steady_clock::now();

    timings.inputUnmapMs =
        0.0;

    timings.outputUnmapMs =
        std::chrono::duration<double,
            std::milli>(
            unmapEnd -
            unmapStart).count();

    timings.finalSynchronizeMs =
        0.0;
    const auto totalEnd =
        std::chrono::steady_clock::now();

    timings.totalMs =
        std::chrono::duration<double,
            std::milli>(
            totalEnd -
            totalStart).count();

    if (kernelMs != nullptr)
    {
        *kernelMs =
            measuredKernelMs;
    }

    /*
        ---------------------------------------------------------
        DIAGNOSTIC OUTPUT
        ---------------------------------------------------------
    */

    if (timings.totalMs > 20.0)
    {
        std::cout
            << "\nCUDA/D3D11 stall: "
            << std::fixed
            << std::setprecision(3)
            << "Total="
            << timings.totalMs
            << " ms"
            << " | InMap="
            << timings.inputMapMs
            << " ms"
            << " | InCopy="
            << timings.inputCopyMs
            << " ms"
            << " | InUnmap="
            << timings.inputUnmapMs
            << " ms"
            << " | KernelCPU="
            << timings.kernelCpuMs
            << " ms"
            << " | KernelGPU="
            << timings.kernelGpuMs
            << " ms"
            << " | OutMap="
            << timings.outputMapMs
            << " ms"
            << " | OutCopy="
            << timings.outputCopyMs
            << " ms"
            << " | OutUnmap="
            << timings.outputUnmapMs
            << " ms"
            << " | FinalSync="
            << timings.finalSynchronizeMs
            << " ms"
            << '\n';
    }

    return true;
}
