#include "easu_reference.h"
#include "easu_tensor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#define EASU_CHECK(call)                                                    \
    do                                                                      \
    {                                                                       \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess)                                             \
        {                                                                    \
            std::cerr << "CUDA error: "                                          \
                      << cudaGetErrorString(err)                            \
                      << " at " << __FILE__                                 \
                      << ":" << __LINE__ << '\n';                            \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

struct EasuFloat2
{
    float x;
    float y;
};

struct EasuFloat3
{
    float x;
    float y;
    float z;
};

struct EasuFloat4
{
    float x;
    float y;
    float z;
    float w;
};

struct EasuConstants
{
    EasuFloat4 con0;
    EasuFloat4 con1;
    EasuFloat4 con2;
    EasuFloat4 con3;
};

struct EasuSamples
{
    EasuFloat3 b;
    EasuFloat3 c;
    EasuFloat3 e;
    EasuFloat3 f;
    EasuFloat3 g;
    EasuFloat3 h;
    EasuFloat3 i;
    EasuFloat3 j;
    EasuFloat3 k;
    EasuFloat3 l;
    EasuFloat3 n;
    EasuFloat3 o;
};

__host__ __device__
float easuLoRcp(float a)
{
#ifdef __CUDA_ARCH__
    return __uint_as_float(
        0x7ef07ebbU -
        __float_as_uint(a));
#else
    uint32_t bits;

    std::memcpy(
        &bits,
        &a,
        sizeof(bits));

    bits =
        0x7ef07ebbU -
        bits;

    float result;

    std::memcpy(
        &result,
        &bits,
        sizeof(result));

    return result;
#endif
}

__host__ __device__
float easuLoRsq(float a)
{
#ifdef __CUDA_ARCH__
    return __uint_as_float(
        0x5f347d74U -
        (__float_as_uint(a) >> 1));
#else
    uint32_t bits;

    std::memcpy(
        &bits,
        &a,
        sizeof(bits));

    bits =
        0x5f347d74U -
        (bits >> 1);

    float result;

    std::memcpy(
        &result,
        &bits,
        sizeof(result));

    return result;
#endif
}

__host__ __device__
float easuSat(float x)
{
    return fminf(
        1.0f,
        fmaxf(
            0.0f,
            x));
}

__host__ __device__
int easuClamp(
    int x,
    int lo,
    int hi)
{
    return x < lo
        ? lo
        : (x > hi ? hi : x);
}

__host__ __device__
EasuFloat3 easuAdd(
    EasuFloat3 a,
    EasuFloat3 b)
{
    return {
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    };
}

__host__ __device__
EasuFloat3 easuMul(
    EasuFloat3 a,
    float b)
{
    return {
        a.x * b,
        a.y * b,
        a.z * b
    };
}

__host__ __device__
float easuLuma(
    EasuFloat3 c)
{
    // Exact simplified luma form used by AMD:
    // B * 0.5 + (R * 0.5 + G)
    return
        c.z * 0.5f +
        (c.x * 0.5f + c.y);
}

__host__ __device__
EasuFloat3 fetchEasu(
    const uint8_t* input,
    int width,
    int height,
    int x,
    int y)
{
    x =
        easuClamp(
            x,
            0,
            width - 1);

    y =
        easuClamp(
            y,
            0,
            height - 1);

    const size_t index =
        (static_cast<size_t>(y) *
         static_cast<size_t>(width) +
         static_cast<size_t>(x)) * 4;

    // AMD EASU operates on [0,1].
    constexpr float scale =
        1.0f / 255.0f;

    return {
        static_cast<float>(input[index + 0]) * scale,
        static_cast<float>(input[index + 1]) * scale,
        static_cast<float>(input[index + 2]) * scale
    };
}

__host__ __device__
EasuConstants makeEasuConstants(
    int inputWidth,
    int inputHeight,
    int outputWidth,
    int outputHeight)
{
    EasuConstants c{};

    const float invOutputX =
        1.0f /
        static_cast<float>(outputWidth);

    const float invOutputY =
        1.0f /
        static_cast<float>(outputHeight);

    const float invInputX =
        1.0f /
        static_cast<float>(inputWidth);

    const float invInputY =
        1.0f /
        static_cast<float>(inputHeight);

    // FsrEasuCon(), with viewport == resource == input.
    c.con0 =
    {
        static_cast<float>(inputWidth) *
            invOutputX,

        static_cast<float>(inputHeight) *
            invOutputY,

        0.5f *
            static_cast<float>(inputWidth) *
            invOutputX -
            0.5f,

        0.5f *
            static_cast<float>(inputHeight) *
            invOutputY -
            0.5f
    };

    c.con1 =
    {
        invInputX,
        invInputY,
        invInputX,
        -invInputY
    };

    c.con2 =
    {
        -invInputX,
        2.0f * invInputY,
        invInputX,
        2.0f * invInputY
    };

    c.con3 =
    {
        0.0f,
        4.0f * invInputY,
        0.0f,
        0.0f
    };

    return c;
}

__host__ __device__
EasuSamples gatherSamples(
    const uint8_t* input,
    int width,
    int height,
    int fpX,
    int fpY)
{
    EasuSamples s{};

    // Exact 12-tap EASU layout:
    //
    //       b c
    //     e f g h
    //     i j k l
    //       n o

    s.b = fetchEasu(input, width, height, fpX + 0, fpY - 1);
    s.c = fetchEasu(input, width, height, fpX + 1, fpY - 1);

    s.e = fetchEasu(input, width, height, fpX - 1, fpY + 0);
    s.f = fetchEasu(input, width, height, fpX + 0, fpY + 0);
    s.g = fetchEasu(input, width, height, fpX + 1, fpY + 0);
    s.h = fetchEasu(input, width, height, fpX + 2, fpY + 0);

    s.i = fetchEasu(input, width, height, fpX - 1, fpY + 1);
    s.j = fetchEasu(input, width, height, fpX + 0, fpY + 1);
    s.k = fetchEasu(input, width, height, fpX + 1, fpY + 1);
    s.l = fetchEasu(input, width, height, fpX + 2, fpY + 1);

    s.n = fetchEasu(input, width, height, fpX + 0, fpY + 2);
    s.o = fetchEasu(input, width, height, fpX + 1, fpY + 2);

    return s;
}

__host__ __device__
void easuSet(
    EasuFloat2& dir,
    float& len,
    EasuFloat2 pp,
    bool biS,
    bool biT,
    bool biU,
    bool biV,
    float lA,
    float lB,
    float lC,
    float lD,
    float lE)
{
    float w = 0.0f;

    if (biS)
        w =
            (1.0f - pp.x) *
            (1.0f - pp.y);

    if (biT)
        w =
            pp.x *
            (1.0f - pp.y);

    if (biU)
        w =
            (1.0f - pp.x) *
            pp.y;

    if (biV)
        w =
            pp.x *
            pp.y;

    const float dc =
        lD - lC;

    const float cb =
        lC - lB;

    float lenX =
        fmaxf(
            fabsf(dc),
            fabsf(cb));

    lenX =
        easuLoRcp(lenX);

    const float dirX =
        lD - lB;

    dir.x +=
        dirX * w;

    lenX =
        easuSat(
            fabsf(dirX) *
            lenX);

    lenX *=
        lenX;

    len +=
        lenX * w;

    const float ec =
        lE - lC;

    const float ca =
        lC - lA;

    float lenY =
        fmaxf(
            fabsf(ec),
            fabsf(ca));

    lenY =
        easuLoRcp(lenY);

    const float dirY =
        lE - lA;

    dir.y +=
        dirY * w;

    lenY =
        easuSat(
            fabsf(dirY) *
            lenY);

    lenY *=
        lenY;

    len +=
        lenY * w;
}

__host__ __device__
void easuTap(
    EasuFloat3& aC,
    float& aW,
    EasuFloat2 off,
    EasuFloat2 dir,
    EasuFloat2 len,
    float lob,
    float clp,
    EasuFloat3 color)
{
    EasuFloat2 v{};

    v.x =
        off.x * dir.x +
        off.y * dir.y;

    v.y =
        off.x * (-dir.y) +
        off.y * dir.x;

    v.x *= len.x;
    v.y *= len.y;

    float d2 =
        v.x * v.x +
        v.y * v.y;

    d2 =
        fminf(
            d2,
            clp);

    float wB =
        0.4f * d2 -
        1.0f;

    float wA =
        lob * d2 -
        1.0f;

    wB *= wB;
    wA *= wA;

    wB =
        (25.0f / 16.0f) *
        wB -
        ((25.0f / 16.0f) - 1.0f);

    const float w =
        wB * wA;

    aC =
        easuAdd(
            aC,
            easuMul(
                color,
                w));

    aW += w;
}

__host__ __device__
EasuFloat3 easuResolve(
    const uint8_t* input,
    int inputWidth,
    int inputHeight,
    int outputX,
    int outputY,
    int outputWidth,
    int outputHeight)
{
    const EasuConstants con =
        makeEasuConstants(
            inputWidth,
            inputHeight,
            outputWidth,
            outputHeight);

    // ------------------------------------------------------------
    // Position of F.
    // ------------------------------------------------------------

    EasuFloat2 pp =
    {
        static_cast<float>(outputX) *
            con.con0.x +
            con.con0.z,

        static_cast<float>(outputY) *
            con.con0.y +
            con.con0.w
    };

    const int fpX =
        static_cast<int>(
            floorf(pp.x));

    const int fpY =
        static_cast<int>(
            floorf(pp.y));

    pp.x -=
        static_cast<float>(fpX);

    pp.y -=
        static_cast<float>(fpY);

    // ------------------------------------------------------------
    // Gather the exact EASU 12 taps.
    // ------------------------------------------------------------

    const EasuSamples s =
        gatherSamples(
            input,
            inputWidth,
            inputHeight,
            fpX,
            fpY);

    const float bL = easuLuma(s.b);
    const float cL = easuLuma(s.c);
    const float eL = easuLuma(s.e);
    const float fL = easuLuma(s.f);
    const float gL = easuLuma(s.g);
    const float hL = easuLuma(s.h);
    const float iL = easuLuma(s.i);
    const float jL = easuLuma(s.j);
    const float kL = easuLuma(s.k);
    const float lL = easuLuma(s.l);
    const float nL = easuLuma(s.n);
    const float oL = easuLuma(s.o);

    // ------------------------------------------------------------
    // Exact FsrEasuSetF sequence.
    // ------------------------------------------------------------

    EasuFloat2 dir =
    {
        0.0f,
        0.0f
    };

    float len = 0.0f;

    easuSet(
        dir,
        len,
        pp,
        true,
        false,
        false,
        false,
        bL,
        eL,
        fL,
        gL,
        jL);

    easuSet(
        dir,
        len,
        pp,
        false,
        true,
        false,
        false,
        cL,
        fL,
        gL,
        hL,
        kL);

    easuSet(
        dir,
        len,
        pp,
        false,
        false,
        true,
        false,
        fL,
        iL,
        jL,
        kL,
        nL);

    easuSet(
        dir,
        len,
        pp,
        false,
        false,
        false,
        true,
        gL,
        jL,
        kL,
        lL,
        oL);

    // ------------------------------------------------------------
    // Exact direction normalization.
    // ------------------------------------------------------------

    const float dirX2 =
        dir.x * dir.x;

    const float dirY2 =
        dir.y * dir.y;

    const float dirR =
        dirX2 + dirY2;

    const bool zero =
        dirR <
        (1.0f / 32768.0f);

    float dirScale =
        easuLoRsq(dirR);

    if (zero)
        dirScale = 1.0f;

    if (zero)
        dir.x = 1.0f;

    dir.x *= dirScale;
    dir.y *= dirScale;

    // ------------------------------------------------------------
    // Exact anisotropic kernel setup.
    // ------------------------------------------------------------

    len *= 0.5f;
    len *= len;

    const float maxDir =
        fmaxf(
            fabsf(dir.x),
            fabsf(dir.y));

    const float stretch =
        (dir.x * dir.x +
         dir.y * dir.y) *
        easuLoRcp(maxDir);

    EasuFloat2 len2 =
    {
        1.0f +
            (stretch - 1.0f) * len,

        1.0f -
            0.5f * len
    };

    const float lob =
        0.5f +
        ((0.25f - 0.04f) - 0.5f) *
        len;

    const float clp =
        easuLoRcp(lob);

    // ------------------------------------------------------------
    // Exact F/G/J/K deringing neighborhood.
    // ------------------------------------------------------------

    EasuFloat3 min4 =
    {
        fminf(
            fminf(
                fminf(
                    s.f.x,
                    s.g.x),
                s.j.x),
            s.k.x),

        fminf(
            fminf(
                fminf(
                    s.f.y,
                    s.g.y),
                s.j.y),
            s.k.y),

        fminf(
            fminf(
                fminf(
                    s.f.z,
                    s.g.z),
                s.j.z),
            s.k.z)
    };

    EasuFloat3 max4 =
    {
        fmaxf(
            fmaxf(
                fmaxf(
                    s.f.x,
                    s.g.x),
                s.j.x),
            s.k.x),

        fmaxf(
            fmaxf(
                fmaxf(
                    s.f.y,
                    s.g.y),
                s.j.y),
            s.k.y),

        fmaxf(
            fmaxf(
                fmaxf(
                    s.f.z,
                    s.g.z),
                s.j.z),
            s.k.z)
    };

    // ------------------------------------------------------------
    // Exact 12-tap accumulation.
    // ------------------------------------------------------------

    EasuFloat3 aC =
    {
        0.0f,
        0.0f,
        0.0f
    };

    float aW = 0.0f;

    easuTap(
        aC,
        aW,
        {0.0f - pp.x, -1.0f - pp.y},
        dir,
        len2,
        lob,
        clp,
        s.b);

    easuTap(
        aC,
        aW,
        {1.0f - pp.x, -1.0f - pp.y},
        dir,
        len2,
        lob,
        clp,
        s.c);

    easuTap(
        aC,
        aW,
        {-1.0f - pp.x, 1.0f - pp.y},
        dir,
        len2,
        lob,
        clp,
        s.i);

    easuTap(
        aC,
        aW,
        {0.0f - pp.x, 1.0f - pp.y},
        dir,
        len2,
        lob,
        clp,
        s.j);

    easuTap(
        aC,
        aW,
        {0.0f - pp.x, 0.0f - pp.y},
        dir,
        len2,
        lob,
        clp,
        s.f);

    easuTap(
        aC,
        aW,
        {-1.0f - pp.x, 0.0f - pp.y},
        dir,
        len2,
        lob,
        clp,
        s.e);

    easuTap(
        aC,
        aW,
        {1.0f - pp.x, 1.0f - pp.y},
        dir,
        len2,
        lob,
        clp,
        s.k);

    easuTap(
        aC,
        aW,
        {2.0f - pp.x, 1.0f - pp.y},
        dir,
        len2,
        lob,
        clp,
        s.l);

    easuTap(
        aC,
        aW,
        {2.0f - pp.x, 0.0f - pp.y},
        dir,
        len2,
        lob,
        clp,
        s.h);

    easuTap(
        aC,
        aW,
        {1.0f - pp.x, 0.0f - pp.y},
        dir,
        len2,
        lob,
        clp,
        s.g);

    easuTap(
        aC,
        aW,
        {1.0f - pp.x, 2.0f - pp.y},
        dir,
        len2,
        lob,
        clp,
        s.o);

    easuTap(
        aC,
        aW,
        {0.0f - pp.x, 2.0f - pp.y},
        dir,
        len2,
        lob,
        clp,
        s.n);

    // ------------------------------------------------------------
    // Normalize and dering.
    // ------------------------------------------------------------

    const float invWeight =
        1.0f / aW;

    EasuFloat3 pix =
        easuMul(
            aC,
            invWeight);

    pix.x =
        fminf(
            max4.x,
            fmaxf(
                min4.x,
                pix.x));

    pix.y =
        fminf(
            max4.y,
            fmaxf(
                min4.y,
                pix.y));

    pix.z =
        fminf(
            max4.z,
            fmaxf(
                min4.z,
                pix.z));

    return pix;
}

__global__
void easuKernel(
    const uint8_t* input,
    int inputWidth,
    int inputHeight,
    uint8_t* output,
    int outputWidth,
    int outputHeight)
{
    const int x =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    const int y =
        blockIdx.y * blockDim.y +
        threadIdx.y;

    if (x >= outputWidth ||
        y >= outputHeight)
        return;

    const EasuFloat3 pixel =
        easuResolve(
            input,
            inputWidth,
            inputHeight,
            x,
            y,
            outputWidth,
            outputHeight);

    const size_t index =
        (static_cast<size_t>(y) *
         static_cast<size_t>(outputWidth) +
         static_cast<size_t>(x)) * 4;

    output[index + 0] =
        static_cast<uint8_t>(
            fminf(
                1.0f,
                fmaxf(
                    0.0f,
                    pixel.x)) *
            255.0f +
            0.5f);

    output[index + 1] =
        static_cast<uint8_t>(
            fminf(
                1.0f,
                fmaxf(
                    0.0f,
                    pixel.y)) *
            255.0f +
            0.5f);

    output[index + 2] =
        static_cast<uint8_t>(
            fminf(
                1.0f,
                fmaxf(
                    0.0f,
                    pixel.z)) *
            255.0f +
            0.5f);

    output[index + 3] = 255;
}

Image upscaleEasuReference(
    const Image& input,
    int outputWidth,
    int outputHeight)
{
    Image output;

    output.width = outputWidth;
    output.height = outputHeight;

    output.pixels.resize(
        static_cast<size_t>(outputWidth) *
        static_cast<size_t>(outputHeight) *
        4);

    for (int y = 0;
         y < outputHeight;
         ++y)
    {
        for (int x = 0;
             x < outputWidth;
             ++x)
        {
            const EasuFloat3 pixel =
                easuResolve(
                    input.pixels.data(),
                    input.width,
                    input.height,
                    x,
                    y,
                    outputWidth,
                    outputHeight);

            const size_t index =
                (static_cast<size_t>(y) *
                 static_cast<size_t>(outputWidth) +
                 static_cast<size_t>(x)) * 4;

            output.pixels[index + 0] =
                static_cast<uint8_t>(
                    pixel.x * 255.0f +
                    0.5f);

            output.pixels[index + 1] =
                static_cast<uint8_t>(
                    pixel.y * 255.0f +
                    0.5f);

            output.pixels[index + 2] =
                static_cast<uint8_t>(
                    pixel.z * 255.0f +
                    0.5f);

            output.pixels[index + 3] = 255;
        }
    }

    return output;
}

Image upscaleEasuCuda(
    const Image& input,
    int outputWidth,
    int outputHeight,
    float* kernelMs)
{
    Image output;

    output.width = outputWidth;
    output.height = outputHeight;

    output.pixels.resize(
        static_cast<size_t>(outputWidth) *
        static_cast<size_t>(outputHeight) *
        4);

    uint8_t* dInput = nullptr;
    uint8_t* dOutput = nullptr;

    EASU_CHECK(
        cudaMalloc(
            &dInput,
            input.pixels.size()));

    EASU_CHECK(
        cudaMalloc(
            &dOutput,
            output.pixels.size()));

    EASU_CHECK(
        cudaMemcpy(
            dInput,
            input.pixels.data(),
            input.pixels.size(),
            cudaMemcpyHostToDevice));

    dim3 threads(16, 16);

    dim3 blocks(
        (outputWidth + 15) / 16,
        (outputHeight + 15) / 16);

    for (int i = 0;
         i < 5;
         ++i)
    {
        easuKernel<<<
            blocks,
            threads
        >>>(
            dInput,
            input.width,
            input.height,
            dOutput,
            outputWidth,
            outputHeight);
    }

    EASU_CHECK(
        cudaGetLastError());

    EASU_CHECK(
        cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    EASU_CHECK(
        cudaEventCreate(&start));

    EASU_CHECK(
        cudaEventCreate(&stop));

    constexpr int iterations = 30;

    EASU_CHECK(
        cudaEventRecord(start));

    for (int i = 0;
         i < iterations;
         ++i)
    {
        easuKernel<<<
            blocks,
            threads
        >>>(
            dInput,
            input.width,
            input.height,
            dOutput,
            outputWidth,
            outputHeight);
    }

    EASU_CHECK(
        cudaEventRecord(stop));

    EASU_CHECK(
        cudaEventSynchronize(stop));

    float elapsed = 0.0f;

    EASU_CHECK(
        cudaEventElapsedTime(
            &elapsed,
            start,
            stop));

    *kernelMs =
        elapsed /
        static_cast<float>(
            iterations);

    EASU_CHECK(
        cudaMemcpy(
            output.pixels.data(),
            dOutput,
            output.pixels.size(),
            cudaMemcpyDeviceToHost));

    EASU_CHECK(
        cudaEventDestroy(start));

    EASU_CHECK(
        cudaEventDestroy(stop));

    EASU_CHECK(
        cudaFree(dInput));

    EASU_CHECK(
        cudaFree(dOutput));

    return output;
}


#if defined(__CUDACC__)

__device__
EasuTensorData easuGetTensorData(
    const uint8_t* input,
    int inputWidth,
    int inputHeight,
    int outputX,
    int outputY,
    int outputWidth,
    int outputHeight)
{
    EasuTensorData result{};

    const EasuConstants con =
        makeEasuConstants(
            inputWidth,
            inputHeight,
            outputWidth,
            outputHeight);

    EasuFloat2 pp =
    {
        static_cast<float>(outputX) *
            con.con0.x +
            con.con0.z,

        static_cast<float>(outputY) *
            con.con0.y +
            con.con0.w
    };

    const int fpX =
        static_cast<int>(
            floorf(pp.x));

    const int fpY =
        static_cast<int>(
            floorf(pp.y));

    result.fpX = fpX;
    result.fpY = fpY;

    pp.x -= static_cast<float>(fpX);
    pp.y -= static_cast<float>(fpY);

    const EasuSamples s =
        gatherSamples(
            input,
            inputWidth,
            inputHeight,
            fpX,
            fpY);

    const float bL = easuLuma(s.b);
    const float cL = easuLuma(s.c);
    const float eL = easuLuma(s.e);
    const float fL = easuLuma(s.f);
    const float gL = easuLuma(s.g);
    const float hL = easuLuma(s.h);
    const float iL = easuLuma(s.i);
    const float jL = easuLuma(s.j);
    const float kL = easuLuma(s.k);
    const float lL = easuLuma(s.l);
    const float nL = easuLuma(s.n);
    const float oL = easuLuma(s.o);

    EasuFloat2 dir =
    {
        0.0f,
        0.0f
    };

    float len = 0.0f;

    easuSet(
        dir,
        len,
        pp,
        true,
        false,
        false,
        false,
        bL,
        eL,
        fL,
        gL,
        jL);

    easuSet(
        dir,
        len,
        pp,
        false,
        true,
        false,
        false,
        cL,
        fL,
        gL,
        hL,
        kL);

    easuSet(
        dir,
        len,
        pp,
        false,
        false,
        true,
        false,
        fL,
        iL,
        jL,
        kL,
        nL);

    easuSet(
        dir,
        len,
        pp,
        false,
        false,
        false,
        true,
        gL,
        jL,
        kL,
        lL,
        oL);

    const float dirR =
        dir.x * dir.x +
        dir.y * dir.y;

    const bool zero =
        dirR <
        (1.0f / 32768.0f);

    float dirScale =
        easuLoRsq(dirR);

    if (zero)
        dirScale = 1.0f;

    if (zero)
        dir.x = 1.0f;

    dir.x *= dirScale;
    dir.y *= dirScale;

    len *= 0.5f;
    len *= len;

    const float maxDir =
        fmaxf(
            fabsf(dir.x),
            fabsf(dir.y));

    const float stretch =
        (dir.x * dir.x +
         dir.y * dir.y) *
        easuLoRcp(maxDir);

    EasuFloat2 len2 =
    {
        1.0f +
            (stretch - 1.0f) * len,

        1.0f -
            0.5f * len
    };

    const float lob =
        0.5f +
        ((0.25f - 0.04f) - 0.5f) *
        len;

    const float clp =
        easuLoRcp(lob);

    int weightIndex = 0;

    // The order MUST match the output tap order below.
    const EasuFloat2 offsets[12] =
    {
        {0.0f - pp.x, -1.0f - pp.y},
        {1.0f - pp.x, -1.0f - pp.y},
        {-1.0f - pp.x, 1.0f - pp.y},
        {0.0f - pp.x, 1.0f - pp.y},
        {0.0f - pp.x, 0.0f - pp.y},
        {-1.0f - pp.x, 0.0f - pp.y},
        {1.0f - pp.x, 1.0f - pp.y},
        {2.0f - pp.x, 1.0f - pp.y},
        {2.0f - pp.x, 0.0f - pp.y},
        {1.0f - pp.x, 0.0f - pp.y},
        {1.0f - pp.x, 2.0f - pp.y},
        {0.0f - pp.x, 2.0f - pp.y}
    };

    float rawWeights[12]{};
    float weightSum = 0.0f;

    for (int i = 0; i < 12; ++i)
    {
        EasuFloat2 v{};

        v.x =
            offsets[i].x * dir.x +
            offsets[i].y * dir.y;

        v.y =
            offsets[i].x * (-dir.y) +
            offsets[i].y * dir.x;

        v.x *= len2.x;
        v.y *= len2.y;

        float d2 =
            v.x * v.x +
            v.y * v.y;

        d2 =
            fminf(
                d2,
                clp);

        float wB =
            0.4f * d2 -
            1.0f;

        float wA =
            lob * d2 -
            1.0f;

        wB *= wB;
        wA *= wA;

        wB =
            (25.0f / 16.0f) *
            wB -
            ((25.0f / 16.0f) - 1.0f);

        rawWeights[i] =
            wB * wA;

        weightSum +=
            rawWeights[i];
    }

    const float invWeight =
        1.0f / weightSum;

    for (int i = 0; i < 12; ++i)
    {
        result.weights[i] =
            rawWeights[i] *
            invWeight;
    }

    (void)weightIndex;

    return result;
}

#endif

