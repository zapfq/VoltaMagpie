#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace nvcuda;

static void CheckCuda(cudaError_t result, const char* operation)
{
    if (result != cudaSuccess)
    {
        std::fprintf(
            stderr,
            "CUDA error in %s: %s\n",
            operation,
            cudaGetErrorString(result));

        std::exit(EXIT_FAILURE);
    }
}

__global__ void WmmaTestKernel(
    const half* a,
    const half* b,
    float* c)
{
    if (threadIdx.x >= 32)
        return;

    wmma::fragment<
        wmma::matrix_a,
        16, 16, 16,
        half,
        wmma::row_major> aFrag;

    wmma::fragment<
        wmma::matrix_b,
        16, 16, 16,
        half,
        wmma::row_major> bFrag;

    wmma::fragment<
        wmma::accumulator,
        16, 16, 16,
        float> cFrag;

    wmma::fill_fragment(cFrag, 0.0f);

    wmma::load_matrix_sync(aFrag, a, 16);
    wmma::load_matrix_sync(bFrag, b, 16);

    wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);

    wmma::store_matrix_sync(
        c,
        cFrag,
        16,
        wmma::mem_row_major);
}

int main()
{
    int deviceCount = 0;

    CheckCuda(
        cudaGetDeviceCount(&deviceCount),
        "cudaGetDeviceCount");

    if (deviceCount == 0)
    {
        std::fprintf(stderr, "No CUDA device found.\n");
        return EXIT_FAILURE;
    }

    cudaDeviceProp props{};

    CheckCuda(
        cudaGetDeviceProperties(&props, 0),
        "cudaGetDeviceProperties");

    std::printf("GPU: %s\n", props.name);
    std::printf(
        "Compute capability: %d.%d\n",
        props.major,
        props.minor);

    if (props.major != 7 || props.minor != 0)
    {
        std::fprintf(
            stderr,
            "Warning: this test is intended for SM 7.0.\n");
    }

    constexpr int N = 16 * 16;

    half* a = nullptr;
    half* b = nullptr;
    float* c = nullptr;

    CheckCuda(
        cudaMallocManaged(&a, N * sizeof(half)),
        "cudaMallocManaged(a)");

    CheckCuda(
        cudaMallocManaged(&b, N * sizeof(half)),
        "cudaMallocManaged(b)");

    CheckCuda(
        cudaMallocManaged(&c, N * sizeof(float)),
        "cudaMallocManaged(c)");

    for (int i = 0; i < N; ++i)
    {
        a[i] = __float2half(1.0f);
        b[i] = __float2half(1.0f);
        c[i] = 0.0f;
    }

    WmmaTestKernel<<<1, 32>>>(a, b, c);

    CheckCuda(
        cudaGetLastError(),
        "WmmaTestKernel launch");

    CheckCuda(
        cudaDeviceSynchronize(),
        "cudaDeviceSynchronize");

    bool passed = true;

    for (int i = 0; i < N; ++i)
    {
        if (std::fabs(c[i] - 16.0f) > 0.001f)
        {
            std::fprintf(
                stderr,
                "Mismatch at element %d: %f\n",
                i,
                c[i]);

            passed = false;
            break;
        }
    }

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    if (!passed)
    {
        std::fprintf(stderr, "WMMA test FAILED.\n");
        return EXIT_FAILURE;
    }

    std::printf("WMMA 16x16x16 test: PASS\n");
    std::printf("Tensor Core path is executing successfully.\n");

    return EXIT_SUCCESS;
}