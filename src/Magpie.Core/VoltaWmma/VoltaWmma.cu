#include "VoltaWmma.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cmath>

using namespace nvcuda;

namespace Magpie::VoltaWmma {

namespace {

__global__ void WmmaTestKernel(
	const half* a,
	const half* b,
	float* c
)
{
	if (threadIdx.x >= 32) {
		return;
	}

	wmma::fragment<
		wmma::matrix_a,
		16, 16, 16,
		half,
		wmma::row_major
	> aFrag;

	wmma::fragment<
		wmma::matrix_b,
		16, 16, 16,
		half,
		wmma::row_major
	> bFrag;

	wmma::fragment<
		wmma::accumulator,
		16, 16, 16,
		float
	> cFrag;

	wmma::fill_fragment(cFrag, 0.0f);

	wmma::load_matrix_sync(aFrag, a, 16);
	wmma::load_matrix_sync(bFrag, b, 16);

	wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);

	wmma::store_matrix_sync(
		c,
		cFrag,
		16,
		wmma::mem_row_major
	);
}

}

bool InitializeAndSelfTest() noexcept
{
	int deviceCount = 0;

	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess ||
		deviceCount <= 0) {
		return false;
	}

	if (cudaSetDevice(0) != cudaSuccess) {
		return false;
	}

	cudaDeviceProp props{};

	if (cudaGetDeviceProperties(&props, 0) != cudaSuccess) {
		return false;
	}

	if (props.major != 7 || props.minor != 0) {
		return false;
	}

	constexpr int N = 16 * 16;

	half* a = nullptr;
	half* b = nullptr;
	float* c = nullptr;

	if (cudaMallocManaged(&a, N * sizeof(half)) != cudaSuccess) {
		return false;
	}

	if (cudaMallocManaged(&b, N * sizeof(half)) != cudaSuccess) {
		cudaFree(a);
		return false;
	}

	if (cudaMallocManaged(&c, N * sizeof(float)) != cudaSuccess) {
		cudaFree(a);
		cudaFree(b);
		return false;
	}

	for (int i = 0; i < N; ++i) {
		a[i] = __float2half(1.0f);
		b[i] = __float2half(1.0f);
		c[i] = 0.0f;
	}

	WmmaTestKernel<<<1, 32>>>(a, b, c);

	const cudaError_t launchError = cudaGetLastError();
	const cudaError_t syncError = cudaDeviceSynchronize();

	bool passed =
		launchError == cudaSuccess &&
		syncError == cudaSuccess;

	if (passed) {
		for (int i = 0; i < N; ++i) {
			if (std::fabs(c[i] - 16.0f) > 0.001f) {
				passed = false;
				break;
			}
		}
	}

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);

	return passed;
}

}