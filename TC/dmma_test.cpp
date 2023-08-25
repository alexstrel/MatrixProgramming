// A.S. : this is an adopted C++2X version of the original CUDA implementation :
// https://github.com/hummingtree/dmma-ptx-gemm
/**
 * Copyright (c) 2021, NVIDIA Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>
#include <vector>
#include <array>

#include <span>
#include <type_traits>
#include <random>

#include <cuda_helper.h>

#include <mma_m8n8k4_fp64_sm80.h>

#define MATRIX_M 4096
#define MATRIX_N 2048 // 192 * 64
#define MATRIX_K 1024

#include <cuda_pipeline.h>

using namespace nvcuda::experimental;

using F = double;

using indx_type = int; 

std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()s
std::uniform_real_distribution<F> dis(0.f, 1.f);

// Static objects views: 
using Strided2DView  = stdex::mdspan<F, stdex::extents<indx_type, stdex::dynamic_extent, stdex::dynamic_extent>, stdex::layout_stride, stdex::default_accessor<F>>;
using Strided2DCView = stdex::mdspan<const F, stdex::extents<indx_type, stdex::dynamic_extent, stdex::dynamic_extent>, stdex::layout_stride, stdex::default_accessor<const F>>;

using Dyn2DMap  = stdex::layout_stride::mapping<stdex::extents<indx_type, stdex::dynamic_extent, stdex::dynamic_extent>>;
using Extents2D = stdex::extents<indx_type, stdex::dynamic_extent, stdex::dynamic_extent>;

/*
 * @brief Load data from global memory (col major) to smem (col major)
 * @brief Load data from global memory (col major) to smem (row major)
 */

template <int block_y, int block_z>
inline void g2s(MDViewTp auto smem_view, MDViewTp auto gmem_view, int row_offset, int col_offset, pipeline &pipe)
{
  constexpr auto row_dim = smem_view.extent(0);
  constexpr auto col_dim = smem_view.extent(1);
#pragma unroll
  for (int col = threadIdx.z; col < col_dim; col += block_z) {
#pragma unroll
    for (int row = threadIdx.y; row < row_dim; row += block_y) {
      memcpy_async(smem_view(row, col), gmem_view((row_offset + row), (col + col_offset)), pipe);
    }
  }
}

/**
  This kernel performs GEMM.
  Data needed for each threadblock is first loaded into shared memory, before the thread-block
  level GEMM is performed.
*/

template <int block_y, int block_z, int bM, int bN, int bK>
void mma_cuda_kernel(const int thread_id, MDViewTp auto view_a, MDViewTp auto view_b, MDViewTp auto view_c, auto smem_ptr, const int m_offset, const int n_offset)
{
  using SmemMatMK = stdex::mdspan<F, stdex::extents<int, bM, bK>, stdex::layout_stride, stdex::default_accessor<F>>;
  using SmemMatKN = stdex::mdspan<F, stdex::extents<int, bK, bN>, stdex::layout_stride, stdex::default_accessor<F>>;
  //
  using SmemMapMK = stdex::layout_stride::mapping<stdex::extents<int, bM, bK>>;
  using SmemMapKN = stdex::layout_stride::mapping<stdex::extents<int, bK, bN>>;
 

  constexpr int tile_row_dim = bM / MMA_M; // number of tiles in the col dimension
  constexpr int tile_col_dim = bN / MMA_N; // number of tiles in the row dimension
  constexpr int tile_acc_dim = bK / MMA_K; // number of tiles in the acc dimension

  constexpr int total_warp = block_y * block_z / 32;

  constexpr int total_tile = tile_row_dim * tile_col_dim;
  constexpr int warp_cycle = total_tile / total_warp;

  static_assert(total_tile % total_warp == 0, "Total number of tiles should be divisible by the number of warps.");

  MmaOperandC op_c[warp_cycle]; // initilized to zero

  const int warp_id   = thread_id / 32;//>>>
  WarpRegisterMapping wrm(thread_id);

  // For shared memory layout we use col majob for operand A and row major for operand B with pads.
  
  constexpr int ld_smem_a = bM + pad;
  constexpr int ld_smem_b = bN + pad;

  MDViewTp auto view_smem_a_compute = SmemMatMK(smem_ptr, SmemMapMK( stdex::extents<int, bM, bK>{}, std::array<int, 2>{1, ld_smem_a}) );

  MDViewTp auto view_smem_b_compute = SmemMatKN(smem_ptr+(ld_smem_a*bK),  SmemMapKN( stdex::extents<int, bK, bN>{}, std::array<int, 2>{ld_smem_b, 1}) );

  MDViewTp auto view_smem_a_memory  = SmemMatMK(smem_ptr+((ld_smem_a+ld_smem_b)*bK), SmemMapMK( stdex::extents<int, bM, bK>{}, std::array<int, 2>{1, ld_smem_a}) );

  MDViewTp auto view_smem_b_memory  = SmemMatKN(smem_ptr+(2*ld_smem_a+ld_smem_b)*bK,  SmemMapKN( stdex::extents<int, bK, bN>{}, std::array<int, 2>{ld_smem_b, 1}) );

  pipeline pipe;

  g2s<block_y, block_z>(view_smem_a_compute, view_a, m_offset, 0, pipe);
  g2s<block_y, block_z>(view_smem_b_compute, view_b, 0, n_offset, pipe);

  pipe.commit();

  const auto KK = view_a.extent(1);//or  view_b.extent(0)

  for (int k_offset = 0; k_offset < KK; k_offset += bK) {
    if (k_offset + bK < KK) {
      g2s<block_y, block_z>(view_smem_a_memory, view_a, m_offset, k_offset + bK, pipe);
      g2s<block_y, block_z>(view_smem_b_memory, view_b, k_offset + bK, n_offset, pipe);
    }
    // We use the one set of smem for compute, and let data being loaded into the other set of smem
    // while doing computation.
    pipe.commit();
    pipe.wait_prior<1>();
    __syncthreads();

    // MMA!
#pragma unroll
    for (int c = 0; c < warp_cycle; c++) {
      // The logical warp assigned to each part of the matrix.
      const int logical_warp_index = warp_id * warp_cycle + c;
      const int tile_m = logical_warp_index / tile_col_dim;
      const int tile_n = logical_warp_index - tile_m * tile_col_dim;

      for (int tile_k = 0; tile_k < tile_acc_dim; tile_k++) {

        MmaOperandA op_a;
        op_a.load(view_smem_a_compute, tile_k, tile_m, wrm);

        MmaOperandB op_b;
        op_b.load(view_smem_b_compute, tile_k, tile_n, wrm);

        mma(op_c[c], op_a, op_b);

      } // tile_k
    }   // c

    if (k_offset + bK < KK) {
      // If we have the next iteration to do, switch the smem buffers:
      // compute -> memory, memory -> compute
      __syncthreads();

      std::swap(view_smem_a_memory, view_smem_a_compute);
      std::swap(view_smem_b_memory, view_smem_b_compute);
    }
  } // k_offset

  // Store result to ptr_c.
#pragma unroll
  for (int c = 0; c < warp_cycle; c++) {
    // The logical warp assigned to each part of the matrix.
    const int logical_warp_index = warp_id * warp_cycle + c;
    const int warp_row = logical_warp_index / tile_col_dim;
    const int warp_col = logical_warp_index - warp_row * tile_col_dim;

    op_c[c].store(view_c, m_offset + warp_row * MMA_M, n_offset + warp_col * MMA_N, wrm);
  }
}

template <typename Float, int block_y, int block_z, int bM, int bN, typename lambda_tp>
__global__ void __launch_bounds__(block_y *block_z, 1) launch_mma_kernel(const lambda_tp mma_kernel){

  const int idx = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
 
  // Declare shared memory.
  extern __shared__ int smem[];

  // Offsets of this thread-block in global memory
  int m_offset = blockIdx.y * bM;
  int n_offset = blockIdx.z * bN;
 
  mma_kernel(idx, m_offset, n_offset, reinterpret_cast<Float*>(smem));
}

template <typename Float, int block_y, int block_z, typename stream_tp, int ...bSize>
void dispatch_mma_kernel(auto&& mma_kernel, int smem_size, stream_tp stream){
  static_assert(sizeof...(bSize) == 3);//only 3D is supported

  constexpr int D{sizeof...(bSize)};
  constexpr std::array<int, D> bm{bSize...};

  // Matrix block dims:
  constexpr int bM = bm[0];
  constexpr int bN = bm[1];

  dim3 blockDim(1, block_y, block_z);
  dim3 gridDim(1, MATRIX_M / bM, MATRIX_N / bN);
  
  auto kernel = launch_mma_kernel<Float, block_y, block_z, bM, bN, std::remove_cvref_t<decltype(mma_kernel)>>;

  cudaFuncSetAttribute((const void *)kernel, cudaFuncAttributePreferredSharedMemoryCarveout,(int)cudaSharedmemCarveoutMaxShared);

  cudaFuncSetAttribute((const void *)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  check_error();

  launch_mma_kernel<Float, block_y, block_z, bM, bN><<<gridDim, blockDim, smem_size, stream>>>(mma_kernel);
  
  check_error();
  
  //wait();
}

int main(int argc, char *argv[])
{
  std::vector<F> a_fp64(MATRIX_M * MATRIX_K);
  std::vector<F> b_fp64(MATRIX_K * MATRIX_N);
  std::vector<F> c_mma(MATRIX_M * MATRIX_N);
  //
  std::vector<F> c_cublas(MATRIX_M * MATRIX_N);

  //curandGenerator_t gen;
  cublasHandle_t cublas_handle;

  cudaEvent_t start_mma;
  cudaEvent_t stop_mma;

  cudaEvent_t start_cublas;
  cudaEvent_t stop_cublas;

  cudaErrCheck(cudaEventCreate(&start_mma));
  cudaErrCheck(cudaEventCreate(&stop_mma));

  cudaErrCheck(cudaEventCreate(&start_cublas));
  cudaErrCheck(cudaEventCreate(&stop_cublas));

  cublasErrCheck(cublasCreate(&cublas_handle));

  // Use tensor cores
  cublasErrCheck(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));
  //
  for (auto &i : a_fp64) i = dis(gen);

  for (auto &i : b_fp64) i = dis(gen);

  for (auto &i : c_mma)  i = dis(gen);

  using ElementA = F;
  using ElementB = F;
  using ElementC = F;

  ElementC alpha = 1.0;
  ElementC beta  = 0.0;

  printf("\nMatrix M = %d, Matrix N = %d, Matrix K = %d.\n\n", MATRIX_M, MATRIX_N, MATRIX_K);

  // First: using MMA
  constexpr int bM = 64;
  constexpr int bN = 64;
  constexpr int bK = 32;

  constexpr int block_y = 8;
  constexpr int block_z = 16;

  int shared_memory_size = ((bM + pad) * bK * sizeof(ElementA) + bK * (bN + pad) * sizeof(ElementB)) * 2;

  auto mma_kernel = [=, a_fp64 = std::span(a_fp64),
                        b_fp64 = std::span(b_fp64),
                        c_mma  = std::span(c_mma)] (const auto i, const int m_offset, const int n_offset, F* smem) {
                          //in:
                          auto view_a = Strided2DView{a_fp64.data(), Dyn2DMap{Extents2D{MATRIX_M, MATRIX_K}, std::array<int, 2>{1, MATRIX_M}}}; 
                          auto view_b = Strided2DView{b_fp64.data(), Dyn2DMap{Extents2D{MATRIX_K, MATRIX_N}, std::array<int, 2>{1, MATRIX_K}}};
                          //out:
                          auto view_c = Strided2DView{c_mma.data(),  Dyn2DMap{Extents2D{MATRIX_M, MATRIX_N}, std::array<int, 2>{1, MATRIX_M}}};

                          mma_cuda_kernel<block_y, block_z, bM, bN, bK>( i, view_a, view_b, view_c, smem, m_offset, n_offset );
                       };

  printf("Running with MMA ...\n");

  printf("Shared memory size = %05d\n", shared_memory_size);

  int nstreams = 1;
  int niter    = 1;

  auto streams = get_streams(nstreams);

  auto stream  = streams[0];//with UVM, we use only one compute stream 

  cudaErrCheck(cudaEventRecord(start_mma));

  for (int i = 0; i < niter; i++) {

    dispatch_mma_kernel<F, block_y, block_z, decltype(stream), bM, bN, bK>(mma_kernel, shared_memory_size, stream);
  }

  cudaErrCheck(cudaEventRecord(stop_mma));
  cudaErrCheck(cudaPeekAtLastError());

  // Now using cuBLAS
  printf("Running with cuBLAS...\n");
  cudaErrCheck(cudaEventRecord(start_cublas));

  for (int i = 0; i < niter; i++) {
    cublasErrCheck(cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, MATRIX_M, MATRIX_N, MATRIX_K, &alpha, a_fp64.data(),
                                CUDA_R_64F, MATRIX_M, b_fp64.data(), CUDA_R_64F, MATRIX_K, &beta, c_cublas.data(), CUDA_R_64F,
                                MATRIX_M, CUDA_R_64F, CUBLAS_GEMM_DFALT_TENSOR_OP));
  }

  cudaErrCheck(cudaEventRecord(stop_cublas));

  // Error checking
  printf("\nChecking results...\n");

  cudaErrCheck(cudaDeviceSynchronize());

  int errors = 0;

  for (int i = 0; i < MATRIX_M * MATRIX_N; i++) {
    if (fabs(c_mma[i] - c_cublas[i]) > 1e-8) {
      errors++;
      if (errors < 10) printf("%05d: %8.4f vs. %8.4f.\n", i, c_mma[i], c_cublas[i]);
    }
  }

  if (errors > 0) {
    printf("WMMA does not agree with cuBLAS! %d errors!\n", errors);
  } else {
    printf("Results verified: cublas and WMMA agree.\n\n");

    float mma_time;
    float cublas_time;

    cudaErrCheck(cudaEventSynchronize(stop_mma));
    cudaErrCheck(cudaEventSynchronize(stop_cublas));
    cudaErrCheck(cudaEventElapsedTime(&mma_time, start_mma, stop_mma));
    cudaErrCheck(cudaEventElapsedTime(&cublas_time, start_cublas, stop_cublas));

    printf("mma took %8.4f ms = %.1f %% cublas.\n", mma_time, 100.0 * cublas_time / mma_time);
    printf("cublas took %8.4f ms, %.1f TFLOPS.\n", cublas_time,
           2.0 * MATRIX_M * MATRIX_N * MATRIX_K * niter / (cublas_time * 1e-3) / 1e+12);
  }

  cudaErrCheck(cudaEventDestroy(start_mma));
  cudaErrCheck(cudaEventDestroy(stop_mma));

  cudaErrCheck(cudaEventDestroy(start_cublas));
  cudaErrCheck(cudaEventDestroy(stop_cublas));

  cudaErrCheck(cudaDeviceReset());

  return 0;
}
