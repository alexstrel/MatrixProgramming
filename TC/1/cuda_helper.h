
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

#pragma once

void check_error(){
  	
  auto error = cudaGetLastError();
  if(error != cudaSuccess) fprintf(stderr, "CUDA Error: %s \n", cudaGetErrorString(error));
  //
  return;
}

decltype(auto) get_streams(const int n){

  std::vector<cudaStream_t> streams;
  streams.reserve(n);

  for (int i = 0; i < n; i++) {  
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    streams.push_back(stream);
  }

  return streams;
}

// Define some error checking macros.
#define cudaErrCheck(stat)                                                                                             \
  {                                                                                                                    \
    cudaErrCheck_((stat), __FILE__, __LINE__);                                                                         \
  }
void cudaErrCheck_(cudaError_t stat, const char *file, int line)
{
  if (stat != cudaSuccess) { fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line); }
}

#define cublasErrCheck(stat)                                                                                           \
  {                                                                                                                    \
    cublasErrCheck_((stat), __FILE__, __LINE__);                                                                       \
  }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line)
{
  if (stat != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line); }
}

#define curandErrCheck(stat)                                                                                           \
  {                                                                                                                    \
    curandErrCheck_((stat), __FILE__, __LINE__);                                                                       \
  }
void curandErrCheck_(curandStatus_t stat, const char *file, int line)
{
  if (stat != CURAND_STATUS_SUCCESS) { fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line); }
}
