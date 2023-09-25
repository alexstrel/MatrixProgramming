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

#ifdef _NVHPC_CUDA
#include <nv/target>
#endif

#include <iostream>
#include <concepts>

#include <experimental/mdspan>

namespace stdex = std::experimental;

constexpr int pad = 4;

constexpr int WARP_M = 4;
constexpr int WARP_N = 4;

constexpr std::array<int, 3> INST{8,8,4};// {16, 8, 4} , {16, 8, 8}, and {16, 8, 16}

constexpr int MMA_M = INST[0] * WARP_M;
constexpr int MMA_N = INST[1] * WARP_N;
constexpr int MMA_K = INST[2];

template<typename T>
class is_mdspan : public std::false_type {};

template<typename T, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
class is_mdspan<stdex::mdspan<T, Extents, LayoutPolicy, AccessorPolicy>> : public std::true_type {};

// MDspan concept:
template <typename T>
concept MDViewTp = is_mdspan<std::remove_cvref_t<T>>::value;

// Extended floaitng point type:
template <typename T>
concept ArithmeticTp = std::is_floating_point_v<T>;

constexpr int warp_size = 32;

enum class OperandType {
	Aoperand = 0,
	Boperand = 1
};

class WarpRegisterMapping {
  
  public:
    static constexpr int group_size = 4;
    static constexpr int ngroups    = warp_size / group_size; 
 
    int laneId;
    int groupId;
    int threadId_in_group;

    WarpRegisterMapping(int threadId) :
      laneId(threadId & 31),
      groupId(laneId >> 2),         // = laneId / 4 (group_size)
      threadId_in_group(laneId & 3) // = laneId % 4 (group_size) 
    {
    }
};

/*
 * For MmaOperand  ( A and B matrices) data needs to be loaded from shared memory to the registers of the threads in a warp
 * according to how the matrix elements are distributed accross the threads in the (D)MMA instruciton.
 * The specific distribution is documented in
 *   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-884-f64
 */
template<ArithmeticTp Float, int warp_tile_, int mma_ldim, int mma_k, OperandType type = OperandType::Aoperand>
class MmaOperandAB {
  public:
    using reg_type = Float;

    static constexpr int warp_tile   = warp_tile_;
  
    std::array<reg_type, warp_tile> reg;

    inline MmaOperandAB( const MDViewTp auto smem, const int tile_k, const int tile_ldim, const WarpRegisterMapping &wrm ){
      constexpr  int ngroups   = std::remove_cvref_t<decltype(wrm)>::ngroups;

      const int k = tile_k * mma_k + wrm.threadId_in_group;//{0,1,2,3}, MMA_K = 4
#pragma unroll
      for (int i = 0; i < warp_tile / 2; i++) {
#pragma unroll
        for (int b = 0; b < 2; b++) {
          int l = tile_ldim * mma_ldim + (i * ngroups + wrm.groupId) * 2 + b;
          if constexpr (type == OperandType::Aoperand) {//A operand
            reg[i * 2 + b] = smem(l, k);  //[k * lda + l];
          } else {//B operand
            reg[i * 2 + b] = smem(k, l);  //[k * ldb + l];
          }
        }
      }
    }
};

/*
 * For MmaOperandC data needs to be stored from registers of the threads in a warp to (global) memory
 * according to how the matrix elements are distributed accross the threads in the (D)MMA instruciton.
 * The specific distribution is documented in
 *   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-884-f64
 */
template<ArithmeticTp Float, int warp_m_, int warp_n_>
struct MmaOperandC {

  public:
    using reg_type = Float;

    static constexpr int warp_m   = warp_m_;
    static constexpr int warp_n   = warp_n_;

    std::array<reg_type, warp_m * warp_n * 2> reg;

    MmaOperandC() {
#pragma unroll
      for (int i = 0; i < warp_m * warp_n * 2; i++) { reg[i] = 0; }
    }

    void store(MDViewTp auto c_view, int m_offset, int n_offset, const WarpRegisterMapping &wrm)
    {
      constexpr  int ngroups    = std::remove_cvref_t<decltype(wrm)>::ngroups;
      constexpr  int group_size = std::remove_cvref_t<decltype(wrm)>::group_size;    

#pragma unroll
    for (int c = 0; c < 2; c++) {
#pragma unroll
      for (int n_i = 0; n_i < warp_n / 2; n_i++) {
#pragma unroll
        for (int m_i = 0; m_i < warp_m / 2; m_i++) {
#pragma unroll
          for (int n_b = 0; n_b < 2; n_b++) {
            double tmp[2];
            int n_iter = n_i * 2 + n_b;
#pragma unroll
            for (int m_b = 0; m_b < 2; m_b++) {
              int m_iter = m_i * 2 + m_b;
              int c_iter = (n_iter * warp_m + m_iter) * 2;

              tmp[m_b] = reg[c_iter + c];
            }
            int gmem_m = m_offset + (m_i * ngroups + wrm.groupId) * 2;
            int gmem_n = n_offset + ((n_i * group_size + wrm.threadId_in_group)*2 + c) * 2 + n_b;            
            asm("st.cs.global.v2.f64 [%0+0], {%1, %2};" ::"l"(&c_view(gmem_m,  gmem_n)), "d"(tmp[0]), "d"(tmp[1]));
          }
        }
      }
    }
  }
};

/*
 * Actually calling the (D)MMA instruction. 
 * (WARNING: without target branching should be inlined for CUDA kernel only) 
 */
template <ArithmeticTp res_type, ArithmeticTp op_type, int inst_m, int inst_n, int inst_k> 
class mma_instruction{ };


template <> class mma_instruction<double, double, 8, 8, 4>{ 
  public:

    template <typename mma_op_c, typename mma_op_a, typename mma_op_b>
    void operator()(mma_op_c &op_c, const mma_op_a &op_a, const mma_op_b &op_b) {

      constexpr int warp_m = mma_op_c::warp_m;
      constexpr int warp_n = mma_op_c::warp_n;

      if target (nv::target::is_device) {
#pragma unroll
        for (int m_iter = 0; m_iter < warp_m; m_iter++) {
#pragma unroll
          for (int n_iter = 0; n_iter < warp_n; n_iter++) {
            int c_iter = (n_iter * warp_m + m_iter) * 2;
            asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};"
                     : "+d"(op_c.reg[c_iter + 0]), "+d"(op_c.reg[c_iter + 1])
                     : "d"(op_a.reg[m_iter]), "d"(op_b.reg[n_iter]));
          }
        }
      } else {
        std::cerr << "Operation is not implemented (not supported)\n" << std::endl;
      }
    }
};

using m8n8k4_instruction_f64 = mma_instruction<double, double, 8, 8, 4>;

