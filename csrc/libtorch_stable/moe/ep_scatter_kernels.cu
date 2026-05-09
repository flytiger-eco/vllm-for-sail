/* Optimal by SAIL */
#include <cuda_runtime.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include <cuda_pipeline_primitives.h>
#include <optional>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "libtorch_stable/dispatch_utils.h"
#include "libtorch_stable/torch_utils.h"

#ifndef USE_ROCM
typedef __half torch_Half;
typedef __nv_bfloat16 torch_BFloat16;
#else
typedef __hip_bfloat16 torch_BFloat16;
#endif

template <bool with_scale, typename T, bool less_scales>
__global__ void ep_scatter_2_kernel_special_optimized(
    int M, int topk_num, int K, int Ks, int num_experts,
    const T* __restrict__ hidden_states, const float* __restrict__ scales,
    const int* __restrict__ topk_ids, int* __restrict__ expert_start_loc,
    T* __restrict__ output_tensor, int* __restrict__ output_index,
    float* __restrict__ output_tensor_scale) {
  extern __shared__ __align__(16) char smem_ptr[];
  T* to_copy = reinterpret_cast<T*>(smem_ptr);
  float* to_copy_s = reinterpret_cast<float*>(to_copy + K);
  int* expert_ids = reinterpret_cast<int*>(to_copy_s + (with_scale ? Ks : 0));

  const int m_idx = blockIdx.x;
  if (m_idx >= M) return;

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;  // tid / 32
  const int lane_id = tid & 31;  // tid % 32

  const T* src_hs_row = hidden_states + m_idx * K;
  constexpr int ELEMENTS_PER_THREAD = 16 / sizeof(T);
  for (int k = tid * ELEMENTS_PER_THREAD; k < K;
       k += blockDim.x * ELEMENTS_PER_THREAD) {
    __pipeline_memcpy_async(&to_copy[k], &src_hs_row[k], 16);
  }

  if constexpr (with_scale) {
    const float* src_s_row = scales + m_idx * Ks;
    if constexpr (less_scales) {
      if (tid < Ks) to_copy_s[tid] = src_s_row[tid];
    } else {
      for (int k = tid << 2; k < Ks; k += blockDim.x << 2) {
        __pipeline_memcpy_async(&to_copy_s[k], &src_s_row[k], 16);
      }
    }
  }

  if (tid < topk_num) {
    expert_ids[tid] = topk_ids[m_idx * topk_num + tid];
  }

  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncthreads();

  if (warp_id < topk_num) {
    const int expert_loc = expert_ids[warp_id];
    if (expert_loc >= 0) {
      int dest_token_index;
      if (lane_id == 0) {
        dest_token_index = atomicAdd(&expert_start_loc[expert_loc], 1);
        output_index[m_idx * topk_num + warp_id] = dest_token_index;
      }
      dest_token_index = __shfl_sync(0xffffffff, dest_token_index, 0);

      size_t hs_base_offset_bytes = (size_t)dest_token_index * K * sizeof(T);
      T* dst_hs_ptr = reinterpret_cast<T*>(
          reinterpret_cast<char*>(output_tensor) + hs_base_offset_bytes);
      for (int k = lane_id * ELEMENTS_PER_THREAD; k < K;
           k += 32 * ELEMENTS_PER_THREAD) {
        *reinterpret_cast<uint4*>(&dst_hs_ptr[k]) =
            *reinterpret_cast<uint4*>(&to_copy[k]);
      }

      if constexpr (with_scale) {
        size_t s_base_offset_bytes =
            (size_t)dest_token_index * Ks * sizeof(float);
        float* dst_s_ptr = reinterpret_cast<float*>(
            reinterpret_cast<char*>(output_tensor_scale) + s_base_offset_bytes);
        if constexpr (less_scales) {
          if (lane_id < Ks) dst_s_ptr[lane_id] = to_copy_s[lane_id];
        } else {
          for (int k = lane_id << 2; k < Ks; k += 128) {
            *reinterpret_cast<uint4*>(&dst_s_ptr[k]) =
                *reinterpret_cast<uint4*>(&to_copy_s[k]);
          }
        }
      }
    }
  }
}

template <typename T, bool with_scale>
void launch_ep_scatter_2_kernel(torch::stable::Tensor& hidden_states,
                                torch::stable::Tensor& scales,
                                torch::stable::Tensor& topk_ids,
                                torch::stable::Tensor& expert_start_loc,
                                torch::stable::Tensor& output_tensor,
                                torch::stable::Tensor& output_index,
                                torch::stable::Tensor& output_tensor_scale) {
  const int M = hidden_states.size(0);
  const int K = hidden_states.size(1);
  const int topk_num = topk_ids.size(1);
  const int num_experts = expert_start_loc.size(0);
  const int Ks = with_scale ? scales.size(1) : 0;

  auto* prop = get_device_prop();
  int maxWarpsPerBlock = prop->maxThreadsPerBlock * 2 / prop->warpSize;
  if (maxWarpsPerBlock < topk_num || topk_num % 2 != 0) {
    STD_TORCH_CHECK(false, "Unsupported topk_num");
  }

  int grid_size = M;
  int block_size = topk_num * prop->warpSize;
  size_t smem_size =
      (K * hidden_states.element_size() +
       (with_scale ? Ks * sizeof(float) : 0) + topk_num * sizeof(int));
  auto stream = get_current_cuda_stream(hidden_states.get_device_index());
  auto kernel =
      (Ks * sizeof(float) < 16)
          ? ep_scatter_2_kernel_special_optimized<with_scale, T, true>
          : ep_scatter_2_kernel_special_optimized<with_scale, T, false>;

  if (smem_size >= 48 * 1024) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_size);
  }

  kernel<<<grid_size, block_size, smem_size, stream>>>(
      M, topk_num, K, Ks, expert_start_loc.size(0),
      reinterpret_cast<const T*>(hidden_states.const_data_ptr()),
      with_scale ? scales.const_data_ptr<float>() : nullptr,
      topk_ids.const_data_ptr<int>(),
      expert_start_loc.mutable_data_ptr<int>(),
      reinterpret_cast<T*>(output_tensor.mutable_data_ptr()),
      output_index.mutable_data_ptr<int>(),
      with_scale ? output_tensor_scale.mutable_data_ptr<float>() : nullptr);
}

void ep_scatter_2_cuda(torch::stable::Tensor hidden_states,
                       std::optional<torch::stable::Tensor> scales_opt,
                       torch::stable::Tensor topk_ids,
                       torch::stable::Tensor expert_start_loc,
                       torch::stable::Tensor output_tensor,
                       torch::stable::Tensor output_index,
                       std::optional<torch::stable::Tensor> output_tensor_scale_opt,
                       bool with_scale) {
  torch::stable::Tensor scales =
      with_scale ? scales_opt.value() : torch::stable::Tensor();
  torch::stable::Tensor output_tensor_scale =
      with_scale ? output_tensor_scale_opt.value() : torch::stable::Tensor();

  torch::stable::accelerator::DeviceGuard guard(
      hidden_states.get_device_index());

  THO_DISPATCH_SWITCH(
      hidden_states.scalar_type(), "ep_scatter_2_cuda",
      THO_DISPATCH_CASE(torch::headeronly::ScalarType::Float8_e4m3fn, [&] {  // fp8
        if (with_scale)
          launch_ep_scatter_2_kernel<__nv_fp8_e4m3, true>(
              hidden_states, scales, topk_ids, expert_start_loc, output_tensor,
              output_index, output_tensor_scale);
        else
          launch_ep_scatter_2_kernel<__nv_fp8_e4m3, false>(
              hidden_states, scales, topk_ids, expert_start_loc, output_tensor,
              output_index, output_tensor_scale);
      }) THO_DISPATCH_CASE(torch::headeronly::ScalarType::Float8_e5m2, [&] {  // fp8
        if (with_scale)
          launch_ep_scatter_2_kernel<__nv_fp8_e5m2, true>(
              hidden_states, scales, topk_ids, expert_start_loc, output_tensor,
              output_index, output_tensor_scale);
        else
          launch_ep_scatter_2_kernel<__nv_fp8_e5m2, false>(
              hidden_states, scales, topk_ids, expert_start_loc, output_tensor,
              output_index, output_tensor_scale);
      }) THO_DISPATCH_CASE(torch::headeronly::ScalarType::Char, [&] {  // int8
        if (with_scale)
          launch_ep_scatter_2_kernel<int8_t, true>(
              hidden_states, scales, topk_ids, expert_start_loc, output_tensor,
              output_index, output_tensor_scale);
        else
          launch_ep_scatter_2_kernel<int8_t, false>(
              hidden_states, scales, topk_ids, expert_start_loc, output_tensor,
              output_index, output_tensor_scale);
      }) THO_DISPATCH_CASE(torch::headeronly::ScalarType::Half, [&] {  // fp16
        if (with_scale)
          launch_ep_scatter_2_kernel<torch_Half, true>(
              hidden_states, scales, topk_ids, expert_start_loc, output_tensor,
              output_index, output_tensor_scale);
        else
          launch_ep_scatter_2_kernel<torch_Half, false>(
              hidden_states, scales, topk_ids, expert_start_loc, output_tensor,
              output_index, output_tensor_scale);
      }) THO_DISPATCH_CASE(torch::headeronly::ScalarType::BFloat16, [&] {  // bf16
        if (with_scale)
          launch_ep_scatter_2_kernel<torch_BFloat16, true>(
              hidden_states, scales, topk_ids, expert_start_loc, output_tensor,
              output_index, output_tensor_scale);
        else
          launch_ep_scatter_2_kernel<torch_BFloat16, false>(
              hidden_states, scales, topk_ids, expert_start_loc, output_tensor,
              output_index, output_tensor_scale);
      }) THO_DISPATCH_CASE(torch::headeronly::ScalarType::Float, [&] {  // fp32
        if (with_scale)
          launch_ep_scatter_2_kernel<float, true>(
              hidden_states, scales, topk_ids, expert_start_loc, output_tensor,
              output_index, output_tensor_scale);
        else
          launch_ep_scatter_2_kernel<float, false>(
              hidden_states, scales, topk_ids, expert_start_loc, output_tensor,
              output_index, output_tensor_scale);
      }));
}
