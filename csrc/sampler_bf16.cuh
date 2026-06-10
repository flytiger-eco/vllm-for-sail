// BF16 top-k prefill kernel for vLLM.
//
// Ported from SGLang's topk_prefill_bf16.cuh. Algorithmic approach:
//   1) 12-bit coarse histogram (4096 bins) — single pass over input
//   2) Prefix-scan to find the threshold bin
//   3) Scatter pass — elements above threshold to output, ties to buffer
//   4) Ballot-based tie resolution
//
// Two kernel variants dispatched at launch time:
//   - Register kernel: rows <= 32K, loads into per-thread registers
//   - Streaming kernel: rows > 32K, cp.async double-buffered pipeline
//
// CUDA-only. ROCm uses a separate implementation path.

#pragma once

#ifndef USE_ROCM

#include "cuda_compat.h"

#include <cub/cub.cuh>

#include <cfloat>
#include <cstdint>
#include <cuda_bf16.h>

namespace vllm {

// ============================================================
// Constants
// ============================================================

static constexpr int kBF16BlockSize = 1024;
static constexpr int kBF16NumWarps = kBF16BlockSize / 32;

static constexpr int kBF16HistBits = 12;
static constexpr int kBF16HistBins = 1 << kBF16HistBits;
static constexpr int kBF16HistItems = kBF16HistBins / kBF16BlockSize;  // 4

static constexpr int kBF16NumStages = 2;
static constexpr int kBF16ElemPerStage = 8;  // bf16: 8 elems per 16B
static constexpr int kBF16SizePerStage =
    kBF16ElemPerStage * kBF16BlockSize;  // 8192
static constexpr int kBF16CpAsyncBytes =
    kBF16ElemPerStage * static_cast<int>(sizeof(__nv_bfloat16));  // 16

static constexpr int kBF16MaxTies = 1024;
static constexpr int kBF16Max1PassLen = 16384;
static constexpr int kBF16Max2PassLen = 2 * kBF16Max1PassLen;  // 32768
static constexpr int kBF16VecsPerThread = 4;

// ============================================================
// POD types
// ============================================================

struct alignas(16) BF16MatchBin {
  uint32_t bin;
  uint32_t aboveCount;
  uint32_t equalCount;
};

struct alignas(8) BF16TieEntry {
  uint32_t idx;
  float score;
};

// ============================================================
// Shared memory — streaming kernel (~50 KB)
// ============================================================

struct alignas(128) BF16StreamSmem {
  alignas(128) uint32_t counterGt;
  alignas(128) uint32_t counterEq;
  alignas(128) BF16MatchBin match;
  alignas(128) uint32_t warpSum[kBF16NumWarps];
  union {
    uint32_t histogram[kBF16HistBins];
    BF16TieEntry tieBuffer[kBF16MaxTies];
  };
  alignas(128)
      __nv_bfloat16 bf16Buffer[kBF16NumStages][kBF16SizePerStage];
};

// ============================================================
// Shared memory — register kernel (~82 KB)
// ============================================================

struct alignas(16) BF16Vec4F {
  float d[4];
  __device__ float& operator[](int i) { return d[i]; }
  __device__ const float& operator[](int i) const { return d[i]; }
  __device__ void fill(float v) { d[0] = d[1] = d[2] = d[3] = v; }
  __device__ void load(const float* buf, uint32_t tid) {
    uint32_t base = tid * 4;
    d[0] = buf[base];
    d[1] = buf[base + 1];
    d[2] = buf[base + 2];
    d[3] = buf[base + 3];
  }
};

struct alignas(16) BF16HistVec {
  uint32_t d[kBF16HistItems];
  __device__ uint32_t& operator[](int i) { return d[i]; }
  __device__ const uint32_t& operator[](int i) const { return d[i]; }
  __device__ void fill(uint32_t v) {
    for (int i = 0; i < kBF16HistItems; ++i) d[i] = v;
  }
};

struct BF16RegisterSmem {
  alignas(128) uint32_t counterGt;
  alignas(128) uint32_t counterEq;
  BF16MatchBin match;
  uint32_t warpSum[kBF16NumWarps];
  union {
    uint32_t histogram[kBF16HistBins];
    BF16HistVec histogramVec[kBF16BlockSize];
    BF16TieEntry tieBuffer[kBF16MaxTies];
  };
  alignas(16) float scoreBuffer[kBF16Max1PassLen];
};

// ============================================================
// Device helpers
// ============================================================

template <int kBits>
static inline __device__ uint32_t extractCoarseBinBF16(float x) {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000u) ? static_cast<uint16_t>(~bits)
                                  : static_cast<uint16_t>(bits | 0x8000u);
  return key >> (16 - kBits);
}

static inline __device__ uint32_t bf16DivCeil(uint32_t a, uint32_t b) {
  return (a + b - 1u) / b;
}

static inline __device__ uint32_t bf16WarpInclusiveSum(uint32_t laneId,
                                                       uint32_t val) {
#pragma unroll
  for (uint32_t offset = 1; offset < 32; offset *= 2) {
    uint32_t n = __shfl_up_sync(0xFFFFFFFF, val, offset);
    if (laneId >= offset) val += n;
  }
  return val;
}

static inline __device__ uint32_t bf16WarpReduceSum(uint32_t val) {
#pragma unroll
  for (uint32_t offset = 16; offset >= 1; offset >>= 1) {
    val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

// ============================================================
// bf16FindThreshold — prefix-scan histogram, locate threshold bin
// ============================================================

template <typename SmemT>
static __device__ void bf16FindThreshold(uint32_t length, int topK,
                                         SmemT* smem) {
  uint32_t tx = threadIdx.x;
  uint32_t laneId = tx % 32;
  uint32_t warpId = tx / 32;

  uint32_t orig[kBF16HistItems];
  uint32_t localSum = 0;
#pragma unroll
  for (int i = 0; i < kBF16HistItems; ++i) {
    orig[i] = smem->histogram[tx * kBF16HistItems + i];
    localSum += orig[i];
  }

  uint32_t warpInc = bf16WarpInclusiveSum(laneId, localSum);
  uint32_t warpExc = warpInc - localSum;
  if (laneId == 31) smem->warpSum[warpId] = warpInc;
  __syncthreads();

  uint32_t tmp = smem->warpSum[laneId];
  uint32_t prefixSum = bf16WarpReduceSum(laneId < warpId ? tmp : 0u);
  prefixSum += warpExc;

#pragma unroll
  for (int i = 0; i < kBF16HistItems; ++i) {
    prefixSum += orig[i];
    uint32_t above = length - prefixSum;
    if (above < static_cast<uint32_t>(topK) &&
        above + orig[i] >= static_cast<uint32_t>(topK)) {
      smem->match.bin = tx * kBF16HistItems + i;
      smem->match.aboveCount = above;
      smem->match.equalCount = orig[i];
    }
  }
  __syncthreads();
}

// ============================================================
// bf16TieHandle — resolve ties in the threshold bin
// ============================================================

template <typename SmemT>
static __device__ void bf16TieHandle(int32_t* __restrict__ sTopK, int topK,
                                     SmemT* smem) {
  uint32_t tx = threadIdx.x;
  uint32_t laneId = tx % 32;
  uint32_t warpId = tx / 32;

  uint32_t numAbove = smem->match.aboveCount;
  uint32_t numEqual = smem->counterEq;
  uint32_t numTies = min(numEqual, static_cast<uint32_t>(kBF16MaxTies));
  uint32_t topkRemain =
      (static_cast<uint32_t>(topK) > numAbove)
          ? (static_cast<uint32_t>(topK) - numAbove)
          : 0u;
  bool needTiebreak = (numAbove + numEqual > static_cast<uint32_t>(topK));

  auto isGreater = [](const BF16TieEntry& a, const BF16TieEntry& b) {
    return (a.score > b.score) || (a.score == b.score && a.idx < b.idx);
  };

  if (topkRemain == 0u) {
    // Already filled by strictly-greater bucket.
  } else if (!needTiebreak) {
    for (uint32_t i = tx; i < numTies; i += kBF16BlockSize) {
      uint32_t pos = numAbove + i;
      if (pos < static_cast<uint32_t>(topK)) {
        sTopK[pos] = static_cast<int32_t>(smem->tieBuffer[i].idx);
      }
    }
  } else if (numTies <= 32u) {
    if (laneId < numTies && warpId < numTies) {
      uint32_t mask =
          (numTies == 32u) ? 0xFFFFFFFFu : ((1u << numTies) - 1u);
      BF16TieEntry tie = smem->tieBuffer[laneId];
      BF16TieEntry target = smem->tieBuffer[warpId];
      bool pred = isGreater(tie, target);
      uint32_t rank =
          static_cast<uint32_t>(__popc(__ballot_sync(mask, pred)));
      if (laneId == 0 && rank < topkRemain) {
        sTopK[numAbove + rank] = static_cast<int32_t>(target.idx);
      }
    }
  } else if (numTies <= 64u) {
    BF16TieEntry invalid;
    invalid.idx = 0xFFFFFFFFu;
    invalid.score = -FLT_MAX;
    BF16TieEntry tie0 = smem->tieBuffer[laneId];
    BF16TieEntry tie1 = (laneId + 32u) < numTies
                            ? smem->tieBuffer[laneId + 32u]
                            : invalid;

    auto rankTarget = [&](const BF16TieEntry& target) {
      bool p0 = isGreater(tie0, target);
      bool p1 = isGreater(tie1, target);
      uint32_t r0 =
          static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFF, p0)));
      uint32_t r1 =
          static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFF, p1)));
      return r0 + r1;
    };

    if (warpId < numTies) {
      BF16TieEntry target = smem->tieBuffer[warpId];
      uint32_t rank = rankTarget(target);
      if (laneId == 0 && rank < topkRemain) {
        sTopK[numAbove + rank] = static_cast<int32_t>(target.idx);
      }
    }
    if (warpId + 32u < numTies) {
      BF16TieEntry target = smem->tieBuffer[warpId + 32u];
      uint32_t rank = rankTarget(target);
      if (laneId == 0 && rank < topkRemain) {
        sTopK[numAbove + rank] = static_cast<int32_t>(target.idx);
      }
    }
  } else {
    for (uint32_t i = warpId; i < numTies; i += kBF16NumWarps) {
      BF16TieEntry target = smem->tieBuffer[i];
      uint32_t localRank = 0;
      for (uint32_t j = laneId; j < numTies; j += 32) {
        BF16TieEntry t = smem->tieBuffer[j];
        if (isGreater(t, target)) ++localRank;
      }
      uint32_t rank = bf16WarpReduceSum(localRank);
      if (laneId == 0 && rank < topkRemain) {
        sTopK[numAbove + rank] = static_cast<int32_t>(target.idx);
      }
    }
  }
  __syncthreads();
}

// ============================================================
// bf16StreamPass — cp.async double-buffered histogram/scatter
// ============================================================

template <bool kIsScatter>
static __device__ void bf16StreamPass(
    const __nv_bfloat16* __restrict__ scores, uint32_t length,
    uint32_t thrBin, int topK, int32_t* __restrict__ sTopK,
    BF16StreamSmem* smem) {
  uint32_t tx = threadIdx.x;
  uint32_t numIters = bf16DivCeil(length, kBF16SizePerStage);

  auto issueStage = [&](uint32_t s) {
    if (s >= numIters) return;
    uint32_t buf = s % kBF16NumStages;
    uint32_t base = s * kBF16SizePerStage;
    uint32_t local = tx * kBF16ElemPerStage;
    uint32_t globalBase = base + local;

    if (globalBase + kBF16ElemPerStage <= length) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
      __pipeline_memcpy_async(&smem->bf16Buffer[buf][local],
                              scores + globalBase, kBF16CpAsyncBytes);
#else
      const uint4 data =
          *reinterpret_cast<const uint4*>(scores + globalBase);
      *reinterpret_cast<uint4*>(&smem->bf16Buffer[buf][local]) = data;
#endif
    } else if (globalBase < length) {
#pragma unroll
      for (int e = 0; e < kBF16ElemPerStage; ++e) {
        uint32_t g = globalBase + e;
        smem->bf16Buffer[buf][local + e] =
            (g < length) ? scores[g] : __float2bfloat16(-FLT_MAX);
      }
    } else {
#pragma unroll
      for (int e = 0; e < kBF16ElemPerStage; ++e) {
        smem->bf16Buffer[buf][local + e] = __float2bfloat16(-FLT_MAX);
      }
    }
  };

  // Prologue: kick off pipeline stages.
#pragma unroll
  for (uint32_t s = 0; s < static_cast<uint32_t>(kBF16NumStages); ++s) {
    issueStage(s);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __pipeline_commit();
#endif
  }

  // Main loop.
  for (uint32_t iter = 0; iter < numIters; ++iter) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __pipeline_wait_prior(kBF16NumStages - 1);
#endif
    __syncthreads();

    uint32_t buf = iter % kBF16NumStages;
    uint32_t base = iter * kBF16SizePerStage;

#pragma unroll
    for (int e = 0; e < kBF16ElemPerStage; ++e) {
      uint32_t local = tx * kBF16ElemPerStage + e;
      uint32_t globalIdx = base + local;
      if (globalIdx >= length) break;

      float val = __bfloat162float(smem->bf16Buffer[buf][local]);
      uint32_t bin = extractCoarseBinBF16<kBF16HistBits>(val);

      if constexpr (kIsScatter) {
        if (bin > thrBin) {
          uint32_t pos = atomicAdd(&smem->counterGt, 1u);
          if (pos < static_cast<uint32_t>(topK)) {
            sTopK[pos] = static_cast<int32_t>(globalIdx);
          }
        } else if (bin == thrBin) {
          uint32_t pos = atomicAdd(&smem->counterEq, 1u);
          if (pos < static_cast<uint32_t>(kBF16MaxTies)) {
            smem->tieBuffer[pos].idx = globalIdx;
            smem->tieBuffer[pos].score = val;
          }
        }
      } else {
        atomicAdd(&smem->histogram[bin], 1u);
      }
    }
    __syncthreads();

    issueStage(iter + kBF16NumStages);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __pipeline_commit();
#endif
  }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  __pipeline_wait_prior(0);
#endif
  __syncthreads();
}

// ============================================================
// bf16ScatterWarpSteal — work-stealing scatter, no block barriers
// ============================================================

static __device__ void bf16ScatterWarpSteal(
    const __nv_bfloat16* __restrict__ scores, uint32_t length,
    uint32_t thrBin, int topK, int32_t* __restrict__ sTopK,
    BF16StreamSmem* smem) {
  uint32_t tx = threadIdx.x;
  uint32_t laneId = tx & 31u;
  uint32_t warpId = tx >> 5u;

  constexpr uint32_t kWarpTileElems =
      32u * static_cast<uint32_t>(kBF16ElemPerStage);  // 256
  uint32_t numWarpTiles = bf16DivCeil(length, kWarpTileElems);

  // Pre-assign first kBF16NumWarps tiles statically.
  if (tx == 0) {
    smem->warpSum[0] = kBF16NumWarps;
  }
  __syncthreads();

  uint32_t myTile = warpId;

  while (myTile < numWarpTiles) {
    uint32_t base = myTile * kWarpTileElems;
    uint32_t globalBase = base + laneId * kBF16ElemPerStage;

    if (globalBase + kBF16ElemPerStage <= length) {
      // Vectorized 16-byte load.
      uint4 data =
          *reinterpret_cast<const uint4*>(scores + globalBase);
      const __nv_bfloat16* vals =
          reinterpret_cast<const __nv_bfloat16*>(&data);

#pragma unroll
      for (int e = 0; e < kBF16ElemPerStage; ++e) {
        float fval = __bfloat162float(vals[e]);
        uint32_t bin = extractCoarseBinBF16<kBF16HistBits>(fval);
        uint32_t globalIdx = globalBase + e;

        if (bin > thrBin) {
          uint32_t pos = atomicAdd(&smem->counterGt, 1u);
          if (pos < static_cast<uint32_t>(topK)) {
            sTopK[pos] = static_cast<int32_t>(globalIdx);
          }
        } else if (bin == thrBin) {
          uint32_t pos = atomicAdd(&smem->counterEq, 1u);
          if (pos < static_cast<uint32_t>(kBF16MaxTies)) {
            smem->tieBuffer[pos].idx = globalIdx;
            smem->tieBuffer[pos].score = fval;
          }
        }
      }
    } else if (globalBase < length) {
      for (int e = 0; e < kBF16ElemPerStage; ++e) {
        uint32_t globalIdx = globalBase + e;
        if (globalIdx >= length) break;

        float fval = __bfloat162float(scores[globalIdx]);
        uint32_t bin = extractCoarseBinBF16<kBF16HistBits>(fval);

        if (bin > thrBin) {
          uint32_t pos = atomicAdd(&smem->counterGt, 1u);
          if (pos < static_cast<uint32_t>(topK)) {
            sTopK[pos] = static_cast<int32_t>(globalIdx);
          }
        } else if (bin == thrBin) {
          uint32_t pos = atomicAdd(&smem->counterEq, 1u);
          if (pos < static_cast<uint32_t>(kBF16MaxTies)) {
            smem->tieBuffer[pos].idx = globalIdx;
            smem->tieBuffer[pos].score = fval;
          }
        }
      }
    }

    // Warp leader steals next tile.
    if (laneId == 0) {
      myTile = atomicAdd(&smem->warpSum[0], 1u);
    }
    myTile = __shfl_sync(0xFFFFFFFF, myTile, 0);
  }
}

// ============================================================
// bf16StreamingTopK — orchestrate streaming flow
// ============================================================

static __device__ void bf16StreamingTopK(
    const __nv_bfloat16* __restrict__ scores, uint32_t length,
    int topK, int32_t* __restrict__ sTopK, BF16StreamSmem* smem) {
  uint32_t tx = threadIdx.x;

  // Init shared state.
#pragma unroll
  for (int i = 0; i < kBF16HistItems; ++i) {
    smem->histogram[tx * kBF16HistItems + i] = 0u;
  }
  if (tx == 0) {
    smem->counterGt = 0u;
    smem->counterEq = 0u;
  }
  for (uint32_t i = tx; i < static_cast<uint32_t>(topK);
       i += kBF16BlockSize) {
    sTopK[i] = -1;
  }
  __syncthreads();

  // Phase A: histogram via streaming pipeline.
  bf16StreamPass<false>(scores, length, 0u, topK, nullptr, smem);

  // Phase B: find threshold bin.
  bf16FindThreshold(length, topK, smem);

  // Reset counters for scatter phase.
  if (tx == 0) {
    smem->counterGt = 0u;
    smem->counterEq = 0u;
  }
  __syncthreads();

  // Phase C: scatter (work-stealing).
  uint32_t thrBin = smem->match.bin;
  bf16ScatterWarpSteal(scores, length, thrBin, topK, sTopK, smem);

  // Barrier before tie resolution.
  __syncthreads();
}

// ============================================================
// bf16RegisterTopK — register fast path for short/medium rows
// ============================================================

template <bool kIs2Pass>
static __device__ void bf16RegisterTopK(
    const __nv_bfloat16* __restrict__ scores,
    int32_t* __restrict__ indices, uint32_t length, int topK,
    void* rawSmem) {
  auto* smem = static_cast<BF16RegisterSmem*>(rawSmem);
  uint32_t tx = threadIdx.x;
  uint32_t laneId = tx % 32;
  uint32_t warpId = tx / 32;

  // Init histogram.
  {
    BF16HistVec hv;
    hv.fill(0);
    smem->histogramVec[tx] = hv;
    if (tx == 0) {
      smem->counterGt = smem->counterEq = 0;
    }
    __syncthreads();
  }

  // Load scores into registers (bf16 -> float).
  BF16Vec4F local[kBF16VecsPerThread];
#pragma unroll
  for (int v = 0; v < kBF16VecsPerThread; ++v) {
    uint32_t base = (tx + v * kBF16BlockSize) * 4;
    if (base >= length) break;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      uint32_t idx = base + e;
      local[v][e] =
          (idx < length) ? static_cast<float>(scores[idx]) : 0.0f;
    }
  }

  // 2-pass: load second chunk into smem scoreBuffer.
  if constexpr (kIs2Pass) {
    uint32_t extraLen = length - kBF16Max1PassLen;
    for (uint32_t i = tx; i < extraLen; i += kBF16BlockSize) {
      smem->scoreBuffer[i] =
          static_cast<float>(scores[kBF16Max1PassLen + i]);
    }
    __syncthreads();
  }

  // Accumulate histogram.
#pragma unroll
  for (int v = 0; v < kBF16VecsPerThread; ++v) {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      if constexpr (!kIs2Pass) {
        uint32_t idx = (tx + v * kBF16BlockSize) * 4 + e;
        if (idx >= length) goto label_acc_done;
      }
      atomicAdd(
          &smem->histogram[extractCoarseBinBF16<kBF16HistBits>(local[v][e])],
          1);
    }
  }
  if constexpr (kIs2Pass) {
    for (uint32_t i = tx;
         i + static_cast<uint32_t>(kBF16Max1PassLen) < length;
         i += kBF16BlockSize) {
      float val = smem->scoreBuffer[i];
      atomicAdd(&smem->histogram[extractCoarseBinBF16<kBF16HistBits>(val)],
                1);
    }
  }
[[maybe_unused]] label_acc_done:
  __syncthreads();

  // Find threshold.
  bf16FindThreshold(length, topK, smem);

  auto [thrBin, numAbove, numEqual] = smem->match;
  bool needTiebreak =
      (numAbove + numEqual > static_cast<uint32_t>(topK));
  auto* topkIndices = indices;
  auto* tieBuf = smem->tieBuffer;

  // Scatter first chunk from registers.
#pragma unroll
  for (int v = 0; v < kBF16VecsPerThread; ++v) {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      uint32_t idx = (tx + v * kBF16BlockSize) * 4 + e;
      if constexpr (!kIs2Pass) {
        if (idx >= length) goto label_scatter_done;
      }
      uint32_t bin = extractCoarseBinBF16<kBF16HistBits>(local[v][e]);
      if (bin > thrBin) {
        topkIndices[atomicAdd(&smem->counterGt, 1)] = idx;
      } else if (bin == thrBin) {
        uint32_t pos = atomicAdd(&smem->counterEq, 1);
        if (needTiebreak) {
          if (pos < static_cast<uint32_t>(kBF16MaxTies)) {
            tieBuf[pos].idx = idx;
            tieBuf[pos].score = local[v][e];
          }
        } else {
          uint32_t which = pos + numAbove;
          if (which < static_cast<uint32_t>(topK)) {
            topkIndices[which] = idx;
          }
        }
      }
    }
    // Reload registers from scoreBuffer for the second chunk.
    if constexpr (kIs2Pass) {
      local[v].load(smem->scoreBuffer, tx + v * kBF16BlockSize);
    }
  }

  // 2-pass: scatter second chunk.
  if constexpr (kIs2Pass) {
#pragma unroll
    for (int v = 0; v < kBF16VecsPerThread; ++v) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        uint32_t idx =
            (tx + v * kBF16BlockSize) * 4 + e + kBF16Max1PassLen;
        if (idx >= length) goto label_scatter_done;
        uint32_t bin =
            extractCoarseBinBF16<kBF16HistBits>(local[v][e]);
        if (bin > thrBin) {
          topkIndices[atomicAdd(&smem->counterGt, 1)] = idx;
        } else if (bin == thrBin) {
          uint32_t pos = atomicAdd(&smem->counterEq, 1);
          if (needTiebreak) {
            if (pos < static_cast<uint32_t>(kBF16MaxTies)) {
              tieBuf[pos].idx = idx;
              tieBuf[pos].score = local[v][e];
            }
          } else {
            uint32_t which = pos + numAbove;
            if (which < static_cast<uint32_t>(topK)) {
              topkIndices[which] = idx;
            }
          }
        }
      }
    }
  }

[[maybe_unused]] label_scatter_done:
  if (!needTiebreak) return;
  __syncthreads();

  // Tie-break.
  uint32_t numTies =
      min(numEqual, static_cast<uint32_t>(kBF16MaxTies));
  uint32_t topkRemain = static_cast<uint32_t>(topK) - numAbove;

  auto isGreater = [](const BF16TieEntry& a, const BF16TieEntry& b) {
    return (a.score > b.score) || (a.score == b.score && a.idx < b.idx);
  };

  if (numTies <= 32u) {
    if (laneId >= numTies || warpId >= numTies) return;
    uint32_t mask =
        (numTies == 32u) ? 0xFFFFFFFFu : ((1u << numTies) - 1u);
    BF16TieEntry tie = tieBuf[laneId];
    BF16TieEntry target = tieBuf[warpId];
    bool pred = isGreater(tie, target);
    uint32_t rank =
        static_cast<uint32_t>(__popc(__ballot_sync(mask, pred)));
    if (laneId == 0 && rank < topkRemain) {
      topkIndices[numAbove + rank] = target.idx;
    }
  } else if (numTies <= 64u) {
    BF16TieEntry invalid;
    invalid.idx = 0xFFFFFFFFu;
    invalid.score = -FLT_MAX;
    BF16TieEntry tie0 = tieBuf[laneId];
    BF16TieEntry tie1 = (laneId + 32u) < numTies
                            ? tieBuf[laneId + 32u]
                            : invalid;
    if (warpId < numTies) {
      BF16TieEntry target = tieBuf[warpId];
      bool p0 = isGreater(tie0, target);
      bool p1 = isGreater(tie1, target);
      uint32_t r0 =
          static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFF, p0)));
      uint32_t r1 =
          static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFF, p1)));
      uint32_t rank = r0 + r1;
      if (laneId == 0 && rank < topkRemain) {
        topkIndices[numAbove + rank] = target.idx;
      }
    }
    if (warpId + 32u < numTies) {
      BF16TieEntry target = tieBuf[warpId + 32u];
      bool p0 = isGreater(tie0, target);
      bool p1 = isGreater(tie1, target);
      uint32_t r0 =
          static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFF, p0)));
      uint32_t r1 =
          static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFF, p1)));
      uint32_t rank = r0 + r1;
      if (laneId == 0 && rank < topkRemain) {
        topkIndices[numAbove + rank] = target.idx;
      }
    }
  } else {
    for (uint32_t i = warpId; i < numTies;
         i += static_cast<uint32_t>(kBF16NumWarps)) {
      BF16TieEntry target = tieBuf[i];
      uint32_t localRank = 0;
      for (uint32_t j = laneId; j < numTies; j += 32) {
        BF16TieEntry t = tieBuf[j];
        if (isGreater(t, target)) ++localRank;
      }
      uint32_t rank = bf16WarpReduceSum(localRank);
      if (laneId == 0 && rank < topkRemain) {
        topkIndices[numAbove + rank] = target.idx;
      }
    }
  }
}

// ============================================================
// Kernel: register path (rows <= 32K)
// ============================================================

static __global__ __launch_bounds__(kBF16BlockSize, 1)
    void topKPerRowPrefillBF16Register(
        const __nv_bfloat16* __restrict__ logits,
        const int* __restrict__ rowStarts,
        const int* __restrict__ rowEnds,
        int* __restrict__ outIndices,
        int stride0, int topK) {
  extern __shared__ char smemRaw[];
  auto* sTopK =
      reinterpret_cast<int32_t*>(smemRaw + sizeof(BF16RegisterSmem));

  int rowIdx = blockIdx.x;
  int rowStart = rowStarts[rowIdx];
  int rowEnd = rowEnds[rowIdx];
  uint32_t length =
      rowEnd > rowStart ? static_cast<uint32_t>(rowEnd - rowStart) : 0u;

  const __nv_bfloat16* scores =
      logits + static_cast<int64_t>(rowIdx) * stride0 + rowStart;
  int* out = outIndices + static_cast<int64_t>(rowIdx) * topK;

  if (length <= static_cast<uint32_t>(topK)) {
    uint32_t tx = threadIdx.x;
    if (tx < length) {
      out[tx] = tx;
    } else if (tx < static_cast<uint32_t>(topK)) {
      out[tx] = -1;
    }
    return;
  }

  // Init sentinel indices.
  for (uint32_t i = threadIdx.x; i < static_cast<uint32_t>(topK);
       i += kBF16BlockSize) {
    sTopK[i] = -1;
  }
  __syncthreads();

  if (length <= static_cast<uint32_t>(kBF16Max1PassLen)) {
    bf16RegisterTopK<false>(scores, sTopK, length, topK, smemRaw);
  } else {
    bf16RegisterTopK<true>(scores, sTopK, length, topK, smemRaw);
  }
  __syncthreads();

  // Copy to output.
  for (uint32_t i = threadIdx.x; i < static_cast<uint32_t>(topK);
       i += kBF16BlockSize) {
    out[i] = sTopK[i];
  }
}

// ============================================================
// Kernel: streaming path (rows > 32K)
// ============================================================

static __global__ __launch_bounds__(kBF16BlockSize, 1)
    void topKPerRowPrefillBF16Stream(
        const __nv_bfloat16* __restrict__ logits,
        const int* __restrict__ rowStarts,
        const int* __restrict__ rowEnds,
        int* __restrict__ outIndices,
        int stride0, int topK) {
  extern __shared__ char smemRaw[];
  auto* smem = reinterpret_cast<BF16StreamSmem*>(smemRaw);
  auto* sTopK =
      reinterpret_cast<int32_t*>(smemRaw + sizeof(BF16StreamSmem));

  int rowIdx = blockIdx.x;
  int rowStart = rowStarts[rowIdx];
  int rowEnd = rowEnds[rowIdx];
  uint32_t length =
      rowEnd > rowStart ? static_cast<uint32_t>(rowEnd - rowStart) : 0u;

  const __nv_bfloat16* scores =
      logits + static_cast<int64_t>(rowIdx) * stride0 + rowStart;
  int* out = outIndices + static_cast<int64_t>(rowIdx) * topK;

  if (length <= static_cast<uint32_t>(topK)) {
    uint32_t tx = threadIdx.x;
    if (tx < length) {
      out[tx] = tx;
    } else if (tx < static_cast<uint32_t>(topK)) {
      out[tx] = -1;
    }
    return;
  }

  bf16StreamingTopK(scores, length, topK, sTopK, smem);
  bf16TieHandle(sTopK, topK, smem);

  // Copy to output.
  for (uint32_t i = threadIdx.x; i < static_cast<uint32_t>(topK);
       i += kBF16BlockSize) {
    out[i] = sTopK[i];
  }
}

}  // namespace vllm

#endif  // USE_ROCM
