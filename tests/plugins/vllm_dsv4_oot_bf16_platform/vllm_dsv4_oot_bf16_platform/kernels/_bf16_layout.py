# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse-MLA cache slot geometry and the FP8-vs-BF16 layout switch.

On platforms without native FP8 compute the sparse-MLA KV cache and the
indexer cache are stored in BF16 instead of the packed ``fp8_ds_mla``
layout. The attention math is BF16 either way; only the cache read/write
and the indexer quant differ, gated on ``kv_cache_uses_fp8()``.
"""

from vllm.platforms import current_platform

# cache_dtype string for the DSv4 BF16 sparse-MLA fallback layout.
DSV4_BF16_DS_MLA = "bf16_ds_mla"

# Sparse-MLA (SWA / compressed) KV-cache slot geometry.
# The value vector is 512 elements: 448 NoPE + 64 RoPE. Two on-disk layouts:
#   * FP8 (``fp8_ds_mla``): 448 fp8 bytes + 64*2 bf16 bytes = 576B token data,
#     plus an 8B UE8M0 scale block => 584B/token, padded to 576B alignment.
#   * BF16 fallback: all 512 elements stored as bf16 => 1024B/token, no scale
#     block. The cache tensor stays ``uint8`` so the kernels' byte-offset
#     arithmetic is identical; only the per-token size and the presence of
#     the fp8/scale regions differ (gated by a ``USE_FP8`` constexpr).
MLA_VALUE_DIM = 512
MLA_NOPE_DIM = 448
MLA_ROPE_DIM = 64

# Per-token bytes of value data, excluding any scale block.
FP8_TOKEN_DATA_SIZE = MLA_NOPE_DIM + MLA_ROPE_DIM * 2  # 576
BF16_TOKEN_BYTES = MLA_VALUE_DIM * 2  # 1024


def kv_cache_uses_fp8() -> bool:
    return current_platform.supports_fp8()
