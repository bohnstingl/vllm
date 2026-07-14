# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 sparse-MLA cache kernels for the DeepSeek V4 OOT fallback.

These are the in-tree FP8 kernels' dual-mode (``USE_FP8`` constexpr)
counterparts, invoked with ``use_fp8=False`` so the FP8 branch is
dead-code-eliminated at Triton compile time. Keeping them here rather than
in-tree keeps the in-tree DeepSeek V4 path FP8-only.

This re-exports the surface used by ``attention_overrides``; the indexer op
imports the remaining kernels from their submodules directly.
"""

from ._bf16_layout import BF16_TOKEN_BYTES, DSV4_BF16_DS_MLA
from .cache_utils import dequantize_and_gather_k_cache
from .fused_compress_quant_cache import compress_norm_rope_store_triton
from .fused_indexer_q import fused_indexer_q_rope_quant
from .triton_qnorm_rope_kv_insert import triton_qnorm_rope_kv_insert
from .triton_sparse_decode import triton_sparse_decode

__all__ = [
    "BF16_TOKEN_BYTES",
    "DSV4_BF16_DS_MLA",
    "compress_norm_rope_store_triton",
    "dequantize_and_gather_k_cache",
    "fused_indexer_q_rope_quant",
    "triton_qnorm_rope_kv_insert",
    "triton_sparse_decode",
]
