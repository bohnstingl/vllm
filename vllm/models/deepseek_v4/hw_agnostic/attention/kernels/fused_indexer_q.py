# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.models.deepseek_v4.hw_agnostic.attention._fp8_support import (
    kv_cache_uses_fp8,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _get_cos_sin(
    cos_sin_cache_ptr,
    cos_sin_cache_stride,
    pos,
    HALF_ROT_DIM: tl.constexpr,
):
    block = tl.arange(0, HALF_ROT_DIM)
    cos = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block)
    cos = cos.to(tl.float32)
    sin = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block + HALF_ROT_DIM)
    sin = sin.to(tl.float32)
    return cos, sin


@triton.jit
def _fused_indexer_q_rope_quant_kernel(
    pos_ptr,
    # Index Q RoPE
    index_q_ptr,
    index_q_stride0,
    index_q_stride1,
    index_q_cos_sin_ptr,
    index_q_cos_sin_stride,
    INDEX_Q_HALF_ROT_DIM: tl.constexpr,
    # Index Q Quantize
    index_q_fp8_ptr,
    index_q_fp8_stride0,
    index_q_fp8_stride1,
    INDEX_Q_HEAD_DIM: tl.constexpr,
    # Index weights
    index_weights_ptr,
    index_weights_stride,
    index_weights_softmax_scale,
    index_weights_head_scale,
    index_weights_out_ptr,
    index_weights_out_stride,
    USE_FP8: tl.constexpr,
):
    # Layout matches the reference (DeepseekV4ScalingRotaryEmbedding +
    # per_token_group_quant_fp8): GPT-J interleaved RoPE applied to the LAST
    # rope_dim dims of each head; the leading [0, NOPE_DIM) passes through
    # unchanged.
    INDEX_Q_ROT_DIM: tl.constexpr = 2 * INDEX_Q_HALF_ROT_DIM
    INDEX_Q_NOPE_DIM: tl.constexpr = INDEX_Q_HEAD_DIM - INDEX_Q_ROT_DIM
    tl.static_assert(INDEX_Q_NOPE_DIM >= 0)

    tok_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    pos = tl.load(pos_ptr + tok_idx)
    cos, sin = _get_cos_sin(
        index_q_cos_sin_ptr,
        index_q_cos_sin_stride,
        pos,
        INDEX_Q_HALF_ROT_DIM,
    )
    half_offset = tl.arange(0, INDEX_Q_HALF_ROT_DIM)
    base_ptr = index_q_ptr + tok_idx * index_q_stride0 + head_idx * index_q_stride1

    # Interleaved (GPT-J) RoPE on dims [NOPE_DIM, HEAD_DIM):
    #   even = q[NOPE_DIM + 2*i],  odd = q[NOPE_DIM + 2*i + 1]
    rot_base = base_ptr + INDEX_Q_NOPE_DIM
    x_even = tl.load(rot_base + half_offset * 2).to(tl.float32)
    x_odd = tl.load(rot_base + half_offset * 2 + 1).to(tl.float32)
    r_even = x_even * cos - x_odd * sin
    r_odd = x_odd * cos + x_even * sin

    # Match reference numerics: fp32 → bf16 → fp32 before the ue8m0 absmax.
    # Same pattern as the K-side compressor kernel (fused_compress_quant_cache.py).
    r_even = r_even.to(tl.bfloat16).to(tl.float32)
    r_odd = r_odd.to(tl.bfloat16).to(tl.float32)

    nope_offset = tl.arange(0, INDEX_Q_NOPE_DIM) if INDEX_Q_NOPE_DIM > 0 else None
    if INDEX_Q_NOPE_DIM > 0:
        x_nope = tl.load(base_ptr + nope_offset).to(tl.float32)

    fp8_base_ptr = (
        index_q_fp8_ptr + tok_idx * index_q_fp8_stride0 + head_idx * index_q_fp8_stride1
    )
    fp8_rot_base = fp8_base_ptr + INDEX_Q_NOPE_DIM

    if USE_FP8:
        amax = tl.maximum(tl.max(tl.abs(r_even)), tl.max(tl.abs(r_odd)))
        if INDEX_Q_NOPE_DIM > 0:
            amax = tl.maximum(amax, tl.max(tl.abs(x_nope)))
        index_q_scale = tl.div_rn(tl.maximum(amax, 1e-4), 448.0)
        index_q_scale = tl.math.exp2(tl.math.ceil(tl.math.log2(index_q_scale)))

        # Store quantized values to index_q_fp8
        if INDEX_Q_NOPE_DIM > 0:
            tl.store(
                fp8_base_ptr + nope_offset,
                tl.div_rn(x_nope, index_q_scale).to(tl.float8e4nv),
            )
        tl.store(
            fp8_rot_base + half_offset * 2,
            tl.div_rn(r_even, index_q_scale).to(tl.float8e4nv),
        )
        tl.store(
            fp8_rot_base + half_offset * 2 + 1,
            tl.div_rn(r_odd, index_q_scale).to(tl.float8e4nv),
        )
    else:
        # BF16 fallback: store raw bf16 Q, no quant. Scale-fold uses 1.0.
        index_q_scale = 1.0
        out_ty = index_q_fp8_ptr.type.element_ty
        if INDEX_Q_NOPE_DIM > 0:
            tl.store(fp8_base_ptr + nope_offset, x_nope.to(out_ty))
        tl.store(fp8_rot_base + half_offset * 2, r_even.to(out_ty))
        tl.store(fp8_rot_base + half_offset * 2 + 1, r_odd.to(out_ty))

    # FP8 weight-fold:
    #   index_weights_out = index_weights * q_scale * softmax_scale * head_scale
    # FP8 Q is stored WITHOUT a companion scale tensor; folding the per-token
    # q_scale (fp32) into the output weights lets the downstream logits kernel
    # apply it inline. In BF16 mode q_scale == 1.0 (no-op fold).
    index_weights = tl.load(
        index_weights_ptr + tok_idx * index_weights_stride + head_idx
    )
    index_weights = index_weights.to(tl.float32)
    index_weights *= index_q_scale
    index_weights *= index_weights_softmax_scale
    index_weights *= index_weights_head_scale
    tl.store(
        index_weights_out_ptr + tok_idx * index_weights_out_stride + head_idx,
        index_weights,
    )


def fused_indexer_q_rope_quant(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    # Index weights
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    use_fp8: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused RoPE + FP8 quantize Q for the sparse indexer.

    Returns:
        q_out: (T, H, HEAD_DIM). ``torch.float8_e4m3fn`` on FP8 HW (scale
            folded into ``weights_out``); ``torch.bfloat16`` on the non-FP8
            fallback (no quant, scale-fold is a no-op).
        weights_out: weights * q_scale * softmax_scale * head_scale.
    """
    assert positions.ndim == 1
    assert index_q.ndim == 3
    assert index_q_cos_sin_cache.ndim == 2

    if use_fp8 is None:
        use_fp8 = kv_cache_uses_fp8()

    num_tokens = positions.shape[0]
    num_index_q_heads = index_q.shape[1]
    index_q_head_dim = index_q.shape[2]

    index_weights_out = torch.empty_like(index_weights, dtype=torch.float32)
    out_dtype = torch.float8_e4m3fn if use_fp8 else torch.bfloat16
    index_q_out = torch.empty_like(index_q, dtype=out_dtype)
    _fused_indexer_q_rope_quant_kernel[(num_tokens, num_index_q_heads)](
        positions,
        index_q,
        index_q.stride(0),
        index_q.stride(1),
        index_q_cos_sin_cache,
        index_q_cos_sin_cache.stride(0),
        index_q_cos_sin_cache.shape[-1] // 2,
        index_q_out,
        index_q_out.stride(0),
        index_q_out.stride(1),
        index_q_head_dim,
        index_weights,
        index_weights.stride(0),
        index_weights_softmax_scale,
        index_weights_head_scale,
        index_weights_out,
        index_weights_out.stride(0),
        USE_FP8=use_fp8,
        num_warps=1,
    )
    return index_q_out, index_weights_out
