# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch of the Triton fused MoE kernel for the hw-agnostic path.

Copies of `invoke_fused_moe_triton_kernel` and `_prepare_expert_assignment` from
`vllm.model_executor.layers.fused_moe.fused_moe` without the Hopper TMA,
swap-AB and XPU tensor-descriptor paths. The Triton `fused_moe_kernel` itself
is shared with the in-tree path.
"""

from typing import Any

import torch

from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe_kernel
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.triton_utils import tl, triton


def invoke_fused_moe_triton_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor | None,
    B_scale: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: list[int] | None = None,
    B_bias: torch.Tensor | None = None,
):
    assert topk_weights is not None or not mul_routed_weight
    if topk_weights is not None and not topk_weights.is_contiguous():
        raise ValueError("topk_weights must be contiguous")
    assert sorted_token_ids is None or sorted_token_ids.stride(0) == 1

    SWAP_AB = False
    use_td = False

    if use_fp8_w8a8 or use_int8_w8a8:
        assert B_scale is not None
        assert block_shape is None or triton.cdiv(
            B.size(-2), block_shape[0]
        ) == B_scale.size(-2)
        assert block_shape is None or triton.cdiv(
            B.size(-1), block_shape[1]
        ) == B_scale.size(-1)
    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        assert block_shape is None or block_shape[0] == 0
    else:
        assert A_scale is None
        assert B_scale is None

    M = A.size(0)
    num_tokens = M * top_k
    if sorted_token_ids is not None:
        EM = sorted_token_ids.size(0)
        if A.size(0) < config["BLOCK_SIZE_M"]:
            # optimize for small batch_size.
            # We assume that top_ids of each token is unique,
            # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
            # and we can skip some invalid blocks.
            EM = min(
                sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"]
            )
    else:
        EM = num_tokens * config["BLOCK_SIZE_M"]
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),
    )
    HAS_BIAS = B_bias is not None

    config = config.copy()
    config["SPLIT_K"] = 1
    BLOCK_SIZE_K = config.pop("BLOCK_SIZE_K")
    if block_shape is not None:
        BLOCK_SIZE_K = min(BLOCK_SIZE_K, min(block_shape[0], block_shape[1]))
    config.pop("USE_TMA", None)
    use_tma_b = False
    b_desc = None

    # Triton treats 0-D tensor arguments as scalar values, but the kernel
    # loads tensor-wise activation scales through a pointer.
    if A_scale is not None and A_scale.ndim == 0:
        A_scale = A_scale.reshape(1)

    fused_moe_kernel[grid](
        A,
        B,
        b_desc,
        C,
        B_bias,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),
        B.size(2),
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
        A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
        B_scale.stride(0) if B_scale is not None else 0,
        B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
        B_scale.stride(1)
        if B_scale is not None and B_scale.ndim >= 2 and B_scale.size(1) > 1
        else 0,
        B_bias.stride(0) if B_bias is not None else 0,
        B_bias.stride(1) if B_bias is not None else 0,
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        per_channel_quant=per_channel_quant,
        naive_block_assignment=(sorted_token_ids is None),
        HAS_BIAS=HAS_BIAS,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        SWAP_AB=SWAP_AB,
        USE_TD=use_td,
        USE_TMA_B=use_tma_b,
        **config,
    )


def _weights_host_resident(*weights: torch.Tensor) -> bool:
    """True when any of the expert weights is read from host memory in place,
    as UVA weight offloading (``--offload-backend uva``) leaves them."""
    return any(getattr(w, "_vllm_is_uva_offloaded", False) for w in weights)


def _prepare_expert_assignment(
    topk_ids: torch.Tensor,
    config: dict[str, Any],
    num_tokens: int,
    top_k_num: int,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    *,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    block_shape: list[int] | None = None,
    ignore_invalid_experts: bool = False,
    weights_host_resident: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Prepare expert assignments for the aligned and low-latency Triton paths."""
    # SPARSITY_FACTOR is a heuristic margin ensuring tokens_in_chunk * top_k
    # activates only a small fraction of total experts
    # Skips moe_align_block_size and activates the `sorted_token_ids is None`
    # path of the fused_moe_kernel kernel, which reads an expert's weights once
    # per (token, expert) pair. Not for weights in host memory: there every
    # repeated read crosses PCIe, so group each expert's tokens instead.
    naive_block_assignment = (
        expert_map is None
        and not weights_host_resident
        and num_tokens * top_k_num * 4 <= global_num_experts
        and not (
            (use_int8_w8a16 or use_int4_w4a16)
            and block_shape is not None
            and block_shape[1] > 0
        )
    )

    if naive_block_assignment:
        return (
            None,
            topk_ids.view(-1),
            torch.full(
                (1,),
                topk_ids.numel() * config["BLOCK_SIZE_M"],
                dtype=torch.int32,
                device=topk_ids.device,
            ),
        )

    return moe_align_block_size(
        topk_ids,
        config["BLOCK_SIZE_M"],
        global_num_experts,
        expert_map,
        ignore_invalid_experts=ignore_invalid_experts,
    )
