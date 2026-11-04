# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The hw-agnostic MoE experts and routing against their in-tree counterparts.

Both must compute what the in-tree versions compute, without the compiled
`_C`/`_moe_C` ops those use, except the `moe_align_block_size` token sort.
"""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.hw_agnostic.layers.fused_moe.experts.triton_moe import (
    TritonExperts as HwTritonExperts,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.router.topk import (
    grouped_topk as hw_grouped_topk,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Compares with CUDA/ROCm kernels."
)

NUM_EXPERTS, HIDDEN, INTERMEDIATE, TOPK = 16, 256, 128, 4


class _CompiledOps(TorchDispatchMode):
    """Records the compiled vLLM ops dispatched while active."""

    def __init__(self):
        super().__init__()
        self.ops: set[str] = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func.namespace in ("_C", "_moe_C"):
            self.ops.add(f"{func.namespace}::{func._opname}")
        return func(*args, **(kwargs or {}))


def _run_experts(experts_cls, activation, num_tokens, expert_parallel, seed=0):
    """Run `experts_cls.apply` on random inputs; workspaces are filled with a
    large constant, so a row the reduction must skip would corrupt the output."""
    torch.manual_seed(seed)
    device, dtype = "cuda", torch.bfloat16
    local_experts = NUM_EXPERTS // 2 if expert_parallel else NUM_EXPERTS
    hidden_states = torch.randn(num_tokens, HIDDEN, device=device, dtype=dtype) / 4
    w1 = torch.randn(
        local_experts, 2 * INTERMEDIATE, HIDDEN, device=device, dtype=dtype
    )
    w2 = torch.randn(local_experts, HIDDEN, INTERMEDIATE, device=device, dtype=dtype)
    topk_weights = torch.rand(num_tokens, TOPK, device=device)
    topk_ids = torch.stack(
        [torch.randperm(NUM_EXPERTS, device=device)[:TOPK] for _ in range(num_tokens)]
    ).to(torch.int32)
    expert_map = None
    if expert_parallel:
        expert_map = torch.full((NUM_EXPERTS,), -1, device=device, dtype=torch.int32)
        expert_map[:local_experts] = torch.arange(local_experts, device=device)

    moe_config = make_dummy_moe_config(
        num_experts=NUM_EXPERTS,
        experts_per_token=TOPK,
        hidden_dim=HIDDEN,
        intermediate_size=INTERMEDIATE,
        activation=activation,
    )
    experts = experts_cls(moe_config, FUSED_MOE_UNQUANTIZED_CONFIG)
    shape13, shape2, out_shape = experts.workspace_shapes(
        num_tokens,
        2 * INTERMEDIATE,
        HIDDEN,
        TOPK,
        NUM_EXPERTS,
        local_experts,
        None,
        activation,
    )
    workspace13 = torch.full(shape13, 1e4, device=device, dtype=dtype)
    workspace2 = torch.full(shape2, 1e4, device=device, dtype=dtype)
    output = torch.empty(out_shape, device=device, dtype=dtype)
    with _CompiledOps() as compiled:
        experts.apply(
            output,
            hidden_states,
            w1 / 16,
            w2 / 16,
            topk_weights,
            topk_ids,
            activation,
            NUM_EXPERTS,
            expert_map,
            None,
            None,
            workspace13.view(-1),
            workspace2.view(-1),
            None,
            False,
        )
    return output, compiled.ops


@pytest.mark.parametrize(
    "activation", [MoEActivation.SILU, MoEActivation.GELU, MoEActivation.GELU_TANH]
)
@pytest.mark.parametrize("num_tokens", [1, 7, 300])
@pytest.mark.parametrize("expert_parallel", [False, True])
def test_triton_experts_match_in_tree(
    default_vllm_config, activation, num_tokens, expert_parallel
):
    """The hw-agnostic experts reproduce the in-tree ones exactly, including the
    slots expert parallelism drops, and leave only the token sort compiled."""
    expected, in_tree_ops = _run_experts(
        TritonExperts, activation, num_tokens, expert_parallel
    )
    actual, hw_ops = _run_experts(
        HwTritonExperts, activation, num_tokens, expert_parallel
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert "_moe_C::moe_sum" in in_tree_ops
    # Few enough tokens skip the token sort, see `_prepare_expert_assignment`
    assert hw_ops <= {"_moe_C::moe_align_block_size"}


@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("logits_dtype", [torch.float32, torch.bfloat16])
def test_single_group_topk_matches_fused_topk(scoring_func, renormalize, logits_dtype):
    """Grouped top-k with a single group is the plain top-k the compiled
    `topk_softmax`/`topk_sigmoid` compute, without compiled ops. Gates of
    Qwen3-MoE-like models produce bf16 logits, which are scored in fp32."""
    torch.manual_seed(0)
    hidden_states = torch.randn(257, 64, device="cuda", dtype=torch.bfloat16)
    logits = torch.randn(257, 64, device="cuda", dtype=logits_dtype)
    expected_weights, expected_ids, _ = fused_topk(
        hidden_states, logits, 6, renormalize, scoring_func=scoring_func
    )
    with _CompiledOps() as compiled:
        weights, ids = hw_grouped_topk(
            hidden_states,
            logits,
            6,
            renormalize,
            num_expert_group=1,
            topk_group=1,
            scoring_func=scoring_func,
        )

    _assert_same_routing(weights, ids, expected_weights, expected_ids)
    assert not compiled.ops


def _assert_same_routing(weights, ids, expected_weights, expected_ids):
    """Same experts per token with the same weights, in any order."""
    order, expected_order = ids.argsort(-1), expected_ids.argsort(-1)
    torch.testing.assert_close(
        ids.gather(-1, order),
        expected_ids.to(ids.dtype).gather(-1, expected_order),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        weights.gather(-1, order), expected_weights.gather(-1, expected_order)
    )
