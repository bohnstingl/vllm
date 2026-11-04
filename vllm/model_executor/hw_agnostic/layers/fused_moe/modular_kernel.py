# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The modular fused MoE kernel for the hw-agnostic path.

Copies of `FusedMoEKernelModularImpl` and `FusedMoEKernel` from
`vllm.model_executor.layers.fused_moe.modular_kernel`, without the CUDA
Humming-experts setup, the CPU workspace allocation and the ROCm AITER output
aliasing. The base classes they compose are shared with the in-tree path.
"""

from math import prod
from typing import final

import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    ExpertTokensMetadata,
    FusedMoEExperts,
    FusedMoEExpertsModular,
    FusedMoEKernelMonolithicImpl,
    FusedMoEPrepareAndFinalize,
    FusedMoEPrepareAndFinalizeModular,
)
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsSink,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.v1.worker.ubatching import (
    dbo_enabled,
    dbo_maybe_run_recv_hook,
    dbo_register_recv_hook,
    dbo_yield,
)
from vllm.v1.worker.workspace import current_workspace_manager


class FusedMoEKernelModularImpl:
    def __init__(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalizeModular,
        fused_experts: FusedMoEExpertsModular,
    ):
        self.prepare_finalize = prepare_finalize
        self.fused_experts = fused_experts
        self.shared_experts: SharedExperts | None = None
        moe_parallel_config = fused_experts.moe_config.moe_parallel_config
        self.moe_parallel_config = moe_parallel_config
        self.is_dp_ep = (
            moe_parallel_config is not None
            and moe_parallel_config.dp_size > 1
            and moe_parallel_config.use_ep
        )

    def _allocate_buffers(
        self,
        out_dtype: torch.dtype,
        device: torch.device,
        M_chunk: int,
        M_full: int,
        N: int,
        K: int,
        top_k: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Allocate temporary and output buffers for the fused experts op.
        Inputs:
        - out_dtype: output type of workspace and output tensors.
        - device: the device of the workspace and output tensors.
        See `workspace_shapes` for a description of the remainder of arguments.
        Returns the workspace tensors, output tensor, and whether the output
        uses a prepare/finalize-provided allocation.
        """
        assert M_full > 0 and M_chunk > 0

        workspace_dtype = self.fused_experts.workspace_dtype(out_dtype)

        # Get intermediate workspace shapes based off the chunked M size.
        workspace13_shape, workspace2_shape, _ = self.fused_experts.workspace_shapes(
            M_chunk,
            N,
            K,
            top_k,
            global_num_experts,
            local_num_experts,
            expert_tokens_meta,
            activation,
        )

        # Get final output shape based on the full M size.
        _, _, fused_out_shape = self.fused_experts.workspace_shapes(
            M_full,
            N,
            K,
            top_k,
            global_num_experts,
            local_num_experts,
            expert_tokens_meta,
            activation,
        )

        fused_out_alias = self.prepare_finalize.allocate_fused_expert_output(
            fused_out_shape,
            workspace_dtype,
            device,
            self.fused_experts.finalize_weight_and_reduce_impl(),
        )

        # We can reuse the memory between cache1 and cache3 because by the
        # time we need cache3, we're done with cache1.
        # Reuse workspace13 for the output since there is only one chunk.
        # Keep the original fused-output reservation even when the output is
        # externally aliased. The workspace manager profiles and locks a
        # process-wide high-water mark; later graph shapes may need this
        # storage for workspace13.
        max_shape_size = max(prod(workspace13_shape), prod(fused_out_shape))

        common_workspace, workspace2 = current_workspace_manager().get_simultaneous(
            ((max_shape_size,), workspace_dtype),
            (workspace2_shape, workspace_dtype),
        )
        workspace13 = _resize_cache(common_workspace, workspace13_shape)
        if fused_out_alias is None:
            fused_out = _resize_cache(common_workspace, fused_out_shape)
        else:
            assert fused_out_alias.shape == fused_out_shape
            assert fused_out_alias.dtype == workspace_dtype
            assert fused_out_alias.device == device
            assert fused_out_alias.is_contiguous()
            fused_out = fused_out_alias

        return workspace13, workspace2, fused_out, fused_out_alias is not None

    def _maybe_apply_shared_experts(
        self,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ):
        if shared_experts is not None:
            assert self.prepare_finalize.supports_async()
            assert shared_experts_input is not None
            shared_experts(
                shared_experts_input,
                SharedExpertsOrder.MK_INTERNAL_OVERLAPPED,
            )

    def _prepare(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        ExpertTokensMetadata | None,
        torch.Tensor,
        torch.Tensor,
    ]:
        """The _prepare method is a wrapper around self.prepare_finalize.prepare
        that handles DBO and async.
        """
        if not self.prepare_finalize.supports_async():
            # We shouldn't be running an a2a kernel that doesn't
            # support async prepare/finalize
            # TODO(lucas): enable in follow-up
            assert not dbo_enabled() or self.prepare_finalize.supports_dbo()

            (
                a1q,
                a1q_scale,
                expert_tokens_meta,
                _expert_topk_ids,
                _expert_topk_weights,
            ) = self.prepare_finalize.prepare(
                hidden_states,
                topk_weights,
                topk_ids,
                global_num_experts,
                expert_map,
                apply_router_weight_on_input,
                self.fused_experts.quant_config,
                defer_input_quant=self.fused_experts.expects_unquantized_inputs,
            )
        else:
            # Overlap shared expert compute with all2all dispatch.
            dbo_maybe_run_recv_hook()
            prepare_ret = self.prepare_finalize.prepare_async(
                hidden_states,
                topk_weights,
                topk_ids,
                global_num_experts,
                expert_map,
                apply_router_weight_on_input,
                self.fused_experts.quant_config,
                defer_input_quant=self.fused_experts.expects_unquantized_inputs,
            )

            # TODO(lucas): refactor this in the alternative schedules followup
            # currently unpack if we have hook + receiver pair or just
            # receiver (see finalize_async docstring)
            hook, receiver = (
                prepare_ret if isinstance(prepare_ret, tuple) else (None, prepare_ret)
            )

            if hook is not None:
                if dbo_enabled():
                    # If DBO is being used, register the hook with the ubatch
                    # context and call it in dbo_maybe_run_recv_hook instead of
                    #  passing it to the receiver.
                    dbo_register_recv_hook(hook)
                    dbo_yield()
                else:
                    hook()

            (
                a1q,
                a1q_scale,
                expert_tokens_meta,
                _expert_topk_ids,
                _expert_topk_weights,
            ) = receiver()

        # Maybe prepare gathered topk_ids and topk_weights from other EP ranks.
        topk_ids = topk_ids if _expert_topk_ids is None else _expert_topk_ids
        topk_weights = (
            topk_weights if _expert_topk_weights is None else _expert_topk_weights
        )

        return a1q, a1q_scale, expert_tokens_meta, topk_ids, topk_weights

    def _fused_experts(
        self,
        in_dtype: torch.dtype,
        a1q: torch.Tensor,
        a1q_scale: torch.Tensor | None,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        local_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        expert_tokens_meta: ExpertTokensMetadata | None,
        output_alias: torch.Tensor | None = None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        _, M_full, N, K, top_k = self.fused_experts.moe_problem_size(
            a1q, w1, w2, topk_ids
        )

        # This happens when none of the tokens from the all2all reach this
        # EP rank. Also, note that this is only relevant for CUDAGraph
        # incompatible all2all kernels like the DeepEP high-throughput
        # kernels. CUDAGraph compatible all2all kernels like the DeepEP
        # low-latency kernels are always batched and can never run into
        # the tensor.numel() == 0 case.
        if M_full == 0:
            out_dim = self.fused_experts.output_hidden_dim(a1q.size(-1))
            return a1q.new_empty((*a1q.shape[:-1], out_dim), dtype=in_dtype)

        workspace13, workspace2, fused_out, fused_out_is_specialized = (
            self._allocate_buffers(
                in_dtype,
                a1q.device,
                M_full,
                M_full,
                N,
                K,
                top_k,
                global_num_experts,
                local_num_experts,
                expert_tokens_meta,
                activation,
            )
        )

        use_output_alias = (
            not fused_out_is_specialized
            and output_alias is not None
            and output_alias.shape == fused_out.shape
            and output_alias.dtype == fused_out.dtype
            and output_alias.device == fused_out.device
            and output_alias.is_contiguous()
        )

        # If caller's output buffer already matches fused_out shape/dtype, alias
        # to skip the redundant copy in TopKWeightAndReduceNoOP.apply downstream.
        # This eliminates ~94% of __amd_rocclr_copyBuffer events (Copy 2 of the
        # double-copy MoE write-back path).
        if use_output_alias:
            fused_out = output_alias

        unfinalized = self.fused_experts.apply(
            output=fused_out,
            hidden_states=a1q,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            a1q_scale=a1q_scale,
            a2_scale=self.fused_experts.a2_scale,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=expert_tokens_meta,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

        # Experts that stopped after GEMM2 wrote nothing into `fused_out`; the
        # top-k reduction they left open belongs to whoever takes this.
        if isinstance(unfinalized, UnfinalizedMoEOutput):
            return unfinalized

        return fused_out

    def _finalize(
        self,
        output: torch.Tensor,
        fused_out: torch.Tensor,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        """The _finalize method is a wrapper around self.prepare_finalize.finalize
        that handles DBO, async and shared expert overlap.

        Args:
            output: Tensor the finalized result is written into.
            fused_out: The unweighted, unreduced output of the fused experts.
            hidden_states: The input tensor to the MoE layer.
            topk_weights: A map of row to expert weights.
            topk_ids: A map of row to expert id.
            apply_router_weight_on_input: True if the router weights were
                already applied on the input.
            shared_experts: SharedExperts | None. The shared experts if any.
            shared_experts_input: Optional separate input for shared experts.
                When latent MoE is used, hidden_states is the latent-projected
                tensor (smaller dimension) used by routed experts, while
                shared_experts_input is the original hidden_states (full
                dimension) needed by the shared expert MLP.

        """
        if not self.prepare_finalize.supports_async():
            assert not dbo_enabled() or self.prepare_finalize.supports_dbo()

            self.prepare_finalize.finalize(
                output,
                fused_out,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input,
                self.fused_experts.finalize_weight_and_reduce_impl(),
            )
        else:
            finalize_ret = self.prepare_finalize.finalize_async(
                output,
                fused_out,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input,
                self.fused_experts.finalize_weight_and_reduce_impl(),
            )
            self._maybe_apply_shared_experts(shared_experts, shared_experts_input)

            # TODO(lucas): refactor this in the alternative schedules followup
            # currently unpack if we have hook + receiver pair or just
            # receiver (see finalize_async docstring)
            hook, receiver = (
                finalize_ret
                if isinstance(finalize_ret, tuple)
                else (None, finalize_ret)
            )

            if hook is not None:
                if dbo_enabled():
                    # If DBO is being used, register the hook with the ubatch
                    # context and call it in dbo_maybe_run_recv_hook instead of
                    #  passing it to the receiver.
                    dbo_register_recv_hook(hook)
                    dbo_yield()
                else:
                    hook()

            receiver()

        return output

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        activation: MoEActivation = MoEActivation.SILU,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        shared_experts: SharedExperts | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        """This function computes a Mixture of Experts (MoE) layer using two sets
        of weights, w1 and w2, and top-k gating mechanism.

        Args:
            hidden_states: (torch.Tensor): The input tensor to the MoE layer.
            w1 (torch.Tensor): The first set of expert weights.
            w2 (torch.Tensor): The second set of expert weights.
            topk_ids (torch.Tensor): A map of row to expert id.
            topk_weights (torch.Tensor): The topk weights applied at the end of
                the layer.
            activation (MoEActivation): The activation function to apply after
                the first MoE layer.
            global_num_experts (int): The total number of experts in the global
                expert space.
            expert_map (Optional[torch.Tensor]): A tensor mapping expert indices
                from the global expert space to the local expert space of the
                expert parallel shard.
            apply_router_weight_on_input (bool): When true, the topk weights are
                applied directly on the inputs. This is only applicable when
                topk is 1.
            shared_experts: SharedExperts | None. The shared experts if any.
            shared_experts_input (Optional[torch.Tensor]): Optional separate
                input for shared experts. For latent MoE, this is the original
                hidden_states before latent projection.

        Returns:
            torch.Tensor: The output tensor after applying the MoE layer, or
            the unfinalized output when the experts left the top-k reduction to
            a fused consumer.

        """
        output = hidden_states.new_empty(
            (
                *hidden_states.shape[:-1],
                self.fused_experts.output_hidden_dim(hidden_states.size(-1)),
            )
        )

        local_num_experts = w1.shape[0]
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        a1q, a1q_scale, expert_tokens_meta, topk_ids, topk_weights = self._prepare(
            hidden_states,
            topk_weights,
            topk_ids,
            global_num_experts,
            expert_map,
            apply_router_weight_on_input,
        )

        # Stash the original unquantized hidden states on the LoRA context
        # so apply_w13_lora sees correct-magnitude activations instead of
        # the potentially quantized values produced by _prepare().
        lora_ctx = getattr(self.fused_experts, "_lora_context", None)
        if lora_ctx is not None:
            lora_ctx.original_hidden_states = hidden_states

        fused_out = self._fused_experts(
            in_dtype=hidden_states.dtype,
            a1q=a1q,
            a1q_scale=a1q_scale,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_tokens_meta=expert_tokens_meta,
            output_alias=output,
        )

        if lora_ctx is not None:
            lora_ctx.original_hidden_states = None

        if isinstance(fused_out, UnfinalizedMoEOutput):
            # Nothing below can run on an unfinalized output: finalize is the
            # local top-k reduction (and, for DP/EP, the combine) that the
            # consumer takes over.
            if not self.prepare_finalize.supports_deferred_moe_finalize():
                raise RuntimeError(
                    f"{type(self.prepare_finalize).__name__} cannot pass through "
                    "a deferred MoE output."
                )
            return fused_out

        return self._finalize(
            output,
            fused_out,
            hidden_states,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )


@final
class FusedMoEKernel:
    def __init__(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        fused_experts: FusedMoEExperts,
    ):
        super().__init__()

        # Initialize the implementation (monolithic or modular).
        self.impl: FusedMoEKernelModularImpl | FusedMoEKernelMonolithicImpl
        if isinstance(
            prepare_finalize, FusedMoEPrepareAndFinalizeModular
        ) and isinstance(fused_experts, FusedMoEExpertsModular):
            self.impl = FusedMoEKernelModularImpl(
                prepare_finalize,
                fused_experts,
            )

        else:
            raise ValueError(
                "prepare_finalize and fused_experts must both be either monolithic "
                f"or non-monolithic but got {prepare_finalize.__class__.__name__} "
                f"and {fused_experts.__class__.__name__}"
            )

        self._post_init_setup()

    @property
    def can_overlap_shared_experts(self) -> bool:
        if isinstance(self.impl, FusedMoEKernelModularImpl):
            return self.impl.prepare_finalize.supports_async()
        else:
            return False

    @property
    def is_monolithic(self) -> bool:
        return isinstance(self.impl, FusedMoEKernelMonolithicImpl)

    @property
    def prepare_finalize(self) -> FusedMoEPrepareAndFinalize:
        return self.impl.prepare_finalize

    @property
    def fused_experts(self) -> FusedMoEExperts:
        return self.impl.fused_experts

    @property
    def moe_config(self) -> FusedMoEConfig:
        return self.fused_experts.moe_config

    def supports_lora(self) -> bool:
        return self.fused_experts.supports_lora()

    def _post_init_setup(self):
        """Resolve any leftover setup dependencies between self.prepare_finalize
        and self.fused_experts here.
        """
        self.prepare_finalize.post_init_setup(self.impl.fused_experts)
        assert (
            self.prepare_finalize.activation_format
            == self.fused_experts.activation_format()
        )

    def output_is_reduced(self) -> bool:
        """Indicates whether or not the output of fused MoE kernel
        is reduced across all ranks.
        """
        return self.prepare_finalize.output_is_reduced()

    def supports_deferred_moe_finalize(self) -> bool:
        return self.prepare_finalize.supports_deferred_moe_finalize()

    def apply_monolithic(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        # grouped topk + fused topk bias parameters
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
        *,
        routing_sink: RoutedExpertsSink | None,
        input_ids: torch.Tensor | None = None,
        hash_indices_table: torch.Tensor | None = None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        assert isinstance(self.impl, FusedMoEKernelMonolithicImpl)
        return self.impl.apply(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            router_logits=router_logits,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            num_expert_group=num_expert_group,
            e_score_correction_bias=e_score_correction_bias,
            routed_scaling_factor=routed_scaling_factor,
            topk_group=topk_group,
            routing_sink=routing_sink,
            input_ids=input_ids,
            hash_indices_table=hash_indices_table,
        )

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        shared_experts: SharedExperts | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        assert isinstance(self.impl, FusedMoEKernelModularImpl)
        return self.impl.apply(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )
