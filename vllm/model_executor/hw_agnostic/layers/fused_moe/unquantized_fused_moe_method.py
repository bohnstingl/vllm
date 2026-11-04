# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.hw_agnostic.custom_op import CustomOp
from vllm.model_executor.hw_agnostic.layers.fused_moe.experts.triton_moe import (
    TritonExperts,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.modular_kernel import (
    FusedMoEKernel,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.oracle.unquantized import (
    make_unquantized_moe_kernel,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    biased_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
)
from vllm.model_executor.utils import (
    is_weights_pre_processed,
    replace_parameter,
    set_weight_attrs,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

logger = init_logger(__name__)


@CustomOp.register("unquantized_fused_moe")
class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization, on the hw-agnostic Triton experts."""

    moe_kernel: FusedMoEKernel | None  # type: ignore[assignment]

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        activation_format = (
            mk.FusedMoEActivationFormat.BatchedExperts
            if moe.moe_parallel_config.use_batched_activation_format
            else mk.FusedMoEActivationFormat.Standard
        )
        supported, reason = TritonExperts.is_supported_config(
            TritonExperts, moe, None, None, activation_format
        )
        if not supported:
            raise NotImplementedError(
                f"The hw-agnostic MoE experts do not support this layer: {reason}. "
                "Unset VLLM_USE_HW_AGNOSTIC."
            )
        logger.info_once("Using hardware agnostic MoE experts TritonExperts")
        self.experts_cls = TritonExperts

    @property
    def supports_eplb(self) -> bool:
        return True

    def create_weights(
        self,
        layer: "RoutedExperts",
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        unpadded_intermediate = self.moe.intermediate_size_per_partition_unpadded
        assert unpadded_intermediate is not None
        if self.moe.is_act_and_mul:
            w13_up_dim = 2 * intermediate_size_per_partition
        else:
            w13_up_dim = intermediate_size_per_partition
        unpadded_hidden = self.moe.hidden_dim_unpadded
        assert unpadded_hidden is not None
        unpadded_up = unpadded_intermediate * (2 if self.moe.is_act_and_mul else 1)
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_up_dim,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        w13_weight.weight_loader_numel = num_experts * unpadded_up * unpadded_hidden
        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(num_experts, w13_up_dim, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
            w13_bias.weight_loader_numel = num_experts * unpadded_up
        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        w2_weight.weight_loader_numel = (
            num_experts * unpadded_hidden * unpadded_intermediate
        )
        if self.moe.has_bias:
            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)
            w2_bias.weight_loader_numel = num_experts * unpadded_hidden

    def _setup_kernel(
        self,
        layer: "RoutedExperts",
        w13: torch.Tensor,
        w2: torch.Tensor,
    ) -> None:
        is_weight_update = self.moe_kernel is not None  # type: ignore[has-type]
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)

        if not is_weight_update:
            # Setup moe kernel only on the first call. For the unquantized
            # method, moe_quant_config carries no quantized scales -- only
            # optional w{13,2}_bias references and SwiGLU gate params. Since
            # weight updates mutate those bias tensors in place, the kernel
            # does not need to be re-built.
            self._init_moe_kernel(layer)

            # No-op by default. Experts that need the layer itself for setup
            # convert_to_unquantized_kernel_format cannot express override
            # it: CPU prepacks into its grouped-gemm layout and captures the
            # router config; MoonEP picks up the [E+B] weight layout for
            # prefetch and the up projection.
            assert self.moe_kernel is not None
            self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    def _init_moe_kernel(self, layer: "RoutedExperts") -> None:
        """Build the MoE kernel from the layer's current (shuffled) weights."""
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.moe_quant_config is not None
        assert self.experts_cls is not None
        self.moe_kernel = make_unquantized_moe_kernel(
            quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            routing_tables=layer._expert_routing_tables(),
        )

    def process_weights_after_loading(self, layer: "RoutedExperts") -> None:
        super().process_weights_after_loading(layer)

        if is_weights_pre_processed():
            # Weights are already in runtime format; rebuild the kernel only.
            self._init_moe_kernel(layer)
            return

        self._setup_kernel(
            layer=layer,
            w13=layer.w13_weight,
            w2=layer.w2_weight,
        )

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        # SwiGLU/swigluoai gate params live on the layer; plumb them into the
        # quant config so the fused activation (e.g. swigluoai_uninterleave on
        # MiniMax-M3) receives gemm1_clamp_limit/alpha/beta.
        gemm1_alpha = getattr(layer, "swiglu_alpha", None)
        gemm1_beta = getattr(layer, "swiglu_beta", None)
        gemm1_clamp_limit = getattr(layer, "swiglu_limit", None)

        if self.moe.has_bias:
            return biased_moe_quant_config(
                layer.w13_bias,
                layer.w2_bias,
                gemm1_alpha=gemm1_alpha,
                gemm1_beta=gemm1_beta,
                gemm1_clamp_limit=gemm1_clamp_limit,
            )

        return FusedMoEQuantConfig.make(
            gemm1_alpha=gemm1_alpha,
            gemm1_beta=gemm1_beta,
            gemm1_clamp_limit=gemm1_clamp_limit,
        )

    def apply(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.forward(
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    def forward_native(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    def apply_monolithic(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            x,
            layer.w13_weight,
            layer.w2_weight,
            router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
            routing_sink=layer.routing_sink,
        )
