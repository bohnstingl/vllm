# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 quantization overrides for the DeepSeek V4 OOT fallback.

The in-tree quant path is FP8-only. On non-FP8 hardware the FP8 checkpoint
is dequantized to BF16:

  * Linear (dense projections): a block-scaled FP8 -> BF16 dequant kernel is
    inserted into the linear-kernel selection list, and the FP8 Triton kernel
    is declined via the ``supports_fp8()`` gate added in the FP8 tree.
  * MoE experts: a ``Fp8MoEMethod`` subclass dequantizes the stacked expert
    weights to BF16 (runtime by default, or once at load if the platform sets
    ``moe_dequant_at_load``) and runs the unquantized grouped GEMM.

The BF16 quant config is swapped in for ``deepseek_v4_fp8`` via
``register_quantization_config`` (its customized entry wins the
``method_to_config`` overlay), so no in-tree quant class is edited.
"""

import torch
import torch.nn.functional as F

from vllm.model_executor.hw_agnostic.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    biased_moe_quant_config,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.fused_moe_forward import (
    fused_moe_forward,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.routed_experts import (
    RoutedExperts,
)
from vllm.model_executor.hw_agnostic.quantization.fp8_moe_method import (
    Fp8MoEMethod,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform


def block_dequantize_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: tuple[int, int],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize a block-scaled FP8 weight to ``out_dtype``.

    The last two dims of ``weight`` are ``(N, K)``; any leading dims (e.g. a
    stacked expert dim ``E``) are preserved. ``weight_scale`` carries one
    scalar per ``block_n x block_k`` tile. Used by both the linear (2D) and
    MoE (3D) BF16-dequant fallbacks.
    """
    block_n, block_k = block_size
    n, k = weight.shape[-2], weight.shape[-1]
    w = weight.to(torch.float32)
    scale = weight_scale.to(torch.float32)
    ndim = w.ndim
    scale = scale.repeat_interleave(block_n, dim=ndim - 2).repeat_interleave(
        block_k, dim=ndim - 1
    )
    scale = scale[..., :n, :k]
    return (w * scale).to(out_dtype)


class BF16Fp8MoEMethod(Fp8MoEMethod):
    """FP8 MoE method that dequantizes experts to BF16 on non-FP8 HW."""

    def __init__(self, quant_config, layer: RoutedExperts):
        super().__init__(quant_config, layer)
        self.fp8_compute = current_platform.supports_fp8()
        # Runtime (default) vs load-time dequant. Runtime keeps weights
        # FP8-resident (memory-safe); load-time trades HBM for per-forward
        # compute and is opted into by the platform attribute.
        self._moe_dequant_at_load = getattr(
            current_platform, "moe_dequant_at_load", False
        )

    def _dequant_expert_weight(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Dequant a stacked expert weight ``[E, N, K]`` to ``out_dtype``."""
        if self.block_quant:
            assert self.weight_block_size is not None
            block_n, block_k = self.weight_block_size
            return block_dequantize_weight(
                weight, weight_scale, (block_n, block_k), out_dtype
            )
        w = weight.to(torch.float32)
        scale = weight_scale.to(torch.float32)
        scale = scale.reshape(scale.shape[0], *([1] * (w.ndim - 1)))
        return (w * scale).to(out_dtype)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = getattr(layer, f"w13_{self.weight_scale_name}")
        w2_scale = getattr(layer, f"w2_{self.weight_scale_name}")
        w13_input_scale = layer.w13_input_scale
        w2_input_scale = layer.w2_input_scale

        if self.quant_config.activation_scheme == "static":
            from vllm.model_executor.hw_agnostic.quantization.utils import (
                process_fp8_input_tensor_strategy_moe,
            )

            assert not self.block_quant
            assert w13_input_scale is not None and w2_input_scale is not None
            w13_input_scale, w2_input_scale = process_fp8_input_tensor_strategy_moe(
                w13_input_scale, w2_input_scale
            )
            replace_parameter(layer, "w13_input_scale", w13_input_scale)
            replace_parameter(layer, "w2_input_scale", w2_input_scale)

        if not self.block_quant:
            from vllm.model_executor.hw_agnostic.quantization.utils import (
                process_fp8_weight_tensor_strategy_moe,
            )

            shard_size = layer.intermediate_size_per_partition
            w13, w13_scale = process_fp8_weight_tensor_strategy_moe(
                w13, w13_scale, shard_size, layer.local_num_experts
            )

        # Load-time dequant (only when the platform opts in): frees the FP8
        # copy so ``apply`` takes the plain (bf16) path.
        if not self.fp8_compute and self._moe_dequant_at_load:
            w13 = self._dequant_expert_weight(w13, w13_scale, layer.orig_dtype)
            w2 = self._dequant_expert_weight(w2, w2_scale, layer.orig_dtype)

        self._setup_kernel(layer, w13, w2, w13_scale, w2_scale)

    def get_fused_moe_quant_config(self, layer: RoutedExperts) -> FusedMoEQuantConfig:
        if not self.fp8_compute:
            # Weights are (or will be) BF16; run the unquantized grouped GEMM.
            return biased_moe_quant_config(
                w1_bias=getattr(layer, "w13_bias", None),
                w2_bias=getattr(layer, "w2_bias", None),
                gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
            )
        return super().get_fused_moe_quant_config(layer)

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.experts is not None
        w13 = layer.w13_weight
        w2 = layer.w2_weight

        # Runtime dequant: weights still FP8-resident (1 byte). Load-time
        # dequant already produced BF16 (2 bytes), so this is skipped.
        if not self.fp8_compute and w13.element_size() == 1:
            w13 = self._dequant_expert_weight(
                w13, getattr(layer, f"w13_{self.weight_scale_name}"), layer.orig_dtype
            )
            w2 = self._dequant_expert_weight(
                w2, getattr(layer, f"w2_{self.weight_scale_name}"), layer.orig_dtype
            )
        return fused_moe_forward(
            self.experts,
            x,
            w13,
            w2,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
        )


def _make_bf16_linear_kernel_cls():
    """Build the BF16-dequant block-scaled linear kernel class.

    Defined lazily so the in-tree base class import happens at registration
    time (after the platform is resolved).
    """
    from vllm.model_executor.hw_agnostic.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (  # noqa: E501
        Fp8BlockScaledMMLinearKernel,
    )

    class Bf16DequantFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
        """Dequant-to-BF16 fallback for block-scaled FP8 linear on non-FP8 HW."""

        # Activations stay BF16 -- no FP8 activation quant.
        apply_input_quant = False

        @classmethod
        def is_supported(cls, compute_capability=None):
            if not current_platform.is_cuda_alike():
                return False, "only CUDA-alike devices are supported."
            if current_platform.supports_fp8():
                return False, "only used as the non-FP8-compute fallback."
            return True, None

        def process_weights_after_loading(self, layer: torch.nn.Module):
            super().process_weights_after_loading(layer)
            params = self._get_layer_params(layer)
            weight = params.weight
            weight_scale = (
                params.weight_scale
                if params.weight_scale_inv is None
                else params.weight_scale_inv
            )
            scale_attr = (
                params.WEIGHT_SCALE
                if params.weight_scale_inv is None
                else params.WEIGHT_SCALE_INV
            )
            block_shape = self.weight_group_shape
            block_size = (block_shape.row, block_shape.col)
            weight_bf16 = block_dequantize_weight(
                weight, weight_scale, block_size, self.config.out_dtype
            )
            replace_parameter(layer, params.WEIGHT, weight_bf16)
            replace_parameter(
                layer, scale_attr, torch.ones_like(weight_scale, dtype=torch.float32)
            )

        def apply_weights(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
            **kwargs,
        ) -> torch.Tensor:
            weight_bf16 = self._get_layer_params(layer).weight
            out_dtype = self.config.out_dtype
            output_shape = [*x.shape[:-1], weight_bf16.shape[0]]
            input_2d = x.view(-1, x.shape[-1]).to(out_dtype)
            output = F.linear(input_2d, weight_bf16, bias)
            return output.view(*output_shape)

        def apply_block_scaled_mm(self, A, B, As, Bs):  # pragma: no cover
            raise NotImplementedError(
                "BF16 dequant kernel overrides apply_weights."
            )

    return Bf16DequantFp8BlockScaledMMKernel


def _make_bf16_quant_config_cls():
    """Build the BF16 DeepSeek V4 quant config subclass lazily."""
    from vllm.models.deepseek_v4.hw_agnostic.quantization.quant_config import (
        DeepseekV4FP8Config,
    )

    class DeepseekV4BF16Config(DeepseekV4FP8Config):
        """DSv4 quant config that routes FP8 MoE experts through the BF16
        dequant method. Linear/attention still use the in-tree FP8 method
        classes; the linear BF16 fallback happens at kernel-selection level.
        """

        def get_quant_method(self, layer: torch.nn.Module, prefix: str):
            method = super().get_quant_method(layer, prefix)
            if isinstance(method, Fp8MoEMethod) and not isinstance(
                method, BF16Fp8MoEMethod
            ):
                return BF16Fp8MoEMethod(self, layer)
            return method

    return DeepseekV4BF16Config


def register() -> None:
    """Register BF16 linear kernel + quant config for the DSv4 fallback."""
    from vllm.model_executor.hw_agnostic.kernels.linear import (
        _POSSIBLE_FP8_BLOCK_KERNELS,
    )
    from vllm.model_executor.layers.quantization import (
        register_quantization_config,
    )
    from vllm.platforms import PlatformEnum

    # 1. Insert the BF16-dequant linear kernel ahead of the portable Torch
    #    fallback (but the in-tree Triton FP8 kernel now declines via its
    #    supports_fp8() gate, so this is what wins on non-FP8 HW).
    kernel_cls = _make_bf16_linear_kernel_cls()
    cuda_kernels = _POSSIBLE_FP8_BLOCK_KERNELS.setdefault(PlatformEnum.CUDA, [])
    if kernel_cls not in cuda_kernels:
        # Place after the Triton kernel so FP8 HW is unaffected; on non-FP8
        # HW the Triton kernel's is_supported() returns False and this wins.
        cuda_kernels.insert(len(cuda_kernels) - 1, kernel_cls)

    # 2. Swap the DSv4 quant config for the BF16 variant. The customized
    #    entry wins the method_to_config overlay in get_quantization_config.
    register_quantization_config("deepseek_v4_fp8")(_make_bf16_quant_config_cls())
