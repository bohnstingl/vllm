# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16-dequant fallback for block-scaled FP8 linear layers.

On platforms without native FP8 compute (e.g. NVIDIA Ampere / sm_80,
where Triton has no ``fp8e4nv`` support) the block-scaled FP8 GEMM and
its activation quant cannot run. This kernel instead dequantizes the
block-scaled FP8 weights to BF16 once at load time and runs a plain BF16
GEMM against the (un-quantized) BF16 activations. It is model-independent
and selected purely on the platform capability.

OOT override seam: an out-of-tree platform forces this fp8->bf16
block-scaled fallback by overriding ``Platform.supports_fp8()`` (return
False). A platform needing a different fallback GEMM prepends its own
kernel under its ``PlatformEnum`` in ``_POSSIBLE_FP8_BLOCK_KERNELS``.
"""

import torch
import torch.nn.functional as F

from vllm.model_executor.hw_agnostic.quantization.fp8_utils import (
    block_dequantize_weight,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .BlockScaledMMLinearKernel import Fp8BlockScaledMMLinearKernel


class Bf16DequantFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    """Dequant-to-BF16 fallback for block-scaled FP8 linear on non-FP8 HW."""

    # Activations stay BF16 -- no FP8 activation quant (which would also
    # fail to compile in Triton on sm_80).
    apply_input_quant = False

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not current_platform.is_cuda_alike():
            return False, "only CUDA-alike devices are supported."
        if current_platform.supports_fp8():
            # Prefer the native FP8 kernel when the HW supports it.
            return False, "only used as the non-FP8-compute fallback."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module):
        # Base class transposes/normalizes the block-scaled fp8 weight. Then
        # dequant to BF16 ONCE here (eager) so ``apply_weights`` never reads
        # fp8 inside the torch.compile region -- Triton can't lower fp8e4nv
        # on sm_80, even for a read.
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
        # Neutralize the scale so nothing downstream re-applies it.
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
        # Weight is already BF16 == out_dtype (dequantized at load time).
        # Plain GEMM; no fp8 anywhere in the compiled graph.
        weight_bf16 = self._get_layer_params(layer).weight
        out_dtype = self.config.out_dtype
        output_shape = [*x.shape[:-1], weight_bf16.shape[0]]
        input_2d = x.view(-1, x.shape[-1]).to(out_dtype)
        output = F.linear(input_2d, weight_bf16, bias)
        return output.view(*output_shape)

    def apply_block_scaled_mm(self, A, B, As, Bs):  # pragma: no cover
        raise NotImplementedError("BF16 dequant kernel overrides apply_weights.")
