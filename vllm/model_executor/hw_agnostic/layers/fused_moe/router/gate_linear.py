# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor import hw_agnostic
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

if TYPE_CHECKING:
    from vllm.model_executor.layers.linear import ReplicatedLinear
else:
    # Fixed when this module is imported, which `hw_agnostic.resolve` does only
    # with VLLM_USE_HW_AGNOSTIC set, so this is the hw-agnostic class once it
    # exists.
    ReplicatedLinear = hw_agnostic.resolve("linear", "ReplicatedLinear")


class GateLinear(ReplicatedLinear):
    """MoE gate linear layer for the hw-agnostic path.

    `vllm.model_executor.layers.fused_moe.router.gate_linear.GateLinear` with only
    its portable tier: `ReplicatedLinear` with the result cast to `out_dtype`. The
    CuTe DSL, cuBLAS and ROCm router GEMMs are not used.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        out_dtype: torch.dtype | None = None,
        params_dtype: torch.dtype | None = None,
        force_fp32_compute: bool = False,
        skip_bias_add: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        return_bias: bool = True,
    ):
        if force_fp32_compute:
            params_dtype = torch.float32
        super().__init__(
            input_size,
            output_size,
            bias=bias,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
        )
        self.out_dtype = out_dtype

    def set_out_dtype(self, out_dtype: torch.dtype) -> None:
        if self.out_dtype is not None:
            raise ValueError("out_dtype has already been set")
        self.out_dtype = out_dtype

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if self.out_dtype is not None and x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        base_output = super().forward(x)
        output_bias: Parameter | None = None
        if isinstance(base_output, tuple):
            output, output_bias = base_output
        else:
            output = base_output
        if self.out_dtype is not None and output.dtype != self.out_dtype:
            output = output.to(self.out_dtype)
        return output if not self.return_bias else (output, output_bias)
