# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F

from vllm.model_executor.hw_agnostic.custom_op import (
    HwAgnosticCustomOp as CustomOp,
)


@CustomOp.register("silu_and_mul")
class SiluAndMul(CustomOp):
    """SwiGLU: ``x -> silu(x[:d]) * x[d:]`` where ``d = x.shape[-1] // 2``."""

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]


@CustomOp.register("gelu_and_mul")
class GeluAndMul(CustomOp):
    """GeGLU: ``x -> gelu(x[:d]) * x[d:]`` where ``d = x.shape[-1] // 2``."""

    def __init__(self, approximate: str = "none"):
        super().__init__()
        if approximate not in ("none", "tanh"):
            raise ValueError(f"Unknown approximate mode: {approximate}")
        self.approximate = approximate

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.gelu(x[..., :d], approximate=self.approximate) * x[..., d:]

    def extra_repr(self) -> str:
        return f"approximate={self.approximate!r}"


# Activation-and-mul ops keyed by HF activation name.
_ACTIVATION_AND_MUL_REGISTRY = {
    "gelu": lambda: GeluAndMul(),
    "geglu": lambda: GeluAndMul(),
    "gelu_pytorch_tanh": lambda: GeluAndMul(approximate="tanh"),
    "silu": lambda: SiluAndMul(),
    "swish": lambda: SiluAndMul(),
}


def get_act_and_mul_fn(act_fn_name: str) -> CustomOp:
    """Build the hw-agnostic activation-and-mul op named `act_fn_name`."""
    return _ACTIVATION_AND_MUL_REGISTRY[act_fn_name.lower()]()
