# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HW-agnostic GEMM dispatch for the hw-agnostic linear layers.

Only the CPU path and the vendor-agnostic ``torch.nn.functional.linear`` floor
are dispatched here. Accelerator-specific GEMMs (e.g. the ROCm aiter/skinny
kernels) are the concern of an out-of-tree platform plugin, which overrides the
layer rather than adding a branch here.
"""

from collections.abc import Callable

import torch

from vllm.model_executor.layers.utils import (
    cpu_unquantized_gemm,
    default_unquantized_gemm,
)
from vllm.platforms import current_platform


def dispatch_unquantized_gemm() -> Callable[..., torch.Tensor]:
    if current_platform.is_cpu():
        return cpu_unquantized_gemm
    return default_unquantized_gemm
