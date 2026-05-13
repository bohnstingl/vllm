# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import torch
from torch import Tensor

from ..op import register_op


@register_op
def gelu_new(x: Tensor) -> Tensor:
    """NewGELU activation (tanh-approximation GELU used by GPT-2/GPT-Neo)."""
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * torch.pow(x, 3.0))))
