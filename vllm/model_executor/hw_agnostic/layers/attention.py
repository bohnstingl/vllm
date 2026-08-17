# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Re-export of the attention layers for hw-agnostic modeling code.

Unlike the other hw-agnostic layers, attention is re-exported rather than
subclassed: the V1 framework keys KV-cache group discovery and attention
dispatch off the class object itself (``is``/``isinstance`` checks and
class-name lookups), so identity must be preserved. The KV-cache and
attention-backend layers underneath these classes are already dispatched
per hardware inside vLLM. The re-export gives the modeling backend a single
import surface and an out-of-tree platform an override seam, without a
separate attention kernel.
"""

from vllm.model_executor.layers.attention import (
    Attention,
    EncoderOnlyAttention,
    MLAAttention,
)

__all__ = ["Attention", "EncoderOnlyAttention", "MLAAttention"]
