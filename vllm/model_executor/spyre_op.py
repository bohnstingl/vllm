# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

import torch.utils._pytree as pytree

from vllm.config import get_cached_compilation_config
from vllm.logger import init_logger
from vllm.model_executor.utils import maybe_disable_graph_partition
from vllm.platforms import current_platform

logger = init_logger(__name__)

from vllm.model_executor.layers.layernorm import RMSNorm

def _prepare_inputs_on_spyre(*args):
    def _convert_to_spyre(arg):
        return arg.to(dtype=torch.float16).to(device=torch.device("spyre")) if isinstance(arg, torch.Tensor) else arg
    
    return pytree.tree_map(_convert_to_spyre, args)[0]

@RMSNorm.register_oot
class SpyreRMSNorm(RMSNorm):
    """OOT version of RMSNorm for IBM's Spyre device"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # # NOTE(woosuk): Here we assume that vLLM was built for only one
        # # specific backend. Currently, we do not support dynamic dispatching.
        # compilation_config = get_cached_compilation_config()
        
        # # NOTE(shen-shanshan): CustomOp object can be enforce enabled, e.g.,
        # # enable device-specific kernels in ViT models when enabling graph
        # # mode. By default, it will follow the compilation_config to determine
        # # whether enable itself.
        # # This enforce_enable mechanism will be removed after we adding a
        # # separate compilation_config for multi-modal part.
        # enabled = self._enforce_enable or self.enabled()
        # if enabled:
        #     compilation_config.enabled_custom_ops.update([self.__class__.name])
        # else:
        #     compilation_config.disabled_custom_ops.update([self.__class__.name])

        enabled = False
        # enabled = True

        if not enabled:
            # # Compile forward_native to avoid eager torch ops if inside
            # # opaque torch custom op (e.g. fused_moe, unified_attention, etc.)
            # self._forward = self.maybe_compile(self.forward_native, enable=True)
            self._fwd_spyre = torch.compile(self._forward_static_spyre,
                                            dynamic=False)
        else:
            self._fwd_spyre = self._forward_static_spyre
        
        # self.forward_static = self.forward_static_spyre
        
    @staticmethod
    def _forward_static_spyre(
        x: torch.Tensor,
        variance_epsilon: float,
        hidden_size: int,
        orig_dtype: torch.dtype,
        weight: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        variance_size_override: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        
        x = x.transpose(1, 0).contiguous()
        
        if residual is not None:
            # residual promoted f16->f32 automatically,
            # otherwise Inductor eliminates the casts to and from f16,
            # increasing memory usage (and complicating pattern matching)
            x = x + residual
            residual = x.to(orig_dtype)

        # if x.shape[-1] != hidden_size:
        #     raise ValueError(
        #         f"Expected hidden_size to be {hidden_size}, but found: {x.shape[-1]}"
        #     )

        if variance_size_override is None:
            x_var = x
        else:
            if hidden_size < variance_size_override:
                raise ValueError(
                    "Expected hidden_size to be at least "
                    f"{variance_size_override}, but found: {hidden_size}"
                )

            x_var = x[:, :, :variance_size_override]

        variance = x_var * x_var
        variance = variance.mean(dim=0)
        x = x * torch.rsqrt(variance + variance_epsilon)[None, :]
        
        x = x.transpose(1, 0).contiguous()

        if weight is not None:
            x = x * weight
            
        if residual is None:
            return x
        else:
            return x, residual

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""

        if residual is not None:
            raise NotImplementedError('TODO!')

        out = self._fwd_spyre(
            _prepare_inputs_on_spyre([x])[0],
            _prepare_inputs_on_spyre([torch.ones((x.shape[0])) * self.variance_epsilon])[0],
            self.hidden_size,
            torch.float16,
            _prepare_inputs_on_spyre([self.weight.data])[0] if self.has_weight else None,
            residual,
            self.variance_size_override,
        )
        
        spyre_out = out.cpu()
        return spyre_out.to(torch.bfloat16)