# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HW-agnostic view of the custom-op plumbing, and the seam for its migration.

`CustomOp`, `PluggableLayer`, the two registries (`op_registry`,
`op_registry_oot`) and `maybe_get_oot_by_class` are identity-bearing: one shared
object each, so an OOT plugin registering against `vllm.model_executor.custom_op`
is honored on the hw-agnostic path too. During the transition the native module
`vllm.model_executor.custom_op` still *owns* (defines) them and this module
*borrows* them by re-export -- the native path is left untouched, since
model-specific development depends on it.

This module is the sole seam through which the hw-agnostic path reaches that
plumbing: every module under `vllm/model_executor/hw_agnostic/` imports it from
here, never from the native module directly (pinned by
`test_hw_agnostic_registry_access_goes_through_the_seam`). That keeps the
eventual ownership flip a one-file change.

Switch-over -- when the native registry is removed and the OOT-override API
becomes hw-agnostic-only:

    1. Move the `CustomOp` / `PluggableLayer` / `op_registry` /
       `op_registry_oot` / `maybe_get_oot_by_class` definitions here, so the
       hw-agnostic module becomes the owner.
    2. Flip the import below: `vllm.model_executor.custom_op` becomes the thin
       re-export shim, so its public import path keeps resolving during the
       native path's own deprecation.
    3. Retire the native readers (`maybe_get_oot_by_class` in the LoRA layers
       and `mm_encoder_attention`) as native layers go hardware-specific.

On top of the borrowed plumbing this module adds two thin wrappers,
`HwAgnosticCustomOp` and `HwAgnosticPluggableLayer`, which the hw-agnostic layers
import under the base classes' own names:

    from vllm.model_executor.hw_agnostic.custom_op import (
        HwAgnosticCustomOp as CustomOp,
    )

    @CustomOp.register("silu_and_mul")
    class SiluAndMul(CustomOp): ...
"""

import importlib

from vllm.logger import init_logger
from vllm.model_executor.custom_op import (
    CustomOp,
    PluggableLayer,
    maybe_get_oot_by_class,
    op_registry,
    op_registry_oot,
)

logger = init_logger(__name__)

__all__ = [
    "CustomOp",
    "HwAgnosticCustomOp",
    "HwAgnosticPluggableLayer",
    "PluggableLayer",
    "maybe_get_oot_by_class",
    "op_registry",
    "op_registry_oot",
    "validate_registered_overrides",
]


def _claim_op_name(op_cls: type, name: str) -> type:
    """Set the op name on `op_cls`, leaving `op_registry[name]` to the in-tree op.

    Only `cls.name` affects what runs: `enabled()` and `dispatch_forward` read it,
    and the out-of-tree swap in either `__new__` keys on `cls.__name__`.
    """
    op_cls.name = name  # type: ignore[attr-defined]
    if name not in op_registry:
        logger.debug(
            "hw-agnostic op %s.%s claims %r, which no in-tree op registered; "
            "`custom_ops` validation will report the name as non-existent.",
            op_cls.__module__,
            op_cls.__qualname__,
            name,
        )
    return op_cls


class HwAgnosticCustomOp(CustomOp):
    """HW-agnostic `CustomOp` wrapper, ensuring proper name registering."""

    @classmethod
    def register(
        cls,
        name: str,
        dynamic_arg_dims: dict[str, int | list[int]] | None = None,
    ):
        def decorator(op_cls):
            op_cls._dynamic_arg_dims = dynamic_arg_dims
            return _claim_op_name(op_cls, name)

        return decorator

    def forward_cuda(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)


class HwAgnosticPluggableLayer(PluggableLayer):
    """HW-agnostic `PluggableLayer` wrapper, ensuring proper name registering."""

    @classmethod
    def register(cls, name: str):
        def decorator(op_cls):
            return _claim_op_name(op_cls, name)

        return decorator


def validate_registered_overrides(hw_modules: dict[str, str]) -> None:
    """Check each registered override against the class it will replace.

    Raises:
        TypeError: If an override is not a subclass of the class published under
            its name.
    """
    for hw_name in hw_modules.values():
        hw_module = importlib.import_module(hw_name)
        for name, hw_cls in vars(hw_module).items():
            # Only classes the module defines itself were published.
            if isinstance(hw_cls, type) and hw_cls.__module__ == hw_module.__name__:
                oot_cls = op_registry_oot.get(name)
                if oot_cls is not None and not issubclass(oot_cls, hw_cls):
                    raise TypeError(
                        f"out-of-tree layer {oot_cls.__module__}."
                        f"{oot_cls.__qualname__} is registered under {name!r}, but "
                        f"it is not a subclass of {hw_cls.__module__}."
                        f"{hw_cls.__qualname__}, which is what this process "
                        f"instantiates under that name (VLLM_USE_HW_AGNOSTIC is "
                        f"set), so its __init__ would be skipped. It most likely "
                        f"captured vllm.model_executor.layers.{name} through an "
                        f"import that `hw_agnostic_layer_names()` does not cover."
                    )
