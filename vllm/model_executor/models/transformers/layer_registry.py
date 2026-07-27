# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer provider resolution for the Transformers modeling backend.

When ``VLLM_USE_HW_AGNOSTIC`` is set, layer classes are resolved from
``vllm.model_executor.hw_agnostic.layers.<module>``, falling back to
``vllm.model_executor.layers.<module>`` for anything not yet ported. The
resolved source of every class is logged so it is clear which layers run
hw-agnostic and which fell back to vLLM.

Resolution happens at import time: the classes below are bound once when this
module is first imported. ``VLLM_USE_HW_AGNOSTIC`` is a launch-time setting
(fixed before the process starts, and the backend is only imported while building
a model), so import-time resolution reflects it correctly in real use.

Limitation: because binding is at import time, a single running process cannot
switch providers. Tests that exercise both settings must run each in a fresh
interpreter (vLLM's engine process provides this).

Use these at *construction* sites only. Do not subclass a resolved class or
``isinstance``-check against it. Symbols used that way (e.g. ``MoERunner``), and
those with no hw-agnostic equivalent (``conv``, ``pooler``), must be imported
directly from ``vllm.model_executor.layers``.
"""

import importlib

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_HW_PKG = "vllm.model_executor.hw_agnostic.layers"
_VLLM_PKG = "vllm.model_executor.layers"


def _resolve(module: str, name: str):
    """Return `name` from the hw-agnostic `module` when enabled and available,
    else from vLLM. Logs which source was used."""
    if envs.VLLM_USE_HW_AGNOSTIC:
        try:
            obj = getattr(importlib.import_module(f"{_HW_PKG}.{module}"), name)
            logger.info("Using hw-agnostic layer: %s", name)
            return obj
        except (ImportError, AttributeError):
            logger.warning(
                "hw-agnostic layer %s is not available; falling back to vLLM", name
            )
    return getattr(importlib.import_module(f"{_VLLM_PKG}.{module}"), name)


RMSNorm = _resolve("layernorm", "RMSNorm")
GemmaRMSNorm = _resolve("layernorm", "GemmaRMSNorm")


def get_act_and_mul_fn(act_fn_name: str):
    """Fused activation-and-mul op for `act_fn_name`, preferring hw-agnostic.

    Resolved per call because the op is name-parameterized: an activation with
    no hw-agnostic equivalent falls back to vLLM individually.
    """
    if envs.VLLM_USE_HW_AGNOSTIC:
        try:
            from vllm.model_executor.hw_agnostic.layers.activation import (
                get_act_and_mul_fn as hw_fn,
            )

            fn = hw_fn(act_fn_name)
            logger.info_once("Using hw-agnostic activation: %s", act_fn_name)
            return fn
        except (ImportError, KeyError):
            logger.warning_once(
                "hw-agnostic activation %s is not available; falling back to vLLM",
                act_fn_name,
            )
    from vllm.model_executor.layers.activation import get_act_and_mul_fn as vllm_fn

    return vllm_fn(act_fn_name)
