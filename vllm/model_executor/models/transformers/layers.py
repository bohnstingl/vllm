# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer provider resolution for the Transformers modeling backend.

When ``VLLM_USE_HW_AGNOSTIC`` is set, layer symbols are imported from
``vllm.model_executor.hw_agnostic.layers.<module>``, falling back to
``vllm.model_executor.layers.<module>`` for anything not yet ported. The
resolved source of every symbol is logged so it is clear which layers run
hw-agnostic and which fell back to vLLM.
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
                "hw-agnostic layer %s is not available; falling back to default", name
            )
    return getattr(importlib.import_module(f"{_VLLM_PKG}.{module}"), name)


RMSNorm = _resolve("layernorm", "RMSNorm")
GemmaRMSNorm = _resolve("layernorm", "GemmaRMSNorm")


def get_vocab_parallel_embedding_cls():
    """`VocabParallelEmbedding` class, preferring hw-agnostic. Resolved per call."""
    return _resolve("vocab_parallel_embedding", "VocabParallelEmbedding")


def get_parallel_lm_head_cls():
    """`ParallelLMHead` class, preferring hw-agnostic. Resolved per call."""
    return _resolve("vocab_parallel_embedding", "ParallelLMHead")


def get_logits_processor_cls():
    """`LogitsProcessor` class, preferring hw-agnostic. Resolved per call."""
    return _resolve("logits_processor", "LogitsProcessor")


def get_replicated_linear_cls():
    """`ReplicatedLinear` class, preferring hw-agnostic. Resolved per call."""
    return _resolve("linear", "ReplicatedLinear")


def get_column_parallel_linear_cls():
    """`ColumnParallelLinear` class, preferring hw-agnostic. Resolved per call."""
    return _resolve("linear", "ColumnParallelLinear")


def get_row_parallel_linear_cls():
    """`RowParallelLinear` class, preferring hw-agnostic. Resolved per call."""
    return _resolve("linear", "RowParallelLinear")


def get_merged_column_parallel_linear_cls():
    """`MergedColumnParallelLinear` class, preferring hw-agnostic. Resolved per call."""
    return _resolve("linear", "MergedColumnParallelLinear")


def get_qkv_parallel_linear_cls():
    """`QKVParallelLinear` class, preferring hw-agnostic. Resolved per call."""
    return _resolve("linear", "QKVParallelLinear")


def get_attention_cls():
    """`Attention` class, preferring hw-agnostic. Resolved per call."""
    return _resolve("attention", "Attention")


def get_mla_attention_cls():
    """`MLAAttention` class, preferring hw-agnostic. Resolved per call."""
    return _resolve("attention", "MLAAttention")


def get_encoder_only_attention_cls():
    """`EncoderOnlyAttention` class, preferring hw-agnostic. Resolved per call."""
    return _resolve("attention", "EncoderOnlyAttention")


def get_attention_backend_cls():
    """Portable Triton attention backend when hw-agnostic is enabled, else None.

    Unlike the layer getters, this resolves a full attention *backend* that has
    no same-named vLLM counterpart. Returning `None` when disabled (or when the
    backend cannot be imported) lets the `Attention` layer run its own
    `get_attn_backend` selector and pick the platform default.
    """
    if not envs.VLLM_USE_HW_AGNOSTIC:
        return None
    try:
        from vllm.model_executor.hw_agnostic.v1.attention.triton_backend import (
            TritonAttentionBackend,
        )

        logger.info("Using hw-agnostic attention backend: TritonAttentionBackend")
        return TritonAttentionBackend
    except ImportError:
        logger.warning(
            "hw-agnostic attention backend is not available; falling back to default"
        )
        return None


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
