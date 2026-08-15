# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Transformers backend's hw-agnostic layer resolution.

`layers._resolve` imports a layer symbol from
`vllm.model_executor.hw_agnostic.layers.<module>` when `VLLM_USE_HW_AGNOSTIC`
is set and the symbol exists, and otherwise falls back to
`vllm.model_executor.layers.<module>`. These tests pin that contract and the
logging that reports which source was used.
"""

import importlib
import logging
import sys
import types
from contextlib import contextmanager
from unittest import mock

import pytest
import torch

from vllm.model_executor.models.transformers import layers

from ...utils import multi_gpu_test

HW_MODULE = "vllm.model_executor.hw_agnostic.layers.layernorm"


@pytest.fixture
def fake_hw_layernorm(monkeypatch):
    """Inject a hw-agnostic `layernorm` module exposing a sentinel `RMSNorm`.

    A `SimpleNamespace` stands in for the module: `importlib.import_module`
    returns it from `sys.modules` and `getattr` resolves `RMSNorm`, while its
    attributes are set at construction (no `ModuleType` attribute-set that mypy
    rejects, no constant `setattr` that ruff rejects)."""
    module = types.SimpleNamespace(RMSNorm=type("HwRMSNorm", (), {}))
    monkeypatch.setitem(sys.modules, HW_MODULE, module)
    return module


def test_falls_back_to_vllm_when_disabled(monkeypatch, fake_hw_layernorm):
    """Disabled: the vLLM class is used even if a hw-agnostic one exists."""
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "0")
    from vllm.model_executor.layers.layernorm import RMSNorm as VllmRMSNorm

    assert layers._resolve("layernorm", "RMSNorm") is VllmRMSNorm


def test_uses_hw_agnostic_when_enabled(monkeypatch, fake_hw_layernorm, caplog):
    """Enabled and available: the hw-agnostic class is used and logged."""
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    with caplog.at_level(logging.INFO):
        resolved = layers._resolve("layernorm", "RMSNorm")
    assert resolved is fake_hw_layernorm.RMSNorm
    assert "Using hw-agnostic layer: RMSNorm" in caplog.text


def test_falls_back_when_symbol_missing(monkeypatch, caplog):
    """Enabled but the symbol is not ported: fall back to vLLM and warn."""
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    # A hw-agnostic module without the requested attribute triggers fallback.
    empty = types.ModuleType(HW_MODULE)
    monkeypatch.setitem(sys.modules, HW_MODULE, empty)
    from vllm.model_executor.layers.layernorm import RMSNorm as VllmRMSNorm

    with caplog.at_level(logging.WARNING):
        resolved = layers._resolve("layernorm", "RMSNorm")
    assert resolved is VllmRMSNorm
    assert "falling back to default" in caplog.text


def test_act_and_mul_falls_back_for_unknown_activation(
    monkeypatch, default_vllm_config
):
    """An activation with no hw-agnostic equivalent falls back to vLLM's.

    `default_vllm_config` supplies the config context the CustomOp needs.
    """
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    from vllm.model_executor.layers.activation import GeluAndMul

    assert isinstance(layers.get_act_and_mul_fn("gelu"), GeluAndMul)


# Each getter and the module/class name it resolves between the two trees.
_CLASS_GETTERS = (
    (
        "get_vocab_parallel_embedding_cls",
        "vocab_parallel_embedding",
        "VocabParallelEmbedding",
    ),
    ("get_parallel_lm_head_cls", "vocab_parallel_embedding", "ParallelLMHead"),
    ("get_logits_processor_cls", "logits_processor", "LogitsProcessor"),
    ("get_replicated_linear_cls", "linear", "ReplicatedLinear"),
    ("get_column_parallel_linear_cls", "linear", "ColumnParallelLinear"),
    ("get_row_parallel_linear_cls", "linear", "RowParallelLinear"),
    ("get_merged_column_parallel_linear_cls", "linear", "MergedColumnParallelLinear"),
    ("get_qkv_parallel_linear_cls", "linear", "QKVParallelLinear"),
)


@pytest.mark.parametrize("getter,module,name", _CLASS_GETTERS)
def test_class_getter_falls_back_when_disabled(monkeypatch, getter, module, name):
    """Disabled: each getter returns the vLLM class."""
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "0")
    vllm_cls = getattr(
        importlib.import_module(f"vllm.model_executor.layers.{module}"), name
    )
    assert getattr(layers, getter)() is vllm_cls


@pytest.mark.parametrize("getter,module,name", _CLASS_GETTERS)
def test_class_getter_uses_hw_agnostic_when_enabled(
    monkeypatch, caplog, getter, module, name
):
    """Enabled: each getter returns the hw-agnostic class and logs it."""
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    hw_cls = getattr(
        importlib.import_module(f"vllm.model_executor.hw_agnostic.layers.{module}"),
        name,
    )
    with caplog.at_level(logging.INFO):
        resolved = getattr(layers, getter)()
    assert resolved is hw_cls
    assert f"Using hw-agnostic layer: {name}" in caplog.text


@pytest.mark.parametrize("getter,module,name", _CLASS_GETTERS)
def test_class_getter_falls_back_when_symbol_missing(
    monkeypatch, caplog, getter, module, name
):
    """Enabled but the symbol is not ported: fall back to vLLM and warn."""
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    hw_module = f"vllm.model_executor.hw_agnostic.layers.{module}"
    monkeypatch.setitem(sys.modules, hw_module, types.ModuleType(hw_module))
    vllm_cls = getattr(
        importlib.import_module(f"vllm.model_executor.layers.{module}"), name
    )
    with caplog.at_level(logging.WARNING):
        resolved = getattr(layers, getter)()
    assert resolved is vllm_cls
    assert "falling back to default" in caplog.text


def _save_tiny_llama(tmp_path_factory, name: str, *, tie_word_embeddings: bool) -> str:
    """A randomly-initialized microscopic Llama saved to disk (with an ungated
    tokenizer) so vLLM can load it like any local checkpoint."""
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=tie_word_embeddings,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)

    path = tmp_path_factory.mktemp(name)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def tiny_llama_path(tmp_path_factory):
    """A tiny Llama with an untied `lm_head`."""
    return _save_tiny_llama(tmp_path_factory, "tiny_llama", tie_word_embeddings=False)


@pytest.fixture(scope="module")
def tiny_llama_tied_path(tmp_path_factory):
    """A tiny Llama whose `lm_head` is tied to the input embedding."""
    return _save_tiny_llama(
        tmp_path_factory, "tiny_llama_tied", tie_word_embeddings=True
    )


# Registered names of the layers the backend can
# currently route to hw-agnostic implementations.
_COVERED_LAYERS = (
    "rms_norm",
    "silu_and_mul",
    "vocab_parallel_embedding",
    "parallel_lm_head",
    "logits_processor",
    # A fused dense Llama has only column- and row-parallel linears: fused
    # qkv/gate_up report `column_parallel_linear` (their registered ancestor),
    # and o_proj/down_proj are rowwise. No bare `ReplicatedLinear` survives.
    "column_parallel_linear",
    "row_parallel_linear",
)


def _layer_providers(model) -> dict[str, str]:
    """Map each covered layer type present in the model to the provider its
    implementation came from (``hw_agnostic`` or ``vllm``).
    """

    def provider_of(module) -> str | None:
        for cls in type(module).__mro__:
            if "hw_agnostic.layers" in cls.__module__:
                return "hw_agnostic"
            if ".model_executor.layers." in cls.__module__:
                return "vllm"
        return None

    providers: dict[str, str] = {}
    for module in model.modules():
        name = getattr(module, "name", None)
        if (
            name in _COVERED_LAYERS
            and name not in providers
            and (prov := provider_of(module)) is not None
        ):
            providers[name] = prov
    return providers


def _serve(vllm_runner, model_path, prompts, tensor_parallel_size=1):
    """Serve the model through the backend; return (layer_providers, logprobs)."""
    with vllm_runner(
        model_path,
        model_impl="transformers",
        max_model_len=64,
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        tensor_parallel_size=tensor_parallel_size,
    ) as runner:
        assert runner.llm.llm_engine.model_config.using_transformers_backend()
        providers = runner.apply_model(_layer_providers)[0]
        outputs = runner.generate_greedy_logprobs(
            prompts, max_tokens=32, num_logprobs=5
        )
        return providers, outputs


def test_hw_agnostic_matches_vllm_end_to_end(monkeypatch, vllm_runner, tiny_llama_path):
    """Serving the tiny model with hw-agnostic layers matches the vLLM baseline."""
    # spawn: worker re-imports layers with the env set (see docstring).
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # apply_model pickles the introspection function.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    from ..utils import check_logprobs_close

    prompts = ["The capital of France is", "vLLM is"]

    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "0")
    vllm_providers, vllm_outputs = _serve(vllm_runner, tiny_llama_path, prompts)
    # Every replaceable layer present in the model must be vLLM's here.
    assert vllm_providers == dict.fromkeys(_COVERED_LAYERS, "vllm")

    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    hw_providers, hw_outputs = _serve(vllm_runner, tiny_llama_path, prompts)
    assert hw_providers == dict.fromkeys(_COVERED_LAYERS, "hw_agnostic")

    check_logprobs_close(
        outputs_0_lst=vllm_outputs,
        outputs_1_lst=hw_outputs,
        name_0="vllm",
        name_1="hw_agnostic",
    )


def test_hw_agnostic_matches_vllm_with_tied_lm_head(
    monkeypatch, vllm_runner, tiny_llama_tied_path
):
    """Tied `lm_head`: the hw-agnostic embedding and head still match vLLM.

    Exercises `ParallelLMHead.tie_weights` across the hw-agnostic classes and the
    `isinstance` check that decides whether to tie; a class mismatch there would
    silently drop the tie, so this guards it end to end.
    """
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    from ..utils import check_logprobs_close

    prompts = ["The capital of France is", "vLLM is"]

    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "0")
    _, vllm_outputs = _serve(vllm_runner, tiny_llama_tied_path, prompts)

    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    hw_providers, hw_outputs = _serve(vllm_runner, tiny_llama_tied_path, prompts)
    assert hw_providers == dict.fromkeys(_COVERED_LAYERS, "hw_agnostic")

    check_logprobs_close(
        outputs_0_lst=vllm_outputs,
        outputs_1_lst=hw_outputs,
        name_0="vllm",
        name_1="hw_agnostic",
    )


@multi_gpu_test(num_gpus=2)
def test_hw_agnostic_matches_vllm_tp(monkeypatch, vllm_runner, tiny_llama_path):
    """With TP=2 the hw-agnostic linears still match vLLM: guards the
    column/row-parallel weight loaders and sharding of the ported classes."""
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    from ..utils import check_logprobs_close

    prompts = ["The capital of France is", "vLLM is"]

    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "0")
    _, vllm_outputs = _serve(
        vllm_runner, tiny_llama_path, prompts, tensor_parallel_size=2
    )

    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    hw_providers, hw_outputs = _serve(
        vllm_runner, tiny_llama_path, prompts, tensor_parallel_size=2
    )
    assert hw_providers == dict.fromkeys(_COVERED_LAYERS, "hw_agnostic")

    check_logprobs_close(
        outputs_0_lst=vllm_outputs,
        outputs_1_lst=hw_outputs,
        name_0="vllm",
        name_1="hw_agnostic",
    )


def test_fp8_subtree_imports():
    """The hw-agnostic FP8 linear closure imports cleanly (guards the subtree)."""
    import vllm.model_executor.hw_agnostic.quantization.fp8_linear_method  # noqa: F401
    from vllm.model_executor.hw_agnostic.kernels.linear import (
        init_fp8_linear_kernel,  # noqa: F401
    )


def _fp8_config(weight_shape=(256, 256)):
    """A block-scaled FP8 kernel config (per-group activation scales)."""
    from vllm.model_executor.hw_agnostic.kernels.linear import (
        FP8ScaledMMLinearLayerConfig,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        GroupShape,
        create_fp8_quant_key,
    )

    return FP8ScaledMMLinearLayerConfig(
        weight_quant_key=create_fp8_quant_key(True, GroupShape(128, 128)),
        activation_quant_key=create_fp8_quant_key(False, GroupShape(1, 128)),
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        weight_shape=weight_shape,
    )


@contextmanager
def _as_cuda():
    """Present as a CUDA platform so kernel selection runs without a GPU."""
    from vllm.platforms import PlatformEnum, current_platform

    with (
        mock.patch.object(current_platform, "_enum", PlatformEnum.CUDA),
        mock.patch.object(current_platform, "is_cuda_alike", return_value=True),
        mock.patch.object(current_platform, "is_xpu", return_value=False),
    ):
        yield


def test_fp8_selector_defaults_to_triton_on_cuda(monkeypatch):
    """`auto` backend selects the Triton block-scaled kernel on CUDA."""
    monkeypatch.setattr("vllm.envs.VLLM_DISABLED_KERNELS", [])
    from vllm.model_executor.hw_agnostic.kernels.linear import (
        _POSSIBLE_FP8_BLOCK_KERNELS,
        TritonFp8BlockScaledMMKernel,
        choose_scaled_mm_linear_kernel,
    )

    with _as_cuda():
        chosen = choose_scaled_mm_linear_kernel(
            _fp8_config(), _POSSIBLE_FP8_BLOCK_KERNELS, compute_capability=90
        )
    assert chosen is TritonFp8BlockScaledMMKernel


def test_fp8_selector_falls_back_to_torch_when_triton_disabled(monkeypatch):
    """Disabling the Triton kernel falls through to the portable torch kernel."""
    from vllm.model_executor.hw_agnostic.kernels.linear import (
        _POSSIBLE_FP8_BLOCK_KERNELS,
        ChannelWiseTorchFP8ScaledMMLinearKernel,
        choose_scaled_mm_linear_kernel,
    )

    monkeypatch.setattr(
        "vllm.envs.VLLM_DISABLED_KERNELS", ["TritonFp8BlockScaledMMKernel"]
    )
    with _as_cuda():
        chosen = choose_scaled_mm_linear_kernel(
            _fp8_config(), _POSSIBLE_FP8_BLOCK_KERNELS, compute_capability=90
        )
    assert chosen is ChannelWiseTorchFP8ScaledMMLinearKernel


def test_fp8_init_rejects_non_block_scaled():
    """Per-tensor FP8 is unsupported: block-scaled (per-group) only."""
    from vllm.model_executor.hw_agnostic.kernels.linear import init_fp8_linear_kernel
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        GroupShape,
        create_fp8_quant_key,
    )

    per_tensor = create_fp8_quant_key(True, GroupShape.PER_TENSOR)
    with pytest.raises(NotImplementedError):
        init_fp8_linear_kernel(
            activation_quant_key=per_tensor,
            weight_quant_key=per_tensor,
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
            weight_shape=(256, 256),
        )
