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
from typing import Any
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
    ("get_attention_cls", "attention", "Attention"),
    ("get_mla_attention_cls", "attention", "MLAAttention"),
    ("get_encoder_only_attention_cls", "attention", "EncoderOnlyAttention"),
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


@pytest.mark.parametrize(
    "getter,name",
    [
        ("get_attention_cls", "Attention"),
        ("get_mla_attention_cls", "MLAAttention"),
        ("get_encoder_only_attention_cls", "EncoderOnlyAttention"),
    ],
)
def test_attention_reexport_preserves_identity(monkeypatch, getter, name):
    vllm_cls = getattr(
        importlib.import_module("vllm.model_executor.layers.attention"), name
    )
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    assert getattr(layers, getter)() is vllm_cls
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "0")
    assert getattr(layers, getter)() is vllm_cls


# --------------------------------------------------------------------------
# Out-of-tree override reachability.
#
# The hw-agnostic layers are self-contained, so `hw.RMSNorm` and
# `layers.RMSNorm` are unrelated classes with the same name. A plugin subclasses
# whichever one `vllm.model_executor.layers.<mod>` gave it and registers the
# result under that name; `__new__` then swaps it in for the class the model
# instantiates. Get the pairing wrong and Python silently skips `__init__` (it
# only runs when `__new__` returns an instance of the class called), which is why
# `validate_registered_overrides` exists and why `hw_agnostic_layer_names()`
# decides what the plugin's import sees. The tests below pin both halves.
# --------------------------------------------------------------------------

# Every layer implemented on both sides, and the module holding it.
_HW_AGNOSTIC_LAYERS = (
    ("layernorm", "RMSNorm"),
    ("activation", "SiluAndMul"),
    ("logits_processor", "LogitsProcessor"),
    ("vocab_parallel_embedding", "VocabParallelEmbedding"),
    ("vocab_parallel_embedding", "ParallelLMHead"),
    ("linear", "ReplicatedLinear"),
    ("linear", "ColumnParallelLinear"),
    ("linear", "MergedColumnParallelLinear"),
    ("linear", "QKVParallelLinear"),
    ("linear", "RowParallelLinear"),
)


def _hw_layer(module: str, name: str) -> Any:
    """The hw-agnostic class `name` from `hw_agnostic.layers.<module>`."""
    return getattr(
        importlib.import_module(f"vllm.model_executor.hw_agnostic.layers.{module}"),
        name,
    )


def _vllm_layer(module: str, name: str) -> Any:
    """The in-tree class `name` from `model_executor.layers.<module>`."""
    return getattr(
        importlib.import_module(f"vllm.model_executor.layers.{module}"), name
    )


def test_custom_op_plumbing_is_shared():
    """One `CustomOp`, one `PluggableLayer`, one pair of registries.

    A second copy of the plumbing is a second, disjoint `op_registry_oot`: a
    plugin decorating the vLLM class writes into a registry the hw-agnostic
    `__new__` never reads, and its override is skipped without an error.
    """
    from vllm.model_executor import custom_op as vllm_plumbing
    from vllm.model_executor.hw_agnostic import custom_op as hw_plumbing

    for attr in (
        "CustomOp",
        "PluggableLayer",
        "op_registry",
        "op_registry_oot",
        "maybe_get_oot_by_class",
    ):
        assert getattr(hw_plumbing, attr) is getattr(vllm_plumbing, attr), attr


@pytest.mark.parametrize("module,name", _HW_AGNOSTIC_LAYERS)
def test_hw_agnostic_layer_is_standalone(module, name):
    """Each hw-agnostic layer is an independent implementation, not a subclass.

    The isolation is deliberate: the hw-agnostic path can be reshaped without
    touching the in-tree layers. What it costs is that the two classes are
    interchangeable only by name, which is what the rest of these tests are
    about.
    """
    hw_cls = _hw_layer(module, name)
    vllm_cls = _vllm_layer(module, name)

    assert hw_cls is not vllm_cls
    assert not issubclass(hw_cls, vllm_cls)
    assert not issubclass(vllm_cls, hw_cls)
    # No in-tree import anywhere in the hw-agnostic layer's own ancestry.
    assert not any(
        c.__module__.startswith("vllm.model_executor.layers.") for c in hw_cls.__mro__
    )


@pytest.mark.parametrize("module,name", _HW_AGNOSTIC_LAYERS)
def test_hw_agnostic_layer_keeps_the_registered_names(module, name):
    """`__name__` and `name` match the in-tree layer, without taking its entry.

    `__name__` is what `__new__` keys the out-of-tree swap on, so drift there
    unregisters every override of the layer. `name` is what identifies the op to
    `CompilationConfig.custom_ops`. The single shared `op_registry` keeps
    pointing at the in-tree class, because the hw-agnostic `register` claims the
    name without writing the entry (see
    `test_hw_agnostic_register_claims_the_name_only`).
    """
    from vllm.model_executor.custom_op import op_registry

    hw_cls = _hw_layer(module, name)
    vllm_cls = _vllm_layer(module, name)

    assert hw_cls.__name__ == vllm_cls.__name__
    # `name` may be inherited -- `MergedColumnParallelLinear` and
    # `QKVParallelLinear` both report `column_parallel_linear` -- which is fine,
    # because the swap keys on `__name__`.
    assert hw_cls.name == vllm_cls.name
    assert op_registry[vllm_cls.name] is not hw_cls
    assert issubclass(vllm_cls, op_registry[vllm_cls.name])


def test_hw_agnostic_register_claims_the_name_only():
    """`register` on the wrappers: same effect on the class, no registry write.

    This is what lets an hw-agnostic layer be written the way its in-tree
    counterpart is -- `@CustomOp.register("rms_norm")` on a class whose name the
    in-tree op already holds -- with no change in
    `vllm.model_executor.custom_op`. Writing the entry instead would make the
    owner depend on which module was imported first, and the loser would be the
    in-tree decorator, whose assert aborts the import.
    """
    from vllm.model_executor.custom_op import CustomOp, PluggableLayer, op_registry
    from vllm.model_executor.hw_agnostic.custom_op import (
        HwAgnosticCustomOp,
        HwAgnosticPluggableLayer,
    )

    before = dict(op_registry)

    # Names the in-tree ops already own; neither call raises.
    @HwAgnosticCustomOp.register("rms_norm", dynamic_arg_dims={"x": 0})
    class _Op(HwAgnosticCustomOp):
        pass

    @HwAgnosticPluggableLayer.register("replicated_linear")
    class _Layer(HwAgnosticPluggableLayer):
        pass

    # Everything `enabled()` and `dispatch_forward` read is set.
    assert _Op.name == "rms_norm"
    assert _Op._dynamic_arg_dims == {"x": 0}
    assert _Layer.name == "replicated_linear"
    # Nothing added, nothing overwritten.
    assert op_registry == before
    # The out-of-tree machinery is inherited untouched, so one `register_oot`
    # still covers both paths.
    assert HwAgnosticCustomOp.register_oot.__func__ is CustomOp.register_oot.__func__
    assert HwAgnosticCustomOp.__new__ is CustomOp.__new__
    assert (
        HwAgnosticPluggableLayer.register_oot.__func__
        is PluggableLayer.register_oot.__func__
    )
    assert HwAgnosticPluggableLayer.__new__ is PluggableLayer.__new__


# --------------------------------------------------------------------------
# Resolving the in-tree layer names to the hw-agnostic classes.
#
# `hw_agnostic_layer_names()` is what makes a single plugin work on both paths:
# inside it, the plugin's `from vllm.model_executor.layers.linear import X`
# yields the class its process will instantiate.
# --------------------------------------------------------------------------


def test_layer_names_scope_is_a_noop_when_disabled(monkeypatch):
    """Disabled: the in-tree names are untouched, so a plugin overrides vLLM."""
    from vllm.model_executor.hw_agnostic.layers._layer_names import (
        hw_agnostic_layer_names,
    )

    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "0")
    with hw_agnostic_layer_names():
        for module, name in _HW_AGNOSTIC_LAYERS:
            assert _vllm_layer(module, name) is _vllm_layer(module, name)
            assert _vllm_layer(module, name) is not _hw_layer(module, name)


def test_layer_names_scope_rebinds_and_restores(monkeypatch):
    """Enabled: every mirrored name resolves to the hw-agnostic class inside the
    block, and to the in-tree class again outside it.

    Restoring matters as much as rebinding: ~200 in-tree modules import from
    these, and a permanent swap would leave `isinstance(x, LinearBase)` answering
    differently depending on import order.
    """
    from vllm.model_executor.hw_agnostic.layers._layer_names import (
        hw_agnostic_layer_names,
    )

    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    before = {(m, n): _vllm_layer(m, n) for m, n in _HW_AGNOSTIC_LAYERS}

    with hw_agnostic_layer_names():
        for module, name in _HW_AGNOSTIC_LAYERS:
            assert _vllm_layer(module, name) is _hw_layer(module, name)

    for key, cls in before.items():
        assert _vllm_layer(*key) is cls


def test_layer_names_scope_restores_on_exception(monkeypatch):
    """A plugin that raises mid-import must not leave the namespace rebound."""
    from vllm.model_executor.hw_agnostic.layers._layer_names import (
        hw_agnostic_layer_names,
    )

    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    original = _vllm_layer("linear", "QKVParallelLinear")

    with (
        pytest.raises(RuntimeError, match="plugin blew up"),
        hw_agnostic_layer_names(),
    ):
        raise RuntimeError("plugin blew up")

    assert _vllm_layer("linear", "QKVParallelLinear") is original


def test_layer_names_scope_leaves_unported_layers_alone(monkeypatch):
    """Names with no hw-agnostic implementation keep their in-tree class.

    The same fallback the modeling side takes in `_resolve`, so a plugin's
    `GeluAndMul` override still lands on the class that gets built.
    """
    from vllm.model_executor.hw_agnostic.layers._layer_names import (
        hw_agnostic_layer_names,
    )

    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    unported = (("activation", "GeluAndMul"), ("layernorm", "GemmaRMSNorm"))
    before = {(m, n): _vllm_layer(m, n) for m, n in unported}

    with hw_agnostic_layer_names():
        for key, cls in before.items():
            assert _vllm_layer(*key) is cls


@pytest.mark.parametrize("hw_agnostic", ["0", "1"])
def test_general_plugins_open_the_scope_only_when_enabled(monkeypatch, hw_agnostic):
    """`load_general_plugins` guards the scope on `VLLM_USE_HW_AGNOSTIC`.

    With it off, the loading path must be what it is upstream: no scope opened,
    and — checked separately in `/tmp/check_plugins_guard.py`, which needs a fresh
    interpreter — no hw-agnostic module imported at all.
    """
    import vllm.plugins as plugins

    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", hw_agnostic)
    monkeypatch.setattr(plugins, "plugins_loaded", False)

    seen: dict[tuple[str, str], Any] = {}
    monkeypatch.setattr(
        plugins,
        "_run_general_plugins",
        lambda: seen.update(
            {(m, n): _vllm_layer(m, n) for m, n in _HW_AGNOSTIC_LAYERS}
        ),
    )
    plugins.load_general_plugins()

    assert seen, "the plugin-loading body never ran"
    for module, name in _HW_AGNOSTIC_LAYERS:
        expected = (
            _hw_layer(module, name) if hw_agnostic == "1" else _vllm_layer(module, name)
        )
        assert seen[(module, name)] is expected


def test_layer_names_scope_covers_the_quant_method(monkeypatch):
    """The linear quant methods are rebound too.

    Plugins gate on `isinstance(self.quant_method, UnquantizedLinearMethod)`, and
    the hw-agnostic layers install their own `UnquantizedLinearMethod`. Without
    the rebinding that check silently answers False on the hw-agnostic path and
    the plugin's fast GEMM never runs.
    """
    import vllm.model_executor.layers.linear as vllm_linear
    from vllm.model_executor.hw_agnostic.layers._layer_names import (
        hw_agnostic_layer_names,
    )
    from vllm.model_executor.hw_agnostic.layers.linear import (
        LinearMethodBase as HwLinearMethodBase,
    )
    from vllm.model_executor.hw_agnostic.layers.linear import (
        UnquantizedLinearMethod as HwUnquantizedLinearMethod,
    )

    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    with hw_agnostic_layer_names():
        assert vllm_linear.UnquantizedLinearMethod is HwUnquantizedLinearMethod
        assert vllm_linear.LinearMethodBase is HwLinearMethodBase


@pytest.mark.parametrize("module,name", _HW_AGNOSTIC_LAYERS)
@pytest.mark.parametrize("hw_agnostic", ["0", "1"])
def test_plugin_import_pattern_reaches_the_entry_point(
    monkeypatch, module, name, hw_agnostic
):
    """The end-to-end contract, both ways round.

    A plugin does exactly one thing: subclass the name it imported from
    `vllm.model_executor.layers.<mod>` while `hw_agnostic_layer_names()` is
    active, and register that under the class name. This asserts the result is
    swapped in for the class the modeling code on that path instantiates, with
    `__init__` reachable -- which is the whole reason the plugin needs no
    `VLLM_USE_HW_AGNOSTIC`-dependent imports of its own.
    """
    from vllm.model_executor.custom_op import op_registry_oot
    from vllm.model_executor.hw_agnostic.layers._layer_names import (
        hw_agnostic_layer_names,
    )

    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", hw_agnostic)

    with hw_agnostic_layer_names():
        imported = _vllm_layer(module, name)  # what the plugin's import yields
        plugin_cls = type(f"Plugin{name}", (imported,), {})

    monkeypatch.setitem(op_registry_oot, name, plugin_cls)

    # What the modeling code instantiates on this path.
    entry_cls = _hw_layer(module, name) if hw_agnostic == "1" else imported
    assert plugin_cls.__bases__ == (entry_cls,)

    obj = entry_cls.__new__(entry_cls)
    assert type(obj) is plugin_cls
    # `type.__call__` runs `__init__` only because this holds.
    assert isinstance(obj, entry_cls)


@pytest.mark.parametrize("module,name", _HW_AGNOSTIC_LAYERS)
def test_override_registered_against_wrong_base_raises(monkeypatch, module, name):
    """A plugin that captured the other path's class is reported, not absorbed.

    This is what happens without `hw_agnostic_layer_names()`: the override is a
    sibling of the class being built, `type.__call__` skips `__init__`, and the
    first symptom is an unrelated `AttributeError` on `_backward_hooks` far from
    the registration that caused it. `__new__` cannot see the mismatch without a
    check of its own, so `validate_registered_overrides` sweeps the registry once
    after plugin loading and names both classes instead.
    """
    from vllm.model_executor.custom_op import op_registry_oot
    from vllm.model_executor.hw_agnostic.custom_op import validate_registered_overrides

    hw_cls = _hw_layer(module, name)
    vllm_cls = _vllm_layer(module, name)
    # Captured the in-tree class, but the hw-agnostic one gets instantiated.
    plugin_cls = type(f"Plugin{name}", (vllm_cls,), {})
    monkeypatch.setitem(op_registry_oot, name, plugin_cls)

    # Correct on the in-tree path, and silently broken on the hw-agnostic one:
    # `__new__` hands back the override either way, but only in the first case is
    # it an instance of the class that was called, so only there runs `__init__`.
    assert type(vllm_cls.__new__(vllm_cls)) is plugin_cls
    assert isinstance(vllm_cls.__new__(vllm_cls), vllm_cls)
    assert not isinstance(hw_cls.__new__(hw_cls), hw_cls)

    mirrored = {
        f"vllm.model_executor.layers.{module}": (
            f"vllm.model_executor.hw_agnostic.layers.{module}"
        )
    }
    with pytest.raises(TypeError, match=f"Plugin{name}.*not a subclass"):
        validate_registered_overrides(mirrored)


def test_correctly_based_override_passes_validation(monkeypatch):
    """The sweep only rejects mismatches.

    Every layer with no hw-agnostic implementation is registered against its
    in-tree class on both paths, so validating the whole published set must leave
    those alone -- otherwise the guard would reject the fallback it is supposed to
    allow.
    """
    from vllm.model_executor.custom_op import op_registry_oot
    from vllm.model_executor.hw_agnostic.custom_op import validate_registered_overrides
    from vllm.model_executor.hw_agnostic.layers._layer_names import _MIRRORED_MODULES

    # What a plugin loaded under `hw_agnostic_layer_names()` registers.
    monkeypatch.setitem(
        op_registry_oot,
        "RMSNorm",
        type("PluginRMSNorm", (_hw_layer("layernorm", "RMSNorm"),), {}),
    )
    # An unported layer: the in-tree class is what gets instantiated on both
    # paths, so an override based on it is right and must not be flagged.
    monkeypatch.setitem(
        op_registry_oot,
        "GeluAndMul",
        type("PluginGeluAndMul", (_vllm_layer("activation", "GeluAndMul"),), {}),
    )

    validate_registered_overrides(_MIRRORED_MODULES)


def test_hw_agnostic_ops_skip_vendor_forwards(monkeypatch):
    """Dispatch stays on the portable implementation, and an out-of-tree subclass
    inherits that discipline.

    `CustomOp.forward_{hip,cpu,tpu,xpu,oot}` already delegate to
    `forward_native`, so `forward_cuda` -- the one branch that raises -- is all
    `HwAgnosticCustomOp` has to redirect. Expressing it that way keeps the
    discipline out of `vllm.model_executor.custom_op` entirely.
    """
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.custom_op import CustomOp, op_registry_oot
    from vllm.model_executor.hw_agnostic.custom_op import HwAgnosticCustomOp
    from vllm.model_executor.hw_agnostic.layers.layernorm import RMSNorm as HwRMSNorm
    from vllm.model_executor.layers.layernorm import RMSNorm as VllmRMSNorm
    from vllm.platforms import current_platform

    assert issubclass(HwRMSNorm, HwAgnosticCustomOp)
    assert not issubclass(VllmRMSNorm, HwAgnosticCustomOp)
    # Inherited, so a plugin's override keeps it.
    plugin_rms: Any = type("PluginRMSNorm", (HwRMSNorm,), {})
    assert issubclass(plugin_rms, HwAgnosticCustomOp)

    config = VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))
    # Register the hw-agnostic class as its own override, so this test builds the
    # class under test whether or not an out-of-tree plugin is loaded in this
    # process (on Spyre, "RMSNorm" resolves to `SpyreRMSNorm`, which is based on
    # the in-tree class and would come back with its `__init__` skipped).
    monkeypatch.setitem(op_registry_oot, "RMSNorm", HwRMSNorm)
    with set_current_vllm_config(config):
        op = HwRMSNorm(8)
        x = torch.randn(2, 8)
        expected = op.forward_native(x)
        # The base class raises here; every vendor branch now lands on the
        # portable implementation, `forward_hip` via `forward_cuda`.
        with pytest.raises(NotImplementedError):
            CustomOp.forward_cuda(op, x)
        for vendor in ("cuda", "hip", "cpu", "tpu", "xpu", "oot"):
            torch.testing.assert_close(getattr(op, f"forward_{vendor}")(x), expected)
        # On an out-of-tree platform the chain is unchanged: `forward_oot`, which
        # is where a plugin's override hooks in.
        monkeypatch.setattr(current_platform, "is_out_of_tree", lambda: True)
        assert HwRMSNorm(8)._forward_method.__name__ == "forward_oot"


def test_hw_agnostic_rms_norm_matches_vllm_native():
    """The hw-agnostic `forward_native` computes what vLLM's does; it differs
    only in bypassing the `vllm.ir` op registry, which can resolve to a vendor
    kernel."""
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.hw_agnostic.layers.layernorm import RMSNorm as HwRMSNorm
    from vllm.model_executor.layers.layernorm import RMSNorm as VllmRMSNorm

    config = VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))
    with set_current_vllm_config(config):
        vllm_norm, hw_norm = VllmRMSNorm(64), HwRMSNorm(64)
    hw_norm.load_state_dict(vllm_norm.state_dict())

    x = torch.randn(4, 64)
    torch.testing.assert_close(vllm_norm.forward_native(x), hw_norm.forward_native(x))

    residual = torch.randn(4, 64)
    expected = vllm_norm.forward_native(x.clone(), residual.clone())
    actual = hw_norm.forward_native(x.clone(), residual.clone())
    for want, got in zip(expected, actual):
        torch.testing.assert_close(want, got)


def test_vllm_attention_forward_applies_module_scaling():
    from types import SimpleNamespace

    from vllm.model_executor.models.transformers import vllm_attention_forward

    impl = SimpleNamespace(scale=4**-0.5)
    self_attn = SimpleNamespace(
        impl=impl, forward=lambda q, k, v: torch.zeros(q.shape[0], q.shape[1])
    )
    module = SimpleNamespace(layer_idx=0)
    qkv = torch.zeros(1, 2, 3, 4)  # [batch, heads, tokens, head_dim]

    vllm_attention_forward(
        module,
        qkv,
        qkv,
        qkv,
        attention_mask=None,
        scaling=0.123,
        attention_instances={0: self_attn},
    )
    assert impl.scale == pytest.approx(0.123)


def test_attention_backend_getter_disabled_returns_none(monkeypatch):
    """Disabled: no hw backend is forced; the layer picks the platform default."""
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "0")
    assert layers.get_attention_backend_cls() is None


def test_attention_backend_getter_enabled_returns_triton(monkeypatch, caplog):
    """Enabled: the getter returns the portable hw-agnostic Triton backend.

    Only where Triton is usable. The backend module imports cleanly without
    Triton — `TritonPlaceholder` makes `@triton.jit` a passthrough — so the
    getter checks `HAS_TRITON` up front rather than letting the first kernel
    launch fail; where it is unusable, the platform default is the right answer.
    """
    from vllm.triton_utils import HAS_TRITON

    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    from vllm.model_executor.hw_agnostic.v1.attention.triton_backend import (
        TritonAttentionBackend,
    )

    with caplog.at_level(logging.INFO):
        resolved = layers.get_attention_backend_cls()
    if HAS_TRITON:
        assert resolved is TritonAttentionBackend
    else:
        assert resolved is None
    assert "hw-agnostic attention backend" in caplog.text


def test_hw_agnostic_triton_backend_is_portable():
    """The vendored backend imports without vendor deps and keeps the enum-valid
    `TRITON_ATTN` name (required by `AttentionBackendEnum[...]`), while living in
    the hw-agnostic tree so the seam is distinguishable from the in-tree one."""
    from vllm.model_executor.hw_agnostic.v1.attention import triton_backend as hw

    assert hw.TritonAttentionBackend.get_name() == "TRITON_ATTN"
    assert hw.TritonAttentionBackend.get_impl_cls() is hw.TritonAttentionImpl
    # The ROCm aiter import is stripped, so the module never binds the symbol,
    # and the vendor fused-rope override is dropped to the base default.
    assert not hasattr(hw, "rocm_aiter_ops")
    assert "do_rope_and_kv_cache_update" not in vars(hw.TritonAttentionImpl)


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
