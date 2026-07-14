# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Out-of-tree BF16 fallback platform for DeepSeek V4 on non-FP8 hardware.

This plugin turns the in-tree FP8-only ``hw_agnostic`` DeepSeek V4 path into
a BF16 fallback on platforms that lack native FP8 compute (e.g. A100 /
sm_80). It does so purely through the seams the in-tree code exposes:

  * ``DSv4OOTBF16Platform.supports_fp8() -> False`` is the single switch the
    generic quant / cache-dtype code keys on.
  * ``PluggableLayer.register_oot`` subclasses override the attention layers'
    per-platform seam methods to read/write a BF16 KV-cache slot and run the
    BF16 cache kernels instead of the FP8 ones.
  * ``register_quantization_config`` swaps in a BF16 DeepSeek V4 quant config
    whose linear / MoE methods dequantize the FP8 checkpoint to BF16.

No BF16 code lives in the in-tree tree; installing this package
(``pip install -e .``) is the opt-in, uninstalling it is the opt-out.
"""


def dsv4_oot_bf16_platform_plugin() -> str | None:
    # Activate unconditionally: having this package installed is the opt-in.
    # OOT plugins take precedence over built-in platforms (see
    # ``vllm.platforms.resolve_current_platform_cls_qualname``).
    import os

    # Breakable cudagraph is a CUDA-only feature; force it off like the FP8
    # sibling plugin does.
    os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "0"

    return "vllm_dsv4_oot_bf16_platform.platform.DSv4OOTBF16Platform"


def register_bf16_overrides() -> None:
    """Register the BF16 PluggableLayer + quant overrides.

    Invoked once per process by the ``vllm.general_plugins`` entry point
    (``vllm.plugins.load_general_plugins``), before any model is built.
    """
    from vllm_dsv4_oot_bf16_platform import attention_overrides  # noqa: F401
    from vllm_dsv4_oot_bf16_platform import quant_overrides  # noqa: F401

    attention_overrides.register()
    quant_overrides.register()
