# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from setuptools import find_packages, setup

setup(
    name="vllm_dsv4_oot_bf16_platform",
    version="0.1",
    packages=find_packages(),
    entry_points={
        # Report an OOT platform so DeepSeek V4 dispatches to the
        # hw_agnostic implementation (see ``vllm.models.deepseek_v4``).
        "vllm.platform_plugins": [
            "dsv4_oot_bf16_platform_plugin = vllm_dsv4_oot_bf16_platform:dsv4_oot_bf16_platform_plugin"  # noqa: E501
        ],
        # Register the BF16 PluggableLayer / quant overrides at startup in
        # every process (driver + workers), before any model is built.
        "vllm.general_plugins": [
            "dsv4_bf16_overrides = vllm_dsv4_oot_bf16_platform:register_bf16_overrides"  # noqa: E501
        ],
    },
)
