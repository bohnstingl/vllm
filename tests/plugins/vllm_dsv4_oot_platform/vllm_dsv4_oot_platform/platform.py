# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.platforms.cuda import NvmlCudaPlatform

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class DSv4OOTPlatform(NvmlCudaPlatform):
    """Test-only OOT platform that piggybacks on CUDA infrastructure."""

    def is_out_of_tree(self) -> bool:
        return True

    @classmethod
    def support_deep_gemm(cls) -> bool:
        return False

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        super().check_and_update_config(vllm_config)

        # This plugin exercises the native-FP8 DeepSeek V4 path only. Fail
        # closed on hardware without FP8 compute (e.g. pre-Ada, cc < 8.9);
        # the BF16 fallback lives in the separate vllm_dsv4_oot_bf16_platform
        # plugin, which overrides the PluggableLayer seams.
        if not cls.supports_fp8():
            raise RuntimeError(
                "DSv4OOTPlatform requires native FP8 compute (compute "
                "capability >= 8.9). Install the vllm_dsv4_oot_bf16_platform "
                "plugin to run DeepSeek V4 on non-FP8 hardware."
            )

        if vllm_config.kernel_config.moe_backend == "auto":
            vllm_config.kernel_config.moe_backend = "triton"
