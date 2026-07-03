# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.platforms.cuda import NvmlCudaPlatform

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class DSv4OOTPlatform(NvmlCudaPlatform):
    """Test-only OOT platform that piggybacks on CUDA infrastructure."""

    # FP8 MoE experts are dequantized to BF16 at RUNTIME (per-forward), the
    # memory-safe default: on the full DeepSeek-V4 model a load-time dequant
    # doubles the expert weights (~35->69 GB/GPU) and OOMs the KV cache. The
    # ``moe_dequant_at_load`` seam still exists for the generic Fp8MoEMethod
    # to read via getattr; tests override it to True on a small layer to
    # exercise the load-time path without OOMing a real model here.
    moe_dequant_at_load: bool = False

    def is_out_of_tree(self) -> bool:
        return True

    @classmethod
    def support_deep_gemm(cls) -> bool:
        return False

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        super().check_and_update_config(vllm_config)

        if vllm_config.kernel_config.moe_backend == "auto":
            vllm_config.kernel_config.moe_backend = "triton"
