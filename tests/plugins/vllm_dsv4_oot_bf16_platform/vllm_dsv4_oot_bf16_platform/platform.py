# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.platforms.cuda import NvmlCudaPlatform

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class DSv4OOTBF16Platform(NvmlCudaPlatform):
    """OOT platform for the DeepSeek V4 BF16 fallback (non-FP8 hardware).

    Piggybacks on CUDA infrastructure (so the model executes on a real GPU)
    but reports ``supports_fp8() -> False`` unconditionally. That single
    switch drives every FP8-vs-BF16 decision in the in-tree code:

      * the quant methods dequantize the FP8 checkpoint to BF16,
      * the attention layers select the ``bf16_ds_mla`` KV-cache layout,
      * the BF16 cache kernels are used in place of the FP8 ones.

    Unlike the FP8-only ``DSv4OOTPlatform``, this platform does NOT fail on
    pre-Ada hardware -- running there is its whole purpose.
    """

    # Dequant FP8 MoE experts to BF16 at RUNTIME (per-forward), the
    # memory-safe default: a load-time dequant doubles the expert weights
    # (~35->69 GB/GPU on the full model) and OOMs the KV cache. The generic
    # ``Fp8MoEMethod`` reads this via ``getattr(current_platform, ...)``.
    moe_dequant_at_load: bool = False

    def is_out_of_tree(self) -> bool:
        return True

    @classmethod
    def supports_fp8(cls) -> bool:
        # Force the BF16 fallback regardless of hardware capability. This is
        # the single knob the whole fallback keys on; forcing it False even
        # on FP8-capable HW makes the fallback testable there too.
        return False

    @classmethod
    def support_deep_gemm(cls) -> bool:
        return False

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        super().check_and_update_config(vllm_config)

        if vllm_config.kernel_config.moe_backend == "auto":
            vllm_config.kernel_config.moe_backend = "triton"
