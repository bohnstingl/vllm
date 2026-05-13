# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

import vllm.kernels  # noqa: F401
from tests.kernels.allclose_default import get_default_rtol
from vllm import ir
from vllm.platforms import current_platform


def gelu_new_inputs(n_tokens: int, d: int, dtype: torch.dtype):
    # Pointwise activation: last dim is not split.
    x = torch.randn(n_tokens, d, dtype=dtype)
    return (x,)


gelu_new_native = ir.ops.gelu_new.impls["native"].impl_fn


@pytest.mark.skipif(
    not current_platform.is_cuda_alike()
    and not current_platform.is_xpu()
    and not current_platform.is_cpu(),
    reason="Currently kernels on CUDA, ROCm, XPU and CPU",
)
def test_gelu_new_registration():
    expected = {
        "native": True,
        # vllm_c covers both CUDA-alike and CPU for this op (see
        # csrc/cpu/torch_bindings.cpp registration of gelu_new).
        "vllm_c": current_platform.is_cuda_alike() or current_platform.is_cpu(),
        "xpu_kernels": current_platform.is_xpu(),  # if registered
    }
    actual = {
        provider: impl.supported for provider, impl in ir.ops.gelu_new.impls.items()
    }
    assert actual == expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n_tokens", [1, 8, 17])
@pytest.mark.parametrize("d", [16, 4096, 8192])
@pytest.mark.skipif(
    not current_platform.is_cuda_alike()
    and not current_platform.is_xpu()
    and not current_platform.is_cpu(),
    reason="Currently kernels on CUDA, ROCm, XPU and CPU",
)
class TestNewGELU:
    @classmethod
    def setup_class(cls):
        torch.set_default_device(current_platform.device_type)

    def test_native_semantics(self, dtype, n_tokens, d):
        (x,) = gelu_new_inputs(n_tokens, d, dtype)
        out = gelu_new_native(x)
        # Pointwise: output shape matches input shape, not halved.
        assert out.shape == x.shape
        assert out.dtype == x.dtype

        # Sanity: gelu_new is odd-symmetric around 0 up to the cubic term,
        # and gelu_new(0) == 0.
        zero = torch.zeros((), dtype=dtype)
        torch.testing.assert_close(gelu_new_native(zero), zero, atol=1e-3, rtol=1e-3)
        # Large positive input → approximately x (sigmoid-tanh saturates to 1).
        big = torch.tensor(5.0, dtype=dtype)
        torch.testing.assert_close(
            gelu_new_native(big),
            big,
            atol=1e-2 if dtype == torch.float16 else 1e-3,
            rtol=1e-2,
        )
        # Large negative input → approximately 0.
        neg_big = torch.tensor(-5.0, dtype=dtype)
        torch.testing.assert_close(
            gelu_new_native(neg_big),
            torch.zeros((), dtype=dtype),
            atol=1e-2 if dtype == torch.float16 else 1e-3,
            rtol=1e-2,
        )

    @pytest.mark.parametrize("provider", ["vllm_c", "xpu_kernels"])
    def test_impls(self, dtype, n_tokens, d, provider):
        impl = ir.ops.gelu_new.impls[provider]
        if not impl.supported:
            pytest.skip(f"{provider} not supported")
        (x,) = gelu_new_inputs(n_tokens, d, dtype)
        out_impl = impl.impl_fn(x)
        out_native = gelu_new_native(x)
        torch.testing.assert_close(
            out_impl, out_native, rtol=get_default_rtol(out_impl), atol=1e-3
        )
        with ir.ops.gelu_new.set_priority([provider, "native"]):
            out_dispatched = ir.ops.gelu_new(x)
        torch.testing.assert_close(out_dispatched, out_impl, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("compile", [False, True])
    def test_native_impl_compile(self, dtype, n_tokens, d, compile):
        impl = ir.ops.gelu_new.impls["native"]
        assert impl.supported, "native implementation must be supported!"
        (x,) = gelu_new_inputs(n_tokens, d, dtype)
        out_impl = impl.impl_fn(x)
        out_native = gelu_new_native(x)
        torch.testing.assert_close(
            out_impl, out_native, rtol=get_default_rtol(out_impl), atol=1e-3
        )
        with ir.ops.gelu_new.set_priority(["native"], compile=compile):
            out_dispatched = ir.ops.gelu_new(x)

            if compile:
                assert isinstance(
                    ir.ops.gelu_new.dispatch(x),
                    ir.op.IrOpImplCompiledWrapper,
                ), (
                    "When `set_priority` with compile=True, the implementation is "
                    "expected to be wrapped with compile."
                )
        torch.testing.assert_close(out_dispatched, out_impl, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("provider", ["vllm_c", "xpu_kernels", "native"])
    def test_torch_opcheck(self, dtype, n_tokens, d, provider):
        if not ir.ops.gelu_new.impls[provider].supported:
            pytest.skip(f"{provider} not supported")
        (x,) = gelu_new_inputs(n_tokens, d, dtype)
        with ir.ops.gelu_new.set_priority([provider, "native"]):
            torch.library.opcheck(torch.ops.vllm_ir.gelu_new, (x,))
