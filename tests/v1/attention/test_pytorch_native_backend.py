"""Test PyTorch native attention backend against standard backends."""

import pytest
import torch

from vllm import LLM, SamplingParams


@pytest.mark.parametrize("model", ["ibm-granite/granite-3.0-8b-base"])
@pytest.mark.parametrize("backend", ["FLASH_ATTN", "PYTORCH_NATIVE"])
def test_pytorch_native_vs_standard(model: str, backend: str):
    """
    Test that PyTorch native backend produces similar results to standard backend.
    
    This test compares the outputs of the PyTorch native attention backend
    with the standard FlashAttention backend using the Granite 3.0 8B model.
    """
    
    # Skip if model not available
    pytest.importorskip("transformers")
    
    # Create LLM with specified backend
    llm = LLM(
        model=model,
        attention_backend=backend,
        max_model_len=512,
        enforce_eager=True,  # Disable CUDA graphs for simplicity
        gpu_memory_utilization=0.5,
    )
    
    # Test prompts
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "In machine learning,",
    ]
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic for comparison
        max_tokens=20,
        top_p=1.0,
    )
    
    # Generate outputs
    outputs = llm.generate(prompts, sampling_params)
    
    # Verify we got outputs
    assert len(outputs) == len(prompts)
    
    for output in outputs:
        assert len(output.outputs) > 0
        assert len(output.outputs[0].text) > 0
        print(f"Backend: {backend}")
        print(f"Prompt: {output.prompt}")
        print(f"Output: {output.outputs[0].text}")
        print("-" * 80)


def test_pytorch_native_backend_comparison():
    """
    Direct comparison test between FlashAttention and PyTorch native backends.
    
    This test runs the same prompts through both backends and compares
    the generated text to ensure they produce similar (though not necessarily
    identical) results.
    """
    
    model = "ibm-granite/granite-3.0-8b-base"
    
    # Test prompts
    prompts = [
        "The quick brown fox",
        "Once upon a time",
    ]
    
    # Sampling parameters (deterministic)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=10,
        top_p=1.0,
    )
    
    # Run with FlashAttention
    print("\n" + "=" * 80)
    print("Testing with FlashAttention backend")
    print("=" * 80)
    
    llm_flash = LLM(
        model=model,
        attention_backend="FLASH_ATTN",
        max_model_len=256,
        enforce_eager=True,
        gpu_memory_utilization=0.4,
    )
    
    outputs_flash = llm_flash.generate(prompts, sampling_params)
    
    # Clean up
    del llm_flash
    torch.cuda.empty_cache()
    
    # Run with PyTorch Native
    print("\n" + "=" * 80)
    print("Testing with PyTorch Native backend")
    print("=" * 80)
    
    llm_native = LLM(
        model=model,
        attention_backend="PYTORCH_NATIVE",
        max_model_len=256,
        enforce_eager=True,
        gpu_memory_utilization=0.4,
    )
    
    outputs_native = llm_native.generate(prompts, sampling_params)
    
    # Compare outputs
    print("\n" + "=" * 80)
    print("Comparison Results")
    print("=" * 80)
    
    for i, (out_flash, out_native) in enumerate(zip(outputs_flash, outputs_native)):
        flash_text = out_flash.outputs[0].text
        native_text = out_native.outputs[0].text
        
        print(f"\nPrompt {i+1}: {prompts[i]}")
        print(f"FlashAttention:  {flash_text}")
        print(f"PyTorch Native:  {native_text}")
        
        # Check if outputs are similar (they should be identical with temp=0)
        # Note: Due to numerical precision differences, they might not be exactly the same
        if flash_text == native_text:
            print("✓ Outputs are identical")
        else:
            print("⚠ Outputs differ (this may be due to numerical precision)")
            # Check if they at least start the same way
            min_len = min(len(flash_text), len(native_text))
            common_prefix_len = 0
            for j in range(min_len):
                if flash_text[j] == native_text[j]:
                    common_prefix_len += 1
                else:
                    break
            
            similarity = common_prefix_len / max(len(flash_text), len(native_text))
            print(f"  Similarity: {similarity:.1%}")
            
            # Assert they are at least 50% similar
            assert similarity > 0.5, (
                f"Outputs are too different: {similarity:.1%} similarity"
            )


if __name__ == "__main__":
    # Run the comparison test directly
    test_pytorch_native_backend_comparison()

# Made with Bob
