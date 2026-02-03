# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure PyTorch implementation of PagedAttention.

This backend uses only PyTorch native operations (matmul, softmax, etc.)
to implement attention with paged KV cache support.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec


@dataclass
class PyTorchNativeAttentionMetadata:
    """Metadata for PyTorch native attention computation."""

    # Batch information
    num_actual_tokens: int
    num_seqs: int
    max_query_len: int
    max_seq_len: int

    # Sequence lengths
    seq_lens: torch.Tensor  # [num_seqs]
    query_start_loc: torch.Tensor  # [num_seqs + 1]

    # Block table for paged KV cache
    block_table: torch.Tensor  # [num_seqs, max_num_blocks_per_seq]
    block_size: int

    # Slot mapping for KV cache updates
    slot_mapping: torch.Tensor  # [num_actual_tokens]

    # Attention mask (optional)
    causal_mask: torch.Tensor | None = None

    # For grouped-query attention
    num_kv_heads: int = 0
    num_heads: int = 0


class PyTorchNativeAttentionMetadataBuilder(
    AttentionMetadataBuilder[PyTorchNativeAttentionMetadata]
):
    """Builds attention metadata from batch information."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size

        model_config = vllm_config.model_config
        self.num_heads = model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.num_kv_heads = model_config.get_num_kv_heads(vllm_config.parallel_config)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> PyTorchNativeAttentionMetadata:
        """Build attention metadata from common metadata."""

        # Extract information from common metadata
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        num_seqs = common_attn_metadata.num_reqs
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len

        seq_lens = common_attn_metadata.seq_lens
        query_start_loc = common_attn_metadata.query_start_loc
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        # Create causal mask if needed
        causal_mask = None
        if common_attn_metadata.causal and max_query_len > 1:
            causal_mask = self._create_causal_mask(
                max_query_len, max_seq_len, self.device
            )

        return PyTorchNativeAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            num_seqs=num_seqs,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            block_table=block_table,
            block_size=self.block_size,
            slot_mapping=slot_mapping,
            causal_mask=causal_mask,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
        )

    def _create_causal_mask(
        self,
        query_len: int,
        kv_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create causal attention mask."""
        # Create indices
        query_idx = torch.arange(query_len, device=device).unsqueeze(1)
        kv_idx = torch.arange(kv_len, device=device).unsqueeze(0)

        # Causal mask: query position can only attend to kv positions <= query position
        mask = query_idx < kv_idx

        return mask


class PyTorchNativeAttentionBackend(AttentionBackend):
    """Pure PyTorch implementation of PagedAttention."""

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Support any block size (no kernel-specific constraints)
        return [MultipleOf(1)]

    @staticmethod
    def get_name() -> str:
        return "PYTORCH_NATIVE"

    @staticmethod
    def get_impl_cls() -> type["PyTorchNativeAttentionImpl"]:
        return PyTorchNativeAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["PyTorchNativeAttentionMetadataBuilder"]:
        return PyTorchNativeAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """KV cache shape: [2, num_blocks, block_size, num_kv_heads, head_size]"""
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # Support any head size
        return True

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        return kv_cache_dtype in cls.supported_kv_cache_dtypes


class PyTorchNativeAttentionImpl(AttentionImpl[PyTorchNativeAttentionMetadata]):
    """PyTorch native implementation of attention with paged KV cache."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type

        # Simplified implementation: don't support these features initially
        if alibi_slopes is not None:
            raise NotImplementedError("ALiBi slopes not supported yet")
        if sliding_window is not None:
            raise NotImplementedError("Sliding window not supported yet")
        if logits_soft_cap is not None:
            raise NotImplementedError("Logits soft cap not supported yet")

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,  # [num_tokens, num_heads, head_size]
        key: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        kv_cache: torch.Tensor,  # [2, num_blocks, block_size, num_kv_heads, head_size]
        attn_metadata: PyTorchNativeAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attention output using PyTorch native operations."""

        assert output is not None, "Output tensor must be provided"

        if attn_metadata is None:
            # Profiling run
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Step 1: Update KV cache
        self._write_to_kv_cache(
            key[:num_actual_tokens],
            value[:num_actual_tokens],
            kv_cache,
            attn_metadata.slot_mapping,
            attn_metadata.block_size,
        )

        # Step 2: Gather keys and values from blocks
        key_cache = kv_cache[0]
        value_cache = kv_cache[1]

        gathered_keys = self._gather_from_kv_cache(
            key_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            attn_metadata.block_size,
        )

        gathered_values = self._gather_from_kv_cache(
            value_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            attn_metadata.block_size,
        )

        # Step 3: Prepare query tensor
        query_per_seq = self._reshape_query_to_sequences(
            query[:num_actual_tokens],
            attn_metadata.query_start_loc,
            attn_metadata.num_seqs,
            attn_metadata.max_query_len,
        )

        # Step 4: Compute attention
        attn_output = self._compute_attention(
            query_per_seq,
            gathered_keys,
            gathered_values,
            attn_metadata,
        )
        
        # import os
        # if os.environ["DEBUG"] == "STOP":
        #     print('I am here')

        # # Step 5: Reshape output back
        attn_output2 = self._reshape_output_from_sequences(
            attn_output,
            attn_metadata.query_start_loc,
            num_actual_tokens,
        )

        # Copy to output tensor
        output[:num_actual_tokens].copy_(attn_output2)

        import os
        if "DEBUG" in os.environ and os.environ["DEBUG"] == "STOP":
            print('I am here')

        return output

    def _write_to_kv_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
    ) -> None:
        """Write keys and values to paged KV cache."""

        num_tokens = key.shape[0]

        # Convert slot indices to block indices and offsets
        block_indices = slot_mapping // block_size
        block_offsets = slot_mapping % block_size

        # Get key and value caches
        key_cache = kv_cache[0]
        value_cache = kv_cache[1]

        # Write keys and values using advanced indexing
        for i in range(num_tokens):
            block_idx = block_indices[i].item()
            offset = block_offsets[i].item()
            key_cache[block_idx, offset] = key[i]
            value_cache[block_idx, offset] = value[i]

    def _gather_from_kv_cache(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """Gather keys or values from paged KV cache."""

        num_seqs = block_table.shape[0]
        max_seq_len = seq_lens.max().item()
        num_kv_heads = cache.shape[2]
        head_size = cache.shape[3]

        # Initialize output tensor
        gathered = torch.zeros(
            num_seqs,
            max_seq_len,
            num_kv_heads,
            head_size,
            dtype=cache.dtype,
            device=cache.device,
        )

        # Gather tokens for each sequence
        for seq_idx in range(num_seqs):
            seq_len = seq_lens[seq_idx].item()

            for token_idx in range(seq_len):
                # Determine which block and offset
                block_idx_in_table = token_idx // block_size
                block_offset = token_idx % block_size

                # Get physical block index
                physical_block_idx = block_table[seq_idx, block_idx_in_table].item()

                # Gather token
                gathered[seq_idx, token_idx] = cache[physical_block_idx, block_offset]

        return gathered

    def _reshape_query_to_sequences(
        self,
        query: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_seqs: int,
        max_query_len: int,
    ) -> torch.Tensor:
        """Reshape query from flat tokens to per-sequence format."""

        num_heads = query.shape[1]
        head_size = query.shape[2]
        device = query.device

        # Initialize output
        query_per_seq = torch.zeros(
            num_seqs,
            max_query_len,
            num_heads,
            head_size,
            dtype=query.dtype,
            device=device,
        )

        # Fill in queries for each sequence
        for seq_idx in range(num_seqs):
            start = query_start_loc[seq_idx].item()
            end = query_start_loc[seq_idx + 1].item()
            seq_len = end - start

            query_per_seq[seq_idx, :seq_len] = query[start:end]

        return query_per_seq

    def _compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: PyTorchNativeAttentionMetadata,
    ) -> torch.Tensor:
        """Compute attention using PyTorch operations."""

        # Handle grouped-query attention
        if self.num_queries_per_kv > 1:
            # Repeat KV heads to match query heads
            key = key.repeat_interleave(self.num_queries_per_kv, dim=2)
            value = value.repeat_interleave(self.num_queries_per_kv, dim=2)

        # Transpose for matmul
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Compute Q @ K^T
        attn_scores = torch.matmul(query, key.transpose(-2, -1))

        # Scale
        attn_scores = attn_scores * self.scale

        # Apply causal mask if needed
        if attn_metadata.causal_mask is not None:
            mask = attn_metadata.causal_mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        # Softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, value)

        # Transpose back
        attn_output = attn_output.transpose(1, 2)

        return attn_output

    def _reshape_output_from_sequences(
        self,
        output_per_seq: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """Reshape output from per-sequence format back to flat tokens."""

        num_seqs = output_per_seq.shape[0]
        num_heads = output_per_seq.shape[2]
        head_size = output_per_seq.shape[3]
        device = output_per_seq.device

        # # Initialize flat output
        # output_flat = torch.zeros(
        #     num_tokens,
        #     num_heads * head_size,
        #     dtype=output_per_seq.dtype,
        #     device=device,
        # )
        # Initialize flat output
        output_flat = torch.zeros(
            num_tokens,
            num_heads,
            head_size,
            dtype=output_per_seq.dtype,
            device=device,
        )

        # Extract outputs for each sequence
        for seq_idx in range(num_seqs):
            start = query_start_loc[seq_idx].item()
            end = query_start_loc[seq_idx + 1].item()
            seq_len = end - start

            # Reshape and copy
            seq_output = output_per_seq[seq_idx, :seq_len]
            # seq_output = seq_output.reshape(seq_len, num_heads * head_size)
            seq_output = seq_output.reshape(seq_len, num_heads, head_size)
            output_flat[start:end] = seq_output

        return output_flat

# Made with Bob
