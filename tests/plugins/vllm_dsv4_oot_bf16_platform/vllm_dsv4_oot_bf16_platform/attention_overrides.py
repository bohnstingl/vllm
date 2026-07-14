# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 PluggableLayer overrides for the DeepSeek V4 attention stack.

Each class below subclasses an in-tree ``PluggableLayer`` and overrides only
the per-platform seam methods, swapping the FP8 cache layout / kernels for
the BF16 fallback. Registration via ``PluggableLayer.register_oot`` makes
``__new__`` instantiate these subclasses in place of the in-tree classes
whenever the model is built under this platform.

The FP8 branch of every kernel is dead-code-eliminated at Triton compile
time because the plugin invokes them with ``use_fp8=False``.
"""

import torch

from vllm.model_executor.hw_agnostic.custom_op import PluggableLayer
from vllm.models.deepseek_v4.hw_agnostic.attention.attention import (
    DeepseekV4Indexer,
    DeepseekV4MLAAttention,
    DeepseekV4MultiHeadLatentAttentionWrapper,
)
from vllm.models.deepseek_v4.hw_agnostic.attention.compressor import (
    DeepseekCompressor,
)
from vllm.models.deepseek_v4.hw_agnostic.attention.sparse_attn_indexer import (
    SparseAttnIndexer,
    _encode_layer_name,
)

from vllm_dsv4_oot_bf16_platform.kernels import (
    BF16_TOKEN_BYTES,
    DSV4_BF16_DS_MLA,
    compress_norm_rope_store_triton,
    dequantize_and_gather_k_cache,
    fused_indexer_q_rope_quant,
    triton_qnorm_rope_kv_insert,
    triton_sparse_decode,
)


class BF16MultiHeadLatentAttentionWrapper(DeepseekV4MultiHeadLatentAttentionWrapper):
    """Wrapper whose sparse-MLA slot is an all-BF16 1024B page (no scale)."""

    def _mla_head_bytes(self) -> int:
        # 512 bf16 values, no UE8M0 scale block.
        return BF16_TOKEN_BYTES

    def _insert_kv(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        swa_kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        positions: torch.Tensor,
        block_size: int,
    ) -> None:
        triton_qnorm_rope_kv_insert(
            q,
            kv,
            swa_kv_cache,
            slot_mapping,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.eps,
            block_size,
            use_fp8=False,
        )


class BF16MLAAttention(DeepseekV4MLAAttention):
    """MLA attention over the BF16 sparse-MLA KV cache."""

    def _resolve_cache_dtype(self) -> str:
        return DSV4_BF16_DS_MLA

    def _kv_cache_alignment(self) -> int | None:
        # The 1024B BF16 slot needs no extra alignment.
        return None

    def _gather_k(
        self,
        out: torch.Tensor,
        k_cache: torch.Tensor,
        seq_lens: torch.Tensor,
        gather_lens: torch.Tensor | None,
        block_table: torch.Tensor,
        block_size: int,
        offset: int,
    ) -> None:
        dequantize_and_gather_k_cache(
            out,
            k_cache,
            seq_lens=seq_lens,
            gather_lens=gather_lens,
            block_table=block_table,
            block_size=block_size,
            offset=offset,
            use_fp8=False,
        )

    def _sparse_decode(self, **kwargs) -> None:
        triton_sparse_decode(use_fp8=False, **kwargs)


class BF16Indexer(DeepseekV4Indexer):
    """Lightning indexer over a BF16 indexer K cache (256B/head_dim)."""

    def _indexer_k_cache_head_dim(self) -> int:
        # BF16 fallback: 128 bf16 = 256 bytes/head_dim, no scale.
        return self.head_dim * 2

    def _indexer_kv_cache_alignment(self) -> int | None:
        return None

    def _quant_index_q(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        indexer_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return fused_indexer_q_rope_quant(
            positions,
            q,
            cos_sin_cache,
            indexer_weights,
            self.softmax_scale,
            self.n_head**-0.5,
            use_fp8=False,
        )


class BF16Compressor(DeepseekCompressor):
    """Compressor writing BF16 (unquantized) state into the packed slot."""

    def _cache_slot_layout(self) -> tuple[int, int]:
        # All-BF16 slot: token_stride = value_dim * 2 bytes, no scale block.
        if self.head_dim == 512:
            token_stride = self.head_dim * 2  # 512 bf16
        else:  # head_dim == 128
            token_stride = self.head_dim * 2  # 128 bf16
        scale_dim = 0
        return token_stride, scale_dim

    def _store_compressed(self, **kwargs) -> None:
        compress_norm_rope_store_triton(use_fp8=False, **kwargs)


class BF16SparseAttnIndexer(SparseAttnIndexer):
    """Sparse indexer that calls the BF16 indexer op."""

    def _indexer_op(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.vllm_hw_agnostic.dsv4_sparse_attn_indexer_bf16(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_quant,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
        )


def register() -> None:
    """Register the BF16 overrides for the attention PluggableLayers.

    Also registers the BF16 sparse-indexer custom op used by
    ``BF16SparseAttnIndexer._indexer_op``.
    """
    from vllm_dsv4_oot_bf16_platform import indexer_op  # noqa: F401

    indexer_op.register_bf16_indexer_op()

    # ``PluggableLayer.__new__`` selects the OOT replacement by the in-tree
    # class's ``__name__`` (not its ``@register`` string), so register each
    # override against its base class: the default ``reg_name = cls.__name__``
    # is exactly the key ``__new__`` looks up.
    DeepseekV4MultiHeadLatentAttentionWrapper.register_oot(
        BF16MultiHeadLatentAttentionWrapper
    )
    DeepseekV4MLAAttention.register_oot(BF16MLAAttention)
    DeepseekV4Indexer.register_oot(BF16Indexer)
    DeepseekCompressor.register_oot(BF16Compressor)
    SparseAttnIndexer.register_oot(BF16SparseAttnIndexer)
