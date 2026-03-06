import mlx.core as mx
import mlx.nn as nn
from ssd.kernels.kvcache_store import store_kvcache
from ssd.utils.context import get_context


def _gather_paged_kv(k_cache, v_cache, block_tables_b, ctx_len, block_size, num_kv_heads, head_dim):
    num_blocks = (ctx_len + block_size - 1) // block_size
    block_ids = block_tables_b[:num_blocks]
    flat_slots = []
    for bi in range(num_blocks):
        block_id = int(block_ids[bi].item())
        start = block_id * block_size
        remaining = ctx_len - bi * block_size
        count = min(block_size, remaining)
        flat_slots.append(mx.arange(start, start + count))
    all_slots = mx.concatenate(flat_slots)
    k_out = k_cache[all_slots].reshape(ctx_len, num_kv_heads, head_dim)
    v_out = v_cache[all_slots].reshape(ctx_len, num_kv_heads, head_dim)
    return k_out, v_out


def _sdpa_causal(q, k, v, scale, num_heads, num_kv_heads):
    q = mx.expand_dims(q, axis=0)
    k = mx.expand_dims(k, axis=0)
    v = mx.expand_dims(v, axis=0)
    q = mx.transpose(q, axes=(0, 2, 1, 3))
    k = mx.transpose(k, axes=(0, 2, 1, 3))
    v = mx.transpose(v, axes=(0, 2, 1, 3))
    if num_heads != num_kv_heads:
        repeats = num_heads // num_kv_heads
        k = mx.repeat(k, repeats, axis=1)
        v = mx.repeat(v, repeats, axis=1)
    q_len = q.shape[2]
    k_len = k.shape[2]
    causal_offset = k_len - q_len
    row_idx = mx.arange(q_len).reshape(-1, 1)
    col_idx = mx.arange(k_len).reshape(1, -1)
    mask = (col_idx <= row_idx + causal_offset).astype(mx.float32)
    mask = mx.where(mask, 0.0, -1e9)
    mask = mask.reshape(1, 1, q_len, k_len)
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    o = mx.transpose(o, axes=(0, 2, 1, 3))
    return o.squeeze(0)


def _sdpa_with_mask(q, k, v, scale, attn_mask, num_heads, num_kv_heads):
    q = mx.expand_dims(q, axis=0)
    k = mx.expand_dims(k, axis=0)
    v = mx.expand_dims(v, axis=0)
    q = mx.transpose(q, axes=(0, 2, 1, 3))
    k = mx.transpose(k, axes=(0, 2, 1, 3))
    v = mx.transpose(v, axes=(0, 2, 1, 3))
    if num_heads != num_kv_heads:
        repeats = num_heads // num_kv_heads
        k = mx.repeat(k, repeats, axis=1)
        v = mx.repeat(v, repeats, axis=1)
    float_mask = mx.where(attn_mask, 0.0, -1e9)
    float_mask = float_mask.reshape(1, 1, attn_mask.shape[0], attn_mask.shape[1])
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=float_mask)
    o = mx.transpose(o, axes=(0, 2, 1, 3))
    return o.squeeze(0)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        draft: bool = False,
        speculate: bool = False,
        draft_async: bool = False,
        use_eagle: bool = False,
        F: int = 1,
        K: int = 1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = None
        self.v_cache = None
        self.draft = draft
        self.speculate = speculate
        self.draft_async = draft_async
        self.use_eagle = use_eagle
        self.F = F
        self.K = K
        self.block_size = 256

    def __call__(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_kv_heads, self.head_dim)
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)

        context = get_context()
        if self.k_cache is not None and self.v_cache is not None:
            store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None and self.k_cache is not None:
                cu_q = context.cu_seqlens_q
                cu_k = context.cu_seqlens_k
                B = cu_q.shape[0] - 1
                mx.eval(cu_q, cu_k, context.context_lens)
                outputs = []
                for b in range(B):
                    q_start = int(cu_q[b].item())
                    q_end = int(cu_q[b + 1].item())
                    ctx_len = int(context.context_lens[b].item())
                    q_b = q[q_start:q_end]
                    k_b, v_b = _gather_paged_kv(
                        self.k_cache, self.v_cache, context.block_tables[b],
                        ctx_len, self.block_size, self.num_kv_heads, self.head_dim,
                    )
                    o_b = _sdpa_causal(q_b, k_b, v_b, self.scale, self.num_heads, self.num_kv_heads)
                    outputs.append(o_b)
                o = mx.concatenate(outputs, axis=0)
            else:
                cu_q = context.cu_seqlens_q
                cu_k = context.cu_seqlens_k
                B = cu_q.shape[0] - 1
                mx.eval(cu_q, cu_k)
                outputs = []
                for b in range(B):
                    q_start = int(cu_q[b].item())
                    q_end = int(cu_q[b + 1].item())
                    k_start = int(cu_k[b].item())
                    k_end = int(cu_k[b + 1].item())
                    q_b = q[q_start:q_end]
                    k_b = k[k_start:k_end]
                    v_b = v[k_start:k_end]
                    o_b = _sdpa_causal(q_b, k_b, v_b, self.scale, self.num_heads, self.num_kv_heads)
                    outputs.append(o_b)
                o = mx.concatenate(outputs, axis=0)
        else:
            verify_or_glue = self.speculate and context.cu_seqlens_q is not None
            decode = not verify_or_glue
            tree_decode = (
                decode and self.speculate and self.draft and self.draft_async
                and not context.is_jit
            )

            if verify_or_glue:
                cu_q = context.cu_seqlens_q
                B = cu_q.shape[0] - 1
                mx.eval(cu_q, context.context_lens)
                outputs = []
                for b in range(B):
                    q_start = int(cu_q[b].item())
                    q_end = int(cu_q[b + 1].item())
                    q_b = q[q_start:q_end]
                    ctx_len = int(context.context_lens[b].item())
                    k_b, v_b = _gather_paged_kv(
                        self.k_cache, self.v_cache, context.block_tables[b],
                        ctx_len, self.block_size, self.num_kv_heads, self.head_dim,
                    )
                    o_b = _sdpa_causal(q_b, k_b, v_b, self.scale, self.num_heads, self.num_kv_heads)
                    outputs.append(o_b)
                o = mx.concatenate(outputs, axis=0)

            elif tree_decode:
                mq_len = self.F * (self.K + 1)
                B = q.shape[0] // mq_len
                mx.eval(context.context_lens)
                outputs = []
                for b in range(B):
                    q_b = q[b * mq_len:(b + 1) * mq_len]
                    ctx_len = int(context.context_lens[b].item())
                    k_b, v_b = _gather_paged_kv(
                        self.k_cache, self.v_cache, context.block_tables[b],
                        ctx_len, self.block_size, self.num_kv_heads, self.head_dim,
                    )
                    if hasattr(context, 'custom_mask') and context.custom_mask is not None:
                        mask_2d = context.custom_mask[b].reshape(mq_len, ctx_len)
                        o_b = _sdpa_with_mask(q_b, k_b, v_b, self.scale, mask_2d, self.num_heads, self.num_kv_heads)
                    else:
                        o_b = _sdpa_causal(q_b, k_b, v_b, self.scale, self.num_heads, self.num_kv_heads)
                    outputs.append(o_b)
                o = mx.concatenate(outputs, axis=0)

            else:
                B = q.shape[0]
                mx.eval(context.context_lens)
                outputs = []
                for b in range(B):
                    q_b = q[b:b + 1]
                    ctx_len = int(context.context_lens[b].item())
                    k_b, v_b = _gather_paged_kv(
                        self.k_cache, self.v_cache, context.block_tables[b],
                        ctx_len, self.block_size, self.num_kv_heads, self.head_dim,
                    )
                    o_b = _sdpa_causal(q_b, k_b, v_b, self.scale, self.num_heads, self.num_kv_heads)
                    outputs.append(o_b)
                o = mx.concatenate(outputs, axis=0)

        o = o.reshape(-1, self.num_heads * self.head_dim)
        return o
