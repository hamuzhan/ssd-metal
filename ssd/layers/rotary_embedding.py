from functools import lru_cache
import mlx.core as mx
import mlx.nn as nn


def apply_rotary_emb(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    cos = mx.expand_dims(cos, axis=-2)
    sin = mx.expand_dims(sin, axis=-2)
    orig_dtype = x.dtype
    x = x.astype(mx.float32)
    x1, x2 = mx.split(x, 2, axis=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return mx.concatenate([y1, y2], axis=-1).astype(orig_dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ):
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base ** (mx.arange(0, rotary_dim, 2, dtype=mx.float32) / rotary_dim))
        t = mx.arange(max_position_embeddings, dtype=mx.float32)
        freqs = t[:, None] * inv_freq[None, :]
        cos = mx.cos(freqs)
        sin = mx.sin(freqs)
        self._cos_sin_cache = mx.concatenate([cos, sin], axis=-1)
        self.freeze(keys=["_cos_sin_cache"])

    def __call__(
        self,
        positions: mx.array,
        query: mx.array,
        key: mx.array,
    ) -> tuple[mx.array, mx.array]:
        cos_sin = self._cos_sin_cache[positions]
        cos, sin = mx.split(cos_sin, 2, axis=-1)
        query_shape = query.shape
        num_q_heads = query.shape[-1] // self.head_size
        query = query.reshape(-1, num_q_heads, self.head_size)
        query = apply_rotary_emb(query, cos, sin).reshape(query_shape)
        key_shape = key.shape
        num_k_heads = key.shape[-1] // self.head_size
        key = key.reshape(-1, num_k_heads, self.head_size)
        key = apply_rotary_emb(key, cos, sin).reshape(key_shape)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
