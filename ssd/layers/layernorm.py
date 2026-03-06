import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((hidden_size,))

    def __call__(self, x: mx.array, residual: mx.array | None = None):
        if residual is not None:
            x = x + residual
            residual = x
        orig_dtype = x.dtype
        x = x.astype(mx.float32)
        variance = mx.mean(x * x, axis=-1, keepdims=True)
        x = x * mx.rsqrt(variance + self.eps)
        x = x.astype(orig_dtype) * self.weight
        if residual is not None:
            return x, residual
        return x


RMSHeadNorm = RMSNorm
RMSDNorm = RMSNorm
