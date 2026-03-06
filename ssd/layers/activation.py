import mlx.core as mx
import mlx.nn as nn


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, x: mx.array) -> mx.array:
        x, y = mx.split(x, 2, axis=-1)
        return nn.silu(x) * y
