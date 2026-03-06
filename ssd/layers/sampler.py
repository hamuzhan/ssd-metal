import mlx.core as mx
import mlx.nn as nn
from ssd.utils.async_helpers.async_spec_helpers import apply_sampler_x_rescaling


class Sampler(nn.Module):

    def __init__(self, sampler_x: float | None = None, async_fan_out: int = 3):
        super().__init__()
        self.sampler_x = sampler_x
        self.F = async_fan_out

    def __call__(self, logits: mx.array, temperatures: mx.array, is_tree: bool = False) -> mx.array:
        logits = logits.astype(mx.float32)
        greedy_tokens = mx.argmax(logits, axis=-1)

        zero_mask = temperatures == 0

        logits = logits / mx.expand_dims(temperatures, axis=-1)
        probs = mx.softmax(logits, axis=-1)

        if self.sampler_x is not None and is_tree:
            probs = apply_sampler_x_rescaling(probs, self.sampler_x, self.F)

        epsilon = 1e-10
        u = mx.random.uniform(shape=probs.shape)
        exp_samples = -mx.log(u + epsilon)
        scores = probs / (exp_samples + epsilon)
        sample_tokens = mx.argmax(scores, axis=-1)

        return mx.where(zero_mask, greedy_tokens, sample_tokens)
