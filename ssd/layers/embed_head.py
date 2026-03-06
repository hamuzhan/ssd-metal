import mlx.core as mx
import mlx.nn as nn
from ssd.utils.context import get_context


class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = mx.zeros((num_embeddings, embedding_dim))

    def weight_loader(self, loaded_weight: mx.array):
        self.weight = loaded_weight

    def __call__(self, x: mx.array) -> mx.array:
        return self.weight[x]


class LMHead(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = mx.zeros((num_embeddings, embedding_dim))

    def weight_loader(self, loaded_weight: mx.array):
        self.weight = loaded_weight

    def __call__(self, x: mx.array, last_only: bool = True) -> mx.array:
        context = get_context()
        if context.cu_seqlens_q is not None:
            if context.is_prefill:
                if last_only:
                    last_indices = context.cu_seqlens_q[1:] - 1
                    x = x[last_indices]
                else:
                    return x @ self.weight.T
            else:
                flat_logits = x @ self.weight.T
                batch_size = context.cu_seqlens_q.shape[0] - 1
                total_tokens = x.shape[0]
                if total_tokens % batch_size == 0:
                    constant_query_len = total_tokens // batch_size
                    return flat_logits.reshape(batch_size, constant_query_len, -1)
                return flat_logits
        return x @ self.weight.T
