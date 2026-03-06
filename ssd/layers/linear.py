import mlx.core as mx
import mlx.nn as nn


class QKVLinear(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        super().__init__()
        self.head_size = head_size
        self.num_heads = total_num_heads
        self.num_kv_heads = total_num_kv_heads or total_num_heads
        output_size = (self.num_heads + 2 * self.num_kv_heads) * head_size
        self.weight = mx.zeros((output_size, hidden_size))
        self.bias = mx.zeros((output_size,)) if bias else None

    def weight_loader(self, loaded_weight: mx.array, loaded_shard_id: str):
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        parts = []
        if shard_offset > 0:
            parts.append(self.weight[:shard_offset])
        parts.append(loaded_weight)
        remainder = self.weight.shape[0] - shard_offset - shard_size
        if remainder > 0:
            parts.append(self.weight[shard_offset + shard_size:])
        self.weight = mx.concatenate(parts, axis=0)

    def __call__(self, x: mx.array) -> mx.array:
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y


class GateUpLinear(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.intermediate_size = intermediate_size
        self.weight = mx.zeros((2 * intermediate_size, hidden_size))
        self.bias = mx.zeros((2 * intermediate_size,)) if bias else None

    def weight_loader(self, loaded_weight: mx.array, loaded_shard_id: int):
        shard_offset = loaded_shard_id * self.intermediate_size
        shard_size = self.intermediate_size
        parts = []
        if shard_offset > 0:
            parts.append(self.weight[:shard_offset])
        parts.append(loaded_weight)
        remainder = self.weight.shape[0] - shard_offset - shard_size
        if remainder > 0:
            parts.append(self.weight[shard_offset + shard_size:])
        self.weight = mx.concatenate(parts, axis=0)

    def __call__(self, x: mx.array) -> mx.array:
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y


class RowLinear(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.weight = mx.zeros((output_size, input_size))
        self.bias = mx.zeros((output_size,)) if bias else None

    def weight_loader(self, loaded_weight: mx.array):
        self.weight = loaded_weight

    def __call__(self, x: mx.array) -> mx.array:
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y


class ReplicatedLinear(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.weight = mx.zeros((output_size, input_size))
        self.bias = mx.zeros((output_size,)) if bias else None

    def weight_loader(self, loaded_weight: mx.array):
        self.weight = loaded_weight

    def __call__(self, x: mx.array) -> mx.array:
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y
