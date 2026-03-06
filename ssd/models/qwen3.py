import mlx.core as mx
import mlx.nn as nn
from transformers import Qwen3Config
from ssd.layers.activation import SiluAndMul
from ssd.layers.attention import Attention
from ssd.layers.layernorm import RMSHeadNorm, RMSDNorm
from ssd.layers.linear import QKVLinear, GateUpLinear, RowLinear
from ssd.layers.rotary_embedding import get_rope
from ssd.layers.embed_head import Embedding, LMHead


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
        draft: bool = False,
        speculate: bool = False,
        spec_k: int = 1,
        async_fan_out: int = 1,
        draft_async: bool = False,
    ):
        super().__init__()
        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = self.total_num_kv_heads
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            draft=draft,
            speculate=speculate,
            draft_async=draft_async,
            F=async_fan_out,
            K=spec_k,
        )
        self.q_norm = RMSHeadNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSHeadNorm(self.head_dim, eps=rms_norm_eps)

    def __call__(self, positions: mx.array, hidden_states: mx.array) -> mx.array:
        qkv = self.qkv_proj(hidden_states)
        q = qkv[..., :self.q_size]
        k = qkv[..., self.q_size:self.q_size + self.kv_size]
        v = qkv[..., self.q_size + self.kv_size:]
        q_by_head = q.reshape(-1, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.reshape(q.shape)
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.reshape(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_up_proj = GateUpLinear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.down_proj = RowLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def __call__(self, x: mx.array) -> mx.array:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        draft: bool,
        speculate: bool,
        spec_k: int,
        async_fan_out: int,
        draft_async: bool,
    ):
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            draft=draft,
            speculate=speculate,
            spec_k=spec_k,
            async_fan_out=async_fan_out,
            draft_async=draft_async,
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSDNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSDNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        positions: mx.array,
        hidden_states: mx.array,
        residual: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        draft: bool = False,
        speculate: bool = False,
        spec_k: int = 1,
        async_fan_out: int = 1,
        draft_async: bool = False,
    ):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Qwen3DecoderLayer(
                config,
                draft=draft,
                speculate=speculate,
                spec_k=spec_k,
                async_fan_out=async_fan_out,
                draft_async=draft_async,
            )
            for _ in range(config.num_hidden_layers)
        ]
        self.norm = RMSDNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, input_ids: mx.array, positions: mx.array) -> mx.array:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config,
        draft: bool = False,
        speculate: bool = False,
        use_eagle: bool = False,
        spec_k: int = 1,
        async_fan_out: int = 1,
        draft_async: bool = False,
    ):
        super().__init__()
        self.draft = draft
        self.model = Qwen3Model(config, draft, speculate, spec_k, async_fan_out, draft_async)
        self.lm_head = LMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def __call__(self, input_ids: mx.array, positions: mx.array) -> mx.array:
        hidden_states = self.model(input_ids, positions)
        return hidden_states

    def compute_logits(self, hidden_states: mx.array, last_only: bool = True) -> mx.array:
        logits = self.lm_head(hidden_states, last_only=last_only)
        return logits
