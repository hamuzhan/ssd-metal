import mlx.core as mx
import mlx.nn as nn
from transformers import LlamaConfig
from ssd.layers.activation import SiluAndMul
from ssd.layers.attention import Attention
from ssd.layers.layernorm import RMSDNorm
from ssd.layers.linear import QKVLinear, GateUpLinear, RowLinear
from ssd.layers.rotary_embedding import get_rope
from ssd.layers.embed_head import Embedding, LMHead


class LlamaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        rope_theta: float = 500000,
        rope_scaling: dict | None = None,
        draft: bool = False,
        speculate: bool = False,
        spec_k: int = 1,
        async_fan_out: int = 1,
        draft_async: bool = False,
    ):
        super().__init__()
        self.draft = draft
        self.draft_async = draft_async

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
            bias=False,
        )
        self.o_proj = RowLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )

        if rope_scaling is not None:
            rope_scaling = None

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

    def __call__(self, positions: mx.array, hidden_states: mx.array) -> mx.array:
        qkv = self.qkv_proj(hidden_states)
        q = qkv[..., :self.q_size]
        k = qkv[..., self.q_size:self.q_size + self.kv_size]
        v = qkv[..., self.q_size + self.kv_size:]
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output


class LlamaMLP(nn.Module):

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


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        draft: bool,
        speculate: bool,
        spec_k: int,
        async_fan_out: int,
        draft_async: bool,
    ):
        super().__init__()
        self.self_attn = LlamaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 500000),
            rope_scaling=getattr(config, "rope_scaling", None),
            draft=draft,
            speculate=speculate,
            spec_k=spec_k,
            async_fan_out=async_fan_out,
            draft_async=draft_async,
        )

        self.mlp = LlamaMLP(
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
            hidden_states, residual = self.input_layernorm(hidden_states, residual), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        draft: bool = False,
        speculate: bool = False,
        spec_k: int = 1,
        async_fan_out: int = 1,
        draft_async: bool = False,
        use_eagle: bool = False,
        eagle_layers: list[int] | None = None,
    ):
        super().__init__()
        self.use_eagle = use_eagle
        self.eagle_layers = eagle_layers
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            LlamaDecoderLayer(
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

    def __call__(self, input_ids: mx.array, positions: mx.array):
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        collected_acts = [] if self.use_eagle else None

        for layer_idx, layer in enumerate(self.layers):
            if collected_acts is not None and layer_idx in self.eagle_layers:
                current_act = hidden_states if residual is None else hidden_states + residual
                collected_acts.append(current_act)
            hidden_states, residual = layer(positions, hidden_states, residual)

        hidden_states, _ = self.norm(hidden_states, residual)

        if collected_acts:
            eagle_acts = mx.concatenate(collected_acts, axis=-1)
            return hidden_states, eagle_acts
        else:
            return hidden_states


class LlamaForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: LlamaConfig,
        draft: bool = False,
        speculate: bool = False,
        use_eagle: bool = False,
        eagle_layers: list[int] | None = None,
        spec_k: int = 1,
        async_fan_out: int = 1,
        draft_async: bool = False,
    ):
        super().__init__()
        self.draft = draft
        self.use_eagle = use_eagle
        self.eagle_layers = eagle_layers

        self.model = LlamaModel(
            config, draft, speculate, spec_k, async_fan_out, draft_async,
            use_eagle=use_eagle, eagle_layers=eagle_layers,
        )
        self.lm_head = LMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def __call__(self, input_ids: mx.array, positions: mx.array):
        out = self.model(input_ids, positions)
        return out

    def compute_logits(self, hidden_states: mx.array, last_only: bool = True) -> mx.array:
        logits = self.lm_head(hidden_states, last_only=last_only)
        return logits
