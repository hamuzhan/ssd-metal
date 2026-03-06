import mlx.core as mx
import mlx.nn as nn
from transformers import LlamaConfig
from ssd.layers.activation import SiluAndMul
from ssd.layers.attention import Attention
from ssd.layers.layernorm import RMSDNorm
from ssd.layers.linear import QKVLinear, GateUpLinear, RowLinear
from ssd.layers.rotary_embedding import get_rope
from ssd.layers.embed_head import Embedding, LMHead
from ssd.models.llama3 import LlamaMLP


class Eagle3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int,
        rms_norm_eps: float,
        head_dim: int | None,
        rope_theta: float,
        rope_scaling: dict | None,
        draft: bool,
        speculate: bool,
        spec_k: int,
        async_fan_out: int,
        draft_async: bool,
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
            2 * hidden_size,
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
            use_eagle=True,
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


class Eagle3DecoderLayer(nn.Module):

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
        self.self_attn = Eagle3Attention(
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
        self.conditioning_feature_ln = RMSDNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSDNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        positions: mx.array,
        token_embeddings: mx.array,
        conditioning_features: mx.array,
    ) -> mx.array:
        normed_tokens = self.input_layernorm(token_embeddings)
        normed_conditioning = self.conditioning_feature_ln(conditioning_features)
        hidden_states = mx.concatenate([normed_tokens, normed_conditioning], axis=-1)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, conditioning_features)
        hidden_states = self.mlp(hidden_states) + residual
        return hidden_states


class Eagle3DraftModel(nn.Module):

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
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        assert config.num_hidden_layers == 1
        self.layer = Eagle3DecoderLayer(
            config,
            draft=draft,
            speculate=speculate,
            spec_k=spec_k,
            async_fan_out=async_fan_out,
            draft_async=draft_async,
        )

    def __call__(
        self,
        input_ids: mx.array,
        target_hidden_states_projected: mx.array,
        positions: mx.array,
    ) -> mx.array:
        token_embeddings = self.embed_tokens(input_ids)
        hidden_states = self.layer(positions, token_embeddings, target_hidden_states_projected)
        return hidden_states


class Eagle3DraftForCausalLM(nn.Module):
    packed_modules_mapping = {
        "midlayer.self_attn.q_proj": ("model.layer.self_attn.qkv_proj", "q"),
        "midlayer.self_attn.k_proj": ("model.layer.self_attn.qkv_proj", "k"),
        "midlayer.self_attn.v_proj": ("model.layer.self_attn.qkv_proj", "v"),
        "midlayer.mlp.gate_proj": ("model.layer.mlp.gate_up_proj", 0),
        "midlayer.mlp.up_proj": ("model.layer.mlp.gate_up_proj", 1),
    }

    def __init__(
        self,
        config: LlamaConfig,
        draft: bool = False,
        speculate: bool = False,
        use_eagle: bool = False,
        eagle_layers: list[int] | None = None,
        d_model_target: int = 4096,
        spec_k: int = 1,
        async_fan_out: int = 1,
        draft_async: bool = False,
    ):
        super().__init__()

        assert draft
        assert use_eagle
        assert eagle_layers is not None

        self.config = config
        self.draft = draft
        self.use_eagle = use_eagle
        self.eagle_layers = eagle_layers if eagle_layers is not None else []
        self.d_model_target = d_model_target
        self.d2t = {}
        self.t2d = {}
        self.d2t_tensor = None
        self.t2d_tensor = None

        self.fc = RowLinear(len(self.eagle_layers) * d_model_target, config.hidden_size, bias=False)
        self.final_norm = RMSDNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.model = Eagle3DraftModel(
            config, draft, speculate, spec_k, async_fan_out, draft_async,
            use_eagle=use_eagle, eagle_layers=eagle_layers,
        )
        self.lm_head = LMHead(config.draft_vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def __call__(
        self,
        input_ids: mx.array,
        positions: mx.array,
        hidden_states: mx.array,
    ) -> mx.array:
        if hidden_states.shape[-1] == len(self.eagle_layers) * self.d_model_target:
            hidden_states_projected = self.fc(hidden_states.astype(self.fc.weight.dtype))
        else:
            hidden_states_projected = hidden_states
        prenorm = self.model(input_ids, hidden_states_projected, positions)
        return prenorm

    def compute_logits(self, hidden_states: mx.array, last_only: bool = True) -> mx.array:
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states, last_only=last_only)

        if logits.ndim == 3:
            logits = logits.reshape(-1, logits.shape[-1])

        assert self.d2t_tensor is not None
        B = logits.shape[0]
        vocab_size = self.config.vocab_size

        base = mx.arange(self.config.draft_vocab_size)
        target_indices = base + self.d2t_tensor

        logits_full = mx.full((B, vocab_size), float('-inf'))
        logits_full[:, target_indices] = logits

        return logits_full
