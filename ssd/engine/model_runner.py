import os
import time
import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer, AutoConfig
from ssd.config import Config
from ssd.engine.sequence import Sequence
from ssd.models.qwen3 import Qwen3ForCausalLM
from ssd.models.llama3 import LlamaForCausalLM
from ssd.models.eagle3_draft_llama3 import Eagle3DraftForCausalLM
from ssd.layers.sampler import Sampler
from ssd.utils.context import set_context, reset_context, get_context
from ssd.utils.loader import load_model
from ssd.engine.helpers.runner_helpers import (
    prepare_decode_tensors_from_seqs,
    prepare_block_tables_from_seqs,
    prepare_prefill_tensors_from_seqs,
)
from ssd.engine.helpers.mask_helpers import get_custom_mask


class ModelRunner:

    def __init__(self, config: Config, is_draft: bool = False):
        self.config = config
        self.is_draft = is_draft
        self.hf_config = config.draft_hf_config if is_draft else config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_path if config.tokenizer_path else config.model,
            use_fast=True,
        )
        self.max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        self.use_eagle = config.use_eagle
        self.verbose = config.verbose
        self.draft_async = config.draft_async

        if self.is_draft and self.draft_async:
            self.prev_fork_keys: mx.array | None = None
            self.prev_fork_block_tables: mx.array | None = None

        self._build_model(config)

        if self.config.draft_async:
            if self.config.fan_out_list is None:
                self.config.fan_out_list = [self.config.async_fan_out] * (self.config.speculate_k + 1)
            self.config.fan_out_t = mx.array(self.config.fan_out_list, dtype=mx.int32)
            self.config.fan_out_t_miss = mx.array(self.config.fan_out_list_miss, dtype=mx.int32)
            assert len(self.config.fan_out_list) == self.config.speculate_k + 1
            assert any(f > 0 for f in self.config.fan_out_list)
            self.config.MQ_LEN = sum(self.config.fan_out_list)

    def _build_model(self, config: Config):
        hf_config = self.hf_config

        if config.use_eagle and self.is_draft:
            model_class = Eagle3DraftForCausalLM
        elif hf_config.model_type == 'llama':
            model_class = LlamaForCausalLM
        elif hf_config.model_type == 'qwen3':
            model_class = Qwen3ForCausalLM
        else:
            raise ValueError(f"Unsupported model type: {hf_config.model_type}")

        kwargs = dict(
            config=hf_config,
            draft=self.is_draft,
            speculate=config.speculate,
            spec_k=config.speculate_k,
            async_fan_out=config.async_fan_out,
            draft_async=config.draft_async,
        )

        if config.use_eagle:
            kwargs['use_eagle'] = True
            kwargs['eagle_layers'] = config.eagle_layers

        if model_class == Eagle3DraftForCausalLM:
            kwargs['d_model_target'] = config.d_model_target

        self.model = model_class(**kwargs)

        model_type = "DRAFT" if self.is_draft else "TARGET"
        if self.verbose:
            print(f'[ModelRunner] Loading {model_type} model', flush=True)

        target_path = getattr(config, 'tokenizer_path', None)
        target_hidden_size = getattr(config, 'd_model_target', None)
        load_model(self.model, config.model if not self.is_draft else config.draft,
                   target_path=target_path, target_hidden_size=target_hidden_size)

        if self.verbose:
            print(f'[ModelRunner] {model_type} model loaded', flush=True)

        self.sampler = Sampler(sampler_x=config.sampler_x, async_fan_out=config.async_fan_out)

        if self.verbose:
            print(f'[ModelRunner] Warming up {model_type} model', flush=True)
        self.warmup_model()

        if self.verbose:
            print(f'[ModelRunner] Allocating {model_type} KV cache', flush=True)
        self.allocate_kv_cache()

        if self.verbose:
            print(f'[ModelRunner] {model_type} model ready', flush=True)

    def warmup_model(self):
        max_num_batched_tokens = self.config.max_num_batched_tokens
        max_model_len = self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]

        hidden_states = None
        if self.config.use_eagle and self.is_draft:
            num_tokens = num_seqs * max_model_len
            d_model_target = self.config.d_model_target or 4096
            hidden_states = mx.zeros((num_tokens, 3 * d_model_target))

        self.run(seqs, True, hidden_states=hidden_states)

    def allocate_kv_cache(self):
        config = self.config
        hf_config = self.hf_config

        metal_info = mx.metal.device_info()
        total_memory = metal_info["memory_size"]
        recommended = metal_info.get("recommended_max_working_set_size", total_memory)
        free = recommended

        num_kv_heads = hf_config.num_key_value_heads
        head_dim = getattr(hf_config, 'head_dim', hf_config.hidden_size // hf_config.num_attention_heads)
        dtype_size = 2

        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * dtype_size
        )
        usable_bytes = free * config.memory_utilization

        if self.is_draft and self.draft_async:
            B = config.max_num_seqs
            K = config.speculate_k
            F = config.async_fan_out
            V = hf_config.vocab_size
            reserved_bytes = B * (K + 1) * F * (K + 1) * V * dtype_size
            usable_bytes = max(usable_bytes - reserved_bytes, 0)
            assert usable_bytes > 0, "Not enough memory for draft KV cache after tree_cache reservation"

        config.num_kvcache_blocks = int(usable_bytes) // block_bytes
        if self.verbose:
            model_type = "TARGET" if not self.is_draft else "DRAFT"
            print(f'[ModelRunner] KV cache allocation for {model_type}:', flush=True)
            print(f'  free={free / 1e9:.2f}GB, util={config.memory_utilization:.2f}', flush=True)
            print(f'  block_bytes={block_bytes}, num_blocks={config.num_kvcache_blocks}', flush=True)

        assert config.num_kvcache_blocks > 0, "KV cache too big for free memory!"

        num_slots = config.num_kvcache_blocks * self.block_size
        self.k_cache = mx.zeros((hf_config.num_hidden_layers, num_slots, num_kv_heads * head_dim))
        self.v_cache = mx.zeros((hf_config.num_hidden_layers, num_slots, num_kv_heads * head_dim))

        if self.verbose:
            print(f'[ModelRunner] k_cache shape={self.k_cache.shape}', flush=True)

        layer_id = 0
        for name, module in self.model.named_modules():
            if hasattr(module, 'k_cache') and hasattr(module, 'v_cache'):
                module.k_cache = self.k_cache[layer_id]
                module.v_cache = self.v_cache[layer_id]
                layer_id += 1

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids, positions, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping = \
            prepare_prefill_tensors_from_seqs(seqs, self.block_size, self.is_draft)

        block_tables = None
        mx.eval(cu_seqlens_q, cu_seqlens_k)
        if int(cu_seqlens_k[-1].item()) > int(cu_seqlens_q[-1].item()):
            block_tables = prepare_block_tables_from_seqs(seqs, self.is_draft)

        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            context_lens=None,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence], verify: bool = False):
        input_ids, positions, slot_mapping, context_lens = \
            prepare_decode_tensors_from_seqs(
                seqs, self.block_size, self.is_draft, verify,
                self.config.speculate_k if verify else -1,
            )

        block_tables = prepare_block_tables_from_seqs(seqs, self.is_draft)

        if verify:
            cu_seqlens_q = mx.zeros((len(seqs) + 1,), dtype=mx.int32)
            seqlen_q = mx.full((len(seqs),), self.config.speculate_k + 1, dtype=mx.int32)
            cu_seqlens_q = mx.concatenate([mx.array([0], dtype=mx.int32), mx.cumsum(seqlen_q)])
            set_context(
                is_prefill=False,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=None,
                max_seqlen_q=self.config.speculate_k + 1,
                max_seqlen_k=0,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
            )
        else:
            set_context(
                is_prefill=False,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=0,
                max_seqlen_k=0,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
            )

        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            if self.is_draft and seq.draft_temperature is not None:
                temperatures.append(seq.draft_temperature)
            else:
                temperatures.append(seq.temperature)
        return mx.array(temperatures, dtype=mx.float32)

    def eager_tree_decode_plan(self, input_ids, positions, step, cache_hits):
        assert self.is_draft and self.config.draft_async
        context = get_context()

        K, F = self.config.speculate_k, self.config.async_fan_out
        MQ_LEN = self.config.MQ_LEN
        flat_batch_size = input_ids.shape[0]
        B = flat_batch_size // MQ_LEN

        custom_mask = get_custom_mask(
            self.config, context.context_lens, step, K, F, B,
            cache_hits=cache_hits,
        )
        context.custom_mask = custom_mask

    def run_model(
        self,
        input_ids: mx.array,
        positions: mx.array,
        is_prefill: bool,
        last_only: bool = True,
        tree_decode_step: int = -1,
        cache_hits: mx.array | None = None,
        hidden_states: mx.array | None = None,
    ):
        is_tree_decode = self.is_draft and self.config.draft_async and tree_decode_step >= 0

        if is_tree_decode:
            self.eager_tree_decode_plan(input_ids, positions, tree_decode_step, cache_hits)

        if self.config.use_eagle:
            if self.is_draft:
                assert hidden_states is not None
                prenorm = self.model(input_ids, positions, hidden_states)
                logits = self.model.compute_logits(prenorm, last_only)
                return logits, prenorm
            else:
                outputs, eagle_acts = self.model(input_ids, positions)
                logits = self.model.compute_logits(outputs, last_only)
                return logits, eagle_acts
        else:
            outputs = self.model(input_ids, positions)
            logits = self.model.compute_logits(outputs, last_only)
            return logits

    def run(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
        last_only: bool = True,
        draft_return_logits: bool = False,
        hidden_states: mx.array | None = None,
        tree_decode_step: int = -1,
        cache_hits: mx.array | None = None,
    ) -> list[int] | tuple[list[int], mx.array]:
        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs, verify=not last_only)
        temperatures = self.prepare_sample(seqs)

        conditioning = None
        if self.config.use_eagle:
            logits, conditioning = self.run_model(
                input_ids, positions, is_prefill, last_only,
                tree_decode_step=tree_decode_step,
                cache_hits=cache_hits,
                hidden_states=hidden_states,
            )
        else:
            logits = self.run_model(
                input_ids, positions, is_prefill, last_only,
                tree_decode_step=tree_decode_step,
                cache_hits=cache_hits,
                hidden_states=hidden_states,
            )

        if last_only:
            mx.eval(logits)
            token_ids = self.sampler(logits, temperatures).tolist()
            reset_context()
            if conditioning is not None:
                return token_ids, conditioning
            return (token_ids, logits) if draft_return_logits else token_ids
        else:
            mx.eval(logits)
            reset_context()
            if conditioning is not None:
                return logits, conditioning
            return logits

    def exit(self):
        pass
