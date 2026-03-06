import os
import time
import dataclasses
import mlx.core as mx

from ssd.engine.model_runner import ModelRunner
from ssd.config import Config
from ssd.utils.context import set_context, reset_context
from ssd.utils.async_helpers.async_spec_helpers import get_forked_recovery_tokens_from_logits, make_glue_decode_input_ids
from ssd.engine.helpers.mask_helpers import get_custom_mask

ttl = 0
ttl_hit = 0


class DraftRunner(ModelRunner):

    @classmethod
    def create_draft_config(cls, cfg: Config) -> Config:
        draft_cfg = dataclasses.replace(
            cfg,
            model=cfg.draft,
            memory_utilization=0.8 if cfg.draft_async else 0.75,
            tokenizer_path=cfg.model if cfg.use_eagle else None,
            d_model_target=cfg.hf_config.hidden_size if cfg.use_eagle and cfg.hf_config else None,
            enforce_eager=cfg.enforce_eager,
        )
        return draft_cfg

    def __init__(self, cfg: Config):
        self.draft_cfg = self.create_draft_config(cfg)
        self.prev_num_tokens = None
        super().__init__(self.draft_cfg, is_draft=True)

        if self.config.use_eagle:
            assert self.config.jit_speculate, \
                "EAGLE requires jit_speculate=True (cache misses need draft activations)"

        self._reset_tree_cache_tensors()
        self._init_prealloc_buffers()

    def draft_prefill(self, seqs, eagle_acts=None):
        hidden_states = eagle_acts
        self.run(seqs, is_prefill=True, hidden_states=hidden_states)

    def _reset_tree_cache_tensors(self):
        self.tree_cache_keys = mx.zeros((0, 3), dtype=mx.int64)
        self.tree_cache_tokens = None
        self.tree_cache_logits = None
        self.tree_cache_activations = None

    def _init_prealloc_buffers(self):
        K = self.config.speculate_k
        MQ_LEN = self.config.MQ_LEN
        self._step_pos_offsets = mx.expand_dims(mx.arange(K, dtype=mx.int64), axis=1) * MQ_LEN
        self._step_rope_offsets = mx.expand_dims(mx.arange(K, dtype=mx.int64), axis=1)
        self._fan_idx_hit = mx.repeat(mx.arange(K + 1, dtype=mx.int64), self.config.fan_out_t)
        self._fan_idx_miss = mx.repeat(mx.arange(K + 1, dtype=mx.int64), self.config.fan_out_t_miss)
        self._arange_mq = mx.arange(MQ_LEN, dtype=mx.int64)
        self._arange_kp1 = mx.arange(K + 1, dtype=mx.int64)
        self._arange_2kp1 = mx.arange(2 * K + 1, dtype=mx.int64)

    def jit_speculate(
        self,
        request_keys: mx.array,
        num_tokens: mx.array,
        out_logits: mx.array,
        out_tokens: mx.array,
        temperatures: mx.array,
        draft_block_tables: mx.array,
        target_recovery_activations: mx.array = None,
    ):
        input_ids = request_keys[:, -1]
        pos_offset = -1 if self.config.use_eagle else 0
        positions = num_tokens - 1 + pos_offset
        context_lens = num_tokens + pos_offset
        block_idx = positions // self.block_size
        pos_in_block = positions % self.block_size
        B = input_ids.shape[0]
        batch_indices = mx.arange(B)
        mx.eval(block_idx, batch_indices)
        slot_map = mx.array([
            int(draft_block_tables[b, int(block_idx[b].item())].item()) * self.block_size + int(pos_in_block[b].item())
            for b in range(B)
        ], dtype=mx.int32)

        hidden_states = None
        spec_activations = None

        if self.config.use_eagle:
            assert target_recovery_activations is not None
            hidden_states = self.model.fc(target_recovery_activations)
            spec_activations = mx.zeros(
                (B, self.config.speculate_k, self.hf_config.hidden_size))

        for i in range(self.config.speculate_k):
            set_context(
                is_prefill=False,
                slot_mapping=slot_map,
                context_lens=context_lens.astype(mx.int32),
                block_tables=draft_block_tables,
                is_jit=True,
            )

            if self.config.use_eagle:
                logits, prenorm = self.run_model(input_ids, positions, is_prefill=False, last_only=True, hidden_states=hidden_states)
                spec_activations[:, i] = prenorm
                hidden_states = prenorm
            else:
                logits = self.run_model(input_ids, positions, is_prefill=False, last_only=True)

            out_logits[:, i, :] = logits
            reset_context()
            mx.eval(logits)
            next_tokens = self.sampler(logits, temperatures, is_tree=True)
            out_tokens[:, i] = next_tokens

            input_ids = next_tokens
            positions = positions + 1
            context_lens = context_lens + 1
            block_idx = positions // self.block_size
            pos_in_block = positions % self.block_size
            mx.eval(block_idx, pos_in_block)
            slot_map = mx.array([
                int(draft_block_tables[b, int(block_idx[b].item())].item()) * self.block_size + int(pos_in_block[b].item())
                for b in range(B)
            ], dtype=mx.int32)

        return spec_activations

    def hit_cache_and_respond(self, request_keys, B, K, num_tokens, temperatures, draft_block_tables, target_recovery_activations=None):
        global ttl, ttl_hit
        V = self.hf_config.vocab_size

        out_logits = mx.random.uniform(shape=(B, K, V))
        out_tokens = mx.argmax(out_logits, axis=-1)
        cache_hits = mx.zeros((B,), dtype=mx.int64)

        hidden_size = self.hf_config.hidden_size
        out_activations = mx.zeros((B, K, hidden_size)) if self.config.use_eagle else None

        ttl += int(B)

        if self.tree_cache_keys.shape[0] > 0:
            eq = mx.expand_dims(request_keys, axis=1) == mx.expand_dims(self.tree_cache_keys, axis=0)
            match = mx.all(eq, axis=2)
            cache_hits = match.any(axis=1).astype(mx.int64)
            mx.eval(cache_hits)
            ttl_hit += int(cache_hits.sum().item())

            if (bool(cache_hits.any().item()) and not self.config.jit_speculate) or (bool(cache_hits.all().item()) and self.config.jit_speculate):
                idx = mx.argmax(match.astype(mx.float32), axis=1).astype(mx.int64)
                mx.eval(idx)
                for b in range(B):
                    if bool(cache_hits[b].item()):
                        i = int(idx[b].item())
                        out_tokens[b] = self.tree_cache_tokens[i]
                        out_logits[b] = self.tree_cache_logits[i]
                        if self.config.use_eagle and self.tree_cache_activations is not None:
                            out_activations[b] = self.tree_cache_activations[i]
            elif self.config.jit_speculate:
                jit_acts = self.jit_speculate(
                    request_keys, num_tokens, out_logits, out_tokens,
                    temperatures, draft_block_tables, target_recovery_activations,
                )
                if self.config.use_eagle:
                    out_activations = jit_acts
        elif self.config.jit_speculate:
            jit_acts = self.jit_speculate(
                request_keys, num_tokens, out_logits, out_tokens,
                temperatures, draft_block_tables, target_recovery_activations,
            )
            if self.config.use_eagle:
                out_activations = jit_acts

        rec_toks = request_keys[:, 2]

        return out_tokens, out_logits, make_glue_decode_input_ids(out_tokens, rec_toks), cache_hits, out_activations

    def prepare_prefill_ctxt(self, num_tokens, draft_block_table):
        B = num_tokens.shape[0]
        mx.eval(num_tokens)
        total = int(num_tokens.sum().item())
        cu_seqlens_q = mx.concatenate([mx.array([0], dtype=mx.int32), mx.cumsum(num_tokens.astype(mx.int32))])

        positions_list = []
        slot_map_list = []
        mx.eval(cu_seqlens_q)
        for b in range(B):
            n = int(num_tokens[b].item())
            start = int(cu_seqlens_q[b].item())
            for j in range(n):
                positions_list.append(j)
                block_i = j // self.block_size
                offset = j % self.block_size
                block_id = int(draft_block_table[b, block_i].item())
                slot_map_list.append(block_id * self.block_size + offset)

        positions = mx.array(positions_list, dtype=mx.int64)
        slot_map = mx.array(slot_map_list, dtype=mx.int32)
        mx.eval(num_tokens)
        max_seqlen_q = int(num_tokens.max().item())

        return {
            "positions": positions,
            "slot_map": slot_map,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": mx.array(cu_seqlens_q),
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_q,
        }

    def prepare_glue_decode_ctxt(self, num_tokens, input_ids, dbt, B):
        K = self.config.speculate_k
        pos_offset = -1 if self.config.use_eagle else 0

        positions_list = []
        slot_map_list = []
        context_lens_list = []

        mx.eval(num_tokens)
        for b in range(B):
            nt = int(num_tokens[b].item())
            base = nt - 1 + pos_offset
            for j in range(K + 1):
                pos = base + j
                positions_list.append(pos)
                block_i = pos // self.block_size
                offset = pos % self.block_size
                block_id = int(dbt[b, block_i].item())
                slot_map_list.append(block_id * self.block_size + offset)
            context_lens_list.append(nt + pos_offset + K)

        positions_flat = mx.array(positions_list, dtype=mx.int64)
        slot_map_flat = mx.array(slot_map_list, dtype=mx.int32)
        context_lens = mx.array(context_lens_list, dtype=mx.int32)

        seqlen_q = mx.full((B,), K + 1, dtype=mx.int32)
        cu_seqlens_q = mx.concatenate([mx.array([0], dtype=mx.int32), mx.cumsum(seqlen_q)])

        return {
            "input_ids": input_ids,
            "positions": positions_flat,
            "slot_map": slot_map_flat,
            "cu_seqlens_q": cu_seqlens_q,
            "max_seqlen_q": K + 1,
            "context_lens": context_lens,
            "block_tables": dbt,
        }

    def prepare_glue_decode_ctxt_eagle(self, num_tokens, fused_ids, fused_hs, extend_counts, seqlens_q, cu_seqlens_q, dbt, B):
        K = self.config.speculate_k
        mx.eval(cu_seqlens_q, num_tokens, extend_counts, seqlens_q)
        total_real = int(cu_seqlens_q[-1].item())

        positions_list = []
        slot_map_list = []
        context_lens_list = []

        for b in range(B):
            nt = int(num_tokens[b].item())
            n_ext = int(extend_counts[b].item())
            base_pos = nt - 2 - n_ext
            sq = int(seqlens_q[b].item())
            for j in range(sq):
                pos = base_pos + j
                positions_list.append(pos)
                block_i = max(0, min(pos // self.block_size, dbt.shape[1] - 1))
                offset = pos % self.block_size
                block_id = int(dbt[b, block_i].item())
                slot_map_list.append(block_id * self.block_size + offset)
            context_lens_list.append(nt - 1 + K)

        positions = mx.array(positions_list, dtype=mx.int64)
        slot_map = mx.array(slot_map_list, dtype=mx.int32)
        context_lens = mx.array(context_lens_list, dtype=mx.int32)

        return {
            "input_ids": fused_ids,
            "positions": positions,
            "slot_map": slot_map,
            "hidden_states": fused_hs,
            "cu_seqlens_q": cu_seqlens_q,
            "max_seqlen_q": 2 * K + 1,
            "context_lens": context_lens,
            "block_tables": dbt,
        }

    def _build_tree_batch(self, partial_tree_decode_args, glue_decode_input_ids):
        K = self.config.speculate_k
        dbt = partial_tree_decode_args["dbt"]
        cache_hits = partial_tree_decode_args["cache_hits"]
        mx.eval(cache_hits)
        cache_hits_list = cache_hits.tolist()
        pos_offset = -1 if self.config.use_eagle else 0

        if self.config.use_eagle:
            B = partial_tree_decode_args["num_tokens"].shape[0]
            extend_counts = partial_tree_decode_args.get("extend_counts")
            if extend_counts is None:
                extend_counts = mx.zeros((B,), dtype=mx.int64)
            extend_eagle_acts_batch = partial_tree_decode_args.get("extend_eagle_acts")
            extend_token_ids_batch = partial_tree_decode_args.get("extend_token_ids")
            target_acts = partial_tree_decode_args["target_recovery_activations"]
            prev_acts = partial_tree_decode_args["previous_activations"]
            hidden_size = self.hf_config.hidden_size

            gd_view = glue_decode_input_ids.reshape(B, K + 1)
            rec_tok_ids = gd_view[:, 0]
            spec_tok_ids = gd_view[:, 1:]

            seqlens_q = (extend_counts + K + 1).astype(mx.int32)
            cu_seqlens_q = mx.concatenate([mx.array([0], dtype=mx.int32), mx.cumsum(seqlens_q)])
            mx.eval(cu_seqlens_q, extend_counts)
            total_real = int(cu_seqlens_q[-1].item())

            fused_ids = mx.zeros((total_real,), dtype=mx.int64)
            fused_hs = mx.zeros((total_real, hidden_size))

            for b in range(B):
                start = int(cu_seqlens_q[b].item())
                n_ext = int(extend_counts[b].item())

                for j in range(n_ext):
                    if extend_eagle_acts_batch is not None:
                        act = extend_eagle_acts_batch[b, j]
                        fused_hs[start + j] = self.model.fc(mx.expand_dims(act, 0)).squeeze(0)
                        fused_ids[start + j] = extend_token_ids_batch[b, j]

                rec_act = target_acts[b]
                fused_hs[start + n_ext] = self.model.fc(mx.expand_dims(rec_act, 0)).squeeze(0)
                fused_ids[start + n_ext] = rec_tok_ids[b]

                for j in range(K):
                    fused_ids[start + n_ext + 1 + j] = spec_tok_ids[b, j]
                    fused_hs[start + n_ext + 1 + j] = prev_acts[b, j]

            glue_decode_ctxt = self.prepare_glue_decode_ctxt_eagle(
                num_tokens=partial_tree_decode_args["num_tokens"],
                fused_ids=fused_ids, fused_hs=fused_hs,
                extend_counts=extend_counts, seqlens_q=seqlens_q,
                cu_seqlens_q=cu_seqlens_q, dbt=dbt, B=B,
            )
        else:
            B = glue_decode_input_ids.shape[0] // (K + 1)
            glue_decode_ctxt = self.prepare_glue_decode_ctxt(
                num_tokens=partial_tree_decode_args["num_tokens"],
                input_ids=glue_decode_input_ids,
                dbt=dbt, B=B,
            )

        MQ_LEN = self.config.MQ_LEN
        b_flat_parts = []
        for b in range(B):
            b_flat_parts.append(mx.full((MQ_LEN,), b, dtype=mx.int64))
        b_flat = mx.concatenate(b_flat_parts)
        fkp1_flat = mx.tile(self._arange_mq, (B,))
        j_idx_parts = []
        for hit in cache_hits_list:
            j_idx_parts.append(self._fan_idx_hit if int(hit) else self._fan_idx_miss)
        j_idx_flat = mx.concatenate(j_idx_parts)
        N_pre = b_flat.shape[0]

        num_tokens_arr = partial_tree_decode_args["num_tokens"]
        seq_ids = partial_tree_decode_args["seq_ids"]
        seq_ids_expanded = seq_ids[b_flat]
        positions_pre = (num_tokens_arr[b_flat] - 1 + pos_offset) + (K + 1) + fkp1_flat
        rope_positions_pre = (num_tokens_arr[b_flat] - 1 + pos_offset) + j_idx_flat + 1
        temperatures_pre = partial_tree_decode_args["temperatures"][b_flat]

        set_context(
            is_prefill=False,
            cu_seqlens_q=glue_decode_ctxt["cu_seqlens_q"],
            max_seqlen_q=glue_decode_ctxt["max_seqlen_q"],
            slot_mapping=glue_decode_ctxt["slot_map"],
            context_lens=glue_decode_ctxt["context_lens"],
            block_tables=glue_decode_ctxt["block_tables"],
        )

        glue_prenorm = None
        if self.config.use_eagle:
            fused_hs_flat = glue_decode_ctxt["hidden_states"]
            glue_decode_logits_flat, glue_prenorm = self.run_model(
                glue_decode_ctxt["input_ids"], glue_decode_ctxt["positions"],
                is_prefill=False, last_only=False, hidden_states=fused_hs_flat)
        else:
            glue_decode_logits_flat = self.run_model(
                glue_decode_ctxt["input_ids"], glue_decode_ctxt["positions"],
                is_prefill=False, last_only=False)

        reset_context()
        mx.eval(glue_decode_logits_flat)

        if self.config.use_eagle:
            cu_q = glue_decode_ctxt["cu_seqlens_q"]
            mx.eval(cu_q)
            extract_indices = []
            for b in range(B):
                rec_off = int(cu_q[b].item()) + int(extend_counts[b].item())
                for j in range(K + 1):
                    extract_indices.append(rec_off + j)
            flat_idx = mx.array(extract_indices, dtype=mx.int32)
            glue_decode_logits = glue_decode_logits_flat[flat_idx].reshape(B, K + 1, -1)
            if glue_prenorm is not None:
                glue_prenorm_kp1 = glue_prenorm[flat_idx].reshape(B, K + 1, -1)
        else:
            glue_decode_logits = glue_decode_logits_flat.reshape(B, K + 1, -1)
            if glue_prenorm is not None:
                glue_prenorm_kp1 = glue_prenorm.reshape(B, K + 1, -1)

        tree_hidden_states = None
        if glue_prenorm is not None:
            fan_hit = self.config.fan_out_t
            fan_miss = self.config.fan_out_t_miss
            parts = []
            for b in range(B):
                hit = bool(cache_hits[b].item())
                fan = fan_hit if hit else fan_miss
                for d in range(K + 1):
                    f = int(fan[d].item())
                    parts.append(mx.tile(mx.expand_dims(glue_prenorm_kp1[b, d], 0), (f, 1)))
            tree_hidden_states = mx.concatenate(parts, axis=0)

        if self.config.use_eagle:
            gd_for_fork = gd_view
        else:
            gd_for_fork = glue_decode_input_ids.reshape(B, K + 1)

        forked_rec_tokens = get_forked_recovery_tokens_from_logits(
            self.config,
            glue_decode_logits,
            cache_hits,
            gd_for_fork,
            tokenizer=self.tokenizer,
        ).reshape(-1)

        tree_decode_args = {
            "metadata_ints": (B, K, self.config.async_fan_out, N_pre),
            "input_ids": forked_rec_tokens,
            "positions": positions_pre,
            "rope_positions": rope_positions_pre,
            "block_tables": dbt,
            "temps": temperatures_pre,
            "rec_flat": forked_rec_tokens,
            "seq_ids_expanded": seq_ids_expanded,
            "cache_hits": cache_hits,
            "cache_hits_list": cache_hits_list,
        }
        tree_decode_args["hidden_states"] = tree_hidden_states
        return tree_decode_args

    def _decode_tree(self, payload):
        B, K, F, N = payload["metadata_ints"]
        V = self.hf_config.vocab_size
        spec_tokens = mx.zeros((N, K), dtype=mx.int64)
        spec_logits = mx.zeros((N, K, V))
        spec_activations = mx.zeros(
            (N, K, self.hf_config.hidden_size)) if self.config.use_eagle else None

        initial_positions = payload["positions"]
        initial_rope_positions = payload["rope_positions"]
        current_input_ids = payload["input_ids"]
        dbt = payload["block_tables"]

        MQ_LEN = self.config.MQ_LEN
        step_positions = mx.expand_dims(initial_positions, 0) + self._step_pos_offsets
        step_rope_positions = mx.expand_dims(initial_rope_positions, 0) + self._step_rope_offsets
        step_context_lens = step_positions.reshape(K, B, MQ_LEN)[:, :, -1] + 1

        b_flat_parts = []
        for b in range(B):
            b_flat_parts.append(mx.full((MQ_LEN,), b, dtype=mx.int64))
        b_flat = mx.concatenate(b_flat_parts)

        mx.eval(step_positions)
        step_slot_maps = mx.zeros((K, N), dtype=mx.int32)
        for depth in range(K):
            slots_list = []
            for i in range(N):
                pos = int(step_positions[depth, i].item())
                b = int(b_flat[i].item())
                block_i = pos // self.block_size
                offset = pos % self.block_size
                block_id = int(dbt[b, block_i].item())
                slots_list.append(block_id * self.block_size + offset)
            step_slot_maps[depth] = mx.array(slots_list, dtype=mx.int32)

        all_greedy = bool((payload["temps"] == 0).all().item())

        for depth in range(K):
            set_context(
                is_prefill=False,
                slot_mapping=step_slot_maps[depth],
                context_lens=step_context_lens[depth].astype(mx.int32),
                block_tables=dbt,
            )

            hidden_states = payload.get("hidden_states")
            if self.config.use_eagle:
                logits, prenorm = self.run_model(
                    current_input_ids, step_rope_positions[depth],
                    is_prefill=False, last_only=False,
                    tree_decode_step=depth, cache_hits=payload["cache_hits"],
                    hidden_states=hidden_states)
                spec_activations[:, depth] = prenorm
                payload["hidden_states"] = prenorm
            else:
                logits = self.run_model(
                    current_input_ids, step_rope_positions[depth],
                    is_prefill=False, last_only=False,
                    tree_decode_step=depth, cache_hits=payload["cache_hits"])

            reset_context()
            mx.eval(logits)

            logits_flat = logits.reshape(-1, V)
            spec_logits[:, depth, :] = logits_flat

            if all_greedy:
                next_tokens = mx.argmax(logits_flat, axis=-1)
            else:
                next_tokens = self.sampler(logits_flat, payload["temps"], is_tree=True)

            spec_tokens[:, depth] = next_tokens
            current_input_ids = next_tokens

        return spec_tokens, spec_logits, spec_activations

    def _populate_tree_cache(self, payload, tokens, logits, cache_hits, activations=None):
        seq_ids_expanded = payload["seq_ids_expanded"].astype(mx.int64)
        rec_flat = payload["rec_flat"].astype(mx.int64)

        k_parts = []
        for hit in payload["cache_hits_list"]:
            k_parts.append(self._fan_idx_hit if int(hit) else self._fan_idx_miss)
        k_flat = mx.concatenate(k_parts)

        keys = mx.stack([seq_ids_expanded, k_flat, rec_flat], axis=1)

        self.tree_cache_keys = keys
        self.tree_cache_tokens = tokens
        self.tree_cache_logits = logits
        self.tree_cache_activations = activations

    def speculate(
        self,
        seqs,
        request_keys: mx.array,
        num_tokens: mx.array,
        temperatures: mx.array,
        draft_block_tables: mx.array,
        target_recovery_activations: mx.array = None,
        extend_counts: mx.array = None,
        extend_eagle_acts: mx.array = None,
        extend_token_ids: mx.array = None,
    ):
        B = request_keys.shape[0]
        K = self.config.speculate_k

        out_tokens, out_logits, glue_decode_input_ids, cache_hits, out_activations = \
            self.hit_cache_and_respond(
                request_keys, B, K, num_tokens, temperatures,
                draft_block_tables, target_recovery_activations,
            )

        self._reset_tree_cache_tensors()

        seq_ids = mx.array([seq.seq_id for seq in seqs], dtype=mx.int64) if seqs else mx.arange(B, dtype=mx.int64)

        partial_tree_decode_args = {
            "num_tokens": num_tokens,
            "seq_ids": seq_ids,
            "temperatures": temperatures,
            "dbt": draft_block_tables,
            "cache_hits": cache_hits,
            "returned_tokens": out_tokens,
            "target_recovery_activations": target_recovery_activations,
            "previous_activations": out_activations,
            "extend_counts": extend_counts,
            "extend_eagle_acts": extend_eagle_acts,
            "extend_token_ids": extend_token_ids,
        }

        tree_decode_args = self._build_tree_batch(partial_tree_decode_args, glue_decode_input_ids)
        tokens, logits, activations = self._decode_tree(tree_decode_args)
        self._populate_tree_cache(tree_decode_args, tokens, logits, tree_decode_args["cache_hits"], activations)

        return out_tokens, out_logits, cache_hits
