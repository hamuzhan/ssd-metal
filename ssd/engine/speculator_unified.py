import mlx.core as mx
from transformers import AutoTokenizer

from ssd.engine.helpers.speculate_types import SpeculateResult, VerifyResult, SpeculatorBase
from ssd.engine.sequence import Sequence
from ssd.engine.draft_runner import DraftRunner
from ssd.utils.misc import decode_tokens


class SpeculatorUnified(SpeculatorBase):

    def __init__(
        self,
        lookahead: int,
        draft_runner: DraftRunner,
        config,
        tokenizer: AutoTokenizer,
    ):
        super().__init__(lookahead)
        self.draft_runner = draft_runner
        self.config = config
        self.tokenizer = tokenizer
        self.K = lookahead
        self.verbose = config.verbose

    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        eagle_acts = verify_result.eagle_acts

        if eagle_acts is not None:
            input_id_list = [seq.token_ids for seq in seqs]
            sliced = []
            offset = 0
            for ids in input_id_list:
                seq_len = len(ids)
                sliced.append(eagle_acts[offset:offset + seq_len - 1])
                offset += seq_len
            eagle_acts = mx.concatenate(sliced, axis=0)

        self.draft_runner.draft_prefill(seqs, eagle_acts=eagle_acts)
        return SpeculateResult([], [])

    def speculate(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        for seq in seqs:
            assert seq.recovery_token_id is not None
            seq.append_token(seq.recovery_token_id)

        B = len(seqs)
        eagle = verify_result.eagle_acts is not None

        if self.config.draft_async:
            return self._speculate_async(seqs, verify_result, eagle)
        else:
            return self._speculate_sync(seqs, verify_result)

    def _speculate_sync(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        B = len(seqs)
        speculations = mx.zeros((B, self.K + 1), dtype=mx.int64)
        logits_q = []

        recovery_tokens = mx.array([seq.recovery_token_id for seq in seqs], dtype=mx.int64)
        speculations[:, 0] = recovery_tokens

        for k in range(self.K + 1):
            token_ids, step_logits_q = self.draft_runner.run(seqs, False, True, True)
            for s in seqs:
                s.num_draft_cached_tokens += 1

            if k == self.K:
                break

            logits_q.append(step_logits_q)

            for i, (seq, token_id) in enumerate(zip(seqs, token_ids)):
                seq.append_token(token_id)

            speculations[:, k + 1] = mx.array(token_ids, dtype=mx.int64)

        logits_q = mx.stack(logits_q, axis=1)
        return SpeculateResult(speculations, logits_q)

    def _speculate_async(self, seqs: list[Sequence], verify_result: VerifyResult, eagle: bool) -> SpeculateResult:
        B = len(seqs)
        K = self.K

        cache_keys = mx.zeros((B, 3), dtype=mx.int64)
        num_tokens = mx.zeros((B,), dtype=mx.int64)
        temperatures = mx.zeros((B,), dtype=mx.float32)
        max_blocks = self.config.max_blocks
        block_tables = mx.full((B, max_blocks), -1, dtype=mx.int32)

        target_recovery_activations = None
        extend_counts = None
        extend_eagle_acts = None
        extend_token_ids = None

        for i, seq in enumerate(seqs):
            cache_keys[i, 0] = seq.seq_id
            cache_keys[i, 1] = seq.last_spec_step_accepted_len - 1
            cache_keys[i, 2] = seq.recovery_token_id
            num_tokens[i] = seq.num_tokens
            temperatures[i] = seq.draft_temperature if seq.draft_temperature is not None else seq.temperature
            bt = seq.draft_block_table
            bt_len = len(bt)
            if bt_len > 0:
                block_tables[i, :bt_len] = mx.array(bt, dtype=mx.int32)

        if eagle:
            target_recovery_activations = mx.stack(
                [seq.last_target_hidden_state for seq in seqs], axis=0)

            act_dim = target_recovery_activations.shape[-1]
            extend_counts = mx.zeros((B,), dtype=mx.int64)
            extend_eagle_acts = mx.zeros((B, K, act_dim))
            extend_token_ids = mx.zeros((B, K), dtype=mx.int64)

            for i, seq in enumerate(seqs):
                extend_counts[i] = seq.extend_count
                n = seq.extend_count
                if n > 0 and seq.extend_eagle_acts is not None:
                    extend_eagle_acts[i, :n] = seq.extend_eagle_acts[:n]
                    extend_token_ids[i, :n] = seq.extend_token_ids[:n]

        out_tokens, out_logits, cache_hits = self.draft_runner.speculate(
            seqs,
            request_keys=cache_keys,
            num_tokens=num_tokens,
            temperatures=temperatures,
            draft_block_tables=block_tables,
            target_recovery_activations=target_recovery_activations,
            extend_counts=extend_counts,
            extend_eagle_acts=extend_eagle_acts,
            extend_token_ids=extend_token_ids,
        )

        speculations = mx.zeros((B, K + 1), dtype=mx.int64)
        speculations[:, 0] = mx.array([seq.recovery_token_id for seq in seqs], dtype=mx.int64)
        speculations[:, 1:] = out_tokens

        for i, seq in enumerate(seqs):
            mx.eval(out_tokens)
            seq.token_ids.extend(out_tokens[i].tolist())
            seq.num_tokens = len(seq.token_ids)
            seq.last_token = seq.token_ids[-1]
            seq.num_draft_cached_tokens += len(out_tokens[i].tolist()) + 1

        return SpeculateResult(speculations, out_logits, cache_hits)
