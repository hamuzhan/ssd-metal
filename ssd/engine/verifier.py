import mlx.core as mx
from time import perf_counter
from transformers import AutoTokenizer

from ssd.engine.sequence import Sequence
from ssd.engine.model_runner import ModelRunner
from ssd.utils.verify import verify
from ssd.engine.helpers.speculate_types import SpeculateResult, VerifyResult, VerifierBase


class Verifier(VerifierBase):
    def __init__(
        self,
        lookahead: int,
        target_model_runner: ModelRunner,
        sampler_x: float | None = None,
        async_fan_out: int | None = None,
        jit_speculate: bool = False,
        tokenizer: AutoTokenizer = None,
        metrics: dict = None,
    ):
        super().__init__(lookahead)
        self.target_model_runner = target_model_runner
        self.sampler_x = sampler_x
        self.async_fan_out = async_fan_out
        self.jit_speculate = jit_speculate
        self.tokenizer = tokenizer
        self.metrics = metrics

    def prefill(self, seqs: list[Sequence], eagle: bool = False) -> VerifyResult:
        result = self.target_model_runner.run(seqs, True)
        if eagle:
            token_ids, eagle_acts = result
        else:
            token_ids = result

        offset = 0
        for seq, token_id in zip(seqs, token_ids):
            seq.recovery_token_id = token_id
            if eagle:
                seq_len = seq.num_prompt_tokens
                eagle_acts_eval = eagle_acts[offset + seq_len - 1]
                seq.last_target_hidden_state = mx.array(eagle_acts_eval)
                offset += seq_len

        return VerifyResult(
            [],
            [seq.recovery_token_id for seq in seqs],
            eagle_acts if eagle else None,
        )

    def verify(self, seqs: list[Sequence], speculate_result: SpeculateResult, eagle: bool = False) -> VerifyResult:
        batch_size = len(seqs)

        _tv0 = perf_counter()
        result = self.target_model_runner.run(seqs, False, False, True)

        if eagle:
            logits_p_flat, eagle_acts_flat = result
        else:
            logits_p_flat = result

        for s in seqs:
            s.num_cached_tokens += self.lookahead + 1

        logits_p = logits_p_flat.reshape(batch_size, self.lookahead + 1, -1)

        temps_target = [seq.temperature for seq in seqs]
        temps_draft = [
            seq.draft_temperature if seq.draft_temperature is not None else seq.temperature
            for seq in seqs
        ]
        temperatures_target = mx.array(temps_target, dtype=mx.float32)
        temperatures_draft = mx.array(temps_draft, dtype=mx.float32)

        new_suffixes, recovery_tokens = verify(
            logits_p=logits_p,
            logits_q=speculate_result.logits_q,
            speculations=speculate_result.speculations,
            temperatures_target=temperatures_target,
            temperatures_draft=temperatures_draft,
            cache_hits=speculate_result.cache_hits,
            sampler_x=self.sampler_x,
            async_fan_out=self.async_fan_out,
            jit_speculate=self.jit_speculate,
        )

        self.metrics["target_verify_times"].append(perf_counter() - _tv0)

        self.metrics["accepted_suffix_lens_with_recovery"].extend(
            [len(s) for s in new_suffixes])

        if speculate_result.cache_hits is not None:
            ch = speculate_result.cache_hits
            mx.eval(ch)
            ch_float = ch.astype(mx.float32)
            self.metrics["cache_hits"].append(float(ch_float.mean().item()))
            for i, suffix_len in enumerate([len(s) for s in new_suffixes]):
                if int(ch[i].item()) == 1:
                    self.metrics["accepted_suffix_lens_on_hit"].append(suffix_len)
                else:
                    self.metrics["accepted_suffix_lens_on_miss"].append(suffix_len)

        eagle_acts = None
        if eagle:
            eagle_acts = eagle_acts_flat.reshape(batch_size, self.lookahead + 1, -1)

        return VerifyResult(
            new_suffixes=new_suffixes,
            recovery_tokens=recovery_tokens,
            eagle_acts=eagle_acts,
        )
