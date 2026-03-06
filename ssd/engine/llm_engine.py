import ssd.paths  # noqa: F401

from ssd.config import Config
from ssd.sampling_params import SamplingParams
from ssd.utils.misc import infer_model_family
from ssd.engine.sequence import Sequence
from ssd.engine.scheduler import Scheduler
from ssd.engine.model_runner import ModelRunner
from ssd.engine.draft_runner import DraftRunner
from ssd.engine.speculator_unified import SpeculatorUnified
from ssd.engine.step import InferenceStep, AutoRegressiveStep, SpecDecodeStep
from ssd.engine.verifier import Verifier

from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer


METRICS = {
    "cache_hits": [],
    "accepted_suffix_lens_with_recovery": [],
    "accepted_suffix_lens_on_hit": [],
    "accepted_suffix_lens_on_miss": [],
    "prefill_total_time": 0,
    "decode_total_time": 0,
    "prefill_total_tokens": 0,
    "decode_total_tokens": 0,
    "target_step_times": [],
    "target_verify_times": [],
}


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        Sequence.block_size = config.kvcache_block_size

        assert config.kvcache_block_size >= (
            2 * config.speculate_k + 2), "ERROR: support for block size < 2*k+2 is not implemented"

        if config.speculate:
            target_family = infer_model_family(config.model)
            draft_family = infer_model_family(config.draft)
            assert target_family == draft_family, "ERROR: target and draft model families must match"

        self.model_runner = ModelRunner(config, is_draft=False)

        if config.speculate:
            self.draft_runner = DraftRunner(config)
            self.draft_cfg = self.draft_runner.draft_cfg
        else:
            self.draft_cfg = None

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config, draft_cfg=self.draft_cfg)

        print(f"[LLMEngine] finished llm_engine init", flush=True)

    def exit(self):
        pass

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self, step: InferenceStep):
        t = perf_counter()
        seqs, is_prefill = self.scheduler.schedule()
        ttl_tokens = step.prefill(seqs) if is_prefill else step.decode(seqs)

        time_taken = perf_counter() - t

        if is_prefill:
            METRICS["prefill_total_time"] += time_taken
            METRICS["prefill_total_tokens"] += ttl_tokens
        else:
            METRICS["decode_total_time"] += time_taken
            METRICS["decode_total_tokens"] += ttl_tokens

        outputs = [(seq.seq_id, seq.completion_token_ids)
                   for seq in seqs if seq.is_finished]

        return outputs

    def is_finished(self):
        return self.scheduler.is_finished()

    def log_metrics(self):
        if METRICS["prefill_total_time"] > 0:
            avg_prefill_throughput = METRICS["prefill_total_tokens"] / METRICS["prefill_total_time"]
            print(f"Final Prefill Throughput: {int(avg_prefill_throughput)}tok/s", flush=True)
        if METRICS["decode_total_time"] > 0:
            avg_decode_throughput = METRICS["decode_total_tokens"] / METRICS["decode_total_time"]
            print(f"Final Decode Throughput: {int(avg_decode_throughput)}tok/s", flush=True)

        if self.config.speculate and METRICS['accepted_suffix_lens_with_recovery']:
            ttl_accepted_with_recovery = sum(METRICS['accepted_suffix_lens_with_recovery'])
            ttl_num_spec_steps = len(METRICS['accepted_suffix_lens_with_recovery'])
            avg_tokens_per_step = ttl_accepted_with_recovery / ttl_num_spec_steps
            print(f"[metrics] Avg Tokens per step (incl recovery): {avg_tokens_per_step:.2f}", flush=True)

            total_accepted = ttl_accepted_with_recovery - ttl_num_spec_steps
            avg_acceptance_rate = (total_accepted / ttl_num_spec_steps) / self.config.speculate_k
            print(f"[metrics] Avg Fraction of Speculated Tokens Accepted: {avg_acceptance_rate:.2f}", flush=True)

            if METRICS['target_step_times']:
                print(f"[metrics] Avg target time per full step (ms): {sum(METRICS['target_step_times']) * 1000 / len(METRICS['target_step_times']):.2f}", flush=True)
            if METRICS['target_verify_times']:
                print(f"[metrics] Avg target verify time (ms): {sum(METRICS['target_verify_times']) * 1000 / len(METRICS['target_verify_times']):.2f}", flush=True)

            if self.config.draft_async and METRICS['cache_hits']:
                print(f"[metrics] Avg Cache Hits: {sum(METRICS['cache_hits']) / len(METRICS['cache_hits']):.2f}", flush=True)
                if METRICS['accepted_suffix_lens_on_hit']:
                    avg_suffix_len_on_hit = sum(METRICS['accepted_suffix_lens_on_hit']) / len(METRICS['accepted_suffix_lens_on_hit'])
                    print(f"[metrics] Avg Tokens per step on Cache Hit: {avg_suffix_len_on_hit:.2f}", flush=True)

                    adjusted_lens = [length - 1 for length in METRICS['accepted_suffix_lens_on_hit']]
                    total_count = len(adjusted_lens)
                    freq_counts = {}
                    for length in adjusted_lens:
                        freq_counts[length] = freq_counts.get(length, 0) + 1

                    print(f"[metrics] Empirical frequencies of accepted_suffix_lens_on_hit - 1:", flush=True)
                    for k in range(self.config.speculate_k + 1):
                        prob = freq_counts.get(k, 0) / total_count
                        print(f"  {k}: {prob:.3f}", flush=True)

                if METRICS['accepted_suffix_lens_on_miss']:
                    avg_suffix_len_on_miss = sum(METRICS['accepted_suffix_lens_on_miss']) / len(METRICS['accepted_suffix_lens_on_miss'])
                    print(f"[metrics] Avg Tokens per step on Cache Miss: {avg_suffix_len_on_miss:.2f}", flush=True)

    def create_inference_step(self, config: Config) -> InferenceStep:
        if config.speculate:
            speculator = SpeculatorUnified(
                lookahead=config.speculate_k,
                draft_runner=self.draft_runner,
                config=config,
                tokenizer=self.tokenizer,
            )

            verifier = Verifier(
                lookahead=config.speculate_k,
                target_model_runner=self.model_runner,
                sampler_x=config.sampler_x,
                async_fan_out=config.async_fan_out,
                jit_speculate=config.jit_speculate,
                tokenizer=self.tokenizer,
                metrics=METRICS,
            )
            return SpecDecodeStep(
                scheduler=self.scheduler,
                speculator=speculator,
                verifier=verifier,
                eagle=config.use_eagle,
                tokenizer=self.tokenizer,
                async_spec=config.draft_async,
            )
        else:
            return AutoRegressiveStep(
                scheduler=self.scheduler,
                model_runner=self.model_runner,
                tokenizer=self.tokenizer,
            )

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        stream_callback=None,
    ) -> list[str]:
        for k in METRICS:
            METRICS[k] = [] if isinstance(METRICS[k], list) else 0

        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}
        inference_step = self.create_inference_step(self.config)
        i = 0
        max_steps = self.config.max_steps if self.config.max_steps is not None else float('inf')
        _stream_lens = {}
        while not self.is_finished() and i < max_steps:
            i += 1
            t = perf_counter()
            output = self.step(inference_step)
            time_taken = perf_counter() - t
            METRICS["target_step_times"].append(time_taken)

            if stream_callback:
                for seq in self.scheduler.running:
                    cur = seq.num_completion_tokens
                    prev = _stream_lens.get(seq.seq_id, 0)
                    if cur > prev:
                        stream_callback(seq.seq_id, seq.completion_token_ids[prev:cur])
                        _stream_lens[seq.seq_id] = cur

            for seq_id, token_ids in output:
                if stream_callback:
                    prev = _stream_lens.get(seq_id, 0)
                    if len(token_ids) > prev:
                        stream_callback(seq_id, token_ids[prev:])
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()

        if not stream_callback:
            self.log_metrics()

        return outputs, METRICS
