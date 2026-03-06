from dataclasses import dataclass
import mlx.core as mx
from ssd.engine.sequence import Sequence
from abc import ABC, abstractmethod


@dataclass
class SpeculateResult:
    speculations: mx.array
    logits_q: mx.array
    cache_hits: mx.array | None = None


@dataclass
class VerifyResult:
    new_suffixes: list[list[int]]
    recovery_tokens: list[int]
    eagle_acts: mx.array | None = None


class SpeculatorBase(ABC):
    def __init__(self, lookahead: int):
        self.lookahead = lookahead

    @abstractmethod
    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        pass

    @abstractmethod
    def speculate(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        pass


class VerifierBase(ABC):
    def __init__(self, lookahead: int):
        self.lookahead = lookahead

    @abstractmethod
    def prefill(self, seqs: list[Sequence], eagle: bool = False) -> VerifyResult:
        pass

    @abstractmethod
    def verify(self, seqs: list[Sequence], speculate_result: SpeculateResult, eagle: bool = False) -> VerifyResult:
        pass
