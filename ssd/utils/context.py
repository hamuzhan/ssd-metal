from dataclasses import dataclass
import mlx.core as mx


@dataclass
class Context:
    is_prefill: bool = False
    is_jit: bool = False
    cu_seqlens_q: mx.array | None = None
    cu_seqlens_k: mx.array | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: mx.array | None = None
    context_lens: mx.array | None = None
    block_tables: mx.array | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, is_jit=False):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, is_jit, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
