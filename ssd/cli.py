"""
SSD-Metal Interactive CLI
Full-screen terminal interface (like htop) with arrow-key navigation.

Usage:
    python -m ssd              # Interactive TUI
    python -m ssd --install    # Direct install (no TUI)
"""

import curses
import locale
import os
import re
import sys
import time
import platform
import subprocess
import threading
import shutil
from dataclasses import dataclass, field


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str = ""
    error: str = ""
    skipped: bool = False
    time_ms: float = 0.0


@dataclass
class PhaseResult:
    name: str
    tests: list[TestResult] = field(default_factory=list)

    @property
    def passed(self):
        return sum(1 for t in self.tests if t.passed and not t.skipped)

    @property
    def failed(self):
        return sum(1 for t in self.tests if not t.passed and not t.skipped)

    @property
    def skipped(self):
        return sum(1 for t in self.tests if t.skipped)


# ─── ANSI Styles (for non-curses / installer output) ─────────────────────────

class Style:
    RESET      = "\033[0m"
    BOLD       = "\033[1m"
    DIM        = "\033[2m"
    ITALIC     = "\033[3m"
    UNDERLINE  = "\033[4m"

    BLACK      = "\033[30m"
    RED        = "\033[31m"
    GREEN      = "\033[32m"
    YELLOW     = "\033[33m"
    BLUE       = "\033[34m"
    MAGENTA    = "\033[35m"
    CYAN       = "\033[36m"
    WHITE      = "\033[37m"

    BRIGHT_RED     = "\033[91m"
    BRIGHT_GREEN   = "\033[92m"
    BRIGHT_YELLOW  = "\033[93m"
    BRIGHT_BLUE    = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN    = "\033[96m"
    BRIGHT_WHITE   = "\033[97m"

    BG_RED     = "\033[41m"
    BG_GREEN   = "\033[42m"
    BG_YELLOW  = "\033[43m"
    BG_BLUE    = "\033[44m"

    CHECK  = "\033[92m\u2713\033[0m"
    CROSS  = "\033[91m\u2717\033[0m"
    WARN   = "\033[93m\u26a0\033[0m"
    BULLET = "\033[36m\u25cf\033[0m"
    ARROW  = "\033[36m\u2192\033[0m"


# ─── Print Banner (stdout, for installer / non-curses) ────────────────────────

BANNER_LINES = [
    " \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557\u2588\u2588\u2588\u2588\u2588\u2588\u2557       \u2588\u2588\u2588\u2557   \u2588\u2588\u2588\u2557\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2557",
    " \u2588\u2588\u2554\u2550\u2550\u2550\u2550\u255d\u2588\u2588\u2554\u2550\u2550\u2550\u2550\u255d\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557      \u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2588\u2588\u2551\u2588\u2588\u2554\u2550\u2550\u2550\u2550\u255d\u255a\u2550\u2550\u2588\u2588\u2554\u2550\u2550\u255d\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2551",
    " \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557\u2588\u2588\u2551  \u2588\u2588\u2551\u2588\u2588\u2588\u2588\u2588\u2557\u2588\u2588\u2554\u2588\u2588\u2588\u2588\u2554\u2588\u2588\u2551\u2588\u2588\u2588\u2588\u2588\u2557     \u2588\u2588\u2551   \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2551\u2588\u2588\u2551",
    " \u255a\u2550\u2550\u2550\u2550\u2588\u2588\u2551\u255a\u2550\u2550\u2550\u2550\u2588\u2588\u2551\u2588\u2588\u2551  \u2588\u2588\u2551\u255a\u2550\u2550\u2550\u2550\u255d\u2588\u2588\u2551\u255a\u2588\u2588\u2554\u255d\u2588\u2588\u2551\u2588\u2588\u2554\u2550\u2550\u255d     \u2588\u2588\u2551   \u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2551\u2588\u2588\u2551",
    " \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2551\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2551\u2588\u2588\u2588\u2588\u2588\u2588\u2554\u255d      \u2588\u2588\u2551 \u255a\u2550\u255d \u2588\u2588\u2551\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557   \u2588\u2588\u2551   \u2588\u2588\u2551  \u2588\u2588\u2551\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557",
    " \u255a\u2550\u2550\u2550\u2550\u2550\u2550\u255d\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u255d\u255a\u2550\u2550\u2550\u2550\u2550\u255d       \u255a\u2550\u255d     \u255a\u2550\u255d\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u255d   \u255a\u2550\u255d   \u255a\u2550\u255d  \u255a\u2550\u255d\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u255d",
]


def print_banner():
    """Print ASCII banner to stdout (non-curses mode)."""
    print()
    for line in BANNER_LINES:
        print(f"  {Style.BRIGHT_CYAN}{Style.BOLD}{line}{Style.RESET}")
    print()
    print(f"  {Style.DIM}{'─' * 68}{Style.RESET}")
    subtitle = "Speculative Decoding Engine for Apple Silicon"
    print(f"  {Style.BRIGHT_WHITE}{Style.BOLD}{subtitle:^68}{Style.RESET}")
    version_line = "v0.3.0"
    print(f"  {Style.DIM}{version_line:^68}{Style.RESET}")
    print(f"  {Style.DIM}{'─' * 68}{Style.RESET}")
    print()


# ─── Hardware Info ────────────────────────────────────────────────────────────

class HardwareInfo:

    @staticmethod
    def _sysctl(key: str) -> str:
        try:
            return subprocess.check_output(
                ["sysctl", "-n", key], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return ""

    @staticmethod
    def gather() -> list[tuple[str, str]]:
        rows = []

        chip = HardwareInfo._sysctl("machdep.cpu.brand_string")
        if not chip:
            chip = platform.processor() or "Unknown"
        rows.append(("Chip", chip))

        phys = HardwareInfo._sysctl("hw.physicalcpu")
        perf = HardwareInfo._sysctl("hw.perflevel0.physicalcpu")
        eff = HardwareInfo._sysctl("hw.perflevel1.physicalcpu")
        if phys:
            core_str = f"{phys} cores"
            if perf and eff:
                core_str += f" ({perf}P + {eff}E)"
            rows.append(("CPU", core_str))

        try:
            mem_bytes = int(HardwareInfo._sysctl("hw.memsize"))
            mem_gb = mem_bytes / (1024 ** 3)
            rows.append(("Memory", f"{mem_gb:.0f} GB"))
        except (ValueError, TypeError):
            rows.append(("Memory", "Unknown"))

        mac_ver = platform.mac_ver()[0]
        if mac_ver:
            rows.append(("macOS", mac_ver))

        chip_lower = chip.lower()
        if any(f"m{n}" in chip_lower for n in ("4", "3", "2", "1")):
            rows.append(("Neural Engine", "16-core"))
        elif "apple" in chip_lower:
            rows.append(("Neural Engine", "Available"))

        rows.append(("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"))

        try:
            import mlx.core as mx
            ver = getattr(mx, "__version__", None)
            if not ver:
                import mlx
                ver = getattr(mlx, "__version__", None)
            rows.append(("MLX", f"v{ver}" if ver else "installed"))
        except ImportError:
            rows.append(("MLX", "Not installed"))
        except Exception:
            rows.append(("MLX", "Error"))

        try:
            import mlx.core as mx
            if hasattr(mx, "device_info"):
                info = mx.device_info()
            else:
                info = mx.metal.device_info()
            metal_mem = info.get("memory_size", 0)
            if metal_mem:
                rows.append(("Metal Memory", f"{metal_mem / (1024**3):.0f} GB"))
            rec_ws = info.get("recommended_max_working_set_size", 0)
            if rec_ws:
                rows.append(("Metal Rec. WS", f"{rec_ws / (1024**3):.0f} GB"))
            arch = info.get("architecture", "")
            if arch:
                rows.append(("GPU Arch", arch))
        except Exception:
            pass

        return rows


# ─── Test Runner ──────────────────────────────────────────────────────────────

class TestRunner:
    """Defines and runs diagnostic tests. Returns results without printing."""

    PHASES = [
        (1, "Core Dependencies"),
        (2, "Foundation Modules"),
        (3, "Layer Primitives"),
        (4, "Kernel Functions"),
        (5, "Model Definitions"),
        (6, "Engine Components"),
        (7, "Tensor Operations"),
        (8, "Integration"),
    ]

    @staticmethod
    def run_test_silent(name: str, fn) -> TestResult:
        """Run a single test, return TestResult (no UI output)."""
        t0 = time.perf_counter()
        try:
            detail = fn()
            elapsed = (time.perf_counter() - t0) * 1000
            return TestResult(name=name, passed=True, detail=detail or "", time_ms=elapsed)
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            err_msg = str(e)
            if len(err_msg) > 150:
                err_msg = err_msg[:150] + "..."
            return TestResult(name=name, passed=False, error=err_msg, time_ms=elapsed)

    @staticmethod
    def get_tests_for_phase(phase_num: int):
        """Return (phase_name, [(test_name, test_fn), ...])."""
        methods = {
            1: ("Core Dependencies",  TestRunner._tests_core_deps),
            2: ("Foundation Modules",  TestRunner._tests_foundation),
            3: ("Layer Primitives",    TestRunner._tests_layers),
            4: ("Kernel Functions",    TestRunner._tests_kernels),
            5: ("Model Definitions",   TestRunner._tests_models),
            6: ("Engine Components",   TestRunner._tests_engine),
            7: ("Tensor Operations",   TestRunner._tests_tensor_ops),
            8: ("Integration",         TestRunner._tests_integration),
        }
        name, fn = methods[phase_num]
        return name, fn()

    # ── Phase 1: Core Dependencies ─────────────────────────────

    @staticmethod
    def _tests_core_deps():
        tests = []
        deps = [
            ("mlx", "mlx"), ("transformers", "transformers"),
            ("safetensors", "safetensors"), ("xxhash", "xxhash"),
            ("numpy", "numpy"), ("tqdm", "tqdm"), ("tiktoken", "tiktoken"),
        ]
        for display_name, module_name in deps:
            def make_test(mod=module_name):
                def test():
                    m = __import__(mod)
                    ver = getattr(m, "__version__", None)
                    return f"v{ver}" if ver else "imported"
                return test
            tests.append((display_name, make_test()))
        return tests

    # ── Phase 2: Foundation Modules ────────────────────────────

    @staticmethod
    def _tests_foundation():
        tests = []

        def test_config():
            from ssd.config import Config
            assert hasattr(Config, '__post_init__'), "Config missing __post_init__"
            assert hasattr(Config, 'max_blocks'), "Config missing max_blocks"
            return "Config class OK"
        tests.append(("ssd.config", test_config))

        def test_paths():
            try:
                import ssd.paths
                return f"HF_CACHE={ssd.paths.HF_CACHE_DIR[:30]}..."
            except RuntimeError as e:
                raise RuntimeError(f"Env var not set: {e}")
        tests.append(("ssd.paths", test_paths))

        def test_sampling():
            from ssd.sampling_params import SamplingParams
            sp = SamplingParams()
            assert sp.temperature == 1.0
            assert sp.max_new_tokens == 256
            return f"T={sp.temperature}"
        tests.append(("ssd.sampling_params", test_sampling))

        def test_sequence():
            from ssd.engine.sequence import Sequence, SequenceStatus
            Sequence.block_size = 256
            seq = Sequence([1, 2, 3])
            assert len(seq) == 3
            assert seq.status == SequenceStatus.WAITING
            return f"len={len(seq)}"
        tests.append(("ssd.engine.sequence", test_sequence))

        return tests

    # ── Phase 3: Layer Primitives ──────────────────────────────

    @staticmethod
    def _tests_layers():
        tests = []
        HIDDEN, HEAD_DIM, NUM_HEADS, NUM_KV_HEADS = 64, 16, 4, 2
        VOCAB, INTER = 256, 128

        def test_rmsnorm():
            import mlx.core as mx
            from ssd.layers.layernorm import RMSNorm
            norm = RMSNorm(HIDDEN)
            x = mx.random.normal((2, HIDDEN))
            y = norm(x); mx.eval(y)
            assert y.shape == (2, HIDDEN)
            return f"RMSNorm({HIDDEN})"
        tests.append(("RMSNorm", test_rmsnorm))

        def test_silu():
            import mlx.core as mx
            from ssd.layers.activation import SiluAndMul
            act = SiluAndMul()
            x = mx.random.normal((2, INTER * 2))
            y = act(x); mx.eval(y)
            assert y.shape == (2, INTER)
            return f"in={INTER*2} out={INTER}"
        tests.append(("SiluAndMul", test_silu))

        def test_rope():
            import mlx.core as mx
            from ssd.layers.rotary_embedding import RotaryEmbedding
            rope = RotaryEmbedding(HEAD_DIM, HEAD_DIM, 512, 500000.0)
            positions = mx.array([0, 1, 2])
            q = mx.random.normal((3, NUM_HEADS * HEAD_DIM))
            k = mx.random.normal((3, NUM_KV_HEADS * HEAD_DIM))
            q_out, k_out = rope(positions, q, k)
            mx.eval(q_out, k_out)
            assert q_out.shape == q.shape
            return f"head_dim={HEAD_DIM}"
        tests.append(("RotaryEmbedding", test_rope))

        def test_qkv():
            import mlx.core as mx
            from ssd.layers.linear import QKVLinear
            qkv = QKVLinear(HIDDEN, HEAD_DIM, NUM_HEADS, NUM_KV_HEADS)
            x = mx.random.normal((2, HIDDEN))
            y = qkv(x); mx.eval(y)
            expected = (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
            assert y.shape == (2, expected)
            return f"{HIDDEN}\u2192{expected}"
        tests.append(("QKVLinear", test_qkv))

        def test_gateup():
            import mlx.core as mx
            from ssd.layers.linear import GateUpLinear
            gu = GateUpLinear(HIDDEN, INTER)
            x = mx.random.normal((2, HIDDEN))
            y = gu(x); mx.eval(y)
            assert y.shape == (2, INTER * 2)
            return f"{HIDDEN}\u2192{INTER*2}"
        tests.append(("GateUpLinear", test_gateup))

        def test_row():
            import mlx.core as mx
            from ssd.layers.linear import RowLinear
            rl = RowLinear(INTER, HIDDEN)
            x = mx.random.normal((2, INTER))
            y = rl(x); mx.eval(y)
            assert y.shape == (2, HIDDEN)
            return f"{INTER}\u2192{HIDDEN}"
        tests.append(("RowLinear", test_row))

        def test_embed():
            import mlx.core as mx
            from ssd.layers.embed_head import Embedding
            emb = Embedding(VOCAB, HIDDEN)
            emb.weight = mx.random.normal((VOCAB, HIDDEN))
            ids = mx.array([0, 5, 10])
            y = emb(ids); mx.eval(y)
            assert y.shape == (3, HIDDEN)
            return f"vocab={VOCAB} dim={HIDDEN}"
        tests.append(("Embedding", test_embed))

        def test_lmhead():
            import mlx.core as mx
            from ssd.layers.embed_head import LMHead
            from ssd.utils.context import set_context, reset_context
            lm = LMHead(VOCAB, HIDDEN)
            lm.weight = mx.random.normal((VOCAB, HIDDEN))
            set_context(
                is_prefill=True,
                cu_seqlens_q=mx.array([0, 3]),
                cu_seqlens_k=mx.array([0, 3]),
            )
            x = mx.random.normal((3, HIDDEN))
            y = lm(x, last_only=True); mx.eval(y)
            assert y.shape == (1, VOCAB)
            reset_context()
            return f"dim={HIDDEN} vocab={VOCAB}"
        tests.append(("LMHead", test_lmhead))

        def test_sampler():
            import mlx.core as mx
            from ssd.layers.sampler import Sampler
            s = Sampler()
            logits = mx.random.normal((4, VOCAB))
            temps = mx.zeros((4,))
            tokens = s(logits, temps); mx.eval(tokens)
            expected = mx.argmax(logits.astype(mx.float32), axis=-1); mx.eval(expected)
            assert all(int(tokens[i].item()) == int(expected[i].item()) for i in range(4))
            return "greedy OK"
        tests.append(("Sampler", test_sampler))

        def test_attention():
            import mlx.core as mx
            from ssd.layers.attention import Attention
            from ssd.utils.context import set_context, reset_context
            scale = HEAD_DIM ** -0.5
            attn = Attention(NUM_HEADS, HEAD_DIM, scale, NUM_KV_HEADS)
            set_context(
                is_prefill=True,
                cu_seqlens_q=mx.array([0, 4]),
                cu_seqlens_k=mx.array([0, 4]),
            )
            q = mx.random.normal((4, NUM_HEADS * HEAD_DIM))
            k = mx.random.normal((4, NUM_KV_HEADS * HEAD_DIM))
            v = mx.random.normal((4, NUM_KV_HEADS * HEAD_DIM))
            o = attn(q, k, v); mx.eval(o)
            assert o.shape == (4, NUM_HEADS * HEAD_DIM)
            reset_context()
            return f"heads={NUM_HEADS} dim={HEAD_DIM}"
        tests.append(("Attention", test_attention))

        return tests

    # ── Phase 4: Kernel Functions ──────────────────────────────

    @staticmethod
    def _tests_kernels():
        def test_kvcache_store():
            import mlx.core as mx
            from ssd.kernels.kvcache_store import store_kvcache
            D, N, SLOTS = 32, 4, 16
            k_cache = mx.zeros((SLOTS, D))
            v_cache = mx.zeros((SLOTS, D))
            key = mx.ones((N, D))
            value = mx.ones((N, D)) * 2.0
            slot_mapping = mx.array([0, 3, 7, -1])
            store_kvcache(key, value, k_cache, v_cache, slot_mapping)
            mx.eval(k_cache, v_cache)
            assert float(k_cache[0, 0].item()) == 1.0
            assert float(v_cache[3, 0].item()) == 2.0
            assert float(k_cache[7, 0].item()) == 1.0
            assert float(k_cache[15, 0].item()) == 0.0
            return f"N={N} slots={SLOTS} padding=OK"
        return [("store_kvcache", test_kvcache_store)]

    # ── Phase 5: Model Definitions ─────────────────────────────

    @staticmethod
    def _tests_models():
        tests = []

        def test_llama():
            from ssd.models.llama3 import LlamaForCausalLM
            assert hasattr(LlamaForCausalLM, 'packed_modules_mapping')
            m = LlamaForCausalLM.packed_modules_mapping
            assert "q_proj" in m or "gate_proj" in m
            return f"packed_modules={len(m)} keys"
        tests.append(("LlamaForCausalLM", test_llama))

        def test_qwen():
            from ssd.models.qwen3 import Qwen3ForCausalLM
            assert hasattr(Qwen3ForCausalLM, 'packed_modules_mapping')
            return f"packed_modules={len(Qwen3ForCausalLM.packed_modules_mapping)} keys"
        tests.append(("Qwen3ForCausalLM", test_qwen))

        def test_eagle():
            from ssd.models.eagle3_draft_llama3 import Eagle3DraftForCausalLM
            assert hasattr(Eagle3DraftForCausalLM, 'packed_modules_mapping')
            return f"packed_modules={len(Eagle3DraftForCausalLM.packed_modules_mapping)} keys"
        tests.append(("Eagle3DraftForCausalLM", test_eagle))

        return tests

    # ── Phase 6: Engine Components ─────────────────────────────

    @staticmethod
    def _tests_engine():
        tests = []
        components = [
            ("ModelRunner",       "ssd.engine.model_runner",        "ModelRunner"),
            ("DraftRunner",       "ssd.engine.draft_runner",        "DraftRunner"),
            ("Scheduler",         "ssd.engine.scheduler",           "Scheduler"),
            ("Verifier",          "ssd.engine.verifier",            "Verifier"),
            ("SpeculatorUnified", "ssd.engine.speculator_unified",  "SpeculatorUnified"),
            ("InferenceStep",     "ssd.engine.step",                "InferenceStep"),
        ]
        for display, module_path, class_name in components:
            def make_test(mod=module_path, cls=class_name):
                def test():
                    import importlib
                    m = importlib.import_module(mod)
                    klass = getattr(m, cls)
                    assert klass is not None
                    return "imported"
                return test
            tests.append((display, make_test()))
        return tests

    # ── Phase 7: Tensor Operations ─────────────────────────────

    @staticmethod
    def _tests_tensor_ops():
        tests = []

        def test_basic():
            import mlx.core as mx
            a = mx.array([1.0, 2.0, 3.0])
            b = mx.array([4.0, 5.0, 6.0])
            c = a + b; mx.eval(c)
            assert float(c[0].item()) == 5.0
            A = mx.random.normal((4, 8))
            B = mx.random.normal((8, 3))
            C = A @ B; mx.eval(C)
            assert C.shape == (4, 3)
            return "add, matmul OK"
        tests.append(("Basic tensor ops", test_basic))

        def test_sdpa():
            import mlx.core as mx
            B, H, S, D = 1, 4, 8, 16
            q = mx.random.normal((B, H, S, D))
            k = mx.random.normal((B, H, S, D))
            v = mx.random.normal((B, H, S, D))
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5)
            mx.eval(o)
            assert o.shape == (B, H, S, D)
            return f"shape={o.shape}"
        tests.append(("mx.fast.sdpa", test_sdpa))

        def test_rope_values():
            import mlx.core as mx
            from ssd.layers.rotary_embedding import RotaryEmbedding
            rope = RotaryEmbedding(16, 16, 512, 500000.0)
            positions = mx.array([5])
            q = mx.ones((1, 64))
            k = mx.ones((1, 32))
            q_out, k_out = rope(positions, q, k)
            mx.eval(q_out, k_out)
            diff = float(mx.sum(mx.abs(q_out - q)).item())
            assert diff > 0.001, f"RoPE didn't change values, diff={diff}"
            return f"diff={diff:.4f}"
        tests.append(("Rotary embedding forward", test_rope_values))

        def test_sampler_greedy():
            import mlx.core as mx
            from ssd.layers.sampler import Sampler
            s = Sampler()
            logits_np = [[0.0] * 100, [0.0] * 100]
            logits_np[0][42] = 10.0
            logits_np[1][77] = 10.0
            logits = mx.array(logits_np)
            temps = mx.zeros((2,))
            tokens = s(logits, temps); mx.eval(tokens)
            assert int(tokens[0].item()) == 42
            assert int(tokens[1].item()) == 77
            return "greedy argmax verified"
        tests.append(("Sampler greedy", test_sampler_greedy))

        def test_attention_pipeline():
            import mlx.core as mx
            from ssd.layers.attention import Attention
            from ssd.utils.context import set_context, reset_context
            NUM_HEADS, HEAD_DIM, NUM_KV = 4, 16, 2
            scale = HEAD_DIM ** -0.5
            attn = Attention(NUM_HEADS, HEAD_DIM, scale, NUM_KV)
            set_context(
                is_prefill=True,
                cu_seqlens_q=mx.array([0, 3, 8]),
                cu_seqlens_k=mx.array([0, 3, 8]),
            )
            q = mx.random.normal((8, NUM_HEADS * HEAD_DIM))
            k = mx.random.normal((8, NUM_KV * HEAD_DIM))
            v = mx.random.normal((8, NUM_KV * HEAD_DIM))
            o = attn(q, k, v); mx.eval(o)
            assert o.shape == (8, NUM_HEADS * HEAD_DIM)
            reset_context()
            return "2-seq prefill OK"
        tests.append(("Attention prefill pipeline", test_attention_pipeline))

        def test_mask():
            import mlx.core as mx
            from ssd.engine.helpers.mask_helpers import get_mask_iter_i
            K, F = 3, 2
            prefix_len = 10
            mask = get_mask_iter_i(0, prefix_len, K, F)
            mx.eval(mask)
            q_len = F * (K + 1)
            expected_cols = prefix_len + (K + 1) + q_len
            assert mask.shape[0] == q_len
            assert mask.shape[1] == expected_cols
            return f"shape={mask.shape}"
        tests.append(("Mask generation", test_mask))

        return tests

    # ── Phase 8: Integration ───────────────────────────────────

    @staticmethod
    def _tests_integration():
        tests = []

        def test_ssd_import():
            try:
                import ssd
                return f"exports={len(dir(ssd))} names"
            except RuntimeError as e:
                raise RuntimeError(f"ssd package import failed: {e}")
        tests.append(("ssd package import", test_ssd_import))

        def test_llm_class():
            from ssd.llm import LLM
            from ssd.engine.llm_engine import LLMEngine
            assert issubclass(LLM, LLMEngine)
            return "LLM \u2190 LLMEngine"
        tests.append(("LLM class hierarchy", test_llm_class))

        return tests


# ─── Installer ────────────────────────────────────────────────────────────────

class Installer:
    """One-click installer for SSD-Metal environment."""

    CONDA_ENV = "ssd-metal"
    PYTHON_VER = "3.12"

    DEPS = [
        "mlx>=0.22.0",
        "transformers>=4.51.0",
        "safetensors>=0.4.0",
        "xxhash>=3.5.0",
        "numpy>=2.0.0",
        "tqdm>=4.65.0",
        "tiktoken",
    ]

    @staticmethod
    def _run(cmd, check=True, capture=False, **kwargs):
        if capture:
            return subprocess.run(cmd, capture_output=True, text=True, check=check, **kwargs)
        return subprocess.run(cmd, check=check, **kwargs)

    @staticmethod
    def _cmd_exists(name):
        return shutil.which(name) is not None

    @staticmethod
    def _step(num, total, msg):
        bar_filled = int((num / total) * 20)
        bar_empty = 20 - bar_filled
        bar = f"{Style.BRIGHT_CYAN}{'\u2588' * bar_filled}{Style.DIM}{'\u2591' * bar_empty}{Style.RESET}"
        print(f"\n  [{bar}] {Style.BOLD}Step {num}/{total}{Style.RESET} {msg}")

    @staticmethod
    def _ok(msg):
        print(f"    {Style.CHECK} {msg}")

    @staticmethod
    def _info(msg):
        print(f"    {Style.BULLET} {Style.DIM}{msg}{Style.RESET}")

    @staticmethod
    def _fail(msg):
        print(f"    {Style.CROSS} {Style.BRIGHT_RED}{msg}{Style.RESET}")

    def run(self):
        print_banner()
        print(f"  {Style.BOLD}{Style.BRIGHT_WHITE}One-Click Installer{Style.RESET}")
        print(f"  {Style.DIM}Setting up SSD-Metal environment for Apple Silicon{Style.RESET}")
        print()

        if platform.system() != "Darwin":
            self._fail("SSD-Metal requires macOS with Apple Silicon.")
            sys.exit(1)

        machine = platform.machine()
        if machine not in ("arm64", "aarch64"):
            self._fail(f"Apple Silicon required, got: {machine}")
            sys.exit(1)

        total_steps = 5

        # Step 1: Homebrew
        self._step(1, total_steps, "Checking Homebrew")
        if self._cmd_exists("brew"):
            result = self._run(["brew", "--version"], capture=True, check=False)
            ver = result.stdout.strip().split("\n")[0] if result.returncode == 0 else "unknown"
            self._ok(f"Homebrew found ({ver})")
        else:
            self._info("Homebrew not found. Installing...")
            print()
            try:
                self._run([
                    "/bin/bash", "-c",
                    "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
                ])
                self._ok("Homebrew installed")
            except subprocess.CalledProcessError:
                self._fail("Failed to install Homebrew. Install manually:")
                self._info('/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"')
                sys.exit(1)

        # Step 2: Conda
        self._step(2, total_steps, "Checking Conda")
        conda_cmd = None
        for name in ("conda", "mamba"):
            if self._cmd_exists(name):
                conda_cmd = name
                break
        if conda_cmd:
            result = self._run([conda_cmd, "--version"], capture=True, check=False)
            ver = result.stdout.strip() if result.returncode == 0 else "unknown"
            self._ok(f"{conda_cmd} found ({ver})")
        else:
            self._info("Conda not found. Installing Miniforge via Homebrew...")
            try:
                self._run(["brew", "install", "miniforge"])
                conda_cmd = "conda"
                self._ok("Miniforge installed")
                self._info("Run 'conda init zsh' (or bash) and restart your terminal after this setup.")
            except subprocess.CalledProcessError:
                self._fail("Failed to install Miniforge. Install manually:")
                self._info("brew install miniforge")
                sys.exit(1)

        # Step 3: Conda env
        self._step(3, total_steps, f"Setting up '{self.CONDA_ENV}' environment")
        result = self._run([conda_cmd, "env", "list"], capture=True, check=False)
        env_exists = self.CONDA_ENV in result.stdout if result.returncode == 0 else False
        if env_exists:
            self._ok(f"Environment '{self.CONDA_ENV}' already exists")
        else:
            self._info(f"Creating conda environment: {self.CONDA_ENV} (Python {self.PYTHON_VER})")
            try:
                self._run([conda_cmd, "create", "-n", self.CONDA_ENV, f"python={self.PYTHON_VER}", "-y"])
                self._ok(f"Environment '{self.CONDA_ENV}' created")
            except subprocess.CalledProcessError:
                self._fail(f"Failed to create conda environment '{self.CONDA_ENV}'")
                sys.exit(1)

        # Step 4: Dependencies
        self._step(4, total_steps, "Installing dependencies")
        result = self._run(
            [conda_cmd, "run", "-n", self.CONDA_ENV, "python", "-c", "import sys; print(sys.executable)"],
            capture=True, check=False,
        )
        if result.returncode != 0:
            self._fail("Cannot find Python in conda environment")
            sys.exit(1)

        self._info(f"Installing: {', '.join(d.split('>=')[0].split('<')[0] for d in self.DEPS)}")
        try:
            self._run([conda_cmd, "run", "-n", self.CONDA_ENV, "pip", "install"] + self.DEPS)
            self._ok("All dependencies installed")
        except subprocess.CalledProcessError as e:
            self._fail(f"Dependency installation failed: {e}")
            sys.exit(1)

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pyproject = os.path.join(project_root, "pyproject.toml")
        if os.path.exists(pyproject):
            self._info("Installing ssd package (editable)...")
            try:
                self._run([conda_cmd, "run", "-n", self.CONDA_ENV, "pip", "install", "-e", project_root])
                self._ok("ssd package installed")
            except subprocess.CalledProcessError:
                self._info("Editable install failed, trying regular install...")
                try:
                    self._run([conda_cmd, "run", "-n", self.CONDA_ENV, "pip", "install", project_root])
                    self._ok("ssd package installed")
                except subprocess.CalledProcessError:
                    self._fail("Failed to install ssd package")

        # Step 5: Verify
        self._step(5, total_steps, "Verifying installation")
        verify_script = (
            "import mlx.core as mx; "
            "print(f'MLX {mx.__version__}'); "
            "from ssd.layers.layernorm import RMSNorm; "
            "x = mx.ones((2,64)); "
            "y = RMSNorm(64)(x); "
            "mx.eval(y); "
            "print('RMSNorm OK'); "
            "print('ALL OK')"
        )
        result = self._run(
            [conda_cmd, "run", "-n", self.CONDA_ENV, "python", "-c", verify_script],
            capture=True, check=False,
        )
        if result.returncode == 0 and "ALL OK" in result.stdout:
            for line in result.stdout.strip().split("\n"):
                self._ok(line.strip())
        else:
            self._fail("Verification failed")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-5:]:
                    self._info(line)

        # Done
        print()
        print(f"  {Style.DIM}{'─' * 68}{Style.RESET}")
        print()
        print(f"  {Style.BRIGHT_GREEN}{Style.BOLD}\u2728 Installation complete!{Style.RESET}")
        print()
        print(f"  {Style.BRIGHT_WHITE}To get started:{Style.RESET}")
        print(f"    {Style.DIM}conda activate {self.CONDA_ENV}{Style.RESET}")
        print(f"    {Style.DIM}export SSD_HF_CACHE=/path/to/huggingface/hub{Style.RESET}")
        print(f"    {Style.DIM}export SSD_DATASET_DIR=/path/to/datasets{Style.RESET}")
        print(f"    {Style.DIM}python -m ssd{Style.RESET}  {Style.DIM}# run diagnostics{Style.RESET}")
        print()


# ─── TUI (Full-Screen Terminal Interface) ─────────────────────────────────────

class TUI:
    """Curses-based full-screen interface, htop-style."""

    # Color pair indices
    C_BANNER   = 1
    C_SELECTED = 2
    C_SUCCESS  = 3
    C_FAIL     = 4
    C_WARN     = 5
    C_HEADER   = 6
    C_ACCENT   = 7
    C_DIM      = 8

    MENU_ITEMS = [
        ("Run Diagnostics",   "System readiness check (39 tests)",   "diag"),
        ("One-Click Install",  "Setup environment & dependencies",   "install"),
        ("Download Models",    "Get models from HuggingFace",        "download"),
        ("Run Benchmark",      "Performance benchmark suite",        "bench"),
        ("Interactive Chat",   "Chat with loaded model",             "chat"),
        ("System Info",        "Hardware & software details",        "sysinfo"),
        ("Exit",               "Quit SSD-Metal",                     "exit"),
    ]

    DOWNLOAD_ITEMS = [
        ("Llama 3.1 (8B)",    "Meta-Llama-3.1-8B target model",     "llama"),
        ("Qwen 3 (32B)",      "Qwen3-32B target model",             "qwen"),
        ("Eagle 3 (Draft)",   "EAGLE3 draft model for Llama",       "eagle"),
        ("All Models",         "Download everything",                "all"),
        ("Back",               "Return to main menu",                "back"),
    ]

    SPINNER = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"

    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.h, self.w = stdscr.getmaxyx()
        self.menu_idx = 0
        self.download_idx = 0
        self.hw_info = []
        self._setup()

    def _setup(self):
        curses.curs_set(0)
        curses.use_default_colors()
        self.stdscr.keypad(True)

        curses.init_pair(self.C_BANNER,   curses.COLOR_CYAN,  -1)
        curses.init_pair(self.C_SELECTED, curses.COLOR_BLACK,  curses.COLOR_CYAN)
        curses.init_pair(self.C_SUCCESS,  curses.COLOR_GREEN,  -1)
        curses.init_pair(self.C_FAIL,     curses.COLOR_RED,    -1)
        curses.init_pair(self.C_WARN,     curses.COLOR_YELLOW, -1)
        curses.init_pair(self.C_HEADER,   curses.COLOR_WHITE,  -1)
        curses.init_pair(self.C_ACCENT,   curses.COLOR_CYAN,   -1)
        curses.init_pair(self.C_DIM,      curses.COLOR_WHITE,  -1)

        # Loading screen
        self.stdscr.erase()
        self._center(self.h // 2 - 1, "SSD-METAL", curses.color_pair(self.C_BANNER) | curses.A_BOLD)
        self._center(self.h // 2 + 1, "Loading...", curses.A_DIM)
        self.stdscr.refresh()
        self.hw_info = HardwareInfo.gather()

    # ─── Drawing Primitives ───────────────────────────────────

    def _s(self, y, x, text, attr=0):
        """Safe addstr — write text, silently ignore out-of-bounds."""
        try:
            if 0 <= y < self.h:
                self.stdscr.addnstr(y, x, text, max(0, self.w - x - 1), attr)
        except curses.error:
            pass

    def _center(self, y, text, attr=0):
        x = max(0, (self.w - len(text)) // 2)
        self._s(y, x, text, attr)

    def _hline(self, y, attr=0):
        self._s(y, 0, '\u2500' * (self.w - 1), attr | curses.A_DIM)

    def _fill(self, y, attr=0):
        self._s(y, 0, ' ' * (self.w - 1), attr)

    # ─── Drawing Components ───────────────────────────────────

    def draw_banner(self, y=0):
        """Draw full ASCII art banner. Returns next y."""
        if self.w < 74:
            # Compact fallback for narrow terminals
            self._center(y, "SSD-METAL", curses.color_pair(self.C_BANNER) | curses.A_BOLD)
            self._center(y + 1, "Speculative Decoding Engine for Apple Silicon", curses.A_DIM)
            self._center(y + 2, "v0.3.0", curses.A_DIM)
            return y + 4

        attr = curses.color_pair(self.C_BANNER) | curses.A_BOLD
        for i, line in enumerate(BANNER_LINES):
            self._center(y + i, line, attr)
        y += len(BANNER_LINES)
        self._hline(y, curses.color_pair(self.C_DIM))
        y += 1
        self._center(y, "Speculative Decoding Engine for Apple Silicon",
                     curses.color_pair(self.C_HEADER) | curses.A_BOLD)
        y += 1
        self._center(y, "v0.3.0", curses.A_DIM)
        y += 1
        self._hline(y, curses.color_pair(self.C_DIM))
        return y + 1

    def draw_compact_header(self, title):
        """Single-line header for sub-views. Returns next y."""
        left = f" SSD-METAL \u2502 {title} "
        pad = '\u2500' * max(0, self.w - len(left) - 1)
        self._fill(0, curses.color_pair(self.C_ACCENT) | curses.A_REVERSE)
        self._s(0, 0, left + pad, curses.color_pair(self.C_ACCENT) | curses.A_REVERSE | curses.A_BOLD)
        return 1

    def draw_hw_summary(self, y):
        """Compact hardware summary line. Returns next y."""
        hw = dict(self.hw_info)
        chip = hw.get("Chip", "Unknown")
        mem = hw.get("Memory", "?")
        mlx_ver = hw.get("MLX", "?")
        macos = hw.get("macOS", "?")

        box_w = min(68, self.w - 4)

        # Top border
        title = " System "
        self._s(y, 2, '\u250c' + title, curses.A_DIM)
        self._s(y, 2 + 1 + len(title), '\u2500' * max(0, box_w - len(title) - 1) + '\u2510', curses.A_DIM)
        y += 1

        # Content line
        info = f"  {chip}  \u00b7  {mem}  \u00b7  macOS {macos}  \u00b7  MLX {mlx_ver}"
        self._s(y, 2, '\u2502', curses.A_DIM)
        self._s(y, 3, info[:box_w - 1], curses.A_DIM)
        self._s(y, box_w + 2, '\u2502', curses.A_DIM)
        y += 1

        # Bottom border
        self._s(y, 2, '\u2514' + '\u2500' * box_w + '\u2518', curses.A_DIM)
        return y + 2

    def draw_menu(self, items, selected, y, label_w=24):
        """Draw menu items with selection highlight. Returns next y."""
        for i, item in enumerate(items):
            name, desc = item[0], item[1]
            if i == selected:
                self._fill(y, curses.color_pair(self.C_SELECTED))
                self._s(y, 4, f" \u25b8 {name:<{label_w}} {desc}",
                        curses.color_pair(self.C_SELECTED) | curses.A_BOLD)
            else:
                self._s(y, 4, f"   {name:<{label_w}}", curses.color_pair(self.C_HEADER))
                self._s(y, 7 + label_w, desc, curses.A_DIM)
            y += 1
        return y

    def draw_footer(self, text):
        """Draw footer bar at bottom of screen."""
        y = self.h - 1
        self._fill(y, curses.color_pair(self.C_ACCENT) | curses.A_REVERSE)
        self._s(y, 1, text, curses.color_pair(self.C_ACCENT) | curses.A_REVERSE | curses.A_BOLD)

    def draw_progress(self, y, current, total, label=""):
        """Draw progress bar at given row."""
        bar_w = min(40, self.w - 30)
        pct = current / total if total > 0 else 0
        filled = int(pct * bar_w)

        self._s(y, 4, '\u2501' * filled, curses.color_pair(self.C_BANNER) | curses.A_BOLD)
        self._s(y, 4 + filled, '\u2591' * (bar_w - filled), curses.A_DIM)

        info = f"  {current}/{total}"
        if label:
            info += f"  {label}"
        self._s(y, 4 + bar_w + 1, info, curses.A_DIM)

    # ─── Views ────────────────────────────────────────────────

    def view_menu(self):
        """Main menu with banner + arrow navigation."""
        while True:
            self.h, self.w = self.stdscr.getmaxyx()
            self.stdscr.erase()

            y = self.draw_banner()
            y += 1
            y = self.draw_hw_summary(y)
            self.draw_menu(self.MENU_ITEMS, self.menu_idx, y)
            self.draw_footer(" \u2191\u2193 Navigate   Enter Select   q Quit")

            self.stdscr.refresh()

            key = self.stdscr.getch()
            if key == curses.KEY_UP:
                self.menu_idx = (self.menu_idx - 1) % len(self.MENU_ITEMS)
            elif key == curses.KEY_DOWN:
                self.menu_idx = (self.menu_idx + 1) % len(self.MENU_ITEMS)
            elif key in (curses.KEY_ENTER, 10, 13):
                action = self.MENU_ITEMS[self.menu_idx][2]
                if action == "exit":
                    return
                self._dispatch(action)
            elif ord('1') <= key <= ord('7'):
                idx = key - ord('1')
                if idx < len(self.MENU_ITEMS):
                    action = self.MENU_ITEMS[idx][2]
                    if action == "exit":
                        return
                    self.menu_idx = idx
                    self._dispatch(action)
            elif key in (ord('q'), ord('Q')):
                return
            elif key == curses.KEY_RESIZE:
                self.h, self.w = self.stdscr.getmaxyx()

    def _dispatch(self, action):
        if action == "diag":
            self.view_diagnostics()
        elif action == "install":
            self._run_in_shell(lambda: Installer().run())
        elif action == "download":
            self.view_download()
        elif action == "bench":
            self._run_in_shell(self._run_benchmark)
        elif action == "chat":
            self._run_in_shell(self._run_chat)
        elif action == "sysinfo":
            self.view_sysinfo()

    def view_diagnostics(self):
        """Run all diagnostic phases with real-time display."""
        # Content buffer: [(text, color_pair, attr), ...]
        content = []
        scroll = 0
        total_tests = 0
        completed = 0
        all_phases = []

        # Count total tests first
        for pn in range(1, 9):
            _, tests = TestRunner.get_tests_for_phase(pn)
            total_tests += len(tests)

        def redraw(phase_label="", running=True):
            nonlocal scroll
            self.h, self.w = self.stdscr.getmaxyx()
            self.stdscr.erase()

            header_y = self.draw_compact_header("Diagnostics")
            footer_h = 3  # progress + blank + footer
            content_top = header_y + 1
            content_bot = self.h - footer_h
            visible = content_bot - content_top

            # Auto-scroll to keep latest visible
            if len(content) > visible:
                scroll = len(content) - visible
            if scroll < 0:
                scroll = 0

            for i in range(visible):
                idx = scroll + i
                if idx < len(content):
                    text, cp, attr = content[idx]
                    self._s(content_top + i, 0, text, curses.color_pair(cp) | attr)

            # Progress bar
            self.draw_progress(self.h - 2, completed, total_tests, phase_label)

            if running:
                self.draw_footer(" Running diagnostics...")
            else:
                self.draw_footer(" \u2191\u2193 Scroll   Enter/Esc Back to Menu")

            self.stdscr.refresh()

        # Run each phase
        t_start = time.perf_counter()

        for phase_num in range(1, 9):
            phase_name, tests = TestRunner.get_tests_for_phase(phase_num)
            phase_result = PhaseResult(name=phase_name)

            # Phase header
            if phase_num > 1:
                content.append(("", 0, 0))
            header = f"  \u2500\u2500 Phase {phase_num}: {phase_name} "
            header += '\u2500' * max(0, min(50, self.w - 6) - len(header))
            content.append((header, self.C_HEADER, curses.A_BOLD))

            for test_name, test_fn in tests:
                # Show spinner line
                test_idx = len(content)
                content.append((f"    {self.SPINNER[0]} {test_name}...", self.C_ACCENT, 0))
                redraw(f"Phase {phase_num}/8")

                # Run test in background thread
                result_holder = [None]
                done = threading.Event()

                def run(fn=test_fn, nm=test_name):
                    result_holder[0] = TestRunner.run_test_silent(nm, fn)
                    done.set()

                thread = threading.Thread(target=run, daemon=True)
                thread.start()

                # Animate spinner while test runs
                si = 0
                while not done.wait(timeout=0.08):
                    si = (si + 1) % len(self.SPINNER)
                    content[test_idx] = (f"    {self.SPINNER[si]} {test_name}...", self.C_ACCENT, 0)
                    redraw(f"Phase {phase_num}/8")

                thread.join(timeout=5.0)
                result = result_holder[0]
                if result is None:
                    result = TestResult(name=test_name, passed=False, error="Timeout")

                # Format result line
                max_name = min(36, self.w - 30)
                display_name = test_name[:max_name]
                dots = '\u00b7' * max(3, 40 - len(display_name))
                time_str = f" {result.time_ms:.0f}ms" if result.time_ms > 0 else ""

                if result.passed:
                    detail = f" ({result.detail})" if result.detail else ""
                    content[test_idx] = (
                        f"    \u2713 {display_name} {dots} ok{detail}{time_str}",
                        self.C_SUCCESS, 0
                    )
                elif result.skipped:
                    content[test_idx] = (
                        f"    \u26a0 {display_name} {dots} SKIP{time_str}",
                        self.C_WARN, 0
                    )
                else:
                    content[test_idx] = (
                        f"    \u2717 {display_name} {dots} FAIL{time_str}",
                        self.C_FAIL, 0
                    )
                    if result.error:
                        for err_line in result.error.split('\n')[:2]:
                            content.append((f"      \u2192 {err_line[:self.w-10]}", self.C_FAIL, curses.A_DIM))

                phase_result.tests.append(result)
                completed += 1
                redraw(f"Phase {phase_num}/8")

            all_phases.append(phase_result)

        elapsed = time.perf_counter() - t_start

        # Summary
        total_passed = sum(p.passed for p in all_phases)
        total_failed = sum(p.failed for p in all_phases)
        total_skipped = sum(p.skipped for p in all_phases)

        content.append(("", 0, 0))
        content.append(("  " + "\u2550" * min(50, self.w - 6), self.C_BANNER, 0))
        content.append(("", 0, 0))

        if total_failed == 0:
            content.append(("  \u2728 All systems operational. SSD-Metal is ready.", self.C_SUCCESS, curses.A_BOLD))
        else:
            content.append((f"  Status: {total_failed} test(s) failed", self.C_FAIL, curses.A_BOLD))

        content.append((
            f"  Passed: {total_passed}  Failed: {total_failed}  Skipped: {total_skipped}  Time: {elapsed:.1f}s",
            0, curses.A_DIM
        ))

        if total_failed > 0:
            content.append(("", 0, 0))
            content.append(("  Issues to resolve:", self.C_FAIL, curses.A_BOLD))
            for phase in all_phases:
                for test in phase.tests:
                    if not test.passed and not test.skipped:
                        content.append((f"    \u2717 [{phase.name}] {test.name}", self.C_FAIL, 0))
                        if test.error:
                            for err_line in test.error.split('\n')[:2]:
                                content.append((f"      \u2192 {err_line[:self.w-10]}", self.C_FAIL, curses.A_DIM))

        # Interactive scrolling after completion
        while True:
            self.h, self.w = self.stdscr.getmaxyx()
            self.stdscr.erase()

            header_y = self.draw_compact_header("Diagnostics \u2014 Complete")
            content_top = header_y + 1
            content_bot = self.h - 1
            visible = content_bot - content_top

            # Clamp scroll
            max_scroll = max(0, len(content) - visible)
            scroll = max(0, min(scroll, max_scroll))

            for i in range(visible):
                idx = scroll + i
                if idx < len(content):
                    text, cp, attr = content[idx]
                    self._s(content_top + i, 0, text, curses.color_pair(cp) | attr)

            # Scroll indicators
            if len(content) > visible:
                if scroll > 0:
                    self._s(content_top, self.w - 2, '\u25b2', curses.A_DIM)
                if scroll < max_scroll:
                    self._s(content_bot - 1, self.w - 2, '\u25bc', curses.A_DIM)

            self.draw_footer(" \u2191\u2193 Scroll   Enter/Esc Back to Menu")
            self.stdscr.refresh()

            key = self.stdscr.getch()
            if key == curses.KEY_UP:
                scroll = max(0, scroll - 1)
            elif key == curses.KEY_DOWN:
                scroll = min(max_scroll, scroll + 1)
            elif key == curses.KEY_PPAGE:  # Page Up
                scroll = max(0, scroll - visible)
            elif key == curses.KEY_NPAGE:  # Page Down
                scroll = min(max_scroll, scroll + visible)
            elif key in (27, curses.KEY_ENTER, 10, 13, ord('q')):
                return
            elif key == curses.KEY_RESIZE:
                self.h, self.w = self.stdscr.getmaxyx()

    def view_sysinfo(self):
        """Detailed system information view."""
        while True:
            self.h, self.w = self.stdscr.getmaxyx()
            self.stdscr.erase()

            y = self.draw_compact_header("System Information")
            y += 1

            box_w = min(64, self.w - 6)

            # Hardware info box
            title = " Hardware "
            self._s(y, 3, '\u250c' + title + '\u2500' * max(0, box_w - len(title) - 1) + '\u2510', curses.A_DIM)
            y += 1

            for label, value in self.hw_info:
                self._s(y, 3, '\u2502', curses.A_DIM)
                self._s(y, 5, f"{label:<22}", curses.color_pair(self.C_HEADER))
                clean_val = re.sub(r'\033\[[0-9;]*m', '', str(value))
                self._s(y, 27, clean_val[:box_w - 26], curses.A_DIM)
                self._s(y, box_w + 3, '\u2502', curses.A_DIM)
                y += 1

            self._s(y, 3, '\u2514' + '\u2500' * (box_w) + '\u2518', curses.A_DIM)
            y += 2

            # Environment info box
            title = " Environment "
            self._s(y, 3, '\u250c' + title + '\u2500' * max(0, box_w - len(title) - 1) + '\u2510', curses.A_DIM)
            y += 1

            env_rows = [
                ("Python Executable", sys.executable),
                ("Working Directory", os.getcwd()),
                ("Platform", platform.platform()),
            ]
            # Check env vars
            for var in ("SSD_HF_CACHE", "SSD_DATASET_DIR"):
                val = os.environ.get(var, "not set")
                env_rows.append((var, val))

            for label, value in env_rows:
                self._s(y, 3, '\u2502', curses.A_DIM)
                self._s(y, 5, f"{label:<22}", curses.color_pair(self.C_HEADER))
                self._s(y, 27, str(value)[:box_w - 26], curses.A_DIM)
                self._s(y, box_w + 3, '\u2502', curses.A_DIM)
                y += 1

            self._s(y, 3, '\u2514' + '\u2500' * (box_w) + '\u2518', curses.A_DIM)

            self.draw_footer(" Enter/Esc Back to Menu")
            self.stdscr.refresh()

            key = self.stdscr.getch()
            if key in (27, curses.KEY_ENTER, 10, 13, ord('q')):
                return
            elif key == curses.KEY_RESIZE:
                self.h, self.w = self.stdscr.getmaxyx()

    def view_download(self):
        """Download models sub-menu."""
        while True:
            self.h, self.w = self.stdscr.getmaxyx()
            self.stdscr.erase()

            y = self.draw_compact_header("Download Models")
            y += 1

            self._s(y, 4, "Select models to download from HuggingFace:", curses.color_pair(self.C_HEADER))
            y += 2

            self.draw_menu(self.DOWNLOAD_ITEMS, self.download_idx, y)
            self.draw_footer(" \u2191\u2193 Navigate   Enter Select   Esc Back")
            self.stdscr.refresh()

            key = self.stdscr.getch()
            if key == curses.KEY_UP:
                self.download_idx = (self.download_idx - 1) % len(self.DOWNLOAD_ITEMS)
            elif key == curses.KEY_DOWN:
                self.download_idx = (self.download_idx + 1) % len(self.DOWNLOAD_ITEMS)
            elif key in (curses.KEY_ENTER, 10, 13):
                action = self.DOWNLOAD_ITEMS[self.download_idx][2]
                if action == "back":
                    return
                self._run_download(action)
            elif key in (27, ord('q')):
                return
            elif key == curses.KEY_RESIZE:
                self.h, self.w = self.stdscr.getmaxyx()

    # ─── Shell Commands ───────────────────────────────────────

    def _run_in_shell(self, fn):
        """Temporarily exit curses, run fn in normal terminal, then restore."""
        curses.def_prog_mode()
        curses.endwin()
        try:
            fn()
        except KeyboardInterrupt:
            print("\n  Interrupted.")
        except Exception as e:
            print(f"\n  Error: {e}")
        print(f"\n  Press Enter to return to menu...")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
        curses.reset_prog_mode()
        self.stdscr.refresh()

    def _project_root(self):
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _run_download(self, model_key):
        """Run model download script."""
        script = os.path.join(self._project_root(), "scripts", "download_from_hf.py")
        if not os.path.exists(script):
            self._run_in_shell(lambda: print(
                f"\n  Download script not found at:\n  {script}\n"
                f"  Run manually: python scripts/download_from_hf.py {model_key}"
            ))
            return
        self._run_in_shell(lambda: subprocess.run([sys.executable, script, model_key]))

    def _run_benchmark(self):
        """Run benchmark suite."""
        bench_script = os.path.join(self._project_root(), "bench", "bench.py")
        if not os.path.exists(bench_script):
            print(f"\n  Benchmark script not found at:\n  {bench_script}")
            print(f"\n  Run manually from the bench/ directory:")
            print(f"  python -O bench.py --llama --size 8 --b 1 --temp 0 --numseqs 128 --output_len 512 --all")
            return

        print(f"\n  {Style.BRIGHT_CYAN}{Style.BOLD}SSD-Metal Benchmark{Style.RESET}\n")
        print(f"  Starting quick benchmark (Llama 8B, 10 sequences)...\n")
        subprocess.run([
            sys.executable, "-O", bench_script,
            "--llama", "--size", "8", "--b", "1", "--temp", "0",
            "--numseqs", "10", "--output_len", "128", "--all"
        ])

    def _run_chat(self):
        """Run interactive chat."""
        chat_script = os.path.join(self._project_root(), "bench", "chat.py")
        if not os.path.exists(chat_script):
            print(f"\n  Chat script not found at:\n  {chat_script}")
            print(f"\n  Run manually from the bench/ directory:")
            print(f"  python -O chat.py --ssd --spec --async --k 7 --f 3 --metrics")
            return

        print(f"\n  {Style.BRIGHT_CYAN}{Style.BOLD}SSD-Metal Interactive Chat{Style.RESET}")
        print(f"  {Style.DIM}Starting with async speculative decoding (k=7, f=3)...{Style.RESET}\n")
        subprocess.run([
            sys.executable, "-O", chat_script,
            "--ssd", "--spec", "--async", "--k", "7", "--f", "3", "--metrics"
        ])

    # ─── Main Loop ────────────────────────────────────────────

    def main_loop(self):
        """Entry point for the TUI."""
        if self.h < 16 or self.w < 60:
            self._center(self.h // 2, f"Terminal too small ({self.w}x{self.h}). Need 60x16 minimum.")
            self.stdscr.refresh()
            self.stdscr.getch()
            return
        self.view_menu()


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    if "--install" in sys.argv or "install" in sys.argv:
        Installer().run()
    else:
        os.environ.setdefault('ESCDELAY', '25')
        locale.setlocale(locale.LC_ALL, '')
        curses.wrapper(lambda stdscr: TUI(stdscr).main_loop())


if __name__ == "__main__":
    main()
