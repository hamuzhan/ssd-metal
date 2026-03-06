"""
SSD-Metal Diagnostic CLI
Professional system readiness checker and installer for the SSD speculative decoding engine on Apple Silicon.

Usage:
    python -m ssd              # Run diagnostics
    python -m ssd --install    # One-click install
    python ssd/cli.py --install
"""

import os
import sys
import time
import platform
import subprocess
import threading
import shutil
from dataclasses import dataclass, field


# ─── Style Constants ──────────────────────────────────────────────────────────

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

    CHECK  = "\033[92m\u2713\033[0m"    # green checkmark
    CROSS  = "\033[91m\u2717\033[0m"    # red cross
    WARN   = "\033[93m\u26a0\033[0m"    # yellow warning
    BULLET = "\033[36m\u25cf\033[0m"    # cyan bullet
    ARROW  = "\033[36m\u2192\033[0m"    # cyan arrow


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


# ─── UI Helpers ───────────────────────────────────────────────────────────────

class UI:
    BOX_WIDTH = 72

    @staticmethod
    def clear_line():
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    @staticmethod
    def banner():
        art = r"""
 ███████╗███████╗██████╗       ███╗   ███╗███████╗████████╗ █████╗ ██╗
 ██╔════╝██╔════╝██╔══██╗      ████╗ ████║██╔════╝╚══██╔══╝██╔══██╗██║
 ███████╗███████╗██║  ██║█████╗██╔████╔██║█████╗     ██║   ███████║██║
 ╚════██║╚════██║██║  ██║╚════╝██║╚██╔╝██║██╔══╝     ██║   ██╔══██║██║
 ███████║███████║██████╔╝      ██║ ╚═╝ ██║███████╗   ██║   ██║  ██║███████╗
 ╚══════╝╚══════╝╚═════╝       ╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝"""

        print()
        for line in art.strip("\n").split("\n"):
            print(f"  {Style.BRIGHT_CYAN}{Style.BOLD}{line}{Style.RESET}")
        print()
        print(f"  {Style.DIM}{'─' * 68}{Style.RESET}")
        subtitle = "Speculative Decoding Engine for Apple Silicon"
        print(f"  {Style.BRIGHT_WHITE}{Style.BOLD}{subtitle:^68}{Style.RESET}")
        version_line = "v0.3.0  \u00b7  Diagnostics"
        print(f"  {Style.DIM}{version_line:^68}{Style.RESET}")
        print(f"  {Style.DIM}{'─' * 68}{Style.RESET}")
        print()

    @staticmethod
    def box(title: str, rows: list[tuple[str, str]]):
        w = UI.BOX_WIDTH
        # Top border with title
        title_str = f" {title} "
        top = f"  \u250c{title_str}{'─' * (w - len(title_str) - 1)}\u2510"
        print(f"  {Style.DIM}\u250c{Style.RESET}{Style.BOLD}{Style.BRIGHT_WHITE} {title} {Style.RESET}{Style.DIM}{'─' * (w - len(title_str) - 1)}\u2510{Style.RESET}")

        for label, value in rows:
            # Pad to box width
            content = f"   {label:<28}{value}"
            padding = w - 1 - len(content)
            if padding < 0:
                padding = 0
            print(f"  {Style.DIM}\u2502{Style.RESET}{content}{' ' * padding}{Style.DIM}\u2502{Style.RESET}")

        # Bottom border
        print(f"  {Style.DIM}\u2514{'─' * w}\u2518{Style.RESET}")
        print()

    @staticmethod
    def phase_header(phase_num: int, name: str):
        header = f"Phase {phase_num}: {name}"
        line_len = UI.BOX_WIDTH - len(header) - 4
        print(f"\n  {Style.DIM}\u2500\u2500{Style.RESET} {Style.BOLD}{Style.BRIGHT_WHITE}{header}{Style.RESET} {Style.DIM}{'─' * max(line_len, 4)}{Style.RESET}\n")

    @staticmethod
    def test_result(result: TestResult):
        UI.clear_line()
        if result.skipped:
            icon = Style.WARN
            status = f"{Style.YELLOW}SKIP{Style.RESET}"
        elif result.passed:
            icon = Style.CHECK
            status = f"{Style.GREEN}ok{Style.RESET}"
        else:
            icon = Style.CROSS
            status = f"{Style.BRIGHT_RED}FAIL{Style.RESET}"

        name = result.name
        detail = result.detail
        # Build the dotted line
        used = len(name) + len(result.detail.replace('\033[', '').split('m')[-1] if '\033' in result.detail else result.detail) + 12
        # Simple approach: fixed dot count
        dot_count = max(3, 48 - len(name))
        dots = f"{Style.DIM}{'·' * dot_count}{Style.RESET}"
        detail_str = f"  ({Style.DIM}{detail}{Style.RESET})" if detail else ""
        time_str = f"  {Style.DIM}{result.time_ms:.0f}ms{Style.RESET}" if result.time_ms > 0 else ""
        print(f"    {icon} {name} {dots} {status}{detail_str}{time_str}")

        if result.error and not result.passed:
            for line in result.error.split("\n"):
                print(f"      {Style.ARROW} {Style.DIM}{line}{Style.RESET}")

    @staticmethod
    def summary(phases: list[PhaseResult], total_time: float):
        total_passed = sum(p.passed for p in phases)
        total_failed = sum(p.failed for p in phases)
        total_skipped = sum(p.skipped for p in phases)
        all_pass = total_failed == 0

        if all_pass:
            status_str = f"{Style.BRIGHT_GREEN}{Style.BOLD}READY{Style.RESET}"
        else:
            status_str = f"{Style.BRIGHT_RED}{Style.BOLD}NOT READY{Style.RESET}"

        rows = [
            ("Tests Passed", f"{Style.GREEN}{total_passed}{Style.RESET}"),
            ("Tests Failed", f"{Style.RED}{total_failed}{Style.RESET}" if total_failed else f"{Style.DIM}0{Style.RESET}"),
            ("Tests Skipped", f"{Style.YELLOW}{total_skipped}{Style.RESET}" if total_skipped else f"{Style.DIM}0{Style.RESET}"),
            ("Time", f"{total_time:.2f}s"),
            ("Status", status_str),
        ]

        UI.box("Summary", rows)

        # List failures
        if total_failed > 0:
            print(f"  {Style.BRIGHT_RED}{Style.BOLD}Issues to resolve:{Style.RESET}")
            for phase in phases:
                for test in phase.tests:
                    if not test.passed and not test.skipped:
                        print(f"    {Style.CROSS} {Style.DIM}[{phase.name}]{Style.RESET} {Style.BRIGHT_WHITE}{test.name}{Style.RESET}")
                        err = test.error or "Unknown error"
                        for line in err.split("\n")[:3]:
                            print(f"      {Style.ARROW} {Style.DIM}{line}{Style.RESET}")
            print()
        else:
            print(f"  {Style.BRIGHT_GREEN}{Style.BOLD}\u2728 All systems operational. SSD-Metal is ready.{Style.RESET}\n")


# ─── Spinner ──────────────────────────────────────────────────────────────────

class Spinner:
    """Braille spinner on a daemon thread, used as context manager."""
    FRAMES = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"

    def __init__(self, text: str = ""):
        self.text = text
        self._stop = threading.Event()
        self._thread = None

    def _spin(self):
        i = 0
        while not self._stop.is_set():
            frame = self.FRAMES[i % len(self.FRAMES)]
            sys.stdout.write(f"\r    {Style.BRIGHT_CYAN}{frame}{Style.RESET} {Style.DIM}{self.text}{Style.RESET}")
            sys.stdout.flush()
            i += 1
            self._stop.wait(0.08)
        UI.clear_line()

    def __enter__(self):
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        UI.clear_line()

    def update(self, text: str):
        self.text = text


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

        # Chip
        chip = HardwareInfo._sysctl("machdep.cpu.brand_string")
        if not chip:
            chip = platform.processor() or "Unknown"
        rows.append(("Chip", chip))

        # CPU cores
        phys = HardwareInfo._sysctl("hw.physicalcpu")
        perf = HardwareInfo._sysctl("hw.perflevel0.physicalcpu")
        eff = HardwareInfo._sysctl("hw.perflevel1.physicalcpu")
        if phys:
            core_str = f"{phys} physical"
            if perf and eff:
                core_str += f" ({perf}P + {eff}E)"
            rows.append(("CPU Cores", core_str))

        # Memory
        try:
            mem_bytes = int(HardwareInfo._sysctl("hw.memsize"))
            mem_gb = mem_bytes / (1024 ** 3)
            rows.append(("Memory", f"{mem_gb:.1f} GB"))
        except (ValueError, TypeError):
            rows.append(("Memory", "Unknown"))

        # macOS
        mac_ver = platform.mac_ver()[0]
        if mac_ver:
            rows.append(("macOS", mac_ver))

        # Neural Engine (infer from chip name)
        chip_lower = chip.lower()
        if "m4" in chip_lower:
            rows.append(("Neural Engine", "16-core"))
        elif "m3" in chip_lower:
            rows.append(("Neural Engine", "16-core"))
        elif "m2" in chip_lower:
            rows.append(("Neural Engine", "16-core"))
        elif "m1" in chip_lower:
            rows.append(("Neural Engine", "16-core"))
        elif "apple" in chip_lower:
            rows.append(("Neural Engine", "Available"))

        # Python
        rows.append(("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"))

        # MLX
        try:
            import mlx.core as mx
            ver = getattr(mx, "__version__", None)
            if not ver:
                import mlx
                ver = getattr(mlx, "__version__", None)
            rows.append(("MLX", f"v{ver}" if ver else "installed"))
        except ImportError:
            rows.append(("MLX", f"{Style.BRIGHT_RED}Not installed{Style.RESET}"))
        except Exception as e:
            rows.append(("MLX", f"{Style.YELLOW}{e}{Style.RESET}"))

        # Metal memory
        try:
            import mlx.core as mx
            # Use mx.device_info if available (newer MLX), fallback to mx.metal.device_info
            if hasattr(mx, "device_info"):
                info = mx.device_info()
            else:
                info = mx.metal.device_info()
            metal_mem = info.get("memory_size", 0)
            if metal_mem:
                rows.append(("Metal Memory", f"{metal_mem / (1024**3):.1f} GB"))
            rec_ws = info.get("recommended_max_working_set_size", 0)
            if rec_ws:
                rows.append(("Metal Recommended WS", f"{rec_ws / (1024**3):.1f} GB"))
            arch = info.get("architecture", "")
            if arch:
                rows.append(("Metal Architecture", arch))
        except Exception:
            pass

        return rows


# ─── Test Runner ──────────────────────────────────────────────────────────────

class TestRunner:

    @staticmethod
    def run_test(name: str, fn, spinner_text: str = "") -> TestResult:
        with Spinner(spinner_text or f"Testing {name}..."):
            t0 = time.perf_counter()
            try:
                detail = fn()
                elapsed = (time.perf_counter() - t0) * 1000
                return TestResult(name=name, passed=True, detail=detail or "", time_ms=elapsed)
            except Exception as e:
                elapsed = (time.perf_counter() - t0) * 1000
                err_msg = str(e)
                # Truncate very long error messages
                if len(err_msg) > 200:
                    err_msg = err_msg[:200] + "..."
                return TestResult(name=name, passed=False, error=err_msg, time_ms=elapsed)

    # ── Phase 1: Core Dependencies ─────────────────────────────────

    @staticmethod
    def phase_core_deps() -> PhaseResult:
        phase = PhaseResult(name="Core Dependencies")

        deps = [
            ("mlx", "mlx"),
            ("transformers", "transformers"),
            ("safetensors", "safetensors"),
            ("xxhash", "xxhash"),
            ("numpy", "numpy"),
            ("tqdm", "tqdm"),
            ("tiktoken", "tiktoken"),
        ]

        for display_name, module_name in deps:
            def make_test(mod=module_name):
                def test():
                    m = __import__(mod)
                    ver = getattr(m, "__version__", None)
                    return f"v{ver}" if ver else "imported"
                return test
            result = TestRunner.run_test(display_name, make_test(), f"Importing {display_name}...")
            phase.tests.append(result)

        return phase

    # ── Phase 2: Foundation Modules ────────────────────────────────

    @staticmethod
    def phase_foundation() -> PhaseResult:
        phase = PhaseResult(name="Foundation Modules")

        # ssd.config — just verify class exists (don't instantiate, requires real model dir)
        def test_config():
            from ssd.config import Config
            assert hasattr(Config, '__post_init__'), "Config missing __post_init__"
            assert hasattr(Config, 'max_blocks'), "Config missing max_blocks property"
            return "Config class OK"
        phase.tests.append(TestRunner.run_test("ssd.config", test_config, "Checking Config..."))

        # ssd.paths — catch RuntimeError if env vars missing
        def test_paths():
            try:
                import ssd.paths
                return f"HF_CACHE={ssd.paths.HF_CACHE_DIR[:40]}..."
            except RuntimeError as e:
                raise RuntimeError(f"Env var not set: {e}")
        phase.tests.append(TestRunner.run_test("ssd.paths", test_paths, "Checking paths..."))

        # ssd.sampling_params
        def test_sampling_params():
            from ssd.sampling_params import SamplingParams
            sp = SamplingParams()
            assert sp.temperature == 1.0
            assert sp.max_new_tokens == 256
            return f"defaults OK (T={sp.temperature})"
        phase.tests.append(TestRunner.run_test("ssd.sampling_params", test_sampling_params, "Checking SamplingParams..."))

        # ssd.engine.sequence
        def test_sequence():
            from ssd.engine.sequence import Sequence, SequenceStatus
            Sequence.block_size = 256
            seq = Sequence([1, 2, 3])
            assert len(seq) == 3
            assert seq.status == SequenceStatus.WAITING
            assert seq.num_prompt_tokens == 3
            return f"Sequence([1,2,3]) len={len(seq)}"
        phase.tests.append(TestRunner.run_test("ssd.engine.sequence", test_sequence, "Checking Sequence..."))

        return phase

    # ── Phase 3: Layer Primitives ──────────────────────────────────

    @staticmethod
    def phase_layers() -> PhaseResult:
        phase = PhaseResult(name="Layer Primitives")

        HIDDEN = 64
        HEAD_DIM = 16
        NUM_HEADS = 4
        NUM_KV_HEADS = 2
        VOCAB = 256
        INTER = 128

        # RMSNorm
        def test_rmsnorm():
            import mlx.core as mx
            from ssd.layers.layernorm import RMSNorm
            norm = RMSNorm(HIDDEN)
            x = mx.random.normal((2, HIDDEN))
            y = norm(x)
            mx.eval(y)
            assert y.shape == (2, HIDDEN), f"shape {y.shape}"
            return f"RMSNorm({HIDDEN})"
        phase.tests.append(TestRunner.run_test("RMSNorm", test_rmsnorm, "Testing RMSNorm..."))

        # SiluAndMul
        def test_silu():
            import mlx.core as mx
            from ssd.layers.activation import SiluAndMul
            act = SiluAndMul()
            x = mx.random.normal((2, INTER * 2))
            y = act(x)
            mx.eval(y)
            assert y.shape == (2, INTER), f"shape {y.shape}"
            return f"in={INTER*2} out={INTER}"
        phase.tests.append(TestRunner.run_test("SiluAndMul", test_silu, "Testing SiluAndMul..."))

        # RotaryEmbedding
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
        phase.tests.append(TestRunner.run_test("RotaryEmbedding", test_rope, "Testing RotaryEmbedding..."))

        # QKVLinear
        def test_qkv():
            import mlx.core as mx
            from ssd.layers.linear import QKVLinear
            qkv = QKVLinear(HIDDEN, HEAD_DIM, NUM_HEADS, NUM_KV_HEADS)
            x = mx.random.normal((2, HIDDEN))
            y = qkv(x)
            mx.eval(y)
            expected = (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
            assert y.shape == (2, expected), f"shape {y.shape} expected (2, {expected})"
            return f"{HIDDEN}\u2192{expected}"
        phase.tests.append(TestRunner.run_test("QKVLinear", test_qkv, "Testing QKVLinear..."))

        # GateUpLinear
        def test_gateup():
            import mlx.core as mx
            from ssd.layers.linear import GateUpLinear
            gu = GateUpLinear(HIDDEN, INTER)
            x = mx.random.normal((2, HIDDEN))
            y = gu(x)
            mx.eval(y)
            assert y.shape == (2, INTER * 2), f"shape {y.shape}"
            return f"{HIDDEN}\u2192{INTER*2}"
        phase.tests.append(TestRunner.run_test("GateUpLinear", test_gateup, "Testing GateUpLinear..."))

        # RowLinear
        def test_row():
            import mlx.core as mx
            from ssd.layers.linear import RowLinear
            rl = RowLinear(INTER, HIDDEN)
            x = mx.random.normal((2, INTER))
            y = rl(x)
            mx.eval(y)
            assert y.shape == (2, HIDDEN), f"shape {y.shape}"
            return f"{INTER}\u2192{HIDDEN}"
        phase.tests.append(TestRunner.run_test("RowLinear", test_row, "Testing RowLinear..."))

        # Embedding
        def test_embed():
            import mlx.core as mx
            from ssd.layers.embed_head import Embedding
            emb = Embedding(VOCAB, HIDDEN)
            emb.weight = mx.random.normal((VOCAB, HIDDEN))
            ids = mx.array([0, 5, 10])
            y = emb(ids)
            mx.eval(y)
            assert y.shape == (3, HIDDEN), f"shape {y.shape}"
            return f"vocab={VOCAB} dim={HIDDEN}"
        phase.tests.append(TestRunner.run_test("Embedding", test_embed, "Testing Embedding..."))

        # LMHead
        def test_lmhead():
            import mlx.core as mx
            from ssd.layers.embed_head import LMHead
            from ssd.utils.context import set_context, reset_context
            lm = LMHead(VOCAB, HIDDEN)
            lm.weight = mx.random.normal((VOCAB, HIDDEN))
            # set context for prefill mode
            set_context(
                is_prefill=True,
                cu_seqlens_q=mx.array([0, 3]),
                cu_seqlens_k=mx.array([0, 3]),
            )
            x = mx.random.normal((3, HIDDEN))
            y = lm(x, last_only=True)
            mx.eval(y)
            assert y.shape == (1, VOCAB), f"shape {y.shape}"
            reset_context()
            return f"dim={HIDDEN} vocab={VOCAB}"
        phase.tests.append(TestRunner.run_test("LMHead", test_lmhead, "Testing LMHead..."))

        # Sampler
        def test_sampler():
            import mlx.core as mx
            from ssd.layers.sampler import Sampler
            s = Sampler()
            logits = mx.random.normal((4, VOCAB))
            temps = mx.zeros((4,))  # greedy
            tokens = s(logits, temps)
            mx.eval(tokens)
            expected = mx.argmax(logits.astype(mx.float32), axis=-1)
            mx.eval(expected)
            assert all(int(tokens[i].item()) == int(expected[i].item()) for i in range(4)), "greedy mismatch"
            return "greedy OK"
        phase.tests.append(TestRunner.run_test("Sampler", test_sampler, "Testing Sampler..."))

        # Attention (prefill path)
        def test_attention():
            import mlx.core as mx
            from ssd.layers.attention import Attention
            from ssd.utils.context import set_context, reset_context
            scale = HEAD_DIM ** -0.5
            attn = Attention(NUM_HEADS, HEAD_DIM, scale, NUM_KV_HEADS)
            # simple prefill without cache
            set_context(
                is_prefill=True,
                cu_seqlens_q=mx.array([0, 4]),
                cu_seqlens_k=mx.array([0, 4]),
            )
            q = mx.random.normal((4, NUM_HEADS * HEAD_DIM))
            k = mx.random.normal((4, NUM_KV_HEADS * HEAD_DIM))
            v = mx.random.normal((4, NUM_KV_HEADS * HEAD_DIM))
            o = attn(q, k, v)
            mx.eval(o)
            assert o.shape == (4, NUM_HEADS * HEAD_DIM), f"shape {o.shape}"
            reset_context()
            return f"heads={NUM_HEADS} dim={HEAD_DIM}"
        phase.tests.append(TestRunner.run_test("Attention", test_attention, "Testing Attention..."))

        return phase

    # ── Phase 4: Kernel Functions ──────────────────────────────────

    @staticmethod
    def phase_kernels() -> PhaseResult:
        phase = PhaseResult(name="Kernel Functions")

        def test_kvcache_store():
            import mlx.core as mx
            from ssd.kernels.kvcache_store import store_kvcache
            D = 32
            N = 4
            SLOTS = 16
            k_cache = mx.zeros((SLOTS, D))
            v_cache = mx.zeros((SLOTS, D))
            key = mx.ones((N, D))
            value = mx.ones((N, D)) * 2.0
            slot_mapping = mx.array([0, 3, 7, -1])  # last one should be skipped
            store_kvcache(key, value, k_cache, v_cache, slot_mapping)
            mx.eval(k_cache, v_cache)
            # Check slot 0 was written
            assert float(k_cache[0, 0].item()) == 1.0, "k_cache[0] not written"
            assert float(v_cache[3, 0].item()) == 2.0, "v_cache[3] not written"
            assert float(k_cache[7, 0].item()) == 1.0, "k_cache[7] not written"
            # Slot -1 should not have been written — check slot 15 is still 0
            assert float(k_cache[15, 0].item()) == 0.0, "k_cache[15] was incorrectly written"
            return f"N={N} slots={SLOTS} padding=OK"
        phase.tests.append(TestRunner.run_test("store_kvcache", test_kvcache_store, "Testing KV cache kernel..."))

        return phase

    # ── Phase 5: Model Definitions ─────────────────────────────────

    @staticmethod
    def phase_models() -> PhaseResult:
        phase = PhaseResult(name="Model Definitions")

        def test_llama():
            from ssd.models.llama3 import LlamaForCausalLM
            assert hasattr(LlamaForCausalLM, 'packed_modules_mapping')
            mapping = LlamaForCausalLM.packed_modules_mapping
            assert "q_proj" in mapping or "gate_proj" in mapping, f"unexpected mapping keys: {list(mapping.keys())}"
            return f"packed_modules={len(mapping)} keys"
        phase.tests.append(TestRunner.run_test("LlamaForCausalLM", test_llama, "Checking Llama model..."))

        def test_qwen():
            from ssd.models.qwen3 import Qwen3ForCausalLM
            assert hasattr(Qwen3ForCausalLM, 'packed_modules_mapping')
            return f"packed_modules={len(Qwen3ForCausalLM.packed_modules_mapping)} keys"
        phase.tests.append(TestRunner.run_test("Qwen3ForCausalLM", test_qwen, "Checking Qwen3 model..."))

        def test_eagle():
            from ssd.models.eagle3_draft_llama3 import Eagle3DraftForCausalLM
            assert hasattr(Eagle3DraftForCausalLM, 'packed_modules_mapping')
            return f"packed_modules={len(Eagle3DraftForCausalLM.packed_modules_mapping)} keys"
        phase.tests.append(TestRunner.run_test("Eagle3DraftForCausalLM", test_eagle, "Checking EAGLE3 model..."))

        return phase

    # ── Phase 6: Engine Components ─────────────────────────────────

    @staticmethod
    def phase_engine() -> PhaseResult:
        phase = PhaseResult(name="Engine Components")

        components = [
            ("ModelRunner", "ssd.engine.model_runner", "ModelRunner"),
            ("DraftRunner", "ssd.engine.draft_runner", "DraftRunner"),
            ("Scheduler", "ssd.engine.scheduler", "Scheduler"),
            ("Verifier", "ssd.engine.verifier", "Verifier"),
            ("SpeculatorUnified", "ssd.engine.speculator_unified", "SpeculatorUnified"),
            ("InferenceStep", "ssd.engine.step", "InferenceStep"),
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
            phase.tests.append(TestRunner.run_test(display, make_test(), f"Importing {display}..."))

        return phase

    # ── Phase 7: Tensor Operations ─────────────────────────────────

    @staticmethod
    def phase_tensor_ops() -> PhaseResult:
        phase = PhaseResult(name="Tensor Operations")

        # Basic ops
        def test_basic():
            import mlx.core as mx
            a = mx.array([1.0, 2.0, 3.0])
            b = mx.array([4.0, 5.0, 6.0])
            c = a + b
            mx.eval(c)
            assert float(c[0].item()) == 5.0
            # matmul
            A = mx.random.normal((4, 8))
            B = mx.random.normal((8, 3))
            C = A @ B
            mx.eval(C)
            assert C.shape == (4, 3)
            return "add, matmul OK"
        phase.tests.append(TestRunner.run_test("Basic tensor ops", test_basic, "Testing basic ops..."))

        # SDPA
        def test_sdpa():
            import mlx.core as mx
            B, H, S, D = 1, 4, 8, 16
            q = mx.random.normal((B, H, S, D))
            k = mx.random.normal((B, H, S, D))
            v = mx.random.normal((B, H, S, D))
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5)
            mx.eval(o)
            assert o.shape == (B, H, S, D), f"shape {o.shape}"
            return f"shape={o.shape}"
        phase.tests.append(TestRunner.run_test("mx.fast.scaled_dot_product_attention", test_sdpa, "Testing SDPA..."))

        # Rotary embedding values
        def test_rope_values():
            import mlx.core as mx
            from ssd.layers.rotary_embedding import RotaryEmbedding
            rope = RotaryEmbedding(16, 16, 512, 500000.0)
            positions = mx.array([5])
            q = mx.ones((1, 64))
            k = mx.ones((1, 32))
            q_out, k_out = rope(positions, q, k)
            mx.eval(q_out, k_out)
            # At position 5 with rotation, values should differ from input
            diff = float(mx.sum(mx.abs(q_out - q)).item())
            assert diff > 0.001, f"RoPE didn't change values, diff={diff}"
            return f"diff={diff:.4f}"
        phase.tests.append(TestRunner.run_test("Rotary embedding forward", test_rope_values, "Testing RoPE values..."))

        # Sampler greedy verification
        def test_sampler_greedy():
            import mlx.core as mx
            from ssd.layers.sampler import Sampler
            s = Sampler()
            # Make logits with clear max
            logits = mx.zeros((2, 100))
            logits_np = [[0.0] * 100, [0.0] * 100]
            logits_np[0][42] = 10.0
            logits_np[1][77] = 10.0
            logits = mx.array(logits_np)
            temps = mx.zeros((2,))  # greedy
            tokens = s(logits, temps)
            mx.eval(tokens)
            assert int(tokens[0].item()) == 42, f"expected 42 got {int(tokens[0].item())}"
            assert int(tokens[1].item()) == 77, f"expected 77 got {int(tokens[1].item())}"
            return "greedy argmax verified"
        phase.tests.append(TestRunner.run_test("Sampler greedy", test_sampler_greedy, "Testing sampler..."))

        # Attention full prefill pipeline
        def test_attention_pipeline():
            import mlx.core as mx
            from ssd.layers.attention import Attention
            from ssd.utils.context import set_context, reset_context
            NUM_HEADS, HEAD_DIM, NUM_KV = 4, 16, 2
            scale = HEAD_DIM ** -0.5
            attn = Attention(NUM_HEADS, HEAD_DIM, scale, NUM_KV)
            # 2 sequences: lengths 3 and 5
            set_context(
                is_prefill=True,
                cu_seqlens_q=mx.array([0, 3, 8]),
                cu_seqlens_k=mx.array([0, 3, 8]),
            )
            q = mx.random.normal((8, NUM_HEADS * HEAD_DIM))
            k = mx.random.normal((8, NUM_KV * HEAD_DIM))
            v = mx.random.normal((8, NUM_KV * HEAD_DIM))
            o = attn(q, k, v)
            mx.eval(o)
            assert o.shape == (8, NUM_HEADS * HEAD_DIM), f"shape {o.shape}"
            reset_context()
            return "2-seq prefill OK"
        phase.tests.append(TestRunner.run_test("Attention prefill pipeline", test_attention_pipeline, "Testing attention pipeline..."))

        # Mask generation
        def test_mask():
            import mlx.core as mx
            from ssd.engine.helpers.mask_helpers import get_mask_iter_i
            K, F = 3, 2
            prefix_len = 10
            mask = get_mask_iter_i(0, prefix_len, K, F)
            mx.eval(mask)
            q_len = F * (K + 1)
            expected_cols = prefix_len + (K + 1) + q_len  # prefix + glue + 1 diag
            assert mask.shape[0] == q_len, f"rows {mask.shape[0]} expected {q_len}"
            assert mask.shape[1] == expected_cols, f"cols {mask.shape[1]} expected {expected_cols}"
            return f"shape={mask.shape}"
        phase.tests.append(TestRunner.run_test("Mask generation", test_mask, "Testing masks..."))

        return phase

    # ── Phase 8: Integration ───────────────────────────────────────

    @staticmethod
    def phase_integration() -> PhaseResult:
        phase = PhaseResult(name="Integration")

        # ssd package import
        def test_ssd_import():
            try:
                import ssd
                return f"exports={len(dir(ssd))} names"
            except RuntimeError as e:
                # paths.py env var issue
                raise RuntimeError(f"ssd package import failed: {e}")
        phase.tests.append(TestRunner.run_test("ssd package import", test_ssd_import, "Importing ssd..."))

        # LLM class check
        def test_llm_class():
            from ssd.llm import LLM
            from ssd.engine.llm_engine import LLMEngine
            assert issubclass(LLM, LLMEngine), "LLM should subclass LLMEngine"
            return "LLM \u2190 LLMEngine"
        phase.tests.append(TestRunner.run_test("LLM class hierarchy", test_llm_class, "Checking LLM class..."))

        return phase


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
    def _run(cmd: list[str], check: bool = True, capture: bool = False, **kwargs) -> subprocess.CompletedProcess:
        if capture:
            return subprocess.run(cmd, capture_output=True, text=True, check=check, **kwargs)
        return subprocess.run(cmd, check=check, **kwargs)

    @staticmethod
    def _cmd_exists(name: str) -> bool:
        return shutil.which(name) is not None

    @staticmethod
    def _step(num: int, total: int, msg: str):
        bar_filled = int((num / total) * 20)
        bar_empty = 20 - bar_filled
        bar = f"{Style.BRIGHT_CYAN}{'█' * bar_filled}{Style.DIM}{'░' * bar_empty}{Style.RESET}"
        print(f"\n  [{bar}] {Style.BOLD}Step {num}/{total}{Style.RESET} {msg}")

    @staticmethod
    def _ok(msg: str):
        print(f"    {Style.CHECK} {msg}")

    @staticmethod
    def _info(msg: str):
        print(f"    {Style.BULLET} {Style.DIM}{msg}{Style.RESET}")

    @staticmethod
    def _fail(msg: str):
        print(f"    {Style.CROSS} {Style.BRIGHT_RED}{msg}{Style.RESET}")

    def run(self):
        UI.banner()
        print(f"  {Style.BOLD}{Style.BRIGHT_WHITE}One-Click Installer{Style.RESET}")
        print(f"  {Style.DIM}Setting up SSD-Metal environment for Apple Silicon{Style.RESET}")
        print()

        # Pre-flight: check we're on macOS + ARM
        if platform.system() != "Darwin":
            self._fail("SSD-Metal requires macOS with Apple Silicon.")
            sys.exit(1)

        machine = platform.machine()
        if machine not in ("arm64", "aarch64"):
            self._fail(f"Apple Silicon required, got: {machine}")
            sys.exit(1)

        total_steps = 5

        # ── Step 1: Homebrew ──────────────────────────────────────────
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

        # ── Step 2: Conda / Miniforge ─────────────────────────────────
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

        # ── Step 3: Create conda environment ──────────────────────────
        self._step(3, total_steps, f"Setting up '{self.CONDA_ENV}' environment")

        # Check if env already exists
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

        # ── Step 4: Install dependencies ──────────────────────────────
        self._step(4, total_steps, "Installing dependencies")

        # Get pip path in the conda env
        result = self._run(
            [conda_cmd, "run", "-n", self.CONDA_ENV, "python", "-c", "import sys; print(sys.executable)"],
            capture=True, check=False,
        )
        if result.returncode != 0:
            self._fail("Cannot find Python in conda environment")
            sys.exit(1)

        conda_python = result.stdout.strip()
        conda_pip = os.path.join(os.path.dirname(conda_python), "pip")

        # Install deps
        self._info(f"Installing: {', '.join(d.split('>=')[0].split('<')[0] for d in self.DEPS)}")
        try:
            self._run([conda_cmd, "run", "-n", self.CONDA_ENV, "pip", "install"] + self.DEPS)
            self._ok("All dependencies installed")
        except subprocess.CalledProcessError as e:
            self._fail(f"Dependency installation failed: {e}")
            sys.exit(1)

        # Install ssd package itself (editable)
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

        # ── Step 5: Verify ────────────────────────────────────────────
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

        # ── Done ──────────────────────────────────────────────────────
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
        print(f"  {Style.BRIGHT_WHITE}Full diagnostics:{Style.RESET}")
        print(f"    {Style.DIM}conda run -n {self.CONDA_ENV} python -m ssd{Style.RESET}")
        print()


# ─── Main Orchestrator ────────────────────────────────────────────────────────

class DiagnosticCLI:

    def run(self):
        total_start = time.perf_counter()

        # Banner
        UI.banner()

        # Hardware info
        with Spinner("Detecting hardware..."):
            hw_rows = HardwareInfo.gather()
        UI.box("Hardware", hw_rows)

        # Check MLX is available — if not, can't continue
        try:
            import mlx.core  # noqa: F401
        except ImportError:
            print(f"  {Style.BRIGHT_RED}{Style.BOLD}FATAL: MLX is not installed.{Style.RESET}")
            print(f"  {Style.DIM}Install with: pip install mlx>=0.22.0{Style.RESET}")
            print(f"  {Style.DIM}Or run: python ssd/cli.py --install{Style.RESET}")
            print()
            sys.exit(1)

        # Run all phases
        phases = []

        phase_runners = [
            (1, "Core Dependencies",  TestRunner.phase_core_deps),
            (2, "Foundation Modules",  TestRunner.phase_foundation),
            (3, "Layer Primitives",    TestRunner.phase_layers),
            (4, "Kernel Functions",    TestRunner.phase_kernels),
            (5, "Model Definitions",   TestRunner.phase_models),
            (6, "Engine Components",   TestRunner.phase_engine),
            (7, "Tensor Operations",   TestRunner.phase_tensor_ops),
            (8, "Integration",         TestRunner.phase_integration),
        ]

        for phase_num, phase_name, runner_fn in phase_runners:
            UI.phase_header(phase_num, phase_name)
            phase_result = runner_fn()
            for test in phase_result.tests:
                UI.test_result(test)
            phases.append(phase_result)

        # Summary
        total_time = time.perf_counter() - total_start
        print()
        UI.summary(phases, total_time)

        # Exit code
        total_failed = sum(p.failed for p in phases)
        sys.exit(1 if total_failed > 0 else 0)


def main():
    if "--install" in sys.argv or "install" in sys.argv:
        Installer().run()
    else:
        DiagnosticCLI().run()


if __name__ == "__main__":
    main()
