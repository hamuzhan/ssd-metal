<h1 align="center">SSD-Metal</h1>
<h3 align="center">Speculative Speculative Decoding for Apple Silicon</h3>

<h4 align="center">
  <a href="https://arxiv.org/pdf/2603.03251">Paper</a>
</h4>

<p align="center">
  <img width="800"
       src="assets/ssd fig1 readme.png" />
</p>

> *"In all fictions, each time a man meets diverse alternatives, he chooses one and eliminates the others; in the work of the almost unfathomable Ts'ui Pên, he chooses — simultaneously — all of them."*
>
> — Jorge Luis Borges, "The Garden of Forking Paths" (1941)

**SSD is a new LLM inference algorithm. It is exact, and it is extremely fast.**

SSD is a new type of speculative decoding (SD). In normal SD, a small and fast model guesses the next few tokens that a larger slower model may generate, and the large model then verifies them in one forward pass: drafting and verification happen one after the other on the same hardware.

In SSD, they happen in parallel, on distinct hardware. The small model anticipates likely verification outcomes in advance, and speculates for all of them at once. If it guessed correctly, the speculation can be returned immediately so drafting overhead is eliminated entirely.

### Apple Silicon Port

This fork ports the entire SSD engine from CUDA/PyTorch to **Apple MLX/Metal**, exploiting Apple Silicon's unified memory architecture for zero-copy draft/target synchronization. All NVIDIA-specific dependencies (FlashInfer, Triton, sgl-kernel, NCCL, CUDA Graphs) are replaced with native MLX primitives and Metal compute shaders.

| Feature | CUDA (original) | MLX/Metal (this fork) |
|---|---|---|
| Tensor ops | `torch.Tensor` (CUDA) | `mx.array` (unified memory) |
| NN modules | `torch.nn.Module` | `mlx.nn.Module` |
| Attention | FlashInfer + Triton | `mx.fast.scaled_dot_product_attention` |
| Graph compilation | CUDA Graphs (903 lines) | `mx.compile()` (shape-polymorphic) |
| Draft/Target sync | NCCL across GPUs | Direct function calls (zero-copy) |
| Tensor parallelism | Multi-GPU all-reduce | Not needed (single unified GPU) |
| RoPE | Precomputed cos/sin cache | `mx.fast.rope()` (Metal-accelerated) |

This engine supports:
- A reference implementation of the SSD algorithm on Apple Silicon
- Optimized SD and autoregressive baselines
- Qwen3 + Llama3 model families
- PagedAttention with prefix caching
- Built-in diagnostic CLI (`python -m ssd`)

---

## Requirements

| Requirement | Minimum | Notes |
|---|---|---|
| **Mac** | Apple Silicon (M1/M2/M3/M4) | Intel Macs are not supported |
| **macOS** | 14.0+ (Sonoma) | Metal 3 support required |
| **Python** | 3.11 – 3.12 | 3.13 not yet tested |
| **Homebrew** | Latest | For installing conda |
| **Conda** | Miniforge recommended | Isolated environment |
| **MLX** | >= 0.22.0 | Apple's ML framework |
| **RAM** | 16 GB+ | 32 GB+ recommended for 8B models, 64 GB+ for larger |

---

## Setup

> We **strongly recommend** using a conda environment. Installing into system Python can cause dependency conflicts. Conda provides a clean, isolated environment.

### Option A: One-Click Install (recommended)

The CLI has a built-in installer that handles everything automatically:

```bash
git clone https://github.com/tanishqkumar/ssd && cd ssd
python3 ssd/cli.py --install
```

This will:
1. Check for Homebrew (install if missing)
2. Check for Conda/Miniforge (install if missing)
3. Create a `ssd-metal` conda environment with Python 3.12
4. Install MLX and all dependencies
5. Run the full diagnostic suite to verify everything works

### Option B: Manual Setup

#### 1. Install Homebrew (if not installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### 2. Install Miniforge (if conda not installed)

```bash
brew install miniforge
conda init zsh  # or: conda init bash
```

Restart your terminal after `conda init`.

#### 3. Create conda environment

```bash
conda create -n ssd-metal python=3.12 -y
conda activate ssd-metal
```

#### 4. Clone and install

```bash
git clone https://github.com/tanishqkumar/ssd && cd ssd
pip install -e .
```

This installs: `mlx`, `transformers`, `safetensors`, `xxhash`, `numpy`, `tqdm`, `tiktoken`.

#### 5. Set environment variables

```bash
export SSD_HF_CACHE=/path/to/huggingface/hub
export SSD_DATASET_DIR=/path/to/processed_datasets
```

`SSD_HF_CACHE` should point to the HuggingFace **hub** directory — the one containing `models--org--name/` subdirectories (e.g. `~/.cache/huggingface/hub`).

Add these to your `~/.zshrc` or `~/.bashrc` to make them permanent.

#### 6. Verify installation

```bash
python -m ssd
```

---

## Diagnostics

SSD-Metal includes a professional diagnostic CLI that validates every component:

```bash
python -m ssd
```

```
 ███████╗███████╗██████╗       ███╗   ███╗███████╗████████╗ █████╗ ██╗
 ██╔════╝██╔════╝██╔══██╗      ████╗ ████║██╔════╝╚══██╔══╝██╔══██╗██║
 ███████╗███████╗██║  ██║█████╗██╔████╔██║█████╗     ██║   ███████║██║
 ╚════██║╚════██║██║  ██║╚════╝██║╚██╔╝██║██╔══╝     ██║   ██╔══██║██║
 ███████║███████║██████╔╝      ██║ ╚═╝ ██║███████╗   ██║   ██║  ██║███████╗
 ╚══════╝╚══════╝╚═════╝       ╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝

              Speculative Decoding Engine for Apple Silicon
                        v0.3.0  ·  Diagnostics
```

The diagnostic runs 39 tests across 8 phases:

| Phase | What it checks |
|---|---|
| Core Dependencies | mlx, transformers, safetensors, xxhash, numpy, tqdm, tiktoken |
| Foundation Modules | Config, paths, SamplingParams, Sequence |
| Layer Primitives | RMSNorm, SiluAndMul, RoPE, QKV/GateUp/Row linear, Embedding, LMHead, Sampler, Attention |
| Kernel Functions | Metal KV cache store kernel |
| Model Definitions | LlamaForCausalLM, Qwen3ForCausalLM, Eagle3DraftForCausalLM |
| Engine Components | ModelRunner, DraftRunner, Scheduler, Verifier, SpeculatorUnified |
| Tensor Operations | Basic ops, SDPA, rotary embeddings, sampling, attention pipeline, mask generation |
| Integration | Full package import, LLM class hierarchy |

---

## Usage

### Download models

```bash
# Requires: pip install huggingface_hub datasets
python scripts/download_from_hf.py llama
```

### Benchmarks

```bash
cd bench

# Autoregressive — Llama 8B
python -O bench.py --llama --size 8 --b 1 --temp 0 --numseqs 128 --output_len 512 --all

# Sync speculative decoding — 8B target + 1B draft, k=6
python -O bench.py --llama --size 8 --spec --k 6 --b 1 --temp 0 --numseqs 128 --output_len 512 --all

# Async speculative decoding (SSD) — k=7, f=3
python -O bench.py --llama --size 8 --spec --async --k 7 --f 3 --b 1 --temp 0 --numseqs 128 --output_len 512 --all
```

Use `--qwen --size 32` for Qwen models. See `bench/bench.py` for full args.

### Chat

Interactive streaming chat:

```bash
cd bench

# Autoregressive
python -O chat.py --ssd

# Sync spec decode, k=6
python -O chat.py --ssd --spec --k 6

# Async spec decode (SSD), k=7, f=3
python -O chat.py --ssd --spec --async --k 7 --f 3 --metrics
```

---

## Technical Overview

### Unified Memory Architecture

Apple Silicon's unified memory eliminates the need for explicit data transfers between CPU and GPU. In the original CUDA implementation, draft and target models run on separate GPUs communicating via NCCL `send`/`recv`. In SSD-Metal, both models share the same address space:

```
Apple Silicon Unified Memory
┌──────────────────────────────────────────────┐
│  Target Model Weights ──┐                    │
│  Draft Model Weights  ──┤ all mx.array,      │
│  Target KV Cache      ──┤ backed by Metal    │
│  Draft KV Cache       ──┤ buffers            │
│  Tree Cache           ──┘                    │
│                                              │
│  Execution (single process):                 │
│  1. target.run() → logits, hidden_states     │
│  2. draft.speculate(hidden_states) → tokens  │
│     ↑ zero-copy: same mx.array reference     │
│  3. verify(logits_target, logits_draft)      │
│  4. postprocess → append accepted tokens     │
└──────────────────────────────────────────────┘
```

### What was removed

- **903-line CUDA graph capture/replay system** — replaced by `mx.compile()` (shape-polymorphic, no bucket management)
- **NCCL multi-process communication** — direct method calls in a single process
- **FlashInfer + Triton kernels** — `mx.fast.scaled_dot_product_attention` + simple Metal kernels
- **Tensor parallelism** (QKVParallel, RowParallel, all-reduce) — plain `mlx.nn.Linear` (single GPU)
- **SharedMemory IPC** — unnecessary with unified memory
- **~2,000 lines of NVIDIA-specific infrastructure**

### Key MLX synchronization points

MLX uses lazy evaluation — computations are only executed when results are needed. Explicit `mx.eval()` calls are required:

1. After model forward → `mx.eval(logits)` before sampling
2. After sampling → `mx.eval(token_ids)` before `.tolist()`
3. After verification → `mx.eval(recovery_tokens)` before postprocessing

---

## Citation

Speculative Speculative Decoding will appear at ICLR 2026.

```bibtex
@misc{kumar2026speculativespeculativedecoding,
      title={Speculative Speculative Decoding},
      author={Tanishq Kumar and Tri Dao and Avner May},
      year={2026},
      eprint={2603.03251},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.03251},
}
```

## History

[![Star History Chart](https://api.star-history.com/svg?repos=tanishqkumar/ssd&type=Date)](https://star-history.com/#tanishqkumar/ssd&Date)
