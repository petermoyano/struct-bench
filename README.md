# Structural Prediction Precision Benchmark

A controlled benchmark to determine whether reducing numerical precision during protein structure prediction inference preserves the practical ranking of protein variant candidates.

## Background: Why This Benchmark Exists

This project supports a cancer research lab located in Mendoza, Argentina that runs a multi-stage computational protein design pipeline. Their workflow (simplified) looks like this:

| Stage | Tool | What it does |
|:-----:|------|--------------|
| 1 | [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) | Generates thousands of 3D protein backbone candidates (GPU, diffusion model) |
| 2 | [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) | Assigns amino acid sequences to each backbone (~1s per design) |
| 3 | **Structure prediction (AlphaFold2)** | Re-predicts the 3D structure from each sequence to validate the design |
| 4 | Manual review in [PyMOL](https://pymol.org/) | Visual inspection, final selection for wet-lab experiments |

**Stage 3 is the bottleneck.** AlphaFold2 is slow and expensive, especially when evaluating thousands of candidate sequences. The researchers want to know if they can speed up this filtering stage by using lower numerical precision or by substituting faster structure prediction models, without changing which candidates end up in the top-k selection.

For more context on the full pipeline, see:
- [RFdiffusion repo](https://github.com/RosettaCommons/RFdiffusion) and its [README](https://github.com/RosettaCommons/RFdiffusion/blob/main/README.md)
- [dl_binder_design repo](https://github.com/nrbennet/dl_binder_design) (contains scripts for ProteinMPNN + AF2 filtering)
- [RFdiffusion paper](https://www.biorxiv.org/content/10.1101/2022.12.09.519842v1)
- [ProteinMPNN paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9997061/)

### Scale reference

The lab's typical campaign: ~10,000 backbones (Stage 1) become ~20,000 sequences (Stage 2), which must all pass through Stage 3. Only a few hundred survive filtering. This makes Stage 3 the most compute-intensive step.

---

## The Core Question

> **Does lowering inference precision (FP32 → BF16 → INT8) change which protein variants end up in the top-k, or does the ranking stay stable enough that cheaper inference is practically useful?**

This is a **ranking stability** benchmark, not an accuracy benchmark. The researchers explicitly stated: "I don't care if the absolute score changes a little — I care if the practical decision changes."

---

## Scope

- **Monomeric proteins only** (single-chain sequences, not multi-chain complexes)
- **Two models** to benchmark:
  - **ESMFold** — fast, sequence-only, no MSA needed
  - **OpenFold** — slower, uses MSA (multiple sequence alignment), higher accuracy
- **No biology expertise required** — the researchers handle biological interpretation; we build the infrastructure
- **No model training** — inference only, using pre-trained checkpoints

---

## The Two Models

### ESMFold

[ESMFold](https://github.com/facebookresearch/esm) is a protein structure predictor from Meta (Facebook AI Research). It is built on top of **ESM-2**, a Transformer protein language model trained on ~250 million protein sequences (it is, genuinely, an LLM — but for protein sequences instead of natural language).

Key properties:
- **Input**: a single amino acid sequence (string of letters like `MKTVRQERLK...`)
- **Output**: a 3D structure (`.pdb` file) + per-residue confidence scores (pLDDT)
- **Speed**: fast — no MSA computation needed, runs in seconds per sequence
- **Install**: via HuggingFace `transformers` (recommended) or `pip install fair-esm[esmfold]` (legacy, requires Python <= 3.9)
- **Repo**: https://github.com/facebookresearch/esm (archived Aug 2024, still functional)
- **PyPI package**: `fair-esm` (legacy) / `transformers` (recommended — no `openfold` dependency, works on Python 3.10+)
- **HuggingFace model**: [`facebook/esmfold_v1`](https://huggingface.co/facebook/esmfold_v1) via `EsmForProteinFolding`
- **Models available**: ESM-2 (8M to 15B params), ESMFold v0/v1

The repo is archived but the weights remain available on HuggingFace Hub. This benchmark uses the `transformers` port, which has simplified dependencies and is actively maintained. Context7 has 262 code snippets indexed under `/facebookresearch/esm`.

### OpenFold

[OpenFold](https://github.com/aqlaboratory/openfold) is a faithful, trainable PyTorch reimplementation of DeepMind's AlphaFold2. It predicts protein structures from sequences but (unlike ESMFold) also uses MSA — evolutionary comparisons with related proteins — for higher accuracy.

Key properties:
- **Input**: amino acid sequence + MSA (precomputed alignments) + template structures
- **Output**: 3D structure (`.pdb` file) + per-residue confidence scores (pLDDT) + pAE (predicted alignment error)
- **Speed**: slower than ESMFold (MSA computation is a significant part of the cost)
- **Install**: conda environment + pip (see [OpenFold docs](https://openfold.readthedocs.io/en/latest/))
- **Repo**: https://github.com/aqlaboratory/openfold
- **Native BF16 support**: `--precision bf16` flag (confirmed, ~1.5x speedup)
- **Native TensorRT support**: `--trt_mode run` (additional speedup)
- **DeepSpeed integration**: memory-efficient attention kernels
- **Docs**: https://openfold.readthedocs.io/en/latest/

Context7 has 251 code snippets indexed under `/aqlaboratory/openfold`.

---

## What We Measure

For each **variant x model x precision** combination, collect:

| Metric | Description | Why |
|--------|-------------|-----|
| **Mean pLDDT** | Average per-residue confidence (0-100) | Primary ranking score — higher means the model is more confident in the predicted structure |
| **Per-residue pLDDT** | Array of confidence per amino acid | Diagnose whether precision changes are spread out or localized to specific regions |
| **Inference time** | Wall-clock seconds | Quantify the speedup from lower precision |
| **Peak GPU memory** | MB/GB | Determine if lower precision enables larger batches or fits on smaller GPUs |

### How We Compare Rankings

For each reduced-precision run vs. the FP32 reference:

| Metric | What it tells us |
|--------|-----------------|
| **Spearman rank correlation (rho)** | Global ranking agreement (-1 to 1, 1 = identical ranking) |
| **Kendall tau** | Pairwise concordance of ranking |
| **Top-k overlap** | Fraction of variants in the top 1%, 5%, 10% that are the same in both conditions — **this is the primary metric the researchers care about** |
| **Score scatter plot** | Visual check: FP32 score (x-axis) vs reduced score (y-axis) — should be a tight diagonal |

---

## Precision Levels

| Precision | Bits | Priority | ESMFold approach | OpenFold approach |
|-----------|------|----------|------------------|-------------------|
| **FP32** | 32 | Reference baseline | Default (`model.eval().cuda()`) | Default |
| **BF16** | 16 | Required | `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` or `model.to(torch.bfloat16)` | Native flag: `--precision bf16` |
| **INT8** | 8 | Required (if feasible) | `bitsandbytes` or `torch.quantization` post-training quantization | Custom quantization (not natively supported) |
| **INT4** | 4 | Optional | `bitsandbytes` 4-bit quantization | Custom quantization (not natively supported) |

### Technical notes on precision

- **FP32** (float32): 32-bit floating point. Full precision. This is the "gold standard" reference.
- **BF16** (bfloat16): 16-bit. Same dynamic range as FP32 but ~3 decimal digits of precision. Generally safe for neural network inference. ~1.5x speedup on Ampere+ GPUs.
- **FP16** (float16): 16-bit. Narrower range than BF16. Can cause numerical instability (overflow/underflow). OpenFold docs explicitly say "fp16 is not recommended." **We should use BF16, not FP16.**
- **INT8**: 8-bit integer. Aggressive quantization. Typically applied via post-training quantization using `bitsandbytes` or PyTorch's native quantization APIs. May introduce measurable accuracy changes.
- **INT4**: 4-bit integer. Very aggressive. Used in LLM inference (GPTQ, QLoRA) but less explored for structure prediction. Treat as exploratory.

---

## Input Data Requirements

The benchmark requires a **fixed set of protein sequence variants** provided as a **FASTA file**. A FASTA file is a simple text format:

```
>variant_001_wild_type
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG
>variant_002_A45V
MKTVRQERLKSIVRILERSKEPVSGVQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG
>variant_003_G78D
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLDYNIVATPRGYVLAGG
```

Each entry has a header line (starts with `>`) followed by the amino acid sequence (string of single-letter codes). A "variant" is a mutated version of a base protein (e.g., one amino acid changed at a specific position).

### What to request from the researchers

- [ ] FASTA file with the variant sequences (or a base protein + mutation list to generate one)
- [x] Approximate number of variants — **tens of thousands** (this is the core motivation for the benchmark; at this scale, even a 1.5× speedup saves hours of compute per campaign)
- [x] Available GPU hardware — **NVIDIA RTX 4060, 8 GB VRAM** (Ampere architecture; supports native BF16; 8 GB is tight for long sequences and will be a hard constraint for OpenFold with MSA — see note below)
- [x] Whether ESMFold / OpenFold are already installed — **ESMFold: yes; OpenFold: not yet installed**
- [ ] Preferred ranking score (default assumption: **mean pLDDT**)

> **Hardware note (RTX 4060, 8 GB):** ESMFold weights alone are ~2.7 GB; with activations during inference of longer sequences (200+ residues), peak VRAM usage can reach 6–8 GB. Very long sequences may require chunked inference. OpenFold with full MSA is significantly more memory-hungry and may not fit in 8 GB for average-length proteins without memory optimizations (DeepSpeed, chunked attention). This is a known constraint for this proof-of-concept phase — cloud GPUs (A100 40/80 GB) are the target for the full-scale run.

If the researchers don't have a ready variant set, a reasonable fallback is to pick a well-characterized protein from the [RCSB PDB](https://www.rcsb.org/) (e.g., GFP, lysozyme, or a kinase domain) and generate systematic single-point mutations. This produces hundreds of variants from one sequence and is standard practice for benchmarking.

---

## Expected Deliverables

1. **Benchmark runner scripts**
   - `run_esmfold.py` — runs ESMFold on a FASTA at a specified precision, outputs per-variant metrics
   - `run_openfold.py` — same for OpenFold
   - Both must produce identical output formats for fair comparison

2. **Per-variant results table** (CSV/Parquet, one row per variant per model per precision)
   - Columns: `variant_id`, `model`, `precision`, `mean_plddt`, `inference_time_s`, `peak_memory_mb`
   - Separate file for per-residue pLDDT arrays (e.g., NumPy `.npz` or pickle)

3. **Ranking analysis script**
   - `analyze_rankings.py` — loads all results, computes:
     - Spearman rho and Kendall tau per (model, precision) vs. (model, FP32)
     - Top-k overlap at k = 1%, 5%, 10%
     - Score scatter plots
     - Summary table

4. **Report / conclusion**
   - Per-model practical recommendation
   - Example: "ESMFold BF16 preserves 98% top-5% overlap while being 1.8x faster and using 40% less memory"
   - Or: "OpenFold INT8 degrades top-1% overlap to 75% — not recommended for candidate selection"

---

## Suggested Project Structure

```
struct-bench/
  README.md
  .python-version             # pyenv — locks Python 3.10.14 for this project
  .gitignore
  requirements-esm.txt        # pip deps for the ESMFold venv
  config.yaml                 # variant FASTA path, output dir, precision levels, etc.
  .venv-esm/                  # ESMFold virtual environment (gitignored)
  data/
    variants.fasta            # input sequences (provided by researchers or generated)
  scripts/
    run_esmfold.py            # ESMFold benchmark runner
    run_openfold.py           # OpenFold benchmark runner
    analyze_rankings.py       # ranking comparison and visualization
    generate_variants.py      # (optional) generate synthetic variants from a base protein
  results/                    # output directory (gitignored, generated by runners)
    esmfold_fp32.csv
    esmfold_bf16.csv
    openfold_fp32.csv
    ...
  notebooks/
    exploration.ipynb         # (optional) interactive analysis
```

---

## Key Technical References

### Models under test

| Model | Repo | Docs | Install | Context7 ID |
|-------|------|------|---------|-------------|
| ESMFold | [facebookresearch/esm](https://github.com/facebookresearch/esm) | [HuggingFace docs](https://huggingface.co/docs/transformers/model_doc/esm) | `pip install transformers torch accelerate` | `/facebookresearch/esm` (262 snippets) |
| OpenFold | [aqlaboratory/openfold](https://github.com/aqlaboratory/openfold) | [ReadTheDocs](https://openfold.readthedocs.io/en/latest/) | conda env + pip | `/aqlaboratory/openfold` (251 snippets) |

### Upstream pipeline tools (for context only — not part of this benchmark)

| Tool | Repo | Role in pipeline |
|------|------|-----------------|
| RFdiffusion | [RosettaCommons/RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) | Stage 1: backbone generation (diffusion model) |
| ProteinMPNN | [dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN) | Stage 2: sequence design (graph neural network) |
| dl_binder_design | [nrbennet/dl_binder_design](https://github.com/nrbennet/dl_binder_design) | Orchestration scripts for ProteinMPNN + AF2 filtering |
| AlphaFold2 | [google-deepmind/alphafold](https://github.com/google-deepmind/alphafold) | Stage 3: structure validation (what this benchmark aims to accelerate) |

### Papers

- [RFdiffusion paper](https://www.biorxiv.org/content/10.1101/2022.12.09.519842v1) — the backbone generation method
- [ProteinMPNN paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9997061/) — sequence design
- [Improving de novo protein binder design with deep learning](https://www.nature.com/articles/s41467-023-38328-5) — binder design pipeline
- [ESM-2 / ESMFold paper](https://www.science.org/doi/abs/10.1126/science.ade2574) — the ESMFold model
- [OpenFold paper](https://www.biorxiv.org/content/10.1101/2022.11.20.517210) — the OpenFold reimplementation

### Alternative / future tools (noted for awareness, not in scope)

- [RFdiffusion2](https://github.com/RosettaCommons/RFdiffusion2) — atom-level enzyme scaffolding (Sept 2025)
- [RFdiffusion3 via Foundry](https://rosettacommons.github.io/foundry/models/rfd3/index.html) — unified atom-level design (Dec 2025)
- [nf-binder-design](https://github.com/Australian-Protein-Design-Initiative/nf-binder-design) — Nextflow pipeline for full binder design workflow
- [Protenix](https://github.com/bytedance/Protenix) — ByteDance structure prediction
- [OpenFold3-preview](https://github.com/aqlaboratory/openfold-3) — AlphaFold3 reproduction (Context7 ID: `/aqlaboratory/openfold-3`, 245 snippets)

---

## Glossary (for AI engineers, not biologists)

| Term | Meaning |
|------|---------|
| **Protein sequence** | A string of amino acid letters (e.g., `MKTVRQERLK...`). Like a sentence, but the alphabet has 20 letters. |
| **Variant** | A mutated version of a protein sequence (e.g., one letter changed). Each variant is a candidate to evaluate. |
| **Structure prediction** | Given a sequence, predict the 3D shape the protein will fold into. This is what ESMFold and OpenFold do. |
| **pLDDT** | Per-residue confidence score (0-100). The model's confidence that each amino acid is correctly positioned. Higher = better. **Mean pLDDT across all residues is the primary ranking score.** |
| **pAE** | Predicted Aligned Error. Confidence in the relative positions of residue pairs. Used for complex (multi-chain) evaluation. Less relevant here since we're doing monomers. |
| **MSA** | Multiple Sequence Alignment. A comparison of the input protein against evolutionarily related proteins. OpenFold uses MSAs; ESMFold does not. MSA computation is a major time cost. |
| **FASTA** | Simple text file format for protein/DNA sequences. Header line (`>name`) followed by the sequence. |
| **PDB** | File format for 3D protein structures. Contains atom coordinates, chain IDs, residue names. |
| **Monomer** | A single protein chain. This benchmark is scoped to monomers only (not multi-chain complexes). |
| **Backbone** | The N-Ca-C atom skeleton of a protein. RFdiffusion generates backbones; ESMFold/OpenFold predict full structures. |
| **Residue** | One amino acid in the chain. Analogous to a "token" in NLP. |
| **Top-k** | The top k% of variants by score. The key question: does the same set of variants appear in the top-k across precision levels? |
| **Checkpoint** | Saved neural network weights (`.pt` file). Each model ships pre-trained checkpoints. |
| **BF16 / FP16 / INT8 / INT4** | Numerical precision formats. Lower bits = faster but potentially less accurate. See Precision Levels section above. |

---

## Getting Started

### Prerequisites

- **OS**: Linux or WSL2 on Windows (tested on Ubuntu 24.04 under WSL2)
- **GPU**: NVIDIA GPU with CUDA support (developed on RTX 4060, 8 GB; cloud A100 40/80 GB for full-scale runs)
- **[pyenv](https://github.com/pyenv/pyenv)**: manages the Python version (similar to `nvm` for Node.js)

### Why two environments?

This benchmark tests two models with **incompatible dependency trees**:

| Environment | Model | Managed by | Python | Key deps |
|-------------|-------|-----------|--------|----------|
| `.venv-esm` | ESMFold (via HuggingFace `transformers`) | **pyenv + venv + pip** | 3.10 | `transformers`, `torch`, `accelerate` |
| `bench-openfold` | OpenFold | **conda / mamba** | Set by `environment.yml` | CUDA toolkit, HHsuite, OpenMM, DeepSpeed |

ESMFold is accessed through the [HuggingFace `transformers` port](https://huggingface.co/facebook/esmfold_v1) (`EsmForProteinFolding`), which is actively maintained, works on modern Python, and has no dependency on the `openfold` pip package. The original `fair-esm[esmfold]` route requires Python <= 3.9 and fragile pinned dependencies — we avoid it.

Both environments run scripts from **this single repo**. The output CSV format is identical regardless of which model produced it, so the analysis script (`analyze_rankings.py`) works in either environment.

### Step 1 — Install pyenv and Python 3.10

```bash
# Install build dependencies (Ubuntu/Debian)
sudo apt update && sudo apt install -y make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
  libffi-dev liblzma-dev

# Install pyenv
curl https://pyenv.run | bash

# Add to ~/.zshrc (or ~/.bashrc if using bash)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc

# Install Python 3.10 and set it for this project
pyenv install 3.10.14
cd /path/to/struct-bench
pyenv local 3.10.14    # creates .python-version file
```

### Step 2 — Create the ESMFold virtual environment

```bash
python -m venv .venv-esm
source .venv-esm/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-esm.txt   # once available
```

### Step 3 — (Later) Create the OpenFold conda environment

OpenFold requires conda/mamba due to non-Python dependencies (C++ compilers, CUDA toolkit, bioinformatics binaries). This will be set up when we reach the OpenFold phase of the benchmark.

```bash
# Install Miniforge: https://github.com/conda-forge/miniforge
git clone https://github.com/aqlaboratory/openfold.git /path/to/openfold
cd /path/to/openfold
mamba env create -n bench-openfold -f environment.yml
conda activate bench-openfold
```

### Step 4 — Run the benchmark

```bash
# ESMFold (activate the venv first)
source .venv-esm/bin/activate
python scripts/run_esmfold.py --precision fp32
python scripts/run_esmfold.py --precision bf16

# OpenFold (activate the conda env first)
conda activate bench-openfold
python scripts/run_openfold.py --precision fp32
python scripts/run_openfold.py --precision bf16

# Analyze results (works in either environment)
python scripts/analyze_rankings.py
```

### Everyday cheatsheet

| Goal | Command |
|------|---------|
| Enter the project | `cd ~/struct-bench` (pyenv auto-selects Python 3.10) |
| Activate ESMFold env | `source .venv-esm/bin/activate` |
| Activate OpenFold env | `conda activate bench-openfold` |
| Deactivate any env | `deactivate` (venv) or `conda deactivate` (conda) |
| Check active Python | `python --version` and `which python` |
| Install a package | `pip install something` |
| Install from requirements | `pip install -r requirements-esm.txt` |
| Check GPU visibility | `python -c "import torch; print(torch.cuda.is_available())"` |

### Files not tracked by git

The following should be in `.gitignore`:

```
.venv-esm/
results/
*.pyc
__pycache__/
.python-version
```
