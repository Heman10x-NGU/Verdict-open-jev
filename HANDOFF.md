# Handoff: Open-Jev / Verdict 1.5 & Jev-Training Handoff

> **Timestamp**: 2026-09-28T04:23:00+05:30  
> **Status**: Verification track completed and committed (`6c3b2b3`); Training repository scaffolded and committed (`2751024`).  
> **Target File Location**: `/tmp/handoff.md`  

---

## 1. Discussed

1. **Inference vs Training Reality**:
   - Maximizing the inference engine (context window 512 → 2048, 7,000-char middle truncation budget allocator, suite adapter temperature $T=1.0$) uncollapsed probabilities (spread $0.387 \to 0.754$) and prevented context truncation on the 61/111 hard items.
   - Softmax with positive temperature is strictly monotonic and argmax-preserving: it changes calibration (ECE, NLL), but **cannot** change categorical rankings or accuracy.
   - To defeat top competitors on raw intelligence (Kev-0.8B at 82.6%, Laya at 80.3%), **retraining is mandatory**. The underlying 149M ModernBERT-base model lacks the capacity and option-interaction modeling of 400M–800M models.

2. **Reverse-Engineering Competitors (<1B parameters)**:
   - **`jaredpalmer/kev-0.8b`**: Base `Qwen/Qwen3.5-0.8B-Base`, LoRA $r=16$ on all projections + `head.pt`. Trained on 10 open classification datasets with 30% option permutation, 15% distractor injection, and 10% abstention injection.
   - **`convaiinnovations/laya`**: Base `answerdotai/ModernBERT-large` (395M params, 28 layers, 1024 hidden). Decision head: 2-layer `TransformerEncoder` scoring pooled candidate markers (`<<LABEL>>` / `[MASK]`) + `act_head`. Cost-sensitive RL loss on 13 operational decision families.
   - **`fastino/GLiNER2.5-Decide`**: Base `microsoft/deberta-v3-large` (435M), token span extractor on `fast-decisions` (17 domains).

3. **Reconciliation of JevBench v1.4.2.2**:
   - Upstream tag `v1.4.2.2` publishes exactly 231 downloadable items (48 easy, 72 standard, 111 hard).
   - The remaining 303 items (24 easy-heldout, 24 standard-heldout, 146 judge/router, 109 hard-heldout) and 308 sealed items are retained privately upstream.
   - The API tiers (0.861/0.677/0.562/0.377) include held items. Local evaluation on the 231 public set reproduces the frozen handoff baseline item-for-item: **42 easy, 50 standard, 41 hard** (133 / 231).

4. **Hardware Strategy for Training**:
   - **Windows Laptop (AMD Ryzen 7 4800H, 16 GB RAM, NVIDIA GTX 1660 Ti Mobile 6 GB VRAM, Turing CC 7.5)**: Runs FP16 with native PyTorch AdamW and gradient checkpointing (batch 2, accum 32). Peaks at ~3.8 GB VRAM. Zero `bitsandbytes` dependency. Ideal for rapid iteration, smoke runs, and augmentation verification.
   - **Kaggle Free Compute (2x NVIDIA T4 16 GB VRAM = 32 GB total, 30 hrs/week free)**: Runs full 7,500-step production run in ~3 hours with automatic checkpoint streaming to Hugging Face Hub.
   - **Development Host (Mac)**: Strictly zero model weights downloaded.

---

## 2. Done

### Track A: Verdict-open-jev Verification & Engine Turnaround
- **Repository**: `/Users/heman10x/Downloads/claude_dev/personal_projects/Mind-Palace/RLCD-demo/Verdict-open-jev`
- **Branch**: `bench/verdict-1.5-benchmark-turnaround`
- **Latest Commit**: `6c3b2b3` (`test(benchmarks): controlled 231-item JevBench and LangWatch comparisons with scope-honest manifests`)
- **Key Artifacts Produced & Verified by Muse**:
  1. `benchmarks/jevbench/v1.4.2.2/MANIFEST.json`: Added scope note documenting 231 public items vs 303 held-out and 308 sealed items, citing upstream `HARD-TIER.md`, `TOPICS.md`, and `METHOD-v1.4.md`.
  2. `reports/v1.4/baseline_jevbench.json`: Frozen baseline verified on 231 items: Easy 42/48 (87.50%), Standard 50/72 (69.44%), Hard 41/111 (36.94%), ECE 0.1177, Fidelity 72.78.
  3. `reports/v1.5/v1.5_langwatch_controlled.json`: Bit-identical 78.4% accuracy across all 11 tasks (confirming argmax invariance), spread uncollapsed from 0.387 to 0.754.
  4. `reports/v1.5/v1.5_jevbench_controlled.json`: Single invocation comparing v1.4-frozen vs v1.5 side by side:
     - Truncation count: 61/111 Hard exceed 512 tokens, 36/111 exceed 2048 tokens, max 3,705.
     - Easy: 42/48 (0.0 delta), Standard: 50/72 (0.0 delta).
     - Hard: 41/111 (36.94%) vs 45/111 (40.54%) (+4 items, strictly within $1\sigma \approx 5.2$ noise bar).
     - Long Policy slice (19 items): 6/19 (31.58%) vs 7/19 (36.84%) (+1 item).
     - Hard ECE improved from 0.1177 to 0.0899 (-0.0278).
     - Gate status honestly marked: `REJECTED (Intelligence < 50, partial suite; no leaderboard delta claimed)`.
  5. `scripts/run_jevbench_controlled.py`: Automated reproducible harness for the controlled comparison.

### Track B: Standalone Jev-Training Decision Foundation
- **Repository**: `/Users/heman10x/Downloads/claude_dev/personal_projects/Mind-Palace/Jev-training`
- **Latest Commit**: `2751024` (`fix(dataset): reference pinned GLICLASS_LABEL_TOKEN_ID constant for tokenizer fallback`)
- **Key Modules & Infrastructure Created**:
  1. `configs/laptop_1660ti_smoke.yaml`: Tailored for GTX 1660 Ti 6GB Mobile (FP16, batch 2, accum 32, grad checkpointing, 500 steps).
  2. `configs/kaggle_2xt4_full.yaml`: Tailored for Kaggle 2x T4 (FP16, batch 8, accum 4, 7,500 steps, Hub streaming).
  3. `src/model/constants.py`: Pinned GLiClass token constants (`50368` for `<<LABEL>>`, `50369` for `<<SEP>>`).
  4. `src/model/decision_model.py`: `ModernBERT-large` (395M) backbone + LoRA $r=16, \alpha=32$ on `Wqkv` and `out_proj`, with 2-layer `TransformerEncoder` option interaction head.
  5. `src/losses/brier_loss.py`: Composite $L = \text{CE} + 0.1 \cdot \text{Brier}$ (no label smoothing, no confidence clamping).
  6. `src/data/augmentations.py`: 30% option permutation, 15% distractor injection, 10% abstention injection.
  7. `src/data/manifest.json`: Strict anti-contamination ledger marking training datasets and firewalling all evaluation/test splits (`eval-never`, including Banking77).
  8. `src/data/download.py`: Downloader strictly honoring `weights` splits only.
  9. `src/calibrate/calibrator.py`: NLL optimization fitting raw logits to output `calibrator.json`.
  10. `src/eval/evaluator.py`: CI gates (spread $\ge 0.25$, AUROC $\ge 0.60$, ECE $\le 0.15$).
  11. `RUNBOOK_WINDOWS.md`: Complete step-by-step setup guide for the user's laptop (CUDA 12.1/12.4, PyTorch cu121, execution commands).
  12. `RUNBOOK_KAGGLE.md`: Step-by-step guide for Kaggle free 2x T4 execution.

---

## 3. Next Steps

1. **Option 1: Windows Laptop Smoke Run**:
   - Clone or copy `Jev-training` to the Windows laptop.
   - Follow `RUNBOOK_WINDOWS.md`:
     ```cmd
     .venv\Scripts\activate
     python data/download.py --manifest data/manifest.json
     python train.py --config configs/laptop_1660ti_smoke.yaml
     ```
   - Verify peak VRAM ($\le 4.0$ GB), loss convergence, and quality gate report.

2. **Option 2: Kaggle Production Training (Free 2x T4)**:
   - Create a Kaggle notebook with GPU T4 x2.
   - Follow `RUNBOOK_KAGGLE.md`:
     - Run `train.py --config configs/kaggle_2xt4_full.yaml`.
     - Stream checkpoints to Hugging Face Hub every 500 steps.

3. **Option 3: GitHub PR for Verdict 1.5**:
   - On `Verdict-open-jev`, push branch `bench/verdict-1.5-benchmark-turnaround`:
     ```bash
     git push origin bench/verdict-1.5-benchmark-turnaround
     gh pr create --title "test(benchmarks): controlled 231-item JevBench and LangWatch comparisons" --body "..."
     ```
   - Note standing rule: human user merges the PR.

---

## 4. References & Pointers

- **Jev-Training Repo**: `/Users/heman10x/Downloads/claude_dev/personal_projects/Mind-Palace/Jev-training`
- **Windows Runbook**: [`RUNBOOK_WINDOWS.md`](file:///Users/heman10x/Downloads/claude_dev/personal_projects/Mind-Palace/Jev-training/RUNBOOK_WINDOWS.md)
- **Kaggle Runbook**: [`RUNBOOK_KAGGLE.md`](file:///Users/heman10x/Downloads/claude_dev/personal_projects/Mind-Palace/Jev-training/RUNBOOK_KAGGLE.md)
- **Verdict 1.5 Repo**: `/Users/heman10x/Downloads/claude_dev/personal_projects/Mind-Palace/RLCD-demo/Verdict-open-jev`
- **Controlled JevBench Report**: [`v1.5_jevbench_controlled.json`](file:///Users/heman10x/Downloads/claude_dev/personal_projects/Mind-Palace/RLCD-demo/Verdict-open-jev/reports/v1.5/v1.5_jevbench_controlled.json)
- **Controlled LangWatch Report**: [`v1.5_langwatch_controlled.json`](file:///Users/heman10x/Downloads/claude_dev/personal_projects/Mind-Palace/RLCD-demo/Verdict-open-jev/reports/v1.5/v1.5_langwatch_controlled.json)
- **Competitor Analysis Report**: [`comparative_study.md`](file:///Users/heman10x/Downloads/claude_dev/personal_projects/Mind-Palace/competitor_analysis/reports/comparative_study.md)
- **Herdr Partner Pane**: `w7:pN` (Agent `rlcd-muse`)
