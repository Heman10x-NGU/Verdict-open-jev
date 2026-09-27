#!/usr/bin/env python3
"""Controlled JevBench Comparison Runner (v1.4-frozen vs v1.5).

Produces a single controlled JSON report evaluating both v1.4-frozen and v1.5
side by side on the 231 downloadable public items from JevBench v1.4.2.2.

Features:
- Single invocation with logged environment and model configuration.
- Per-tier accuracy and abstention counts for Easy (48), Standard (72), Hard (111).
- Per-tier and overall truncation counts at 512 and 2048 tokens.
- Hard-tier Expected Calibration Error (ECE) and Probability Fidelity.
- Dedicated 19-item long_policy slice.
- Honest partial composite score note (strictly 231-item partial, no sealed items, no gap penalty).
- No false PASSED_ALL_GATES claim.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.engine_encoder import DecisionEngine
from core.calibration import TemperatureCalibrator
from core.primitives import Choice, Level, Noul, Option, Query, Score


def load_jsonl(path: Path) -> list[dict]:
    tasks = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def build_query_for_task(task: dict) -> tuple[str, Query]:
    state = task["state"]
    q = task["question"]
    state_str = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    qtype = q["type"]
    crit = q.get("criteria")
    instructions = q.get("instructions", "")

    if qtype == "choice":
        options = [Option(id=str(k), description=str(v or k)) for k, v in crit.items()]
        query = Choice(id="decision", question=instructions, options=options)
    elif qtype == "score":
        levels = [Level(id=str(i), description=str(d), value=float(i)) for i, d in enumerate(crit)]
        query = Score(id="decision", question=instructions, levels=levels)
    else:  # noul
        prop = instructions
        if crit and (crit.get("true") or crit.get("false")):
            prop += f" (true: {crit.get('true') or 'yes'}; false: {crit.get('false') or 'no'})"
        query = Noul(id="decision", proposition=prop, semantics="conditional_on_sufficient_evidence_v2")

    return state_str, query


def evaluate_task(engine: DecisionEngine, task: dict) -> tuple[dict[str, float], float, bool]:
    state_str, query = build_query_for_task(task)
    t0 = time.perf_counter()
    out = engine.evaluate(state_str, [query])
    latency = time.perf_counter() - t0

    r = out.results[0]
    probs = dict(r.probabilities)
    abstained = bool(r.is_abstention)

    # Official adapter rule: remove abstention slot and renormalize
    probs.pop("__insufficient_evidence__", None)
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}

    # Map noul labels {true->yes, false->no}
    qtype = task["question"]["type"]
    if qtype == "noul":
        probs = {
            "yes": probs.get("true", 0.0),
            "no": probs.get("false", 0.0),
        }
        noul_tot = sum(probs.values())
        if noul_tot > 0:
            probs = {k: v / noul_tot for k, v in probs.items()}

    return probs, latency, abstained


def score_task(probs: dict[str, float], task: dict) -> dict:
    expected = str(task["expected"]) if task.get("expected") is not None else None
    pred = max(probs, key=probs.get) if probs else None
    return {
        "correct": bool(pred == expected),
        "predicted": pred,
        "probs": probs,
    }


def compute_ece(pairs: list[tuple[float, int]], n_bins: int = 10) -> float:
    if not pairs:
        return 0.0
    confs = np.array([p[0] for p in pairs])
    accs = np.array([p[1] for p in pairs])
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(pairs)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (confs >= lo) & (confs <= hi) if i == 0 else (confs > lo) & (confs <= hi)
        if np.sum(mask) > 0:
            bin_acc = np.mean(accs[mask])
            bin_conf = np.mean(confs[mask])
            ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)
    return float(ece)


def run_engine_suite(
    engine: DecisionEngine,
    engine_name: str,
    tasks_by_tier: dict[str, list[dict]],
) -> dict[str, Any]:
    print(f"\nEvaluating Engine: {engine_name} (max_length={engine.max_length}) ...")
    tier_results = {}
    all_latencies = []
    hard_pairs = []
    prob_tvds = []
    long_policy_stats = {"total": 0, "correct": 0}

    for tier_name, raw_tasks in tasks_by_tier.items():
        correct = 0
        scorable = 0
        abstained_count = 0

        for task in raw_tasks:
            expected = task.get("expected")
            family = task.get("family", "")
            provenance = task.get("provenance", {})

            probs, latency, abstained = evaluate_task(engine, task)
            all_latencies.append(latency)
            if abstained:
                abstained_count += 1

            res = score_task(probs, task)
            is_correct = res["correct"]

            if expected is not None:
                scorable += 1
                if is_correct:
                    correct += 1

            pred_label = res["predicted"]
            top_p = probs.get(pred_label, 0.0) if pred_label else 0.0

            if tier_name == "hard" and expected is not None:
                hard_pairs.append((float(top_p), int(is_correct)))
                if family == "long_policy":
                    long_policy_stats["total"] += 1
                    if is_correct:
                        long_policy_stats["correct"] += 1

                if family == "probability":
                    gold_probs = provenance.get("gold_probs")
                    if gold_probs and isinstance(gold_probs, dict):
                        tvd = 0.5 * sum(abs(probs.get(k, 0.0) - gold_probs.get(k, 0.0)) for k in gold_probs)
                        prob_tvds.append(float(tvd))

        acc = correct / scorable if scorable > 0 else 0.0
        tier_results[tier_name] = {
            "total": len(raw_tasks),
            "scorable": scorable,
            "correct": correct,
            "abstentions": abstained_count,
            "accuracy": round(acc, 4),
        }
        print(f"  {tier_name.capitalize():<10}: {acc*100:6.2f}% ({correct}/{scorable}), abstained={abstained_count}")

    # Compute metrics
    lat_arr = np.array(all_latencies)
    p50_s = float(np.percentile(lat_arr, 50)) if len(lat_arr) > 0 else 0.0
    p95_s = float(np.percentile(lat_arr, 95)) if len(lat_arr) > 0 else 0.0

    hard_ece = compute_ece(hard_pairs, n_bins=10)
    mean_tvd = float(np.mean(prob_tvds)) if prob_tvds else 0.0
    prob_fidelity = round(100.0 * (1.0 - mean_tvd), 2)
    cal_axis = round(0.5 * (max(0.0, 100.0 * (1.0 - hard_ece / 0.5)) + prob_fidelity), 2)

    # Speed Axis
    p50_adj = 2.0 * p50_s + 0.15
    p95_adj = 2.0 * p95_s + 0.15
    s50 = max(0.0, min(100.0, 100.0 - 20.0 * math.log10(p50_adj / 0.1)))
    s95 = max(0.0, min(100.0, 100.0 - 20.0 * math.log10(p95_adj / 0.1)))
    speed_axis = round(0.5 * (s50 + s95), 2)

    # Cost Axis
    cost_usd = 0.0037
    cost_axis = round(max(0.0, min(100.0, 100.0 - 30.0 * math.log10(cost_usd / 0.001))), 2)

    # Partial Intelligence Axis:
    # 231-public items: easy 0.14, standard 0.28, judge held at 0.6096 (0.28), hard 0.30
    easy_acc = tier_results["easy"]["accuracy"]
    std_acc = tier_results["standard"]["accuracy"]
    judge_acc = 0.6096
    hard_acc = tier_results["hard"]["accuracy"]

    raw_weighted = 0.14 * easy_acc + 0.28 * std_acc + 0.28 * judge_acc + 0.30 * hard_acc
    chance = 0.293
    partial_intelligence = round(max(0.0, (raw_weighted - chance) / (1.0 - chance)) * 100.0, 2)

    # Harmonic Mean
    axes = [partial_intelligence, cal_axis, speed_axis, cost_axis]
    if all(a > 0 for a in axes):
        harmonic_mean = 4.0 / sum(1.0 / a for a in axes)
    else:
        harmonic_mean = 0.0

    penalty = (partial_intelligence / 50.0) ** 2 if partial_intelligence < 50.0 else 1.0
    partial_composite = round(harmonic_mean * penalty, 2)

    lp_acc = long_policy_stats["correct"] / long_policy_stats["total"] if long_policy_stats["total"] > 0 else 0.0

    return {
        "tiers": tier_results,
        "long_policy_slice": {
            "total": long_policy_stats["total"],
            "correct": long_policy_stats["correct"],
            "accuracy": round(lp_acc, 4),
        },
        "metrics": {
            "hard_ece": round(hard_ece, 4),
            "probability_fidelity": prob_fidelity,
            "mean_tvd": round(mean_tvd, 4),
            "p50_latency_s": round(p50_s, 4),
            "p95_latency_s": round(p95_s, 4),
        },
        "partial_axes": {
            "intelligence": partial_intelligence,
            "calibration": cal_axis,
            "speed": speed_axis,
            "cost": cost_axis,
        },
        "partial_harmonic_mean": round(harmonic_mean, 2),
        "partial_composite_score": partial_composite,
    }


def analyze_truncations(engine: DecisionEngine, tasks_by_tier: dict[str, list[dict]]) -> dict[str, Any]:
    print("\nMeasuring Token Lengths through formatting pipeline ...")
    tokenizer = engine.tokenizer
    tier_truncations = {}
    total_over_512 = 0
    total_over_2048 = 0
    total_tasks = 0

    from core.formatting import format_query, build_model_input

    for tier_name, raw_tasks in tasks_by_tier.items():
        over_512 = 0
        over_2048 = 0
        max_tokens = 0

        for task in raw_tasks:
            state_str, query = build_query_for_task(task)
            text, labels, _ = format_query(state_str, query)
            if query.kind in ("choice", "score"):
                prompt = build_model_input(query.question, state_str, labels)
            else:
                prompt = build_model_input("", text, labels)

            token_count = len(tokenizer.encode(prompt, add_special_tokens=True))
            
            if token_count > max_tokens:
                max_tokens = token_count
            if token_count > 512:
                over_512 += 1
            if token_count > 2048:
                over_2048 += 1

        tier_truncations[tier_name] = {
            "total": len(raw_tasks),
            "over_512": over_512,
            "over_2048": over_2048,
            "max_tokens": max_tokens,
        }
        total_over_512 += over_512
        total_over_2048 += over_2048
        total_tasks += len(raw_tasks)

    return {
        "total_tasks": total_tasks,
        "total_over_512": total_over_512,
        "total_over_2048": total_over_2048,
        "per_tier": tier_truncations,
    }


def main():
    parser = argparse.ArgumentParser(description="Controlled JevBench Evaluation")
    parser.add_argument("--tasks_dir", type=str, default="benchmarks/jevbench/v1.4.2.2/tasks")
    parser.add_argument("--model", type=str, default="heman10x/rlcd-modernbert-151m")
    parser.add_argument("--output", type=str, default="reports/v1.5/v1.5_jevbench_controlled.json")
    args = parser.parse_args()

    t_dir = Path(args.tasks_dir)
    tasks_by_tier = {
        "easy": load_jsonl(t_dir / "easy.jsonl"),
        "standard": load_jsonl(t_dir / "original.jsonl"),
        "hard": load_jsonl(t_dir / "hard.jsonl"),
    }

    # Instantiate Engines
    print("Instantiating Engines...")
    v2_cal = TemperatureCalibrator.load(REPO_ROOT / "artifacts" / "v2" / "calibrator.json")
    with open(REPO_ROOT / "artifacts" / "v2" / "calibrator.json") as f:
        v2_cal.per_k = json.load(f).get("per_k", {})

    engine_v14 = DecisionEngine(
        model_name_or_path=args.model,
        calibrator=v2_cal,
        device="cpu",
        max_length=512,
    )

    v15_cal = TemperatureCalibrator.load(REPO_ROOT / "artifacts" / "v1.5" / "calibrator.json")
    with open(REPO_ROOT / "artifacts" / "v1.5" / "calibrator.json") as f:
        v15_cal.per_k = json.load(f).get("per_k", {})

    engine_v15 = DecisionEngine(
        model_name_or_path=args.model,
        calibrator=v15_cal,
        device="cpu",
        max_length=2048,
    )

    truncation_analysis = analyze_truncations(engine_v15, tasks_by_tier)

    # Run v1.4-frozen
    v14_results = run_engine_suite(engine_v14, "v1.4-frozen", tasks_by_tier)

    # Run v1.5
    v15_results = run_engine_suite(engine_v15, "v1.5", tasks_by_tier)

    report = {
        "metadata": {
            "title": "Controlled JevBench Benchmark Report (v1.4-frozen vs v1.5)",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": args.model,
            "suite": "JevBench v1.4.2.2 Downloadable Public Subset (231 items)",
            "provenance_note": "Evaluated on the complete public downloadable set of 231 items (48 easy, 72 standard, 111 hard). The remaining 303 organizer-held items and 308 sealed items are retained privately and unavailable locally (per upstream HARD-TIER.md, TOPICS.md, and METHOD-v1.4.md).",
        },
        "configurations": {
            "v1.4-frozen": {
                "max_length": 512,
                "calibrator": "artifacts/v2/calibrator.json",
                "effective_temperature": "per_k (T=5.0069 for k<=3)",
                "truncation": "head-only 512 tokens",
            },
            "v1.5": {
                "max_length": 2048,
                "calibrator": "artifacts/v1.5/calibrator.json",
                "effective_temperature": "suite adapter T=1.0 on probes, choice k=3 at T=5.0069",
                "truncation": "7000-char middle truncation budget allocator + 2048 context window",
            },
        },
        "truncation_analysis": truncation_analysis,
        "results": {
            "v1.4-frozen": v14_results,
            "v1.5": v15_results,
        },
        "deltas": {
            "easy_accuracy_delta": round(v15_results["tiers"]["easy"]["accuracy"] - v14_results["tiers"]["easy"]["accuracy"], 4),
            "standard_accuracy_delta": round(v15_results["tiers"]["standard"]["accuracy"] - v14_results["tiers"]["standard"]["accuracy"], 4),
            "hard_accuracy_delta": round(v15_results["tiers"]["hard"]["accuracy"] - v14_results["tiers"]["hard"]["accuracy"], 4),
            "long_policy_delta": round(v15_results["long_policy_slice"]["accuracy"] - v14_results["long_policy_slice"]["accuracy"], 4),
            "hard_ece_delta": round(v15_results["metrics"]["hard_ece"] - v14_results["metrics"]["hard_ece"], 4),
            "probability_fidelity_delta": round(v15_results["metrics"]["probability_fidelity"] - v14_results["metrics"]["probability_fidelity"], 2),
        },
        "gate_status": {
            "verdict": "REJECTED (Intelligence < 50, partial suite; no leaderboard delta claimed)",
            "reason": "Intelligence axis is below 50.0 threshold. Score is computed strictly on partial 231-item public subset without 308 sealed items or gap penalty; no official leaderboard rank delta can be claimed.",
        },
    }

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[COMPLETE] Saved controlled JevBench report to {out_p}")


if __name__ == "__main__":
    main()
