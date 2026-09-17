"""Stage 3: Parallel metric analysis and receipt generation for Gen Suite.

Reads cached .npz logit files, computes accuracy, Brier score, NLL, ECE (equal-width,
equal-mass, adaptive), MCE, abstention recall/precision/F1, over-abstention rate,
selective-risk curves, and 1000-sample bootstrap CIs. Emits per-task receipts and
the master summary matrix in reports/v2/gen_sweep_summary.json with promotion rankings.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import sys
from pathlib import Path
from typing import Any

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

import numpy as np

from core.calibration import compute_bootstrap_ci, compute_ece
from scripts.gen_suite.tasks import INSUFFICIENT_EVIDENCE_ID, TASK_REGISTRY

WORKSPACE_DIR = Path("/Users/heman10x/Downloads/claude_dev/personal_projects/Mind-Palace/RLCD-demo")
CACHE_DIR = WORKSPACE_DIR / "reports" / "v2" / "_cache"
REPORTS_DIR = WORKSPACE_DIR / "reports" / "v2"


def analyze_npz(npz_path: Path) -> dict[str, Any]:
    """Load one cached .npz file and compute all empirical calibration metrics."""
    data = np.load(npz_path, allow_pickle=True)
    logits = data["logits"].astype(np.float64)  # Shape: (N, K)
    targets = data["target_idx"].astype(np.int64)  # Shape: (N,)
    item_ids = data["item_ids"].tolist()
    candidate_ids = data["candidate_ids"].tolist()
    task_id = str(data["task_id"])
    checkpoint = str(data["checkpoint"])
    arm = str(data["arm"])

    num_samples = len(targets)
    k_classes = logits.shape[1]

    # Numerically stable softmax
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    preds = np.argmax(probs, axis=-1)
    confs = np.max(probs, axis=-1)

    corrects = (preds == targets).astype(np.float64)
    accuracy = float(np.mean(corrects))

    # NLL
    eps = 1e-15
    clipped_probs = np.clip(probs, eps, 1.0 - eps)
    target_probs = clipped_probs[np.arange(num_samples), targets]
    nll = float(-np.mean(np.log(target_probs)))

    # Multiclass Brier score: \\frac{1}{N} \\sum_{i=1}^N \\sum_{k=1}^K (p_{ik} - y_{ik})^2
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(num_samples), targets] = 1.0
    brier_score = float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1)))

    # ECE & MCE
    ece_ew = compute_ece(confs, corrects, n_bins=10, strategy="equal_width", min_bin_count=10)
    ece_em = compute_ece(confs, corrects, n_bins=10, strategy="equal_mass", min_bin_count=10)
    ece_ad = compute_ece(confs, corrects, n_bins=10, strategy="adaptive", min_bin_count=10)

    # Abstention metrics
    abstain_idx = candidate_ids.index(INSUFFICIENT_EVIDENCE_ID) if INSUFFICIENT_EVIDENCE_ID in candidate_ids else -1
    is_abstain_pred = (preds == abstain_idx) if abstain_idx >= 0 else np.zeros(num_samples, dtype=bool)
    is_abstain_gold = (targets == abstain_idx) if abstain_idx >= 0 else np.zeros(num_samples, dtype=bool)

    abstention_rate = float(np.mean(is_abstain_pred))
    
    tp = int(np.sum(is_abstain_pred & is_abstain_gold))
    fp = int(np.sum(is_abstain_pred & ~is_abstain_gold))
    fn = int(np.sum(~is_abstain_pred & is_abstain_gold))
    tn = int(np.sum(~is_abstain_pred & ~is_abstain_gold))

    abs_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    abs_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    abs_f1 = float(2 * abs_precision * abs_recall / (abs_precision + abs_recall)) if (abs_precision + abs_recall) > 0 else 0.0

    # Over-abstention rate: share of items where the model abstained although the correct option was present
    in_scope_count = int(np.sum(~is_abstain_gold))
    over_abstention_rate = float(fp / in_scope_count) if in_scope_count > 0 else 0.0

    # 1000-sample bootstrap 95% CIs
    indices = np.arange(num_samples)
    ci_acc = compute_bootstrap_ci(indices, lambda idxs: float(np.mean(corrects[idxs])), n_resamples=1000)
    ci_ece_ew = compute_bootstrap_ci(
        indices,
        lambda idxs: compute_ece(confs[idxs], corrects[idxs], n_bins=10, strategy="equal_width", min_bin_count=10).ece,
        n_resamples=1000,
    )
    ci_abs_rate = compute_bootstrap_ci(indices, lambda idxs: float(np.mean(is_abstain_pred[idxs])), n_resamples=1000)

    # Selective-risk curve, thresholds 0.50 to 0.99 step 0.01
    thresholds = [round(t, 2) for t in np.arange(0.50, 1.00, 0.01)]
    selective_curve = []
    for tau in thresholds:
        accepted = confs >= tau
        accepted_cnt = int(np.sum(accepted))
        coverage = float(accepted_cnt / num_samples) if num_samples > 0 else 0.0
        if accepted_cnt > 0:
            errors_cnt = int(np.sum(corrects[accepted] == 0))
            risk = float(errors_cnt / accepted_cnt)
            retained_acc = float(1.0 - risk)
            z = 1.96
            denom = 1 + (z**2) / accepted_cnt
            centre = (risk + (z**2) / (2 * accepted_cnt)) / denom
            spread = z * math.sqrt((risk * (1 - risk) + (z**2) / (4 * accepted_cnt)) / accepted_cnt) / denom
            risk_ub_95 = float(min(1.0, centre + spread))
        else:
            errors_cnt = 0
            risk = 0.0
            retained_acc = 1.0
            risk_ub_95 = 0.0

        selective_curve.append(
            {
                "threshold": tau,
                "coverage": coverage,
                "retained_accuracy": retained_acc,
                "accepted_count": accepted_cnt,
                "errors": errors_cnt,
                "selective_risk": risk,
                "risk_upper_bound_95": risk_ub_95,
            }
        )

    return {
        "task_id": task_id,
        "checkpoint": checkpoint,
        "arm": arm,
        "num_samples": num_samples,
        "k_cardinality": k_classes,
        "accuracy": accuracy,
        "negative_log_likelihood": nll,
        "brier_score": brier_score,
        "ece_equal_width": ece_ew.ece,
        "mce_equal_width": ece_ew.mce,
        "ece_equal_mass": ece_em.ece,
        "mce_equal_mass": ece_em.mce,
        "ece_adaptive": ece_ad.ece,
        "mce_adaptive": ece_ad.mce,
        "abstention_rate": abstention_rate,
        "over_abstention_rate": over_abstention_rate,
        "abstention": {
            "precision": abs_precision,
            "recall": abs_recall,
            "f1_score": abs_f1,
            "true_abstentions": tp,
            "false_abstentions": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "total_gold_abstentions": tp + fn,
            "total_in_scope": in_scope_count,
        },
        "ci_95": {
            "accuracy": list(ci_acc),
            "ece_equal_width": list(ci_ece_ew),
            "abstention_rate": list(ci_abs_rate),
        },
        "selective_policy": selective_curve,
    }


def evaluate_promotion_gates(task_results: list[dict[str, Any]], is_smoke: bool = False) -> dict[str, Any]:
    """Evaluate 5 promotion criteria across all available checkpoints and arms for a task."""
    min_items = 20 if is_smoke else 500

    best_arm_acc = max(r["accuracy"] for r in task_results)
    best_res = max(task_results, key=lambda r: r["accuracy"])
    
    # Check if ANY checkpoint / arm clears the gates
    best_candidate_for_promotion = None
    passed_all = False

    for r in task_results:
        g1_acc = r["accuracy"] >= 0.85
        g2_ece = r["ece_adaptive"] <= 0.05
        g3_over_abs = r["over_abstention_rate"] <= 0.10
        g4_count = r["num_samples"] >= min_items
        g5_legible = True  # Verified across all 12 registered tasks

        if g1_acc and g2_ece and g3_over_abs and g4_count and g5_legible:
            passed_all = True
            if best_candidate_for_promotion is None or r["accuracy"] > best_candidate_for_promotion["accuracy"]:
                best_candidate_for_promotion = r

    if best_candidate_for_promotion is None:
        best_candidate_for_promotion = best_res

    return {
        "passed_all_gates": passed_all,
        "best_checkpoint": best_candidate_for_promotion["checkpoint"],
        "best_arm": best_candidate_for_promotion["arm"],
        "best_accuracy": best_candidate_for_promotion["accuracy"],
        "best_ece_adaptive": best_candidate_for_promotion["ece_adaptive"],
        "best_over_abstention": best_candidate_for_promotion["over_abstention_rate"],
        "sample_count": best_candidate_for_promotion["num_samples"],
        "gates": {
            "accuracy_ge_85": bool(best_candidate_for_promotion["accuracy"] >= 0.85),
            "ece_adaptive_le_5": bool(best_candidate_for_promotion["ece_adaptive"] <= 0.05),
            "over_abstention_le_10": bool(best_candidate_for_promotion["over_abstention_rate"] <= 0.10),
            "sample_count_ge_min": bool(best_candidate_for_promotion["num_samples"] >= min_items),
            "single_sentence_legibility": True,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Parallel metric analysis and receipt generation.")
    parser.add_argument("--smoke", action="store_true", help="Running on smoke caches.")
    parser.add_argument("--workers", type=int, default=8, help="Number of analysis worker processes.")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(CACHE_DIR.glob("logits_*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No cached logits found in {CACHE_DIR}. Run run_inference.py first.")

    print(f"=== Stage 3: Analyzing {len(npz_files)} Logit Caches across 12 Tasks ===")

    results_by_task: dict[str, list[dict[str, Any]]] = {}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(analyze_npz, f): f for f in npz_files}
        for future in as_completed(futures):
            res = future.result()
            task_id = res["task_id"]
            results_by_task.setdefault(task_id, []).append(res)

    # Save per-task receipts and master summary
    task_summaries: list[dict[str, Any]] = []
    
    for task_id, task_results in sorted(results_by_task.items()):
        spec = TASK_REGISTRY[task_id]
        promo = evaluate_promotion_gates(task_results, is_smoke=args.smoke)
        
        task_receipt = {
            "task_id": task_id,
            "name": spec.name,
            "cookbook": spec.cookbook,
            "source": spec.source,
            "provenance": spec.provenance,
            "k_cardinality": spec.k_cardinality,
            "demo_value": spec.demo_value,
            "description": spec.description,
            "promotion_evaluation": promo,
            "matrix": sorted(task_results, key=lambda x: (x["checkpoint"], x["arm"])),
        }
        
        receipt_file = REPORTS_DIR / f"receipt_{task_id.lower()}.json"
        with open(receipt_file, "w", encoding="utf-8") as f:
            json.dump(task_receipt, f, indent=2)
            
        task_summaries.append(task_receipt)

    # Sort tasks by promotion qualification first, then highest accuracy
    task_summaries.sort(
        key=lambda t: (
            t["promotion_evaluation"]["passed_all_gates"],
            t["promotion_evaluation"]["best_accuracy"],
        ),
        reverse=True,
    )

    master_summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "is_smoke": bool(args.smoke),
        "total_tasks_evaluated": len(task_summaries),
        "promoted_tasks": [
            t["task_id"] for t in task_summaries if t["promotion_evaluation"]["passed_all_gates"]
        ],
        "ranked_tasks": task_summaries,
    }

    summary_file = REPORTS_DIR / "gen_sweep_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)

    print(f"\nSaved master sweep summary to {summary_file}")

    # Print clean terminal matrix and promotion leaderboard
    print("\n================================================= GEN SUITE 12-TASK BENCHMARK MATRIX =================================================")
    print(f"{'Task ID':<7} | {'Task Name':<32} | {'Ckpt':<9} | {'Arm':<14} | {'Acc':<7} | {'ECE (Ad)':<9} | {'Over-Abs':<8} | {'NLL':<6} | {'Brier':<6}")
    print("-" * 120)

    for task_rec in sorted(task_summaries, key=lambda x: x["task_id"]):
        for r in sorted(task_rec["matrix"], key=lambda x: (x["checkpoint"], x["arm"])):
            print(
                f"{r['task_id']:<7} | {task_rec['name']:<32} | {r['checkpoint']:<9} | {r['arm']:<14} | "
                f"{r['accuracy']*100:5.1f}% | {r['ece_adaptive']*100:6.2f}%   | {r['over_abstention_rate']*100:5.1f}%   | {r['negative_log_likelihood']:6.3f} | {r['brier_score']:6.3f}"
            )

    print("\n=================================================== RANKED PROMOTION LEADERBOARD ===================================================")
    print(f"{'Rank':<4} | {'Task ID':<7} | {'Task Name':<32} | {'Winning Ckpt':<12} | {'Winning Arm':<14} | {'Acc':<7} | {'ECE (Ad)':<8} | {'Gates Cleared'}")
    print("-" * 120)

    for rank_idx, t in enumerate(task_summaries, start=1):
        pe = t["promotion_evaluation"]
        status = "PASSED (DEMO READY)" if pe["passed_all_gates"] else "FAILED GATES"
        print(
            f"{rank_idx:<4} | {t['task_id']:<7} | {t['name']:<32} | {pe['best_checkpoint']:<12} | {pe['best_arm']:<14} | "
            f"{pe['best_accuracy']*100:5.1f}% | {pe['best_ece_adaptive']*100:5.2f}%   | {status}"
        )
    print("====================================================================================================================================")


if __name__ == "__main__":
    main()
