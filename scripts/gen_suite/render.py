"""Stage 4: Markdown receipt rendering and documentation integration for Gen Suite.

Renders empirical tables from reports/v2/gen_sweep_summary.json into README.md
between <!-- BEGIN GENERATED: cookbook_sweep_* --> markers without manual transcription.
Supports --check flag to verify documentation freshness in CI.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

REPORTS_DIR = WORKSPACE_DIR / "reports" / "v2"
README_PATH = WORKSPACE_DIR / "README.md"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required receipt {path} does not exist.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_pct(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


def format_num(val: float | None, decimals: int = 4) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def render_top3_table(summary: dict[str, Any]) -> str:
    """Render top 3 promoted cookbook use cases."""
    ranked = summary.get("ranked_tasks", [])
    top3 = ranked[:3]

    lines = [
        "| Rank | Task ID | Cookbook Recipe | Evaluated Use Case | Optimal Checkpoint | Framing Arm | Accuracy | Adaptive ECE | Over-Abstention |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for rank_idx, t in enumerate(top3, start=1):
        pe = t["promotion_evaluation"]
        lines.append(
            f"| {rank_idx} | `{t['task_id']}` | [{t['name']}](RLCD%20Cookbook/{t['cookbook']}.md) | {t['description']} | `{pe['best_checkpoint']}` | `{pe['best_arm']}` | {format_pct(pe['best_accuracy'])} | {format_pct(pe['best_ece_adaptive'])} | {format_pct(pe['best_over_abstention'])} |"
        )

    return "\n".join(lines)


def render_full_matrix_table(summary: dict[str, Any]) -> str:
    """Render the full 12-task cross-checkpoint matrix."""
    ranked = summary.get("ranked_tasks", [])
    
    lines = [
        "| Task ID | Task Name | Cardinality ($K$) | Winning Checkpoint | Winning Arm | Accuracy | Adaptive ECE | Over-Abstention | Proper Brier | Gate Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for t in sorted(ranked, key=lambda x: x["task_id"]):
        pe = t["promotion_evaluation"]
        status = "PASSED (Promoted)" if pe["passed_all_gates"] else "Failed Gate"
        # Find best arm result object
        best_obj = next((m for m in t["matrix"] if m["checkpoint"] == pe["best_checkpoint"] and m["arm"] == pe["best_arm"]), t["matrix"][0])
        lines.append(
            f"| `{t['task_id']}` | {t['name']} | $K={t['k_cardinality']}$ | `{pe['best_checkpoint']}` | `{pe['best_arm']}` | {format_pct(pe['best_accuracy'])} | {format_pct(pe['best_ece_adaptive'])} | {format_pct(pe['best_over_abstention'])} | {format_num(best_obj.get('brier_score', 0.0))} | {status} |"
        )

    return "\n".join(lines)


def replace_generated_section(content: str, section_name: str, new_content: str) -> str:
    start_tag = f"<!-- BEGIN GENERATED: {section_name} -->"
    end_tag = f"<!-- END GENERATED: {section_name} -->"
    pattern = re.compile(rf"{re.escape(start_tag)}.*?{re.escape(end_tag)}", re.DOTALL)
    replacement = f"{start_tag}\n{new_content}\n{end_tag}"

    if pattern.search(content):
        return pattern.sub(replacement, content)
    else:
        raise ValueError(f"Marker '{start_tag}' not found in document.")


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Render Gen Suite receipts into README.")
    parser.add_argument("--check", action="store_true", help="Check for documentation drift without writing.")
    args = parser.parse_args()

    summary_file = REPORTS_DIR / "gen_sweep_summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Master summary {summary_file} not found. Run analyze.py first.")

    summary = load_json(summary_file)
    top3_table = render_top3_table(summary)
    full_matrix_table = render_full_matrix_table(summary)

    if not README_PATH.exists():
        raise FileNotFoundError(f"{README_PATH} does not exist.")

    original_content = README_PATH.read_text(encoding="utf-8")
    updated_content = original_content

    # If markers don't exist yet, insert them cleanly into the cookbook / evaluation section
    if "<!-- BEGIN GENERATED: cookbook_sweep_top3 -->" not in updated_content:
        section_to_append = f"""
## Gen Suite Cookbook Evaluation Matrix

Empirical evaluation of OpenJev across 12 structured decision tasks, comparing the zero-shot base checkpoint (`knowledgator/gliclass-modern-base-v2.0`) against the Banking77 fine-tuned checkpoint (`artifacts/v2/model.safetensors`) across three label-framing arms (`neutral`, `verbose`, `banking_framed`).

### Promoted Top Cookbooks

<!-- BEGIN GENERATED: cookbook_sweep_top3 -->
{top3_table}
<!-- END GENERATED: cookbook_sweep_top3 -->

### Full 12-Task Cross-Domain Evaluation Matrix

<!-- BEGIN GENERATED: cookbook_sweep_matrix -->
{full_matrix_table}
<!-- END GENERATED: cookbook_sweep_matrix -->
"""
        # Append before the last horizontal rule or at the end
        updated_content = updated_content + section_to_append
    else:
        updated_content = replace_generated_section(updated_content, "cookbook_sweep_top3", top3_table)
        updated_content = replace_generated_section(updated_content, "cookbook_sweep_matrix", full_matrix_table)

    if original_content != updated_content:
        if args.check:
            print("Documentation drift detected in README.md:")
            diff = difflib.unified_diff(
                original_content.splitlines(),
                updated_content.splitlines(),
                fromfile="README.md",
                tofile="README.md (rendered)",
                lineterm="",
            )
            print("\n".join(diff))
            sys.exit(1)
        else:
            README_PATH.write_text(updated_content, encoding="utf-8")
            print("Rendered Gen Suite receipts into README.md")
    else:
        print("README.md is already up to date with reports/v2/gen_sweep_summary.json.")


if __name__ == "__main__":
    main()
