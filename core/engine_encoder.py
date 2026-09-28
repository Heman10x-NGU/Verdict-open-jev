"""Primary non-autoregressive decision engine using ModernBERT and GLiClass.

Evaluates typed decision queries (Choice, Score, Noul) in a single batched
forward pass over dynamic labels with explicit abstention candidates and
properly calibrated confidence scores.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from core.calibration import TemperatureCalibrator
from core.formatting import (
    CapacityError,
    MAX_SUBSTANTIVE_CANDIDATES,
    MAX_SUPPORTED_CANDIDATES,
    build_model_input,
    format_prompt,
    format_query,
    frame_hypothesis,
)
from core.primitives import (
    ChoiceResult,
    DecisionBatchResult,
    DecisionResult,
    INSUFFICIENT_EVIDENCE_DESC,
    INSUFFICIENT_EVIDENCE_ID,
    NoulResult,
    Query,
    ScoreResult,
)


def _compute_concentration(probs: Sequence[float]) -> float:
    """Compute normalized negative entropy: 1.0 - (H(p) / ln(K)).

    Returns a distribution shape statistic in [0.0, 1.0] where 1.0 is a point mass
    and 0.0 is a uniform distribution.
    """
    k = len(probs)
    if k <= 1:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probs if p > 1e-12)
    max_entropy = math.log(k)
    return max(0.0, min(1.0, 1.0 - (entropy / max_entropy)))


def _tokenize_text(text: str) -> list[str]:
    """Extract lowercase alphanumeric tokens."""
    return [m.lower() for m in re.findall(r"[a-zA-Z0-9]+", text)]


def _query_aware_pack_context(
    context: str,
    query: Query,
    max_chars: int = 7000,
    header_target_chars: int = 1200,
    header_max_chars: int = 1600,
    reserved_chars: int = 0,
) -> str:
    """Paragraph-relevance packing preserving header declarations and doc order.

    1. If context length <= max_chars, returns context as is.
    2. Identifies paragraph boundaries (\\n\\s*\\n). If no paragraph boundaries exist
       (e.g., unindented JSON dumps), falls back cleanly to middle truncation.
    3. Retains opening paragraphs up to header_target_chars (extending to next break,
       capped at header_max_chars).
    4. Computes within-document IDF weights for question + candidate description tokens.
    5. Scores remaining body paragraphs by IDF-weighted query term match.
    6. Selects highest-scoring paragraphs within budget (accounting for reserved
       label/question budget) and packs them in original document order.
    """
    effective_budget = max(1000, max_chars - reserved_chars)
    if len(context) <= effective_budget:
        return context

    raw_paras = [p.strip() for p in re.split(r"\n\s*\n+", context) if p.strip()]

    # Fallback to middle truncation if no paragraph breaks exist
    if len(raw_paras) <= 1:
        half = effective_budget // 2
        return (
            context[:half]
            + "\n...[middle content omitted for length]...\n"
            + context[-half:]
        )

    # Step 1: Accumulate opening header paragraphs
    header_paras: list[str] = []
    header_len = 0
    body_paras_with_idx: list[tuple[int, str]] = []

    for idx, p in enumerate(raw_paras):
        p_len = len(p)
        sep_len = 2 if header_paras else 0
        if header_len + sep_len + p_len <= header_target_chars:
            header_paras.append(p)
            header_len += sep_len + p_len
        elif header_len < header_target_chars and (
            header_len + sep_len + p_len <= header_max_chars
        ):
            header_paras.append(p)
            header_len += sep_len + p_len
        else:
            body_paras_with_idx.append((idx, p))

    if not body_paras_with_idx:
        return "\n\n".join(header_paras)

    body_budget = max(0, effective_budget - (header_len + 2))
    if body_budget <= 0:
        return "\n\n".join(header_paras)

    # Step 2: Query tokens (question + option descriptions / proposition)
    query_text = ""
    if query.kind == "choice":
        query_text = f"{query.question} " + " ".join(
            opt.description for opt in query.options
        )
    elif query.kind == "score":
        query_text = f"{query.question} " + " ".join(
            f"{lvl.description} {lvl.value}" for lvl in query.levels
        )
    elif query.kind == "noul":
        query_text = f"{query.proposition}"

    query_tokens = set(_tokenize_text(query_text))
    if not query_tokens:
        packed_body = []
        cur_len = 0
        for _, p in body_paras_with_idx:
            if cur_len + len(p) + 2 <= body_budget:
                packed_body.append(p)
                cur_len += len(p) + 2
            else:
                break
        return "\n\n".join(header_paras + packed_body)

    # Step 3: Within-document IDF across paragraphs
    N = len(raw_paras)
    para_token_sets = [_tokenize_text(p) for _, p in body_paras_with_idx]

    df: dict[str, int] = {}
    for t_list in para_token_sets:
        for t in set(t_list):
            if t in query_tokens:
                df[t] = df.get(t, 0) + 1

    idf = {t: math.log((N + 1.0) / (count + 1.0)) + 1.0 for t, count in df.items()}

    # Step 4: Score each body paragraph
    scored_paras: list[tuple[float, int, str]] = []
    for (orig_idx, p), tokens in zip(body_paras_with_idx, para_token_sets):
        score = 0.0
        tf: dict[str, int] = {}
        for t in tokens:
            if t in idf:
                tf[t] = tf.get(t, 0) + 1
        for t, freq in tf.items():
            score += idf[t] * (1.0 + math.log(freq))
        scored_paras.append((score, orig_idx, p))

    # Stable sort descending by score
    scored_paras.sort(key=lambda x: x[0], reverse=True)

    # Step 5: Greedily pack highest-scoring paragraphs
    selected: list[tuple[int, str]] = []
    cur_body_len = 0
    for _, orig_idx, p in scored_paras:
        sep_len = 2 if selected else 0
        if cur_body_len + sep_len + len(p) <= body_budget:
            selected.append((orig_idx, p))
            cur_body_len += sep_len + len(p)

    # Step 6: Restore original document order
    selected.sort(key=lambda x: x[0])
    selected_paras = [p for _, p in selected]

    return "\n\n".join(header_paras + selected_paras)


class DecisionEngine:
    """Non-autoregressive decision engine evaluating typed queries concurrently."""

    def __init__(
        self,
        model_name_or_path: str = "knowledgator/gliclass-modern-base-v2.0",
        model: Any = None,
        tokenizer: Any = None,
        calibrator: TemperatureCalibrator | None = None,
        device: str = "cpu",
        max_length: int = 2048,
        suite_temperature: float | None = None,
        enable_smart_framing: bool = True,
        enable_2perm_tta: bool = False,
        enable_query_aware_packing: bool = False,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.max_length = max_length
        self.calibrator = calibrator
        self.suite_temperature = suite_temperature
        self.enable_smart_framing = enable_smart_framing
        self.enable_2perm_tta = enable_2perm_tta
        self.enable_query_aware_packing = enable_query_aware_packing

        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        elif model == "mock":
            self.model = None
            self.tokenizer = None
        else:
            # Lazy or eager load from HuggingFace
            from gliclass import GLiClassModel
            from transformers import AutoTokenizer

            self.model = GLiClassModel.from_pretrained(model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model.to(self.device)
            self.model.eval()

        self.ort_session = None
        if model != "mock" and model is None and self.device == "cpu":
            try:
                import onnxruntime as ort
                onnx_file = None
                if isinstance(model_name_or_path, str) and os.path.isdir(model_name_or_path):
                    cand = os.path.join(model_name_or_path, "model.onnx")
                    if os.path.exists(cand):
                        onnx_file = cand
                if not onnx_file:
                    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                    for cand in [
                        os.path.join(repo_dir, "artifacts", "v1.5", "model.onnx"),
                        os.path.join(repo_dir, "artifacts", "v2", "model.onnx"),
                        os.path.join(repo_dir, "artifacts", "openjev_modernbert.onnx"),
                        os.path.join(repo_dir, "..", "artifacts", "v1.5", "model.onnx"),
                        os.path.join(repo_dir, "..", "artifacts", "v2", "model.onnx"),
                        os.path.join(repo_dir, "..", "artifacts", "openjev_modernbert.onnx"),
                        "artifacts/v1.5/model.onnx",
                        "artifacts/v2/model.onnx",
                        "artifacts/openjev_modernbert.onnx",
                    ]:
                        if os.path.exists(cand):
                            onnx_file = cand
                            break
                if onnx_file and os.path.exists(onnx_file):
                    sess_opts = ort.SessionOptions()
                    sess_opts.intra_op_num_threads = 4
                    sess_opts.inter_op_num_threads = 1
                    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    self.ort_session = ort.InferenceSession(
                        onnx_file,
                        sess_opts,
                        providers=["CPUExecutionProvider"],
                    )
                    logging.info("Preferred FP32 ONNX session loaded from %s", onnx_file)
            except Exception as exc:
                logging.warning("Failed to initialize ONNX session; falling back to PyTorch: %s", exc)
                self.ort_session = None

        if self.calibrator is None:
            cal_file = None
            if isinstance(model_name_or_path, str) and os.path.isdir(model_name_or_path):
                for fname in ["calibrator.json", "calibrator_modernbert.json"]:
                    cand = os.path.join(model_name_or_path, fname)
                    if os.path.exists(cand):
                        cal_file = cand
                        break
            if not cal_file:
                repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                for cand in [
                    os.path.join(repo_dir, "artifacts", "v1.5", "calibrator.json"),
                    os.path.join(repo_dir, "artifacts", "v2", "calibrator.json"),
                    os.path.join(repo_dir, "artifacts", "calibrator.json"),
                    os.path.join(repo_dir, "artifacts", "calibrator_modernbert.json"),
                    os.path.join(repo_dir, "..", "artifacts", "v1.5", "calibrator.json"),
                    os.path.join(repo_dir, "..", "artifacts", "v2", "calibrator.json"),
                    os.path.join(repo_dir, "..", "artifacts", "calibrator.json"),
                    os.path.join(repo_dir, "..", "artifacts", "calibrator_modernbert.json"),
                    "artifacts/v1.5/calibrator.json",
                    "artifacts/v2/calibrator.json",
                    "artifacts/calibrator.json",
                    "artifacts/calibrator_modernbert.json",
                ]:
                    if os.path.exists(cand):
                        cal_file = cand
                        break
            if not cal_file and isinstance(model_name_or_path, str) and not os.path.exists(model_name_or_path):
                try:
                    from huggingface_hub import hf_hub_download
                    cal_file = hf_hub_download(model_name_or_path, "calibrator.json")
                except Exception:
                    pass
            if cal_file and os.path.exists(cal_file):
                try:
                    self.calibrator = TemperatureCalibrator.load(cal_file)
                    try:
                        with open(cal_file, "r", encoding="utf-8") as f:
                            cal_data = json.load(f)
                        if "per_k" in cal_data:
                            self.calibrator.per_k = cal_data["per_k"]
                    except Exception:
                        pass
                    logging.info("Auto-loaded calibrator from %s (T=%s)", cal_file, self.calibrator.temperature)
                except Exception as exc:
                    logging.warning("Failed to auto-load calibrator from %s: %s", cal_file, exc)

    def evaluate(
        self,
        context: str,
        queries: Sequence[Query],
        return_raw_logits: bool = False,
        temperature: float | None = None,
        enable_smart_framing: bool | None = None,
        enable_2perm_tta: bool | None = None,
        enable_query_aware_packing: bool | None = None,
    ) -> DecisionBatchResult:
        """Evaluate multiple typed queries in a single batched forward pass."""
        if not queries:
            return DecisionBatchResult(
                results=(),
                forward_call_count=0,
                total_latency_ms=0.0,
                execution_mode="single_call",
            )

        start_time = time.perf_counter()

        use_smart_framing = (
            self.enable_smart_framing
            if enable_smart_framing is None
            else enable_smart_framing
        )
        use_2perm_tta = (
            self.enable_2perm_tta if enable_2perm_tta is None else enable_2perm_tta
        )
        use_query_aware_packing = (
            self.enable_query_aware_packing
            if enable_query_aware_packing is None
            else enable_query_aware_packing
        )

        def _allocate_middle_truncation(ctx: str, max_chars: int = 7000) -> str:
            if len(ctx) <= max_chars:
                return ctx
            half = max_chars // 2
            return ctx[:half] + "\n...[middle content omitted for length]...\n" + ctx[-half:]

        query_plans: list[dict[str, Any]] = []
        all_prompts: list[str] = []
        all_labels_for_eval: list[list[str]] = []

        for q in queries:
            # 1. Budget & pack context
            if use_query_aware_packing:
                reserved = len(getattr(q, "question", "")) + sum(
                    len(getattr(opt, "description", ""))
                    for opt in getattr(q, "options", [])
                ) + 150
                budgeted_context = _query_aware_pack_context(
                    context, q, max_chars=7000, reserved_chars=reserved
                )
            else:
                budgeted_context = _allocate_middle_truncation(context, max_chars=7000)

            # 2. Format Pass 1
            text_p1, labels_p1, ids_p1 = format_query(
                budgeted_context, q, smart_framing=use_smart_framing
            )
            if len(labels_p1) > MAX_SUPPORTED_CANDIDATES:
                logging.warning(
                    "Candidate count %d exceeds MAX_SUPPORTED_CANDIDATES %d; degrading by truncating.",
                    len(labels_p1),
                    MAX_SUPPORTED_CANDIDATES,
                )
                labels_p1 = labels_p1[:MAX_SUPPORTED_CANDIDATES]
                ids_p1 = ids_p1[:MAX_SUPPORTED_CANDIDATES]

            if q.kind in ("choice", "score"):
                prompt_p1 = build_model_input(q.question, budgeted_context, labels_p1)
            else:
                prompt_p1 = build_model_input("", text_p1, labels_p1)

            idx_p1 = len(all_prompts)
            all_prompts.append(prompt_p1)
            all_labels_for_eval.append(labels_p1)

            # 3. 2-Permutation Averaging (TTA):
            # Exclude score queries; skip when substantive count <= 1
            is_tta = False
            idx_p2 = None
            ids_p2: list[str] = []

            if use_2perm_tta and q.kind == "choice":
                opts = list(q.options)[:MAX_SUBSTANTIVE_CANDIDATES]
                if len(opts) >= 2:
                    is_tta = True
                    reversed_opts = list(reversed(opts))
                    if use_smart_framing:
                        labels_p2 = [
                            frame_hypothesis(opt.description) for opt in reversed_opts
                        ] + [INSUFFICIENT_EVIDENCE_DESC]
                    else:
                        labels_p2 = [
                            f"It is {opt.description}" for opt in reversed_opts
                        ] + [INSUFFICIENT_EVIDENCE_DESC]
                    ids_p2 = [opt.id for opt in reversed_opts] + [
                        INSUFFICIENT_EVIDENCE_ID
                    ]
                    prompt_p2 = build_model_input(
                        q.question, budgeted_context, labels_p2
                    )
                    idx_p2 = len(all_prompts)
                    all_prompts.append(prompt_p2)
                    all_labels_for_eval.append(labels_p2)

            elif use_2perm_tta and q.kind == "noul":
                is_tta = True
                labels_p2 = [
                    f"false: not {q.proposition}",
                    f"true: {q.proposition}",
                    INSUFFICIENT_EVIDENCE_DESC,
                ]
                ids_p2 = ["false", "true", INSUFFICIENT_EVIDENCE_ID]
                text_p2 = f"Context:\n{budgeted_context}\n\nEvaluate proposition: {q.proposition}"
                prompt_p2 = build_model_input("", text_p2, labels_p2)
                idx_p2 = len(all_prompts)
                all_prompts.append(prompt_p2)
                all_labels_for_eval.append(labels_p2)

            query_plans.append(
                {
                    "query": q,
                    "ids": ids_p1,
                    "is_tta": is_tta,
                    "idx_p1": idx_p1,
                    "ids_p1": ids_p1,
                    "idx_p2": idx_p2,
                    "ids_p2": ids_p2,
                }
            )

        any_tta_run = any(plan["is_tta"] for plan in query_plans)

        if self.model is None:
            # Mock mode for testing without GPU/checkpoint dependencies
            batch_logits = []
            for plan in query_plans:
                row_logits = [3.0] + [0.5] * (len(plan["ids"]) - 1)
                batch_logits.append(row_logits)
        else:
            tokenized_inputs = self.tokenizer(
                all_prompts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            raw_logits = None
            if self.ort_session is not None:
                try:
                    ort_inputs = {
                        "input_ids": tokenized_inputs["input_ids"].cpu().numpy(),
                        "attention_mask": tokenized_inputs["attention_mask"].cpu().numpy(),
                    }
                    ort_outs = self.ort_session.run(None, ort_inputs)
                    raw_logits = torch.from_numpy(ort_outs[0]).to(self.device)
                except Exception as exc:
                    logging.warning("ONNX execution failed (%s); falling back to PyTorch", exc)
                    raw_logits = None

            if raw_logits is None:
                with torch.inference_mode():
                    outputs = self.model(**tokenized_inputs)
                    raw_logits = outputs.logits  # shape: (batch_size, max_num_classes)

            if not torch.all(torch.isfinite(raw_logits)):
                raise ValueError("Model output contains non-finite logits (NaN or Inf).")

            batch_logits = []
            for plan in query_plans:
                q_ids = plan["ids"]
                idx_1 = plan["idx_p1"]
                ids_1 = plan["ids_p1"]
                logits_1 = raw_logits[idx_1, : len(ids_1)].float().cpu().tolist()

                if plan["is_tta"]:
                    idx_2 = plan["idx_p2"]
                    ids_2 = plan["ids_p2"]
                    logits_2 = raw_logits[idx_2, : len(ids_2)].float().cpu().tolist()

                    map_1 = {oid: val for oid, val in zip(ids_1, logits_1)}
                    map_2 = {oid: val for oid, val in zip(ids_2, logits_2)}

                    # 2-permutation raw logit averaging: average logit per option ID before temperature
                    averaged_logits = [
                        0.5 * (map_1[oid] + map_2[oid]) for oid in q_ids
                    ]
                    batch_logits.append(averaged_logits)
                else:
                    batch_logits.append(logits_1)

        # Process each field prediction
        results: list[DecisionResult] = []
        total_latency_ms = (time.perf_counter() - start_time) * 1000.0
        per_query_latency = total_latency_ms / len(queries)

        for idx, q in enumerate(queries):
            logits_tensor = torch.tensor(
                [batch_logits[idx]], dtype=torch.float32, device=self.device
            )
            ids = query_plans[idx]["ids"]

            # Temperature calibration and scope determination
            effective_t = temperature if temperature is not None else self.suite_temperature
            if effective_t is not None and not return_raw_logits:
                cal_logits = logits_tensor / float(effective_t)
                cal_status = "suite_adapter_calibrated"
            elif self.calibrator is not None and not return_raw_logits:
                per_k = getattr(self.calibrator, "per_k", None)
                k_val = str(len(ids))
                if per_k and k_val in per_k:
                    t_val = float(per_k[k_val])
                    cal_logits = logits_tensor / t_val
                elif per_k:
                    k_int = len(ids)
                    int_keys = sorted([int(k) for k in per_k.keys() if k.isdigit()])
                    if int_keys and k_int <= int_keys[0]:
                        t_val = float(per_k[str(int_keys[0])])
                    elif int_keys and k_int >= int_keys[-1]:
                        t_val = float(per_k[str(int_keys[-1])])
                    elif int_keys:
                        low_k = max(k for k in int_keys if k <= k_int)
                        high_k = min(k for k in int_keys if k >= k_int)
                        if low_k == high_k:
                            t_val = float(per_k[str(low_k)])
                        else:
                            t_low = float(per_k[str(low_k)])
                            t_high = float(per_k[str(high_k)])
                            t_val = t_low + (t_high - t_low) * (k_int - low_k) / (high_k - low_k)
                    else:
                        t_val = getattr(self.calibrator, "temperature", 2.0)
                    cal_logits = logits_tensor / t_val
                else:
                    try:
                        cal_logits = self.calibrator(logits_tensor, k=len(ids))
                    except TypeError:
                        cal_logits = self.calibrator(logits_tensor)
                cal_status = "calibrated_for_scope"
            else:
                cal_logits = logits_tensor
                cal_status = "raw_logits" if return_raw_logits else "uncalibrated"

            raw_probs = F.softmax(cal_logits, dim=-1)[0].cpu().tolist()
            # Normalize to sum to exactly 1.0 within floating point precision
            p_sum = sum(raw_probs)
            probs = [float(p / p_sum) for p in raw_probs]
            prob_dict = {opt_id: p for opt_id, p in zip(ids, probs)}

            max_idx = int(torch.argmax(cal_logits[0]).item())
            selected_id = ids[max_idx]
            selected_prob = prob_dict[selected_id]
            is_abstention = (selected_id == INSUFFICIENT_EVIDENCE_ID)
            concentration = _compute_concentration(probs)

            if q.kind == "choice":
                results.append(
                    ChoiceResult(
                        id=q.id,
                        selected_id=selected_id,
                        selected_probability=selected_prob,
                        probabilities=prob_dict,
                        is_abstention=is_abstention,
                        concentration=concentration,
                        model_id=self.model_name_or_path,
                        calibration_status=cal_status,
                        latency_ms=per_query_latency,
                    )
                )
            elif q.kind == "score":
                substantive_mass = sum(
                    prob_dict[lvl.id] for lvl in q.levels if lvl.id in prob_dict
                )
                if substantive_mass > 0.0 and not is_abstention:
                    expected_val = sum(
                        lvl.value * (prob_dict[lvl.id] / substantive_mass)
                        for lvl in q.levels
                    )
                    selected_val = next(
                        (lvl.value for lvl in q.levels if lvl.id == selected_id), None
                    )
                else:
                    expected_val = None
                    selected_val = None

                results.append(
                    ScoreResult(
                        id=q.id,
                        selected_level_id=selected_id,
                        selected_value=selected_val,
                        expected_score=expected_val,
                        probabilities=prob_dict,
                        is_abstention=is_abstention,
                        abstention_probability=prob_dict.get(INSUFFICIENT_EVIDENCE_ID, 0.0),
                        model_id=self.model_name_or_path,
                        calibration_status=cal_status,
                        latency_ms=per_query_latency,
                    )
                )
            elif q.kind == "noul":
                p_abstain = prob_dict.get(INSUFFICIENT_EVIDENCE_ID, 0.0)
                substantive_mass = prob_dict.get("true", 0.0) + prob_dict.get("false", 0.0)
                if substantive_mass > 0.0 and not is_abstention:
                    p_true_cond = prob_dict.get("true", 0.0) / substantive_mass
                else:
                    p_true_cond = None

                results.append(
                    NoulResult(
                        id=q.id,
                        selected_outcome=selected_id,  # type: ignore[arg-type]
                        p_true_given_sufficient_evidence=p_true_cond,
                        p_insufficient_evidence=p_abstain,
                        probabilities=prob_dict,
                        is_abstention=is_abstention,
                        model_id=self.model_name_or_path,
                        calibration_status=cal_status,
                        latency_ms=per_query_latency,
                    )
                )

        return DecisionBatchResult(
            results=tuple(results),
            forward_call_count=1,
            total_latency_ms=total_latency_ms,
            execution_mode="2perm_tta" if any_tta_run else "single_call",
        )

    def predict_logits(
        self,
        context: str,
        queries: Sequence[Query],
    ) -> list[dict[str, float]]:
        """Return raw uncalibrated logits for each candidate option per query."""
        res = self.evaluate(context, queries, return_raw_logits=True)
        return [dict(r.probabilities) for r in res.results]

