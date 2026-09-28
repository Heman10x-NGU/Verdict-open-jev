"""Unit and integration tests for primary ModernBERT / GLiClass DecisionEngine."""

import pytest

from core.engine_encoder import DecisionEngine, _compute_concentration
from core.primitives import (
    Choice,
    INSUFFICIENT_EVIDENCE_ID,
    Level,
    Noul,
    Option,
    Score,
)


def test_compute_concentration() -> None:
    # Uniform distribution: concentration should be 0.0
    uniform_p = [0.25, 0.25, 0.25, 0.25]
    assert pytest.approx(_compute_concentration(uniform_p), abs=1e-5) == 0.0

    # Point mass distribution: concentration should be 1.0
    point_mass = [1.0, 0.0, 0.0, 0.0]
    assert pytest.approx(_compute_concentration(point_mass), abs=1e-5) == 1.0


def test_decision_engine_mock_batch_execution() -> None:
    engine = DecisionEngine(model="mock")

    q1 = Choice(
        id="q1",
        question="Select category",
        options=[
            Option(id="billing", description="Billing question"),
            Option(id="tech", description="Technical issue"),
        ],
    )
    q2 = Score(
        id="q2",
        question="Rate severity",
        levels=[
            Level(id="p1", description="Low", value=1.0),
            Level(id="p2", description="High", value=5.0),
        ],
    )
    q3 = Noul(
        id="q3",
        proposition="User is requesting a refund.",
        semantics="conditional_on_sufficient_evidence_v2",
    )

    batch_res = engine.evaluate("Customer wants a refund on order #1234", [q1, q2, q3])
    assert batch_res.forward_call_count == 1
    assert batch_res.execution_mode == "single_call"
    assert len(batch_res.results) == 3

    # Check Choice result
    c_res = batch_res.results[0]
    assert c_res.id == "q1"
    assert c_res.selected_id == "billing"
    assert not c_res.is_abstention
    assert c_res.concentration is not None

    # Check Score result
    s_res = batch_res.results[1]
    assert s_res.id == "q2"
    assert s_res.selected_level_id == "p1"
    assert s_res.selected_value == 1.0
    assert s_res.expected_score is not None

    # Check Noul result
    n_res = batch_res.results[2]
    assert n_res.id == "q3"
    assert n_res.selected_outcome in ("true", "false", INSUFFICIENT_EVIDENCE_ID)


def test_decision_engine_injected_model_and_tokenizer() -> None:
    from gliclass import GLiClassModel
    from transformers import AutoTokenizer

    model_name = "knowledgator/gliclass-modern-base-v2.0"
    model = GLiClassModel.from_pretrained(model_name).cpu().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    engine = DecisionEngine(model=model, tokenizer=tokenizer, device="cpu")
    q = Choice(
        id="c1",
        question="Select category",
        options=[
            Option(id="card", description="Card replacement"),
            Option(id="loan", description="Mortgage loan query"),
        ],
    )
    res = engine.evaluate("I need a new debit card sent to my home address", [q])
    assert len(res.results) == 1
    assert res.results[0].selected_id == "card"


def test_decision_engine_real_gliclass_inference() -> None:
    # Test real checkpoint inference on CPU
    engine = DecisionEngine(model_name_or_path="knowledgator/gliclass-modern-base-v2.0", device="cpu")

    context = (
        "Hello, I was double-charged on my credit card statement for invoice INV-9021. "
        "Please reverse the duplicate charge of $49.00."
    )

    q_choice = Choice(
        id="intent",
        question="What is the primary customer request?",
        options=[
            Option(id="billing_dispute", description="Billing dispute or duplicate charge refund"),
            Option(id="password_reset", description="Account login or password reset request"),
            Option(id="shipping_status", description="Package delivery or shipping tracking"),
        ],
    )

    q_noul = Noul(
        id="has_invoice_id",
        proposition="The customer explicitly provides an invoice identifier in the context.",
        semantics="conditional_on_sufficient_evidence_v2",
    )

    batch_res = engine.evaluate(context, [q_choice, q_noul])
    assert batch_res.forward_call_count == 1
    assert len(batch_res.results) == 2

    # Intent should clearly be billing_dispute
    intent_res = batch_res.results[0]
    assert intent_res.selected_id == "billing_dispute"
    assert intent_res.selected_probability > 0.65
    assert not intent_res.is_abstention

    # Proposition evaluation
    noul_res = batch_res.results[1]
    assert noul_res.selected_outcome == "true"
    assert not noul_res.is_abstention


def test_query_aware_context_packing() -> None:
    from core.engine_encoder import _query_aware_pack_context

    # Case 1: Short context (< 7000 chars) passes through completely unchanged
    short_ctx = "Short context that easily fits."
    q = Choice(
        id="c1",
        question="What is this?",
        options=[
            Option(id="short", description="Short text"),
            Option(id="long", description="Long text"),
        ],
    )
    assert _query_aware_pack_context(short_ctx, q, max_chars=7000) == short_ctx

    # Case 2: Multi-paragraph long context with header + body
    header_para = "Header declarations: Section 1.0 General Terms and Policy Definitions. " * 15  # ~1050 chars
    body_relevant = "Critical policy: The building water damage is strictly excluded under clause 4.2."
    body_irrelevant_1 = "Filler text about unrelated office equipment and furniture. " * 30
    body_irrelevant_2 = "More filler text about parking regulations and cafeteria hours. " * 30

    long_ctx = f"{header_para}\n\n{body_irrelevant_1}\n\n{body_relevant}\n\n{body_irrelevant_2}"
    q_water = Choice(
        id="c2",
        question="Is water damage covered?",
        options=[
            Option(id="excluded", description="The building water damage is excluded"),
            Option(id="covered", description="Water damage is covered"),
        ],
    )
    packed = _query_aware_pack_context(long_ctx, q_water, max_chars=2000)
    # Header must be preserved
    assert "Header declarations" in packed
    # High-IDF matching paragraph must be selected over irrelevant filler
    assert "building water damage is strictly excluded" in packed

    # Case 3: No-paragraph breaks (unindented string) falls back to middle truncation
    unindented_long = "Unbroken text stream without any double newlines. " * 100
    packed_fallback = _query_aware_pack_context(unindented_long, q_water, max_chars=1000)
    assert "...[middle content omitted for length]..." in packed_fallback


def test_2perm_tta_execution_modes() -> None:
    engine = DecisionEngine(model="mock")

    q_choice_multi = Choice(
        id="c_multi",
        question="Select option",
        options=[
            Option(id="o1", description="First option"),
            Option(id="o2", description="Second option"),
        ],
    )
    q_score = Score(
        id="s1",
        question="Rate severity",
        levels=[Level(id="l1", description="Low", value=1.0), Level(id="l2", description="High", value=5.0)],
    )

    # 1. Without TTA flag: execution_mode is single_call
    res_default = engine.evaluate("Some context", [q_choice_multi], enable_2perm_tta=False)
    assert res_default.execution_mode == "single_call"

    # 2. With TTA flag on Choice with 2 options: execution_mode is 2perm_tta
    res_tta = engine.evaluate("Some context", [q_choice_multi], enable_2perm_tta=True)
    assert res_tta.execution_mode == "2perm_tta"
    assert res_tta.forward_call_count == 1

    # 3. With TTA flag on Score query: TTA is skipped, execution_mode is single_call
    res_score = engine.evaluate("Some context", [q_score], enable_2perm_tta=True)
    assert res_score.execution_mode == "single_call"
