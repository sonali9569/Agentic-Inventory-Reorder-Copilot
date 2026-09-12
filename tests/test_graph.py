import re
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph import (
    BatchCritique,
    Critique,
    ExtractedContextSignal,
    RankedItem,
    RankedRecommendations,
    _format_item_facts,
    _is_rationale_grounded,
    _trend_note,
    build_graph,
    make_context_extraction_node,
    make_gather_signals_node,
)


# ---------------------------------------------------------------- fixtures

@pytest.fixture
def small_dataset():
    dates = pd.date_range("2026-06-01", periods=70)
    items = pd.DataFrame([
        dict(item_id="LOW", item_name="Low Stock Item", category="Staples", supplier_id="S1",
             unit_cost_inr=10, unit_price_inr=15),
        dict(item_id="OK", item_name="Healthy Item", category="Staples", supplier_id="S1",
             unit_cost_inr=10, unit_price_inr=15),
    ])
    suppliers = pd.DataFrame([dict(supplier_id="S1", supplier_name="Test Supplier",
                                    mean_lead_time_days=4, lead_time_std_days=1)])
    # LOW: steady demand, almost nothing on hand -> stockout_risk
    # OK: steady demand, plenty on hand -> healthy, should never reach the LLM
    rows = []
    for d in dates:
        rows.append(dict(date=d, item_id="LOW", units_sold=8, stockout_flag=0, on_hand_end=3))
        rows.append(dict(date=d, item_id="OK", units_sold=8, stockout_flag=0, on_hand_end=200))
    sales = pd.DataFrame(rows)
    festival_calendar = pd.DataFrame(columns=["date", "festival_name", "affected_categories", "uplift_multiplier", "ramp_days_before"])
    festival_overrides = pd.DataFrame(columns=["festival_name", "item_id", "uplift_multiplier"])
    promotions = pd.DataFrame(columns=["date", "item_id", "promo_uplift"])
    return dict(sales=sales, items=items, suppliers=suppliers, festival_calendar=festival_calendar,
                festival_overrides=festival_overrides, promotions=promotions, as_of=dates[-1])


class _FakeStructuredLLM:
    def __init__(self, response=None, raise_n_times=0):
        self.response = response
        self.raise_n_times = raise_n_times
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        if self.calls <= self.raise_n_times:
            raise ValueError("simulated structured-output parse failure")
        return self.response


class _FakeLLM:
    def __init__(self, response=None, raise_n_times=0):
        self._structured = _FakeStructuredLLM(response, raise_n_times)

    def with_structured_output(self, schema):
        return self._structured


# ---------------------------------------------------------------- trend note

def test_trend_note_fires_for_a_genuine_ramp():
    assert "3.2x" in _trend_note(dict(trend_ratio=3.2))


def test_trend_note_fires_for_a_genuine_drop():
    note = _trend_note(dict(trend_ratio=0.2))
    assert note is not None and "0.2x" in note


def test_trend_note_silent_for_ordinary_day_to_day_variation():
    assert _trend_note(dict(trend_ratio=1.1)) is None
    assert _trend_note(dict(trend_ratio=0.9)) is None


def test_trend_note_silent_when_no_signal_at_all():
    assert _trend_note(dict(trend_ratio=None)) is None


def test_format_item_facts_surfaces_a_notable_trend_with_no_festival_involved():
    # this is the Raincoat/Umbrella case: a real seasonal ramp with no festival
    # behind it at all -- the trend note is the ONLY way this reaches the model
    consumption = {"X": dict(item_id="X", item_name="Raincoat", category="Durables",
                              risk="stockout_risk", on_hand=0, days_of_cover=0.0,
                              suggested_order_qty=39, trend_ratio=3.6)}
    context = {"X": dict(is_festival_window=False, days_to_next_festival=None)}
    facts = _format_item_facts("X", consumption, context)
    assert "3.6x" in facts
    assert "no festival involved" in facts


def test_format_item_facts_does_not_claim_no_festival_when_one_is_active():
    # an item can be BOTH inside a festival window AND showing a real
    # trend signal at once (e.g. Rakhi Set, actively selling during its own
    # festival) -- the "no festival involved" caveat must not fire for it
    consumption = {"X": dict(item_id="X", item_name="Rakhi Set", category="Festive",
                              risk="stockout_risk", on_hand=0, days_of_cover=0.0,
                              suggested_order_qty=2, trend_ratio=15.6)}
    context = {"X": dict(is_festival_window=True, active_festival="Raksha Bandhan", uplift_multiplier=14.0)}
    facts = _format_item_facts("X", consumption, context)
    assert "15.6x" in facts
    assert "no festival involved" not in facts
    assert "Raksha Bandhan" in facts


def test_format_item_facts_stays_quiet_when_trend_is_ordinary(small_dataset):
    node = make_gather_signals_node(
        small_dataset["sales"], small_dataset["items"], small_dataset["suppliers"],
        small_dataset["festival_calendar"], small_dataset["festival_overrides"], small_dataset["promotions"],
    )
    out = node({"as_of_date": str(small_dataset["as_of"].date())})
    facts = _format_item_facts("LOW", out["consumption_signals"], out["context_signals"])
    assert "usual rate" not in facts  # LOW's demand is flat in the fixture -- nothing notable to say


# ---------------------------------------------------------------- gather_signals

def test_gather_signals_excludes_healthy_items(small_dataset):
    node = make_gather_signals_node(
        small_dataset["sales"], small_dataset["items"], small_dataset["suppliers"],
        small_dataset["festival_calendar"], small_dataset["festival_overrides"], small_dataset["promotions"],
    )
    out = node({"as_of_date": str(small_dataset["as_of"].date())})
    assert "LOW" in out["consumption_signals"]
    assert "OK" not in out["consumption_signals"]  # healthy items never reach the LLM


def test_gather_signals_applies_per_item_parameter_overrides(small_dataset):
    # a seller's feedback adjusts z/lead_time_buffer_days,
    # but that's worthless if a future gather_signals call ignores it -- the whole
    # point of the Feedback Agent is that it reaches the NEXT recommendation.
    # LOW's demand needs real variance for z to matter at all: reorder_point's
    # z*sigma*sqrt(L) term is 0 regardless of z when sigma is 0, which is exactly
    # LOW's default fixture shape (constant 8/day) -- so this uses its own
    # alternating-demand sales series instead of the shared fixture's.
    dates = small_dataset["sales"]["date"].unique()
    variable_low_sales = pd.DataFrame({
        "date": dates, "item_id": "LOW",
        "units_sold": [4, 12] * (len(dates) // 2), "stockout_flag": 0, "on_hand_end": 3,
    })
    other_sales = small_dataset["sales"][small_dataset["sales"].item_id != "LOW"]
    dataset = {**small_dataset, "sales": pd.concat([other_sales, variable_low_sales], ignore_index=True)}

    default_node = make_gather_signals_node(
        dataset["sales"], dataset["items"], dataset["suppliers"],
        dataset["festival_calendar"], dataset["festival_overrides"], dataset["promotions"],
    )
    adjusted_node = make_gather_signals_node(
        dataset["sales"], dataset["items"], dataset["suppliers"],
        dataset["festival_calendar"], dataset["festival_overrides"], dataset["promotions"],
        params_by_item={"LOW": {"z": 1.0, "lead_time_buffer_days": 0.0}},  # z well below the 1.65 default
    )
    default_out = default_node({"as_of_date": str(dataset["as_of"].date())})
    adjusted_out = adjusted_node({"as_of_date": str(dataset["as_of"].date())})

    assert adjusted_out["consumption_signals"]["LOW"]["reorder_point"] < default_out["consumption_signals"]["LOW"]["reorder_point"]


def test_gather_signals_caps_to_the_most_severe_items(small_dataset):
    # LOW already has on_hand_end=3 (days_of_cover 3/8=0.375). Add four more at-risk
    # items at distinct, non-tying severities, then cap to 3 and confirm only the
    # three lowest days_of_cover survive -- while total_flagged_count still reports
    # the true count of all five.
    on_hand_by_item = {"RISK1": 1, "RISK2": 2, "RISK3": 5, "RISK4": 6}  # -> cover 0.125, 0.25, 0.625, 0.75
    extra_items = pd.DataFrame([
        dict(item_id=iid, item_name=iid, category="Staples", supplier_id="S1",
             unit_cost_inr=10, unit_price_inr=15)
        for iid in on_hand_by_item
    ])
    items = pd.concat([small_dataset["items"], extra_items], ignore_index=True)
    extra_rows = [
        dict(date=d, item_id=iid, units_sold=8, stockout_flag=0, on_hand_end=on_hand)
        for iid, on_hand in on_hand_by_item.items()
        for d in small_dataset["sales"].date.unique()
    ]
    sales = pd.concat([small_dataset["sales"], pd.DataFrame(extra_rows)], ignore_index=True)

    node = make_gather_signals_node(
        sales, items, small_dataset["suppliers"], small_dataset["festival_calendar"],
        small_dataset["festival_overrides"], small_dataset["promotions"], max_items=3,
    )
    out = node({"as_of_date": str(small_dataset["as_of"].date())})

    assert out["total_flagged_count"] == 5  # LOW + all 4 RISK items, uncapped count
    assert len(out["consumption_signals"]) == 3
    assert set(out["consumption_signals"]) == {"RISK1", "RISK2", "LOW"}  # the 3 most severe


def test_severity_key_sorts_infinite_cover_last_not_first():
    from graph import _severity_key
    no_demand_but_stocked = dict(risk="overstock", days_of_cover=None)
    genuinely_urgent = dict(risk="stockout_risk", days_of_cover=0.5)
    ranked = sorted([no_demand_but_stocked, genuinely_urgent], key=_severity_key)
    assert ranked[0] is genuinely_urgent  # None (infinite cover) must not look more urgent than 0.5 days


# ---------------------------------------------------------------- reasoning + merge safety

def test_reasoning_merge_pulls_quantity_from_engine_not_the_llm(small_dataset):
    # even if the LLM's structured output had a quantity field, the merge only
    # ever takes item_id/urgency/rationale from it -- qty always comes from consumption_signals
    fake_response = RankedRecommendations(recommendations=[
        RankedItem(item_id="LOW", urgency="high", rationale="Nearly out of stock."),
    ])
    llm = _FakeLLM(response=fake_response)
    graph = build_graph(**{k: v for k, v in small_dataset.items() if k != "as_of"}, llm=llm)

    config = {"configurable": {"thread_id": "test-merge"}}
    result = graph.invoke({"as_of_date": str(small_dataset["as_of"].date())}, config)

    rec = result["ranked_recommendations"][0]
    assert rec["item_id"] == "LOW"
    assert rec["urgency"] == "high"
    # the quantity must equal what assess_item computed, not something the LLM invented
    from engine import assess_item
    expected_qty = assess_item("LOW", small_dataset["as_of"], small_dataset["sales"],
                                small_dataset["items"], small_dataset["suppliers"])["suggested_order_qty"]
    assert rec["suggested_order_qty"] == expected_qty


def test_reasoning_replaces_ungrounded_rationale_with_safe_fallback(small_dataset):
    # a model batching many items can borrow a festival name from a DIFFERENT
    # item in the same prompt and paste it onto one that festival doesn't affect (LOW is Staples; this festival affects FMCG)
    festival_calendar = pd.DataFrame([dict(
        date=pd.Timestamp("2026-08-28"), festival_name="Raksha Bandhan",
        affected_categories="FMCG", uplift_multiplier=1.4, ramp_days_before=7,
    )])
    dataset = {**small_dataset, "festival_calendar": festival_calendar}
    fake_response = RankedRecommendations(recommendations=[
        RankedItem(item_id="LOW", urgency="high",
                   rationale="Low stock during the Raksha Bandhan window uplift period."),
    ])
    llm = _FakeLLM(response=fake_response)
    graph = build_graph(**{k: v for k, v in dataset.items() if k != "as_of"}, llm=llm)

    config = {"configurable": {"thread_id": "test-grounding-catch"}}
    result = graph.invoke({"as_of_date": str(dataset["as_of"].date())}, config)

    rationale = result["ranked_recommendations"][0]["rationale"]
    assert "Raksha Bandhan" not in rationale  # fabricated for this item -- must not survive
    assert "days of cover" in rationale       # replaced with the fact-only fallback


def test_reasoning_keeps_a_correctly_grounded_rationale(small_dataset):
    festival_calendar = pd.DataFrame([dict(
        date=pd.Timestamp("2026-08-28"), festival_name="Raksha Bandhan",
        affected_categories="Staples", uplift_multiplier=1.4, ramp_days_before=7,  # LOW *is* Staples here
    )])
    dataset = {**small_dataset, "festival_calendar": festival_calendar}
    original = "Low stock ahead of the Raksha Bandhan window uplift period."
    fake_response = RankedRecommendations(recommendations=[
        RankedItem(item_id="LOW", urgency="high", rationale=original),
    ])
    llm = _FakeLLM(response=fake_response)
    graph = build_graph(**{k: v for k, v in dataset.items() if k != "as_of"}, llm=llm)

    config = {"configurable": {"thread_id": "test-grounding-keep"}}
    result = graph.invoke({"as_of_date": str(dataset["as_of"].date())}, config)

    assert result["ranked_recommendations"][0]["rationale"] == original  # true claim, kept as-is


def test_reasoning_keeps_a_low_confidence_but_grounded_rationale_as_is(small_dataset):
    # low confidence is the model flagging its own uncertainty, not a fabricated
    # claim -- the text must survive unchanged, unlike an ungrounded rationale
    fake_response = RankedRecommendations(recommendations=[
        RankedItem(item_id="LOW", urgency="high", rationale="Nearly out of stock.", confidence=0.2),
    ])
    llm = _FakeLLM(response=fake_response)
    graph = build_graph(**{k: v for k, v in small_dataset.items() if k != "as_of"}, llm=llm)

    config = {"configurable": {"thread_id": "test-low-confidence-kept"}}
    result = graph.invoke({"as_of_date": str(small_dataset["as_of"].date())}, config)

    rec = result["ranked_recommendations"][0]
    assert rec["rationale"] == "Nearly out of stock."  # unchanged, not replaced
    assert rec["confidence"] == 0.2


def test_reasoning_defaults_to_full_confidence_when_the_model_omits_it(small_dataset):
    fake_response = RankedRecommendations(recommendations=[
        RankedItem(item_id="LOW", urgency="high", rationale="Nearly out of stock."),
    ])
    llm = _FakeLLM(response=fake_response)
    graph = build_graph(**{k: v for k, v in small_dataset.items() if k != "as_of"}, llm=llm)

    config = {"configurable": {"thread_id": "test-default-confidence"}}
    result = graph.invoke({"as_of_date": str(small_dataset["as_of"].date())}, config)

    assert result["ranked_recommendations"][0]["confidence"] == 1.0


def test_low_confidence_items_count_toward_the_retry_threshold(four_flagged):
    # every rationale is grounded (no fabricated claim) but reported at low
    # confidence -- the router must still escalate, same as a grounding failure
    class _UnconfidentLLM:
        def __init__(self):
            self.batch_sizes = []

        def with_structured_output(self, schema):
            outer = self

            class _Bound:
                def invoke(self, prompt):
                    ids = re.findall(r"^- (\S+) ", prompt, re.MULTILINE)
                    outer.batch_sizes.append(len(ids))
                    return RankedRecommendations(recommendations=[
                        RankedItem(item_id=i, urgency="high",
                                   rationale="Stock is low and cover is short.", confidence=0.1)
                        for i in ids
                    ])
            return _Bound()

    llm = _UnconfidentLLM()
    graph = build_graph(**{k: v for k, v in four_flagged.items() if k != "as_of"}, llm=llm)

    config = {"configurable": {"thread_id": "test-low-confidence-retry"}}
    paused = graph.invoke({"as_of_date": str(four_flagged["as_of"].date())}, config)
    recs = paused["__interrupt__"][0].value["recommendations"]

    assert len(llm.batch_sizes) > 1  # the router sent it back at least once
    assert min(llm.batch_sizes) < llm.batch_sizes[0]  # on a smaller batch
    # nothing was fabricated, so the original (not fact-only) rationale survives
    assert all(r["rationale"] == "Stock is low and cover is short." for r in recs)


# ---------------------------------------------------------------- writer/critic

class _WriterAndCriticLLM:
    """Schema-aware, unlike the single-response _FakeLLM above: returns the
    writer's ranking for RankedRecommendations and a scripted critique for
    BatchCritique, so the two independent calls can be tested as genuinely
    independent -- the writer never sees the critic's verdict or vice versa."""
    def __init__(self, writer_response, critiques: list[Critique]):
        self.writer_response = writer_response
        self.critiques = critiques

    def with_structured_output(self, schema):
        outer = self

        class _Bound:
            def invoke(self, prompt):
                if schema is RankedRecommendations:
                    return outer.writer_response
                if schema is BatchCritique:
                    return BatchCritique(critiques=outer.critiques)
                raise AssertionError(f"unexpected schema: {schema}")
        return _Bound()


def test_critic_replaces_a_rationale_the_regex_check_cannot_catch(small_dataset):
    # nothing here trips _is_rationale_grounded (no borrowed festival, invented
    # trend, or supplier word) -- only an independent critic call catches this
    writer_response = RankedRecommendations(recommendations=[
        RankedItem(item_id="LOW", urgency="high", rationale="This item will sell out within the hour."),
    ])
    critiques = [Critique(item_id="LOW", is_supported=False, confidence=0.1,
                           issue="facts say nothing about an hourly timeframe")]
    llm = _WriterAndCriticLLM(writer_response, critiques)
    graph = build_graph(**{k: v for k, v in small_dataset.items() if k != "as_of"}, llm=llm)

    config = {"configurable": {"thread_id": "test-critic-catch"}}
    result = graph.invoke({"as_of_date": str(small_dataset["as_of"].date())}, config)

    rationale = result["ranked_recommendations"][0]["rationale"]
    assert "within the hour" not in rationale
    assert "days of cover" in rationale  # replaced with the fact-only fallback


def test_critic_confidence_overrides_the_writers_self_report(small_dataset):
    # the writer claims full confidence; the critic disagrees -- the critic wins
    writer_response = RankedRecommendations(recommendations=[
        RankedItem(item_id="LOW", urgency="high", rationale="Nearly out of stock.", confidence=1.0),
    ])
    critiques = [Critique(item_id="LOW", is_supported=True, confidence=0.3)]
    llm = _WriterAndCriticLLM(writer_response, critiques)
    graph = build_graph(**{k: v for k, v in small_dataset.items() if k != "as_of"}, llm=llm)

    config = {"configurable": {"thread_id": "test-critic-confidence"}}
    result = graph.invoke({"as_of_date": str(small_dataset["as_of"].date())}, config)

    rec = result["ranked_recommendations"][0]
    assert rec["confidence"] == 0.3           # critic's number, not the writer's 1.0
    assert rec["rationale"] == "Nearly out of stock."  # supported, so kept as-is


def test_critic_approval_keeps_the_rationale_and_raises_confidence(small_dataset):
    writer_response = RankedRecommendations(recommendations=[
        RankedItem(item_id="LOW", urgency="high", rationale="Nearly out of stock.", confidence=0.4),
    ])
    critiques = [Critique(item_id="LOW", is_supported=True, confidence=0.95)]
    llm = _WriterAndCriticLLM(writer_response, critiques)
    graph = build_graph(**{k: v for k, v in small_dataset.items() if k != "as_of"}, llm=llm)

    config = {"configurable": {"thread_id": "test-critic-approve"}}
    result = graph.invoke({"as_of_date": str(small_dataset["as_of"].date())}, config)

    rec = result["ranked_recommendations"][0]
    assert rec["confidence"] == 0.95
    assert rec["rationale"] == "Nearly out of stock."


def test_a_regex_failure_is_never_sent_to_the_critic(small_dataset):
    # LOW is Staples; this festival only affects FMCG -- a borrowed claim the
    # regex check catches on its own. The critic must never see it: sending a
    # critique payload with no critiques for it (and asserting via the fake's
    # scripted response) proves it was never included in the review batch.
    festival_calendar = pd.DataFrame([dict(
        date=pd.Timestamp("2026-08-28"), festival_name="Raksha Bandhan",
        affected_categories="FMCG", uplift_multiplier=1.4, ramp_days_before=7,
    )])
    dataset = {**small_dataset, "festival_calendar": festival_calendar}
    writer_response = RankedRecommendations(recommendations=[
        RankedItem(item_id="LOW", urgency="high",
                   rationale="Low stock during the Raksha Bandhan window uplift period."),
    ])
    # no critique for LOW at all -- if the item were (wrongly) sent to the
    # critic, .get("LOW") would just return None and behave the same either
    # way, so this alone wouldn't prove much; the real assertion is the
    # rationale below, which must already be fact-only BEFORE any critique step
    llm = _WriterAndCriticLLM(writer_response, critiques=[])
    graph = build_graph(**{k: v for k, v in dataset.items() if k != "as_of"}, llm=llm)

    config = {"configurable": {"thread_id": "test-regex-first"}}
    result = graph.invoke({"as_of_date": str(dataset["as_of"].date())}, config)

    rationale = result["ranked_recommendations"][0]["rationale"]
    assert "Raksha Bandhan" not in rationale
    assert "days of cover" in rationale


def test_critic_failure_falls_back_to_the_writers_own_confidence(small_dataset):
    # the critic call raises on every attempt -- must degrade to pre-critic
    # behaviour (the writer's self-reported confidence), never crash the graph
    class _CriticAlwaysFails(_WriterAndCriticLLM):
        def with_structured_output(self, schema):
            if schema is BatchCritique:
                class _Raising:
                    def invoke(self, prompt):
                        raise ValueError("critic unavailable")
                return _Raising()
            return super().with_structured_output(schema)

    writer_response = RankedRecommendations(recommendations=[
        RankedItem(item_id="LOW", urgency="high", rationale="Nearly out of stock.", confidence=0.77),
    ])
    llm = _CriticAlwaysFails(writer_response, critiques=[])
    graph = build_graph(**{k: v for k, v in small_dataset.items() if k != "as_of"}, llm=llm)

    config = {"configurable": {"thread_id": "test-critic-outage"}}
    result = graph.invoke({"as_of_date": str(small_dataset["as_of"].date())}, config)

    rec = result["ranked_recommendations"][0]
    assert rec["confidence"] == 0.77  # the writer's own number, untouched
    assert rec["rationale"] == "Nearly out of stock."


def test_reasoning_never_drops_an_item_the_model_omitted(small_dataset):
    # a model handling a long list can silently omit items rather than erroring.
    # Add a third flagged item here and have the fake response cover only one of
    # the two flagged items -- the omitted one must still appear, not vanish.
    items = pd.concat([small_dataset["items"], pd.DataFrame([
        dict(item_id="LOW2", item_name="Also Low Stock", category="Staples", supplier_id="S1",
             unit_cost_inr=10, unit_price_inr=15),
    ])], ignore_index=True)
    sales = pd.concat([small_dataset["sales"], pd.DataFrame([
        dict(date=d, item_id="LOW2", units_sold=8, stockout_flag=0, on_hand_end=3)
        for d in small_dataset["sales"].date.unique()
    ])], ignore_index=True)
    dataset = {**small_dataset, "items": items, "sales": sales}

    fake_response = RankedRecommendations(recommendations=[
        RankedItem(item_id="LOW", urgency="high", rationale="Nearly out of stock."),
        # LOW2 is flagged too but the model's response never mentions it
    ])
    llm = _FakeLLM(response=fake_response)
    graph = build_graph(**{k: v for k, v in dataset.items() if k != "as_of"}, llm=llm)

    config = {"configurable": {"thread_id": "test-omission"}}
    result = graph.invoke({"as_of_date": str(dataset["as_of"].date())}, config)

    ids = {r["item_id"] for r in result["ranked_recommendations"]}
    assert ids == {"LOW", "LOW2"}  # LOW2 must still show up despite the model omitting it
    low2 = next(r for r in result["ranked_recommendations"] if r["item_id"] == "LOW2")
    assert low2["source"] == "backfill"


def test_reasoning_drops_a_hallucinated_item_id(small_dataset):
    fake_response = RankedRecommendations(recommendations=[
        RankedItem(item_id="LOW", urgency="high", rationale="Real item."),
        RankedItem(item_id="DOES_NOT_EXIST", urgency="high", rationale="Invented by the model."),
    ])
    llm = _FakeLLM(response=fake_response)
    graph = build_graph(**{k: v for k, v in small_dataset.items() if k != "as_of"}, llm=llm)

    config = {"configurable": {"thread_id": "test-hallucination"}}
    result = graph.invoke({"as_of_date": str(small_dataset["as_of"].date())}, config)

    ids = [r["item_id"] for r in result["ranked_recommendations"]]
    assert ids == ["LOW"]  # the invented id never made it through


def test_reasoning_falls_back_when_llm_fails_every_attempt(small_dataset):
    llm = _FakeLLM(response=None, raise_n_times=99)  # always raises
    graph = build_graph(**{k: v for k, v in small_dataset.items() if k != "as_of"}, llm=llm)

    config = {"configurable": {"thread_id": "test-fallback"}}
    result = graph.invoke({"as_of_date": str(small_dataset["as_of"].date())}, config)

    assert len(result["ranked_recommendations"]) == 1
    assert result["ranked_recommendations"][0]["source"] == "fallback"


def test_reasoning_retries_before_succeeding(small_dataset):
    fake_response = RankedRecommendations(recommendations=[
        RankedItem(item_id="LOW", urgency="medium", rationale="Recovered after retry."),
    ])
    llm = _FakeLLM(response=fake_response, raise_n_times=2)  # fails twice, succeeds on the 3rd
    graph = build_graph(**{k: v for k, v in small_dataset.items() if k != "as_of"}, llm=llm)

    config = {"configurable": {"thread_id": "test-retry"}}
    result = graph.invoke({"as_of_date": str(small_dataset["as_of"].date())}, config)

    assert result["ranked_recommendations"][0]["rationale"] == "Recovered after retry."


# ---------------------------------------------------------------- interrupt / resume

def test_graph_pauses_for_approval_then_resumes_with_the_decision(small_dataset):
    from langgraph.types import Command

    fake_response = RankedRecommendations(recommendations=[
        RankedItem(item_id="LOW", urgency="high", rationale="Nearly out of stock."),
    ])
    llm = _FakeLLM(response=fake_response)
    graph = build_graph(**{k: v for k, v in small_dataset.items() if k != "as_of"}, llm=llm)

    config = {"configurable": {"thread_id": "test-interrupt"}}
    paused = graph.invoke({"as_of_date": str(small_dataset["as_of"].date())}, config)
    assert "__interrupt__" in paused
    assert paused["__interrupt__"][0].value["recommendations"][0]["item_id"] == "LOW"

    resumed = graph.invoke(Command(resume={"decision": "approve"}), config)
    assert resumed["seller_decision"] == {"decision": "approve"}
    assert "__interrupt__" not in resumed


# ---------------------------------------------------------------- grounding checks

def _ctx(active=None, next_name=None):
    return {"I1": {"active_festival": active, "next_festival_name": next_name}}


def _cons(trend_ratio=None):
    return {"I1": {"item_id": "I1", "trend_ratio": trend_ratio, "on_hand": 3, "days_of_cover": 1.0}}


ALL_FESTIVALS = {"Diwali", "Holi", "Raksha Bandhan"}


def test_grounding_allows_the_items_own_festival():
    from graph import _is_rationale_grounded
    assert _is_rationale_grounded(
        "I1", "Diwali is 6 days away and stock is low.",
        _cons(), _ctx(next_name="Diwali"), ALL_FESTIVALS)


def test_grounding_rejects_a_festival_borrowed_from_another_item():
    from graph import _is_rationale_grounded
    assert not _is_rationale_grounded(
        "I1", "Stock up ahead of Raksha Bandhan.",
        _cons(), _ctx(next_name="Diwali"), ALL_FESTIVALS)


def test_grounding_rejects_a_trend_claim_when_the_item_has_no_trend_signal():
    from graph import _is_rationale_grounded
    # trend_ratio None -> _trend_note() emits nothing -> the facts block never
    # mentioned a trend, so the model invented it
    assert not _is_rationale_grounded(
        "I1", "Demand is surging for this item.", _cons(None), _ctx(), ALL_FESTIVALS)


def test_grounding_allows_a_trend_claim_when_the_signal_is_really_there():
    from graph import _is_rationale_grounded
    assert _is_rationale_grounded(
        "I1", "Demand is surging well above its usual rate.",
        _cons(trend_ratio=2.4), _ctx(), ALL_FESTIVALS)


def test_grounding_rejects_any_supplier_claim():
    from graph import _is_rationale_grounded
    # _format_item_facts never emits supplier information at all, so the whole
    # class is out of bounds -- no allow-list needed
    for text in ["The supplier is unreliable.",
                 "Long lead time on this one.",
                 "Expect a delivery delay."]:
        assert not _is_rationale_grounded("I1", text, _cons(2.4), _ctx(), ALL_FESTIVALS)


# ---------------------------------------------------------------- escalation routing

class _ContaminatingLLM:
    """Reproduces the observed failure: given a large batch it pastes a festival
    name from one item's facts onto every item. Given a small batch it behaves.
    Records the batch size of every call so a test can prove the graph shrank it."""

    def __init__(self, contaminate_above: int = 2):
        self.contaminate_above = contaminate_above
        self.batch_sizes = []

    def with_structured_output(self, schema):
        outer = self

        class _Bound:
            def invoke(self, prompt):
                ids = re.findall(r"^- (\S+) ", prompt, re.MULTILINE)
                outer.batch_sizes.append(len(ids))
                dirty = len(ids) > outer.contaminate_above
                return RankedRecommendations(recommendations=[
                    RankedItem(
                        item_id=i, urgency="high",
                        rationale=("Stock up for Raksha Bandhan." if dirty
                                   else "Stock is low and cover is short."),
                    ) for i in ids
                ])
        return _Bound()


def test_router_returns_to_reasoning_when_too_many_rationales_fail_grounding():
    from graph import make_route_after_reasoning
    route = make_route_after_reasoning(failure_limit=2, max_attempts=3)
    assert route({"grounding_failures": 0, "reasoning_attempt": 1}) == "approve"
    assert route({"grounding_failures": 1, "reasoning_attempt": 1}) == "approve"
    assert route({"grounding_failures": 5, "reasoning_attempt": 1}) == "retry_smaller"


def test_router_stops_retrying_at_the_attempt_ceiling():
    from graph import make_route_after_reasoning
    route = make_route_after_reasoning(failure_limit=2, max_attempts=3)
    # still failing, but out of budget -- fact-only rationales are the floor
    assert route({"grounding_failures": 9, "reasoning_attempt": 3}) == "approve"


def test_shrink_batch_halves_the_prompt_size():
    from graph import shrink_batch
    assert shrink_batch({"chunk_size": 8})["chunk_size"] == 4
    assert shrink_batch({"chunk_size": 1})["chunk_size"] == 1  # never reaches zero
    # first pass has no chunk_size set: derive it from the number of flagged items
    assert shrink_batch({"consumption_signals": {"A": {}, "B": {}, "C": {}, "D": {}}})["chunk_size"] == 2


@pytest.fixture
def four_flagged():
    """Four at-risk SKUs and a festival calendar that lists Raksha Bandhan without
    it applying to any of them -- so a rationale citing it is provably borrowed."""
    dates = pd.date_range("2026-06-01", periods=70)
    ids = ["A1", "A2", "A3", "A4"]
    items = pd.DataFrame([
        dict(item_id=i, item_name=f"Item {i}", category="Staples", supplier_id="S1",
             unit_cost_inr=10, unit_price_inr=15) for i in ids
    ])
    suppliers = pd.DataFrame([dict(supplier_id="S1", supplier_name="T",
                                    mean_lead_time_days=4, lead_time_std_days=1)])
    rows = [dict(date=d, item_id=i, units_sold=8, stockout_flag=0, on_hand_end=3)
            for d in dates for i in ids]
    festival_calendar = pd.DataFrame([dict(
        date=pd.Timestamp("2026-12-01"), festival_name="Raksha Bandhan",
        affected_categories="Festive", uplift_multiplier=1.4, ramp_days_before=7)])
    return dict(sales=pd.DataFrame(rows), items=items, suppliers=suppliers,
                festival_calendar=festival_calendar,
                festival_overrides=pd.DataFrame(columns=["festival_name", "item_id", "uplift_multiplier"]),
                promotions=pd.DataFrame(columns=["date", "item_id", "promo_uplift"]),
                as_of=dates[-1])


def test_graph_escalates_to_smaller_batches_and_recovers_the_rationales(four_flagged):
    """End to end: one contaminated pass, the router sends it back, the batch is
    halved until rationales survive grounding -- and the seller never sees the
    borrowed festival, on either path."""
    llm = _ContaminatingLLM(contaminate_above=2)
    graph = build_graph(**{k: v for k, v in four_flagged.items() if k != "as_of"}, llm=llm)

    config = {"configurable": {"thread_id": "test-escalation"}}
    paused = graph.invoke({"as_of_date": str(four_flagged["as_of"].date())}, config)
    recs = paused["__interrupt__"][0].value["recommendations"]

    # the loop ran: more than one call, and later calls were smaller than the first
    assert len(llm.batch_sizes) > 1
    assert min(llm.batch_sizes) < llm.batch_sizes[0]
    # and nothing borrowed ever reached the output
    assert not any("Raksha Bandhan" in r["rationale"] for r in recs)


# ---------------------------------------------------------------- context_extraction

class _FakeExtractionLLM:
    """Only ever asked for ExtractedContextSignal -- unlike the reasoning fakes
    above, so a wrong-schema call here is a real bug, not something to shrug off."""
    def __init__(self, response=None):
        self.response = response
        self.calls = 0

    def with_structured_output(self, schema):
        assert schema is ExtractedContextSignal
        return self

    def invoke(self, prompt):
        self.calls += 1
        return self.response


def test_context_extraction_is_a_noop_without_seller_notes():
    llm = _FakeExtractionLLM(response=ExtractedContextSignal(
        item_id="X", signal_type="festival", note="local mela this week", confidence=0.9))
    node = make_context_extraction_node(llm)
    out = node({"consumption_signals": {"X": {}}, "as_of_date": "2026-01-01"})
    assert out == {}
    assert llm.calls == 0  # no LLM call made at all -- the whole point of the no-op


def test_context_extraction_is_a_noop_with_no_flagged_items():
    llm = _FakeExtractionLLM(response=ExtractedContextSignal(
        item_id="X", signal_type="festival", note="local mela", confidence=0.9))
    node = make_context_extraction_node(llm)
    out = node({"seller_notes": "there's a local mela this week", "consumption_signals": {}})
    assert out == {}
    assert llm.calls == 0


def test_context_extraction_merges_a_valid_signal_into_context_signals():
    llm = _FakeExtractionLLM(response=ExtractedContextSignal(
        item_id="X", signal_type="promotion", note="running a 20% off sale this weekend",
        confidence=0.85))
    node = make_context_extraction_node(llm)
    out = node({
        "seller_notes": "I'm running a sale this weekend on X",
        "consumption_signals": {"X": {}, "Y": {}},
        "context_signals": {"X": {"is_festival_window": False}, "Y": {}},
    })
    assert out["context_signals"]["X"]["seller_reported_note"] == "running a 20% off sale this weekend"
    assert out["context_signals"]["X"]["is_festival_window"] is False  # existing keys untouched
    assert "seller_reported_note" not in out["context_signals"]["Y"]  # only the named item changes


def test_context_extraction_drops_a_hallucinated_item_id():
    # the model named an item that isn't actually flagged today -- never trusted
    llm = _FakeExtractionLLM(response=ExtractedContextSignal(
        item_id="NOT_FLAGGED", signal_type="festival", note="...", confidence=0.9))
    node = make_context_extraction_node(llm)
    out = node({"seller_notes": "something", "consumption_signals": {"X": {}}})
    assert out == {}


def test_context_extraction_drops_low_confidence_signals():
    llm = _FakeExtractionLLM(response=ExtractedContextSignal(
        item_id="X", signal_type="festival", note="maybe something", confidence=0.3))
    node = make_context_extraction_node(llm)
    out = node({"seller_notes": "not sure but maybe", "consumption_signals": {"X": {}}})
    assert out == {}


def test_context_extraction_drops_signal_type_none():
    llm = _FakeExtractionLLM(response=ExtractedContextSignal(
        item_id="X", signal_type="none", note="just chatting", confidence=0.95))
    node = make_context_extraction_node(llm)
    out = node({"seller_notes": "nothing relevant here", "consumption_signals": {"X": {}}})
    assert out == {}


def test_context_extraction_swallows_llm_failure():
    class _Raising(_FakeExtractionLLM):
        def invoke(self, prompt):
            raise ValueError("simulated failure")
    node = make_context_extraction_node(_Raising())
    out = node({"seller_notes": "something", "consumption_signals": {"X": {}}})
    assert out == {}


def test_grounding_check_allows_a_festival_named_in_a_seller_reported_note():
    # without the fix, a real festival name introduced via context_extraction
    # would be flagged as "borrowed from another item" and stripped
    consumption = {"X": dict(trend_ratio=None)}
    context = {"X": dict(seller_reported_note="Diwali crowd is already showing up early")}
    rationale = "Stock up for Diwali based on the seller's own note."
    assert _is_rationale_grounded("X", rationale, consumption, context, all_festival_names={"Diwali"})


def test_grounding_check_still_rejects_an_unrelated_borrowed_festival():
    consumption = {"X": dict(trend_ratio=None)}
    context = {"X": dict()}  # no seller note, no calendar signal for this item at all
    rationale = "Stock up for Diwali."
    assert not _is_rationale_grounded("X", rationale, consumption, context, all_festival_names={"Diwali"})
