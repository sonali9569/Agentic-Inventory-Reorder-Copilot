"""
LangGraph orchestrator that ranks flagged inventory items and generates a
rationale for each. engine.py, context_agent.py and forecast.py supply data;
this is the only module that calls an LLM.

Graph flow:
  START -> gather_signals -> context_extraction -> reasoning -> [route] -> human_approval -> END
                                                        ^                |
                                                        +---- shrink_batch ----+

gather_signals filters and ranks flagged items (no LLM). context_extraction
parses an optional free-text note from the seller into a structured signal;
it is a no-op if no note is supplied. reasoning generates a rationale per
item, checks each one against the supplied facts with a regex-based grounding
check, and sends surviving rationales to a second LLM call that critiques
them independently. A rationale that fails either check is replaced with a
fact-only sentence built from the underlying data.

If more than a set number of rationales fail grounding in one pass, the graph
routes back to reasoning with a smaller batch size, up to three attempts,
before falling back to fact-only text.

human_approval pauses the graph with interrupt() and returns the seller's
decision into state. The dashboard reads this payload directly and applies
each decision through feedback_agent.submit_feedback(), since approval is
per item while the interrupt covers the whole batch.

Two invariants hold throughout: order quantities always come from
assess_item()'s deterministic output, never from the model, and any item_id
not present in the input batch is dropped rather than trusted.
"""

import re
import sys
from pathlib import Path
from typing import Literal, Optional, TypedDict

import pandas as pd
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_agent import get_context
from engine import assess_item, days_to_next_arrival, on_order_qty


class SellerSenseState(TypedDict):
    as_of_date: str
    consumption_signals: dict
    context_signals: dict
    total_flagged_count: int
    ranked_recommendations: list[dict]
    seller_decision: Optional[dict]
    # routing state: how many rationales failed the grounding check on the last
    # pass, how many passes have run, and how many items go into one prompt
    grounding_failures: int
    reasoning_attempt: int
    chunk_size: Optional[int]
    # optional free text from the seller, consumed only by context_extraction;
    # absent or empty means that node is a no-op (see its docstring)
    seller_notes: Optional[str]


LOW_CONFIDENCE_THRESHOLD = 0.5


class RankedItem(BaseModel):
    item_id: str = Field(description="the item_id exactly as given -- never invent a new one")
    urgency: Literal["high", "medium", "low"]
    rationale: str = Field(
        description="one or two plain-language sentences a busy shop owner can read in five "
        "seconds, citing the SPECIFIC reason from the facts given (a festival, a growth trend, "
        "low stock, a slow-moving supplier) -- not a generic restatement of the numbers"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="how confident you are in THIS item's rationale specifically, not the "
        "ranking overall -- report low confidence when the facts are thin or ambiguous rather "
        "than writing a confident-sounding guess"
    )


class RankedRecommendations(BaseModel):
    recommendations: list[RankedItem]


class Critique(BaseModel):
    item_id: str = Field(description="exactly one of the item_ids under review")
    is_supported: bool = Field(
        description="false if the rationale claims anything -- a festival, a trend, a "
        "supplier issue, a conclusion -- that these facts don't actually support"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="your confidence that this rationale is fully accurate, independent of "
        "how confident the original writer sounded"
    )
    issue: Optional[str] = Field(default=None, description="what's unsupported, if is_supported is false")


class BatchCritique(BaseModel):
    critiques: list[Critique]


# ---------------------------------------------------------------- gather_signals

def _severity_key(c: dict) -> tuple:
    """Sorts stockout risk ahead of overstock, then by days of cover ascending --
    None means infinite cover (nothing on hand to run out of, or no demand at all),
    which must sort LAST, not first, so it doesn't look falsely urgent."""
    cover = c["days_of_cover"]
    return (c["risk"] != "stockout_risk", cover if cover is not None else float("inf"))


def make_gather_signals_node(sales, items, suppliers, festival_calendar, festival_overrides,
                              promotions, max_items: int = 8, params_by_item: dict | None = None,
                              purchase_orders=None, forecast_tables: dict | None = None):
    """Deterministic, no LLM. Filters to items actually worth the Reasoning node's
    attention -- healthy items never reach the model at all -- then caps to the
    max_items most severe. Two independent reasons for the cap: a
    seller reviewing 18 alerts at once defeats the "top few, not a flood" premise
    of the product, and a live model takes noticeably longer (and has more room to
    silently omit something) the more items it has to handle in one batch. The
    full flagged count still travels in state so nothing pretends fewer items
    needed attention than actually did.

    params_by_item threads the Feedback Agent's per-item adjustments (z, lead_time_buffer_days)
    back into the assessment -- without this, a seller's feedback would change
    what the Inventory tab shows but never actually reach a future recommendation,
    which defeats the entire point of the Feedback Agent existing.

    purchase_orders nets off stock already in transit. Optional so existing callers
    and tests keep working, but the dashboard passes it: without it the same order
    is recommended every morning until the goods physically arrive.

    forecast_tables is {item_id: pd.Series} of pre-computed forecasts -- see
    forecast_cache.py, which is the only place that fits or loads a Prophet
    model. Optional and per-item: an item missing from the dict (no cached
    model yet, or the toolchain to fit one unavailable) just falls back to
    assess_item's own trailing-rate estimate for that one item, same as before
    this parameter existed."""
    params_by_item = params_by_item or {}
    forecast_tables = forecast_tables or {}

    def gather_signals(state: SellerSenseState) -> dict:
        as_of = pd.Timestamp(state["as_of_date"])
        flagged = {}
        for item_id in items["item_id"]:
            params = params_by_item.get(item_id, {})
            assessment = assess_item(item_id, as_of, sales, items, suppliers,
                                      z=params.get("z", 1.65),
                                      lead_time_buffer_days=params.get("lead_time_buffer_days", 0.0),
                                      on_order=on_order_qty(item_id, as_of, purchase_orders),
                                      arriving_in_days=days_to_next_arrival(item_id, as_of, purchase_orders),
                                      forecast_table=forecast_tables.get(item_id))
            if assessment["risk"] == "healthy":
                continue
            flagged[item_id] = assessment

        top_ids = sorted(flagged, key=lambda iid: _severity_key(flagged[iid]))[:max_items]
        consumption = {iid: flagged[iid] for iid in top_ids}
        context = {
            iid: get_context(iid, as_of, items, festival_calendar, festival_overrides, promotions)
            for iid in top_ids
        }
        return {
            "consumption_signals": consumption,
            "context_signals": context,
            "total_flagged_count": len(flagged),
        }
    return gather_signals


# ---------------------------------------------------------------- context extraction

class ExtractedContextSignal(BaseModel):
    item_id: Optional[str] = Field(
        description="must be exactly one of today's flagged item_ids, or null if the "
        "note isn't clearly about one of them -- never invent an item_id"
    )
    signal_type: Literal["festival", "promotion", "none"] = Field(
        description="'none' if the note doesn't actually describe a demand-relevant event"
    )
    note: str = Field(description="a short, plain restatement of what the seller said")
    confidence: float = Field(ge=0.0, le=1.0)


CONTEXT_EXTRACTION_CONFIDENCE_THRESHOLD = 0.6


def make_context_extraction_node(llm):
    """
    Turns a seller's free-text note into a structured signal, validated by
    Pydantic, and merges it into context_signals for one item -- never into
    consumption_signals, so it can enrich a rationale but can never touch a
    quantity. Scoped to festival/promotion signals only, not supplier claims,
    since _is_rationale_grounded() bans any supplier language regardless of
    source.

    Same failure mode as everywhere an LLM output meets state here: a
    hallucinated item_id, a signal_type of "none", or confidence below threshold
    all mean the note is dropped, not guessed into the facts block. Costs
    nothing when the state has no seller_notes -- no LLM call is made.
    """
    def context_extraction(state: SellerSenseState) -> dict:
        notes = state.get("seller_notes")
        if not notes:
            return {}

        flagged_ids = sorted(state.get("consumption_signals", {}))
        if not flagged_ids:
            return {}

        prompt = (
            "A shop owner wrote this free-text note about their shop. Extract a "
            "festival or promotion signal ONLY if it clearly concerns one of "
            f"today's flagged items: {flagged_ids}. item_id must be one of those "
            "exact ids, or null if none apply. signal_type is 'none' if the note "
            "doesn't describe an actual demand-relevant event.\n\n"
            f"Note: \"{notes}\""
        )
        try:
            extracted = llm.with_structured_output(ExtractedContextSignal).invoke(prompt)
        except Exception:
            return {}

        if (extracted.signal_type == "none"
                or extracted.item_id not in flagged_ids
                or extracted.confidence < CONTEXT_EXTRACTION_CONFIDENCE_THRESHOLD):
            return {}

        context = dict(state.get("context_signals", {}))
        item_ctx = dict(context.get(extracted.item_id, {}))
        item_ctx["seller_reported_note"] = extracted.note
        context[extracted.item_id] = item_ctx
        return {"context_signals": context}
    return context_extraction


# ---------------------------------------------------------------- reasoning

def _trend_note(c: dict) -> str | None:
    """Surfaces engine.py's recent_trend_ratio() as a plain-language note -- this
    is the signal that catches a seasonal ramp like the Raincoat/Umbrella monsoon
    case, which the festival calendar alone has no way to see (no festival is
    involved, it's pure seasonality). Only worth mentioning when it's a genuine
    departure from the item's own usual rate, not routine day-to-day noise."""
    ratio = c.get("trend_ratio")
    if ratio is None:
        return None
    if ratio >= 1.5:
        return f"recent demand is running {ratio}x its usual rate"
    if ratio <= 0.5:
        return f"recent demand is running at only {ratio}x its usual rate"
    return None


def _format_item_facts(item_id: str, consumption: dict, context: dict) -> str:
    c = consumption[item_id]
    ctx = context.get(item_id, {})
    line = (
        f"- {item_id} {c['item_name']} ({c['category']}): risk={c['risk']}, on_hand={c['on_hand']}, "
        f"days_of_cover={c['days_of_cover']}, suggested_order_qty={c['suggested_order_qty']}"
        f"{' (sized off a real forecast)' if c.get('used_forecast') else ''}"
    )
    # stock already in transit is the difference between "order this now" and
    # "it is handled, watch the date" -- the model has no way to tell those apart
    # from a quantity of 0 alone
    if c.get("on_order"):
        arriving = c.get("arriving_in_days")
        when = f", arriving in {arriving} day(s)" if arriving is not None else ""
        line += f"\n  {c['on_order']} unit(s) already on order{when}"
    has_festival_signal = False
    if ctx.get("is_festival_window"):
        line += f"\n  currently inside the {ctx['active_festival']} window (uplift x{ctx['uplift_multiplier']})"
        has_festival_signal = True
    elif ctx.get("days_to_next_festival") is not None:
        line += f"\n  {ctx['next_festival_name']} is {ctx['days_to_next_festival']} days away"
        has_festival_signal = True
    trend = _trend_note(c)
    if trend:
        # only claim "no festival involved" when that's actually true for this item --
        # a festival-active item can also have a real trend signal on top of it
        qualifier = "" if has_festival_signal else " (no festival involved -- likely a seasonal or demand shift)"
        line += f"\n  {trend}{qualifier}"
    if ctx.get("seller_reported_note"):
        line += f"\n  seller reported: {ctx['seller_reported_note']}"
    return line


def _plain_stock_sentence(c: dict) -> str:
    """Stock position in one sentence, no context signals -- shared by the
    model-outage fallback so it never contradicts the grounded rationale used
    everywhere else."""
    s = f"On hand: {c['on_hand']} units, {c['days_of_cover']} days of cover."
    if c.get("on_order"):
        arriving = c.get("arriving_in_days")
        when = f", arriving in {arriving} day(s)" if arriving is not None else ""
        s += f" {c['on_order']} unit(s) already on order{when}."
    return s


def _fallback_ranking(consumption: dict) -> list[dict]:
    """Used only if the LLM fails on every retry -- deterministic, no rationale
    beyond saying so plainly, but never blocks the graph on a model outage."""
    ranked = sorted(consumption.values(), key=_severity_key)
    return [
        dict(
            item_id=c["item_id"], item_name=c["item_name"], category=c["category"],
            urgency="high" if c["risk"] == "stockout_risk" else "medium",
            rationale=_plain_stock_sentence(c), confidence=1.0,
            suggested_order_qty=c["suggested_order_qty"], days_of_cover=c["days_of_cover"], risk=c["risk"],
            on_order=c.get("on_order", 0), arriving_in_days=c.get("arriving_in_days"),
            source="fallback",
        )
        for c in ranked
    ]


def _safe_fallback_rationale(item_id: str, consumption: dict, context: dict) -> str:
    """Built entirely from verified facts, no LLM involved -- used whenever the
    model's own rationale fails the grounding check below, so a seller never sees
    a fabricated cause even when the ranking/urgency itself is fine. Covers both
    signal types available: the festival calendar and the recent-trend
    proxy -- the latter is what catches a Raincoat/Umbrella-style
    seasonal ramp that has no festival behind it at all."""
    c = consumption[item_id]
    ctx = context.get(item_id, {})
    parts = [f"On hand: {c['on_hand']} units, {c['days_of_cover']} days of cover."]
    if c.get("on_order"):
        arriving = c.get("arriving_in_days")
        when = f", arriving in {arriving} day(s)" if arriving is not None else ""
        parts.append(f"{c['on_order']} unit(s) already on order{when}.")
    if ctx.get("is_festival_window"):
        parts.append(f"Currently inside the {ctx['active_festival']} window (demand uplift x{ctx['uplift_multiplier']}).")
    elif ctx.get("days_to_next_festival") is not None:
        parts.append(f"{ctx['next_festival_name']} is {ctx['days_to_next_festival']} days away.")
    trend = _trend_note(c)
    if trend:
        parts.append(trend.capitalize() + ".")
    return " ".join(parts)


_TREND_WORDS = re.compile(
    r"\b(trend(ing)?|surg\w+|spik\w+|rising|risen|climb\w+|soar\w+|"
    r"falling|fallen|declin\w+|slow\w+ down|picking up)\b", re.IGNORECASE)
_SUPPLIER_WORDS = re.compile(
    r"\b(supplier|vendor|lead[\s-]?time|restock\w* delay|delivery delay|"
    r"unreliable|shipment delay)\b", re.IGNORECASE)


def _is_rationale_grounded(item_id: str, rationale: str, consumption: dict, context: dict,
                            all_festival_names: set[str]) -> bool:
    """
    Three checks, one per class of claim the facts block can actually support.

    1. Borrowed festival. The original failure: a smaller model batching many items
       in one prompt pattern-matches a festival name from a DIFFERENT item onto this
       one. True names for this item are allowed; any other real festival name in the
       text means it borrowed a reason that isn't this item's own.
    2. Invented trend. _format_item_facts only emits a trend line when
       _trend_note() fires. A rationale claiming a surge or a decline for an item
       that has no trend signal is asserting something it was never told.
    3. Invented supplier problem. The facts block never mentions supplier
       reliability or lead time at all, so ANY supplier claim is necessarily
       fabricated -- no allow-list needed, the whole class is out of bounds.

    Deliberately not checked: the numbers. Those are merged back from the
    deterministic dict by item_id lookup and never read from the model's text, so
    a number in a rationale can only be right or ignorable, never load-bearing.
    """
    ctx = context.get(item_id, {})
    true_names = {ctx.get("active_festival"), ctx.get("next_festival_name")} - {None}
    # a seller-reported note (context_extraction) can legitimately name a real
    # festival the calendar itself doesn't have this item marked for yet -- treat
    # any festival name that appears inside the seller's own note as true for
    # this item too, rather than flagging it as borrowed from another item
    note = ctx.get("seller_reported_note") or ""
    true_names |= {name for name in all_festival_names if name and name in note}
    if any(name in rationale for name in (all_festival_names - true_names) if name):
        return False

    if _trend_note(consumption[item_id]) is None and _TREND_WORDS.search(rationale):
        return False

    if _SUPPLIER_WORDS.search(rationale):
        return False

    return True


def make_reasoning_node(llm, festival_calendar: pd.DataFrame, max_attempts: int = 3):
    """
    Ranks and explains. Reads `chunk_size` off state: the whole flagged set in one
    call on the first pass, smaller batches on an escalation pass (see
    route_after_reasoning). Reports `grounding_failures` so the router can decide
    whether this output is trustworthy enough to show a seller.
    """
    structured_llm = llm.with_structured_output(RankedRecommendations)
    all_festival_names = set(festival_calendar["festival_name"])

    def _critique_batch(candidates: list[dict], consumption: dict, context: dict) -> dict:
        """
        A second, independent LLM call reviewing the first one's rationales
        against the same facts -- the writer doesn't grade its own homework.
        Only ever called on rationales that already passed the regex-based
        _is_rationale_grounded() check: that check is free, deterministic, and
        already airtight for its three known failure classes (borrowed festival,
        invented trend, invented supplier claim), so there is no reason to spend
        an LLM call re-checking what a denylist already caught. The critic exists
        for what a denylist structurally cannot catch: a conclusion the facts
        don't actually support, in a way nobody wrote a regex for.

        Returns {item_id: Critique}, or {} on any failure -- a bad response shape
        (including from a test fake or a provider with no real critic behind it),
        an exception, or a schema mismatch all degrade to "no critique happened",
        never to a crash or a blocked pipeline. When this returns {}, every
        candidate keeps the writer's own self-reported confidence exactly as
        before this feature existed.
        """
        if not candidates:
            return {}
        blocks = []
        for c in candidates:
            facts = _format_item_facts(c["item_id"], consumption, context)
            blocks.append(f"{facts}\n  Rationale under review: \"{c['rationale']}\"")
        prompt = (
            "You are reviewing another analyst's rationales for accuracy, one item at a "
            "time. For each item below, decide whether its rationale claims anything -- a "
            "festival, a trend, a supplier issue, a conclusion -- that the item's OWN facts "
            "don't actually support. Report your own confidence in each rationale's accuracy, "
            "independent of how confident the writer sounded.\n\n" + "\n\n".join(blocks)
        )
        for _ in range(max_attempts):
            try:
                result = llm.with_structured_output(BatchCritique).invoke(prompt)
                return {cr.item_id: cr for cr in result.critiques}
            except Exception:
                continue
        return {}

    def _rank_batch(batch_ids, consumption, context):
        """One LLM call over one batch, then one critique call over what survived
        the regex check. Returns (ranked_items, failed_grounding_count)."""
        facts = "\n".join(_format_item_facts(iid, consumption, context) for iid in batch_ids)
        prompt = (
            "You are a retail inventory analyst reviewing today's flagged items for a small "
            "store owner. The facts below are already computed -- do not recompute or second-"
            "guess the numbers, and do not state a quantity yourself. For each item, assign an "
            "urgency (high/medium/low) and write a one-to-two sentence rationale citing the "
            "specific reason from THAT item's OWN facts only -- never mention a festival, trend, "
            "or supplier issue that isn't explicitly listed for this exact item, even if it was "
            "mentioned for another item above. When an item's facts do name an active festival, "
            "lead with it and give its name: an approaching event is the most useful thing a shop "
            "owner can be told. Rank most urgent first.\n\n" + facts
        )

        result = None
        for _ in range(max_attempts):
            try:
                result = structured_llm.invoke(prompt)
                break
            except Exception:
                continue
        if result is None:
            return None, 0

        allowed = set(batch_ids)
        ranked, failures, to_critique = [], 0, []
        for r in result.recommendations:
            if r.item_id not in allowed:
                continue  # hallucinated id, or one from another batch -- never trust it
            c = consumption[r.item_id]
            rationale = r.rationale
            grounded = _is_rationale_grounded(r.item_id, rationale, consumption, context, all_festival_names)
            item = dict(
                item_id=r.item_id, item_name=c["item_name"], category=c["category"],
                urgency=r.urgency, rationale=rationale, confidence=round(r.confidence, 2),
                suggested_order_qty=c["suggested_order_qty"], days_of_cover=c["days_of_cover"],
                risk=c["risk"], on_order=c.get("on_order", 0),
                arriving_in_days=c.get("arriving_in_days"), source="model",
            )
            if not grounded:
                # a trust violation, not just uncertainty -- the text is replaced,
                # not merely flagged, so a fabricated claim never reaches a seller.
                # Not sent to the critic: it already failed the cheap check, so
                # there is nothing left for a second opinion to add.
                item["rationale"] = _safe_fallback_rationale(r.item_id, consumption, context)
                failures += 1
            else:
                to_critique.append(item)
            ranked.append(item)

        critiques = _critique_batch(to_critique, consumption, context)
        for item in to_critique:
            critique = critiques.get(item["item_id"])
            if critique is None:
                # no critic opinion for this item (critic unavailable, or omitted
                # it) -- fall back to the writer's own self-reported confidence,
                # exactly as this behaved before the critic existed
                if item["confidence"] < LOW_CONFIDENCE_THRESHOLD:
                    failures += 1
                continue
            if not critique.is_supported:
                # the critic caught something the regex denylist could not --
                # same treatment as a regex failure: replace, don't just flag
                item["rationale"] = _safe_fallback_rationale(item["item_id"], consumption, context)
                failures += 1
            else:
                # the critic's independent judgement replaces the writer's own
                # self-report -- a model grading its own uncertainty is a weaker
                # signal than a second call actually checking the claim
                item["confidence"] = round(critique.confidence, 2)
                if critique.confidence < LOW_CONFIDENCE_THRESHOLD:
                    failures += 1
        return ranked, failures

    def reasoning(state: SellerSenseState) -> dict:
        consumption, context = state["consumption_signals"], state["context_signals"]
        attempt = state.get("reasoning_attempt", 0) + 1
        if not consumption:
            return {"ranked_recommendations": [], "grounding_failures": 0,
                    "reasoning_attempt": attempt}

        item_ids = list(consumption)
        chunk = state.get("chunk_size") or len(item_ids)
        batches = [item_ids[i:i + chunk] for i in range(0, len(item_ids), chunk)]

        merged, seen_ids, failures, any_result = [], set(), 0, False
        for batch in batches:
            ranked, batch_failures = _rank_batch(batch, consumption, context)
            if ranked is None:
                continue  # this batch's model call failed outright; backfill covers it
            any_result = True
            failures += batch_failures
            merged.extend(ranked)
            seen_ids.update(r["item_id"] for r in ranked)

        if not any_result:
            return {"ranked_recommendations": _fallback_ranking(consumption),
                    "grounding_failures": 0, "reasoning_attempt": attempt}

        # a model handling a long list can silently omit items rather than erroring --
        # a long list makes this more likely. Never let a flagged item
        # vanish just because the model's response didn't mention it.
        for item_id, c in consumption.items():
            if item_id in seen_ids:
                continue
            merged.append(dict(
                item_id=item_id, item_name=c["item_name"], category=c["category"],
                urgency="high" if c["risk"] == "stockout_risk" else "medium",
                rationale=_safe_fallback_rationale(item_id, consumption, context), confidence=1.0,
                suggested_order_qty=c["suggested_order_qty"], days_of_cover=c["days_of_cover"],
                risk=c["risk"], on_order=c.get("on_order", 0),
                arriving_in_days=c.get("arriving_in_days"), source="backfill",
            ))
        return {"ranked_recommendations": merged, "grounding_failures": failures,
                "reasoning_attempt": attempt}
    return reasoning


# ---------------------------------------------------------------- routing

GROUNDING_FAILURE_LIMIT = 2
MAX_REASONING_ATTEMPTS = 3


def make_route_after_reasoning(failure_limit: int = GROUNDING_FAILURE_LIMIT,
                                max_attempts: int = MAX_REASONING_ATTEMPTS):
    """
    The one conditional edge in this graph. With many items in a single prompt,
    a smaller model can paste a festival name from one item's facts onto
    another's; _is_rationale_grounded() catches each instance and swaps in a
    fact-only sentence, but a run with several failures loses most of the
    value the model was there to add.

    Batch size is the lever: contamination scales with how many items' facts
    share one context, so a bad pass routes back into reasoning with the
    batch halved, trading one extra call for rationales that survive the
    check. Bounded at max_attempts so a confused model degrades to fact-only
    text instead of looping.
    """
    def route_after_reasoning(state: SellerSenseState) -> str:
        if state.get("grounding_failures", 0) < failure_limit:
            return "approve"
        if state.get("reasoning_attempt", 0) >= max_attempts:
            return "approve"  # out of retries: fact-only rationales are the floor, not a failure
        return "retry_smaller"
    return route_after_reasoning


def shrink_batch(state: SellerSenseState) -> dict:
    """Halve the batch for the next reasoning pass. Separate node rather than
    mutation inside the router because a LangGraph conditional edge must stay a
    pure read of state -- it decides where to go, never what changes."""
    current = state.get("chunk_size") or len(state.get("consumption_signals", {})) or 1
    return {"chunk_size": max(1, current // 2)}


# ---------------------------------------------------------------- human approval

def human_approval(state: SellerSenseState) -> dict:
    decision = interrupt({
        "recommendations": state["ranked_recommendations"],
        "prompt": "Approve, reject (with a reason), or snooze each recommendation.",
    })
    return {"seller_decision": decision}


# ---------------------------------------------------------------- graph assembly

def build_graph(sales, items, suppliers, festival_calendar, festival_overrides, promotions, llm,
                 max_items: int = 8, params_by_item: dict | None = None, purchase_orders=None,
                 forecast_tables: dict | None = None):
    graph = StateGraph(SellerSenseState)
    graph.add_node("gather_signals", make_gather_signals_node(
        sales, items, suppliers, festival_calendar, festival_overrides, promotions, max_items,
        params_by_item, purchase_orders, forecast_tables))
    graph.add_node("context_extraction", make_context_extraction_node(llm))
    graph.add_node("reasoning", make_reasoning_node(llm, festival_calendar))
    graph.add_node("human_approval", human_approval)
    graph.add_node("shrink_batch", shrink_batch)

    graph.add_edge(START, "gather_signals")
    graph.add_edge("gather_signals", "context_extraction")
    graph.add_edge("context_extraction", "reasoning")
    graph.add_conditional_edges("reasoning", make_route_after_reasoning(),
                                 {"retry_smaller": "shrink_batch", "approve": "human_approval"})
    graph.add_edge("shrink_batch", "reasoning")
    graph.add_edge("human_approval", END)

    # A checkpointer is required for interrupt() to work at all, but nothing
    # resumes from it -- see the module docstring. InMemorySaver is the right
    # choice precisely because that state is not meant to outlive the call.
    return graph.compile(checkpointer=InMemorySaver())
