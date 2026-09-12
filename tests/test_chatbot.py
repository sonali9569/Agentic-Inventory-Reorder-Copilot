import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chatbot import (
    answer_with_tools,
    make_tools,
    ChatAnswer,
    ChatIntent,
    answer_question,
    classify_intent,
    plain_facts,
    resolve_item_id,
    respond,
)


class _Structured:
    def __init__(self, response=None, raises=False):
        self.response = response
        self.raises = raises

    def invoke(self, prompt):
        if self.raises:
            raise ValueError("simulated LLM failure")
        return self.response


class _FakeLLM:
    """Hands back a different canned response depending on which schema was
    requested, since respond() can chain classify_intent() then answer_question()
    -- each asks with_structured_output() for a different schema in sequence."""
    def __init__(self, intent=None, answer=None, intent_raises=False, answer_raises=False):
        self.intent = intent
        self.answer = answer
        self.intent_raises = intent_raises
        self.answer_raises = answer_raises

    def with_structured_output(self, schema):
        if schema is ChatIntent:
            return _Structured(self.intent, raises=self.intent_raises)
        if schema is ChatAnswer:
            return _Structured(self.answer, raises=self.answer_raises)
        raise AssertionError(f"unexpected schema requested: {schema}")


CONSUMPTION = {
    "I021": dict(item_id="I021", item_name="Raincoat", category="Durables",
                 risk="stockout_risk", on_hand=0, days_of_cover=0.0, reorder_point=39,
                 suggested_order_qty=39, trend_ratio=None),
    "I016": dict(item_id="I016", item_name="Rakhi Set", category="Festive",
                 risk="stockout_risk", on_hand=0, days_of_cover=0.0, reorder_point=2,
                 suggested_order_qty=2, trend_ratio=15.6),
    "I013": dict(item_id="I013", item_name="Toothpaste 150g", category="FMCG",
                 risk="stockout_risk", on_hand=1, days_of_cover=0.3, reorder_point=39,
                 suggested_order_qty=38, trend_ratio=None),
}
CONTEXT = {
    "I021": dict(is_festival_window=False, days_to_next_festival=None),
    "I016": dict(is_festival_window=True, active_festival="Raksha Bandhan", uplift_multiplier=14.0),
    "I013": dict(is_festival_window=False, days_to_next_festival=None),
}


# ---------------------------------------------------------------- resolve_item_id (deterministic, no LLM)

def test_resolve_item_id_matches_by_name_case_insensitively():
    assert resolve_item_id("why is my Rakhi Set low?", CONSUMPTION) == "I016"
    assert resolve_item_id("RAINCOAT stock?", CONSUMPTION) == "I021"


def test_resolve_item_id_matches_without_the_size_suffix():
    # a seller says "toothpaste", never "Toothpaste 150g" -- requiring the full
    # name including the size made every realistic question fail to resolve
    assert resolve_item_id("why toothpaste is recommended", CONSUMPTION) == "I013"
    assert resolve_item_id("do I need more toothpaste?", CONSUMPTION) == "I013"


def test_resolve_item_id_still_matches_the_full_name_with_size():
    assert resolve_item_id("Toothpaste 150g is required?", CONSUMPTION) == "I013"


def test_resolve_item_id_matches_by_bare_id():
    assert resolve_item_id("what about I021", CONSUMPTION) == "I021"


def test_resolve_item_id_returns_none_when_nothing_matches():
    assert resolve_item_id("what about my umbrellas", CONSUMPTION) is None


def test_resolve_item_id_matches_whole_words_only():
    # "tea" must not match inside "steal"/"instead"
    assert resolve_item_id("instead of that, what should I steal", CONSUMPTION) is None


def test_resolve_item_id_returns_none_when_ambiguous():
    assert resolve_item_id("raincoat and rakhi set both need restocking", CONSUMPTION) is None


# ---------------------------------------------------------------- classify_intent

def test_classify_intent_passes_through_the_structured_result():
    canned = ChatIntent(kind="question")
    result = classify_intent(_FakeLLM(intent=canned), "why is it low", "Raincoat")
    assert result.kind == "question"


def test_classify_intent_falls_back_to_unclear_on_llm_failure():
    result = classify_intent(_FakeLLM(intent_raises=True), "anything", "Raincoat")
    assert result.kind == "unclear"


# ---------------------------------------------------------------- answer_question

def test_answer_question_falls_back_to_facts_when_llm_fails():
    text = answer_question(_FakeLLM(answer_raises=True), "why is this low?", "I021", CONSUMPTION, CONTEXT)
    assert "Raincoat" in text
    assert "In stock right now: 0 units" in text  # fact-only fallback, in plain language


# ---------------------------------------------------------------- plain_facts

def test_plain_facts_spells_out_what_flagged_actually_means():
    # the model can't answer "do I need to order this?" from `risk=stockout_risk`
    # alone -- the facts have to say what that flag implies
    facts = plain_facts("I013", CONSUMPTION, CONTEXT)
    assert "recommending reordering 38 units" in facts
    assert "0.3 days of stock left" in facts


def test_plain_facts_includes_an_active_festival_and_its_uplift():
    facts = plain_facts("I016", CONSUMPTION, CONTEXT)
    assert "Raksha Bandhan is on right now" in facts
    assert "14.0x" in facts


def test_plain_facts_has_no_raw_field_names():
    facts = plain_facts("I021", CONSUMPTION, CONTEXT)
    for jargon in ("risk=", "on_hand=", "days_of_cover=", "suggested_order_qty="):
        assert jargon not in facts


# ---------------------------------------------------------------- respond (the full loop)

def test_respond_resolves_the_item_by_name_before_ever_calling_the_llm():
    llm = _FakeLLM(intent=ChatIntent(kind="question"), answer=ChatAnswer(answer="Zero stock, no cover."))
    result = respond(llm, "why is rakhi set flagged?", CONSUMPTION, CONTEXT)
    assert result["kind"] == "answer"
    assert "Zero stock" in result["text"]


def test_respond_is_unclear_immediately_when_no_item_name_matches_no_llm_call_needed():
    # the LLM is never given a response here -- if respond() tried to call it,
    # the fake would raise AssertionError from the unexpected-schema branch
    result = respond(_FakeLLM(), "how's business today", CONSUMPTION, CONTEXT)
    assert result["kind"] == "unclear"


def test_respond_detects_a_command_without_executing_it():
    llm = _FakeLLM(intent=ChatIntent(kind="command", command_action="reject", reject_reason="qty_too_high"))
    result = respond(llm, "reject the rakhi set order, the quantity is too high", CONSUMPTION, CONTEXT)
    assert result["kind"] == "command"
    assert result["item_id"] == "I016"
    assert result["action"] == "reject"
    assert result["reject_reason"] == "qty_too_high"
    # respond() never imports or calls anything from feedback_agent -- detection only


def test_respond_ignores_a_reject_reason_the_seller_never_gave():
    # a small model fills in optional enum fields on nearly every call. A reason
    # invented here would land in the audit log as the seller's own words and
    # drive a real parameter change, so it is only ever read from the message.
    llm = _FakeLLM(intent=ChatIntent(kind="command", command_action="reject", reject_reason="not_needed_now"))
    result = respond(llm, "reject the raincoat", CONSUMPTION, CONTEXT)  # no reason stated
    assert result["kind"] == "unclear"
    assert "reason" in result["text"].lower()


def test_respond_reads_the_action_from_the_sellers_words():
    # the model returning no action at all must not block a plainly worded command
    llm = _FakeLLM(intent=ChatIntent(kind="command", command_action=None, reject_reason=None))
    result = respond(llm, "reject the raincoat, supplier is always late", CONSUMPTION, CONTEXT)
    assert result["kind"] == "command"
    assert result["action"] == "reject"
    assert result["reject_reason"] == "supplier_unreliable"


def test_respond_does_not_render_a_bogus_none_action():
    # kind=="command" with command_action left unset by the model must
    # not silently render as "Got it -- None on <item>" -- ask instead of guessing
    llm = _FakeLLM(intent=ChatIntent(kind="command", command_action=None))
    result = respond(llm, "do something about the raincoat", CONSUMPTION, CONTEXT)
    assert result["kind"] == "unclear"
    assert "None" not in result["text"]


def test_respond_asks_for_a_reason_when_reject_has_none():
    llm = _FakeLLM(intent=ChatIntent(kind="command", command_action="reject", reject_reason=None))
    result = respond(llm, "reject the raincoat", CONSUMPTION, CONTEXT)
    assert result["kind"] == "unclear"
    assert "reason" in result["text"].lower()


def test_respond_handles_unclear_intent_gracefully():
    llm = _FakeLLM(intent=ChatIntent(kind="unclear"))
    result = respond(llm, "raincoat, hmm", CONSUMPTION, CONTEXT)
    assert result["kind"] == "unclear"


# ---------------------------------------------------------------- tool-calling path

class _ToolCallingLLM:
    """A provider that supports bind_tools. Scripted: emits the tool calls it is
    told to, then a final answer. Records what the model actually chose to call."""

    def __init__(self, script):
        self.script = list(script)   # list of (tool_name, args) lists, then a string
        self.chosen = []
        self.tools = None

    def bind_tools(self, tools):
        self.tools = {t.name: t for t in tools}
        return self

    def with_structured_output(self, schema):
        """Intent classification still runs first and still uses structured output:
        a command has to be routed deterministically, never executed by the model.
        Only the QUESTION branch reaches the tool loop."""
        outer = self

        class _Intent:
            def invoke(self, prompt):
                if schema is ChatAnswer:
                    raise AssertionError("question answering must use the tool path here")
                return ChatIntent(kind="question")
        return _Intent()

    def invoke(self, messages):
        step = self.script.pop(0)
        if isinstance(step, str):
            return AIMessage(content=step)
        calls = [dict(name=n, args=a, id=f"c{i}") for i, (n, a) in enumerate(step)]
        self.chosen.extend(n for n, _ in step)
        return AIMessage(content="", tool_calls=calls)


def _state():
    consumption = {
        "I013": dict(item_id="I013", item_name="Toothpaste 150g", category="FMCG",
                     on_hand=1, days_of_cover=0.3, reorder_point=39,
                     suggested_order_qty=0, risk="stockout_risk",
                     on_order=49, arriving_in_days=2, trend_ratio=None),
    }
    context = {"I013": dict(is_festival_window=False, days_to_next_festival=None,
                            promo_active=False, promo_uplift=1.0)}
    return consumption, context


def test_model_picks_its_own_tools_and_answers_from_them():
    consumption, context = _state()
    llm = _ToolCallingLLM([
        [("get_item_status", {"item_id": "I013"})],
        [("get_open_orders", {"item_id": "I013"})],
        "Toothpaste is nearly out, but 49 units land in about 2 days — no need to order.",
    ])
    out = respond(llm, "should I order more toothpaste?", consumption, context)
    assert out["kind"] == "answer"
    assert out["source"] == "tools"
    assert llm.chosen == ["get_item_status", "get_open_orders"]


def _two_item_state():
    consumption = {
        "I020": dict(item_id="I020", item_name="Umbrella", category="Seasonal",
                     on_hand=2, days_of_cover=1.0, reorder_point=30,
                     suggested_order_qty=28, risk="stockout_risk",
                     on_order=0, arriving_in_days=None, trend_ratio=3.1),
        "I021": dict(item_id="I021", item_name="Raincoat", category="Durables",
                     on_hand=0, days_of_cover=0.0, reorder_point=39,
                     suggested_order_qty=39, risk="stockout_risk",
                     on_order=0, arriving_in_days=None, trend_ratio=None),
    }
    context = {
        "I020": dict(is_festival_window=False, days_to_next_festival=None,
                     promo_active=False, promo_uplift=1.0),
        "I021": dict(is_festival_window=False, days_to_next_festival=None,
                     promo_active=False, promo_uplift=1.0),
    }
    return consumption, context


def test_model_can_compare_two_items_within_the_step_budget():
    # a genuine cross-item question: needs status AND context for BOTH items
    # before there's enough to compare -- four tool calls, then the answer.
    # Regression target for MAX_TOOL_STEPS: this must fit without falling back.
    consumption, context = _two_item_state()
    llm = _ToolCallingLLM([
        [("get_item_status", {"item_id": "I020"})],
        [("get_item_status", {"item_id": "I021"})],
        [("get_item_context", {"item_id": "I020"})],
        [("get_item_context", {"item_id": "I021"})],
        "Raincoat needs more urgent attention -- it's completely out, while "
        "Umbrella still has a day of cover left.",
    ])
    out = respond(llm, "compare umbrella and raincoat, which needs more attention?",
                  consumption, context)
    assert out["kind"] == "answer"
    assert out["source"] == "tools"
    assert llm.chosen == ["get_item_status", "get_item_status", "get_item_context", "get_item_context"]


def test_tools_let_the_model_answer_a_question_naming_no_item():
    # the old path returned "I'm not sure which item that's about" here, because
    # resolve_item_id had nothing to match on
    consumption, context = _state()
    llm = _ToolCallingLLM([
        [("list_flagged_items", {})],
        "One item needs attention: Toothpaste 150g is at risk of running out.",
    ])
    out = respond(llm, "what needs attention today?", consumption, context)
    assert out["kind"] == "answer"
    assert llm.chosen == ["list_flagged_items"]


def test_ungrounded_first_reply_gets_one_nudge_then_recovers():
    # a smaller model occasionally answers (or asks an off-topic clarifying
    # question) on the very first turn without calling any tool at all, even
    # though it was given zero facts -- this must get one corrective retry
    # rather than surfacing that ungrounded guess as the answer
    consumption, context = _state()
    llm = _ToolCallingLLM([
        "I'm not sure which area you mean -- news, work, or your schedule?",
        [("list_flagged_items", {})],
        "One item needs attention: Toothpaste 150g is at risk of running out.",
    ])
    out = answer_with_tools(llm, "what is the most demanding thing today?", consumption, context)
    assert out == "One item needs attention: Toothpaste 150g is at risk of running out."
    assert llm.chosen == ["list_flagged_items"]


def test_ungrounded_reply_after_all_nudges_is_still_returned():
    # the nudge budget (MAX_UNGROUNDED_NUDGES) is bounded, not a second
    # unbounded loop -- a model that stays ungrounded through every nudge
    # still gets its last answer text back rather than losing it to None
    consumption, context = _state()
    llm = _ToolCallingLLM([
        "Not sure what you mean.",
        "Still not sure what you mean.",
        "Really still not sure what you mean.",
    ])
    out = answer_with_tools(llm, "what is the most demanding thing today?", consumption, context)
    assert out == "Really still not sure what you mean."


def test_unknown_item_id_is_answered_not_raised():
    consumption, context = _state()
    tools = {t.name: t for t in make_tools(consumption, context)}
    out = tools["get_item_status"].invoke({"item_id": "NOPE"})
    assert "No item NOPE" in out
    assert "list_flagged_items" in out


def test_tool_loop_is_bounded_and_falls_back_rather_than_spinning():
    consumption, context = _state()
    # never stops calling tools: the budget must end the loop
    llm = _ToolCallingLLM([[("get_item_status", {"item_id": "I013"})]] * 12)
    assert answer_with_tools(llm, "why?", consumption, context, max_steps=3) is None
    assert len(llm.chosen) == 3


def test_commands_never_reach_the_tool_path():
    # tools are read-only by construction; a command must still go through the
    # deterministic extract/confirm route, not be executed by the model
    consumption, context = _state()
    assert all(t.name.startswith("get_") or t.name.startswith("list_")
               for t in make_tools(consumption, context))


# ---------------------------------------------------------------- resolve_item_id: typo & pronoun fallbacks

def test_resolve_item_id_tolerates_a_typo():
    assert resolve_item_id("what about my raincot", CONSUMPTION) == "I021"


def test_resolve_item_id_typo_fallback_never_fires_when_nothing_is_close():
    # "umbrellas" isn't a near-miss of any item name in this fixture -- must
    # still return None rather than guessing the nearest thing available
    assert resolve_item_id("what about my umbrellas", CONSUMPTION) is None


def test_resolve_item_id_resolves_a_pronoun_against_last_item_id():
    assert resolve_item_id("when's it arriving?", CONSUMPTION, last_item_id="I021") == "I021"


def test_resolve_item_id_pronoun_does_nothing_without_last_item_id():
    assert resolve_item_id("when's it arriving?", CONSUMPTION) is None


def test_resolve_item_id_pronoun_ignored_if_last_item_id_not_currently_flagged():
    # the previously-discussed item is no longer in today's flagged set --
    # never resolve a pronoun to an item that isn't actually in play
    assert resolve_item_id("when's it arriving?", CONSUMPTION, last_item_id="I999") is None


def test_resolve_item_id_exact_match_still_wins_over_pronoun_or_typo():
    # a message naming a real item should never be redirected to last_item_id
    assert resolve_item_id("what about the toothpaste", CONSUMPTION, last_item_id="I021") == "I013"


def test_respond_carries_resolved_item_id_for_the_next_turn():
    llm = _FakeLLM(intent=ChatIntent(kind="question"), answer=ChatAnswer(answer="It's low."))
    out = respond(llm, "why is raincoat low?", CONSUMPTION, CONTEXT)
    assert out["resolved_item_id"] == "I021"

    # next turn: no item named, but the pronoun resolves via last turn's result
    out2 = respond(llm, "when's it arriving?", CONSUMPTION, CONTEXT, last_item_id=out["resolved_item_id"])
    assert out2["resolved_item_id"] == "I021"


def test_respond_resolved_item_id_is_none_when_nothing_resolves():
    out = respond(_FakeLLM(), "how's business today", CONSUMPTION, CONTEXT)
    assert out["resolved_item_id"] is None
