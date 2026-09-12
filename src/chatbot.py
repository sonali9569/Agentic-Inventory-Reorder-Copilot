"""
Chatbot for answering seller questions using data already computed elsewhere
in the system. Does not make inventory decisions.

resolve_item_id() matches a message to a flagged item by comparing its
identifying words, with fallbacks for a typo (token similarity) or a pronoun
referring to the last item discussed. classify_intent() then determines
whether the message is a question or a command.

For a question, if the provider supports tool binding, the model is given
four read-only tools (list_flagged_items, get_item_status, get_item_context,
get_open_orders) and chooses which to call, within a step limit. A provider
without tool support falls back to a single grounded call using the same
computed facts (plain_facts()).

A detected command (approve/reject/snooze) is returned as a structured
payload but not executed here; the caller applies it. Reject reasons are read
from the message text directly, never inferred by the model, since a
fabricated reason would land in the feedback log and could drive a real
parameter change.

No retrieval or embeddings: the flagged set and recent feedback history fit
comfortably in a single prompt at this scale.
"""

import difflib
import re
import sys
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from feedback_agent import REASON_CODES
from graph import _trend_note


class ChatIntent(BaseModel):
    kind: Literal["question", "command", "unclear"]
    command_action: Optional[Literal["approve", "reject", "snooze"]] = Field(
        default=None, description="set only when kind == 'command'"
    )
    reject_reason: Optional[Literal[*REASON_CODES]] = Field(
        default=None, description="set only for a reject command, if a reason is stated or clearly implied"
    )


class ChatAnswer(BaseModel):
    answer: str = Field(
        description="a short, plain-language answer grounded ONLY in the facts given -- "
        "if the facts don't actually answer the question, say so plainly instead of guessing"
    )


_MEASUREMENT = re.compile(r"^\d+(\.\d+)?(g|kg|ml|l|gm|ltr)$")
_GENERIC_TOKENS = {"pack", "set", "box", "small", "large", "of", "and", "assorted"}


def _content_tokens(name: str) -> set[str]:
    """The words in an item name that actually identify it. Drops sizes ("150g",
    "5kg") and generic packaging words ("pack", "set", "box") -- nobody asks
    about their "Toothpaste 150g", they ask about toothpaste, and "box" on its
    own points at three different items."""
    tokens = re.findall(r"[a-z0-9]+", name.lower())
    return {
        t for t in tokens
        if t not in _GENERIC_TOKENS and not _MEASUREMENT.match(t) and not t.isdigit()
    }


_REFERENCE_WORDS = {"it", "that", "this", "same", "those"}


def resolve_item_id(message: str, consumption: dict, last_item_id: Optional[str] = None) -> Optional[str]:
    """
    Deterministic name matching, no LLM -- still true after the two fallbacks
    below, since neither one asks a model to guess. Scores each item by how many
    of its identifying words appear in the message, and takes the clear winner --
    so "why is toothpaste low" finds "Toothpaste 150g" without the seller typing
    the size. Whole-word matching only, so "tea" doesn't match "steal". The bare
    item_id matches too, for a UI that already knows it.

    Two things a purely literal match misses in real conversation, added as
    fallbacks that only fire when the exact match found nothing:

    1. Typos. "raincot", "umbrela" -- get_close_matches compares whole tokens
       with a high similarity cutoff, so it tolerates a slip without loosening
       the whole-word rule into a substring match.
    2. Pronouns. "what about it", "is that still low" -- resolved against
       last_item_id, the item the conversation was just about, so a seller
       doesn't have to re-type the name every single turn. Only used when the
       message actually contains a reference word; an unrelated question with
       no item named still returns None here (the tool-calling path handles
       those by listing the flagged set itself).

    Returns None -- never a guess -- when nothing matches, when two items tie
    (a bare "gift" could mean either the Gift Hamper or the Sweets Gift Box), or
    when a typo is ambiguous between two items.
    """
    words = set(re.findall(r"[a-z0-9]+", message.lower()))

    for item_id in consumption:
        if item_id.lower() in words:
            return item_id

    scores = {iid: len(_content_tokens(c["item_name"]) & words) for iid, c in consumption.items()}
    best = max(scores.values(), default=0)
    if best > 0:
        winners = [iid for iid, s in scores.items() if s == best]
        return winners[0] if len(winners) == 1 else None

    fuzzy_hits = {
        item_id for item_id, c in consumption.items()
        if any(len(word) >= 4 and difflib.get_close_matches(word, _content_tokens(c["item_name"]), n=1, cutoff=0.82)
               for word in words)
    }
    if fuzzy_hits:
        return fuzzy_hits.pop() if len(fuzzy_hits) == 1 else None

    if last_item_id in consumption and words & _REFERENCE_WORDS:
        return last_item_id

    return None


_ACTION_CUES = {
    "reject": ("reject", "decline", "cancel", "skip", "don't order", "dont order", "no thanks"),
    "approve": ("approve", "confirm", "go ahead", "place the order", "order it", "yes do it"),
    "snooze": ("snooze", "later", "remind me", "not now", "tomorrow"),
}

_REASON_CUES = {
    "qty_too_high": ("too high", "too many", "too much", "fewer", "less than", "reduce", "lower the qty"),
    "qty_too_low": ("too low", "too few", "not enough", "more than that", "increase"),
    "supplier_unreliable": ("supplier", "late", "delay", "unreliable", "never delivers"),
    "not_needed_now": ("not needed", "don't need", "dont need", "no cash", "no money", "next week"),
}


def _cue_match(message: str, cues: dict) -> Optional[str]:
    lowered = message.lower()
    hits = [key for key, phrases in cues.items() if any(p in lowered for p in phrases)]
    return hits[0] if len(hits) == 1 else None


def extract_reject_reason(message: str) -> Optional[str]:
    """
    Read the reason out of what the seller actually wrote. Deliberately never
    taken from the model: a small model fills in optional enum fields on nearly
    every call, and a fabricated reason here would land in the audit log as the
    seller's own words and drive a real parameter change. No cue in the message
    means no reason -- ask them, don't guess.
    """
    return _cue_match(message, _REASON_CUES)


def extract_command_action(message: str) -> Optional[str]:
    """Deterministic first pass at what the seller is asking for. Unlike the
    reason, a wrong guess here only produces a confirmation prompt, so the
    model's own answer is still usable as a fallback when no cue matches."""
    return _cue_match(message, _ACTION_CUES)


def classify_intent(llm, message: str, item_name: str) -> ChatIntent:
    prompt = (
        f"A shop owner sent this message about \"{item_name}\", an item currently flagged "
        "by their inventory copilot. Classify it.\n"
        "'command' means they're telling the system to DO something right now (approve an "
        "order, reject a recommendation, snooze it). 'question' means they're asking about "
        "it. 'unclear' if it's neither.\n\n"
        f"Message: \"{message}\""
    )
    try:
        return llm.with_structured_output(ChatIntent).invoke(prompt)
    except Exception:
        return ChatIntent(kind="unclear")


def plain_facts(item_id: str, consumption: dict, context: dict) -> str:
    """
    The same underlying numbers the Reasoning Agent gets, written as sentences
    instead of key=value pairs. The terse form is fine for ranking, where the
    model is sorting known fields; it is not fine for answering a person's
    question, where the model has to know that "flagged" means "reorder this"
    before it can say anything useful. Spelling that out here is what turns
    "the facts do not specify whether this is required" into a real answer.
    """
    c = consumption[item_id]
    ctx = context.get(item_id, {})
    lines = [f"Item: {c['item_name']} ({c['category']} category)",
             f"In stock right now: {c['on_hand']} units"]

    if c["days_of_cover"] is not None:
        lines.append(f"At the current selling rate that is roughly {c['days_of_cover']} days of stock left.")

    if c["risk"] == "stockout_risk":
        lines.append(
            f"This item is flagged as at risk of running out, which is why the system is "
            f"recommending reordering {c['suggested_order_qty']} units now. "
            f"It reorders once stock falls to {c['reorder_point']} units."
        )
    elif c["risk"] == "overstock":
        lines.append("This item is flagged as overstocked: there is more stock on hand than "
                     "the current selling rate justifies, so money is tied up in it.")
    else:
        lines.append(f"This item is healthy: stock is above its reorder point of "
                     f"{c['reorder_point']} units, so no order is needed right now.")

    if ctx.get("is_festival_window"):
        lines.append(f"{ctx['active_festival']} is on right now, which typically multiplies demand "
                     f"for this item by about {ctx['uplift_multiplier']}x.")
    elif ctx.get("days_to_next_festival") is not None:
        lines.append(f"{ctx['next_festival_name']} is {ctx['days_to_next_festival']} days away.")

    trend = _trend_note(c)
    if trend:
        lines.append(trend.capitalize() + ", compared with how it normally sells.")

    return "\n".join(lines)


def answer_question(llm, message: str, item_id: str, consumption: dict, context: dict) -> str:
    facts = plain_facts(item_id, consumption, context)
    prompt = (
        "You are an inventory assistant answering a shop owner's question about one item "
        "in their shop. Answer in one or two short, plain sentences they'd find useful -- "
        "no jargon, no field names.\n\n"
        "Use only the facts below and never invent a number. If they're asking whether they "
        "need to order something, the line about what the system is recommending is the answer. "
        "If they're asking why something was flagged, explain the reason in the facts (running "
        "low, a festival driving demand, or selling faster than usual).\n\n"
        f"Facts:\n{facts}\n\nTheir question: \"{message}\""
    )
    try:
        return llm.with_structured_output(ChatAnswer).invoke(prompt).answer
    except Exception:
        return f"Here's what I have on {c_name(consumption, item_id)}:\n{facts}"


def c_name(consumption: dict, item_id: str) -> str:
    return consumption[item_id]["item_name"]


def respond(llm, message: str, consumption: dict, context: dict,
            last_item_id: Optional[str] = None) -> dict:
    """
    The single entry point. Returns one of:
      {"kind": "answer", "text": "...", "source": "tools" | "grounded"}
      {"kind": "command", "item_id": ..., "action": ..., "reject_reason": ...,
       "text": "..."}                                   -- detected, NOT executed
      {"kind": "unclear", "text": "..."}
    Every branch also carries "resolved_item_id": the item this turn was about,
    or None. A caller keeping a chat session open should pass it back in as the
    next call's last_item_id -- that's the whole of this system's conversation
    memory: which single item was last discussed, not a transcript. It's what
    lets "why is umbrella low" followed by "when's it arriving" resolve "it"
    without the seller repeating the name (see resolve_item_id's pronoun case).
    """
    item_id = resolve_item_id(message, consumption, last_item_id=last_item_id)

    if item_id is None:
        # deterministic matching failed. With tool calling the model can still
        # answer by listing the flagged set itself, which is exactly the class of
        # question ("what needs ordering today?") the old path could never handle.
        if supports_tool_calling(llm):
            # no item was named, so this is exactly the class of question
            # ("what needs attention today?") where a smaller model most often
            # answers without calling anything -- seed the one always-safe,
            # zero-argument tool deterministically rather than hoping it does
            answer = answer_with_tools(llm, message, consumption, context, seed_tools=("list_flagged_items",))
            if answer:
                return dict(kind="answer", text=answer, source="tools", resolved_item_id=None)
        return dict(kind="unclear", text="I'm not sure which item that's about -- try naming it directly.",
                    resolved_item_id=None)

    item_name = consumption[item_id]["item_name"]
    intent = classify_intent(llm, message, item_name)

    if intent.kind == "command":
        # the seller's own words win over the model's for both fields; the reason
        # is taken ONLY from the message, never from the model (see extract_reject_reason)
        action = extract_command_action(message) or intent.command_action
        reason = extract_reject_reason(message)

        if action is None:
            return dict(kind="unclear", resolved_item_id=item_id,
                        text=f"I can tell this is about {item_name}, but not what you want done with it -- approve, reject, or snooze?")
        if action == "reject" and reason is None:
            return dict(kind="unclear", resolved_item_id=item_id,
                        text=f"Got that you want to reject {item_name} -- what's the reason? "
                             "(quantity too high, quantity too low, supplier unreliable, or not needed right now)")
        return dict(
            kind="command", item_id=item_id, action=action, reject_reason=reason, resolved_item_id=item_id,
            text=f"Got it -- {action} on {item_name}"
                 + (f" ({reason.replace('_', ' ')})" if reason else "") + ". Confirm to apply this.",
        )

    if intent.kind == "question":
        # tool path first: the model picks its own lookups. Falls back to the
        # single-call grounded path when the provider has no bind_tools (demo
        # cache) or the loop runs out of steps -- same answer quality floor either
        # way, since both read from the same computed facts.
        if supports_tool_calling(llm):
            answer = answer_with_tools(llm, message, consumption, context)
            if answer:
                return dict(kind="answer", text=answer, source="tools", resolved_item_id=item_id)
        return dict(kind="answer", resolved_item_id=item_id,
                    text=answer_question(llm, message, item_id, consumption, context),
                    source="grounded")

    return dict(kind="unclear", resolved_item_id=item_id,
                text=f"I found {item_name} but I'm not sure what you're asking -- could you rephrase?")


# ---------------------------------------------------------------- tool-calling path

# 4 was enough for a single-item question (status + context, or status + open
# orders, then a final answer -- see answer_with_tools's docstring) but not for
# a genuine cross-item question ("compare umbrella and raincoat this month"),
# which needs a status and a context call for EACH item before it has enough to
# compare -- 4 calls alone, leaving no room for the final answer. 6 covers two
# items at full depth (2 x status + 2 x context + 1 final, with one call to
# spare) without opening the loop up to an unbounded multi-item sweep.
MAX_TOOL_STEPS = 6

# How many times an ungrounded opening reply (no tool call at all) gets a
# corrective nudge before its text is simply returned as-is. 2, not 1: a
# smaller model occasionally stays ungrounded through a first retry too
# (observed directly against a live provider), and this still leaves at
# least one step of the 6-step budget free for the model's actual tool
# calls and final answer even in that worst case.
MAX_UNGROUNDED_NUDGES = 2

# Marks a reply that declines to answer despite already having facts in
# context (a seeded tool result, or its own earlier tool call) -- as
# distinct from a reply that has no facts at all yet. Deliberately narrow
# (English hedge phrases only, observed directly against a live provider)
# rather than a broad sentiment check: a false negative here just costs the
# unhelpful reply going out as-is, exactly the pre-existing behaviour; a
# false positive costs one harmless extra nudge round-trip.
_REFUSAL_WORDS = re.compile(
    r"\b(not sure|don'?t have (enough|any)?|do not have (enough|any)?|"
    r"unable to (determine|tell)|no information|can'?t tell|cannot tell)\b",
    re.IGNORECASE)


def make_tools(consumption: dict, context: dict):
    """
    Read-only lookups over the current cycle's computed state. Deliberately
    close-ended: each returns text the model can quote, never a handle it can
    compute with, and none of them mutates anything. An unknown item_id comes back
    as an ordinary sentence rather than an exception, because a model that guessed
    a SKU should be corrected in the conversation, not crash the turn.
    """

    @tool
    def list_flagged_items() -> str:
        """List every item currently flagged, with its risk type, days of cover, and
        any demand trend -- usually enough on its own to answer a broad question
        like "what's most urgent" or "what's in most demand" without a follow-up
        call per item. Use this when the question is about the shop overall, or
        when you are unsure which item is meant."""
        if not consumption:
            return "Nothing is flagged right now."
        lines = []
        for iid, c in consumption.items():
            line = (f"{iid}: {c['item_name']} ({c['category']}) — {c['risk'].replace('_', ' ')}, "
                    f"{c['days_of_cover']} days of cover")
            trend = _trend_note(c)
            if trend:
                line += f", {trend}"
            lines.append(line)
        return "\n".join(lines)

    @tool
    def get_item_status(item_id: str) -> str:
        """Stock position for one item: units on hand, days of cover, reorder point,
        the recommended order quantity and why it is flagged. Takes the item_id."""
        if item_id not in consumption:
            return (f"No item {item_id} in today's flagged list. "
                    f"Call list_flagged_items to see what is available.")
        return plain_facts(item_id, consumption, context)

    @tool
    def get_item_context(item_id: str) -> str:
        """Demand context for one item: whether a festival is active or approaching,
        any uplift, and whether a promotion is running. Takes the item_id."""
        if item_id not in consumption:
            return f"No item {item_id} in today's flagged list."
        ctx = context.get(item_id, {})
        if ctx.get("is_festival_window"):
            base = (f"{ctx['active_festival']} is running now, typically about "
                    f"{ctx['uplift_multiplier']}x normal demand for this item.")
        elif ctx.get("days_to_next_festival") is not None:
            base = f"{ctx['next_festival_name']} is {ctx['days_to_next_festival']} days away."
        else:
            base = "No festival applies to this item right now."
        if ctx.get("promo_active"):
            base += f" A promotion is running at about {ctx['promo_uplift']}x."
        return base

    @tool
    def get_open_orders(item_id: str) -> str:
        """Whether stock is already on the way for one item, and when it arrives.
        Takes the item_id. Use this before telling anyone to order more."""
        if item_id not in consumption:
            return f"No item {item_id} in today's flagged list."
        c = consumption[item_id]
        if not c.get("on_order"):
            return f"Nothing on order for {c['item_name']} right now."
        arriving = c.get("arriving_in_days")
        when = f", arriving in about {arriving} day(s)" if arriving is not None else ""
        return f"{c['on_order']} units of {c['item_name']} are already on order{when}."

    return [list_flagged_items, get_item_status, get_item_context, get_open_orders]


def supports_tool_calling(llm) -> bool:
    """The demo cache and some wrappers expose only with_structured_output."""
    return hasattr(llm, "bind_tools")


def answer_with_tools(llm, message: str, consumption: dict, context: dict,
                       max_steps: int = MAX_TOOL_STEPS, seed_tools: tuple[str, ...] = ()) -> Optional[str]:
    """
    Bind the read-only lookups and let the model choose. Returns None if the loop
    cannot produce an answer, so the caller can fall back rather than showing an
    empty reply.

    The step budget is the important part. An unbounded tool loop is how an agent
    burns a rate limit on a question it was never going to answer; six steps
    covers every question this surface is meant to handle, including a genuine
    two-item comparison ("compare umbrella and raincoat this month"), which needs
    a status and a context call for EACH item before there's enough to compare --
    four calls alone, before the model has said anything back. Still bounded: a
    seller asking about a whole aisle at once is out of scope for a budget sized
    for two items, and should fall back rather than spiral.

    seed_tools names zero-argument, side-effect-free tools (list_flagged_items
    is the only one that qualifies) to call before the model's own turn, so a
    general question is grounded from the start rather than depending on the
    model choosing to call a tool on its own.

    An ungrounded reply -- no tool call, from the model or from a seed -- gets
    up to MAX_UNGROUNDED_NUDGES corrective retries within the same step
    budget before its text is returned as-is. A reply that follows at least
    one real tool call is trusted immediately.
    """
    tools = make_tools(consumption, context)
    registry = {t.name: t for t in tools}

    try:
        bound = llm.bind_tools(tools)
    except Exception:
        return None

    messages = [
        SystemMessage(content=(
            "You are SellerSense, an inventory assistant for a small shop owner. Every "
            "message from this user is about their shop's inventory, sales, or orders -- "
            "never about anything else, so never ask which domain or area they mean. You "
            "start each question with NO information: never invent a number, an item, or "
            "a reason, and never guess an answer before checking -- call a tool whenever "
            "you need a fact, including for a general question like 'what needs "
            "attention today', which means calling list_flagged_items first. When you "
            "have enough facts, reply in one or two short plain sentences with no jargon, "
            "no field names, and no internal codes -- tool results are labelled with an "
            "item_id like 'I016' for your own lookups only; a shop owner never sees that, "
            "only the plain item name. If the tools genuinely don't answer the question "
            "after checking, say so plainly. Before suggesting anyone orders more of "
            "something, check whether it is already on order. For a broad question like "
            "'what's most urgent' or 'what's in most demand', list_flagged_items already "
            "includes days of cover and demand trend for every item -- that is usually "
            "enough to answer directly; only call get_item_context for one or two items "
            "you are actually unsure about, not every item in the list."
        )),
        HumanMessage(content=message),
    ]

    grounded = False   # whether at least one real tool call has happened yet
    nudges_used = 0     # corrective retries spent so far, capped below

    for name in seed_tools:
        fn = registry.get(name)
        if fn is None:
            continue
        try:
            output = fn.invoke({})
        except Exception:
            continue  # a seed is a courtesy, not a requirement -- the model's own turn still follows
        # appended as plain context rather than a synthetic ToolMessage: a
        # bare ToolMessage with no matching preceding assistant tool_calls
        # is invalid on an OpenAI-compatible API and would itself error
        messages.append(SystemMessage(content=f"Already looked up for you -- {name}: {output}"))
        grounded = True

    for _ in range(max_steps):
        try:
            reply = bound.invoke(messages)
        except Exception:
            return None
        messages.append(reply)

        calls = getattr(reply, "tool_calls", None) or []
        if not calls:
            text = (reply.content or "").strip()
            # A seeded fact guarantees the model WAS given something to work
            # with; it does not guarantee the model actually used it -- a
            # seeded turn can still open with a refusal ("I don't have
            # enough data") despite the facts sitting right there in
            # context. That is a different failure than a genuinely
            # ungrounded reply, but it is still wrong, and it must not be
            # trusted just because `grounded` is already true from the seed.
            refuses_despite_facts = grounded and _REFUSAL_WORDS.search(text)
            if (grounded and not refuses_despite_facts) or nudges_used >= MAX_UNGROUNDED_NUDGES or not text:
                return text or None
            nudges_used += 1
            nudge = (
                "The facts you need are already given above (see 'Already looked up for "
                "you') -- answer directly from that instead of saying you lack data."
                if refuses_despite_facts else
                "You haven't looked anything up yet, so you don't have an answer. This "
                "is always about the shop's inventory -- call list_flagged_items or "
                "another tool now, then answer from what it returns."
            )
            messages.append(HumanMessage(content=nudge))
            continue

        grounded = True
        for call in calls:
            fn = registry.get(call["name"])
            # a hallucinated tool name is answered, not raised -- the model gets a
            # chance to correct itself inside the same turn
            output = (fn.invoke(call["args"]) if fn
                      else f"No tool named {call['name']}. Available: {', '.join(registry)}.")
            messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))

    return None  # out of steps: caller falls back
