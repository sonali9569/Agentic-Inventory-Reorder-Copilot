"""
Visualization only. Makes the internal "agent chatter" visible in the terminal --
which node ran, what the router decided, and every tool call the chatbot's model
makes -- none of which the product prints today (see chatbot.py and graph.py
docstrings: the system is deliberately silent about its own control flow, since
a seller wants an answer, not a transcript).

Nothing here changes graph.py or chatbot.py. Part 1 uses LangGraph's own
graph.stream(..., stream_mode="updates") instead of build_graph()'s .invoke(),
which is a property of how the compiled graph is called, not how it's built.
Part 2 is a traced re-implementation of chatbot.answer_with_tools's loop --
duplicated, not imported, specifically so every step can be printed; if that
loop changes in chatbot.py, update the copy here too.

Run:
    python3 src/run_agent_trace.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chatbot import (answer_question, classify_intent, make_tools,
                      plain_facts, resolve_item_id, supports_tool_calling)
from engine import assess_item, days_to_next_arrival, on_order_qty
from context_agent import get_context
from graph import build_graph
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

sales = pd.read_csv(DATA / "daily_sales.csv", parse_dates=["date"])
items = pd.read_csv(DATA / "items.csv")
suppliers = pd.read_csv(DATA / "suppliers.csv")
festival_calendar = pd.read_csv(DATA / "festival_calendar.csv", parse_dates=["date"])
festival_overrides = pd.read_csv(DATA / "festival_item_overrides.csv")
promotions = pd.read_csv(DATA / "promotions.csv", parse_dates=["date"])
purchase_orders = pd.read_csv(DATA / "purchase_orders.csv", parse_dates=["date_ordered", "date_expected"])

AS_OF = "2026-08-24"  # mid-monsoon (Raincoat/Umbrella) + inside the Raksha Bandhan window (Rakhi Set)


def _get_llm():
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model="qwen3:4b", temperature=0)
        llm.invoke("ping")
        print("using a real local Ollama model (qwen3:4b)\n")
        return llm
    except Exception as e:
        print(f"Ollama not reachable ({e!r}) -- both traces below need a real model "
              "with tool support, stopping.\n"
              "Install Ollama and `ollama pull qwen3:4b`, then re-run this script.")
        sys.exit(1)


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ==================================================================== PART 1
# The ordering graph: gather_signals -> reasoning -> [route] -> human_approval
# streamed node-by-node instead of invoked as one blocking call.

def trace_graph(llm) -> None:
    hr("PART 1 -- ordering graph, node by node")

    graph = build_graph(sales, items, suppliers, festival_calendar, festival_overrides,
                         promotions, llm, purchase_orders=purchase_orders)
    config = {"configurable": {"thread_id": f"trace-{AS_OF}"}}

    for update in graph.stream({"as_of_date": AS_OF}, config, stream_mode="updates"):
        for node_name, node_output in update.items():
            print(f"\n[{node_name}]")

            if node_name == "gather_signals":
                print(f"  computed risk for {node_output['total_flagged_count']} flagged item(s) "
                      f"out of {len(items)} total")

            elif node_name == "context_extraction":
                if node_output:
                    print(f"  extracted a seller-reported signal: {node_output}")
                else:
                    print("  no seller_notes in state -- skipped, no LLM call made")

            elif node_name == "reasoning":
                attempt = node_output.get("reasoning_attempt")
                failures = node_output.get("grounding_failures")
                recs = node_output.get("ranked_recommendations", [])
                print(f"  attempt {attempt}: LLM ranked {len(recs)} item(s), "
                      f"{failures} rationale(s) failed the grounding check")
                for r in recs:
                    tag = "REJECTED->fact-only" if r["rationale"].startswith("(") else "kept"
                    print(f"    - {r['item_id']} [{r['urgency']}] ({tag}): \"{r['rationale']}\"")

            elif node_name == "shrink_batch":
                print(f"  router sent it back: too many grounding failures, "
                      f"batch size now {node_output.get('chunk_size')}")

            elif node_name == "human_approval":
                print("  graph paused here, awaiting seller approval "
                      "(this is where the dashboard would show the ranked list)")

            else:
                print(f"  {node_output}")

    print("\n(graph run complete -- this is the point run_graph_check.py "
          "would print only the final list)")


# ==================================================================== PART 2
# The chatbot's tool-calling loop, traced turn by turn. Mirrors
# chatbot.answer_with_tools() exactly, with a print at each step.

def trace_chatbot(llm, question: str, consumption: dict, context: dict) -> None:
    hr(f'PART 2 -- chatbot tool loop for: "{question}"')

    item_id = resolve_item_id(question, consumption)
    if item_id is None and not supports_tool_calling(llm):
        print("  no item named and this model has no tool support -- would return 'unclear'")
        return

    if item_id is not None:
        intent = classify_intent(llm, question, consumption[item_id]["item_name"])
        print(f"  classify_intent -> kind={intent.kind}")
        if intent.kind == "command":
            print("  (command path -- no tool loop; see chatbot.respond for that branch)")
            return

    tools = make_tools(consumption, context)
    registry = {t.name: t for t in tools}
    try:
        bound = llm.bind_tools(tools)
    except Exception as e:
        print(f"  this model has no bind_tools ({e!r}) -- would fall back to answer_question()")
        return

    messages = [
        SystemMessage(content=(
            "You are an inventory assistant for a small shop owner. Answer using the "
            "tools provided and nothing else -- never invent a number, an item or a "
            "reason. Call a tool when you need a fact; when you have enough, reply in "
            "one or two short plain sentences with no jargon and no field names. If the "
            "tools do not answer the question, say so plainly. Before suggesting anyone "
            "orders more of something, check whether it is already on order."
        )),
        HumanMessage(content=question),
    ]
    print(f"  USER -> \"{question}\"")

    for step in range(1, 5):  # mirrors chatbot.MAX_TOOL_STEPS
        reply = bound.invoke(messages)
        messages.append(reply)
        calls = getattr(reply, "tool_calls", None) or []

        if not calls:
            text = (reply.content or "").strip()
            print(f"  MODEL (step {step}, final) -> \"{text}\"")
            return

        for call in calls:
            print(f"  MODEL (step {step}) -> calls {call['name']}({call['args']})")
            fn = registry.get(call["name"])
            output = (fn.invoke(call["args"]) if fn
                      else f"No tool named {call['name']}. Available: {', '.join(registry)}.")
            print(f"  TOOL  {call['name']} -> \"{output}\"")
            messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))

    print("  (ran out of steps -- caller would fall back to answer_question())")


if __name__ == "__main__":
    llm = _get_llm()

    trace_graph(llm)

    WATCH_ITEMS = ["I016", "I020"]  # Rakhi Set (festival), Umbrella (trend)
    as_of_ts = pd.Timestamp(AS_OF)
    consumption = {iid: assess_item(iid, as_of_ts, sales, items, suppliers,
                                     on_order=on_order_qty(iid, as_of_ts, purchase_orders),
                                     arriving_in_days=days_to_next_arrival(iid, as_of_ts, purchase_orders))
                   for iid in WATCH_ITEMS}
    context = {iid: get_context(iid, as_of_ts, items, festival_calendar, festival_overrides, promotions)
               for iid in WATCH_ITEMS}

    for q in ["what needs attention today?", "why is umbrella low on stock?"]:
        trace_chatbot(llm, q, consumption, context)

    print()
