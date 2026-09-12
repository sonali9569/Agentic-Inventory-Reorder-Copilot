# SellerSense

**[Live demo](https://agentic-inventory-reorder-copilot-bkuphp2kqpwiw5ni5icze9.streamlit.app/)** — starts in demo-safe mode, no setup needed.

An inventory copilot for a small retailer. It reads sales history, flags what
is about to run out or sitting dead on the shelf, ranks what needs attention
today, drafts a reorder, and adjusts its own thresholds based on seller
feedback.

**Design rule:** all arithmetic is deterministic Python. The language model
ranks items, writes explanations, and classifies intent — it never computes a
quantity or decides what counts as at-risk. Every rationale the model writes
is checked against the item's own computed facts before being shown, and text
that fails the check is replaced with a fact-only sentence generated from the
data.

## Quick start

Try the [live demo](https://agentic-inventory-reorder-copilot-bkuphp2kqpwiw5ni5icze9.streamlit.app/) —
no setup needed. It starts in **Cached (demo-safe)** mode, which replays
pre-recorded model responses so it works without any API key. Today's
flagged items and reorder recommendations appear on the **Recommendations**
tab; the **Ask** tab answers natural-language questions about any item.

To run it locally instead:

```bash
pip install -r requirements.txt
streamlit run src/dashboard.py
```

Then open `http://localhost:8501`.

## Setup

Requires Python 3.11+.

```bash
pip install -r requirements.txt
```

Prophet needs its CmdStan backend compiled once (a few minutes; requires
Xcode command line tools on macOS):

```bash
python3 -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
python3 src/prefit_forecasts.py   # cache a Prophet model per SKU
```

The dashboard and test suite run without this step; the backtest and live
forecast do not.

## Choosing a model

Install at least one LLM provider:

```bash
pip install langchain-groq          # then export GROQ_API_KEY
pip install langchain-google-genai  # then export GOOGLE_API_KEY
pip install langchain-openai        # then export OPENAI_API_KEY
```

Or run locally with [Ollama](https://ollama.com):

```bash
ollama pull qwen3:4b
```

The provider is auto-selected from what's installed and keyed. To pin one
explicitly:

```bash
export SELLERSENSE_LLM_PROVIDER=groq
export SELLERSENSE_LLM_MODEL=openai/gpt-oss-120b
```

## Components

| Module | Responsibility |
|---|---|
| `src/engine.py` | Demand rate, reorder point, safety stock, risk classification |
| `src/context_agent.py` | Festival calendar, per-item overrides, promotions |
| `src/forecast.py` / `src/forecast_cache.py` | Per-SKU Prophet forecasting, cached to disk |
| `src/graph.py` | LangGraph pipeline: gather signals, rank, check, approve |
| `src/feedback_agent.py` | Applies approve/reject feedback to per-item parameters |
| `src/chatbot.py` | Tool-calling Q&A over current state |
| `src/llm_provider.py` / `src/llm_cache.py` | Model selection and response caching |
| `src/dashboard.py` | Streamlit interface |

### How recommendations are produced

Demand rate uses a winsorized mean for dense sales series, switching to
Croston's method for intermittent (mostly-zero) series. Reorder point and
safety stock follow standard inventory formulas, sized off a Prophet forecast
when one is cached, falling back to a trailing-rate estimate otherwise.

The LangGraph pipeline has 5 nodes: `gather_signals` (deterministic filtering
and ranking), `context_extraction` (parses an optional free-text seller note
into a structured signal), `reasoning` (the model ranks and explains flagged
items), a conditional retry step that halves the batch size if too many
rationales fail a grounding check, and `human_approval`, which pauses for the
seller's decision. A second, independent LLM call critiques the first one's
output before it is shown.

The chat surface binds the model to four read-only tools
(`list_flagged_items`, `get_item_status`, `get_item_context`,
`get_open_orders`) and lets it choose which to call. Approve/reject/snooze
commands are detected from the message text but applied outside the model.

Feedback guardrails: a parameter only changes once the same rejection reason
appears in an item's last three rejections, and only for reasons that
indicate a miscalibration (not for circumstantial ones).

## The interface

| Section | What it shows |
|---|---|
| Overview | What needs attention, cost to restock, margin at risk, upcoming demand events |
| Recommendations | Ranked list with a reason for each, and approve/reject |
| Inventory | Every SKU with its risk state, cover, and reorder point |
| Ask | Natural-language questions about any flagged item |
| Evidence | Backtest: four policies over the same held-out window |
| Activity | Every decision recorded, and which parameter it moved |

## Running

```bash
streamlit run src/dashboard.py     # the interface, on :8501
pytest                             # test suite
python3 src/run_backtest.py        # four-policy comparison
python3 data/generate_dataset.py   # regenerate the dataset (seeded)
python3 src/record_demo_cache.py   # record model responses for demo mode
python3 src/prefit_forecasts.py    # cache a Prophet model per SKU
```

## Demo mode

The dashboard defaults to cached responses (`data/demo_llm_cache.json`), so
it runs without a live model. Switch to "Live model" in the sidebar to call
the configured provider directly. Re-record the cache after changing the
dataset, prompts, or provider:

```bash
python3 src/record_demo_cache.py
```

Feedback and learned parameters persist to `data/feedback_log.csv` and
`data/seller_parameters.csv`. Use "Reset feedback history" in the sidebar
between demo runs.

## Deployment

Deployed on [Streamlit Community Cloud](https://agentic-inventory-reorder-copilot-bkuphp2kqpwiw5ni5icze9.streamlit.app/).

Ollama is local-only and unreachable from a deployed container. Deploy with a
hosted provider (Groq, Gemini, or OpenAI) instead — on Streamlit Community
Cloud, put the API key in the app's secrets and set
`SELLERSENSE_LLM_PROVIDER` explicitly.

## Data

Synthetic, one store, 25 SKUs, 365 days, generated with a fixed seed:

- Fast-moving staples and FMCG lines on a growth trend
- Festival-driven demand across six events
- Monsoon seasonality on raincoats and umbrellas
- Slow movers, intermittent demand on accessories
- Three supplier profiles (reliable, slow, variable lead time)

`daily_sales.csv` includes `true_demand_uncensored` alongside observed sales,
used only to score the backtest fairly — never a model input.

## Results

45-day holdout, all 25 SKUs, fill rate weighted by demand across the store:

| Policy | Stockout days | Fill rate | Capital tied up |
|---|---|---|---|
| Seller's own rule, closed-loop | 150 | 0.828 | ₹63,672 |
| Recorded orders, replayed | 178 | 0.807 | ₹55,037 |
| Plain reorder point | 302 | 0.643 | ₹20,235 |
| Context-aware (SellerSense) | 171 | 0.802 | ₹30,837 |

The context-aware policy does not beat the seller on availability — it
reaches 97% of the seller's fill rate on 48% of the working capital. Against
a plain reorder point it wins outright: 43% fewer stockout days and 16 more
points of fill rate, for about 1.5x the capital.

Lead times are drawn from each supplier's distribution, seeded per item and
policy for reproducibility. Across ten alternative seed sets, the plain
reorder point's own numbers moved by at most 24 stockout days and 0.022 fill
rate, an order of magnitude below the gaps above.

## Known limitations

- Raksha Bandhan occurs once in this year of data and falls inside the
  holdout window, so its effect can't be learned from history alone.
- Demand estimation excludes stockout days rather than modelling them as
  censored; this leaves the demand rate biased low on the SKUs that stock
  out most (Umbrella and Raincoat sit at 0.71 and 0.81 of true demand).
- The festival uplift multipliers the context-aware policy reads are the
  same values used to generate the synthetic demand, so that portion of the
  backtest margin is not independent evidence. The monsoon seasonality
  result does not have this problem, since Prophet learns it from the data.
