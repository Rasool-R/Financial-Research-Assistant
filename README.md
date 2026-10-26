# Financial Research Assistant

A multi-agent workflow (built with [LangGraph](https://langchain-ai.github.io/langgraph/))
that produces a short, evidence-based research report for forex and metal
instruments. Technical indicators are computed in code; the LLMs only
interpret the numbers, summarize news, write the report and review it.

> **Disclaimer:** this project is for research and education. Its output is
> not financial advice and makes no guarantee about future prices.

## How it works

```mermaid
flowchart LR
    S([start]) --> M[market_agent]
    S --> R[research_agent]
    M --> A[analysis_agent]
    A --> W[report_agent]
    R --> W
    W --> V[reviewer_agent]
    V -- "NEEDS_REVISION (up to MAX_REVISIONS)" --> W
    V -- "PASS / UNREVIEWED" --> E([end])
```

| Node | What it does |
| --- | --- |
| `market_agent` | Trend and momentum analyst. Fetches closed OHLC candles (Twelve Data) for the requested timeframe **and the higher ones** (`1h` also loads `4h` and `1day`), computes SMA/EMA/RSI/MACD/ATR/Bollinger Bands and swing support/resistance in pandas, builds a market summary (including a data-freshness check and a timeframe-alignment verdict) and asks the LLM for a directional bias with a confidence level. The answer ends with a machine-readable `FORECAST:` line that is parsed and logged for backtesting. |
| `analysis_agent` | Risk and levels analyst. Works from the same summary but covers a different job: volatility (ATR, Bollinger), nearest support/resistance in ATR units, bull and bear conditions, and invalidation levels. It does not see the trend analysis, so the two views stay independent. |
| `research_agent` | Runs several short keyword queries per instrument (plus your own question) in parallel, keeps only recent articles, ranks and deduplicates them, and extracts validated findings with `relevance` (high/medium/low) and `sentiment` (bullish/bearish/neutral). URLs, titles and dates are copied from the real search results, never trusted from the LLM. |
| `report_agent` | Writes the final report from the evidence. On a revision it receives its previous draft plus the reviewer's feedback. |
| `reviewer_agent` | Returns a structured verdict (`PASS` / `NEEDS_REVISION`) with critical issues and recommendations. Checks done in code (currently the word limit) are added to the LLM's findings and force a revision even if the LLM says `PASS`. |

Design notes:

- `market_agent` and `research_agent` run **in parallel**; `report_agent`
  waits for both.
- A failure in market data, analysis or the report stops the workflow (after
  retrying transient errors such as timeouts, 429 and 5xx). A failure in
  research or review **degrades gracefully**: the problem is recorded in
  `errors`, passed to the report as a limitation, and the workflow continues.
- If data is stale (for example over a weekend), the report is told to say the
  prices may not be live.
- The requested timeframe must load, but a higher timeframe that fails is only
  recorded (`multi_timeframe.unavailable`, plus a warning passed to the
  report). The report then says higher-timeframe confirmation is incomplete.
- News search is limited to the last `NEWS_LOOKBACK_DAYS` days, both in the
  Tavily request and again in code. If some of the queries fail, the others
  are still used and the gap is reported as a limitation.
- Web pages are untrusted input. Search results are passed to the LLM as JSON
  inside `<search_results>` tags (a page cannot close the block), the agents
  are told to treat them as data, and the LLM can never change URLs, titles or
  dates. This lowers the risk of prompt injection but cannot remove it, so the
  evaluation includes a scenario with a planted injection.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # installs the package + dev tools
cp .env.example .env             # then fill in your keys
```

| Variable | Required | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | yes | API key for the LLM provider. |
| `OPENAI_MODEL` | yes | Model name. |
| `OPENAI_BASE_URL` | no | Custom OpenAI-compatible endpoint. |
| `TWELVE_DATA_API_KEY` | yes | Market data. |
| `TAVILY_API_KEY` | yes | News search. |
| `MAX_REVISIONS` | no | Max reviewer-requested revisions (default `2`). |
| `NEWS_LOOKBACK_DAYS` | no | Only use news from the last N days (default `7`). |
| `FORECAST_LOG_PATH` | no | Where forecasts are logged for backtesting (default `data/forecasts.jsonl`). |
| `LANGSMITH_*` | no | Optional tracing. |

Check your LLM connection with `python scripts/check_model.py`.

## Quick start: ask a question

```bash
python main.py                         # interactive: type your question
python main.py "قیمت دو ساعت آینده طلا چطور خواهد بود؟"
python main.py "will gold go up in the next 2 hours?"
```

The whole workflow runs (market data, indicators, news, report, review), but
you only see a short answer: a header, one or two sentences about what drives
the picture, the forecast direction and a one-line disclaimer.

```text
طلا (XAU/USD) · 2 ساعت آینده
روند کوتاه‌مدت صعودی و RSI خنثی است؛ مقاومت نزدیک حدود یک ATR بالاتر است.
پیش‌بینی: صعودی ↑ (اطمینان: متوسط)
این توصیه‌ی مالی نیست.
```

The direction line is written by code from the model's machine-readable
forecast, never by the free-text sentence, so the two cannot contradict each
other. Add `--full` (or type `details` in interactive mode) for the complete
English report, `--lang en` / `--lang fa` to force the answer language and
`--no-log` to keep the forecast out of the backtest log.

What it understands (Persian or English):

| You write | Understood as |
| --- | --- |
| `طلا ۲ ساعت آینده`, `gold in 2 hours` | gold, forecast 2 hourly candles ahead |
| `یورو فردا`, `euro tomorrow` | EUR/USD, 24 hours ahead (6 candles of 4h) |
| `gold in 3 days` | 3 daily candles ahead |
| `طلا ۴ ساعته`, `gold 4h` | gold on the 4h timeframe, default horizon |
| `gold 2 hours ahead on the 4h chart` | both: 4h timeframe, 1 candle ahead |

"In 4h" is a horizon (four hours ahead) while "4h" or "۴ ساعته" is a timeframe
(candle size). If the symbol or the period is missing, the assistant asks;
anything shorter than one hour is rounded up to one hourly candle, and the
answer tells you when the period was rounded. The longest horizon is 48
candles. The same is available as `financial-ask`.

Things to know:

* **"طلا" / "gold" means world spot gold (XAU/USD, per ounce), not the local
  price of gold, coins or other currencies.** Only gold and EUR/USD are
  supported.
* A two-hour forecast is mostly noise. Short horizons are the hardest to call,
  so treat the direction as one input, not a signal; `financial-backtest`
  (below) measures how well the forecasts actually did.
* Persian output needs a terminal that shows right-to-left text. Classic
  Windows consoles often display it disconnected or reversed; use Windows
  Terminal, or `--lang en`.
* Every answer's forecast is appended to `data/forecasts.jsonl` (with its
  horizon) so it can be scored later. Use `--no-log` to opt out.

## Usage

The report command (explicit symbol and timeframe, full report):

```bash
financial-assistant                                   # XAU/USD, 1h
financial-assistant --symbol EUR/USD --interval 4h
financial-assistant --query "How does the dollar affect gold?" --show-review
financial-assistant -o report.md                      # also save the report
financial-assistant --json                            # full workflow state
financial-assistant --no-log                          # do not log this forecast
```

`python -m financial_assistant ...` works as well. Supported symbols and
intervals are listed by `financial-assistant --help`.

From Python:

```python
from financial_assistant.graph import financial_graph

result = financial_graph.invoke(
    {"user_query": "Analyze gold", "symbol": "XAU/USD", "interval": "1h"}
)
print(result["final_report"])
print(result["review"]["verdict"], result.get("errors"))
```

## Tests

```bash
pytest          # offline: no API keys, no network
ruff check src tests evaluation scripts
```

All tests use fake models and fake data. Scripts that call the real APIs
(and cost money / quota) live in `scripts/manual/` and are never collected by
pytest, for example `python scripts/manual/run_graph.py`.

## Evaluation

`evaluation/` runs the **real workflow with frozen inputs** (synthetic but
deterministic market data and fixed news; only the LLM calls are real) and
scores each report with deterministic checks:

| Check | Verifies |
| --- | --- |
| `required_sections` | All seven sections are present. |
| `word_limit` | Report stays within 600 words. The reviewer enforces this by counting in code and sends an over-long report back for revision. |
| `grounded_numbers` | Every price / indicator number in the report exists in the market data or news, matched at the precision it is written. Understands `4,397.92`, `4 397.92`, `4'397.92`, `4.397,92`; round numbers (`7,000`) and `about ...` are accepted as roundings. Numbers the model *derives* itself (differences, ratios) cannot be verified and are listed for you to judge. |
| `sources_from_research` | Every URL in the report comes from the research findings. |
| `no_guarantees_or_advice` | No guarantees or buy/sell instructions. |
| `stale_data_disclosed` | Stale data is not presented as live. |
| `timeframe_conflict_disclosed` | If the higher timeframes disagree with the main one, the report says so. |
| `forecast_stated` | The report states the forecast bias it was given. |
| `fresh_news` | Cited news is inside the lookback window. |
| `no_prompt_injection` | A phrase planted in a news item never appears in the report. |

```bash
python -m evaluation.run_eval                      # all scenarios
python -m evaluation.run_eval --scenario gold_stale_weekend_data
python -m evaluation.run_eval --fail-on-error      # non-zero exit if a check fails
```

This needs your LLM keys. Results are written to `evaluation/results/`
(git-ignored). Use it to compare prompt or model changes before committing
them.

## Backtesting

The question every forecasting tool must answer: **do the calls beat chance?**
Every run of `financial-assistant` appends its directional forecast (bullish /
bearish / neutral, with confidence) to a JSON Lines log. `financial-backtest`
then compares forecasts with what the market actually did.

```bash
financial-backtest resolve        # score the forecasts you have really made
financial-backtest replay         # test the rule-based bias on past candles
financial-backtest replay --symbol EUR/USD --interval 4h --candles 3000
```

**`resolve` (the honest test).** Reads the log, downloads the candles that came
after each forecast and judges it by the sign of the return over the horizon
(24 candles for `1h`, 6 for `4h`, 5 for `1day`; override with
`--horizon-candles`). It fills up slowly: you need at least 30 *independent*
directional forecasts per interval before it will say anything. Forecasts on
stale (market closed) data are excluded unless you pass `--include-stale`, and
two forecasts for the same candle count once.

**`replay` (a quick sanity check).** Walks through past candles and scores the
deterministic *rule-based* bias (SMA trend and MACD must agree) at each point,
using only the candles that existed at that moment, so there is no look-ahead.
It needs no LLM and no API credits beyond one candle download, so it gives many
samples immediately. It does **not** test the LLM, the news research or the
higher-timeframe confirmation, because past news cannot be replayed without
look-ahead.

How to read the output:

| Item | Meaning |
| --- | --- |
| hit rate, 95% CI, p | Share of directional calls that matched the sign of the return, with a Wilson interval and an exact two-sided binomial test against 50%. |
| `always_bullish` | Baseline: always predicting "up" on the same dates. A strategy that does not beat it is just riding the market's drift. |
| `trend_only`, `rule_based` | Simpler baselines scored on the same data. A forecast should beat these to justify the LLM. |
| mean follow-return | Average return (in %) of following each call, so a bearish call earns when price falls. Ignores costs. |
| By confidence | Whether "high" confidence calls hit more often than "low" ones. |
| independent | Forecasts closer than the horizon judge overlapping price moves. The conclusion is computed only from a non-overlapping subset. |

The tool never reports "it works" on its own: with fewer than 30 independent
forecasts the conclusion is `INSUFFICIENT DATA`, and even a significant result
should be read next to the baselines. The statistics are descriptive, ignore
trading costs and slippage, and are not financial advice.

## Project structure

```text
src/financial_assistant/
  app.py                  `main.py` / `financial-ask`: questions in plain language
  request_parser.py       Persian/English question -> symbol, timeframe, horizon
  cli.py, __main__.py     report command (`financial-assistant`)
  forecast.py             FORECAST line parser, rule-based bias, forecast log
  backtest.py             statistics, resolve (live) and replay (historical)
  backtest_cli.py         `financial-backtest` command
  graph.py                LangGraph workflow and revision routing
  state.py, schemas.py    shared state and Pydantic schemas
  config.py               model factory and settings
  retry.py, parsing.py    transient-error retries, parse-and-retry helper
  clock.py                UTC clock (frozen in tests)
  nodes/                  the five agents
  tools/                  market data, indicators, market structure (swing
                          levels, SMA crosses), market summary, web search
main.py                   ask a question (python main.py)
tests/                    offline unit and end-to-end tests
evaluation/               frozen scenarios, evaluators, runner
scripts/                  manual scripts that call real APIs
```

## Known limitations

- Only `XAU/USD` and `EUR/USD`, intervals `1h`, `4h`, `1day`.
- News queries are hard-coded per instrument (`tools/news_queries.py`).
- No macro data yet (dollar index, yields, economic calendar).
- Support/resistance are plain swing highs/lows (no clustering of nearby
  levels), and the thresholds (RSI 70/30, ATR ratio 1.25/0.75, Bollinger
  0.2/0.8) are fixed.
- Timeframe alignment is judged only by the SMA 20/50 trend of each timeframe.
- A forecast is judged only by the direction of the return over a fixed
  horizon. Magnitude, trading costs and risk management are not modelled.
- `replay` covers only the rule-based bias, not the LLM or the news research;
  only `resolve` tests the real system, and it needs time to collect forecasts.
- Whether the reports are directionally useful is an open question until
  `financial-backtest resolve` has enough forecasts to answer it.
