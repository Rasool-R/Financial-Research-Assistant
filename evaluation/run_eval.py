"""Run the real workflow on frozen scenarios and score the reports.

Usage:
    python -m evaluation.run_eval
    python -m evaluation.run_eval --scenario gold_stale_weekend_data
    python -m evaluation.run_eval --fail-on-error

Market data and news are frozen; the LLM calls are real, so this needs
OPENAI_API_KEY / OPENAI_MODEL (and optionally OPENAI_BASE_URL) to be set.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from evaluation.evaluators import evaluate_report
from evaluation.scenarios import SCENARIOS, SCENARIOS_BY_NAME, Scenario

RESULTS_DIR = Path(__file__).parent / "results"


def run_scenario(scenario: Scenario, graph=None) -> dict:
    """Run one scenario and return its evaluation record."""

    if graph is None:
        from financial_assistant.graph import financial_graph as graph

    def frozen_market_data(**kwargs):
        return scenario.make_candles(kwargs["interval"])

    def frozen_search(**kwargs):
        return [dict(item) for item in scenario.news]

    with (
        patch(
            "financial_assistant.tools.market_summary.get_market_data",
            frozen_market_data,
        ),
        patch(
            "financial_assistant.nodes.research_agent.search_web",
            frozen_search,
        ),
    ):
        state = graph.invoke(
            {
                "user_query": scenario.query,
                "symbol": scenario.symbol,
                "interval": scenario.interval,
            }
        )

    results = evaluate_report(state)
    summary = state.get("market_summary") or {}
    actual_trend = (summary.get("market_conditions") or {}).get("trend")
    actual_alignment = (summary.get("multi_timeframe") or {}).get("alignment")

    return {
        "scenario": scenario.name,
        "description": scenario.description,
        "passed": all(result.passed for result in results),
        "evaluators": [result.to_dict() for result in results],
        "reviewer_verdict": (state.get("review") or {}).get("verdict"),
        "revisions": state.get("revision_count", 0),
        "errors": state.get("errors", []),
        "trend_matches_fixture": (
            scenario.expected_trend is None
            or actual_trend == scenario.expected_trend
        ),
        "alignment_matches_fixture": (
            scenario.expected_alignment is None
            or actual_alignment == scenario.expected_alignment
        ),
        "report": state.get("final_report", ""),
    }


def print_table(records: list[dict]) -> None:
    names = [e["name"] for e in records[0]["evaluators"]] if records else []

    print(f"\n{'scenario':32s} " + " ".join(f"{n[:12]:>12s}" for n in names))
    print("-" * (33 + 13 * len(names)))

    for record in records:
        cells = []
        for evaluator in record["evaluators"]:
            mark = "PASS" if evaluator["passed"] else "FAIL"
            cells.append(f"{mark:>12s}")
        print(f"{record['scenario']:32s} " + " ".join(cells))

    print()

    for record in records:
        for evaluator in record["evaluators"]:
            if not evaluator["passed"]:
                print(
                    f"[{record['scenario']}] {evaluator['name']}: "
                    f"{evaluator['details']}"
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS_BY_NAME),
        help="Run a single scenario (default: all).",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit with status 1 if any evaluator fails.",
    )
    args = parser.parse_args(argv)

    scenarios = (
        [SCENARIOS_BY_NAME[args.scenario]] if args.scenario else SCENARIOS
    )

    records = []
    for scenario in scenarios:
        print(f"Running {scenario.name} ...", file=sys.stderr)
        records.append(run_scenario(scenario))

    print_table(records)

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = RESULTS_DIR / f"eval_{stamp}.json"
    output.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Detailed results saved to {output}")

    failed = any(not record["passed"] for record in records)
    return 1 if (failed and args.fail_on_error) else 0


if __name__ == "__main__":
    sys.exit(main())
