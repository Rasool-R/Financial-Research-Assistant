import argparse
import json
import sys
from pathlib import Path

from financial_assistant.tools.market_data import (
    SUPPORTED_INTERVALS,
    SUPPORTED_SYMBOLS,
)

DEFAULT_SYMBOL = "XAU/USD"
DEFAULT_INTERVAL = "1h"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="financial-assistant",
        description=(
            "Generate a financial research report with the multi-agent "
            "workflow (market data, news research, report, review)."
        ),
    )

    parser.add_argument(
        "--symbol",
        default=DEFAULT_SYMBOL,
        choices=sorted(SUPPORTED_SYMBOLS),
        help=f"Instrument to analyze (default: {DEFAULT_SYMBOL}).",
    )
    parser.add_argument(
        "--interval",
        default=DEFAULT_INTERVAL,
        choices=sorted(SUPPORTED_INTERVALS),
        help=f"Candle interval (default: {DEFAULT_INTERVAL}).",
    )
    parser.add_argument(
        "--query",
        help="Research question. Defaults to a general technical + news view.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write the report to this file (Markdown).",
    )
    parser.add_argument(
        "--show-review",
        action="store_true",
        help="Also print the reviewer's verdict and any non-fatal errors.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print the full workflow result as JSON instead of the report.",
    )

    return parser


def default_query(symbol: str) -> str:
    return (
        f"Analyze the technical conditions of {symbol} "
        "and investigate relevant recent market news."
    )


def run_workflow(symbol: str, interval: str, query: str) -> dict:
    """Run the LangGraph workflow and return its final state."""

    from financial_assistant.graph import financial_graph

    return financial_graph.invoke(
        {
            "user_query": query,
            "symbol": symbol,
            "interval": interval,
        }
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    query = args.query or default_query(args.symbol)

    try:
        result = run_workflow(args.symbol, args.interval, query)
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    report = result.get("final_report", "")
    review = result.get("review") or {}
    errors = result.get("errors") or []

    if args.as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        print(report)

    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"\nReport saved to {args.output}", file=sys.stderr)

    if args.show_review and not args.as_json:
        print("\n--- Review ---")
        print(result.get("review_feedback", "No review available."))
        print(f"\nRevisions: {result.get('revision_count', 0)}")

        if errors:
            print("\n--- Non-fatal errors ---")
            for error in errors:
                print(f"- {error}")

    if review.get("verdict") == "NEEDS_REVISION":
        print(
            "Warning: the reviewer still reported critical issues after the "
            "maximum number of revisions. Treat the report with caution.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
