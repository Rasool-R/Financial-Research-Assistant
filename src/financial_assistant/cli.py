import argparse
import json
import sys
from pathlib import Path

from financial_assistant.config import FORECAST_LOG_PATH
from financial_assistant.forecast import append_record, build_forecast_record
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
        "--no-log",
        action="store_true",
        help="Do not append this run's forecast to the backtest log.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path(FORECAST_LOG_PATH),
        help=f"Forecast log used by financial-backtest (default: "
        f"{FORECAST_LOG_PATH}).",
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


def run_workflow(
    symbol: str,
    interval: str,
    query: str,
    horizon_candles: int | None = None,
) -> dict:
    """Run the LangGraph workflow and return its final state."""

    from financial_assistant.graph import financial_graph

    state = {
        "user_query": query,
        "symbol": symbol,
        "interval": interval,
    }

    if horizon_candles:
        state["horizon_candles"] = horizon_candles

    return financial_graph.invoke(state)


def log_forecast(result: dict, args) -> None:
    """Append the run's forecast to the log and tell the user what happened."""

    forecast = result.get("forecast") or {}
    record = build_forecast_record(result)

    if record is None:
        if forecast.get("bias") == "unknown":
            print(
                "Note: this run made no directional forecast, so nothing "
                "was logged for backtesting.",
                file=sys.stderr,
            )
        return

    summary = (
        f"Forecast: {record['bias']} ({record['confidence']} confidence) "
        f"for the next {forecast.get('horizon_candles')} "
        f"{record['interval']} candles."
    )

    if args.no_log:
        print(f"{summary} Not logged (--no-log).", file=sys.stderr)
        return

    append_record(args.log_path, record)

    note = " Data was stale; it is excluded from scoring by default." if (
        record["is_stale"]
    ) else ""
    print(f"{summary} Logged to {args.log_path}.{note}", file=sys.stderr)


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

    log_forecast(result, args)

    if review.get("verdict") == "NEEDS_REVISION":
        print(
            "Warning: the reviewer still reported critical issues after the "
            "maximum number of revisions. Treat the report with caution.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
