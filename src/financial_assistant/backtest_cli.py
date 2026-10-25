import argparse
import json
import sys
from pathlib import Path

from financial_assistant.backtest import format_report, replay, resolve_forecasts
from financial_assistant.config import FORECAST_LOG_PATH
from financial_assistant.forecast import load_records
from financial_assistant.tools.market_data import (
    SUPPORTED_INTERVALS,
    SUPPORTED_SYMBOLS,
    get_market_data,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="financial-backtest",
        description=(
            "Measure whether the assistant's directional forecasts beat "
            "chance. Descriptive statistics only, not financial advice."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--horizon-candles",
            type=int,
            help="Candles ahead a forecast is judged on "
            "(default: 24 for 1h, 6 for 4h, 5 for 1day).",
        )
        sub.add_argument(
            "--json",
            action="store_true",
            dest="as_json",
            help="Print the full result as JSON.",
        )
        sub.add_argument(
            "-o", "--output", type=Path, help="Also save the result as JSON."
        )

    resolve = commands.add_parser(
        "resolve",
        help="Score the forecasts logged by `financial-assistant` "
        "(the honest, forward-looking test).",
    )
    resolve.add_argument(
        "--log",
        type=Path,
        default=Path(FORECAST_LOG_PATH),
        help=f"Forecast log (default: {FORECAST_LOG_PATH}).",
    )
    resolve.add_argument(
        "--include-stale",
        action="store_true",
        help="Also score forecasts made on stale (market closed) data.",
    )
    common(resolve)

    history = commands.add_parser(
        "replay",
        help="Walk-forward test of the rule-based bias on past candles "
        "(no LLM, no news, no look-ahead).",
    )
    history.add_argument(
        "--symbol", default="XAU/USD", choices=sorted(SUPPORTED_SYMBOLS)
    )
    history.add_argument(
        "--interval", default="1h", choices=sorted(SUPPORTED_INTERVALS)
    )
    history.add_argument(
        "--candles",
        type=int,
        default=2000,
        help="How many past candles to download (max 5000, default 2000).",
    )
    history.add_argument(
        "--step",
        type=int,
        help="Candles between samples (default: the horizon, so samples "
        "do not overlap).",
    )
    common(history)

    return parser


def fetch_candles(symbol: str, interval: str, outputsize: int):
    return get_market_data(
        symbol=symbol, interval=interval, outputsize=outputsize
    )


def run_resolve(args, fetch=None) -> tuple[dict, str]:
    fetch = fetch or fetch_candles

    if not args.log.exists():
        raise FileNotFoundError(
            f"No forecast log at {args.log}. Run `financial-assistant` "
            "first; every run appends its forecast there."
        )

    records, skipped = load_records(args.log)

    result = resolve_forecasts(
        records,
        fetch,
        horizon=args.horizon_candles,
        include_stale=args.include_stale,
    )
    result["meta"]["unreadable_log_lines"] = skipped

    meta = result["meta"]
    parts = [
        f"{meta['records']} logged forecasts",
        f"{meta['duplicates_ignored']} duplicates ignored",
        f"{meta['stale_excluded']} on stale data excluded",
        f"{meta['without_directional_forecast']} without a direction",
    ]
    if skipped:
        parts.append(f"{skipped} unreadable lines skipped")

    sections = [f"Log: {args.log} - " + ", ".join(parts)]

    if not result["intervals"]:
        sections.append("\nNothing to score yet.")

    for interval, report in sorted(result["intervals"].items()):
        sections.append("\n" + format_report(report, f"Live forecasts, {interval}"))

    return result, "\n".join(sections)


def run_replay(args, fetch=None) -> tuple[dict, str]:
    fetch = fetch or fetch_candles

    candles = fetch(args.symbol, args.interval, args.candles)

    report = replay(
        candles,
        args.symbol,
        args.interval,
        horizon=args.horizon_candles,
        step=args.step,
    )

    header = (
        f"Replay on {report['candles_used']} {args.interval} candles of "
        f"{args.symbol} ({report['first_reference']} -> "
        f"{report['last_reference']}), one sample every "
        f"{report['step_candles']} candles"
    )

    title = f"Replay, {args.symbol} {args.interval}"

    return report, header + "\n\n" + format_report(report, title)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if getattr(args, "candles", 1) > 5000 or getattr(args, "candles", 1) < 1:
        print("Error: --candles must be between 1 and 5000.", file=sys.stderr)
        return 2

    try:
        if args.command == "resolve":
            result, text = run_resolve(args)
        else:
            result, text = run_replay(args)
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, default=str) if args.as_json else text)

    if args.output:
        args.output.write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nSaved to {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
