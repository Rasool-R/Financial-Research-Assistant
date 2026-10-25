"""Machine-readable forecasts, a deterministic rule-based bias, and the log.

The trend analyst ends its answer with one line such as::

    FORECAST: bias=bullish; confidence=medium

That line is parsed here so every run produces a forecast that can later be
compared with what the market actually did (see ``backtest.py``).
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from financial_assistant.clock import utc_now

BIASES = ("bullish", "bearish", "neutral")
CONFIDENCES = ("low", "medium", "high")

# How many candles ahead a forecast is judged on (about one trading day).
DEFAULT_HORIZON_CANDLES = {"1h": 24, "4h": 6, "1day": 5}

FORECAST_LINE = re.compile(
    r"^[ \t>*_`-]*FORECAST\s*:\s*bias\s*=\s*(bullish|bearish|neutral)"
    r"\s*[;,]\s*confidence\s*=\s*(low|medium|high)[ \t*_`.]*$",
    re.IGNORECASE | re.MULTILINE,
)


def parse_forecast(text: str) -> tuple[dict[str, str] | None, str]:
    """Extract the forecast line from an analysis.

    Returns ``(forecast, text_without_the_line)``. If several lines match,
    the last one wins. ``forecast`` is None when no valid line is present.
    """

    matches = list(FORECAST_LINE.finditer(text))

    if not matches:
        return None, text

    last = matches[-1]
    forecast = {
        "bias": last.group(1).lower(),
        "confidence": last.group(2).lower(),
    }

    cleaned = FORECAST_LINE.sub("", text).rstrip()

    return forecast, cleaned


def horizon_for(interval: str) -> int:
    return DEFAULT_HORIZON_CANDLES[interval]


def unknown_forecast(interval: str) -> dict[str, Any]:
    return {
        "bias": "unknown",
        "confidence": "unknown",
        "horizon_candles": horizon_for(interval),
    }


def rule_based_bias(summary: dict[str, Any]) -> dict[str, str]:
    """Deterministic bias from the summary, used as a baseline.

    A direction is called only when the SMA trend and the MACD momentum
    agree; otherwise the call is neutral. Confidence is high when the
    higher timeframes point the same way, medium otherwise.
    """

    conditions = summary["market_conditions"]
    trend = conditions["trend"]
    momentum = conditions["macd_momentum"]

    if trend == momentum and trend in {"bullish", "bearish"}:
        alignment = (summary.get("multi_timeframe") or {}).get("alignment", "")
        confidence = "high" if alignment == f"aligned_{trend}" else "medium"
        return {"bias": trend, "confidence": confidence}

    return {"bias": "neutral", "confidence": "low"}


def build_forecast_record(
    result: dict[str, Any],
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """One log line for a finished run, or None if there is nothing to score."""

    forecast = result.get("forecast") or {}
    summary = result.get("market_summary")

    if forecast.get("bias") not in BIASES or not summary:
        return None

    now = now or utc_now()
    rule = rule_based_bias(summary)
    latest = summary["latest_candle"]

    return {
        "created_at": now.isoformat(),
        "symbol": summary["symbol"],
        "interval": summary["interval"],
        "bias": forecast["bias"],
        "confidence": forecast["confidence"],
        "rule_bias": rule["bias"],
        "rule_confidence": rule["confidence"],
        "trend": summary["market_conditions"]["trend"],
        "alignment": (summary.get("multi_timeframe") or {}).get("alignment"),
        "reference_candle_start": latest["datetime"],
        "reference_price": latest["close"],
        "atr_percent": summary["volatility"]["atr_percent"],
        "is_stale": summary["data_quality"]["is_stale"],
        "review_verdict": (result.get("review") or {}).get("verdict"),
        "revisions": result.get("revision_count", 0),
    }


def append_record(path: str | Path, record: dict[str, Any]) -> None:
    """Append one record to the JSON Lines log (the file is never rewritten)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_records(path: str | Path) -> tuple[list[dict[str, Any]], int]:
    """Read the log. Returns ``(records, number_of_unreadable_lines)``."""

    records = []
    skipped = 0

    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            required = {"symbol", "interval", "bias", "reference_candle_start"}
            if isinstance(record, dict) and required <= record.keys():
                records.append(record)
            else:
                skipped += 1

    return records, skipped
