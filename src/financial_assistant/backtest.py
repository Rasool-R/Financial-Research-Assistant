"""Did the forecasts beat chance?

Two ways to get an answer:

* ``resolve_forecasts`` scores the forecasts the assistant really made (read
  from the log) against what the market did afterwards. This is the honest,
  forward-looking test, but it fills up slowly.
* ``replay`` walks through past candles and scores the *rule-based* bias on
  each of them, using only the candles available at that moment (no
  look-ahead). It needs no LLM and no news, so it gives many samples
  immediately, but it says nothing about the LLM or the news research.

A forecast is judged by the sign of the close-to-close return over
``horizon`` candles. Everything here is descriptive statistics, not advice.
"""

import math
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from financial_assistant.clock import utc_now
from financial_assistant.forecast import (
    CONFIDENCES,
    horizon_for,
    rule_based_bias,
)
from financial_assistant.tools.market_data import INTERVAL_DELTAS
from financial_assistant.tools.market_summary import (
    MIN_CANDLES,
    build_market_summary,
)
from financial_assistant.tools.technical_indicators import (
    add_technical_indicators,
)

# Fewer independent directional forecasts than this: no conclusion at all.
MIN_SAMPLE = 30

# A neutral call counts as right if the move stayed inside this fraction of
# the expected move (ATR% scaled by the square root of the horizon).
NEUTRAL_BAND_FRACTION = 0.5

MAX_FETCH_CANDLES = 5000
Z_95 = 1.96

DIRECTION = {"bullish": 1, "bearish": -1}


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def wilson_interval(hits: int, n: int, z: float = Z_95):
    """Wilson score interval for a proportion (95% by default)."""

    if n == 0:
        return None, None

    p = hits / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = (
        z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    )

    return max(0.0, centre - half), min(1.0, centre + half)


def binomial_two_sided_p(hits: int, n: int, p: float = 0.5) -> float | None:
    """Exact two-sided binomial test (sum of outcomes no likelier than hits)."""

    if n == 0:
        return None

    def log_pmf(i: int) -> float:
        return (
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            + i * math.log(p)
            + (n - i) * math.log(1 - p)
        )

    reference = log_pmf(hits)
    total = sum(
        math.exp(log_pmf(i))
        for i in range(n + 1)
        if log_pmf(i) <= reference + 1e-9
    )

    return min(1.0, total)


def forward_return_percent(
    closes: np.ndarray,
    index: int,
    horizon: int,
) -> float | None:
    """Return in percent from candle ``index`` to ``horizon`` candles later."""

    if index < 0 or index + horizon >= len(closes):
        return None

    base = float(closes[index])

    if base == 0:
        return None

    return (float(closes[index + horizon]) - base) / base * 100


def _directional_stats(pairs: list[tuple[int, float]]) -> dict[str, Any]:
    """Hit rate and mean return for (direction, forward_return) pairs."""

    n = len(pairs)

    if n == 0:
        return {
            "n": 0,
            "hits": 0,
            "hit_rate": None,
            "ci95": None,
            "p_value": None,
            "mean_follow_return_percent": None,
            "mean_follow_ci95": None,
        }

    hits = sum(1 for direction, r in pairs if direction * r > 0)
    follow = np.array([direction * r for direction, r in pairs], dtype=float)
    mean = float(follow.mean())

    mean_ci = None
    if n >= 2:
        standard_error = float(follow.std(ddof=1) / math.sqrt(n))
        mean_ci = [mean - Z_95 * standard_error, mean + Z_95 * standard_error]

    low, high = wilson_interval(hits, n)

    return {
        "n": n,
        "hits": hits,
        "hit_rate": hits / n,
        "ci95": [low, high],
        "p_value": binomial_two_sided_p(hits, n),
        "mean_follow_return_percent": mean,
        "mean_follow_ci95": mean_ci,
    }


def select_non_overlapping(samples: list[dict], horizon: int) -> list[dict]:
    """A subset of forecasts that are independent of each other.

    Forecasts on the same series closer than ``horizon`` candles judge
    overlapping stretches of price, so they are not independent evidence.
    The earliest forecast is kept, then the next one at least ``horizon``
    candles later, and so on (per symbol and interval).
    """

    groups: dict[tuple, list[dict]] = defaultdict(list)
    selected = []

    for sample in samples:
        if sample.get("reference_index") is None:
            selected.append(sample)
        else:
            groups[(sample.get("symbol"), sample.get("interval"))].append(sample)

    for group in groups.values():
        last = None

        for sample in sorted(group, key=lambda item: item["reference_index"]):
            index = sample["reference_index"]

            if last is None or index - last >= horizon:
                selected.append(sample)
                last = index

    return selected


def count_non_overlapping(samples: list[dict], horizon: int) -> int:
    """How many forecasts are independent of each other."""

    return len(select_non_overlapping(samples, horizon))


def _conclusion(stats: dict[str, Any]) -> str:
    """Verdict from the stats of INDEPENDENT forecasts only."""

    if stats["n"] < MIN_SAMPLE:
        return "insufficient_data"

    low, high = stats["ci95"]
    p_value = stats["p_value"]

    if p_value < 0.05 and low > 0.5:
        return "better_than_chance"

    if p_value < 0.05 and high < 0.5:
        return "worse_than_chance"

    return "not_distinguishable_from_chance"


def evaluate_samples(
    samples: list[dict[str, Any]],
    horizon: int,
    primary_label: str = "system",
) -> dict[str, Any]:
    """Score samples against what happened.

    Each sample needs ``bias`` and ``forward_return_percent``; ``confidence``,
    ``rule_bias``, ``trend``, ``atr_percent``, ``symbol``, ``interval`` and
    ``reference_index`` are used when present.
    """

    usable = [s for s in samples if s.get("forward_return_percent") is not None]

    def pairs(key: str, subset: list[dict] | None = None):
        rows = usable if subset is None else subset
        return [
            (DIRECTION[s[key]], s["forward_return_percent"])
            for s in rows
            if s.get(key) in DIRECTION
        ]

    directional = [s for s in usable if s.get("bias") in DIRECTION]

    methods = {primary_label: _directional_stats(pairs("bias"))}

    if primary_label != "rule_based" and any(s.get("rule_bias") for s in usable):
        methods["rule_based"] = _directional_stats(pairs("rule_bias"))

    if any(s.get("trend") for s in usable):
        methods["trend_only"] = _directional_stats(pairs("trend"))

    # Market drift: always calling "up" on the same dates as the primary calls.
    methods["always_bullish"] = _directional_stats(
        [(1, s["forward_return_percent"]) for s in directional]
    )

    neutral = [s for s in usable if s.get("bias") == "neutral"]
    scored = correct = 0

    for sample in neutral:
        atr_percent = sample.get("atr_percent")

        if atr_percent is None:
            continue

        scored += 1
        band = NEUTRAL_BAND_FRACTION * atr_percent * math.sqrt(horizon)

        if abs(sample["forward_return_percent"]) <= band:
            correct += 1

    by_confidence = {}
    for confidence in CONFIDENCES:
        subset = [s for s in directional if s.get("confidence") == confidence]
        stats = _directional_stats(pairs("bias", subset))
        by_confidence[confidence] = {
            "n": stats["n"],
            "hits": stats["hits"],
            "hit_rate": stats["hit_rate"],
        }

    primary = methods[primary_label]

    independent_samples = select_non_overlapping(directional, horizon)
    independent_stats = _directional_stats(pairs("bias", independent_samples))
    independent = independent_stats["n"]

    warnings = []

    if primary["n"] < MIN_SAMPLE:
        warnings.append(
            f"Only {primary['n']} directional forecasts; at least "
            f"{MIN_SAMPLE} independent ones are needed for any conclusion."
        )
    elif independent < MIN_SAMPLE:
        warnings.append(
            f"The forecasts overlap: only about {independent} of "
            f"{primary['n']} are independent (horizon {horizon} candles)."
        )

    if primary["n"] >= MIN_SAMPLE and independent < primary["n"]:
        warnings.append(
            "Overlapping forecasts are not independent evidence, so the "
            "table's confidence intervals and p-values are too optimistic. "
            "The conclusion uses only the independent forecasts."
        )

    baseline = methods["always_bullish"]
    if (
        primary["n"]
        and baseline["hit_rate"] is not None
        and primary["hit_rate"] <= baseline["hit_rate"]
    ):
        warnings.append(
            "The forecasts did not beat simply predicting 'up' on the same "
            "dates."
        )

    return {
        "horizon_candles": horizon,
        "primary": primary_label,
        "samples": len(usable),
        "directional_calls": primary["n"],
        "independent_directional_calls": independent,
        "independent": {
            "n": independent_stats["n"],
            "hits": independent_stats["hits"],
            "hit_rate": independent_stats["hit_rate"],
            "ci95": independent_stats["ci95"],
            "p_value": independent_stats["p_value"],
        },
        "methods": methods,
        "neutral": {
            "calls": len(neutral),
            "scored": scored,
            "correct": correct,
            "accuracy": correct / scored if scored else None,
        },
        "by_confidence": by_confidence,
        "conclusion": _conclusion(independent_stats),
        "warnings": warnings,
    }


# --------------------------------------------------------------------------
# Live forecasts (from the log)
# --------------------------------------------------------------------------


def _timestamp(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)

    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")

    return stamp.tz_convert("UTC")


def _horizon_of(record: dict[str, Any], override: int | None) -> int:
    """Horizon of a logged forecast: the override, else the one it was made
    with, else the default for its interval (older log lines have none)."""

    if override:
        return override

    stored = record.get("horizon_candles")

    if isinstance(stored, int) and stored >= 1:
        return stored

    return horizon_for(record["interval"])


def _report_label(interval: str, horizon: int, override: int | None) -> str:
    """Forecasts with different horizons are never mixed into one statistic."""

    if override or horizon == horizon_for(interval):
        return interval

    return f"{interval} (+{horizon} candles)"


def resolve_forecasts(
    records: list[dict[str, Any]],
    fetch: Callable[[str, str, int], pd.DataFrame],
    horizon: int | None = None,
    include_stale: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Score logged forecasts whose horizon has already passed.

    ``fetch(symbol, interval, outputsize)`` returns closed candles. Results
    are grouped by interval because each interval has its own horizon.
    """

    now = now or utc_now()

    seen = set()
    unique = []
    duplicates = 0

    for record in records:
        # The horizon is part of the forecast: "in 2 hours" and "in 6 hours"
        # asked on the same candle are different forecasts.
        key = (
            record["symbol"],
            record["interval"],
            record["reference_candle_start"],
            _horizon_of(record, horizon),
        )

        if key in seen:
            duplicates += 1
            continue

        seen.add(key)
        unique.append(record)

    stale_excluded = 0
    no_call = 0
    usable = []

    for record in unique:
        if record["bias"] not in ("bullish", "bearish", "neutral"):
            no_call += 1
        elif record.get("is_stale") and not include_stale:
            stale_excluded += 1
        else:
            usable.append(record)

    series: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in usable:
        series[(record["symbol"], record["interval"])].append(record)

    samples: dict[str, list[dict]] = defaultdict(list)
    pending: Counter = Counter()
    unmatched: Counter = Counter()
    horizons: dict[str, int] = {}

    for (symbol, interval), group in series.items():
        candle_horizons = [_horizon_of(r, horizon) for r in group]

        earliest = min(_timestamp(r["reference_candle_start"]) for r in group)
        age_in_candles = int((now - earliest) / INTERVAL_DELTAS[interval])
        outputsize = max(
            100,
            min(MAX_FETCH_CANDLES, age_in_candles + max(candle_horizons) + 10),
        )

        candles = fetch(symbol, interval, outputsize).reset_index(drop=True)
        closes = candles["close"].to_numpy(dtype=float)

        for record, horizon_candles in zip(group, candle_horizons, strict=True):
            label = _report_label(interval, horizon_candles, horizon)
            horizons[label] = horizon_candles

            reference = _timestamp(record["reference_candle_start"])
            matches = np.flatnonzero((candles["datetime"] == reference).to_numpy())

            if len(matches) == 0:
                unmatched[label] += 1
                continue

            index = int(matches[0])
            change = forward_return_percent(closes, index, horizon_candles)

            if change is None:
                pending[label] += 1
                continue

            samples[label].append(
                {
                    "bias": record["bias"],
                    "confidence": record.get("confidence"),
                    "rule_bias": record.get("rule_bias"),
                    "trend": record.get("trend"),
                    "atr_percent": record.get("atr_percent"),
                    "forward_return_percent": change,
                    "reference_index": index,
                    "symbol": symbol,
                    "interval": interval,
                    "reference_time": record["reference_candle_start"],
                }
            )

    intervals = {}
    for label, horizon_candles in horizons.items():
        report = evaluate_samples(samples[label], horizon_candles)
        report["pending"] = pending[label]
        report["unmatched"] = unmatched[label]
        intervals[label] = report

    return {
        "mode": "live",
        "intervals": intervals,
        "meta": {
            "records": len(records),
            "duplicates_ignored": duplicates,
            "stale_excluded": stale_excluded,
            "without_directional_forecast": no_call,
        },
    }


# --------------------------------------------------------------------------
# Historical replay of the rule-based bias
# --------------------------------------------------------------------------


def replay_samples(
    candles: pd.DataFrame,
    symbol: str,
    interval: str,
    horizon: int,
    step: int,
) -> list[dict[str, Any]]:
    """Rule-based bias at past points, using only the data known at the time.

    Indicators are causal (each value depends only on earlier candles), so
    they are computed once. The summary at point ``i`` is then built from the
    prefix ``candles[: i + 1]``, and the outcome is read from the candle
    ``horizon`` steps later.
    """

    candles = add_technical_indicators(candles.reset_index(drop=True))
    closes = candles["close"].to_numpy(dtype=float)
    delta = INTERVAL_DELTAS[interval]

    samples = []

    for index in range(MIN_CANDLES - 1, len(candles) - horizon, step):
        window = candles.iloc[: index + 1]
        candle_end = window["datetime"].iloc[-1] + delta

        summary = build_market_summary(window, symbol, interval, now=candle_end)
        rule = rule_based_bias(summary)

        samples.append(
            {
                "bias": rule["bias"],
                "confidence": rule["confidence"],
                "trend": summary["market_conditions"]["trend"],
                "atr_percent": summary["volatility"]["atr_percent"],
                "forward_return_percent": forward_return_percent(
                    closes, index, horizon
                ),
                "reference_index": index,
                "symbol": symbol,
                "interval": interval,
                "reference_time": window["datetime"].iloc[-1].isoformat(),
            }
        )

    return samples


def replay(
    candles: pd.DataFrame,
    symbol: str,
    interval: str,
    horizon: int | None = None,
    step: int | None = None,
) -> dict[str, Any]:
    """Walk-forward test of the rule-based bias on historical candles.

    ``step`` defaults to the horizon so the samples do not overlap.
    """

    horizon = horizon or horizon_for(interval)
    step = step or horizon

    samples = replay_samples(candles, symbol, interval, horizon, step)
    report = evaluate_samples(samples, horizon, primary_label="rule_based")

    report.update(
        {
            "mode": "replay",
            "symbol": symbol,
            "interval": interval,
            "step_candles": step,
            "candles_used": len(candles),
            "first_reference": samples[0]["reference_time"] if samples else None,
            "last_reference": samples[-1]["reference_time"] if samples else None,
            "scope_note": (
                "Rule-based bias only (SMA trend + MACD agreement). No LLM, "
                "no news and no higher-timeframe confirmation are replayed."
            ),
        }
    )

    return report


# --------------------------------------------------------------------------
# Text report
# --------------------------------------------------------------------------

CONCLUSIONS = {
    "insufficient_data": "INSUFFICIENT DATA - no conclusion can be drawn yet.",
    "better_than_chance": (
        "Hit rate is significantly above 50% (still check the baselines)."
    ),
    "worse_than_chance": "Hit rate is significantly BELOW 50%.",
    "not_distinguishable_from_chance": (
        "Hit rate cannot be told apart from a coin flip."
    ),
}


def _percent(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value * 100:.{digits}f}%"


def _signed(value: float | None) -> str:
    return "-" if value is None else f"{value:+.3f}%"


def format_report(report: dict[str, Any], title: str) -> str:
    """Readable summary of an ``evaluate_samples`` report."""

    lines = [
        f"== {title} ==",
        (
            f"Horizon: next {report['horizon_candles']} candles | "
            f"scored samples: {report['samples']} | "
            f"directional calls: {report['directional_calls']} "
            f"(independent: {report['independent_directional_calls']})"
        ),
    ]

    for key in ("pending", "unmatched"):
        if report.get(key):
            lines.append(f"{key.capitalize()} forecasts not scored: {report[key]}")

    lines += [
        "",
        f"{'method':16s} {'calls':>5s} {'hit rate':>9s} {'95% CI':>15s} "
        f"{'p (vs 50%)':>11s} {'mean follow-return':>19s}",
    ]

    for name, stats in report["methods"].items():
        interval = (
            "-"
            if stats["ci95"] is None
            else f"[{stats['ci95'][0] * 100:.0f}, {stats['ci95'][1] * 100:.0f}]%"
        )
        p_value = "-" if stats["p_value"] is None else f"{stats['p_value']:.3f}"
        lines.append(
            f"{name:16s} {stats['n']:5d} {_percent(stats['hit_rate']):>9s} "
            f"{interval:>15s} {p_value:>11s} "
            f"{_signed(stats['mean_follow_return_percent']):>19s}"
        )

    confidence_parts = [
        f"{name}: {_percent(item['hit_rate'], 0)} of {item['n']}"
        for name, item in report["by_confidence"].items()
        if item["n"]
    ]
    if confidence_parts:
        lines += ["", "By confidence: " + "; ".join(confidence_parts)]

    neutral = report["neutral"]
    if neutral["calls"]:
        lines.append(
            f"Neutral calls: {neutral['calls']} "
            f"({neutral['correct']} of {neutral['scored']} stayed in range)"
        )

    independent = report["independent"]
    if independent["n"] < report["directional_calls"] and independent["n"]:
        low, high = independent["ci95"]
        lines += [
            "",
            f"Independent forecasts only (n={independent['n']}): "
            f"hit rate {_percent(independent['hit_rate'])} "
            f"[{low * 100:.0f}, {high * 100:.0f}]%, "
            f"p={independent['p_value']:.3f}",
        ]

    lines += ["", f"Conclusion: {CONCLUSIONS[report['conclusion']]}"]

    for warning in report["warnings"]:
        lines.append(f"  ! {warning}")

    if report.get("scope_note"):
        lines.append(f"  Note: {report['scope_note']}")

    return "\n".join(lines)
