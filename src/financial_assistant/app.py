"""Ask a market question in plain language and get a short answer.

    python main.py                                # interactive
    python main.py "قیمت دو ساعت آینده طلا چطور خواهد بود؟"
    python main.py "will gold go up in the next 2 hours?"

The whole workflow runs (market data, indicators, news, report, review), but
only a short answer is printed: one or two sentences, the forecast direction
and a one-line disclaimer. The full report is available with ``--full`` or by
typing ``details``.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from financial_assistant.cli import run_workflow
from financial_assistant.config import FORECAST_LOG_PATH, get_model
from financial_assistant.forecast import (
    append_record,
    build_forecast_record,
    horizon_for,
)
from financial_assistant.request_parser import (
    MAX_CANDLES,
    SYMBOL_EURO,
    SYMBOL_GOLD,
    ParsedRequest,
    normalize,
    parse_request,
)
from financial_assistant.retry import call_with_retries

MAX_ASKS = 3  # follow-up questions before giving up or using a default
MAX_BRIEF_SENTENCES = 2
MAX_BRIEF_CHARS = 200
MAX_REASON_CHARS = 120

EXIT_WORDS = {"exit", "quit", "q", "خروج"}
DETAILS_WORDS = {"details", "report", "جزئیات", "گزارش"}

LANGUAGE_NAMES = {"fa": "Persian (Farsi)", "en": "English"}

TEXT = {
    "fa": {
        "banner": (
            "دستیار تحلیل بازار. سؤالت را بنویس، مثلاً "
            "«قیمت دو ساعت آینده طلا چطور خواهد بود؟»\n"
            "نمادها: طلا (XAU/USD) و یورو/دلار (EUR/USD). "
            "گزارش کامل: «جزئیات» | خروج: «خروج»"
        ),
        "ask_symbol": "کدام نماد؟ (طلا یا یورو): ",
        "ask_horizon": (
            "برای چه بازه‌ای؟ مثلاً «۲ ساعت آینده» یا «فردا» "
            "(Enter = ۲۴ ساعت آینده): "
        ),
        "no_symbol": (
            "نماد مشخص نیست. طلا یا یورو را بنویس، "
            "مثلاً «طلا ۲ ساعت آینده»."
        ),
        "ambiguous_symbol": "هر دو نماد را نوشتی؛ فقط یکی را انتخاب کن.",
        "unsupported_timeframe": (
            "کوچک‌ترین تایم‌فریم پشتیبانی‌شده ۱ ساعته است "
            "(۱ ساعته، ۴ ساعته یا روزانه)."
        ),
        "horizon_too_long": f"حداکثر {MAX_CANDLES} کندل جلوتر قابل پیش‌بینی است.",
        "working": "در حال تحلیل {symbol}، {horizon} … (چند لحظه صبر کن)",
        "forecast": "پیش‌بینی",
        "confidence": "اطمینان",
        "bias": {
            "bullish": "صعودی ↑",
            "bearish": "نزولی ↓",
            "neutral": "خنثی →",
        },
        "conf": {"low": "کم", "medium": "متوسط", "high": "زیاد"},
        "unknown": "نامشخص (مدل جهت مشخصی نداد؛ دوباره امتحان کن)",
        "stale": "⚠ داده‌ها کهنه‌اند (بازار احتمالاً بسته است)؛ قیمت‌ها زنده نیستند.",
        "review": "⚠ بازبین هنوز ایراد جدی دیده؛ با احتیاط بخوان.",
        "reason": "  علت (به انگلیسی): {reason}",
        "more": " (+{n} مورد دیگر)",
        "errors": "⚠ بعضی منابع در دسترس نبودند؛ برای جزئیات «جزئیات» را بنویس.",
        "disclaimer": "این توصیه‌ی مالی نیست.",
        "error": "خطا: {message}",
        "no_details": "هنوز تحلیلی انجام نشده.",
        "note": "(درخواست تو {requested} بود؛ نزدیک‌ترین بازه‌ی ممکن {actual} است)",
        "details": "گزارش کامل (به انگلیسی)",
        "review_title": "بازبینی",
        "warnings_title": "هشدارها",
        "symbols": {SYMBOL_GOLD: "طلا (XAU/USD)", SYMBOL_EURO: "یورو/دلار (EUR/USD)"},
        "minute_one": "{n} دقیقه",
        "minute_other": "{n} دقیقه",
        "hour_one": "{n} ساعت",
        "hour_other": "{n} ساعت",
        "day_one": "{n} روز",
        "day_other": "{n} روز",
        "ahead": "{duration} آینده",
    },
    "en": {
        "banner": (
            "Market assistant. Ask a question, e.g. "
            '"will gold go up in the next 2 hours?"\n'
            "Symbols: gold (XAU/USD) and euro/dollar (EUR/USD). "
            "Full report: 'details' | quit: 'q'"
        ),
        "ask_symbol": "Which symbol? (gold or euro): ",
        "ask_horizon": (
            "For what period? e.g. '2 hours' or 'tomorrow' "
            "(Enter = next 24 hours): "
        ),
        "no_symbol": (
            "I could not tell the symbol. Write gold or euro, "
            "e.g. 'gold in 2 hours'."
        ),
        "ambiguous_symbol": "You named both symbols; please pick one.",
        "unsupported_timeframe": (
            "The smallest supported timeframe is 1 hour "
            "(1h, 4h or daily)."
        ),
        "horizon_too_long": f"The longest horizon is {MAX_CANDLES} candles ahead.",
        "working": "Analysing {symbol}, {horizon} ... (this takes a moment)",
        "forecast": "Forecast",
        "confidence": "confidence",
        "bias": {
            "bullish": "BULLISH ↑",
            "bearish": "BEARISH ↓",
            "neutral": "NEUTRAL →",
        },
        "conf": {"low": "low", "medium": "medium", "high": "high"},
        "unknown": "unclear (the model gave no direction; try again)",
        "stale": (
            "⚠ Data is stale (the market is probably closed); "
            "prices are not live."
        ),
        "review": "⚠ The reviewer still saw a serious problem; read with care.",
        "reason": "  Reason: {reason}",
        "more": " (+{n} more)",
        "errors": "⚠ Some sources were unavailable; type 'details' for more.",
        "disclaimer": "Not financial advice.",
        "error": "Error: {message}",
        "no_details": "Nothing has been analysed yet.",
        "note": "(you asked for {requested}; the closest possible is {actual})",
        "details": "Full report",
        "review_title": "Review",
        "warnings_title": "Warnings",
        "symbols": {SYMBOL_GOLD: "Gold (XAU/USD)", SYMBOL_EURO: "EUR/USD"},
        "minute_one": "{n} minute",
        "minute_other": "{n} minutes",
        "hour_one": "{n} hour",
        "hour_other": "{n} hours",
        "day_one": "{n} day",
        "day_other": "{n} days",
        "ahead": "next {duration}",
    },
}

BRIEF_SYSTEM_PROMPT = """
You write the one-to-two sentence answer shown to the user of a market
assistant. The user is not a trader: keep it short, plain and calm.

Rules:
- Use only facts from the supplied report. Never invent numbers, price levels,
  news or dates.
- At most 2 short sentences, about 25 words in total. No lists, headings,
  markdown or parentheses.
- Say what mainly drives the picture and, if it matters, the main risk. Use
  plain words (trend, momentum, overbought, volatility, resistance) instead of
  indicator names such as SMA, MACD, RSI, ATR or Bollinger, unless one of them
  is the main reason. Use at most one number.
- The forecast direction is printed separately. Never contradict it and do not
  repeat it as a command.
- No price targets and no financial advice.
- Write in the requested language, in natural everyday wording. Do not
  translate jargon word for word, and avoid English abbreviations in Persian.
"""

SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟…])\s+")


@dataclass
class Outcome:
    request: ParsedRequest
    result: dict
    language: str


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------


def duration_text(hours: float, language: str) -> str:
    """'2 hours', '1 hour', '30 minutes', '3 days' in the user's language."""

    labels = TEXT[language]

    if hours < 1:
        kind, value = "minute", round(hours * 60)
    elif hours >= 24 and hours % 24 == 0:
        kind, value = "day", hours / 24
    else:
        kind, value = "hour", hours

    form = "one" if value == 1 else "other"

    return labels[f"{kind}_{form}"].format(n=f"{value:g}")


def horizon_text(request: ParsedRequest, language: str) -> str:
    """'next 2 hours' / '۲ ساعت آینده' for the horizon actually used."""

    labels = TEXT[language]
    duration = duration_text(request.actual_hours, language)

    return labels["ahead"].format(duration=duration)


def english_query(request: ParsedRequest) -> str:
    """The research question handed to the (English) workflow."""

    names = {
        SYMBOL_GOLD: "gold (XAU/USD)",
        SYMBOL_EURO: "the euro against the US dollar (EUR/USD)",
    }
    duration = duration_text(request.actual_hours, "en")

    return (
        f"Where is {names[request.symbol]} heading over the next {duration}? "
        "Assess the technical conditions and the relevant recent news."
    )


def limit_sentences(text: str, count: int) -> str:
    return " ".join(SENTENCE_SPLIT.split(text.strip())[:count])


def clean_brief(text: str) -> str:
    """At most two sentences and 200 characters, with no markdown.

    If two sentences are too long, the first one alone is kept: a whole
    sentence reads better than one that is cut off.
    """

    text = re.sub(r"[*_`#>]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = limit_sentences(text, MAX_BRIEF_SENTENCES)

    if len(text) > MAX_BRIEF_CHARS:
        first = limit_sentences(text, 1)

        if len(first) <= MAX_BRIEF_CHARS:
            text = first

    if len(text) > MAX_BRIEF_CHARS:
        text = text[:MAX_BRIEF_CHARS].rsplit(" ", 1)[0].rstrip(",;:") + "…"

    return text


def review_reason(review: dict, language: str) -> str | None:
    """One line saying WHY the reviewer objected (its first critical issue)."""

    labels = TEXT[language]
    critical = review.get("critical_issues") or []
    issues = critical or review.get("other_issues") or []

    if not issues:
        return None

    text = re.sub(r"\s+", " ", str(issues[0])).strip()

    if len(text) > MAX_REASON_CHARS:
        cut = text[:MAX_REASON_CHARS].rsplit(" ", 1)[0]
        text = cut.rstrip(",;:.") + "…"

    line = labels["reason"].format(reason=text)

    if len(critical) > 1:
        line += labels["more"].format(n=len(critical) - 1)

    return line


def fallback_brief(report: str) -> str:
    """Used when the LLM cannot write the short answer: the report's own
    Executive Summary, cut down to two sentences."""

    lines = report.splitlines()
    heading = re.compile(r"^\s*(?:#+|\d+[.)]|\*\*)")
    body: list[str] = []
    inside = False

    for line in lines:
        if "executive summary" in line.lower():
            inside = True
            continue

        if inside:
            if heading.match(line) and body:
                break
            if line.strip():
                body.append(line)

    text = " ".join(body)

    if not text:  # no such section: use the first prose of the report
        text = " ".join(
            line for line in lines if line.strip() and not heading.match(line)
        )

    return clean_brief(text)


def generate_brief(result: dict, language: str, question: str) -> str:
    """One or two sentences about what drives the picture."""

    report = str(result.get("final_report") or "")

    try:
        model = get_model()
        response = call_with_retries(
            model.invoke,
            [
                SystemMessage(content=BRIEF_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Language: {LANGUAGE_NAMES[language]}\n"
                        f"User question: {question}\n"
                        f"Forecast (printed separately): "
                        f"{json.dumps(result.get('forecast') or {})}\n"
                        f"Warnings: {json.dumps(result.get('errors') or [])}\n\n"
                        f"Report:\n{report[:6000]}"
                    )
                ),
            ],
        )
        text = clean_brief(str(response.content))

        if text:
            return text
    except Exception:  # the short answer must never break the run
        pass

    return fallback_brief(report)


def format_answer(
    request: ParsedRequest,
    result: dict,
    brief: str,
    language: str,
) -> str:
    """The text the user sees: a header, the brief, the forecast, warnings."""

    labels = TEXT[language]
    forecast = result.get("forecast") or {}
    summary = result.get("market_summary") or {}

    header = f"{labels['symbols'][request.symbol]} · {horizon_text(request, language)}"

    if (
        request.requested_hours is not None
        and abs(request.requested_hours - request.actual_hours) > 1e-6
    ):
        header += " " + labels["note"].format(
            requested=duration_text(request.requested_hours, language),
            actual=duration_text(request.actual_hours, language),
        )

    bias = forecast.get("bias")

    if bias in labels["bias"]:
        confidence = labels["conf"].get(forecast.get("confidence"), "?")
        verdict = (
            f"{labels['forecast']}: {labels['bias'][bias]} "
            f"({labels['confidence']}: {confidence})"
        )
    else:
        verdict = f"{labels['forecast']}: {labels['unknown']}"

    lines = [header]

    if brief:
        lines.append(brief)

    lines.append(verdict)

    if (summary.get("data_quality") or {}).get("is_stale"):
        lines.append(labels["stale"])

    review = result.get("review") or {}

    if review.get("verdict") == "NEEDS_REVISION":
        lines.append(labels["review"])
        reason = review_reason(review, language)

        if reason:
            lines.append(reason)

    if result.get("errors"):
        lines.append(labels["errors"])

    lines.append(labels["disclaimer"])

    return "\n".join(lines)


def format_details(result: dict, language: str) -> str:
    """The full English report with the review and warnings."""

    labels = TEXT[language]
    parts = [f"=== {labels['details']} ===", str(result.get("final_report", ""))]

    if result.get("review_feedback"):
        parts += [f"\n=== {labels['review_title']} ===", result["review_feedback"]]

    if result.get("errors"):
        parts += [f"\n=== {labels['warnings_title']} ==="]
        parts += [f"- {error}" for error in result["errors"]]

    return "\n".join(parts)


# --------------------------------------------------------------------------
# Understanding the question
# --------------------------------------------------------------------------


def _say_problems(problems: list[str], language: str) -> None:
    for problem in dict.fromkeys(problems):
        message = TEXT[language].get(problem)

        if message:
            print(message, file=sys.stderr)


def _read(ask, prompt: str) -> str | None:
    """One answer from the user, or None if input ended (Ctrl-D / Ctrl-Z)."""

    try:
        return ask(prompt)
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
        return None


def _is_exit(text: str) -> bool:
    return normalize(text) in EXIT_WORDS


def complete_request(request: ParsedRequest, ask, language: str) -> bool:
    """Fill in what the question did not say, asking if we can.

    ``ask`` is None for one-shot use: a missing symbol is then an error and a
    missing period falls back to the default (next 24 hourly candles).
    Returns False if the request cannot be completed.
    """

    labels = TEXT[language]
    _say_problems(request.problems, language)

    asked = 0
    while request.symbol is None:
        if ask is None or asked >= MAX_ASKS:
            print(labels["no_symbol"], file=sys.stderr)
            return False

        answer = _read(ask, labels["ask_symbol"])
        asked += 1

        if answer is None or _is_exit(answer):
            return False

        reply = parse_request(answer)
        _say_problems(reply.problems, language)
        request.symbol = reply.symbol

    asked = 0
    while request.horizon_candles is None:
        answer = ""

        if ask is not None and asked < MAX_ASKS:
            answer = _read(ask, labels["ask_horizon"])
            asked += 1

        if answer and answer.strip():
            reply = parse_request(answer)
            _say_problems(reply.problems, language)

            if reply.horizon_candles:
                request.interval = reply.interval
                request.horizon_candles = reply.horizon_candles
                request.requested_hours = reply.requested_hours

            continue  # ask again, or use the default after MAX_ASKS tries

        # blank answer, input ended, one-shot use, or too many tries
        request.interval = "1h"
        request.horizon_candles = horizon_for("1h")
        request.requested_hours = None

    return True


# --------------------------------------------------------------------------
# One question, end to end
# --------------------------------------------------------------------------


def log_run(result: dict, args) -> None:
    """Save the forecast for the backtest (quietly)."""

    if args.no_log:
        return

    record = build_forecast_record(result)

    if record is None:
        return

    try:
        append_record(args.log_path, record)
    except OSError as exc:
        print(f"(could not write the forecast log: {exc})", file=sys.stderr)


def handle_question(text: str, args, ask=None) -> tuple[int, Outcome | None]:
    """Understand, analyse and print the short answer.

    Returns ``(exit_code, outcome)``: 0 ok, 1 analysis failed, 2 not understood.
    """

    request = parse_request(text)
    language = request.language if args.lang == "auto" else args.lang
    labels = TEXT[language]

    if not complete_request(request, ask, language):
        return 2, None

    print(
        labels["working"].format(
            symbol=labels["symbols"][request.symbol],
            horizon=horizon_text(request, language),
        ),
        file=sys.stderr,
    )

    try:
        result = run_workflow(
            request.symbol,
            request.interval,
            english_query(request),
            request.horizon_candles,
        )
        log_run(result, args)
        brief = generate_brief(result, language, text)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(labels["error"].format(message=message), file=sys.stderr)
        return 1, None

    print(format_answer(request, result, brief, language))

    if args.full:
        print("\n" + format_details(result, language))

    return 0, Outcome(request, result, language)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "Ask about gold (XAU/USD) or the euro (EUR/USD) in plain Persian "
            "or English and get a short answer with a bullish/bearish "
            "forecast. Not financial advice."
        ),
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Your question. Leave empty for interactive mode.",
    )
    parser.add_argument(
        "--lang",
        choices=["auto", "fa", "en"],
        default="auto",
        help="Answer language (default: the language of the question).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also print the full report after the short answer.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Do not save the forecast for later backtesting.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path(FORECAST_LOG_PATH),
        help=f"Forecast log (default: {FORECAST_LOG_PATH}).",
    )

    return parser


def _use_utf8() -> None:
    """Persian must not crash a console or a redirected stream."""

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def interactive(args, ask=input) -> int:
    print(TEXT["fa"]["banner"])
    print(TEXT["en"]["banner"])
    last: Outcome | None = None

    while True:
        text = _read(ask, "\n> ")

        if text is None:
            return 0

        text = text.strip()

        if not text:
            continue

        key = normalize(text)

        if key in EXIT_WORDS:
            return 0

        if key in DETAILS_WORDS:
            if last is None:
                print(TEXT["en"]["no_details"])
            else:
                print(format_details(last.result, last.language))
            continue

        _, outcome = handle_question(text, args, ask=ask)

        if outcome is not None:
            last = outcome


def main(argv: list[str] | None = None, ask=input) -> int:
    _use_utf8()
    args = build_parser().parse_args(argv)

    if args.question:
        code, _ = handle_question(" ".join(args.question), args, ask=None)
        return code

    return interactive(args, ask=ask)


if __name__ == "__main__":
    sys.exit(main())
