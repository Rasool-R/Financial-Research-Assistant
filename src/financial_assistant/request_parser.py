"""Understand a free-text market question in Persian or English.

Examples this module understands::

    قیمت دو ساعت آینده طلا چطور خواهد بود؟   -> XAU/USD, 2 x 1h candles ahead
    gold in 4 hours                          -> XAU/USD, 4 x 1h candles ahead
    EUR/USD 4h                               -> EUR/USD on the 4h timeframe
    طلا فردا                                 -> XAU/USD, 24 hours ahead

It is deliberately rule-based (no LLM): free, instant, testable, and the
caller can show what was understood and ask when something is missing.

A *horizon* ("in 2 hours") is how far ahead the forecast looks. A
*timeframe* ("4h chart", "۴ ساعته") is the candle size used for the analysis.
"""

import math
import re
from dataclasses import dataclass, field

from financial_assistant.forecast import horizon_for

SYMBOL_GOLD = "XAU/USD"
SYMBOL_EURO = "EUR/USD"

HOURS_PER_CANDLE = {"1h": 1, "4h": 4, "1day": 24}
MAX_CANDLES = 48

_TIMEFRAME_BY_HOURS = {1.0: "1h", 4.0: "4h", 24.0: "1day"}
_HOURS_PER_UNIT = {"minute": 1 / 60, "hour": 1.0, "day": 24.0, "week": 168.0}

NUMBER_WORDS = {
    "یک": 1, "دو": 2, "سه": 3, "چهار": 4, "پنج": 5, "شش": 6, "هفت": 7,
    "هشت": 8, "نه": 9, "ده": 10, "یازده": 11, "دوازده": 12, "نیم": 0.5,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "half": 0.5, "a": 1, "an": 1,
}  # fmt: skip

UNIT_KINDS = {
    "ساعته": "hour", "ساعت": "hour", "روزه": "day", "روز": "day",
    "دقیقه": "minute", "هفته": "week",
    "hours": "hour", "hour": "hour", "hrs": "hour", "hr": "hour", "h": "hour",
    "days": "day", "day": "day", "d": "day",
    "minutes": "minute", "minute": "minute", "mins": "minute", "min": "minute",
    "m": "minute",
    "weeks": "week", "week": "week", "w": "week",
}  # fmt: skip

# Persian "-hour" adjectives: "4 ساعته" is a timeframe, "4 ساعت" a duration
_ADJECTIVE_UNITS = {"ساعته", "روزه"}

_WORDS = "|".join(sorted(map(re.escape, NUMBER_WORDS), key=len, reverse=True))
_UNITS = "|".join(sorted(map(re.escape, UNIT_KINDS), key=len, reverse=True))
# One-letter units ("4h", "1d") only make sense right after a digit. After a
# number word they would turn "and" into "an" + "d" (one day) or "am" into
# "a" + "m" (one minute).
_LONG_UNITS = "|".join(
    sorted(
        (re.escape(unit) for unit in UNIT_KINDS if len(unit) > 1),
        key=len,
        reverse=True,
    )
)

DURATION = re.compile(
    rf"(?<![\w.])(?:"
    rf"(?P<dnum>\d+(?:\.\d+)?)(?P<dgap>[\s\-]*)(?P<dunit>{_UNITS})"
    rf"|(?P<wnum>{_WORDS})(?P<wgap>[\s\-]+)(?P<wunit>{_LONG_UNITS})"
    rf")(?!\w)"
)

GOLD = re.compile(
    r"(?<!\w)طلا|(?<!\w)(?:اونس|انس)(?!\w)|\bgold\b|\bxau(?:/?usd)?\b"
)
EURO = re.compile(r"(?<!\w)یورو|\beuro\b|\beur(?:/?usd)?\b")

HORIZON_CUE = re.compile(
    r"آینده|بعدی|بعد|دیگه|دیگر|آتی|\bnext\b|\bahead\b|\blater\b|"
    r"\bwithin\b|from now|\bin\b|\bafter\b|\bover\b"
)
TIMEFRAME_CUE = re.compile(
    r"تایم ?فریم|timeframe|time frame|\btf\b|\bchart\b|\bcandles?\b|"
    r"کندل|نمودار"
)

DAILY_ALIAS = re.compile(r"روزانه|\bdaily\b|\bd1\b")
HOURLY_ALIAS = re.compile(r"ساعتی|\bhourly\b|\bh1\b")
FOUR_HOUR_ALIAS = re.compile(r"\bh4\b")
TOMORROW = re.compile(r"فردا|\btomorrow\b")

_DIGITS = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"
)


@dataclass
class ParsedRequest:
    language: str  # "fa" or "en"
    symbol: str | None = None
    interval: str | None = None
    horizon_candles: int | None = None
    requested_hours: float | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return bool(self.symbol and self.interval and self.horizon_candles)

    @property
    def actual_hours(self) -> float | None:
        if not (self.interval and self.horizon_candles):
            return None

        return float(self.horizon_candles * HOURS_PER_CANDLE[self.interval])


def normalize(text: str) -> str:
    """Lower-case, ASCII digits, unified Persian/Arabic letters, no marks."""

    text = text.translate(_DIGITS).replace("٫", ".")
    text = text.replace("ي", "ی").replace("ك", "ک").replace("ۀ", "ه")
    text = re.sub("[\u200c\u200e\u200f]", " ", text)  # ZWNJ and direction marks
    text = re.sub("[\u064b-\u065f\u0670]", "", text)  # diacritics
    text = text.lower()
    text = re.sub(r"[?!؟،,;:؛()\[\]{}\"“”«»]", " ", text)

    return re.sub(r"\s+", " ", text).strip()


def plan_horizon(hours: float, interval: str | None) -> tuple[str, int]:
    """Candle size and number of candles that best cover ``hours`` ahead.

    Without an explicit timeframe: up to 12 hours uses 1h candles, up to
    48 hours 4h candles, longer uses daily candles. The result is rounded to
    whole candles (at least one).
    """

    if interval is None:
        interval = "1h" if hours <= 12 else "4h" if hours <= 48 else "1day"

    candles = max(1, math.floor(hours / HOURS_PER_CANDLE[interval] + 0.5))

    return interval, candles


def _is_timeframe(
    text: str,
    match: re.Match,
    attached: bool,
    global_horizon: bool,
    global_timeframe: bool,
) -> bool:
    """Is this duration a candle size (True) or a forecast horizon (False)?"""

    window = text[max(0, match.start() - 14) : match.end() + 14]
    near_horizon = bool(HORIZON_CUE.search(window))
    near_timeframe = bool(TIMEFRAME_CUE.search(window))

    if near_horizon != near_timeframe:
        return near_timeframe

    if attached:  # "4h", "۴ ساعته"
        return not (global_horizon and not global_timeframe)

    return False  # "2 hours", "دو ساعت"


def parse_request(text: str) -> ParsedRequest:
    """Extract symbol, timeframe and forecast horizon from free text."""

    request = ParsedRequest(
        language="fa" if re.search("[\u0600-\u06ff]", text) else "en"
    )
    normalized = normalize(text)

    gold = bool(GOLD.search(normalized))
    euro = bool(EURO.search(normalized))

    if gold and euro:
        request.problems.append("ambiguous_symbol")
    elif gold:
        request.symbol = SYMBOL_GOLD
    elif euro:
        request.symbol = SYMBOL_EURO

    interval = None
    if DAILY_ALIAS.search(normalized):
        interval = "1day"
    elif HOURLY_ALIAS.search(normalized):
        interval = "1h"
    elif FOUR_HOUR_ALIAS.search(normalized):
        interval = "4h"

    global_horizon = bool(HORIZON_CUE.search(normalized))
    global_timeframe = bool(TIMEFRAME_CUE.search(normalized))

    horizon_hours = None

    for match in DURATION.finditer(normalized):
        if match.group("dnum"):
            value = float(match.group("dnum"))
            unit, gap = match.group("dunit"), match.group("dgap")
        else:
            value = NUMBER_WORDS[match.group("wnum")]
            unit, gap = match.group("wunit"), match.group("wgap")

        hours = value * _HOURS_PER_UNIT[UNIT_KINDS[unit]]
        attached = not re.search(r"\s", gap) or unit in _ADJECTIVE_UNITS

        if _is_timeframe(
            normalized, match, attached, global_horizon, global_timeframe
        ):
            timeframe = _TIMEFRAME_BY_HOURS.get(round(hours, 6))

            if timeframe:
                interval = interval or timeframe
            else:
                request.problems.append("unsupported_timeframe")
        elif horizon_hours is None:
            horizon_hours = hours

    if horizon_hours is None and TOMORROW.search(normalized):
        horizon_hours = 24.0

    if horizon_hours is not None:
        request.requested_hours = horizon_hours
        chosen, candles = plan_horizon(horizon_hours, interval)

        if candles > MAX_CANDLES:
            request.problems.append("horizon_too_long")
        else:
            request.interval, request.horizon_candles = chosen, candles
    elif interval is not None:
        request.interval = interval
        request.horizon_candles = horizon_for(interval)

    return request
