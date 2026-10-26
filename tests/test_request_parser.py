import pytest

from financial_assistant.request_parser import (
    MAX_CANDLES,
    normalize,
    parse_request,
    plan_horizon,
)

GOLD, EURO = "XAU/USD", "EUR/USD"


def understood(text):
    request = parse_request(text)
    return (request.symbol, request.interval, request.horizon_candles)


# --------------------------------------------------------------------------
# Horizon: "how far ahead"
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("قیمت دو ساعت آینده طلا چطور خواهد بود؟", (GOLD, "1h", 2)),
        ("قیمت ۲ ساعت آینده طلا؟", (GOLD, "1h", 2)),
        ("طلا ۲ ساعت دیگه", (GOLD, "1h", 2)),
        ("اونس طلا یک ساعت دیگر", (GOLD, "1h", 1)),
        ("دو ساعت آینده طلا", (GOLD, "1h", 2)),
        ("what will gold do in the next 2 hours?", (GOLD, "1h", 2)),
        ("gold in 4 hours", (GOLD, "1h", 4)),
        ("gold 2 hours ahead", (GOLD, "1h", 2)),
        ("gold 2 hours", (GOLD, "1h", 2)),
        ("Will gold go up in three hours?", (GOLD, "1h", 3)),
        ("euro in an hour", (EURO, "1h", 1)),
        ("one hour euro", (EURO, "1h", 1)),
    ],
)
def test_short_horizons_use_hourly_candles(text, expected):
    assert understood(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("طلا فردا", (GOLD, "4h", 6)),
        ("gold tomorrow", (GOLD, "4h", 6)),
        ("gold in a day", (GOLD, "4h", 6)),
        ("gold in 24 hours", (GOLD, "4h", 6)),
        ("gold in 16 hours", (GOLD, "4h", 4)),
        ("gold in 2 days", (GOLD, "4h", 12)),
        ("gold in 3 days", (GOLD, "1day", 3)),
        ("gold in 1 week", (GOLD, "1day", 7)),
    ],
)
def test_longer_horizons_use_bigger_candles(text, expected):
    assert understood(text) == expected


def test_a_horizon_that_is_not_a_whole_number_of_candles_is_rounded():
    request = parse_request("gold 15 hours ahead")

    assert (request.interval, request.horizon_candles) == ("4h", 4)
    assert request.requested_hours == 15
    assert request.actual_hours == 16  # the caller can tell the user


def test_less_than_an_hour_becomes_one_candle():
    request = parse_request("euro 30 minutes")

    assert (request.interval, request.horizon_candles) == ("1h", 1)
    assert request.requested_hours == 0.5
    assert request.actual_hours == 1


@pytest.mark.parametrize("text", ["gold in half an hour", "طلا نیم ساعت دیگه"])
def test_half_an_hour(text):
    assert understood(text)[1:] == ("1h", 1)


# --------------------------------------------------------------------------
# Timeframe: "which candles"
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("طلا ۴ ساعته", (GOLD, "4h", 6)),
        ("یورو دلار تایم فریم ۴ ساعته", (EURO, "4h", 6)),
        ("طلا تایم‌فریم ۱ ساعته", (GOLD, "1h", 24)),
        ("gold 4h", (GOLD, "4h", 6)),
        ("EUR/USD 4h", (EURO, "4h", 6)),
        ("gold 1h", (GOLD, "1h", 24)),
        ("gold 1d", (GOLD, "1day", 5)),
        ("gold daily", (GOLD, "1day", 5)),
        ("طلا روزانه", (GOLD, "1day", 5)),
        ("gold hourly", (GOLD, "1h", 24)),
        ("gold h4", (GOLD, "4h", 6)),
        ("gold 4 hour chart", (GOLD, "4h", 6)),
        ("gold tf 4h", (GOLD, "4h", 6)),
    ],
)
def test_timeframes_keep_the_default_horizon_for_that_timeframe(text, expected):
    assert understood(text) == expected


def test_a_horizon_and_a_timeframe_can_be_given_together():
    request = parse_request("gold 2 hours ahead on the 1h chart")

    assert (request.interval, request.horizon_candles) == ("1h", 2)


def test_a_horizon_is_counted_in_candles_of_the_chosen_timeframe():
    request = parse_request("gold in 8 hours on the 4h chart")

    assert (request.interval, request.horizon_candles) == ("4h", 2)


def test_the_same_number_is_a_horizon_or_a_timeframe_by_context():
    assert understood("gold 4h")[1:] == ("4h", 6)  # candle size
    assert understood("gold in 4h")[1:] == ("1h", 4)  # four hours ahead


# --------------------------------------------------------------------------
# Symbols
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "طلا",
        "قیمت طلای جهانی",
        "اونس طلا",
        "انس",
        "gold",
        "XAU",
        "xauusd",
        "XAU/USD",
    ],
)
def test_gold_is_recognised(text):
    assert parse_request(f"{text} 2 hours").symbol == GOLD


@pytest.mark.parametrize(
    "text", ["یورو", "یورو دلار", "euro", "EUR", "eurusd", "EUR/USD"]
)
def test_euro_is_recognised(text):
    assert parse_request(f"{text} 2 hours").symbol == EURO


def test_both_symbols_in_one_question_is_a_problem_not_a_guess():
    request = parse_request("gold and euro in 2 hours")

    assert request.symbol is None
    assert request.problems == ["ambiguous_symbol"]


@pytest.mark.parametrize(
    "text", ["تحلیل بده", "hello", "", "dollar in 2 hours", "silver"]
)
def test_no_symbol_is_left_for_the_caller_to_ask(text):
    assert parse_request(text).symbol is None


def test_europe_is_not_the_euro():
    assert parse_request("news from europe in 2 hours").symbol is None


# --------------------------------------------------------------------------
# Things that must NOT be misread (regressions)
# --------------------------------------------------------------------------


def test_the_word_and_is_not_one_day():
    # "and" used to be read as "an" + "d" = one day = 24 hours
    request = parse_request("gold and silver in 2 hours")

    assert (request.interval, request.horizon_candles) == ("1h", 2)
    assert request.requested_hours == 2


@pytest.mark.parametrize("text", ["gold at 9 am", "gold and silver", "gold am"])
def test_am_and_and_do_not_create_a_duration(text):
    request = parse_request(text)

    assert request.requested_hours is None
    assert request.interval is None


def test_a_bare_number_is_not_a_duration():
    assert parse_request("gold 2000").requested_hours is None


# --------------------------------------------------------------------------
# Problems
# --------------------------------------------------------------------------


def test_a_timeframe_below_one_hour_is_not_supported():
    request = parse_request("gold 5m chart")

    assert request.problems == ["unsupported_timeframe"]
    assert request.interval is None


def test_a_horizon_that_is_too_far_is_not_accepted():
    request = parse_request("gold in 100 days")

    assert request.problems == ["horizon_too_long"]
    assert request.interval is None and request.horizon_candles is None


def test_the_longest_accepted_horizon():
    request = parse_request(f"gold in {MAX_CANDLES} days")

    assert (request.interval, request.horizon_candles) == ("1day", MAX_CANDLES)
    assert request.problems == []


# --------------------------------------------------------------------------
# Language, digits and the rest
# --------------------------------------------------------------------------


def test_language_is_detected_from_the_script():
    assert parse_request("قیمت طلا").language == "fa"
    assert parse_request("gold price").language == "en"
    assert parse_request("gold قیمت").language == "fa"


@pytest.mark.parametrize("digits", ["۲", "٢", "2"])
def test_persian_arabic_and_latin_digits_are_the_same(digits):
    assert understood(f"طلا {digits} ساعت آینده") == (GOLD, "1h", 2)


def test_persian_normalisation():
    assert normalize("طلاي  «۲»  ساعت؟") == "طلای 2 ساعت"
    assert normalize("می\u200cخواهم") == "می خواهم"  # half-space
    assert normalize("GOLD  Price?!") == "gold price"


def test_ready_needs_symbol_interval_and_horizon():
    assert parse_request("gold in 2 hours").ready
    assert not parse_request("gold").ready
    assert not parse_request("in 2 hours").ready


@pytest.mark.parametrize(
    "hours, interval, expected",
    [
        (1, None, ("1h", 1)),
        (12, None, ("1h", 12)),
        (13, None, ("4h", 3)),
        (48, None, ("4h", 12)),
        (49, None, ("1day", 2)),
        (0.2, None, ("1h", 1)),
        (8, "4h", ("4h", 2)),
        (30, "1day", ("1day", 1)),
    ],
)
def test_plan_horizon(hours, interval, expected):
    assert plan_horizon(hours, interval) == expected
