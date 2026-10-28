import runpy
from pathlib import Path

import pytest

from financial_assistant import app
from financial_assistant.forecast import load_records
from financial_assistant.request_parser import parse_request
from tests.fakes import FakeModel

FA = app.TEXT["fa"]
EN = app.TEXT["en"]

REPORT = (
    "1. Executive Summary\n"
    "Gold is in a short-term uptrend. The nearest resistance is one ATR away. "
    "A stronger dollar is the main risk.\n\n"
    "2. Market Snapshot\nClose 4400.50.\n"
)


def make_result(bias="bullish", confidence="medium", horizon=2, **extra):
    result = {
        "final_report": REPORT,
        "forecast": {
            "bias": bias,
            "confidence": confidence,
            "horizon_candles": horizon,
        },
        "market_summary": {
            "symbol": "XAU/USD",
            "interval": "1h",
            "latest_candle": {
                "datetime": "2026-10-23T10:00:00+00:00",
                "close": 4400.5,
            },
            "market_conditions": {"trend": "bullish", "macd_momentum": "bullish"},
            "volatility": {"atr_percent": 0.2},
            "data_quality": {"is_stale": False},
        },
        "review": {"verdict": "PASS"},
        "review_feedback": "VERDICT: PASS",
        "errors": [],
    }
    result.update(extra)
    return result


@pytest.fixture
def workflow(monkeypatch, tmp_path):
    """Fake analysis and short-answer writer; records every call."""

    state = {"calls": [], "result": make_result(), "brief": "SHORT SUMMARY."}

    def run(symbol, interval, query, horizon_candles=None):
        state["calls"].append((symbol, interval, query, horizon_candles))
        if isinstance(state["result"], Exception):
            raise state["result"]
        return state["result"]

    monkeypatch.setattr(app, "run_workflow", run)
    monkeypatch.setattr(
        app, "generate_brief", lambda result, language, question: state["brief"]
    )
    state["log"] = tmp_path / "forecasts.jsonl"
    state["argv"] = ["--log-path", str(state["log"])]

    return state


def ask_from(answers):
    """A scripted replacement for input(); records the prompts it was given."""

    answers = iter(answers)
    prompts = []

    def ask(prompt):
        prompts.append(prompt)
        value = next(answers)
        if isinstance(value, BaseException):
            raise value
        return value

    ask.prompts = prompts
    return ask


def run_once(workflow, question, *flags):
    return app.main([question, *flags, *workflow["argv"]])


# --------------------------------------------------------------------------
# The short answer
# --------------------------------------------------------------------------


def test_a_persian_question_gets_a_short_persian_answer(workflow, capsys):
    code = run_once(workflow, "قیمت دو ساعت آینده طلا چطور خواهد بود؟")

    lines = capsys.readouterr().out.strip().splitlines()
    assert code == 0
    assert len(lines) == 4  # header, summary, forecast, disclaimer
    assert lines[0] == "طلا (XAU/USD) · 2 ساعت آینده"
    assert lines[1] == "SHORT SUMMARY."
    assert lines[2] == "پیش‌بینی: صعودی ↑ (اطمینان: متوسط)"
    assert lines[3] == FA["disclaimer"]


def test_the_workflow_gets_symbol_timeframe_horizon_and_an_english_question(
    workflow,
):
    run_once(workflow, "قیمت دو ساعت آینده طلا چطور خواهد بود؟")

    (symbol, interval, query, horizon), = workflow["calls"]
    assert (symbol, interval, horizon) == ("XAU/USD", "1h", 2)
    assert query.isascii()  # Persian never reaches the news search
    assert "gold (XAU/USD)" in query and "next 2 hours" in query


def test_an_english_question_gets_an_english_answer(workflow, capsys):
    run_once(workflow, "will gold go up in the next 2 hours?")

    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "Gold (XAU/USD) · next 2 hours"
    assert lines[2] == "Forecast: BULLISH ↑ (confidence: medium)"
    assert lines[3] == "Not financial advice."


@pytest.mark.parametrize(
    "bias, expected",
    [
        ("bullish", "صعودی ↑"),
        ("bearish", "نزولی ↓"),
        ("neutral", "خنثی →"),
    ],
)
def test_the_forecast_direction_comes_from_the_forecast(
    workflow, capsys, bias, expected
):
    workflow["result"] = make_result(bias=bias)

    run_once(workflow, "طلا ۲ ساعت آینده")

    assert expected in capsys.readouterr().out


def test_the_direction_cannot_be_overridden_by_the_summary_text(
    workflow, capsys
):
    workflow["result"] = make_result(bias="bearish")
    workflow["brief"] = "Everything looks bullish and strong."

    run_once(workflow, "gold in 2 hours")

    out = capsys.readouterr().out
    assert "BEARISH ↓" in out  # printed by code from the forecast
    assert "BULLISH" not in out


def test_an_unknown_forecast_is_shown_as_such(workflow, capsys):
    workflow["result"] = make_result(bias="unknown", confidence="unknown")

    run_once(workflow, "gold in 2 hours")

    out = capsys.readouterr().out
    assert EN["unknown"] in out
    assert "BULLISH" not in out and "confidence:" not in out


@pytest.mark.parametrize("confidence", ["low", "medium", "high"])
def test_the_confidence_is_shown(workflow, capsys, confidence):
    workflow["result"] = make_result(confidence=confidence)

    run_once(workflow, "طلا ۲ ساعت آینده")

    assert f"اطمینان: {FA['conf'][confidence]}" in capsys.readouterr().out


def test_a_clean_run_prints_no_warnings(workflow, capsys):
    run_once(workflow, "gold in 2 hours")

    out = capsys.readouterr().out
    assert "⚠" not in out
    assert len(out.strip().splitlines()) == 4


def test_each_warning_is_one_short_line(workflow, capsys):
    stale = {"data_quality": {"is_stale": True}}
    workflow["result"] = make_result(
        market_summary={**make_result()["market_summary"], **stale},
        review={"verdict": "NEEDS_REVISION"},
        errors=["research_agent: Web search failed"],
    )

    run_once(workflow, "gold in 2 hours")

    lines = capsys.readouterr().out.strip().splitlines()
    assert EN["stale"] in lines
    assert EN["review"] in lines
    assert EN["errors"] in lines
    assert len(lines) == 7  # the four lines plus three warnings


def test_a_rounded_horizon_is_explained(workflow, capsys):
    run_once(workflow, "gold 15 hours ahead")

    assert "you asked for 15 hours; the closest possible is 16 hours" in (
        capsys.readouterr().out
    )
    assert workflow["calls"][0][1:2] == ("4h",)


def test_half_an_hour_is_explained_too(workflow, capsys):
    run_once(workflow, "طلا نیم ساعت دیگه")

    out = capsys.readouterr().out
    assert "30 دقیقه" in out and "1 ساعت" in out


def test_a_timeframe_without_a_horizon_uses_its_default(workflow, capsys):
    run_once(workflow, "طلا روزانه")

    (_, interval, _, horizon), = workflow["calls"]
    assert (interval, horizon) == ("1day", 5)
    assert "5 روز آینده" in capsys.readouterr().out


def test_the_language_can_be_forced(workflow, capsys):
    run_once(workflow, "قیمت دو ساعت آینده طلا؟", "--lang", "en")

    assert "Gold (XAU/USD)" in capsys.readouterr().out

    run_once(workflow, "gold in 2 hours", "--lang", "fa")

    assert "طلا (XAU/USD)" in capsys.readouterr().out


def test_full_prints_the_report_after_the_short_answer(workflow, capsys):
    run_once(workflow, "gold in 2 hours", "--full")

    out = capsys.readouterr().out
    assert out.index("Not financial advice.") < out.index("=== Full report ===")
    assert "Market Snapshot" in out
    assert "VERDICT: PASS" in out


def test_progress_goes_to_stderr_not_stdout(workflow, capsys):
    run_once(workflow, "gold in 2 hours")

    captured = capsys.readouterr()
    assert "Analysing" in captured.err
    assert "Analysing" not in captured.out


# --------------------------------------------------------------------------
# Missing information
# --------------------------------------------------------------------------


def test_a_one_shot_question_without_a_symbol_is_refused(workflow, capsys):
    code = run_once(workflow, "تحلیل بده")

    assert code == 2
    assert workflow["calls"] == []
    assert FA["no_symbol"] in capsys.readouterr().err


def test_a_one_shot_question_without_a_period_uses_the_default(workflow):
    code = run_once(workflow, "gold")

    assert code == 0
    (_, interval, _, horizon), = workflow["calls"]
    assert (interval, horizon) == ("1h", 24)


def test_the_assistant_asks_for_the_symbol_and_then_the_period(workflow, capsys):
    ask = ask_from(["تحلیل بده", "طلا", "۲ ساعت", "q"])

    code = app.main(workflow["argv"], ask=ask)

    assert code == 0
    (symbol, interval, _, horizon), = workflow["calls"]
    assert (symbol, interval, horizon) == ("XAU/USD", "1h", 2)
    assert FA["ask_symbol"] in ask.prompts
    assert FA["ask_horizon"] in ask.prompts


def test_pressing_enter_at_the_period_prompt_uses_the_default(workflow):
    ask = ask_from(["gold", "", "q"])

    app.main(workflow["argv"], ask=ask)

    (_, interval, _, horizon), = workflow["calls"]
    assert (interval, horizon) == ("1h", 24)


def test_a_symbol_in_a_later_answer_is_understood(workflow):
    ask = ask_from(["analyse please", "blah", "euro", "4h", "q"])

    app.main(workflow["argv"], ask=ask)

    (symbol, interval, _, horizon), = workflow["calls"]
    assert (symbol, interval, horizon) == ("EUR/USD", "4h", 6)


def test_the_assistant_gives_up_after_three_unclear_answers(workflow, capsys):
    ask = ask_from(["analyse please", "x", "y", "z", "q"])

    app.main(workflow["argv"], ask=ask)

    assert workflow["calls"] == []
    assert EN["no_symbol"] in capsys.readouterr().err


def test_typing_exit_at_a_follow_up_cancels_it(workflow):
    ask = ask_from(["analyse please", "exit", "q"])

    app.main(workflow["argv"], ask=ask)

    assert workflow["calls"] == []


def test_problems_are_explained_when_asking_again(workflow, capsys):
    ask = ask_from(["gold 5m chart", "", "q"])

    app.main(workflow["argv"], ask=ask)

    assert EN["unsupported_timeframe"] in capsys.readouterr().err
    (_, interval, _, horizon), = workflow["calls"]
    assert (interval, horizon) == ("1h", 24)  # the default was used


def test_both_symbols_in_one_question_asks_which(workflow, capsys):
    ask = ask_from(["gold and euro in 2 hours", "euro", "q"])

    app.main(workflow["argv"], ask=ask)

    assert EN["ambiguous_symbol"] in capsys.readouterr().err
    assert workflow["calls"][0][0] == "EUR/USD"
    assert workflow["calls"][0][3] == 2  # the horizon was kept


# --------------------------------------------------------------------------
# Interactive session
# --------------------------------------------------------------------------


def test_the_session_shows_a_banner_in_both_languages(workflow, capsys):
    app.main(workflow["argv"], ask=ask_from(["q"]))

    out = capsys.readouterr().out
    assert FA["banner"] in out and EN["banner"] in out


@pytest.mark.parametrize("word", ["q", "exit", "QUIT", "خروج"])
def test_exit_words(workflow, word):
    assert app.main(workflow["argv"], ask=ask_from([word])) == 0
    assert workflow["calls"] == []


@pytest.mark.parametrize("error", [EOFError(), KeyboardInterrupt()])
def test_end_of_input_ends_the_session_quietly(workflow, error):
    assert app.main(workflow["argv"], ask=ask_from([error])) == 0


def test_blank_lines_are_ignored(workflow):
    ask = ask_from(["", "   ", "q"])

    app.main(workflow["argv"], ask=ask)

    assert workflow["calls"] == []


def test_several_questions_in_one_session(workflow, capsys):
    ask = ask_from(["gold in 2 hours", "یورو ۴ ساعته", "q"])

    app.main(workflow["argv"], ask=ask)

    assert [c[0] for c in workflow["calls"]] == ["XAU/USD", "EUR/USD"]
    out = capsys.readouterr().out
    assert "Gold (XAU/USD)" in out and "یورو/دلار (EUR/USD)" in out


@pytest.mark.parametrize("word", ["details", "جزئیات"])
def test_details_prints_the_last_full_report(workflow, capsys, word):
    ask = ask_from(["gold in 2 hours", word, "q"])

    app.main(workflow["argv"], ask=ask)

    out = capsys.readouterr().out
    assert "Market Snapshot" in out
    assert "VERDICT: PASS" in out


def test_details_before_any_analysis(workflow, capsys):
    app.main(workflow["argv"], ask=ask_from(["details", "q"]))

    assert EN["no_details"] in capsys.readouterr().out
    assert workflow["calls"] == []


def test_details_lists_the_warnings(workflow, capsys):
    workflow["result"] = make_result(errors=["research_agent: Web search failed"])
    ask = ask_from(["gold in 2 hours", "details", "q"])

    app.main(workflow["argv"], ask=ask)

    out = capsys.readouterr().out
    assert "=== Warnings ===" in out
    assert "- research_agent: Web search failed" in out


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


def test_a_failed_analysis_is_reported_briefly(workflow, capsys):
    workflow["result"] = RuntimeError("Twelve Data is down")

    code = run_once(workflow, "gold in 2 hours")

    captured = capsys.readouterr()
    assert code == 1
    assert "Error: RuntimeError: Twelve Data is down" in captured.err
    assert captured.out == ""  # nothing misleading on stdout


def test_the_session_survives_a_failed_analysis(workflow, capsys):
    workflow["result"] = RuntimeError("boom")
    ask = ask_from(["gold in 2 hours", "q"])

    assert app.main(workflow["argv"], ask=ask) == 0
    assert "boom" in capsys.readouterr().err


def test_a_failing_summary_never_breaks_the_answer(monkeypatch, workflow, capsys):
    monkeypatch.undo()  # use the real generate_brief with a broken model
    monkeypatch.setattr(
        app, "run_workflow", lambda *a: make_result()
    )

    def broken_model():
        raise RuntimeError("no key")

    monkeypatch.setattr(app, "get_model", broken_model)

    code = app.main(["gold in 2 hours", "--no-log"])

    out = capsys.readouterr().out
    assert code == 0
    assert "Gold is in a short-term uptrend." in out  # the report's own summary
    assert "BULLISH" in out


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------


def test_the_forecast_is_logged_with_its_horizon(workflow):
    run_once(workflow, "gold in 2 hours")

    records, _ = load_records(workflow["log"])
    assert len(records) == 1
    assert records[0]["bias"] == "bullish"
    assert records[0]["horizon_candles"] == 2
    assert records[0]["interval"] == "1h"


def test_no_log_flag(workflow):
    run_once(workflow, "gold in 2 hours", "--no-log")

    assert not workflow["log"].exists()


def test_an_unknown_forecast_is_not_logged(workflow):
    workflow["result"] = make_result(bias="unknown", confidence="unknown")

    run_once(workflow, "gold in 2 hours")

    assert not workflow["log"].exists()


def test_an_unwritable_log_does_not_lose_the_answer(workflow, capsys, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")  # a file where a directory is needed

    code = app.main(
        ["gold in 2 hours", "--log-path", str(blocker / "sub" / "log.jsonl")]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "BULLISH" in captured.out
    assert "could not write the forecast log" in captured.err


# --------------------------------------------------------------------------
# The short summary
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("One. Two. Three.", "One. Two."),
        ("**Bold** claim. _Next_ one. Third.", "Bold claim. Next one."),
        ("Line one\nstill one. Second!\n\nThird?", "Line one still one. Second!"),
        ("نخست. دوم؟ سوم.", "نخست. دوم؟"),
        ("Price 4397.92 held. Next.", "Price 4397.92 held. Next."),
        ("  # Heading  \n", "Heading"),
        ("", ""),
    ],
)
def test_clean_brief(text, expected):
    assert app.clean_brief(text) == expected


def test_a_very_long_sentence_is_cut_at_a_word():
    brief = app.clean_brief("word " * 200)

    assert len(brief) <= app.MAX_BRIEF_CHARS + 1
    assert brief.endswith("…")
    assert "wor…" not in brief


@pytest.mark.parametrize(
    "report",
    [
        REPORT,
        "# Executive Summary\nGold is rising. Risk is a firmer dollar. Third.\n"
        "# Market Snapshot\nClose 4400.50.",
        "**Executive Summary**\nGold is rising. Risk is a firmer dollar. Third.\n"
        "**Market Snapshot**\nx",
    ],
)
def test_the_fallback_is_the_reports_own_summary(report):
    brief = app.fallback_brief(report)

    assert brief.startswith("Gold is")
    assert "Market Snapshot" not in brief
    assert brief.count(".") <= 2 + brief.count("4400")


def test_the_fallback_without_a_summary_section_uses_the_first_prose():
    brief = app.fallback_brief("1. Overview\nGold is rising. Dollar is soft. Third.")

    assert brief == "Gold is rising. Dollar is soft."


def test_the_fallback_of_an_empty_report_is_empty():
    assert app.fallback_brief("") == ""


def test_generate_brief_asks_for_the_users_language_and_gives_the_forecast(
    monkeypatch,
):
    model = FakeModel(["**Uptrend** with a firm RSI. Dollar risk remains. Extra."])
    monkeypatch.setattr(app, "get_model", lambda: model)

    brief = app.generate_brief(make_result(), "fa", "قیمت طلا؟")

    system, human = model.calls[0]
    assert "Persian (Farsi)" in human.content
    assert '"bias": "bullish"' in human.content
    assert "قیمت طلا؟" in human.content
    assert "Executive Summary" in human.content
    assert "At most 2 short sentences" in system.content
    assert brief == "Uptrend with a firm RSI. Dollar risk remains."


def test_an_empty_reply_from_the_model_falls_back(monkeypatch):
    monkeypatch.setattr(app, "get_model", lambda: FakeModel(["   "]))

    brief = app.generate_brief(make_result(), "en", "q")

    assert brief.startswith("Gold is in a short-term uptrend.")


# --------------------------------------------------------------------------
# Small helpers and the entry point
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hours, language, expected",
    [
        (1, "en", "1 hour"),
        (2, "en", "2 hours"),
        (0.5, "en", "30 minutes"),
        (24, "en", "1 day"),
        (72, "en", "3 days"),
        (36, "en", "36 hours"),
        (2, "fa", "2 ساعت"),
        (48, "fa", "2 روز"),
    ],
)
def test_duration_text(hours, language, expected):
    assert app.duration_text(hours, language) == expected


def test_horizon_text_uses_the_horizon_actually_analysed():
    request = parse_request("gold 15 hours ahead")

    assert app.horizon_text(request, "en") == "next 16 hours"
    assert app.horizon_text(request, "fa") == "16 ساعت آینده"


def test_every_language_has_every_message():
    assert set(app.TEXT["fa"]) == set(app.TEXT["en"])
    assert set(app.TEXT["fa"]["bias"]) == set(app.TEXT["en"]["bias"])


def test_main_py_runs_the_app():
    namespace = runpy.run_path(
        str(Path(__file__).parent.parent / "main.py"), run_name="not_main"
    )

    assert namespace["main"] is app.main


def test_the_parser_defaults():
    args = app.build_parser().parse_args([])

    assert args.question == [] and args.lang == "auto"
    assert not args.full and not args.no_log


# --------------------------------------------------------------------------
# A shorter, plainer summary
# --------------------------------------------------------------------------


def test_the_summary_cap_is_200_characters():
    assert app.MAX_BRIEF_CHARS == 200
    assert app.MAX_BRIEF_SENTENCES == 2


def test_two_sentences_that_are_too_long_are_reduced_to_the_first_whole_one():
    first = "The short term trend is up and momentum is positive."
    second = "The main risk is " + "a firmer dollar and rising yields " * 6 + "."

    brief = app.clean_brief(f"{first} {second}")

    assert brief == first  # a whole sentence, not one cut in the middle
    assert "…" not in brief


def test_a_single_sentence_that_is_too_long_is_cut_at_a_word_with_an_ellipsis():
    brief = app.clean_brief("trend " * 100)

    assert len(brief) <= app.MAX_BRIEF_CHARS + 1
    assert brief.endswith("…")
    assert not brief.endswith("tren…")


def test_the_users_real_brief_is_shortened_to_its_first_sentence():
    brief = app.clean_brief(
        "قیمت در ساعت‌های آینده تحت تأثیر روند صعودی کوتاه‌مدت و مومنتوم مثبت "
        "(SMA‑20 SMA‑50، MACD مثبت، RSI ≈ 65) است که نزدیک باند بالایی بولینگر "
        "(~4022) قرار دارد. ریسک اصلی نزدیکی به سقف نرم و سیگنال‌های نزولی "
        "چارچوب‌های بالاتر و اخبار اخیر نرخ بهره فدرال است."
    )

    assert len(brief) <= app.MAX_BRIEF_CHARS
    assert brief.endswith("قرار دارد.")


def test_the_prompt_asks_for_plain_short_wording():
    prompt = " ".join(app.BRIEF_SYSTEM_PROMPT.split())  # ignore line breaks

    assert "about 25 words" in prompt
    assert "At most 2 short sentences" in prompt
    assert "instead of indicator names such as SMA, MACD, RSI" in prompt
    assert "at most one number" in prompt.lower()
    assert "Do not translate jargon word for word" in prompt
    assert "avoid English abbreviations in Persian" in prompt
    assert "Never contradict" in prompt  # the earlier safety rules remain


# --------------------------------------------------------------------------
# Why the reviewer objected
# --------------------------------------------------------------------------


def reviewed(**review):
    return make_result(review={"verdict": "NEEDS_REVISION", **review})


def lines_for(workflow, result, question="gold in 2 hours"):
    workflow["result"] = result
    return result


def test_the_reviewers_reason_is_shown_under_the_warning(workflow, capsys):
    lines_for(
        workflow,
        reviewed(critical_issues=["The report has 627 words, over the limit."]),
    )

    run_once(workflow, "gold in 2 hours")

    lines = capsys.readouterr().out.strip().splitlines()
    warning = lines.index(EN["review"])
    assert lines[warning + 1] == (
        "  Reason: The report has 627 words, over the limit."
    )
    assert lines[-1] == EN["disclaimer"]


def test_a_long_reason_is_cut_at_a_word_and_counts_the_others(workflow, capsys):
    long_issue = "The report says the 1day trend is bullish " + "but " * 60
    lines_for(workflow, reviewed(critical_issues=[long_issue, "b", "c"]))

    run_once(workflow, "gold in 2 hours")

    reason = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("  Reason:")
    )
    assert "…" in reason
    assert reason.endswith("(+2 more)")
    assert len(reason) < app.MAX_REASON_CHARS + 40


def test_the_reason_is_labelled_in_persian_too(workflow, capsys):
    lines_for(workflow, reviewed(critical_issues=["Wrong RSI value."]))

    run_once(workflow, "طلا ۲ ساعت آینده")

    out = capsys.readouterr().out
    assert "علت (به انگلیسی): Wrong RSI value." in out


def test_without_any_issue_text_there_is_just_the_warning(workflow, capsys):
    lines_for(workflow, reviewed())

    run_once(workflow, "gold in 2 hours")

    out = capsys.readouterr().out
    assert EN["review"] in out
    assert "Reason:" not in out


def test_other_issues_are_used_when_there_is_no_critical_one():
    review = {"verdict": "NEEDS_REVISION", "other_issues": ["Minor wording."]}

    assert app.review_reason(review, "en") == "  Reason: Minor wording."


def test_a_passing_review_shows_no_warning_and_no_reason(workflow, capsys):
    lines_for(
        workflow,
        make_result(review={"verdict": "PASS", "critical_issues": ["x"]}),
    )

    run_once(workflow, "gold in 2 hours")

    out = capsys.readouterr().out
    assert "⚠" not in out and "Reason:" not in out


def test_the_reason_is_only_one_extra_line(workflow, capsys):
    lines_for(workflow, reviewed(critical_issues=["Wrong RSI value."]))

    run_once(workflow, "gold in 2 hours")

    # header, summary, forecast, warning, reason, disclaimer
    assert len(capsys.readouterr().out.strip().splitlines()) == 6
