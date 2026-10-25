import pytest

from financial_assistant import cli

RESULT = {
    "final_report": "REPORT BODY",
    "review": {"verdict": "PASS"},
    "review_feedback": "VERDICT: PASS",
    "revision_count": 1,
    "errors": ["research_agent: Web search failed"],
}


@pytest.fixture
def fake_workflow(monkeypatch):
    calls = []

    def run(symbol, interval, query):
        calls.append((symbol, interval, query))
        return RESULT

    monkeypatch.setattr(cli, "run_workflow", run)
    return calls


def test_defaults(fake_workflow, capsys):
    assert cli.main([]) == 0

    symbol, interval, query = fake_workflow[0]
    assert (symbol, interval) == ("XAU/USD", "1h")
    assert "XAU/USD" in query
    assert capsys.readouterr().out.strip() == "REPORT BODY"


def test_custom_arguments(fake_workflow):
    cli.main(["--symbol", "EUR/USD", "--interval", "4h", "--query", "why?"])

    assert fake_workflow[0] == ("EUR/USD", "4h", "why?")


def test_rejects_unsupported_symbol(fake_workflow):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--symbol", "BTC/USD"])

    assert exc.value.code == 2
    assert fake_workflow == []


def test_show_review_prints_review_and_errors(fake_workflow, capsys):
    cli.main(["--show-review"])

    out = capsys.readouterr().out
    assert "VERDICT: PASS" in out
    assert "Revisions: 1" in out
    assert "Web search failed" in out


def test_output_file(fake_workflow, tmp_path):
    target = tmp_path / "report.md"

    cli.main(["-o", str(target)])

    assert target.read_text(encoding="utf-8") == "REPORT BODY"


def test_json_output(fake_workflow, capsys):
    import json

    cli.main(["--json"])

    assert json.loads(capsys.readouterr().out)["final_report"] == "REPORT BODY"


def test_workflow_error_returns_exit_code_1(monkeypatch, capsys):
    def boom(symbol, interval, query):
        raise RuntimeError("no data")

    monkeypatch.setattr(cli, "run_workflow", boom)

    assert cli.main([]) == 1
    assert "RuntimeError: no data" in capsys.readouterr().err


def test_warns_when_review_still_failing(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "run_workflow",
        lambda *a: {**RESULT, "review": {"verdict": "NEEDS_REVISION"}},
    )

    assert cli.main([]) == 0
    assert "still reported critical issues" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Forecast logging
# --------------------------------------------------------------------------

LOGGED_RESULT = {
    "final_report": "REPORT BODY",
    "forecast": {"bias": "bullish", "confidence": "medium", "horizon_candles": 24},
    "market_summary": {
        "symbol": "XAU/USD",
        "interval": "1h",
        "latest_candle": {"datetime": "2026-10-23T10:00:00+00:00", "close": 4400.5},
        "market_conditions": {"trend": "bullish", "macd_momentum": "bullish"},
        "volatility": {"atr_percent": 0.2},
        "data_quality": {"is_stale": False},
    },
    "review": {"verdict": "PASS"},
}


def run_with(monkeypatch, result, *args):
    monkeypatch.setattr(cli, "run_workflow", lambda *a: result)
    return cli.main(list(args))


def test_the_forecast_is_logged_after_a_run(monkeypatch, tmp_path, capsys):
    from financial_assistant.forecast import load_records

    log = tmp_path / "data" / "forecasts.jsonl"

    assert run_with(monkeypatch, LOGGED_RESULT, "--log-path", str(log)) == 0

    records, _ = load_records(log)
    assert len(records) == 1
    assert records[0]["bias"] == "bullish"
    assert records[0]["reference_price"] == 4400.5
    err = capsys.readouterr().err
    assert "Forecast: bullish (medium confidence)" in err
    assert "next 24 1h candles" in err
    assert f"Logged to {log}" in err


def test_no_log_flag_skips_the_log(monkeypatch, tmp_path, capsys):
    log = tmp_path / "forecasts.jsonl"

    run_with(monkeypatch, LOGGED_RESULT, "--no-log", "--log-path", str(log))

    assert not log.exists()
    assert "Not logged (--no-log)" in capsys.readouterr().err


def test_runs_without_a_direction_are_not_logged(monkeypatch, tmp_path, capsys):
    log = tmp_path / "forecasts.jsonl"
    result = {
        **LOGGED_RESULT,
        "forecast": {"bias": "unknown", "confidence": "unknown"},
    }

    run_with(monkeypatch, result, "--log-path", str(log))

    assert not log.exists()
    assert "no directional forecast" in capsys.readouterr().err


def test_stale_forecasts_are_logged_with_a_note(monkeypatch, tmp_path, capsys):
    from financial_assistant.forecast import load_records

    log = tmp_path / "forecasts.jsonl"
    stale = {
        **LOGGED_RESULT,
        "market_summary": {
            **LOGGED_RESULT["market_summary"],
            "data_quality": {"is_stale": True},
        },
    }

    run_with(monkeypatch, stale, "--log-path", str(log))

    assert load_records(log)[0][0]["is_stale"] is True
    assert "excluded from scoring by default" in capsys.readouterr().err


def test_each_run_appends_a_new_line(monkeypatch, tmp_path):
    log = tmp_path / "forecasts.jsonl"

    for _ in range(3):
        run_with(monkeypatch, LOGGED_RESULT, "--log-path", str(log))

    assert len(log.read_text().splitlines()) == 3


def test_json_output_stays_parseable_when_logging(monkeypatch, tmp_path, capsys):
    import json

    log = tmp_path / "forecasts.jsonl"

    run_with(monkeypatch, LOGGED_RESULT, "--json", "--log-path", str(log))

    assert json.loads(capsys.readouterr().out)["final_report"] == "REPORT BODY"
