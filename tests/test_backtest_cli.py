import json

import numpy as np
import pandas as pd
import pytest

from financial_assistant import backtest_cli as cli
from financial_assistant.forecast import append_record


def candles(n=300):
    rng = np.random.default_rng(5)
    close = 4000 + np.cumsum(rng.normal(0, 4, n))

    return pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
            "open": close,
            "high": close + 2,
            "low": close - 2,
            "close": close,
        }
    )


FRAME = candles()


def log_record(index, bias="bullish", **extra):
    data = {
        "symbol": "XAU/USD",
        "interval": "1h",
        "bias": bias,
        "confidence": "medium",
        "reference_candle_start": FRAME.loc[index, "datetime"].isoformat(),
        "is_stale": False,
        "atr_percent": 0.2,
    }
    data.update(extra)
    return data


@pytest.fixture
def fake_fetch(monkeypatch):
    requests = []

    def fetch(symbol, interval, outputsize):
        requests.append((symbol, interval, outputsize))
        return FRAME

    monkeypatch.setattr(cli, "fetch_candles", fetch)
    return requests


def test_resolve_prints_a_report_for_the_log(tmp_path, fake_fetch, capsys):
    log = tmp_path / "forecasts.jsonl"
    for index in (10, 50, 90):
        append_record(log, log_record(index))
    append_record(log, log_record(297))  # 297 + 5 > last candle: pending

    assert cli.main(["resolve", "--log", str(log), "--horizon-candles", "5"]) == 0

    out = capsys.readouterr().out
    assert "4 logged forecasts" in out
    assert "Live forecasts, 1h" in out
    assert "Pending forecasts not scored: 1" in out
    assert "INSUFFICIENT DATA" in out  # 3 forecasts are far too few
    assert fake_fetch[0][:2] == ("XAU/USD", "1h")


def test_resolve_reports_what_it_skipped(tmp_path, fake_fetch, capsys):
    log = tmp_path / "forecasts.jsonl"
    append_record(log, log_record(10))
    append_record(log, log_record(10))  # duplicate
    append_record(log, log_record(20, is_stale=True))
    append_record(log, log_record(30, bias="unknown"))
    with log.open("a") as handle:
        handle.write("{broken\n")

    cli.main(["resolve", "--log", str(log), "--horizon-candles", "5"])

    out = capsys.readouterr().out
    assert "1 duplicates ignored" in out
    assert "1 on stale data excluded" in out
    assert "1 without a direction" in out
    assert "1 unreadable lines skipped" in out


def test_resolve_include_stale(tmp_path, fake_fetch, capsys):
    log = tmp_path / "forecasts.jsonl"
    append_record(log, log_record(20, is_stale=True))

    cli.main(
        ["resolve", "--log", str(log), "--horizon-candles", "5", "--include-stale"]
    )

    assert "0 on stale data excluded" in capsys.readouterr().out


def test_resolve_with_nothing_to_score(tmp_path, fake_fetch, capsys):
    log = tmp_path / "forecasts.jsonl"
    log.write_text("")

    assert cli.main(["resolve", "--log", str(log)]) == 0
    assert "Nothing to score yet" in capsys.readouterr().out
    assert fake_fetch == []


def test_resolve_without_a_log_explains_what_to_do(tmp_path, capsys):
    code = cli.main(["resolve", "--log", str(tmp_path / "missing.jsonl")])

    assert code == 1
    err = capsys.readouterr().err
    assert "No forecast log" in err
    assert "financial-assistant" in err


def test_resolve_json_and_output_file(tmp_path, fake_fetch, capsys):
    log = tmp_path / "forecasts.jsonl"
    append_record(log, log_record(10))
    target = tmp_path / "result.json"

    cli.main(
        [
            "resolve",
            "--log",
            str(log),
            "--horizon-candles",
            "5",
            "--json",
            "-o",
            str(target),
        ]
    )

    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(target.read_text())
    assert printed == saved
    assert printed["mode"] == "live"
    assert printed["intervals"]["1h"]["horizon_candles"] == 5


def test_api_errors_exit_with_code_1(tmp_path, monkeypatch, capsys):
    log = tmp_path / "forecasts.jsonl"
    append_record(log, log_record(10))

    def broken(symbol, interval, outputsize):
        raise RuntimeError("Twelve Data is down")

    monkeypatch.setattr(cli, "fetch_candles", broken)

    assert cli.main(["resolve", "--log", str(log)]) == 1
    assert "Twelve Data is down" in capsys.readouterr().err


def test_replay_prints_a_report(fake_fetch, capsys):
    argv = ["replay", "--symbol", "EUR/USD", "--interval", "4h", "--candles", "300"]

    assert cli.main(argv) == 0

    out = capsys.readouterr().out
    assert "Replay on 300 4h candles of EUR/USD" in out
    assert "one sample every 6 candles" in out
    assert "rule_based" in out
    assert "No LLM, no news" in out
    assert fake_fetch == [("EUR/USD", "4h", 300)]


def test_replay_options_and_json(fake_fetch, capsys):
    cli.main(
        [
            "replay",
            "--candles",
            "300",
            "--horizon-candles",
            "10",
            "--step",
            "5",
            "--json",
        ]
    )

    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "replay"
    assert result["horizon_candles"] == 10
    assert result["step_candles"] == 5


@pytest.mark.parametrize("candles_arg", ["0", "5001", "-5"])
def test_replay_rejects_an_invalid_candle_count(fake_fetch, candles_arg, capsys):
    assert cli.main(["replay", "--candles", candles_arg]) == 2
    assert "--candles must be between 1 and 5000" in capsys.readouterr().err
    assert fake_fetch == []


def test_a_command_is_required():
    with pytest.raises(SystemExit) as exc:
        cli.main([])

    assert exc.value.code == 2


def test_unsupported_symbol_is_rejected():
    with pytest.raises(SystemExit) as exc:
        cli.main(["replay", "--symbol", "BTC/USD"])

    assert exc.value.code == 2
