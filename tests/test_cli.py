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
