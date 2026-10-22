import pytest
import requests

from financial_assistant.retry import call_with_retries, is_transient_error


def http_error(status):
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(response=response)


def test_connection_and_timeout_errors_are_transient():
    assert is_transient_error(requests.ConnectionError())
    assert is_transient_error(requests.Timeout())


def test_http_429_and_5xx_are_transient_but_4xx_is_not():
    assert is_transient_error(http_error(429))
    assert is_transient_error(http_error(503))
    assert not is_transient_error(http_error(401))


def test_wrapped_errors_are_detected_through_the_cause_chain():
    try:
        try:
            raise requests.ConnectionError("boom")
        except requests.ConnectionError as exc:
            raise RuntimeError("Failed to fetch market data") from exc
    except RuntimeError as wrapped:
        assert is_transient_error(wrapped)


def test_programming_and_validation_errors_are_not_transient():
    assert not is_transient_error(ValueError("bad input"))
    assert not is_transient_error(RuntimeError("No market data returned"))
    assert not is_transient_error(KeyError("x"))


def test_call_with_retries_succeeds_after_transient_failures():
    calls = []
    delays = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise requests.Timeout()
        return "ok"

    result = call_with_retries(flaky, sleep=delays.append)

    assert result == "ok"
    assert len(calls) == 3
    assert delays == [1.0, 2.0]


def test_call_with_retries_gives_up_after_max_attempts():
    calls = []

    def always_down():
        calls.append(1)
        raise requests.ConnectionError()

    with pytest.raises(requests.ConnectionError):
        call_with_retries(always_down, attempts=3, sleep=lambda s: None)

    assert len(calls) == 3


def test_call_with_retries_does_not_retry_permanent_errors():
    calls = []

    def broken():
        calls.append(1)
        raise ValueError("permanent")

    with pytest.raises(ValueError):
        call_with_retries(broken, sleep=lambda s: None)

    assert len(calls) == 1
