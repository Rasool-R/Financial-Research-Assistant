from datetime import datetime, timezone


def utc_now() -> datetime:
    """Current time in UTC. Isolated so tests can freeze it."""

    return datetime.now(timezone.utc)
