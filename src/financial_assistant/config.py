import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Maximum number of report revisions requested by the reviewer.
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "2"))

# Longest allowed report. The report prompt asks for it and the reviewer
# enforces it by counting words in code (an LLM cannot count reliably).
MAX_REPORT_WORDS = 600

# News older than this many days is not sent to the research agent.
NEWS_LOOKBACK_DAYS = int(os.getenv("NEWS_LOOKBACK_DAYS", "7"))

# JSON Lines file where every run's forecast is appended for backtesting.
FORECAST_LOG_PATH = os.getenv("FORECAST_LOG_PATH", "data/forecasts.jsonl")


def get_model() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL")
    # OPENAI_URL_BASE is the legacy (misspelled) name, kept as a fallback.
    base_url = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_URL_BASE")
        or None
    )

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured.")

    if not model_name:
        raise ValueError("OPENAI_MODEL is not configured.")

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )