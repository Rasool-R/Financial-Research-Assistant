import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_model() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL")
    base_url = os.getenv("OPENAI_URL_BASE") or None

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