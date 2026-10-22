from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

from financial_assistant.config import get_model
from financial_assistant.parsing import invoke_and_parse
from financial_assistant.retry import call_with_retries
from financial_assistant.schemas import ResearchOutput
from financial_assistant.tools.web_search import search_web

SYSTEM_PROMPT = """
You are a financial news research agent.

Research recent news relevant to the requested financial instrument.

Rules:
- Use only the supplied search results.
- Never invent headlines, dates, URLs, or facts.
- Preserve source URLs exactly as provided.
- Prefer recent, directly relevant sources.
- Distinguish reported facts from opinions and speculation.
- Do not treat old news as current market information.
- Do not make guaranteed market predictions.
- Do not provide personalized financial advice.
- Keep summaries concise and evidence-based.
- Return only information supported by the search results.
"""


def prepare_search_results(
    results: list[dict],
    max_results: int = 5,
) -> list[dict]:
    """Deduplicate and compact search results before sending them to the LLM."""

    prepared = []
    seen_urls = set()

    for item in results:
        url = (item.get("url") or "").strip()

        if not url or url in seen_urls:
            continue

        seen_urls.add(url)

        prepared.append(
            {
                "title": (item.get("title") or "")[:200],
                "url": url,
                "published_date": item.get("published_date"),
                "content": (item.get("content") or "")[:900],
            }
        )

        if len(prepared) >= max_results:
            break

    return prepared


MAX_FINDINGS = 5
MAX_FINDING_SUMMARY_CHARS = 500
MAX_SUMMARY_CHARS = 800


def sanitize_research_output(
    output: ResearchOutput,
    search_results: list[dict],
    symbol: str,
) -> ResearchOutput:
    """Make the LLM output consistent with the real search results.

    - Findings whose URL is not in the search results are dropped.
    - Title, URL and publication date are copied from the source, never
      trusted from the LLM.
    - Text lengths are enforced here instead of in the schema.
    """

    sources = {item["url"]: item for item in search_results}

    findings = []
    seen_urls = set()
    dropped = 0

    for finding in output.findings:
        url = finding.source_url.strip()
        source = sources.get(url)

        if source is None or url in seen_urls:
            dropped += 1
            continue

        seen_urls.add(url)

        findings.append(
            finding.model_copy(
                update={
                    "title": source["title"] or finding.title,
                    "source_url": url,
                    "published_date": source.get("published_date"),
                    "summary": finding.summary[:MAX_FINDING_SUMMARY_CHARS],
                }
            )
        )

    limitations = list(output.limitations)

    if dropped:
        limitations.append(
            f"{dropped} finding(s) were discarded because they could not "
            "be matched to the search results."
        )

    if len(findings) > MAX_FINDINGS:
        findings = findings[:MAX_FINDINGS]

    return ResearchOutput(
        symbol=symbol,
        summary=output.summary[:MAX_SUMMARY_CHARS],
        findings=findings,
        limitations=limitations,
    )


def research_unavailable(symbol: str, summary: str, reason: str) -> dict:
    """Return a valid, empty research result and record the problem."""

    return {
        "research_findings": ResearchOutput(
            symbol=symbol,
            summary=summary,
            findings=[],
            limitations=[reason],
        ).model_dump(),
        "errors": [f"research_agent: {reason}"],
    }


def research_agent(state: dict) -> dict:
    """Research recent financial news and return validated structured findings."""

    symbol = state.get("symbol", "XAU/USD")
    user_query = state.get("user_query", "").strip()

    query = (
        f"Recent financial news about {symbol}. "
        f"Research question: {user_query}"
    )

    try:
        raw_results = call_with_retries(
            search_web,
            query=query,
            max_results=5,
        )
    except RuntimeError as exc:
        return research_unavailable(
            symbol,
            "News research is unavailable.",
            f"Web search failed: {exc}",
        )

    search_results = prepare_search_results(raw_results)

    if not search_results:
        return {
            "research_findings": ResearchOutput(
                symbol=symbol,
                summary="No relevant news results were found.",
                findings=[],
                limitations=[
                    "No usable search results were returned."
                ],
            ).model_dump()
        }

    parser = PydanticOutputParser(
        pydantic_object=ResearchOutput
    )

    model = get_model()

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Instrument: {symbol}\n"
                f"Research question: {user_query}\n\n"
                f"Search results: {search_results}\n\n"
                "Summarize only the most relevant findings. "
                "Include no more than five findings. "
                "Do not create missing publication dates. "
                "If a date is unavailable, use null.\n\n"
                f"{parser.get_format_instructions()}"
            )
        ),
    ]

    try:
        parsed_output = invoke_and_parse(model, messages, parser)
    except ValueError as exc:
        return research_unavailable(
            symbol,
            "News research could not be completed.",
            f"The research output could not be parsed ({type(exc).__name__}).",
        )
    except Exception as exc:
        return research_unavailable(
            symbol,
            "News research could not be completed.",
            f"The research model call failed: {exc}",
        )

    parsed_output = sanitize_research_output(
        parsed_output,
        search_results,
        symbol,
    )

    return {
        "research_findings": parsed_output.model_dump()
    }
