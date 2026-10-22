from typing import Literal

from pydantic import BaseModel, Field


class ResearchFinding(BaseModel):
    """A single relevant financial news finding."""

    title: str = Field(description="Original article headline")
    published_date: str | None = Field(
        default=None,
        description="Publication date exactly as provided by the source",
    )
    source_url: str = Field(description="Exact source URL")
    relevance: str = Field(
        description="Relevance to the requested financial instrument",
    )
    summary: str = Field(
        description="Concise summary grounded in the article (max 500 characters)",
    )


class ResearchOutput(BaseModel):
    """Structured output produced by the financial research agent."""

    symbol: str
    summary: str = Field(
        description="Concise overview of the collected news (max 800 characters)",
    )
    findings: list[ResearchFinding] = Field(
        description="Up to five relevant news findings",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Important research limitations",
    )


class ReviewResult(BaseModel):
    """Structured verdict produced by the reviewer agent."""

    verdict: Literal["PASS", "NEEDS_REVISION"] = Field(
        description="NEEDS_REVISION if there is at least one critical issue",
    )
    critical_issues: list[str] = Field(
        default_factory=list,
        description="Material factual errors; empty list if none",
    )
    other_issues: list[str] = Field(
        default_factory=list,
        description="Minor problems; empty list if none",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Specific improvements; empty list if none",
    )
