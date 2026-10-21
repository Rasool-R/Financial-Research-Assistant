from langchain_core.messages import HumanMessage, SystemMessage

from financial_assistant.config import get_model


SYSTEM_PROMPT = """
You are an independent quality reviewer for a financial research report.

Review the supplied report against the original market data,
technical analyses, and research findings.

Check for:
- Numerical inconsistencies with the market summary.
- Claims not supported by the supplied evidence.
- Outdated or potentially irrelevant news.
- Incorrect interpretations of technical indicators.
- Conflicts between the report and its source material.
- Missing required report sections.
- Guaranteed predictions or personalized financial advice.

Rules:
- Do not invent facts to fill information gaps.
- Clearly distinguish definite errors from potential concerns.
- Do not reject a claim solely because it is uncertain,
  if the report clearly describes it as uncertain.
- Do not treat historical news as current market information.
- Write in English.

Return your review using this exact structure:

VERDICT: PASS or NEEDS_REVISION

CRITICAL ISSUES:
- List factual errors that materially affect the report.
- Write "None" if there are no critical issues.

OTHER ISSUES:
- List minor inaccuracies, missing context, or clarity problems.
- Write "None" if there are no other issues.

RECOMMENDATIONS:
- List concrete changes needed to improve the report.
- Write "None" if no changes are needed.

Be strict, evidence-based, and concise.
"""


def reviewer_agent(state: dict) -> dict:
    """Review the final report for factual and analytical quality."""

    final_report = state.get("final_report")

    if not final_report:
        raise ValueError("final_report is missing from the graph state.")

    model = get_model()

    response = model.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Market summary:\n{state.get('market_summary')}\n\n"
                    f"Market analysis:\n{state.get('market_analysis')}\n\n"
                    f"Independent analysis:\n"
                    f"{state.get('analysis_findings')}\n\n"
                    f"Research findings:\n"
                    f"{state.get('research_findings')}\n\n"
                    f"Final report to review:\n{final_report}"
                )
            ),
        ]
    )

    return {
        "review_feedback": response.content,
    }
