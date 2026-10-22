from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

from financial_assistant.retry import call_with_retries

MAX_PARSE_ATTEMPTS = 2


def invoke_and_parse(
    model,
    messages: list[BaseMessage],
    parser: PydanticOutputParser,
    max_attempts: int = MAX_PARSE_ATTEMPTS,
):
    """Call the model and parse its answer, asking it to fix invalid output.

    Transient API errors are retried. If the answer still cannot be parsed
    after ``max_attempts``, the last parsing error is raised.
    """

    last_error: Exception | None = None

    for _ in range(max_attempts):
        response = call_with_retries(model.invoke, messages)

        try:
            return parser.parse(response.content)
        except ValueError as exc:  # includes OutputParserException
            last_error = exc
            messages = messages + [
                response,
                HumanMessage(
                    content=(
                        "Your previous answer could not be parsed: "
                        f"{exc}\nReturn only valid JSON that follows the "
                        "format instructions."
                    )
                ),
            ]

    raise last_error  # type: ignore[misc]
