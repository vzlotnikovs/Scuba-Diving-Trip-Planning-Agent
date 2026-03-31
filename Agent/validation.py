import re
import structlog
from typing import Optional, TypedDict
from langchain_openai import ChatOpenAI
from constants import (
    LLM_MODEL,
    MAX_INPUT_LENGTH,
    INJECTION_PATTERNS,
    RELEVANCE_CHECK_TEMPERATURE,
    RELEVANCE_CHECK_PROMPT,
    RELEVANCE_ERROR_MESSAGE,
    MIN_TRIP_DAYS,
    MAX_TRIP_DAYS,
)

log = structlog.get_logger()
_llm_relevance = ChatOpenAI(model=LLM_MODEL, temperature=RELEVANCE_CHECK_TEMPERATURE)


class RelevanceSchema(TypedDict, total=False):
    relevant: bool
    reason: Optional[str]


def check_relevance(sanitized: str) -> tuple[bool, int, float]:
    """Check if the sanitized user input is relevant to scuba diving trip planning.

    Uses an LLM to determine if the query aligns with the agent's purpose. Fails
    open (returns True) if the LLM call encounters an error.

    Args:
        sanitized (str): The sanitized user input string.

    Returns:
        tuple[bool, int, float]: A tuple containing:
            - is_relevant (bool): True if relevant or on error, False otherwise.
            - tokens (int): The number of tokens used in the LLM check.
            - cost (float): The estimated cost of the LLM check in USD.
    """
    from langchain_community.callbacks import get_openai_callback

    try:
        structured_llm = _llm_relevance.with_structured_output(RelevanceSchema)

        with get_openai_callback() as cb:
            result = structured_llm.invoke(
                RELEVANCE_CHECK_PROMPT.format(query=sanitized)
            )
            tokens = cb.total_tokens
            cost = cb.total_cost

        if isinstance(result, dict) and result.get("relevant") is False:
            log.info(
                "relevance_check_failed",
                reason=result.get("reason", "none"),
                tokens=tokens,
                cost=cost,
            )
            return False, tokens, cost

        log.info("relevance_check_passed", tokens=tokens, cost=cost)
        return True, tokens, cost
    except Exception as e:
        log.warning("relevance_check_error", error=str(e))
        return True, 0, 0.0


def validate_user_text(
    content: str,
) -> tuple[bool, Optional[str], Optional[str], int, float]:
    """Validate and sanitize user input text.

    Checks for empty messages, length limits, prompt injection patterns, and
    topic relevance. Consolidates whitespace and strips the input.

    Args:
        content (str): The raw text input from the user.

    Returns:
        tuple[bool, Optional[str], Optional[str], int, float]: A tuple containing:
            - is_valid (bool): True if the input passes all checks, False otherwise.
            - sanitized_content (Optional[str]): The cleaned input string if valid,
              None if invalid.
            - error_message (Optional[str]): A descriptive error message if invalid,
              None if valid.
            - tokens (int): Total token usage from validation checks (e.g., relevance).
            - cost (float): Total estimated cost from validation checks.
    """
    if not content or not content.strip():
        log.info("validate_user_text_rejected", reason="empty_message")
        return False, None, "Empty message. Please try again.", 0, 0.0

    if len(content) > MAX_INPUT_LENGTH:
        log.info(
            "validate_user_text_rejected",
            reason="too_long",
            length=len(content),
        )
        return (
            False,
            None,
            f"Query too long. Please limit to {MAX_INPUT_LENGTH} characters.",
            0,
            0.0,
        )

    sanitized = re.sub(r"\s+", " ", content.strip())

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, sanitized, re.IGNORECASE):
            log.warning(
                "validate_user_text_injection_detected",
                preview=sanitized[:100],
            )
            return (
                False,
                None,
                "Potential prompt injection detected. Please only use this agent for its intended purpose (scuba diving trip planning).",
                0,
                0.0,
            )

    is_relevant, tokens, cost = check_relevance(sanitized)
    if not is_relevant:
        return False, None, RELEVANCE_ERROR_MESSAGE, tokens, cost

    return True, sanitized, None, tokens, cost


def validate_trip_duration(days: Optional[int]) -> bool:
    """Validate that the trip duration is within the allowed range.

    Args:
        days (Optional[int]): The requested trip duration in days.

    Returns:
        bool: True if the duration is valid (or None), False otherwise.
    """
    if days is None:
        return True
    return isinstance(days, int) and MIN_TRIP_DAYS <= days <= MAX_TRIP_DAYS