import re
import structlog
from typing import Optional, Any
from constants import (
    MAX_INPUT_LENGTH,
    INJECTION_PATTERNS,
    MIN_TRIP_DAYS,
    MAX_TRIP_DAYS,
)

log = structlog.get_logger()


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
            )

    return True, sanitized, None


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

def sanitize_text_for_model(value: Any) -> str:
    """Coerce text into JSON-safe UTF-8 before passing back into model messages."""
    text = value if isinstance(value, str) else str(value)
    return text.encode("utf-8", "ignore").decode("utf-8")