import structlog
import tenacity
from ratelimit import limits, sleep_and_retry
from typing_extensions import TypedDict
from typing import Dict, Any, Optional, Annotated, Callable

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_core.runnables.config import RunnableConfig
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_tavily import TavilySearch
from langgraph.types import Command

from Agent.RAG_System_Class import RAGSystem
from Agent.validation import validate_trip_duration, sanitize_text_for_model
from constants import (
    LLM_MODEL,
    PLAN_TRIP_TEMPERATURE,
    SAFETY_CHECK_TEMPERATURE,
    TAVILY_SEARCH_TEMPERATURE,
    TAVILY_SEARCH_MAX_RESULTS,
    TAVILY_SEARCH_INCLUDE_ANSWER,
    TAVILY_SEARCH_SEARCH_DEPTH,
    TAVILY_SEARCH_QUERY,
    SYSTEM_PROMPT,
    SAFETY_CHECK_PROMPT,
    TRIP_SUMMARY_KEYS,
    MIN_TRIP_DAYS,
    MAX_TRIP_DAYS,
)

log = structlog.get_logger()

# max_retries uses the OpenAI client's built-in exponential backoff, which
# correctly handles Retry-After headers from rate-limit responses.
plan_trip_llm = ChatOpenAI(model=LLM_MODEL, temperature=PLAN_TRIP_TEMPERATURE, max_retries=3)
safety_check_llm = ChatOpenAI(model=LLM_MODEL, temperature=SAFETY_CHECK_TEMPERATURE, max_retries=3)

tavily = TavilySearch(
    max_results=TAVILY_SEARCH_MAX_RESULTS,
    include_answer=TAVILY_SEARCH_INCLUDE_ANSWER,
    search_depth=TAVILY_SEARCH_SEARCH_DEPTH,
    temperature=TAVILY_SEARCH_TEMPERATURE,
)


@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
    stop=tenacity.stop_after_attempt(3),
    reraise=True,
)
def _invoke_tavily(query: str) -> str:
    """Non-generator wrapper around tavily.invoke() so Tenacity retry works correctly."""
    return tavily.invoke(query)

def certified_reducer(a: Optional[bool], b: Optional[bool]) -> Optional[bool]:
    """Safety-first reducer: disqualification (False) is permanent and always wins."""
    if a is False or b is False:
        return False
    return b if b is not None else a

def dict_merge_reducer(
    a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Merge two dicts so parallel tool updates to different keys are combined without data loss."""
    if a is None:
        return b
    if b is None:
        return a
    return {**a, **b}

def scalar_reducer(a: Any, b: Any) -> Any:
    """Prefer the incoming non-None value; fall back to the existing value."""
    return b if b is not None else a

class AgentState(TypedDict, total=False):
    """Structured Agent State"""

    messages: Annotated[list[AnyMessage], add_messages]
    certified: Annotated[Optional[bool], certified_reducer]
    certification_type: Annotated[Optional[str], scalar_reducer]
    destination: Annotated[Optional[str], scalar_reducer]
    trip_month: Annotated[Optional[str], scalar_reducer]
    trip_duration: Annotated[Optional[int], scalar_reducer]
    nitrox: Annotated[Optional[bool], scalar_reducer]
    trip_summary: Annotated[Optional[Dict[str, Any]], dict_merge_reducer]
    total_tokens: Annotated[Optional[int], scalar_reducer]
    total_cost: Annotated[Optional[float], scalar_reducer]


@tool(parse_docstring=True)
def save_trip_summary(
    runtime: ToolRuntime,
    destination: Optional[str] = None,
    trip_month: Optional[str] = None,
    trip_duration: Optional[int] = None,
    certification_type: Optional[str] = None,
    nitrox: Optional[bool] = None,
) -> Command:
    """Save the user's trip preferences to state. Call this progressively whenever
    you learn ANY new information — you do not need all fields at once.
    If certification_type indicates the user is not certified (e.g. 'None', 'N/a',
    'Never dived'), this tool will automatically disqualify them — do NOT also
    call `disqualify_user`.

    Args:
        destination: The dive trip destination (e.g. 'Maldives', 'Great Barrier Reef').
        trip_month: The month of the planned trip (e.g. 'June', 'October').
        trip_duration: The length of the trip in whole days (must be between 1 and 14).
        certification_type: The diver's certification level (e.g. 'Open Water', 'AOW',
            'Divemaster'). Use 'None' or 'N/a' if the user is not certified.
        nitrox: Whether the diver will use Nitrox (enriched air). True for Nitrox,
            False for regular air.
    """
    state = runtime.state
    update_dict = {}

    if destination is not None:
        update_dict["destination"] = destination
    if trip_month is not None:
        update_dict["trip_month"] = trip_month

    if trip_duration is not None:
        if validate_trip_duration(trip_duration):
            update_dict["trip_duration"] = trip_duration
        else:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Error: Invalid trip duration. Must be between {MIN_TRIP_DAYS} and {MAX_TRIP_DAYS} days. Please ask user to adjust.",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )

    if certification_type is not None:
        update_dict["certification_type"] = certification_type
        cert_lower = str(certification_type).strip().lower()
        if cert_lower in ("not certified", "none", "n/a", ""):
            update_dict["certified"] = False
        else:
            update_dict["certified"] = True

    if nitrox is not None:
        update_dict["nitrox"] = nitrox

    old_summary = state.get("trip_summary") or {}
    new_summary = {**old_summary}
    for k in TRIP_SUMMARY_KEYS:
        if k in update_dict:
            new_summary[k] = update_dict[k]

    update_dict["trip_summary"] = new_summary

    is_complete = all(new_summary.get(k) is not None for k in TRIP_SUMMARY_KEYS)
    if is_complete:
        success_msg = (
            f"Successfully updated trip details. All 5 required fields are collected: {new_summary}. "
            "CRITICAL INSTRUCTION: DO NOT ask the user for any special preferences or permission to proceed. "
            "You MUST IMMEDIATELY run `search_tavily` and `validate_safety_with_rag` tools."
        )
    else:
        success_msg = f"Successfully updated trip details. Current known fields in summary: {new_summary}"

    log.info("save_trip_summary_called", updates=new_summary)

    return Command(
        update={
            **update_dict,
            "messages": [
                ToolMessage(
                    content=success_msg,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def disqualify_user(runtime: ToolRuntime) -> Command:
    """Call this tool IMMEDIATELY if the user reveals they are not a certified scuba diver.
    Do NOT call `save_trip_summary` in the same turn."""
    log.info("disqualify_user_called")
    return Command(
        update={
            "certified": False,
            "messages": [
                ToolMessage(
                    content="User has been permanently disqualified. CRITICAL INSTRUCTION: Your final response MUST ONLY be: 'I am afraid I cannot plan or book scuba dives for someone who isn't certified.' Do NOT offer any alternatives, explanations, or next steps. Say exactly and only that.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def search_tavily(runtime: ToolRuntime) -> str:
    """Search the web for scuba diving site recommendations.
    ONLY call this once all 5 trip preference fields are collected (destination, month,
    duration, certification type, nitrox). This tool reads trip details directly from
    state — do NOT pass them as arguments.
    Do NOT add specific site names, liveaboard options, or any detail not explicitly
    provided by the user to the internal query."""
    log.info("search_tavily_called")

    state = runtime.state or {}
    destination = state.get("destination")
    trip_month = state.get("trip_month")
    certification_type = state.get("certification_type")
    trip_duration = state.get("trip_duration")
    nitrox = state.get("nitrox")

    query = TAVILY_SEARCH_QUERY.format(
        destination=destination,
        trip_month=trip_month,
        certification_type=certification_type,
        trip_duration=trip_duration,
        nitrox="Nitrox / Enriched Air" if nitrox else "Regular Air",
    )
    try:
        results = _invoke_tavily(query)
        return sanitize_text_for_model(results)
    except Exception as e:
        log.exception("search_tavily_error", query=query, error=str(e))
        return f"Web search failed: {e}. Please inform the user and ask them to try again."


@tool(parse_docstring=True)
def validate_safety_with_rag(itinerary_draft: str, nitrox: bool) -> str:
    """Validate a draft itinerary against DAN/PADI safety guidelines using RAG.
    ONLY call this once you have drafted a complete, day-by-day itinerary from the
    Tavily search results. Pass the full itinerary text as `itinerary_draft`.
    Your final response to the user MUST be exactly and only this tool's output —
    do not paraphrase, summarise, or add anything else.

    Args:
        itinerary_draft: The complete draft itinerary text to validate, written as a
            full day-by-day dive plan including sites, depths, and any relevant details.
        nitrox: Whether the diver is using Nitrox (enriched air). True for Nitrox,
            False for regular air. Must match what was saved in trip preferences.
    """
    log.info("validate_safety_with_rag_called", nitrox=nitrox)
    gas_context = "Nitrox (enriched air)" if nitrox else "Regular air"
    try:
        rag = RAGSystem.get_instance()
        retrieval_query = (
            f"Safety-check this dive itinerary for {gas_context}. "
            f"Itinerary:\n{itinerary_draft}"
        )
        retrieved_context = rag.retrieve_context(retrieval_query)
    except RuntimeError as e:
        log.exception("validate_safety_rag_error", error=str(e))
        return f"Safety validation unavailable (RAG error: {e}). Present the draft itinerary to the user as-is."

    safety_check_prompt = SAFETY_CHECK_PROMPT.format(
        itinerary_text=itinerary_draft,
        gas_context=gas_context,
        retrieved_context=retrieved_context,
    )

    try:
        result = safety_check_llm.invoke(safety_check_prompt)
        if hasattr(result, "content"):
            return sanitize_text_for_model(result.content)
        return "Safety validation could not be completed."
    except Exception as e:
        log.exception("validate_safety_llm_error", error=str(e))
        return f"Safety validation failed (LLM error: {e}). Present the draft itinerary to the user as-is."


@wrap_model_call
def enforce_tool_sequence(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Middleware to enforce that tools are only available when their prerequisites are met."""
    state = request.state
    summary = state.get("trip_summary") or {}

    is_complete = True
    for key in TRIP_SUMMARY_KEYS:
        if summary.get(key) is None:
            is_complete = False
            break

    if state.get("certified") is False:
        visible_tools = ["disqualify_user"]
    elif not is_complete:
        visible_tools = ["save_trip_summary", "disqualify_user"]
    else:
        visible_tools = [
            "save_trip_summary",
            "disqualify_user",
            "search_tavily",
            "validate_safety_with_rag",
        ]

    relevant_tools = [t for t in request.tools if t.name in visible_tools]
    return handler(request.override(tools=relevant_tools))


agent_tools = [
    save_trip_summary,
    disqualify_user,
    search_tavily,
    validate_safety_with_rag,
]


memory = MemorySaver()

react_graph = create_agent(
    model=plan_trip_llm,
    tools=agent_tools,
    state_schema=AgentState,
    system_prompt=SYSTEM_PROMPT,
    middleware=[enforce_tool_sequence],
    checkpointer=memory,
)


@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
    stop=tenacity.stop_after_attempt(3),
    reraise=True,
)
@sleep_and_retry
@limits(calls=10, period=60)
def invoke_graph(input_state: dict, config: RunnableConfig) -> dict:
    """Invoke the graph synchronously and return the final state."""
    return react_graph.invoke(input_state, config=config)
