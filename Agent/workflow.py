import structlog
from typing import TypedDict, Dict, Any, Optional, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.callbacks import get_openai_callback
from langchain_core.runnables.config import RunnableConfig
from langchain_tavily import TavilySearch
from Agent.RAG_System_Class import RAGSystem
from Agent.validation import validate_user_text, validate_trip_duration

from constants import (
    LLM_MODEL,
    EXTRACT_INFO_PROMPT,
    EXTRACT_INFO_TEMPERATURE,
    MIN_TRIP_DAYS,
    MAX_TRIP_DAYS,
    SAFETY_CHECK_PROMPT,
    TAVILY_SEARCH_TEMPERATURE,
    TAVILY_SEARCH_MAX_RESULTS,
    TAVILY_SEARCH_INCLUDE_ANSWER,
    TAVILY_SEARCH_SEARCH_DEPTH,
    TAVILY_SEARCH_QUERY,
    PLAN_TRIP_PROMPT,
    PLAN_TRIP_TEMPERATURE,
    NOT_CERTIFIED_MESSAGE,
    STATUS_ALL_COLLECTED,
    TRIP_SUMMARY_KEYS,
    SUMMARY_DISPLAY,
)

log = structlog.get_logger()

_llm_extract = ChatOpenAI(model=LLM_MODEL, temperature=EXTRACT_INFO_TEMPERATURE)
_llm_plan = ChatOpenAI(model=LLM_MODEL, temperature=PLAN_TRIP_TEMPERATURE)
_tavily = TavilySearch(
    max_results=TAVILY_SEARCH_MAX_RESULTS,
    include_answer=TAVILY_SEARCH_INCLUDE_ANSWER,
    search_depth=TAVILY_SEARCH_SEARCH_DEPTH,
    temperature=TAVILY_SEARCH_TEMPERATURE,
)


class AgentState(TypedDict, total=False):
    """Structured Agent State with the following schema:
    - messages: list of messages
    - certified: boolean
    - certification_type: string
    - destination: string
    - trip_month: string
    - trip_duration: int
    - nitrox: boolean
    - next_node: string - internal: "plan_trip", "collect_info", or None → END
    - sanitized_content: string - set by "validate_input"
    - trip_summary: dict - set by update_trip_summary node when all required info collected (for UI)
    """

    messages: Annotated[list[AnyMessage], add_messages]
    certified: Optional[bool]
    certification_type: Optional[str]
    destination: Optional[str]
    trip_month: Optional[str]
    trip_duration: Optional[int]
    nitrox: Optional[bool]
    next_node: Optional[str]
    sanitized_content: Optional[str]
    trip_summary: Optional[Dict[str, Any]]
    workflow_complete: Optional[bool]
    total_tokens: Optional[int]
    total_cost: Optional[float]


def validate_input(state: AgentState) -> dict[str, Any]:
    """Validate user input before any further processing in the graph.

    Uses the shared `validate_user_text` function to check for safety, relevance,
    and valid formatting.

    Args:
        state (AgentState): The current state of the agent graph.

    Returns:
        dict[str, Any]: State updates with either the sanitized content and next
            node to route to, or an error message and routing to END.
    """
    msg = state["messages"][-1]
    content = msg.content if hasattr(msg, "content") else str(msg)
    if not isinstance(content, str):
        content = str(content)
    is_valid, sanitized, error_message, tokens, cost = validate_user_text(content)

    new_tokens = (state.get("total_tokens") or 0) + tokens
    new_cost = (state.get("total_cost") or 0.0) + cost

    if not is_valid:
        log.info("validate_input_rejected", reason=error_message, tokens=tokens)
        return {
            "messages": [AIMessage(content=error_message or "Invalid input.")],
            "next_node": None,
            "total_tokens": new_tokens,
            "total_cost": new_cost,
        }
    log.info("validate_input_passed", tokens=tokens)
    return {
        "sanitized_content": sanitized,
        "next_node": "collect_info",
        "total_tokens": new_tokens,
        "total_cost": new_cost,
    }


def collect_info(state: AgentState) -> dict[str, Any]:
    """Extract structured trip information from the user's message.

    Uses an LLM with structured output to extract fields like destination, month,
    duration, certification, and nitrox preferences.

    Args:
        state (AgentState): The current state of the agent graph.

    Returns:
        dict[str, Any]: State updates containing the newly extracted fields,
            along with accumulated token usage and costs.
    """
    query = state.get("sanitized_content") or next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )

    if not query:
        return {}

    class ExtractSchema(TypedDict):
        certified: Optional[bool]
        certification_type: Optional[str]
        destination: Optional[str]
        trip_month: Optional[str]
        trip_duration: Optional[int]
        nitrox: Optional[bool]

    structured_llm = _llm_extract.with_structured_output(ExtractSchema)

    chain = ChatPromptTemplate.from_template(EXTRACT_INFO_PROMPT) | structured_llm

    with get_openai_callback() as cb:
        extracted = chain.invoke({"query": query})
        tokens = cb.total_tokens
        cost = cb.total_cost

    updates: dict[str, Any] = {k: v for k, v in extracted.items() if v is not None}

    if any(updates.get(k) != state.get(k) for k in updates if state.get(k) is not None):
        updates["messages"] = [
            AIMessage(content="Got it — I've updated your trip details.")
        ]

    updates["total_tokens"] = (state.get("total_tokens") or 0) + tokens
    updates["total_cost"] = (state.get("total_cost") or 0.0) + cost

    if "trip_duration" in updates and updates["trip_duration"] is not None:
        try:
            updates["trip_duration"] = int(updates["trip_duration"])
        except (TypeError, ValueError):
            updates["trip_duration"] = None

    cert_type = updates.get("certification_type") or state.get("certification_type")
    if cert_type is not None:
        cert_lower = str(cert_type).strip().lower()
        if cert_lower in ("not certified", "none", "n/a", ""):
            updates["certified"] = False
        else:
            updates["certified"] = True

    log.info(
        "collect_info_extracted",
        fields=list(updates.keys()),
        tokens=tokens,
        cost=cost,
    )

    return updates


def router(state: AgentState) -> dict[str, Any]:
    """Determine the next step based on the collected information.

    - Checks if all required information has been collected
    - If yes, routes to update_trip_summary
    - If not, prompts the user for missing details
    - Handles early termination if the user is not certified
    - Request a revised trip duration in case it does not meet the min/max duration requirements.

    Args:
        state (AgentState): The current state of the agent graph.

    Returns:
        dict[str, Any]: State updates containing messages to the user and the
            name of the next node to execute in the graph.
    """
    if state.get("certified") is False:
        log.info("router_not_certified")
        return {
            "messages": [AIMessage(content=NOT_CERTIFIED_MESSAGE)],
            "next_node": None,
        }

    cert_type = (state.get("certification_type") or "").lower().strip()
    if cert_type == "not certified":
        log.info("router_not_certified", cert_type=cert_type)
        return {
            "messages": [AIMessage(content=NOT_CERTIFIED_MESSAGE)],
            "certified": False,
            "next_node": None,
        }

    duration = state.get("trip_duration")
    if duration is not None and not validate_trip_duration(duration):
        log.info("router_invalid_duration", duration=duration)
        return {
            "messages": [
                AIMessage(
                    content=f"Only trips between {MIN_TRIP_DAYS} and {MAX_TRIP_DAYS} days (inclusive) are supported. Please request a shorter trip."
                )
            ],
            "trip_duration": None,
            "next_node": None,
        }

    required = [
        "certification_type",
        "destination",
        "trip_month",
        "trip_duration",
        "nitrox",
    ]
    missing = [f for f in required if state.get(f) is None]
    log.info(
        "router_state_check",
        certified=state.get("certified"),
        certification_type=state.get("certification_type"),
        destination=state.get("destination"),
        trip_month=state.get("trip_month"),
        trip_duration=state.get("trip_duration"),
        nitrox=state.get("nitrox"),
        missing_fields=missing,
    )

    if missing:
        labels = (f.replace("_", " ").title() for f in missing)
        question = (
            f"Thanks! Please also confirm the following details: {'; '.join(labels)}."
        )
        return {"messages": [AIMessage(content=question)], "next_node": None}

    return {
        "messages": [AIMessage(content=STATUS_ALL_COLLECTED)],
        "next_node": "update_trip_summary",
    }


def update_trip_summary(state: AgentState) -> dict[str, Any]:
    """Build the trip summary from the current state.

    Extracts key trip details to populate the `trip_summary` state variable,
    which is then used by the UI to display the summary before planning the trip.

    Args:
        state (AgentState): The current state of the agent graph.

    Returns:
        dict[str, Any]: State updates containing the `trip_summary` dictionary.
    """
    trip_summary = {k: state.get(k) for k in TRIP_SUMMARY_KEYS}
    log.info("update_trip_summary", summary=trip_summary)
    return {"trip_summary": trip_summary}


def plan_trip(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Generate the initial dive trip itinerary.

    Uses Tavily search to find relevant information about the destination and
    then prompts an LLM to generate a structured itinerary based on the
    collected details and search results.

    Args:
        state (AgentState): The current state of the agent graph.
        config (RunnableConfig): Configuration parameters for the run.

    Returns:
        Dict[str, Any]: State updates containing the generated itinerary message,
            along with updated cumulative token usage and costs.
    """
    try:
        destination = state.get("destination")
        trip_month = state.get("trip_month")
        certification_type = state.get("certification_type")
        trip_duration = state.get("trip_duration")
        nitrox = state.get("nitrox")

        search_query = TAVILY_SEARCH_QUERY.format(
            destination=destination,
            trip_month=trip_month,
            certification_type=certification_type,
        )

        log.info("plan_trip_search", query=search_query)
        search_results = _tavily.invoke(search_query)
        search_results_str = str(search_results)

        plan_trip_prompt = PLAN_TRIP_PROMPT.format(
            destination=destination,
            trip_month=trip_month,
            trip_duration=trip_duration,
            certification_type=certification_type,
            nitrox=nitrox,
            search_results=search_results_str,
        )

        messages = [HumanMessage(content=plan_trip_prompt)]

        with get_openai_callback() as cb:
            response = _llm_plan.invoke(messages)

            new_tokens = (state.get("total_tokens") or 0) + cb.total_tokens
            new_cost = (state.get("total_cost") or 0.0) + cb.total_cost

            log.info(
                "plan_trip_complete",
                tokens=cb.total_tokens,
                cost=cb.total_cost,
                cumulative_tokens=new_tokens,
                cumulative_cost=new_cost,
            )  # NEW

        trip_header = (
            f"{SUMMARY_DISPLAY['destination'][0]} **{destination}** | "
            f"{SUMMARY_DISPLAY['trip_month'][0]} **{trip_month}** | "
            f"{SUMMARY_DISPLAY['trip_duration'][0]} **{trip_duration} days** | "
            f"{SUMMARY_DISPLAY['certification_type'][0]} **{certification_type}** | "
            f"{SUMMARY_DISPLAY['nitrox'][0]} Nitrox: **{'Yes' if nitrox else 'No'}**\n\n"
        )

        return {
            "messages": [
                AIMessage(
                    content=(
                        f"{trip_header}"
                        f"**Your Dive Trip Plan:**\n\n"
                        f"{response.content}\n\n"
                        f"_Tokens: {new_tokens} | Cost: ${new_cost:.4f}_"
                    )
                )
            ],
            "total_tokens": new_tokens,
            "total_cost": new_cost,
        }

    except Exception as e:
        log.exception("plan_trip_error", error=str(e))
        return {
            "messages": [
                AIMessage(content="Error planning your dive trip. Please try again.")
            ]
        }


def RAG_check_trip(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Validate the suggested itinerary against safety guidelines using RAG.

    Uses a Retrieval-Augmented Generation system to review the LLM-generated
    itinerary against official PADI and DAN guidelines for potential safety issues.

    Args:
        state (AgentState): The current state of the agent graph.
        config (RunnableConfig): Configuration parameters for the run.

    Returns:
        Dict[str, Any]: State updates containing the final, safety-validated
            itinerary message and the `workflow_complete` flag, along with
            final cumulative token usage and costs.
    """
    try:
        all_messages = state.get("messages") or []
        last_ai = next(
            (m for m in reversed(all_messages) if isinstance(m, AIMessage)),
            None,
        )
        itinerary_text = (
            last_ai.content
            if last_ai and hasattr(last_ai, "content")
            else str(last_ai or "")
        )
        nitrox = state.get("nitrox") is True
        gas_context = "Nitrox (enriched air)" if nitrox else "Regular air"

        rag = RAGSystem().get_instance()
        agent = rag.agent

        safety_check_prompt = SAFETY_CHECK_PROMPT
        rag_messages = [
            HumanMessage(
                content=safety_check_prompt.format(
                    itinerary_text=itinerary_text, gas_context=gas_context
                )
            )
        ]

        thread_id = config.get("configurable", {}).get("thread_id", "unknown")
        log.info("RAG_check_trip_started", thread_id=thread_id)

        with get_openai_callback() as cb:
            result = agent.invoke(
                {"messages": rag_messages},
                config=config,
            )

            new_tokens = (state.get("total_tokens") or 0) + cb.total_tokens
            new_cost = (state.get("total_cost") or 0.0) + cb.total_cost

            log.info(
                "RAG_check_trip_complete",
                thread_id=thread_id,
                tokens=cb.total_tokens,
                cost=cb.total_cost,
                cumulative_tokens=new_tokens,
                cumulative_cost=new_cost,
            )

        result_messages = result.get("messages") or []
        if not result_messages:
            return {
                "messages": [
                    AIMessage(
                        content="Dive trip itinerary generated - validating it in terms of safety..."
                    ),
                    AIMessage(content="Safety validation could not be completed."),
                ],
            }
        last = result_messages[-1]
        validation_content = last.content if hasattr(last, "content") else str(last)

        return {
            "messages": [
                AIMessage(
                    content="Dive trip itinerary generated - validating it in terms of safety..."
                ),
                AIMessage(
                    content=(
                        f"**Updated & safety-checked itinerary**\n\n{validation_content}\n\n"
                        f"_Tokens: {new_tokens} | Cost: ${new_cost:.4f}_"
                    )
                ),
            ],
            "workflow_complete": True,
            "total_tokens": new_tokens,
            "total_cost": new_cost,
        }
    except Exception as e:
        log.exception("RAG_check_trip_error", error=str(e))
        return {
            "messages": [
                AIMessage(
                    content="Dive trip itinerary generated - validating it in terms of safety..."
                ),
                AIMessage(content=("Safety validation could not be completed.")),
            ],
            "workflow_complete": True,
        }


builder = StateGraph(AgentState)

builder.add_node("validate_input", validate_input)
builder.add_node("collect_info", collect_info)
builder.add_node("router", router)
builder.add_node("update_trip_summary", update_trip_summary)
builder.add_node("plan_trip", plan_trip)
builder.add_node("RAG_check_trip", RAG_check_trip)

builder.add_edge(START, "validate_input")
builder.add_conditional_edges(
    "validate_input",
    lambda state: state.get("next_node") or END,
    {"collect_info": "collect_info", END: END},
)
builder.add_edge("collect_info", "router")


builder.add_conditional_edges(
    "router",
    lambda state: state.get("next_node") or END,
    {"update_trip_summary": "update_trip_summary", END: END},
)
builder.add_edge("update_trip_summary", "plan_trip")
builder.add_edge("plan_trip", "RAG_check_trip")
builder.add_edge("RAG_check_trip", END)

memory = MemorySaver()
react_graph = builder.compile(checkpointer=memory)
