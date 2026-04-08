import structlog
from typing import TypedDict, Dict, Any, Optional, Annotated, Callable

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_tavily import TavilySearch
from langgraph.types import Command

from Agent.RAG_System_Class import RAGSystem
from Agent.validation import validate_trip_duration
from constants import (
    LLM_MODEL,
    PLAN_TRIP_TEMPERATURE,
    TAVILY_SEARCH_TEMPERATURE,
    TAVILY_SEARCH_MAX_RESULTS,
    TAVILY_SEARCH_INCLUDE_ANSWER,
    TAVILY_SEARCH_SEARCH_DEPTH,
    SAFETY_CHECK_PROMPT,
    TRIP_SUMMARY_KEYS,
)

log = structlog.get_logger()

# Standard LLM for the agent ReAct loop
_llm = ChatOpenAI(model=LLM_MODEL, temperature=PLAN_TRIP_TEMPERATURE)

# Tavily client for the custom tool
_tavily = TavilySearch(
    max_results=TAVILY_SEARCH_MAX_RESULTS,
    include_answer=TAVILY_SEARCH_INCLUDE_ANSWER,
    search_depth=TAVILY_SEARCH_SEARCH_DEPTH,
    temperature=TAVILY_SEARCH_TEMPERATURE,
)


class AgentState(TypedDict, total=False):
    """Structured Agent State"""
    messages: Annotated[list[AnyMessage], add_messages]
    certified: Optional[bool]
    certification_type: Optional[str]
    destination: Optional[str]
    trip_month: Optional[str]
    trip_duration: Optional[int]
    nitrox: Optional[bool]
    trip_summary: Optional[Dict[str, Any]]
    total_tokens: Optional[int]
    total_cost: Optional[float]


@tool
def save_trip_summary(
    runtime: ToolRuntime,
    destination: Optional[str] = None,
    trip_month: Optional[str] = None,
    trip_duration: Optional[int] = None,
    certification_type: Optional[str] = None,
    nitrox: Optional[bool] = None,
) -> Command:
    """Save the user's trip preferences. Call this tool progressively whenever you learn ANY new information. You do not need all fields at once."""
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
                            content="Error: Invalid trip duration. Must be between 1 and 14 days. Please ask user to adjust.",
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

    # Generate new trip_summary locally to push up to state
    old_summary = state.get("trip_summary") or {}
    new_summary = {**old_summary}
    for k in TRIP_SUMMARY_KEYS:
        if k in update_dict:
            new_summary[k] = update_dict[k]

    update_dict["trip_summary"] = new_summary

    is_complete = all(new_summary.get(k) is not None for k in TRIP_SUMMARY_KEYS)
    if is_complete:
        success_msg = (
            f"Successfully updated trip preferences. All 5 required fields are collected: {new_summary}. "
            "CRITICAL INSTRUCTION: DO NOT ask the user for any special preferences or permission to proceed. "
            "You MUST IMMEDIATELY run `search_tavily` and `validate_safety_with_rag` in this exact turn."
        )
    else:
        success_msg = f"Successfully updated trip preferences. Current known fields in summary: {new_summary}"

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
    """Call this tool IMMEDIATELY if the user reveals they are not a certified scuba diver."""
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
def search_tavily(query: str) -> str:
    """Search the web for diving site recommendations. ONLY call this once all trip preferences are gathered."""
    log.info("search_tavily_called", query=query)
    results = _tavily.invoke(query)
    return str(results)


@tool
def validate_safety_with_rag(itinerary_draft: str, nitrox: bool) -> str:
    """Validate a draft itinerary against safety guidelines using RAG. ONLY call this once you have drafted a full itinerary."""
    log.info("validate_safety_with_rag_called", nitrox=nitrox)
    gas_context = "Nitrox (enriched air)" if nitrox else "Regular air"
    rag = RAGSystem().get_instance()
    agent = rag.agent
    
    safety_check_prompt = SAFETY_CHECK_PROMPT.format(
        itinerary_text=itinerary_draft, gas_context=gas_context
    )
    
    result = agent.invoke({"messages": [HumanMessage(content=safety_check_prompt)]})
    result_messages = result.get("messages") or []
    
    if result_messages:
        return result_messages[-1].content if hasattr(result_messages[-1], "content") else str(result_messages[-1])
        
    return "Safety validation could not be completed."


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

    # Strictly enforce logical paths
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

SYSTEM_PROMPT = """You are a scuba diving trip planning assistant. 
Your goal is to collect user preferences, draft an itinerary, and safety-check it.

1. Information Collection:
You must collect exactly 5 pieces of information: Destination, Month, Duration, Certification Type, and whether they want Nitrox.
Ask the user explicitly for anything you are missing.
Whenever you learn ANY new information, ALWAYS call `save_trip_summary` progressively. Do not wait until you have all 5.

2. Certification Check:
If the user indicates they are not certified (or "None", "N/a", etc.), you MUST immediately call `disqualify_user`. When responding, you must adhere strictly to the refusal message instruction and never offer alternative activities or training options.

3. Drafting and Safety:
Once you have collected all 5 preferences, you will be granted access to the `search_tavily` and `validate_safety_with_rag` tools.
You MUST IMMEDIATELY call `search_tavily` without asking the user for permission or special preferences. Do not pause the conversation. 
Then, draft an itinerary using the search results and pass the string to `validate_safety_with_rag` to ensure it is compliant with DAN/PADI guidelines.

When presenting the final validated itinerary to the user, strictly adhere to these rules:
- Include a suggested itinerary, short description of each dive site & anticipated marine life, and any seasonal considerations.
- Be concise. Respond in up to 350 words.
- CRITICAL: DO NOT add any concluding questions or suggestions at the end (e.g. "Would you like...", "I can send...", "Which would you prefer?"). Simply state the itinerary and stop generating text.
"""

memory = MemorySaver()

react_graph = create_agent(
    model=_llm,
    tools=agent_tools,
    state_schema=AgentState,
    system_prompt=SYSTEM_PROMPT,
    middleware=[enforce_tool_sequence],
    checkpointer=memory,
)
