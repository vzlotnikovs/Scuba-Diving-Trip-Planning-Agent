from typing import List, Dict, Optional, Generator, Tuple, Union, Literal, Any, cast
import tenacity
import structlog
from ratelimit import limits, sleep_and_retry
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig
from Agent.workflow import react_graph
from constants import (
    STATUS_ALL_COLLECTED,
    STATUS_SAFETY_VALIDATING,
    TRIP_SUMMARY_KEYS,
    FALLBACK_RESPONSE_EMPTY_MESSAGES,
    FALLBACK_RESPONSE_ONLY_USER_MSG,
)

log = structlog.get_logger()


def response_and_summary_from_state(
    state: Dict[str, Any],
) -> tuple[str, Dict[str, Any], Optional[bool], Optional[bool]]:
    """Extract final response text and trip summary details from the graph state.

    Args:
        state (Dict[str, Any]): The final state dictionary from the LangGraph run.

    Returns:
        tuple[str, Dict[str, Any], Optional[bool], Optional[bool]]: A tuple containing:
            - response (str): The final AI message content to display.
            - trip_summary (Dict[str, Any]): A dictionary of extracted trip details.
            - certified (Optional[bool]): The user's certification status.
            - workflow_complete (Optional[bool]): True if the safety check finished.
    """
    response: str = FALLBACK_RESPONSE_EMPTY_MESSAGES
    if state.get("messages"):
        last_ai = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
            None,
        )
        if last_ai:
            content = last_ai.content
            response = content if isinstance(content, str) else str(content)
        else:
            response = FALLBACK_RESPONSE_ONLY_USER_MSG

    trip_summary = {k: state.get(k) for k in TRIP_SUMMARY_KEYS}
    certified = state.get("certified")
    workflow_complete = state.get("workflow_complete", False)
    return response, trip_summary, certified, workflow_complete


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


def scuba_diving_trip_planning_agent(
    history: List[Dict[str, str]],
    config: RunnableConfig,
) -> Generator[
    Union[
        Tuple[Literal["status"], str],
        Tuple[Literal["trip_summary"], dict],
        Tuple[Literal["done"], str, dict, Optional[bool], Optional[bool]],
    ],
    None,
    None,
]:
    """Stream the trip-planning graph execution and yield progress updates.

    Initializes the state with the chat history and runs the compiled LangGraph.
    Yields interim status messages (e.g., when info is collected or safety validation starts)
    and ultimately yields the final agent response, summary, and completion status.

    Args:
        history (List[Dict[str, str]]): A list of dictionaries representing the chat
            history, where each dict has 'role' and 'content' keys.
        config (RunnableConfig): Configuration parameters for the LangGraph execution
            (e.g., thread ID for memory tracking).

    Yields:
        Union[...]: A tuple that can be one of three forms:
            - ("status", message): Interim UX status messages.
            - ("trip_summary", summary_dict): The current extracted trip details.
            - ("done", response, trip_summary, certified, workflow_complete): The final outcome.
    """
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")  # NEW

    input_state: Dict[str, Any] = {
        "messages": [
            HumanMessage(content=m["content"])
            if m["role"] == "user"
            else AIMessage(content=m["content"])
            for m in history
        ]
    }

    typed_input_state = cast(Any, input_state)

    last_state: Optional[Dict[str, Any]] = None

    try:
        stream = react_graph.stream(
            typed_input_state,
            config=config,
            stream_mode=["updates", "values"],
        )
        multi_mode = True
    except TypeError:
        stream = react_graph.stream(
            typed_input_state, config=config, stream_mode="updates"
        )
        multi_mode = False

    log.info(
        "agent_stream_started",
        thread_id=thread_id,
        history_length=len(history),
        multi_mode=multi_mode,
    )

    for chunk in stream:
        if multi_mode and isinstance(chunk, tuple) and len(chunk) == 2:
            mode, data = chunk
        elif isinstance(chunk, dict) and "type" in chunk and "data" in chunk:
            mode = chunk["type"]
            data = chunk["data"]
        else:
            mode = "updates"
            data = chunk

        if mode == "values":
            last_state = data
            continue

        if mode != "updates" or not isinstance(data, dict):
            continue

        for node_name, update in data.items():
            if node_name == "router" and isinstance(update, dict):
                if update.get("next_node") == "update_trip_summary":
                    log.info("agent_all_info_collected", thread_id=thread_id)  # NEW
                    yield ("status", STATUS_ALL_COLLECTED)
            elif node_name == "update_trip_summary" and isinstance(update, dict):
                if "trip_summary" in update:
                    log.info(
                        "agent_trip_summary_updated",
                        thread_id=thread_id,
                        summary=update["trip_summary"],
                    )
                    yield ("trip_summary", update["trip_summary"])
            elif node_name == "plan_trip" and isinstance(update, dict):
                log.info("agent_safety_validating", thread_id=thread_id)
                yield ("status", STATUS_SAFETY_VALIDATING)

    if last_state is None:
        log.warning(
            "agent_stream_no_state_fallback_to_invoke",
            thread_id=thread_id,
        )
        try:
            last_state = invoke_graph(typed_input_state, config=config)
        except Exception as e:
            log.exception(
                "agent_invoke_fallback_error",
                thread_id=thread_id,
                error=str(e),
            )
            last_state = typed_input_state

    assert last_state is not None
    response, trip_summary, certified, workflow_complete = (
        response_and_summary_from_state(last_state)
    )
    log.info(
        "agent_done",
        thread_id=thread_id,
        certified=certified,
        workflow_complete=workflow_complete,
    )
    yield ("done", response, trip_summary, certified, workflow_complete)
