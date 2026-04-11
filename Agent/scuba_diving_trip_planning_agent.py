from typing import List, Dict, Optional, Generator, Tuple, Union, Literal, Any
import structlog
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_community.callbacks import get_openai_callback

from Agent.workflow import invoke_react_agent, react_agent
from Agent.validation import validate_user_text
from constants import (
    STATUS_ALL_COLLECTED,
    STATUS_TRIP_GENERATING,
    STATUS_TRIP_VALIDATING,
    TRIP_SUMMARY_KEYS,
    SUMMARY_DISPLAY,
)

log = structlog.get_logger()


def scuba_diving_trip_planning_agent(
    history: List[Dict[str, str]],
    config: RunnableConfig,
) -> Generator[
    Union[
        Tuple[Literal["status"], str],
        Tuple[Literal["trip_summary"], dict],
        Tuple[Literal["token"], str],
        Tuple[Literal["done"], str, dict, Optional[bool]],
    ],
    None,
    None,
]:
    """Stream the scuba diving trip planning agent for UI updates, then finalize with ``invoke_react_agent``.

    Args:
        history (List[Dict[str, str]]): A list of dictionaries representing the chat history
        config (RunnableConfig): Configuration parameters for the LangGraph execution (e.g., thread ID for memory tracking)

    Yields:
        Union[...]: A tuple that can be one of four forms:
            - ("status", message): Interim UX status messages.
            - ("trip_summary", summary_dict): The current extracted trip details.
            - ("token", text): Streaming token output.
            - ("done", response, trip_summary, certified): The final outcome.
    """
    if not history:
        yield ("done", "No history provided.", {}, None)
        return

    latest_prompt_dict = history[-1]
    if latest_prompt_dict["role"] != "user":
        yield ("done", "Expected user message.", {}, None)
        return

    latest_prompt = latest_prompt_dict["content"]

    is_valid, sanitized, error_message = validate_user_text(latest_prompt)
    if not is_valid:
        log.info("validate_input_rejected", reason=error_message)
        yield ("done", error_message or "Invalid input.", {}, None)
        return

    input_state: Dict[str, Any] = {"messages": [HumanMessage(content=sanitized)]}

    last_state: Optional[Dict[str, Any]] = None
    last_trip_summary = None

    try:
        stream = react_agent.stream(
            input_state,
            config=config,
            stream_mode=["values", "messages"],
        )
    except Exception as e:
        log.exception("agent_stream_init_error", error=str(e))
        yield ("done", "Fatal error initializing the agent.", {}, None)
        return

    full_response = ""
    emitted_all_collected = False
    emitted_trip_header = False
    safety_validation_result_ready = False
    safety_tool_call_ids: set[str] = set()

    with get_openai_callback() as cb:
        for event in stream:
            try:
                if not isinstance(event, tuple) or len(event) != 2:
                    continue

                mode, chunk = event

                if mode == "values" and isinstance(chunk, dict):
                    last_state = chunk

                    current_summary = chunk.get("trip_summary")
                    if current_summary and current_summary != last_trip_summary:
                        last_trip_summary = current_summary.copy()
                        yield ("trip_summary", last_trip_summary)
                        is_complete = all(
                            last_trip_summary.get(k) is not None
                            for k in TRIP_SUMMARY_KEYS
                        )
                        if is_complete and not emitted_all_collected:
                            if chunk.get("certified") is not False:
                                yield ("status", STATUS_ALL_COLLECTED)
                                emitted_all_collected = True

                    messages = chunk.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        if isinstance(last_msg, ToolMessage):
                            tool_call_id = getattr(last_msg, "tool_call_id", None)
                            if tool_call_id in safety_tool_call_ids:
                                safety_validation_result_ready = True
                        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                if tc["name"] in ["search_tavily"]:
                                    yield ("status", STATUS_TRIP_GENERATING)
                                elif tc["name"] == "validate_safety_with_rag":
                                    tc_id = tc.get("id")
                                    if tc_id:
                                        safety_tool_call_ids.add(tc_id)
                                    yield ("status", STATUS_TRIP_VALIDATING)

                elif mode == "messages":
                    msg_chunk = chunk[0]
                    if isinstance(msg_chunk, AIMessageChunk) and msg_chunk.content:
                        text_content = msg_chunk.content
                        if isinstance(text_content, str):
                            if (
                                emitted_all_collected
                                and not safety_validation_result_ready
                            ):
                                continue
                            if emitted_all_collected and not emitted_trip_header:
                                summary = last_trip_summary or {}
                                d = summary.get("destination", "")
                                m = summary.get("trip_month", "")
                                dur = summary.get("trip_duration", "")
                                c = summary.get("certification_type", "")
                                n = summary.get("nitrox")
                                trip_header = (
                                    f"{SUMMARY_DISPLAY['destination'][0]} **{d}** | "
                                    f"{SUMMARY_DISPLAY['trip_month'][0]} **{m}** | "
                                    f"{SUMMARY_DISPLAY['trip_duration'][0]} **{dur} days** | "
                                    f"{SUMMARY_DISPLAY['certification_type'][0]} **{c}** | "
                                    f"{SUMMARY_DISPLAY['nitrox'][0]} Nitrox: **{'Yes' if n else 'No'}**\n\n"
                                    f"**Your Dive Trip Plan:**\n\n"
                                )
                                full_response += trip_header
                                yield ("token", trip_header)
                                emitted_trip_header = True

                            full_response += text_content
                            yield ("token", text_content)
            except Exception as e:
                log.exception("agent_stream_event_error", error=str(e))
                continue

    if last_state is None:
        last_state = react_agent.get_state(config).values

    try:
        final_state = invoke_react_agent(None, config)
    except Exception as e:
        log.exception("invoke_react_agent_final_error", error=str(e))
        final_state = last_state

    trip_summary = (
        {k: final_state.get(k) for k in TRIP_SUMMARY_KEYS} if final_state else {}
    )
    certified = final_state.get("certified") if final_state else None

    done_full_response = ""
    if final_state and final_state.get("messages"):
        messages = final_state["messages"]
        last_ai_msgs = [m for m in messages if isinstance(m, AIMessage)]
        if last_ai_msgs:
            final_content = last_ai_msgs[-1].content
            if isinstance(final_content, str):
                done_full_response = final_content

    is_complete = all(trip_summary.get(k) is not None for k in TRIP_SUMMARY_KEYS)
    if is_complete and certified is not False and done_full_response:
        summary = trip_summary
        d = summary.get("destination", "") or ""
        m = summary.get("trip_month", "") or ""
        dur = summary.get("trip_duration", "") or ""
        c = summary.get("certification_type", "") or ""
        n = summary.get("nitrox")
        trip_header = (
            f"{SUMMARY_DISPLAY['destination'][0]} **{d}** | "
            f"{SUMMARY_DISPLAY['trip_month'][0]} **{m}** | "
            f"{SUMMARY_DISPLAY['trip_duration'][0]} **{dur} days** | "
            f"{SUMMARY_DISPLAY['certification_type'][0]} **{c}** | "
            f"{SUMMARY_DISPLAY['nitrox'][0]} Nitrox: **{'Yes' if n else 'No'}**\n\n"
            f"**Your Dive Trip Plan:**\n\n"
        )
        done_full_response = trip_header + done_full_response

    if done_full_response and cb.total_tokens > 0:
        done_full_response += (
            f"\n\n_Tokens: {cb.total_tokens} | Cost: ${cb.total_cost:.4f}_"
        )

    yield ("done", done_full_response, trip_summary, certified)
