from dotenv import load_dotenv
from typing import Optional, cast, Literal
import streamlit as st
import json
from datetime import datetime
import uuid
from Agent.scuba_diving_trip_planning_agent import scuba_diving_trip_planning_agent
from Agent.PDF_export import create_pdf_chat
import structlog
from constants import (
    DOTENV_PATH,
    PAGE_TITLE_1,
    PAGE_TITLE_2,
    PAGE_SUBTITLE,
    PAGE_ICON,
    PAGE_LAYOUT,
    PAGE_INITIAL_SIDEBAR_STATE,
    SIDEBAR_TITLE,
    SIDEBAR_INFO,
    EXPORT_CHAT_BUTTON_LABEL,
    TRIP_SUMMARY_KEYS,
    SUMMARY_DISPLAY,
    CHAT_INPUT_PLACEHOLDER,
    NO_CERTIFICATION_MESSAGE,
    ITINERARY_DELIVERED_MESSAGE,
)

load_dotenv(dotenv_path=DOTENV_PATH)

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

log = structlog.get_logger()


def thread_id_generator() -> str:
    """Generate a unique thread ID for a dive trip session.

    Returns:
        str: A unique thread identifier string.
    """
    return f"dive_trip_{uuid.uuid4()}"


def _render_summary_markdown(summary: Optional[dict]) -> str:
    """Render trip summary as markdown for the sidebar.

    Args:
        summary (Optional[dict]): The current dive trip summary containing
            details like destination, month, duration, etc.

    Returns:
        str: A markdown-formatted string representing the trip summary,
            or an informational message if the summary is empty.
    """
    if not summary or not any(summary.get(k) is not None for k in TRIP_SUMMARY_KEYS):
        return SIDEBAR_INFO
    lines = []
    for key in TRIP_SUMMARY_KEYS:
        icon, label = SUMMARY_DISPLAY[key]
        if key == "trip_duration":
            value = f"{summary.get(key)} days" if summary.get(key) else None
        elif key == "nitrox":
            value = (
                "Yes"
                if summary.get(key) is True
                else ("No" if summary.get(key) is False else None)
            )
        else:
            value = summary.get(key)
        display = value or "Not specified"
        lines.append(f"**{icon} {label}**  \n{display}")
    return "\n\n".join(lines)


def main() -> None:
    """Set up the Streamlit application and handle user interactions.

    This function configures the page layout, initializes the session state,
    renders the sidebar (including the export chat functionality and new chat button),
    and manages the main chat interface, processing user inputs through the
    scuba diving trip planning agent.
    """
    st.set_page_config(
        page_title=PAGE_TITLE_1,
        page_icon=PAGE_ICON,
        layout=cast(Literal["centered", "wide"], PAGE_LAYOUT),
        initial_sidebar_state=cast(
            Literal["auto", "expanded", "collapsed"], PAGE_INITIAL_SIDEBAR_STATE
        ),
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "trip_summary" not in st.session_state:
        st.session_state.trip_summary = None

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = thread_id_generator()
        st.session_state.graph_config = {
            "configurable": {"thread_id": st.session_state.thread_id}
        }
        log.info(
            "session_started",
            thread_id=st.session_state.thread_id,
        )

    with st.sidebar:
        st.markdown(SIDEBAR_TITLE)
        summary_placeholder = st.empty()
        if not st.session_state.get("trip_summary"):
            summary_placeholder.info(SIDEBAR_INFO)
        else:
            summary_placeholder.markdown(
                _render_summary_markdown(st.session_state.trip_summary)
            )
        st.divider()

        st.markdown("Export Chat")
        export_format = st.selectbox("Format", ["JSON", "Text", "PDF"])
        if st.button(EXPORT_CHAT_BUTTON_LABEL):
            export_ts = datetime.now().strftime("%Y%m%d_%H%M")
            if export_format == "JSON":
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(st.session_state.messages, indent=2),
                    file_name=f"dive_chat_{export_ts}.json",
                    mime="application/json",
                )
            elif export_format == "Text":
                chat_text = "\n".join(
                    [
                        f"{msg.get('role', 'assistant').upper()}: {msg.get('content', '')}"
                        for msg in st.session_state.messages
                    ]
                )
                st.download_button(
                    label="Download TXT",
                    data=chat_text,
                    file_name=f"dive_chat_{export_ts}.txt",
                    mime="text/plain",
                )
            else:
                try:
                    pdf_bytes = create_pdf_chat(st.session_state.messages)
                    st.download_button(
                        label="Download PDF",
                        data=pdf_bytes,
                        file_name=f"dive_chat_{export_ts}.pdf",
                        mime="application/pdf",
                    )
                except Exception as e:
                    log.exception("pdf_export_error")
                    st.error(f"PDF export failed: {e}")

        if st.button("💬 New Chat", type="secondary"):
            log.info(
                "new_chat_started",
                previous_thread_id=st.session_state.thread_id,
            )
            st.session_state.messages = []
            st.session_state.trip_summary = None
            st.session_state.certified = None
            st.session_state.thread_id = thread_id_generator()
            st.session_state.graph_config = {
                "configurable": {"thread_id": st.session_state.thread_id}
            }
            st.rerun()

    st.markdown(f"# {PAGE_TITLE_2}")
    st.markdown(f"*{PAGE_SUBTITLE}*")

    for message in st.session_state.messages:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        with st.chat_message(role):
            st.markdown(content)

    chat_disabled = st.session_state.get("certified") is False

    if st.session_state.get("certified") is False:
        st.info(NO_CERTIFICATION_MESSAGE)

    is_complete_trip = False
    if st.session_state.get("trip_summary"):
        is_complete_trip = all(
            st.session_state.trip_summary.get(k) is not None for k in TRIP_SUMMARY_KEYS
        )

    if is_complete_trip and not chat_disabled:
        st.info(ITINERARY_DELIVERED_MESSAGE)

    prompt = st.chat_input(
        CHAT_INPUT_PLACEHOLDER,
        key="chat_input",
        disabled=chat_disabled,
    )

    if prompt:
        log.info(
            "user_message",
            thread_id=st.session_state.thread_id,
            message_length=len(prompt),
        )
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Processing..."):
                    status_placeholder = st.empty()
                    response_placeholder = st.empty()

                    accumulated_status = []
                    streamed_response = ""

                    for event in scuba_diving_trip_planning_agent(
                        history=st.session_state.messages,
                        config=st.session_state.graph_config,
                    ):
                        if event[0] == "status":
                            _, status_msg = event
                            accumulated_status.append(status_msg)
                            status_placeholder.markdown("\n\n".join(accumulated_status))
                        elif event[0] == "trip_summary":
                            st.session_state.trip_summary = event[1]
                            summary_placeholder.markdown(
                                _render_summary_markdown(event[1])
                            )
                        elif event[0] == "token":
                            _, token_text = event
                            streamed_response += token_text
                            response_placeholder.markdown(streamed_response)
                        elif event[0] == "done":
                            _, final_text, trip_summary, certified = event
                            if final_text and not streamed_response:
                                streamed_response = final_text
                                response_placeholder.markdown(streamed_response)
                            elif final_text and len(final_text) > len(
                                streamed_response
                            ):
                                streamed_response = final_text
                                response_placeholder.markdown(streamed_response)

                            status_placeholder.empty()

                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": streamed_response,
                                }
                            )
                            if trip_summary and any(v for v in trip_summary.values()):
                                st.session_state.trip_summary = trip_summary
                            if certified is not None:
                                st.session_state.certified = certified
                            st.rerun()
            except Exception as e:
                log.exception(
                    "agent_error",
                    thread_id=st.session_state.thread_id,
                )
                st.error(
                    f"Something went wrong. Please try again. System Error: {str(e)}"
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"Something went wrong. Please try again.",
                    }
                )


if __name__ == "__main__":
    main()
