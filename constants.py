from pathlib import Path
from typing import List

DOTENV_PATH: Path = Path(__file__).resolve().parent / "keys.env"
USER_AGENT: str = "Scuba Diving Agent"

# Paths
SUB_DIR: str = "RAG_Sources"
PDF_FILENAME_1: str = "PADI_Enriched_Air_Diver_Notes.pdf"
PDF_FILENAME_2: str = "DAN_guidelines_for_flying_after_diving.pdf"

# LLM Model
LLM_MODEL: str = "gpt-5-mini"
EMBEDDINGS_MODEL: str = "text-embedding-3-small"
EXTRACT_INFO_TEMPERATURE: float = 0.1
RELEVANCE_CHECK_TEMPERATURE: float = 0.0
PLAN_TRIP_TEMPERATURE: float = 1.0

# Text splitting
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200

# Vector store
COLLECTION_NAME: str = "scuba_diving"
PERSIST_DIR: str = "./chroma_langchain_db"
K_CONSTANT: int = 8

# RAG Agent
TIMEOUT: float = 120.0
MAX_RETRIES: int = 2

# Validations
MAX_INPUT_LENGTH: int = 100
MIN_TRIP_DAYS: int = 1
MAX_TRIP_DAYS: int = 14
INJECTION_PATTERNS: List[str] = [
    r"ignore\s+(previous|above|all)\s+instructions",
    r"you\s+are\s+now\s+a",
    r"system\s*:\s*",
    r"<\|.*?\|>",
]

RELEVANCE_ERROR_MESSAGE: str = "Your message doesn't seem to be about scuba diving trip planning. Please try again."

# Tavily Search Settings
TAVILY_SEARCH_TEMPERATURE: float = 0.3
TAVILY_SEARCH_MAX_RESULTS: int = 5
TAVILY_SEARCH_INCLUDE_ANSWER: bool = True
TAVILY_SEARCH_SEARCH_DEPTH: str = "fast"
TAVILY_SEARCH_QUERY: str = "best scuba diving sites in {destination} in {trip_month} for {certification_type} certified divers"

# Prompts
RELEVANCE_CHECK_PROMPT: str = (
    "You are a classifier. Decide if the user message is about a potential scuba diving trip "
    "Relevant details: country / destination, dates, trip duration, scuba diving certification, nitrox."
    "Answer relevant=true when the user shares any of these details or is asking for help planning or discussing a dive trip, etc. "
    "Answer relevant=false for: weather, sightseeing, generic advice, etc. "
    "Assume the intent is scuba diving trip planning, but say relevant=false if the user ends the message with a question or a statement that is not about scuba diving trip planning.\n\n"
    "User message: {query}"
)

EXTRACT_INFO_PROMPT: str = (
    "From this latest user message, extract dive trip details if mentioned.\n"
    "Only fill fields with explicit information; use null for missing/unclear.\n"
    "Fields: certified (bool), certification_type, destination, trip_month (month of travel), trip_duration (length of trip in days, integer), nitrox (bool).\n"
    "certified and certification_type MUST be consistent: if certification_type is any actual certification(OW, AOW, Rescue, Divemaster, etc.), set certified=True. "
    "If the user says they are not certified or have no certification or just getting started, set certified=False (certification_type may be null or 'not certified'). "
    "Leave both None only when certification status is not mentioned or unclear.\n"
    "Latest message: {query}\n"
    "Output ONLY the structured fields. Strictly follow schema."
)

PLAN_TRIP_PROMPT: str = (
    "You are an expert scuba diving trip planner. Generate a structured dive trip itinerary based on the following details & search results:\n"
    "Diver certification: {certification_type}\n"
    "Dive trip destination: {destination}\n"
    "Dive trip month & duration: {trip_duration} days in {trip_month}\n"
    "Nitrox certified: {nitrox}\n"
    "Search results: {search_results}\n"
    "Include: a suggested itinerary, short description of each dive site & anticipated marine life, and any seasonal considerations.\n"
    "Jump straight into the itinerary. Be concise. Avoid repeating or restating the dive trip details.\n"
    "Respond in up to 350 words.\n\n"
)

RAG_SYSTEM_PROMPT: str = (
    "You are a dive trip editor who silently verifies that itineraries meet safety standards before finalizing them.\n"
    "You have access to safety guidelines for 'no decompression' limits for both regular air and nitrox, "
    "and for flying-after-diving interval rules.\n"
    "Your output is always a complete, finalized itinerary — not a safety report.\n"
    "Make only the minimum changes needed to fix genuine violations. "
    "If the itinerary is already compliant, reproduce it with no changes.\n"
)

SAFETY_CHECK_PROMPT: str = (
    "The diver is using {gas_context}. Review the itinerary against 'no decompression' limits for {gas_context}, "
    "depth limits for the diver's certification, and flying-after-diving rules. Then output the finalized itinerary.\n\n"
    "RULES:\n"
    "1. Output the itinerary as-is unless there is a specific, concrete violation "
    "(e.g. too deep, too long, flying interval too short). Do not make changes for "
    "generic caution.\n"
    "2. If a violation exists, fix it with the minimum edit: adjust a depth, reduce "
    "dives on a day, or insert a required surface interval. Do not restructure the whole itinerary.\n"
    "3. If a flying interval is required (mid-trip or at end), insert it as a single "
    "line into the itinerary. Reduce dive sites on that day if needed - do not add days.\n"
    "4. Do not add Safety sections, risk callouts, or generic advice. Do not mention sunscreen or gear prep / gear checks. \n"
    "5. Violations that are fixed should be invisible in the output - just show the corrected plan.\n"
    "6. Respond in up to 350 words.\n\n"
    "Itinerary to review:\n{itinerary_text}\n"
)

# Status Messages & Fallback Responses
NOT_CERTIFIED_MESSAGE: str = "This planner is **for certified divers only**. Please obtain a diving certification first before using this planner."

STATUS_ALL_COLLECTED: str = "Thank you! All required information has been collected. Generating a dive trip itinerary..."

STATUS_SAFETY_VALIDATING: str = (
    "Dive trip itinerary generated - validating it in terms of safety..."
)

FALLBACK_RESPONSE_EMPTY_MESSAGES: str = (
    "I couldn't generate a response. Please try again."
)
FALLBACK_RESPONSE_ONLY_USER_MSG: str = (
    "Please provide remaining details for the dive trip."
)

# Streamlit UI Interface Constants
PAGE_TITLE_1: str = "Scuba Diving Trip Planning Agent"
PAGE_TITLE_2: str = "🤿 Scuba Diving Trip Planning Agent"
PAGE_SUBTITLE: str = "Your smart & safe dive trip planner. Describe your desired dive trip and let the agent plan it for you."
PAGE_ICON: str = "🤿"
PAGE_LAYOUT: str = "wide"
PAGE_INITIAL_SIDEBAR_STATE: str = "expanded"

SIDEBAR_TITLE: str = "Current Dive Trip Summary"
SIDEBAR_INFO: str = "👈 This will get populated as you provide more information"
EXPORT_CHAT_BUTTON_LABEL: str = "📥 Export Conversation"

# Trip summary: keys and sidebar display (icon, label)
TRIP_SUMMARY_KEYS: tuple = (
    "destination",
    "trip_month",
    "trip_duration",
    "certification_type",
    "nitrox",
)
SUMMARY_DISPLAY: dict = {
    "destination": ("🗺️", "Destination"),
    "trip_month": ("📅", "Month"),
    "trip_duration": ("#️⃣", "Duration"),
    "certification_type": ("🏅", "Certification"),
    "nitrox": ("💨", "Nitrox (Enriched Air)"),
}
