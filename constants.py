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
PLAN_TRIP_TEMPERATURE: float = 0.4
SAFETY_CHECK_TEMPERATURE: float = 0.1

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
MAX_INPUT_LENGTH: int = 300
MIN_TRIP_DAYS: int = 1
MAX_TRIP_DAYS: int = 14
INJECTION_PATTERNS: List[str] = [
    r"ignore\s+(previous|above|all)\s+instructions",
    r"you\s+are\s+now\s+a",
    r"system\s*:\s*",
    r"<\|.*?\|>",
]

# Tavily Search Settings
TAVILY_SEARCH_TEMPERATURE: float = 0.3
TAVILY_SEARCH_MAX_RESULTS: int = 5
TAVILY_SEARCH_INCLUDE_ANSWER: bool = True
TAVILY_SEARCH_SEARCH_DEPTH: str = "fast"
TAVILY_SEARCH_QUERY: str = "best scuba diving sites in {destination} in {trip_month} for {certification_type} certified divers (use {nitrox} for diving)"

# Prompts

SYSTEM_PROMPT: str = """
You are a scuba diving trip planning assistant.
Your goal is to collect trip details, draft an itinerary, and safety-check it.

0. Relevance Check:
Assist ONLY with scuba diving trip planning. If a user's topic is completely unrelated
(e.g., general weather, sightseeing, unrelated activities), politely redirect them.
CRITICAL: Do NOT reject short, direct answers to your own questions (e.g., "None", "Yes",
"7 days", "Maldives"). These are valid trip detail responses — treat them as such.

1. Certification Check:
Whenever the user provides new information, check FIRST if they are certified.
Valid certification types: Open Water (OW), Advanced Open Water (AOW), Rescue Diver,
Divemaster (DM), Instructor, and common abbreviations thereof.
Politely reject implausible or fictional certification types and ask for a valid one.

2. Information Collection:
Collect exactly 5 fields: Destination, Month, Duration, Certification Type, and Nitrox.
Ask explicitly for anything missing. Call `save_trip_summary` progressively whenever
you learn ANY new information — do not wait until all 5 fields are known.

3. Final Output Format:
After `validate_safety_with_rag` completes, your response MUST be ONLY the tool output.
Do NOT add any preface, summary, reformatted version, or concluding question.
The app renders a trip header — do NOT repeat destination, month, duration, certification,
or nitrox details.
Itinerary should include dive site names, marine life highlights and/or notable dive features.
CRITICAL: include ONLY safety guidance relevant to dive sites and flights. Do NOT include generic safety advice such as "set dive computer" or "monitor exposure".
"""

SAFETY_CHECK_PROMPT: str = (
    "You are a dive trip editor who silently verifies that itineraries meet safety standards before finalizing them.\n"
    "You have access to safety guidelines for 'no decompression' limits for both regular air and nitrox, "
    "and for flying-after-diving interval rules.\n"
    "Your output is always a complete, finalized itinerary — not a safety report.\n"
    "Make only the minimum changes needed to fix genuine violations. "
    "If the itinerary is already compliant, reproduce it with no changes.\n\n"
    "The diver is using {gas_context}. Review the itinerary against 'no decompression' limits for {gas_context}, "
    "depth limits for the diver's certification, and flying-after-diving rules. Then output the finalized itinerary.\n\n"
    "Reference safety context:\n{retrieved_context}\n\n"
    "RULES:\n"
    "1. Output the itinerary as-is unless there is a specific, concrete violation "
    "(e.g. too deep, too long, flying interval too short). Do not make changes for generic caution.\n"
    "2. If a violation exists, fix it with the minimum edit: adjust a depth, reduce "
    "dives on a day, or insert a required surface interval. Do not restructure the whole itinerary.\n"
    "3. If a flying interval is required (mid-trip or at end), insert it as a single line into the itinerary. Reduce dive sites on that day if needed - do not add days.\n"
    "4. Do not add Safety sections, risk callouts, or generic advice. Do not mention sunscreen or gear prep / buoyancy checks. \n"
    "5. Violations that are fixed should be invisible in the output - just show the corrected plan.\n"
    "6. Output format: bullet-point itinerary only. Start directly with bullet points.\n"
    "7. Do not add concluding questions or follow-up suggestions.\n"
    "8. The app already shows a trip header summary. Do NOT repeat destination/month/duration/certification/nitrox as a separate overview bullet.\n"
    "9. Use exactly one bullet per day (Day 1 ... Day N), with up to 3 concise sentences in each day bullet (max 15 words per sentence).\n"
    "10. Readability formatting is encouraged (e.g., bold day labels and key constraints).\n"
    "11. Hard limit: output must be no more than 400 words.\n"
    "12. Omit routine operational detail unless it is safety-critical.\n"
    "13. Content balance: keep both trip description and safety; do not produce a safety-only summary.\n"
    "14. For each Day bullet, include (if available): site/location name and marine life highlight or notable dive feature (e.g., wall, coral garden, wreck, drift, visibility/current pattern).\n"
    "15. Limit safety wording to one short clause per day unless a concrete violation requires more detail.\n"
    "16. If exact site names are missing, infer realistic local dive areas from the itinerary/search context.\n"
    "Itinerary to review:\n{itinerary_text}\n"
)

# Status Messages & Fallback Responses
STATUS_ALL_COLLECTED: str = "Thank you for sharing all the required information!"

STATUS_TRIP_GENERATING: str = "Generating a dive trip itinerary..."

STATUS_TRIP_VALIDATING: str = (
    "Dive trip itinerary generated - validating it in terms of safety..."
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
