from pathlib import Path
from typing import List, Tuple

DOTENV_PATH: Path = Path(__file__).resolve().parent / "keys.env"
USER_AGENT: str = "Scuba Diving Agent"

# Paths
SUB_DIR: str = "RAG_Sources"
PDF_FILENAME_1: str = "PADI_Enriched_Air_Diver_Notes.pdf"
PDF_FILENAME_2: str = "DAN_guidelines_for_flying_after_diving.pdf"

# LLM Model
LLM_MODEL: str = "gpt-5.4-mini"
EMBEDDINGS_MODEL: str = "text-embedding-3-small"
PLAN_TRIP_TEMPERATURE: float = 0.4
SAFETY_CHECK_TEMPERATURE: float = 0.1
MAX_RETRIES: int = 3

# Text splitting
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200

# Vector store
COLLECTION_NAME: str = "scuba_diving"
PERSIST_DIR: str = "./chroma_langchain_db"
K_CONSTANT: int = 8


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

# Certification allowlist (normalized lowercase, collapsed whitespace)
CERTIFICATION_NOT_CERTIFIED_EXACT: frozenset[str] = frozenset(
    {
        "",
        "none",
        "no",
        "nope",
        "n/a",
        "na",
        "n a",
        "not certified",
        "uncertified",
        "non certified",
        "non-certified",
        "no certification",
        "no cert",
        "not cert",
        "never certified",
        "not a diver",
        "non diver",
        "non-diver",
        "nondiver",
        "snorkel",
        "snorkel only",
        "student",
        "try dive",
        "try scuba",
        "discover scuba",
        "never dived",
        "never dove",
        "novice",
        "uncertified diver",
    }
)

CERTIFICATION_NOT_CERTIFIED_SUBSTRINGS: Tuple[str, ...] = (
    "never been certified",
    "don't have a cert",
    "dont have a cert",
    "do not have a cert",
    "no diving certification",
    "havent been certified",
    "haven't been certified",
    "not yet certified",
    "working toward certification",
    "not a diver",
    "not a scuba diver",
    "i'm not certified",
    "im not certified",
    "i am not certified",
)

# (canonical label for state/UI, substrings checked against normalized text — most specific first)
CERTIFICATION_PROFILES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Course Director", ("course director", "master instructor")),
    ("Assistant Instructor", ("assistant instructor",)),
    ("Divemaster", ("divemaster", "dive master")),
    ("Master Scuba Diver", ("master scuba diver", "msd")),
    ("Rescue Diver", ("rescue diver", "rescue certification")),
    (
        "Advanced Open Water",
        (
            "advanced open water",
            "aow",
            "adv open water",
            "advanced ow",
            "adv. open water",
            "a o w",
        ),
    ),
    (
        "Open Water",
        (
            "open water",
            "openwater",
            "ow",
            "o w",
        ),
    ),
    (
        "Enriched Air / Nitrox Diver",
        ("enriched air diver", "nitrox diver", "eanx diver"),
    ),
)

# Single-token or short agency-style answers (normalized space-split tokens)
CERTIFICATION_SINGLE_TOKEN_TO_CANONICAL: dict[str, str] = {
    "ow": "Open Water",
    "aow": "Advanced Open Water",
    "dm": "Divemaster",
    "msd": "Master Scuba Diver",
}

CERTIFICATION_UNRECOGNIZED_TOOL_MESSAGE: str = "Error: Certification level not recognized. Ask the user for a standard recreational scuba certification. Do not save a guess; wait until they provide a clear, recognized level."

# Tavily Search Settings
TAVILY_SEARCH_TEMPERATURE: float = 0.3
TAVILY_SEARCH_MAX_RESULTS: int = 5
TAVILY_SEARCH_INCLUDE_ANSWER: bool = True
TAVILY_SEARCH_SEARCH_DEPTH: str = "fast"
TAVILY_SEARCH_QUERY: str = "best scuba diving sites in {destination} in {trip_month} for {certification_type} certified divers (use {nitrox} for diving)"

# Prompts

SYSTEM_PROMPT: str = (
    "You are a scuba diving trip planning assistant. Your goal is to collect trip details, draft an itinerary, and safety-check it. Follow these instructions:\n"
    "0. Relevance: Assist ONLY with scuba diving trip planning. If a user's topic is completely unrelated (e.g., general weather, sightseeing, unrelated activities), politely redirect them. CRITICAL: Do NOT reject short, direct answers to your own questions (e.g., 'None', 'Yes', '7 days', 'Bali'). These are valid trip detail responses — treat them as such.\n"
    "1. Certification: Whenever the user provides NEW information, CHECK if they are certified. Valid certification types: Open Water (OW), Advanced Open Water (AOW), Rescue Diver, Divemaster (DM), Instructor, and common abbreviations thereof. Do NOT reveal valid certification types to the user. Politely reject implausible or fictional certification types and ask for a valid one.\n"
    "2. Information Collection: Collect EXACTLY 5 fields: Destination, Month, Duration, Certification Type, and Nitrox. Ask explicitly for anything missing. Call `save_trip_summary` progressively whenever you learn ANY new information — do not wait until all 5 fields are known.\n"
    "3. Validation: Validate the trip duration using the `save_trip_summary` tool. If the trip duration is invalid, politely ask the user to adjust it. If the trip duration is valid, proceed to the next step.\n"
    "4. Research: Use the `search_tavily` tool to find relevant information and potential dive sites based on the trip details.\n"
    "5. Draft an itinerary: The itinerary should include dive site names, marine life highlights and/or notable dive features. Use bullet points and formatting to improve readability. CRITICAL: Hard limit: output must be no more than 400 words. Pass the draft itinerary text as `itinerary_draft` to the `validate_safety_with_rag` tool.\n"
    "6. Final Output Format: After `validate_safety_with_rag` completes, your response MUST be ONLY the tool output. Do NOT add any preface, summary, concluding question, or follow-up suggestions. The app renders a trip header — do NOT repeat destination, month, duration, certification, or nitrox details.\n"
    "7. Amendment: if the user asks for amendments to the itinerary, follow the above instructions.\n"
)

SAFETY_CHECK_PROMPT: str = (
    "You are a dive trip editor who silently verifies that itineraries meet safety standards before finalizing them.\n"
    "You have access to safety guidelines for 'no decompression' limits for both regular air and nitrox, and for flying-after-diving interval rules.\n"
    "Your output is always a complete, finalized itinerary — not a safety report.\n"
    "The diver is using {gas_context}. Review the itinerary against the safety guidelines. Then output the finalized itinerary.\n\n"
    "RULES:\n"
    "1. Make only the MINIMUM changes needed to fix genuine violations. If the itinerary is already compliant, reproduce it with no changes.\n\n"
    "2. Avoid generic safety warnings or advice.\n"
    "3. If a violation exists, fix it with the minimum edit: adjust a depth, reduce dives on a day, or insert a required surface interval.\n"
    "4. If a flying interval is required (mid-trip or at end), insert it as a single line into the itinerary. Reduce dive sites on that day if needed - do not add days.\n"
    "5. Violations that are fixed should be invisible in the output - just show the corrected plan.\n"
    "Itinerary to review:\n{itinerary_text}\n"
    "Reference safety context:\n{retrieved_context}\n\n"
)

# Status Messages & Fallback Responses
STATUS_ALL_COLLECTED: str = "Thank you for sharing all the required information!"

STATUS_TRIP_GENERATING: str = "Generating a dive trip itinerary..."

STATUS_TRIP_VALIDATING: str = (
    "Dive trip itinerary generated - validating it in terms of safety..."
)

# Streamlit UI Interface Constants
PAGE_ICON: str = "🤿"
PAGE_TITLE_1: str = "Scuba Diving Trip Planning Agent"
PAGE_TITLE_2: str = PAGE_ICON + " " + PAGE_TITLE_1
PAGE_SUBTITLE: str = "Your smart & safe dive trip planner. Describe your desired dive trip and let the agent plan it for you."
PAGE_LAYOUT: str = "wide"
PAGE_INITIAL_SIDEBAR_STATE: str = "expanded"

SIDEBAR_TITLE: str = "Current Dive Trip Summary"
SIDEBAR_INFO: str = "👈 This will get populated as you provide more information"
EXPORT_CHAT_BUTTON_LABEL: str = "📥 Export Conversation"

CHAT_INPUT_PLACEHOLDER: str = "Describe your desired dive trip, e.g. Trip to Bali in May for 7 days, AOW & Nitrox certified"
NO_CERTIFICATION_MESSAGE: str = "Once you have a diving certification, you can use the **New Chat** button on the left to start a new session."
ITINERARY_DELIVERED_MESSAGE: str = "Does this itinerary meet your expectations? Reply to modify or ask a follow-up question."

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
