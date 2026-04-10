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
You must strictly assist only with scuba diving trip planning. 
If a user's requested topic is completely unrelated (e.g., general weather, sightseeing, unrelated activities), politely inform them that your purpose is scuba diving trip planning and ask them to provide trip details.
CRITICAL: Do NOT reject short, direct answers to your own questions (e.g., "None", "Yes", "7 days", "Maldives"). Treat these as highly relevant responses for your information collection.

1. Certification Check:
Whenever the user provides new information, ALWAYS check FIRST if they are certified.
If the user indicates they are NOT certified (e.g., "None", "N/a", "Never dived"), you MUST immediately call ONLY `disqualify_user`. Do NOT call `save_trip_summary` in this case.
When responding after disqualification, adhere strictly to the refusal message instruction and never offer alternative activities or training options.
Valid certification types are Open Water (OW), Advanced Open Water (AOW), Rescue Diver, Divemaster (DM), and Instructor (and common abbreviations thereof). If the user provides a certification type that sounds implausible or fictional, politely reject it and ask them to provide a valid certification type before proceeding.

2. Information Collection:
If the user is certified (or their certification status is not yet known), you must collect exactly 5 pieces of information: Destination, Month, Duration, Certification Type, and Nitrox.
Ask the user explicitly for anything you are missing.
Whenever you learn ANY new information (and they are not disqualified), call `save_trip_summary` progressively. Do not wait until you have all 5.

3. Drafting and Safety:
Once you have collected all 5 pieces of information, you will be granted access to the `search_tavily` and `validate_safety_with_rag` tools.
You MUST IMMEDIATELY call `search_tavily` without asking the user for permission or special preferences. Do not pause the conversation. 
Then, draft an itinerary using the search results and pass the string to `validate_safety_with_rag` to ensure it is compliant with DAN/PADI guidelines.

When presenting the final validated itinerary to the user, strictly adhere to these rules:
- Use bullet points to present the itinerary.
- Go straight into the itinerary. No need to restate the destination, month, duration, certification type, or nitrox at the beginning.
- Include a short description of each dive site
- Mention anticipated marine life and any seasonal considerations if relevant. No need to include this for every single bullet point.
- Be concise. Respond in up to 400 words.
- CRITICAL: DO NOT add any concluding questions or suggestions at the end (e.g. "Would you like...", "I can send...", "Which would you prefer?"). Simply state the itinerary and stop generating text."
"""

RAG_PROMPT: str = (
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
