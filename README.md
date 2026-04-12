# Scuba Diving Trip Planning Agent

- **Purpose**: The Scuba Diving Trip Planning Agent helps create diving trip itineraries that are both enjoyable and safe.
- **Target users**: certified recreational scuba divers. Users must be certified to use the agent.
- **Why This Agent is Useful**: unlike generic travel planners, the Agent accounts for scuba diving-specific factors such as:
  - Certification level - a complex (advanced) dive site will not be recommended to a (basic) Open Water diver
  - Seasonal factors, e.g. dry or wet season in tropical locations, etc.
  - Safety - itineraries are reviewed & validated from a safety perspective (using information from reputable sources) to ensure there is no risk of decompression sickness (DCS) due to too many dives per day or flying too soon after diving

## SCR Framework

This project can be framed using **Situation**, **Complication**, and **Resolution** (SCR):

- **Situation**: Certified recreational divers often plan trips across unfamiliar destinations, seasons, and logistics. They need itineraries that match their **certification level**, account for **seasonal conditions**, and ensure the trips are **safe** in terms of daily dive load, surface intervals, and adequate time before flying.
- **Complication**: General-purpose trip planners or LLMs can suggest sites that are too advanced or ignore dive-specific safety constraints. That gap increases **risk** if users treat suggestions as professional dive planning or medical advice.
- **Resolution**: This application addresses the complication by offering a way to construct an itinerary that is both fun and safe. The application is agentic, enabling a seamless conversational experience while leveraging a diverse range of tools in the background. The **Streamlit** chat interface allows the user to make edits and adjustments to the proposed itinerary.

## User Interface & Interaction

1. **User Prompt**: Users describe their desired trip via Streamlit user interface (e.g. "diving in Bali for 1 week in April")
2. **Input Validation & Information Gathering**: The Agent validates user input and ensures all the required information is provided. If the user is not certified, the agent will disable the chat functionality and ask the user to get certified first. If the user does not provide all the required information, the agent will keep asking for it.
3. **Itinerary Generation & Safety Review**: once all the required information is provided, the Agent will:
   - Update the dive trip summary in Streamlit UI
   - Use Tavily search to research dive site options given the trip details provided by the user
   - Generate a draft dive trip itinerary based on the information obtained via Tavily search 
   - Review the draft dive trip itinerary from a safety perspective via Retrieval Augmented Generation (RAG) using two reputable sources: **Professional Association of Diving Instructors (PADI)** Enriched Air (Nitrox) Diver Course Notes and **Divers Alert Network (DAN)** "Flying after Diving" guidance (`validate_safety_with_rag` tool)
4. **Agent Response**: the updated, safety-checked itinerary is then shared with the user via Streamlit UI.
5. **Human in the Loop**: the user can use Streamlit UI to request edits to the itinerary, e.g. make it shorter, change the starting and ending locations, etc.
6. **Export & New Chat**: the user can export the conversation in a variety of formats or start a new chat by clicking respective Streamlit UI buttons.

## Validations
- **User Input Validation**: Validates user input to prevent irrelevant requests, inputs that are too long, trips that are too long, prompt injection (using potential patterns), and other misuse of the agent.
- **Required Information Gathering**: Ensures all the required information is gathered from the user before moving forward. Only certified divers can use the agent.
- **Dive Safety Rules**: Validates no more than 3 dives per day, mandatory rest days after 3+ dive days, and appropriate fly-dive gaps.
- **Environment Verification**: Checks OpenAI & Tavily API keys

## Features
- **Agentic RAG**: RAG functionality (leveraging official PADI and DAN guidelines) has been implemented to validate a generated itinerary from a safety perspective.
- **Token Usage & Cost**: Cumulative tokens used and costs incurred are tracked and displayed in the Streamlit UI when the final itinerary is generated.
- **External API Call**: Tavily API is called to search for potential dive sites based on the diving trip information collected.
- **Retry logic**: retry logic has been implemented throughout the app.
- **Short-term memory**: short-term memory has been implemented using LangChain checkpointer.

## Ethical Considerations

Building and operating this kind of agent raises issues worth keeping explicit:

- **Safety and reliance**: Automated itineraries and RAG retrieval are **assistive**, not authoritative. Users should treat output as a **starting point** and follow **local dive professionals**, **boat briefings**, **medical guidance**, and their own limits.

- **Privacy and data flows**: Trip details and chat are processed by configured providers (**OpenAI** for the model, **Tavily** for search). If **LangSmith** tracing is enabled (`keys.env.example`), prompts and traces may be stored according to that product’s policies. API keys belong only in local `keys.env` (not in git).

- **Certification and honesty**: The app relies on **declared** certification level; it does not verify credentials. Misrepresentation could lead to unsuitable site suggestions despite safeguards.

- **Search and representation**: Web results can skew toward commercial operators or popular sites.

## Installation

### Prerequisites / Dependencies

- Python 3.11
- OpenAI API key
- Tavily API key
- See `pyproject.toml` for full dependency list

### Steps

1. Clone the repository.
2. Install dependencies:

   ```bash
   pip install -e .
   ```

   Or with `uv`:
   
   ```bash
   uv sync
   ```

3. Copy `keys.env.example` to `keys.env` and add your `OPENAI_API_KEY` and `TAVILY_API_KEY`. For optional LangSmith tracing, set `LANGSMITH_API_KEY` and adjust `LANGCHAIN_TRACING_V2` / `LANGCHAIN_PROJECT` as in the example file.

## Usage

Start the Streamlit application:

```bash
streamlit run main.py
```

Access at `http://localhost:8501`. Use the chat interface to plan dive trips.

## Examples

- "Philippines 1 week diving trip, AOW"
- "Egypt for 10 days in July, Open Water with Nitrox"

## Running Tests

```bash
pytest tests/
```

## Type Checking

```bash
mypy .
```

## Code Formatting

```bash
ruff format .
ruff check --fix .
```

## Limitations

- No real-time weather / seasonal conditions integration
- No travel pricing / booking integration
- No long-term memory - each chat is independent