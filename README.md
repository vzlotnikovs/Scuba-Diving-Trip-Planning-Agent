# Scuba Diving Trip Planning Agent

- **Purpose**: The Scuba Diving Trip Planning Agent helps create diving trip itineraries that are both enjoyable and safe.
- **Target users**: certified recreational scuba divers holding such certifications as PADI Open Water (OW), Advanced Open Water (AOW), Enriched Air Diver (Nitrox), etc. Users must be certified to use the agent.
- **Why This Agent is Useful**: unlike generic travel planners, the Agent accounts for scuba diving-specific factors such as:
  - Certification level - a complex (advanced) dive site will not be recommended to a (basic) Open Water diver
  - Seasonal factors, e.g. dry or wet season in tropical locations, etc.
  - Safety - itineraries are reviewed & validated from a safety perspective (using information from reputable sources) to ensure there is no risk of decompression sickness (DCS) due to too many dives per day or flying too soon after diving

## Workflow

1. **User Prompt**: Users describe their desired trip via Streamlit user interface (e.g. "diving in Bali for 1 week in April")
2. **Input Validation & Information Gathering**: The Agent validates user input and ensures all the required information is provided. If the user is not certified, the agent will disable the chat functionality and ask the user to get certified first. If the user does not provide all the required information, the agent will keep asking for it.
3. **Itinerary Generation & Safety Review**: once all the required information is provided, the Agent will:
   - Update the dive trip summary in Streamlit UI ("update_trip_summary" node) 
   - Generate an initial dive trip itinerary based on the information provided using Tavily search ("plan_trip" node) 
   - Review the initial dive trip itinerary from a safety perspective - this is achieved via Retrieval Augmented Generation (RAG) using two reputable sources: **Professional Association of Diving Instructors (PADI)** Enriched Air (Nitrox) Diver Course Notes and **Divers Alert Network (DAN)** "Flying after Diving" guidance ("RAG_check_trip" node)
4. **Agent Response**: the updated, safety-checked itinerary is then shared with the user via Streamlit UI.
5. **Export & New Chat**: the user can export the conversation in a variety of formats or start a new chat by clicking respective Streamlit UI buttons.

## LangGraph Visualization
![LangGraph Workflow](assets/graph.png)

## Validations
- **User Input Validation**: Validates user input to prevent irrelevant requests, inputs that are too long (over 100 characters), trips that are too long (over 2 weeks / 14 days), prompt injection (using potential patterns), and other misuse of the agent.
- **Required Information Gathering**: Ensures all the required information is gathered from the user before moving forward. Only certified divers can use the agent.
- **Dive Safety Rules**: Validates no more than 3 dives per day, mandatory rest days after 3+ dive days, and appropriate fly-dive gaps.
- **Environment Verification**: Checks OpenAI & Tavily API keys

## Optional Tasks Completed

- **Agentic RAG**: RAG functionality (leveraging official PADI and DAN guidelines) has been implemented to validate a generated itinerary from a safety perspective.
- **Token Usage & Cost**: Cumulative tokens used and costs incurred are tracked and displayed in the Streamlit UI when the final itinerary is generated.
- **External API Call**: Tavily API is called to search for potential dive sites based on the diving trip information collected.
- **Retry logic**: retry logic has been implemented for the core agent function scuba_diving_trip_planning_agent.
- **Short-term memory**: short-term memory has been implemented in LangGraph using checkpointer.

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

3. Copy `keys.env.example` to `keys.env` and add your `OPENAI_API_KEY`, `TAVILY_API_KEY`, and `LANGSMITH_API_KEY`.

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
- No travel options / pricing / booking integration
- No long-term memory - each chat is independent

CAPSTONE - more complete MVP
- human in the loop logic
- more agentic - agent should call tools such as Tavily, ADD MORE TOOLS (RAG as a tool, etc.)
- don't initialize LLM every time you call a node
- observability (langgraph / chain)
- long-term memory 
- critic agent?
- deploy on internet