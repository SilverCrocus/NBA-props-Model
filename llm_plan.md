The following plan outlines the steps and architecture required to implement the Hybrid Intelligence Betting System, designed to be passed onto an execution-focused LLM or developer for implementation.

The overarching goal of this project is to create a **disciplined, data-driven, and augmented decision-making framework**. The final system must enhance the existing quantitative model by systematically integrating crucial, unstructured qualitative context using a Large Language Model (LLM).

---

## Hybrid Intelligence System Implementation Plan

### SYSTEM ARCHITECTURE DEFINITION

The system must adopt a **"Two-Brain" architecture** with a **Human-in-the-Loop (HITL) supervisor**.

| Component | Role | Function | Technology Focus |
| :--- | :--- | :--- | :--- |
| **Left Brain (Quantitative Engine)** | Reactive Layer / Signal Generator | Continuously scans real-time odds and executes the existing machine learning PRA model. Generates a preliminary, unfiltered list of potential value bets where the projection differs significantly from the market line. | Existing ML Model (Revised), Real-Time Odds API |
| **Right Brain (Qualitative Contextualizer)** | Deliberative Layer / Research Assistant | Triggered **only** for potential bets flagged by the Left Brain. Performs deep, structured qualitative analysis on unstructured data (news, injury reports, social media). | LLM API (e.g., GPT-5), Search/News APIs |
| **Executive Function (Human-in-the-Loop)** | Decision-Maker / Supervisor | Integrates quantitative data and qualitative analysis . Applies domain knowledge and executes the final decision (Place Bet, Pass, or Monitor). | Custom Dashboard Interface |

---

### PHASE 1: DIAGNOSIS AND MODEL REFINEMENT (Prerequisite)

Before integrating any LLM component, the existing quantitative model's performance decay must be clinically diagnosed and fixed.

**Target Outcome:** A robust performance baseline that accounts for real-world market friction.

1.  **Eliminate Data Leakage:** Manually audit the feature generation code for historical games. Implement a **strict chronological split** of the data, ensuring that features like rolling averages or opponent statistics for a given game are calculated **only** using data that occurred *before* that game's date.
2.  **Account for Market Frictions:** Re-run the backtest using the **actual historical odds** from a specific bookmaker for every simulated bet to meticulously account for the **bookmaker's margin ("vig")**. The simulation must also estimate the impact of adverse line movement (**"slippage"**).
3.  **Overfitting Check:** Implement a **strict chronological cross-validation** (e.g., train on 2021-2022 and test on 2023) to assess model generalizability.

---

### PHASE 2: DATA PIPELINE DEVELOPMENT

The system requires parallel, real-time streams of structured and unstructured data.

1.  **Quantitative Data Feeds (for Left Brain):** Secure APIs for:
    *   **Real-Time Betting Odds:** Low-latency feeds for PRA markets from multiple sportsbooks (e.g., SportsDataIO, BallDontLie).
    *   **Historical and Live Player Statistics:** Comprehensive game logs and play-by-play data (e.g., Sportradar, BallDontLie).
2.  **Qualitative Data Feeds (for Right Brain):** Secure APIs/services for:
    *   **Real-Time News and Injury Reports:** Official, time-stamped NBA injury reports and aggregation from reputable sports journalism (e.g., Sportradar, SportsDataIO).
    *   **Web Scraping/Search APIs:** Targeted search functionality to gather articles and press conference summaries (e.g., Serper).
    *   **Social Media APIs:** Access to commentary from verified journalists and insiders for sentiment analysis.

---

### PHASE 3: LLM PROMPT ENGINEERING AND CONSTRAINTS

The LLM's success depends entirely on a **sophisticated, multi-step prompt chain** that forces structured reasoning.

#### Mandatory Prompt Chain Execution

For every potential bet flagged by the Left Brain, the Right Brain LLM must execute the following four steps sequentially:

1.  **Step 1: Information Gathering and Triage:**
    *   **Instruction:** Assume the persona of a sports data analyst. Perform targeted web searches related to the specific player and game (injury status, expected playing time/role, team strategy, specific defensive matchups).
    *   **Output:** Provide **URLs for the top 5 most relevant sources**.
2.  **Step 2: Data Extraction and Structured Summarization:**
    *   **Instruction:** Read the content from the provided URLs. Extract key information.
    *   **Output Format:** Force output into a precise **JSON format**, citing the source URL for each fact. The required fields are:
        `{ 'injury_status': '...', 'expected_minutes': '...', 'key_quotes': ['...', '...'], 'matchup_analysis': '...' }`
3.  **Step 3: Risk and Factor Analysis (Persona-Based):**
    *   **Instruction:** Assume the persona of a **senior risk analyst** for a betting fund. Based *only* on the extracted information, list the top 3 qualitative factors that **SUPPORT** the bet and the top 3 qualitative factors that **ARGUE AGAINST** the bet.
    *   **Output Constraint:** Each factor must be a concise bullet point and **cite its source** (URL).
4.  **Step 4: Final Synthesis and Confidence Score:**
    *   **Instruction:** Synthesize the supporting and opposing factors into a final summary paragraph. Conclude with a justification.
    *   **Output:** Provide a **"Qualitative Confidence Score" on a scale of 1 to 5**, where 1 indicates significant qualitative risks and 5 indicates strong qualitative support.

#### Critical LLM Mitigation Strategies

The LLM must be constrained to mitigate risks inherent to probabilistic models:

| Risk | Mitigation Instruction for LLM |
| :--- | :--- |
| **Factual Inaccuracy (Hallucination)** | **Implement a "fact-checking" step**. After Step 2, prompt the LLM to cross-reference extracted key facts against a structured, reliable data source (e.g., official NBA injury API) before proceeding to Step 3. **Never rely solely on the LLM for critical, verifiable facts**. |
| **Inherited Bias**  | Explicitly instruct the LLM to act as an objective analyst. It must be tasked to **"identify and discount media hype"** and to **"weigh long-term performance data more heavily than short-term trends"**. |
| **Data Privacy**  | **Minimize sensitive information**. Only send generic data (e.g., "Analyze qualitative factors for Player X's PRA line of 21.5") to the third-party LLM API. **Do not send proprietary model outputs** (e.g., the exact predicted edge) in the prompt. |
| **Cost Overrun** | Design prompts for efficiency. Use **smaller, cheaper models** (e.g., GPT-5 mini) for less critical tasks like initial text summarization. Implement API monitoring and budget alerts. |

---

### PHASE 4: SYSTEM INTEGRATION AND VALIDATION

1.  **HITL Dashboard Development:** Create an efficient user interface displaying a dedicated card for each potential bet. This card must present:
    *   The **Quantitative Signal** (Projection, Market Line, Calculated Edge).
    *   The full **Qualitative Analysis** (Summary, Pro/Con Factors, Confidence Score) from the LLM.
    *   **Source Verification:** Clickable links to the source URLs used by the LLM, allowing for manual audit.
2.  **Forward Testing (Paper Trading):** Execute a **non-negotiable** forward testing phase for at least ** one month**.
    *   **Parallel Tracking:** Track the performance of the full hybrid system (ML signal + LLM overlay + Human Decision). Simultaneously, track the performance of the raw quantitative model signals *without* the LLM overlay.
    *   **Evaluation:** Use this parallel data to quantify the specific, tangible, positive expected value added by the qualitative LLM layer, ensuring it exceeds the operational API costs before risking real capital.