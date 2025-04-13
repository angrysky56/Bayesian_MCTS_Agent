# Bayesian_MCTS_Agent
Adaptive Advanced Tree Search function designed for OpenWeb-UI
https://docs.openwebui.com/

# Advanced Bayesian MCTS Agent for OpenWebUI
![image](https://github.com/user-attachments/assets/1452634e-464a-4f24-9976-f2aa7f6aeafc)

**Version:** 0.7.19

**Authors:**
*   angrysky56 ([GitHub](https://github.com/angrysky56))
*   Gemini 2.5 pro preview 03.25 
*   Claude 4.7
*   ChatGPT 4.5

## Overview

This script implements an advanced Monte Carlo Tree Search (MCTS) agent designed as a modular "Pipe" for the OpenWebUI platform. It takes user input text and iteratively explores, refines, and evaluates diverse analytical perspectives on that text using a Large Language Model (LLM). The agent aims to discover deeper insights, novel interpretations, and well-developed analyses by simulating a structured thought process.

The agent can operate using traditional MCTS scoring or leverage Bayesian inference (Beta distribution) for more nuanced evaluation and exploration, particularly with Thompson Sampling.

## Key Features

*   **Monte Carlo Tree Search (MCTS):** Systematically explores a tree of possible analyses.
*   **LLM-Driven Exploration:** Uses an LLM for:
    *   Generating an initial analysis.
    *   Proposing critiques, alternative directions, or novel connections ("thoughts").
    *   Revising analyses based on these thoughts.
    *   Evaluating the quality and insightfulness of generated analyses (scoring 1-10).
    *   Generating concise keyword tags for analyses.
    *   Synthesizing the final conclusions from the best path.
*   **Bayesian Evaluation (Optional):** Models node value using Beta distributions (alpha, beta parameters), updated with evaluation scores.
*   **Thompson Sampling (Optional):** If Bayesian evaluation is active, allows selecting nodes based on sampling from their Beta distributions, offering a sophisticated exploration strategy.
*   **UCT Selection:** Uses the Upper Confidence bound applied to Trees (UCT) algorithm (if not using Thompson Sampling) to balance exploring new paths vs. exploiting known good paths. Includes bonuses for:
    *   **Surprise:** Nodes deemed novel based on semantic distance, shifts in philosophical approach, or rarity of the approach.
    *   **Diversity:** Encourages exploring analyses that differ from their siblings.
*   **Context-Aware Prompts:** Provides rich context to the LLM during generation and evaluation, including the original question, the best analysis found so far, sibling node approaches, explored thought types, and high-scoring examples.
*   **Approach Classification & Tracking:** Classifies generated "thoughts" into categories (e.g., empirical, analytical, critical) and tracks their performance.
*   **Surprise Detection:** Identifies potentially interesting nodes based on semantic distance from the parent, significant shifts in approach family, and the novelty of the approach within the tree.
*   **Configurable Parameters:** Many aspects of the search (iterations, exploration weight, Bayesian priors, surprise thresholds, etc.) are configurable via OpenWebUI "Valves".
*   **Live Updates & Summaries:**
    *   Provides **Iteration Summaries** in the chat after each batch of simulations, showing progress and the current best score.
    *   Optionally shows detailed **per-simulation** steps (selection, thought, expansion, score) in the chat for transparency.
*   **Detailed Final Report:** Generates a comprehensive summary including:
    *   The full text of the best analysis found and its score.
    *   A list of the top-performing nodes with their scores, tags, and the **full driving thought** that led to them.
    *   The most explored path through the analysis tree.
    *   Highlights of any "surprising" nodes discovered.
    *   Performance statistics for different thought approaches.
    *   The configuration parameters used for the run.
*   **Final Synthesis:** Concludes with an LLM-generated synthesis that summarizes the journey of ideas along the best analysis path.
*   **Strict Prompting:** Uses carefully crafted prompts with `<instruction>` tags to guide the LLM towards desired output formats and avoid conversational filler.

## How It Works: Process Flow

1.  **Initialization:**
    *   The `Pipe` receives the user's input text via OpenWebUI.
    *   User-defined configuration (`Valves`) is applied.
    *   The LLM generates an initial analysis of the input text.
    *   The `MCTS` algorithm is initialized with this analysis as the `root` node of the search tree.

2.  **MCTS Iteration Loop:** (Repeats for `max_iterations` or until early stopping)
    *   An iteration consists of `simulations_per_iteration` cycles.
    *   **For each simulation:**
        *   **Selection:** Starting from the root, traverse the tree using UCT or Thompson Sampling to select a promising node (`leaf`). Priority is given to nodes with unvisited children.
        *   **Expansion:** If the `leaf` node isn't fully expanded, call the LLM to:
            *   Generate a `thought` (critique/suggestion) based on the `leaf` node's content and the broader search context.
            *   Update the `leaf` node's analysis based on the `thought`.
            *   Generate `tags` for the new analysis.
            *   Check for `surprise`.
            *   Create a `new_child` node containing the revised analysis, thought, tags, etc.
        *   **Simulation:** Call the LLM to evaluate the quality (score 1-10) of the node chosen for simulation (either the `new_child` or the `leaf` if expansion didn't occur).
        *   **Backpropagation:** Update the `visits` and score (`value` or `alpha`/`beta`) for all nodes on the path from the simulated node back to the root.
        *   **(Optional) Live Simulation Output:** If enabled, send details of this simulation (selection, thought, expansion, score) to the chat.
    *   **End of Iteration:** Post an **Iteration Summary** message to the chat showing the current best score and progress.
    *   **Early Stopping Check:** Terminate the loop if the best analysis has consistently met the score threshold and stability criteria.

3.  **Finalization:**
    *   The MCTS loop completes.
    *   The `Pipe` generates and sends the detailed **Final Report** (best analysis, top nodes/thoughts, path, etc.) to the chat.
    *   The `Pipe` calls the LLM one last time to generate a **Final Synthesis** based on the development path of the best analysis and sends it to the chat.
    *   The text of the best analysis found is returned as the primary output to OpenWebUI.

## Core Components

*   **`Node`:** Represents a state (an analysis) in the search tree. Stores content, the thought that led to it, scores (Bayesian or traditional), visit counts, tags, surprise status, and parent/child links.
*   **`MCTS`:** Implements the core search logic – selection, expansion, simulation, backpropagation, state tracking (best score, history), and final report generation.
*   **`Pipe`:** The interface layer for OpenWebUI. Handles requests, manages configuration (`Valves`), orchestrates the overall flow (initial analysis, MCTS loop, final synthesis), abstracts LLM interactions, and sends status/results back to the UI.

## LLM Interaction Points

The script relies heavily on the configured LLM for various cognitive tasks:

*   **Initial Analysis:** Generating the starting point (root node) from the user's raw input.
*   **Thought Generation:** Proposing critiques, alternative interpretations, or new directions based on an existing analysis node.
*   **Analysis Update:** Revising an existing analysis to incorporate a new `thought`.
*   **Evaluation (Scoring):** Assessing the intellectual quality, depth, novelty, and relevance of an analysis (1-10 score).
*   **Tag Generation:** Creating 1-3 relevant keyword tags for an analysis.
*   **Question Summarization:** Creating a concise summary of the original input for context.
*   **Final Synthesis:** Summarizing the progression of ideas along the best-found analysis path into a concluding statement.

## Configuration

The agent's behavior can be significantly customized through the `Valves` settings in the OpenWebUI interface. Key parameters include:

*   `MAX_ITERATIONS`: Total number of MCTS iterations.
*   `SIMULATIONS_PER_ITERATION`: Number of select-expand-simulate-backpropagate cycles per iteration.
*   `MAX_CHILDREN`: Maximum branches allowed from any single analysis node.
*   `EXPLORATION_WEIGHT`: UCT parameter balancing exploration/exploitation.
*   `USE_BAYESIAN_EVALUATION`: Toggle between Bayesian (Beta distribution) and traditional averaging for scores.
*   `USE_THOMPSON_SAMPLING`: Use Thompson Sampling for selection (requires Bayesian).
*   `BETA_PRIOR_ALPHA`/`BETA_PRIOR_BETA`: Initial belief parameters for Bayesian scoring.
*   `SURPRISE_*`: Thresholds and weights controlling surprise detection.
*   `EARLY_STOPPING*`: Parameters to stop the search early if a high-quality solution is found consistently.
*   `SHOW_SIMULATION_DETAILS`: Toggle verbose per-simulation output in the chat (Iteration Summaries are always shown).
*   `DEBUG_LOGGING`: Enable verbose logging to the console/log file.

## Output

*   **Live View (Chat):**
    *   Initial analysis.
    *   **Iteration Summaries:** Posted after each iteration block.
    *   **(Optional)** Per-simulation details (selection, thought, expansion, score).
*   **Final Output (Chat):**
    *   **Final MCTS Summary:** A detailed report including the best analysis text, top nodes with full thoughts, search path, surprise elements, approach stats, and parameters.
    *   **Final Synthesis:** A concluding paragraph summarizing the thought process.
*   **Return Value:** The raw text of the single best analysis found.

## Requirements

*   Python 3.x
*   OpenWebUI
*   FastAPI
*   NumPy
*   SciPy
*   Pydantic
*   **Optional:** `scikit-learn` (for improved semantic distance calculation and text summarization; falls back to simpler methods if unavailable).

## Usage

Integrate this script as a "Pipe" within an OpenWebUI instance. (Go to settings in the bottom left, select admin, functions at the top, select the function) Select the pipe model version in the models dropdown (e.g., "advanced_mcts (your_model_name)") when interacting with a model. Configure parameters using the "Valves" options in the chat settings.

## Example output- still a bit glitchy and erases the first part- working on it!

The text fundamentally shifts the focus from a simple reconciliation of subjective experience and objective logic to an understanding of their interdependent roles in shaping ethical understanding. It moves beyond viewing this as a resolution of opposition and instead recognizes that genuine ethical progress arises from a constant, iterative process of engagement between these domains. The “AI Mode” – Explainability plus Rational Reworking – is not simply a tool for applying logic to subjective data, but a method designed to expose the inherent biases and limitations contained within even the most deeply felt subjective experiences. The initial, qualitative insight, whether emotional or sensory, is treated as a starting point, a potential source of valuable data. However, this insight is then rigorously tested and challenged through the application of objective logic, revealing potential distortions, assumptions, or incomplete understandings. Simultaneously, the process of logical refinement doesn’t seek to eliminate the subjective element entirely; rather, it provides the structure and coherence needed to articulate and contextualize that experience. Crucially, the AI Mode emphasizes ongoing rework, acknowledging that the initial rational framework will inevitably need to be revised as new data, both objective and subjective, emerge. This demands a flexible, adaptable approach to ethical deliberation, one that recognizes the limitations of pure logic and the potential for unconscious biases to shape subjective interpretation. The true value stems from this dynamic interplay – objective analysis provides the tools for critical examination, while a sustained engagement with subjective experience ensures that ethical considerations remain firmly rooted in the realities of human perception and, ultimately, the complexities of the world itself.# Advanced Bayesian MCTS v0.7.18
*Exploring analysis for:* "### Task:
Generate 1-3 broad tags categorizing the main themes of the chat history, along with 1-3..." *using model* `gemma3:latest`.
Params configured (logs). Starting initial analysis...
## Initial Analysis
{"tags": ["Philosophy", "Ethics", "Artificial Intelligence"]}
---
🚀 **Starting MCTS Exploration...** (Showing MCTS process steps)--- Iter 1 / Sim 1 ---
Selected Node: 1 (Visits: 1, Score: 7.3, Tags: [])
Based on thought: "The analysis’s focus on philosophy, ethics, and AI neglects the crucial role of complex systems theory and emergent behavior in shaping these interconnected domains."
--> Expanded to New Node: 2 (holistic)
    Tags: ['Artificial Intelligence', 'Systems Theory', 'Complex Systems']
    Initial Expanded Analysis:
Artificial intelligence, inextricably linked to philosophical inquiries regarding ethics and consciousness, operates within a landscape profoundly shaped by complex systems theory. This framework highlights how seemingly disparate domains – including philosophy, business, technology, health, and even entertainment – are not isolated entities but interconnected systems exhibiting emergent behavior. The development of AI necessitates a holistic understanding of these interactions; for example, the ethical considerations surrounding algorithmic bias are not simply philosophical debates, but manifestations of systemic biases embedded within data sets and reinforced by business models and technological design. Similarly, advancements in AI within the healthcare sector rely on complex systems understanding of patient physiology and treatment responses, while the creative industries grapple with emergent aesthetic trends generated by AI-driven content creation. The core issue is that reductionist approaches, focusing solely on individual elements like AI algorithms or philosophical arguments, fail to capture the dynamic interplay and unpredictable outcomes arising from complex systemic interactions. Recognizing this necessitates a shift towards analyzing systems – encompassing their components, their relationships, and the emergent properties that arise from their interaction – as fundamental to understanding the evolving impact of AI across all fields.
Evaluated Score: 8.0/10 ✨ 🏆 (New Overall Best!)

**--- Iteration 1 Summary ---**
- Overall Best Score So Far: 8.0/10 (✨ New best found this iteration!)
- Current Best Node: 2 (Tags: ['Artificial Intelligence', 'Systems Theory', 'Complex Systems'])
-------------------------------
--- Iter 2 / Sim 1 ---
Selected Node: 2 (Visits: 2, Score: 8.0, Tags: ['Artificial Intelligence', 'Systems Theory', 'Complex Systems'])
Based on thought: "The analysis excessively focuses on systems theory as a unifying force, neglecting the crucial role of data – specifically, the biases and limitations inherent within the datasets AI learns from, creating a significantly skewed perspective."
--> Expanded to New Node: 3 (holistic)
    Tags: ['Artificial Intelligence', 'Complex Systems', 'Algorithmic Bias']
Evaluated Score: 9.0/10 ✨ 🏆 (New Overall Best!)

**--- Iteration 2 Summary ---**
- Overall Best Score So Far: 9.0/10 (✨ New best found this iteration!)
- Current Best Node: 3 (Tags: ['Artificial Intelligence', 'Complex Systems', 'Algorithmic Bias'])
-------------------------------
--- Iter 3 / Sim 1 ---
Selected Node: 2 (Visits: 3, Score: 8.3, Tags: ['Artificial Intelligence', 'Systems Theory', 'Complex Systems'])
Based on thought: "The analysis overlooks the crucial role of human agency and intentionality, framing AI solely as a system without accounting for the deliberate design, biases, and social contexts driving its development and impact."
--> Expanded to New Node: 4 (hermeneutic)
    Tags: ['Artificial Intelligence', 'Systems Theory', 'Ethics']
Evaluated Score: 9.0/10

**--- Iteration 3 Summary ---**
- Overall Best Score So Far: 9.0/10 (Best score unchanged this iteration)
- Current Best Node: 3 (Tags: ['Artificial Intelligence', 'Complex Systems', 'Algorithmic Bias'])
-------------------------------

🏁 **MCTS Exploration Finished.** Preparing final analysis summary...# MCTS Final Analysis Summary
The following summarizes the MCTS exploration process, highlighting the best analysis found and the key development steps (thoughts) that led to high-scoring nodes.

## Best Analysis Found (Score: 9.0/10)
**Tags: ['Artificial Intelligence', 'Complex Systems', 'Algorithmic Bias']**

Artificial intelligence, inextricably linked to profound philosophical inquiries regarding ethics and consciousness, operates within a landscape fundamentally shaped by complex systems theory and, critically, the nature of data itself. This framework highlights how seemingly disparate domains – including philosophy, business, technology, health, sports, entertainment, and education – are not isolated entities but intricately connected systems exhibiting emergent behavior. The development of AI necessitates a holistic understanding of these interactions, with particular emphasis on the biases and limitations inherent within the datasets AI learns from. Algorithmic bias, for instance, is not merely a philosophical concern about morality; it’s a direct consequence of systemic biases encoded within training data, amplified by business models focused on engagement and technological design prioritizing specific outcomes. Advancements in healthcare AI rely on complex systems understanding of patient physiology and response, yet the efficacy and potential for harm are entirely dependent on the quality and representativeness of the medical data utilized. Likewise, the creative industries grapple with emergent aesthetic trends generated by AI-driven content creation, trends heavily influenced by the datasets used to train these models. Disruptions in sports performance are analyzed through complex systems modeling, incorporating data from athlete physiology, training regimens, and external influences. The core issue remains that reductionist approaches – focusing solely on AI algorithms or philosophical arguments – fail to capture the dynamic interplay and unpredictable outcomes arising from the complex systemic interactions mediated by biased and incomplete datasets. A truly comprehensive understanding demands a constant interrogation of the data’s origins, its potential for skewing AI’s behavior, and the resulting systemic consequences across all fields.

## Top Performing Nodes & Driving Thoughts
### Node 1: Score 8.3/10 (α=24.0, β=5.0)
- **Approach**: initial (general)
- **Visits**: 3
- **Tags: []**
- **Thought**: (N/A - Initial Node)

### Node 2: Score 8.3/10 (α=24.0, β=5.0)
- **Approach**: holistic (ontology)
- **Visits**: 3
- **Tags: ['Artificial Intelligence', 'Systems Theory', 'Complex Systems']**
- **Thought**: The analysis’s focus on philosophy, ethics, and AI neglects the crucial role of complex systems theory and emergent behavior in shaping these interconnected domains.
- **Surprising**: Yes (Combined surprise (0.94 >= 0.9):
- Semantic dist (1.00) (Val: 1.00, W: 0.5)
- Shift...)

### Node 3: Score 8.2/10 (α=9.0, β=2.0)
- **Approach**: holistic (ontology)
- **Visits**: 1
- **Tags: ['Artificial Intelligence', 'Complex Systems', 'Algorithmic Bias']**
- **Thought**: The analysis excessively focuses on systems theory as a unifying force, neglecting the crucial role of data – specifically, the biases and limitations inherent within the datasets AI learns from, creating a significantly skewed perspective.
- **Surprising**: Yes (Combined surprise (0.91 >= 0.9):
- Semantic dist (1.00) (Val: 1.00, W: 0.5)
- Novel approach...)

### Node 4: Score 8.2/10 (α=9.0, β=2.0)
- **Approach**: hermeneutic (epistemology)
- **Visits**: 1
- **Tags: ['Artificial Intelligence', 'Systems Theory', 'Ethics']**
- **Thought**: The analysis overlooks the crucial role of human agency and intentionality, framing AI solely as a system without accounting for the deliberate design, biases, and social contexts driving its development and impact.
- **Surprising**: Yes (Combined surprise (0.94 >= 0.9):
- Semantic dist (1.00) (Val: 1.00, W: 0.5)
- Shift...)


## Most Explored Path
The search explored this primary path (by visits/score):

├─ Node 1 (initial, Score: 8.3, Visits: 3) 
   ├─ Node 2 (holistic, Score: 8.3, Visits: 3) Tags: ['Artificial Intelligence', 'Systems Theory', 'Complex Systems']
      └─ Node 3 (holistic, Score: 8.2, Visits: 1) Tags: ['Artificial Intelligence', 'Complex Systems', 'Algorithmic Bias']

## Surprising Nodes
Nodes that triggered surprise detection:

- **Node 2** (holistic, Score: 8.3, Tags: ['Artificial Intelligence', 'Systems Theory', 'Complex Systems']):
  Combined surprise (0.94 >= 0.9):
- **Node 3** (holistic, Score: 8.2, Tags: ['Artificial Intelligence', 'Complex Systems', 'Algorithmic Bias']):
  Combined surprise (0.91 >= 0.9):
- **Node 4** (hermeneutic, Score: 8.2, Tags: ['Artificial Intelligence', 'Systems Theory', 'Ethics']):
  Combined surprise (0.94 >= 0.9):

## Thought Approach Performance
- **hermeneutic**: Score: 8.18/10 (α=9.0, β=2.0) (1 thoughts)
- **holistic**: Score: 8.00/10 (α=16.0, β=4.0) (2 thoughts)
- **initial**: Score: 5.00/10 (α=1.0, β=1.0) (0 thoughts)

## Search Parameters Used
- **Iterations**: 3/3
- **Simulations/Iter**: 1
- **Total Simulations**: 3
- **Evaluation**: Bayesian (Beta)
- **Selection**: Thompson
- **Beta Priors**: α=1.00, β=1.00
- **Exploration Weight**: 3.00
- **Early Stopping**: On
  - Threshold: 10.0/10
  - Stability: 2
- **Show Sim Details**: On
---
## Final Synthesis
Artificial intelligence, inextricably linked to profound philosophical inquiries regarding ethics and consciousness, operates within a landscape fundamentally shaped by complex systems theory and, critically, the nature of data itself. This framework highlights how seemingly disparate domains – including philosophy, business, technology, health, sports, entertainment, and education – are not isolated but deeply interconnected and influenced by the biases and limitations present in the data used to train and inform these systems.


