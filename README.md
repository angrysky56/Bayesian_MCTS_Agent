# Bayesian_MCTS_Agent
Adaptive Advanced Tree Search function designed for OpenWeb-UI
https://docs.openwebui.com/

# Advanced Bayesian MCTS Agent for OpenWebUI

**Version:** 0.7.19

**Authors:**
*   angrysky56 ([GitHub](https://github.com/angrysky56))
*   ChatGPT
*   Claude

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

Integrate this script as a "Pipe" within an OpenWebUI instance. Select the pipe (e.g., "advanced_mcts (your_model_name)") when interacting with a model. Configure parameters using the "Valves" options in the chat settings.


