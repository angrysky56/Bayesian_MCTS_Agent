# -*- coding: utf-8 -*-
"""
title: advanced_bayesian_mcts_stateful
version: 0.8.0

author: angrysky56 (original) + AI Collaboration
author_url: https://github.com/angrysky56
Project Link: https://github.com/angrysky56/Bayesian_MCTS_Agent

Where I found my stored functions, replace ty with your user name:
/home/ty/.open-webui/cache/functions

The way I launch openweb-ui:
DATA_DIR=~/.open-webui uvx --python 3.11 open-webui@latest serve
http://localhost:8080


**How to Use and Test:**

1.  **Save:** Save this code as a Python file (e.g., `advanced_mcts_stateful_pipe.py`) in the Admin panel accesed from your user name in the bottom left, then top- functions tab, where Open WebUI loads custom pipes/functions from.
2.  **Configure `DB_FILE`:** **CRITICAL:** Edit the `DB_FILE` constant near the top of the script to point to a real, writable path.
3.  **Restart Open WebUI:** Ensure Open WebUI picks up the new/updated pipe.
4.  **Select Pipe:** In the Open WebUI chat interface, select the `advanced_mcts_stateful` model/pipe.
5.  **Test Scenarios:**
    *   **New Analysis:** Provide text like "Analyze the risks of AI misalignment." Check the output.
    *   **Continue Analysis:** In the *same chat*, enter a follow-up like "Explore the regulatory aspect further." Observe if it seems to consider the previous context (check logs for "Loaded state..." messages).
    *   **Ask About Last Run:** Ask "What was the score of the last analysis?" or "Summarize the last run." Check if it retrieves and presents the saved state info.
    *   **Ask About Process/Config:** Ask "How do you work?" or "Show config."
    *   **Start New Analysis After Continuation:** Enter "Analyze the benefits of decentralized AI." Check if it starts fresh (intent `ANALYZE_NEW`).
    *   **Check DB File:** Verify that the `mcts_pipe_state.db` file is created and populated after successful analysis runs.
    *   **Toggle Valves:** Change settings like `SHOW_PROCESSING_DETAILS` or `MAX_ITERATIONS` in the UI and see if they affect the run and are reported correctly by `ASK_CONFIG`.

This code provides the architectural foundation. Expect to iterate and refine the state serialization, context injection, and intent handling based on your specific testing results and desired behavior. Good luck!

description: >
  Stateful Advanced Bayesian MCTS v0.8.0:
  - Integrates SQLite database for state persistence across user turns within a chat session.
  - Uses LLM-based intent classification to handle different user requests (analyze new, continue, ask about results, etc.).
  - Implements Selective State Persistence: Saves/Loads learned approach priors, best results, and basic "unfit" markers.
  - Injects loaded state context into MCTS prompts to guide exploration.
  - Maintains core MCTS logic, quiet/verbose modes, and configuration via Valves.

Key improvements in v0.8.0:
- Stateful Analysis: Remembers key aspects of the last MCTS run per chat session.
- Intent Classification: Understands user intent beyond just analysis tasks.
- Guided Continuation: Uses saved state (priors, unfit markers) to inform subsequent MCTS runs.
- Database Persistence: Uses a separate SQLite DB for state management.

Requires:
- User to configure DB_FILE path correctly. Currently broken idea, db path does not matter.
- Original dependencies + standard Python libs (sqlite3, json, datetime).

*** NOTE: This version ensures correct indentation for methods within the Pipe class, which is the likely cause of the 'AttributeError' seen in the logs. ***
"""

from fastapi import Request, Response
import logging
import random
import math
import asyncio
import json
import re
import gc
import sqlite3
import os
from datetime import datetime
from collections import Counter

import numpy as np
from scipy import stats
from numpy.random import beta as beta_sample

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    TfidfVectorizer, ENGLISH_STOP_WORDS, cosine_similarity = None, None, None

import aiohttp

from typing import (
    List,
    Optional,
    AsyncGenerator,
    Callable,
    Awaitable,
    Generator,
    Iterator,
    Dict,
    Any,
    Tuple,
    Set,
    Union,
)
from pydantic import BaseModel, Field, field_validator
from open_webui.constants import TASKS
import open_webui.routers.ollama as ollama
from open_webui.main import app

# ==============================================================================
# <<< NEW: Database Configuration >>>
# !!! IMPORTANT: Set this path to a writable location for the backend process !!!
# DB_FILE = os.path.join(os.path.dirname(__file__), "mcts_pipe_state.db") # Same directory as this script, might not work.
# ----->>>>>>>> CHANGE THIS PATH <<<<<<<<<<-----
DB_FILE = "/home/ty/Repositories/sqlite-db/mcts_state.db"  # My personal path - REPLACE with your own.
# ----->>>>>>>> CHANGE THIS PATH <<<<<<<<<<-----
# ==============================================================================

name = "advanced_mcts_stateful"  # Renamed to reflect statefulness

# --- DEFAULT Global Configuration (Unchanged from 0.7.21) ---
default_config = {
    "max_children": 10,
    "exploration_weight": 3.0,
    "max_iterations": 1,
    "simulations_per_iteration": 10,
    "surprise_threshold": 0.66,
    "use_semantic_distance": True,
    "relative_evaluation": False,  # Note: relative eval not fully implemented in original code
    "score_diversity_bonus": 0.7,
    "force_exploration_interval": 4,
    "debug_logging": False,
    "global_context_in_prompts": True,
    "track_explored_approaches": True,
    "sibling_awareness": True,
    "memory_cutoff": 5,
    "early_stopping": True,
    "early_stopping_threshold": 10,
    "early_stopping_stability": 2,
    "surprise_semantic_weight": 0.6,
    "surprise_philosophical_shift_weight": 0.3,
    "surprise_novelty_weight": 0.3,
    "surprise_overall_threshold": 0.9,
    "use_bayesian_evaluation": True,
    "use_thompson_sampling": True,
    "beta_prior_alpha": 1.0,
    "beta_prior_beta": 1.0,
    "show_processing_details": True,
    "unfit_score_threshold": 4.0,  # NEW: Score below which nodes might be marked unfit
    "unfit_visit_threshold": 3,  # NEW: Min visits before marking unfit by score
}
# ==============================================================================
# Approach Taxonomy & Metadata (Unchanged)
approach_taxonomy = {
    "empirical": ["evidence", "data", "observation", "experiment"],
    "rational": ["logic", "reason", "deduction", "principle"],
    "phenomenological": ["experience", "perception", "consciousness"],
    "hermeneutic": ["interpret", "meaning", "context", "understanding"],
    "reductionist": ["reduce", "component", "fundamental", "elemental"],
    "holistic": ["whole", "system", "emergent", "interconnected"],
    "materialist": ["physical", "concrete", "mechanism"],
    "idealist": ["concept", "ideal", "abstract", "mental"],
    "analytical": ["analyze", "dissect", "examine", "scrutinize"],
    "synthetic": ["synthesize", "integrate", "combine", "unify"],
    "dialectical": ["thesis", "antithesis", "contradiction"],
    "comparative": ["compare", "contrast", "analogy"],
    "critical": ["critique", "challenge", "question", "flaw"],
    "constructive": ["build", "develop", "formulate"],
    "pragmatic": ["practical", "useful", "effective"],
    "normative": ["should", "ought", "value", "ethical"],
    "structural": ["structure", "organize", "framework"],
    "alternative": ["alternative", "different", "another way"],
    "complementary": ["missing", "supplement", "add"],
}
approach_metadata = {
    "empirical": {"family": "epistemology"},
    "rational": {"family": "epistemology"},
    "phenomenological": {"family": "epistemology"},
    "hermeneutic": {"family": "epistemology"},
    "reductionist": {"family": "ontology"},
    "holistic": {"family": "ontology"},
    "materialist": {"family": "ontology"},
    "idealist": {"family": "ontology"},
    "analytical": {"family": "methodology"},
    "synthetic": {"family": "methodology"},
    "dialectical": {"family": "methodology"},
    "comparative": {"family": "methodology"},
    "critical": {"family": "perspective"},
    "constructive": {"family": "perspective"},
    "pragmatic": {"family": "perspective"},
    "normative": {"family": "perspective"},
    "structural": {"family": "general"},
    "alternative": {"family": "general"},
    "complementary": {"family": "general"},
    "variant": {"family": "general"},
    "initial": {"family": "general"},
}
# ==============================================================================

# --- Prompts (MODIFIED to include context from loaded state) ---
initial_prompt = """<instruction>Provide an initial analysis and interpretation of the core themes, arguments, and potential implications presented. Identify key concepts. Respond with clear, natural language text ONLY.</instruction><question>{question}</question>"""

# <<< MODIFIED: Added placeholders for loaded context >>>
thoughts_prompt = """<instruction>Critically examine the current analysis below. Suggest a SIGNIFICANTLY DIFFERENT interpretation, identify a MAJOR underlying assumption or weakness, or propose a novel connection to another domain or concept. Push the thinking in a new direction.

**Previous Run Context (If Available):**
- Previous Best Analysis Summary: {previous_best_summary}
- Previously Marked Unfit Concepts/Areas: {unfit_markers_summary}
- Learned Approach Preferences: {learned_approach_summary}

Consider this context. Avoid repeating unfit areas unless you have a novel mutation. Build upon previous success if appropriate, or diverge strongly if needed.</instruction>
<context>Original Text Summary: {question_summary}\nBest Overall Analysis (Score {best_score}/10): {best_answer}\nCurrent Analysis (Node {current_sequence}): {current_answer}\nCurrent Analysis Tags: {current_tags}</context>
Generate your critique or alternative direction.</instruction>"""

# <<< MODIFIED: Added placeholders for loaded context >>>
update_prompt = """<instruction>Substantially revise the draft analysis below to incorporate the core idea from the critique. Develop the analysis further based on this new direction.

**Previous Run Context (If Available):**
- Previous Best Analysis Summary: {previous_best_summary}
- Previously Marked Unfit Concepts/Areas: {unfit_markers_summary}

Ensure the revision considers past findings and avoids known unproductive paths unless the critique justifies revisiting them.</instruction>
<context>Original Text Summary: {question_summary}\nBest Overall Analysis (Score {best_score}/10): {best_answer}\nCurrent Analysis Tags: {current_tags}</context>
<draft>{answer}</draft>
<critique>{improvements}</critique>
Write the new, revised analysis text."""

# <<< MODIFIED: Added placeholders for loaded context >>>
eval_answer_prompt = """<instruction>Evaluate the intellectual quality and insightfulness of the analysis below (1-10) concerning the original input. Higher scores for depth, novelty, and relevance. Use the full 1-10 scale. Reserve 9-10 for truly exceptional analyses that significantly surpass previous best analysis ({best_score}/10).

**Previous Run Context (If Available):**
- Previous Best Analysis Summary: {previous_best_summary}
- Previously Marked Unfit Concepts/Areas: {unfit_markers_summary}

Consider if this analysis productively builds upon or diverges from past findings.</instruction>
<context>Original Text Summary: {question_summary}\nBest Overall Analysis (Score {best_score}/10): {best_answer}\nAnalysis Tags: {current_tags}</context>
<answer_to_evaluate>{answer_to_evaluate}</answer_to_evaluate>
How insightful, deep, relevant, and well-developed is this analysis compared to the best so far? Does it offer a genuinely novel perspective or intelligently navigate known issues? Rate 1-10 based purely on merit. Respond with a logical rating from 1 to 10.</instruction>"""

tag_generation_prompt = """<instruction>Generate concise keyword tags summarizing the main concepts in the following text. Output the tags, separated by commas.</instruction>\n<text_to_tag>{analysis_text}</text_to_tag>"""
final_synthesis_prompt = """<instruction>Synthesize the key insights developed along the primary path of analysis below into a conclusive statement addressing the original question. Focus on the progression of ideas represented.</instruction>
<original_question_summary>{question_summary}</original_question_summary>
<initial_analysis>{initial_analysis_summary}</initial_analysis>
<best_analysis_score>{best_score}/10</best_analysis_score>
<development_path>
{path_thoughts}
</development_path>
<final_best_analysis>{final_best_analysis_summary}</final_best_analysis>
Synthesize into a final conclusion:"""

# <<< NEW: Intent classification prompt >>>
intent_classifier_prompt = """Classify user requests. Choose the *single best* category from the list. Respond appropriately.

Categories:
- CONTINUE_ANALYSIS: User wants to continue, refine, or build upon the previous MCTS analysis run (e.g., "elaborate", "explore X further", "what about Y?").
- ANALYZE_NEW: User wants a fresh MCTS analysis on a new provided text/task, ignoring any previous runs in this chat.
- GENERAL_CONVERSATION: The input is conversational.

User Input:
"{raw_input_text}"

Classification:
"""
# ==============================================================================


# Logger Setup Function (Unchanged - Top Level)
def setup_logger(level=None):
    logger = logging.getLogger(__name__)
    log_level = (
        level
        if level is not None
        else (
            logging.DEBUG
            if default_config.get("debug_logging", False)
            else logging.INFO
        )
    )
    logger.setLevel(log_level)
    handler_name = f"{name}_handler"
    # Ensure only one handler is added
    if not any(handler.get_name() == handler_name for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.set_name(handler_name)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s [%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False  # Prevent duplication if root logger is configured
    else:
        # Update level of existing handler if it exists
        for handler in logger.handlers:
            if handler.get_name() == handler_name:
                handler.setLevel(log_level)
                break
    return logger


logger = setup_logger()


# Admin User Mock (Unchanged - Top Level)
class AdminUserMock:
    def __init__(self):
        self.role = "admin"


admin = AdminUserMock()
# ==============================================================================


# Text processing functions (Unchanged - Top Level)
def truncate_text(text, max_length=200):
    if not text:
        return ""
    text = str(text).strip()
    # Remove leading markdown code block indicators
    text = re.sub(
        r"^```(json|markdown)?\s*", "", text, flags=re.IGNORECASE | re.MULTILINE
    )
    # Remove trailing markdown code block indicators
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE).strip()
    if len(text) <= max_length:
        return text
    # Find the last space within the limit
    last_space = text.rfind(" ", 0, max_length)
    return text[:last_space] + "..." if last_space != -1 else text[:max_length] + "..."


def calculate_semantic_distance(text1, text2, llm=None, current_config=None):
    # <<< TODO (Optional Enhancement): Replace this with embedding-based distance calculation >>>
    debug = current_config.get("debug_logging", False) if current_config else False
    if not text1 or not text2:
        return 1.0
    text1, text2 = str(text1), str(text2)
    if SKLEARN_AVAILABLE:
        try:
            custom_stop_words = list(ENGLISH_STOP_WORDS) + [
                "analysis",
                "however",
                "therefore",
                "furthermore",
                "perspective",
            ]
            vectorizer = TfidfVectorizer(
                stop_words=custom_stop_words, max_df=0.9, min_df=1
            )
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            # Check matrix shape before accessing elements
            if tfidf_matrix.shape[0] < 2 or tfidf_matrix.shape[1] == 0:
                raise ValueError(
                    "TF-IDF matrix issue (shape: {}).".format(tfidf_matrix.shape)
                )
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            similarity = max(0.0, min(1.0, similarity))  # Clamp value
            return 1.0 - similarity
        except Exception as e:
            logger.warning(
                f"TF-IDF semantic distance error: {e}. Falling back to Jaccard."
            )
    # Fallback to Jaccard Similarity
    try:
        words1 = set(re.findall(r"\w+", text1.lower()))
        words2 = set(re.findall(r"\w+", text2.lower()))
        if not words1 or not words2:
            return 1.0  # Avoid division by zero if one set is empty
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        if union == 0:
            return 0.0  # If both texts are empty or contain no words
        jaccard_similarity = intersection / union
        return 1.0 - jaccard_similarity
    except Exception as fallback_e:
        logger.error(f"Jaccard similarity fallback failed: {fallback_e}")
        return 1.0


# ==============================================================================
# Node class (Unchanged - Top Level)
class Node(BaseModel):
    id: str = Field(
        default_factory=lambda: "node_"
        + "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=4))
    )
    content: str = ""
    parent: Optional["Node"] = None  # Keep as optional Node for in-memory structure
    children: List["Node"] = Field(default_factory=list)
    visits: int = 0
    raw_scores: List[Union[int, float]] = Field(default_factory=list)
    sequence: int = 0
    is_surprising: bool = False
    surprise_explanation: str = ""
    approach_type: str = "initial"
    approach_family: str = "general"
    thought: str = ""
    max_children: int = default_config["max_children"]
    use_bayesian_evaluation: bool = default_config["use_bayesian_evaluation"]
    alpha: Optional[float] = None
    beta: Optional[float] = None
    value: Optional[float] = None  # Used if use_bayesian_evaluation is False
    descriptive_tags: List[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("parent", "children", mode="before")
    @classmethod
    def _validate_optional_fields(cls, v):
        return v

    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.use_bayesian_evaluation:
            self.alpha = max(
                1e-9, float(data.get("alpha", default_config["beta_prior_alpha"]))
            )
            self.beta = max(
                1e-9, float(data.get("beta", default_config["beta_prior_beta"]))
            )
            self.value = None  # Ensure value is None if using Bayesian
        else:
            self.value = float(data.get("value", 0.0))
            self.alpha = None
            self.beta = None

    def add_child(self, child: "Node") -> "Node":
        child.parent = self
        self.children.append(child)
        return child

    def fully_expanded(self) -> bool:
        return len(self.children) >= self.max_children

    def get_bayesian_mean(self) -> float:
        if (
            self.use_bayesian_evaluation
            and self.alpha is not None
            and self.beta is not None
        ):
            alpha_safe = max(1e-9, self.alpha)
            beta_safe = max(1e-9, self.beta)
            if alpha_safe + beta_safe < 1e-18:
                return 0.5  # Avoid division by zero
            return alpha_safe / (alpha_safe + beta_safe)
        return 0.5  # Default if not Bayesian or invalid state

    def get_average_score(self) -> float:
        if self.use_bayesian_evaluation:
            return self.get_bayesian_mean() * 10
        else:
            return (
                (self.value / max(1, self.visits))
                if self.visits > 0 and self.value is not None
                else 5.0
            )  # Return midpoint score if no visits

    def thompson_sample(self) -> float:
        if (
            self.use_bayesian_evaluation
            and self.alpha is not None
            and self.beta is not None
        ):
            alpha_safe = max(1e-9, self.alpha)
            beta_safe = max(1e-9, self.beta)
            try:
                return float(beta_sample(alpha_safe, beta_safe))
            except Exception as e:
                logger.warning(
                    f"Thompson sample failed for node {self.sequence} (α={alpha_safe}, β={beta_safe}): {e}. Using mean."
                )
                return self.get_bayesian_mean()
        return 0.5  # Default if not Bayesian or invalid state

    def best_child(self):  # Find best child based on visits then score
        if not self.children:
            return None
        valid_children = [c for c in self.children if c is not None]
        if not valid_children:
            return None

        max_visits = -1
        most_visited_children = []
        for child in valid_children:
            if child.visits > max_visits:
                max_visits = child.visits
                most_visited_children = [child]
            elif child.visits == max_visits:
                most_visited_children.append(child)

        if not most_visited_children:
            return None  # Should not happen if valid_children exist
        if len(most_visited_children) == 1:
            return most_visited_children[0]

        # Tie-breaking logic using score
        if self.use_bayesian_evaluation:
            return max(most_visited_children, key=lambda c: c.get_bayesian_mean())
        else:
            # Ensure value and visits are valid before dividing
            return max(
                most_visited_children,
                key=lambda c: (
                    (c.value / max(1, c.visits))
                    if c.visits > 0 and c.value is not None
                    else 0.0
                ),
            )

    def node_to_json(self) -> Dict:  # For summary export
        score = self.get_average_score()
        valid_children = [child for child in self.children if child is not None]
        base_json = {
            "id": self.id,
            "sequence": self.sequence,
            "content_summary": truncate_text(self.content, 150),
            "visits": self.visits,
            "approach_type": self.approach_type,
            "approach_family": self.approach_family,
            "is_surprising": self.is_surprising,
            "thought_summary": truncate_text(self.thought, 100),
            "descriptive_tags": self.descriptive_tags,
            "score": round(score, 2),
            "children": [
                child.node_to_json() for child in valid_children
            ],  # Recursive call for children summaries
        }
        if (
            self.use_bayesian_evaluation
            and self.alpha is not None
            and self.beta is not None
        ):
            base_json["value_alpha"] = round(self.alpha, 3)
            base_json["value_beta"] = round(self.beta, 3)
            base_json["value_mean"] = round(self.get_bayesian_mean(), 3)
        elif not self.use_bayesian_evaluation and self.value is not None:
            base_json["value_cumulative"] = round(self.value, 2)
        return base_json

    # <<< NEW: Method for selective state serialization >>>
    def node_to_state_dict(self) -> Dict:
        """Creates a dictionary representation suitable for state persistence."""
        score = self.get_average_score()
        state = {
            "id": self.id,
            "sequence": self.sequence,
            "content_summary": truncate_text(
                self.content, 200
            ),  # Slightly more context for state
            "visits": self.visits,
            "approach_type": self.approach_type,
            "approach_family": self.approach_family,
            "thought": self.thought,
            "descriptive_tags": self.descriptive_tags,
            "score": round(score, 2),
            "is_surprising": self.is_surprising,
            # Include priors/value based on mode
        }
        if (
            self.use_bayesian_evaluation
            and self.alpha is not None
            and self.beta is not None
        ):
            state["alpha"] = round(self.alpha, 4)
            state["beta"] = round(self.beta, 4)
        elif not self.use_bayesian_evaluation and self.value is not None:
            state["value"] = round(
                self.value, 2
            )  # Save cumulative value if not Bayesian
        return state


# ==============================================================================


# MCTS class (MODIFIED for state loading/context - Top Level)
class MCTS:
    def __init__(self, **kwargs):
        self.config = kwargs.get("mcts_config", default_config.copy())
        self.llm = kwargs.get("llm")  # This will be an instance of the Pipe class
        self.question = kwargs.get("question")  # Original question/task for this run
        self.question_summary = self._summarize_question(self.question)

        # <<< NEW: Handle initial state loading >>>
        self.loaded_initial_state = kwargs.get("initial_state", None)  # Loaded from DB
        self.node_sequence = 0  # Reset sequence for each MCTS instance

        self.selected = None
        self.current_simulation_in_iteration = 0
        self.thought_history = []
        self.debug_history = []
        self.surprising_nodes = []
        self.iterations_completed = 0
        self.simulations_completed = 0
        self.high_score_counter = 0
        self.random_state = random.Random()
        self.approach_types = ["initial"]
        self.explored_approaches = {}
        self.explored_thoughts = set()
        self.approach_scores = {}
        self.memory = {"depth": 0, "branches": 0, "high_scoring_nodes": []}
        self.iteration_json_snapshots = []
        self.debug_logging = self.config.get("debug_logging", False)
        self.show_chat_details = self.config.get("show_processing_details", False)

        mcts_start_log = (
            f"# MCTS Analysis Start (Stateful)\nQ Summary: {self.question_summary}\n"
        )
        if self.loaded_initial_state:
            mcts_start_log += f"Loaded state from previous run (Best score: {self.loaded_initial_state.get('best_score', 'N/A')}).\n"
        self.thought_history.append(mcts_start_log)

        cfg = self.config
        prior_alpha = max(1e-9, cfg["beta_prior_alpha"])
        prior_beta = max(1e-9, cfg["beta_prior_beta"])

        # Initialize approach priors (potentially from loaded state)
        initial_approach_priors = (
            self.loaded_initial_state.get("approach_priors")
            if self.loaded_initial_state
            else None
        )
        if (
            initial_approach_priors
            and "alpha" in initial_approach_priors
            and "beta" in initial_approach_priors
        ):
            self.approach_alphas = {
                k: max(1e-9, v) for k, v in initial_approach_priors["alpha"].items()
            }
            self.approach_betas = {
                k: max(1e-9, v) for k, v in initial_approach_priors["beta"].items()
            }
            logger.info("Loaded approach priors from previous state.")
        else:
            self.approach_alphas = {
                approach: prior_alpha for approach in approach_taxonomy.keys()
            }
            self.approach_alphas.update(
                {"initial": prior_alpha, "variant": prior_alpha}
            )
            self.approach_betas = {
                approach: prior_beta for approach in approach_taxonomy.keys()
            }
            self.approach_betas.update({"initial": prior_beta, "variant": prior_beta})

        # Initialize best solution/score (potentially from loaded state)
        self.best_score = (
            self.loaded_initial_state.get("best_score", 0.0)
            if self.loaded_initial_state
            else 0.0
        )
        # Note: best_solution content will come from root initially, updated during search
        self.best_solution = kwargs.get(
            "initial_analysis_content", "No analysis yet."
        )  # Initial root content
        if (
            self.loaded_initial_state
            and "best_solution_content" in self.loaded_initial_state
        ):
            # If continuing, track previous best but root starts with fresh initial analysis
            self.best_solution = self.loaded_initial_state.get(
                "best_solution_content", self.best_solution
            )
            logger.info(
                f"Initialized best score ({self.best_score}) and solution tracker from previous state."
            )

        # Initialize root node
        root_content = kwargs.get(
            "initial_analysis_content", "Analysis root content missing."
        )
        self.root_node_content = root_content  # Store for reference if needed
        self.root = Node(
            content=root_content,
            sequence=self.get_next_sequence(),
            parent=None,
            max_children=cfg["max_children"],
            use_bayesian_evaluation=cfg["use_bayesian_evaluation"],
            alpha=prior_alpha,
            beta=prior_beta,  # Root starts with default priors
            approach_type="initial",
            approach_family="general",
        )
        self.selected = self.root

        # <<< NEW: Store unfit markers from loaded state >>>
        self.unfit_markers = (
            self.loaded_initial_state.get("unfit_markers", [])
            if self.loaded_initial_state
            else []
        )
        if self.unfit_markers:
            logger.info(
                f"Loaded {len(self.unfit_markers)} unfit markers from previous state."
            )

    def _summarize_question(self, question_text: str, max_words=50) -> str:
        if not question_text:
            return "N/A"
        words = re.findall(r"\w+", question_text)
        if len(words) <= max_words:
            return question_text.strip()
        try:
            # Use simple truncation if scikit-learn is not available
            if not SKLEARN_AVAILABLE:
                logger.warning(
                    "Scikit-learn not available for TF-IDF summary. Truncating question."
                )
                return " ".join(words[:max_words]) + "..."
            sentences = re.split(r"[.!?]+\s*", question_text)
            sentences = [s for s in sentences if len(s.split()) > 3]
            if not sentences:
                return (
                    " ".join(words[:max_words]) + "..."
                )  # Fallback if no suitable sentences
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(sentences)
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            num_summary_sentences = max(1, min(3, len(sentences) // 5))
            top_sentence_indices = sentence_scores.argsort()[-num_summary_sentences:][
                ::-1
            ]
            top_sentence_indices.sort()  # Sort by original order
            summary = " ".join([sentences[i] for i in top_sentence_indices])
            summary_words = summary.split()
            # Final length check
            if (
                len(summary_words) > max_words * 1.2
            ):  # Allow slightly longer if summary is good
                return " ".join(summary_words[:max_words]) + "..."
            return summary + (
                "..." if len(words) > len(summary_words) else ""
            )  # Add ellipsis if truncated
        except Exception as e:
            logger.warning(f"TF-IDF summary failed ({e}). Truncating.")
            return " ".join(words[:max_words]) + "..."

    def get_next_sequence(self) -> int:
        self.node_sequence += 1
        return self.node_sequence

    def export_tree_as_json(self) -> Dict:  # For debug/verbose output
        try:
            return self.root.node_to_json()
        except Exception as e:
            logger.error(f"JSON export error: {e}", exc_info=self.debug_logging)
            return {"error": f"Export failed: {e}"}

    # <<< MODIFIED: To incorporate loaded state into context >>>
    def get_context_for_node(self, node: Node) -> Dict[str, str]:
        cfg = self.config
        debug = self.debug_logging
        best_answer_str = str(self.best_solution) if self.best_solution else "N/A"
        context = {
            "question_summary": self.question_summary,
            "best_answer": truncate_text(best_answer_str, 300),
            "best_score": f"{self.best_score:.1f}",
            "current_answer": truncate_text(node.content, 300),
            "current_sequence": str(node.sequence),
            "current_approach": node.approach_type,
            "current_tags": (
                ", ".join(node.descriptive_tags) if node.descriptive_tags else "None"
            ),
            "tree_depth": str(self.memory.get("depth", 0)),
            "branches": str(self.memory.get("branches", 0)),
            "approach_types": ", ".join(self.approach_types),
            # <<< NEW: Context from loaded state >>>
            "previous_best_summary": "N/A",
            "unfit_markers_summary": "None",
            "learned_approach_summary": "Default priors",
        }

        # Add context from loaded state if available
        if self.loaded_initial_state:
            context["previous_best_summary"] = self.loaded_initial_state.get(
                "best_solution_summary", "N/A"
            )
            unfit_markers = self.loaded_initial_state.get("unfit_markers", [])
            if unfit_markers:
                markers_str = "; ".join(
                    [
                        f"'{m.get('summary', m.get('id', 'Unknown'))}' ({m.get('reason', 'Unknown reason')})"
                        for m in unfit_markers[:5]
                    ]
                )
                context["unfit_markers_summary"] = markers_str + (
                    "..." if len(unfit_markers) > 5 else ""
                )
            else:
                context["unfit_markers_summary"] = "None recorded"

            priors = self.loaded_initial_state.get("approach_priors")
            if priors and "alpha" in priors and "beta" in priors:
                # Summarize learned priors (e.g., top 3 highest mean)
                means = {}
                for app, alpha in priors["alpha"].items():
                    beta = priors["beta"].get(
                        app, 1.0
                    )  # Default beta if missing for some reason
                    alpha = max(1e-9, alpha)
                    beta = max(1e-9, beta)  # Ensure > 0
                    if alpha + beta > 1e-9:
                        means[app] = (alpha / (alpha + beta)) * 10
                sorted_means = sorted(
                    means.items(), key=lambda item: item[1], reverse=True
                )
                top_approaches = [
                    f"{app} ({score:.1f})" for app, score in sorted_means[:3]
                ]
                context["learned_approach_summary"] = (
                    f"Favors: {', '.join(top_approaches)}"
                    + ("..." if len(sorted_means) > 3 else "")
                )
            else:
                context["learned_approach_summary"] = "Priors not loaded or incomplete"

        # Add other context elements (unchanged logic)
        try:  # Explored Thought Types
            if cfg["track_explored_approaches"]:
                exp_app_text = []
                # Use current run's alphas/betas for reporting context
                current_alphas = self.approach_alphas
                current_betas = self.approach_betas
                sorted_approach_keys = sorted(self.explored_approaches.keys())

                for app in sorted_approach_keys:
                    thoughts = self.explored_approaches.get(app, [])
                    if thoughts:
                        count = len(thoughts)
                        score_text = ""
                        # Use current priors for display
                        if cfg["use_bayesian_evaluation"]:
                            alpha = current_alphas.get(app, 1)
                            beta = current_betas.get(app, 1)
                            alpha = max(1e-9, alpha)
                            beta = max(1e-9, beta)  # Safety
                            if (alpha + beta) > 1e-9:
                                score_text = (
                                    f"(β-Mean: {alpha / (alpha + beta):.2f}, N={count})"
                                )
                            else:
                                score_text = (
                                    "(N=?" + str(count) + ")"
                                )  # Indicate prior issue but show count
                        else:
                            score = self.approach_scores.get(app, 0)
                            score_text = (
                                f"(Avg: {score:.1f}, N={count})"  # Non-Bayesian avg
                            )
                        # Show last 1 or 2 thoughts as examples
                        sample_count = min(2, len(thoughts))
                        sample = thoughts[-sample_count:]
                        exp_app_text.append(
                            f"- {app} {score_text}: {'; '.join([f'{truncate_text(str(t), 50)}' for t in sample])}"
                        )
                context["explored_approaches"] = (
                    "\n".join(exp_app_text) if exp_app_text else "None yet."
                )
        except Exception as e:
            logger.error(f"Ctx err (approaches): {e}")
            context["explored_approaches"] = "Error generating approach context."

        try:  # High Scoring Examples
            if self.memory["high_scoring_nodes"]:
                high_score_text = [
                    f"- Score {score:.1f} ({app}): {truncate_text(content, 70)}"
                    for score, content, app, thought in self.memory[
                        "high_scoring_nodes"
                    ]
                ]
                context["high_scoring_examples"] = "\n".join(
                    ["Top Examples:"] + high_score_text
                )
            else:
                context["high_scoring_examples"] = "None yet."
        except Exception as e:
            logger.error(f"Ctx err (high scores): {e}")
            context["high_scoring_examples"] = "Error generating high score context."

        try:  # Sibling Context
            if (
                cfg["sibling_awareness"]
                and node.parent
                and len(node.parent.children) > 1
            ):
                siblings = [
                    c for c in node.parent.children if c is not None and c != node
                ]
                if siblings:
                    sib_app_text = []
                    sorted_siblings = sorted(siblings, key=lambda s: s.sequence)
                    for s in sorted_siblings:
                        if (
                            s.thought and s.visits > 0
                        ):  # Only show siblings that have thoughts and were visited
                            score = s.get_average_score()
                            tags_str = (
                                f"Tags: [{', '.join(s.descriptive_tags)}]"
                                if s.descriptive_tags
                                else ""
                            )
                            sib_app_text.append(
                                f'"{truncate_text(str(s.thought), 50)}" -> (Score: {score:.1f} {tags_str})'
                            )
                    if sib_app_text:
                        context["sibling_approaches"] = "\n".join(
                            ["Siblings:"] + [f"- {sa}" for sa in sib_app_text]
                        )
                    # else: context["sibling_approaches"] = "None." # Uncomment if you want explicit 'None'
        except Exception as e:
            logger.error(f"Ctx err (siblings): {e}")
            context["sibling_approaches"] = "Error generating sibling context."

        # Ensure all values are strings for formatting
        safe_context = {k: str(v) if v is not None else "" for k, v in context.items()}
        return safe_context

    def _calculate_uct(self, node: Node, parent_visits: int) -> float:
        # <<< TODO (Optional Enhancement): Check against self.unfit_markers here >>>
        # If node corresponds to an unfit marker (e.g., by ID or semantic similarity),
        # return a very low score unless node.is_surprising is True.
        cfg = self.config
        if node.visits == 0:
            return float("inf")  # Prioritize unvisited nodes

        exploitation = (
            node.get_bayesian_mean()
            if cfg["use_bayesian_evaluation"]
            else (
                node.value / node.visits
                if node.value is not None and node.visits > 0
                else 0.5
            )
        )

        # Basic check against unfit markers (can be made more sophisticated)
        is_unfit = False
        if hasattr(self, "unfit_markers") and self.unfit_markers:
            for marker in self.unfit_markers:
                # Match by ID or sequence (quick check)
                if (
                    marker.get("id") == node.id
                    or marker.get("sequence") == node.sequence
                ):
                    is_unfit = True
                    logger.debug(
                        f"Node {node.sequence} matches unfit marker {marker.get('id', '')} by ID/Seq. Penalizing UCT."
                    )
                    break
                # Optional: Add semantic check between node.thought/content and marker['summary']
                # if calculate_semantic_distance(node.thought or node.content, marker.get('summary')) < 0.2: # Example threshold
                #    is_unfit = True
                #    logger.debug(f"Node {node.sequence} semantically matches unfit marker. Penalizing UCT.")
                #    break

        # Apply penalty if unfit, unless surprising (allow surprising nodes to bypass penalty)
        if is_unfit and not node.is_surprising:
            return 0.0  # Heavily penalize selection of unfit nodes unless surprising

        # Standard exploration term
        log_parent_visits = math.log(max(1, parent_visits))
        exploration = cfg["exploration_weight"] * math.sqrt(
            log_parent_visits / node.visits
        )

        # Surprise Bonus
        surprise_bonus = 0.3 if node.is_surprising else 0  # Fixed bonus value

        # Diversity Bonus (relative to siblings)
        diversity_bonus = 0.0
        if (
            node.parent
            and len(node.parent.children) > 1
            and cfg["score_diversity_bonus"] > 0
        ):
            my_score_normalized = node.get_average_score() / 10.0  # Normalize score
            sibling_scores = []
            for sibling in node.parent.children:
                # Include only visited siblings for comparison
                if sibling is not None and sibling != node and sibling.visits > 0:
                    sibling_scores.append(
                        sibling.get_average_score() / 10.0
                    )  # Normalize score for comparison
            if sibling_scores:
                sibling_avg = sum(sibling_scores) / len(sibling_scores)
                # Bonus is proportional to absolute difference from average sibling score
                diversity_bonus = cfg["score_diversity_bonus"] * abs(
                    my_score_normalized - sibling_avg
                )

        uct_value = exploitation + exploration + surprise_bonus + diversity_bonus
        return uct_value if math.isfinite(uct_value) else 0.0  # Ensure finite return

    # --- Other MCTS methods ---
    def _collect_non_leaf_nodes(self, node, non_leaf_nodes, max_depth, current_depth=0):
        # (implementation unchanged)
        if current_depth > max_depth:
            return
        if node is None:
            return
        # Node is non-leaf if it HAS children AND is not fully expanded yet
        if node.children and not node.fully_expanded():
            non_leaf_nodes.append(node)

        for child in node.children:
            if child is not None:
                self._collect_non_leaf_nodes(
                    child, non_leaf_nodes, max_depth, current_depth + 1
                )

    async def select(self) -> Node:
        # (implementation unchanged - relies on _calculate_uct's potential penalty)
        cfg = self.config
        debug = self.debug_logging
        if debug:
            logger.debug("Selecting node...")
        node = self.root
        selection_path = [node]
        debug_info = "### Selection Path Decisions:\n"
        force_interval = cfg["force_exploration_interval"]

        # Branch Enhancement (unchanged)
        if (
            force_interval > 0
            and self.simulations_completed > 0
            and self.simulations_completed % force_interval == 0
            and self.memory.get("depth", 0) > 1
        ):
            candidate_nodes = []
            self._collect_non_leaf_nodes(
                self.root, candidate_nodes, max_depth=max(1, self.memory["depth"] // 2)
            )
            expandable_candidates = [
                n for n in candidate_nodes if not n.fully_expanded()
            ]
            if expandable_candidates:
                selected_node = self.random_state.choice(expandable_candidates)
                debug_info += (
                    f"BRANCH ENHANCE: Forcing selection {selected_node.sequence}\n"
                )
                if debug:
                    logger.debug(f"BRANCH ENHANCE: Selected {selected_node.sequence}")
                temp_path = []
                curr_path_node = selected_node
                while curr_path_node:
                    temp_path.append(f"Node {curr_path_node.sequence}")
                    curr_path_node = curr_path_node.parent
                path_str = " → ".join(reversed(temp_path))
                self.thought_history.append(
                    f"### Selection Path (Forced)\n{path_str}\n"
                )
                if debug:
                    self.debug_history.append(debug_info)
                return selected_node  # Return the forced selection

        # Standard selection loop
        while node.children:
            valid_children = [child for child in node.children if child is not None]
            if not valid_children:
                if debug:
                    logger.warning(
                        f"Node {node.sequence} has children list but contains only None. Stopping selection."
                    )
                break  # Cannot proceed down this path

            parent_visits = node.visits
            unvisited = [child for child in valid_children if child.visits == 0]

            if unvisited:
                selected_child = self.random_state.choice(unvisited)
                debug_info += f"Selected unvisited {selected_child.sequence}\n"
                node = selected_child
                break  # Selected an unvisited child, stop selection here

            # If all children visited, use selection strategy
            if cfg["use_thompson_sampling"] and cfg["use_bayesian_evaluation"]:
                # Thompson Sampling
                samples = []
                for child in valid_children:
                    try:
                        sample_val = child.thompson_sample()
                        if math.isfinite(sample_val):
                            samples.append((child, sample_val))
                        else:
                            if debug:
                                logger.warning(
                                    f"Node {child.sequence} TS returned non-finite value ({sample_val}). Skipping."
                                )
                    except Exception as ts_err:
                        logger.error(
                            f"Thompson Sampling error for node {child.sequence}: {ts_err}"
                        )

                if not samples:
                    if debug:
                        logger.warning(
                            f"No valid Thompson samples for children of {node.sequence}. Selecting randomly."
                        )
                    selected_child = self.random_state.choice(valid_children)
                else:
                    selected_child, best_sample = max(samples, key=lambda x: x[1])
                    debug_info += (
                        f"TS: Node {selected_child.sequence} ({best_sample:.3f})\n"
                    )
                node = selected_child
            else:
                # UCT Selection
                uct_values = []
                for child in valid_children:
                    try:
                        uct = self._calculate_uct(
                            child, parent_visits
                        )  # UCT now includes unfit penalty check
                        if math.isfinite(uct):  # Ensure UCT value is valid
                            uct_values.append((child, uct))
                        else:
                            logger.warning(
                                f"UCT for child {child.sequence} was non-finite. Skipping."
                            )
                    except Exception as uct_err:
                        logger.error(
                            f"UCT calculation error for node {child.sequence}: {uct_err}"
                        )

                if not uct_values:
                    if debug:
                        logger.warning(
                            f"No valid UCT values for children of {node.sequence}. Selecting randomly."
                        )
                    # Ensure there's at least one valid child before choosing randomly
                    if not valid_children:
                        logger.error(
                            f"Selection error: No valid children for Node {node.sequence} and no valid UCT values. Cannot proceed."
                        )
                        return node  # Return current node as selection cannot advance
                    selected_child = self.random_state.choice(valid_children)
                else:
                    # Select child with highest UCT score
                    uct_values.sort(key=lambda x: x[1], reverse=True)
                    selected_child = uct_values[0][0]
                    debug_info += f"UCT: Node {selected_child.sequence} ({uct_values[0][1]:.3f})\n"
                node = selected_child

            selection_path.append(node)  # Add selected node to path

            # Stop if the newly selected node is a leaf or not fully expanded (it will be expanded next)
            if not node.children or not node.fully_expanded():
                break

        # Log the selection path
        path_str = " → ".join(
            [
                f"Node {n.sequence} (Tags: {', '.join(n.descriptive_tags) if n.descriptive_tags else '[]'})"
                for n in selection_path
            ]
        )
        self.thought_history.append(f"### Selection Path\n{path_str}\n")
        if debug:
            self.debug_history.append(debug_info)
            logger.debug(f"Selection path: {path_str}\n{debug_info}")

        current_depth = len(selection_path) - 1
        self.memory["depth"] = max(self.memory.get("depth", 0), current_depth)
        return node  # Return the selected leaf or expandable node

    def _classify_approach(self, thought: str) -> Tuple[str, str]:
        approach_type = "variant"
        approach_family = "general"
        if not thought or not isinstance(thought, str):
            return approach_type, approach_family
        thought_lower = thought.lower()
        approach_scores = {
            app: sum(1 for kw in kws if kw in thought_lower)
            for app, kws in approach_taxonomy.items()
        }
        positive_scores = {
            app: score for app, score in approach_scores.items() if score > 0
        }
        if positive_scores:
            max_score = max(positive_scores.values())
            best_approaches = [
                app for app, score in positive_scores.items() if score == max_score
            ]
            approach_type = self.random_state.choice(best_approaches)
        if approach_type in approach_metadata:
            approach_family = approach_metadata[approach_type].get("family", "general")
        if self.debug_logging:
            logger.debug(
                f"Classified thought '{truncate_text(thought, 50)}' as: {approach_type} ({approach_family})"
            )
        return approach_type, approach_family

    def _check_surprise(
        self, parent_node, new_content, new_approach_type, new_approach_family
    ) -> Tuple[bool, str]:
        cfg = self.config
        debug = self.debug_logging
        surprise_factors = []
        is_surprising = False
        surprise_explanation = ""
        # 1. Semantic Distance Check
        if cfg["use_semantic_distance"]:
            try:
                parent_content_str = (
                    str(parent_node.content) if parent_node.content else ""
                )
                new_content_str = str(new_content) if new_content else ""
                if parent_content_str and new_content_str:
                    dist = calculate_semantic_distance(
                        parent_content_str, new_content_str, self.llm, cfg
                    )
                    if dist > cfg["surprise_threshold"]:
                        surprise_factors.append(
                            {
                                "type": "semantic",
                                "value": dist,
                                "weight": cfg["surprise_semantic_weight"],
                                "desc": f"Semantic dist ({dist:.2f})",
                            }
                        )
            except Exception as e:
                logger.warning(f"Semantic distance check failed: {e}")

        # 2. Shift in Thought Approach Family
        parent_family = getattr(parent_node, "approach_family", "general")
        if parent_family != new_approach_family and new_approach_family != "general":
            surprise_factors.append(
                {
                    "type": "family_shift",
                    "value": 1.0,
                    "weight": cfg["surprise_philosophical_shift_weight"],
                    "desc": f"Shift '{parent_family}'->'{new_approach_family}'",
                }
            )

        # 3. Novelty of Thought Approach Family (BFS)
        try:
            family_counts = Counter()
            queue = []
            nodes_visited = 0
            MAX_NODES = 100
            MAX_DEPTH = 5
            if self.root:
                queue.append((self.root, 0))
            else:
                logger.error("Novelty check cannot start: Root node is None.")

            processed_in_bfs = set()  # Prevent cycles/reprocessing in BFS
            while queue and nodes_visited < MAX_NODES:
                curr_node_in_loop = None
                try:
                    if not queue:
                        break
                    curr_node_in_loop, depth = queue.pop(0)

                    if curr_node_in_loop is None:
                        logger.warning(
                            "Popped None node during novelty check BFS. Skipping."
                        )
                        continue
                    if curr_node_in_loop.id in processed_in_bfs:
                        continue  # Skip already processed
                    processed_in_bfs.add(curr_node_in_loop.id)

                    if depth > MAX_DEPTH:
                        continue
                    nodes_visited += 1
                    fam = getattr(curr_node_in_loop, "approach_family", "general")
                    family_counts[fam] += 1

                    if depth + 1 <= MAX_DEPTH:
                        for child in curr_node_in_loop.children:
                            if child is not None and child.id not in processed_in_bfs:
                                queue.append((child, depth + 1))
                            elif child is None:
                                parent_id = getattr(curr_node_in_loop, "id", "UNK")
                                logger.warning(
                                    f"Node {parent_id} contains a None child reference during novelty check BFS."
                                )
                except Exception as node_proc_err:
                    node_id = (
                        getattr(curr_node_in_loop, "id", "UNK")
                        if curr_node_in_loop
                        else "UNK_None"
                    )
                    logger.error(
                        f"Err processing node {node_id} in novelty check BFS: {node_proc_err}",
                        exc_info=debug,
                    )
                    continue  # Continue BFS with next node

            # Check if the new family is novel based on counts
            if (
                family_counts.get(new_approach_family, 0) <= 1
                and new_approach_family != "general"
            ):
                # Use triple quotes for the f-string to safely handle internal single quotes
                surprise_factors.append(
                    {
                        "type": "novelty",
                        "value": 0.8,
                        "weight": cfg["surprise_novelty_weight"],
                        "desc": f"Novel approach family ('{new_approach_family}')",
                    }
                )
        except Exception as e:
            logger.warning(f"Novelty check BFS failed overall: {e}", exc_info=debug)  #

        # Calculate combined score
        if surprise_factors:
            total_weighted_score = sum(
                f["value"] * f["weight"] for f in surprise_factors
            )
            total_weight = sum(f["weight"] for f in surprise_factors)
            combined_score = (
                (total_weighted_score / total_weight) if total_weight > 1e-6 else 0.0
            )
            if combined_score >= cfg["surprise_overall_threshold"]:
                is_surprising = True
                factor_descs = [
                    f"- {f['desc']} (Val: {f['value']:.2f}, W: {f['weight']:.1f})"
                    for f in surprise_factors
                ]
                surprise_explanation = (
                    f"Combined surprise ({combined_score:.2f} >= {cfg['surprise_overall_threshold']}):\n"
                    + "\n".join(factor_descs)
                )
                if debug:
                    logger.debug(
                        f"Surprise DETECTED for node sequence {parent_node.sequence+1}: Score={combined_score:.2f}\n{surprise_explanation}"
                    )

        return is_surprising, surprise_explanation

    async def expand(self, node: Node) -> Tuple[Optional[Node], bool]:
        # (Uses context from get_context_for_node)
        cfg = self.config
        debug = self.debug_logging
        if debug:
            logger.debug(
                f"Expanding node {node.sequence} ('{truncate_text(node.content, 50)}') Tags: {node.descriptive_tags}"
            )
        try:
            # Ensure the llm object (Pipe instance) is available
            if not self.llm:
                logger.error(
                    f"LLM object not available in MCTS for expansion of node {node.sequence}"
                )
                return None, False

            await self.llm.progress(
                f"Expanding Node {node.sequence} (Generating thought)..."
            )
            context = self.get_context_for_node(
                node
            )  # Context now includes previous state info

            # Generate Thought using the context-aware prompt
            thought = await self.llm.generate_thought(
                node.content, context, self.config
            )
            if (
                not isinstance(thought, str)
                or not thought.strip()
                or "Error:" in thought
            ):
                logger.error(
                    f"Invalid thought generation result: '{thought}' for node {node.sequence}"
                )
                return None, False
            thought = thought.strip()
            if debug:
                logger.debug(f"Node {node.sequence} Thought: '{thought}'")

            # <<< TODO (Optional Enhancement): Check thought against self.unfit_markers here before proceeding >>>
            # If thought is semantically similar to an unfit marker, maybe ask for another thought?

            thought_entry = (
                f"### Expanding Node {node.sequence}\n... Thought: {thought}\n"
            )
            approach_type, approach_family = self._classify_approach(thought)
            thought_entry += (
                f"... Approach: {approach_type} (Family: {approach_family})\n"
            )
            self.explored_thoughts.add(thought)
            if approach_type not in self.approach_types:
                self.approach_types.append(approach_type)
            if approach_type not in self.explored_approaches:
                self.explored_approaches[approach_type] = []
            self.explored_approaches[approach_type].append(thought)

            await self.llm.progress(
                f"Expanding Node {node.sequence} (Updating analysis)..."
            )
            context_for_update = context.copy()
            context_for_update["answer"] = node.content
            context_for_update.pop("current_answer", None)
            context_for_update.pop("current_sequence", None)
            # Generate new content using context-aware prompt
            new_content = await self.llm.update_approach(
                node.content, thought, context_for_update, self.config
            )
            if (
                not isinstance(new_content, str)
                or not new_content.strip()
                or "Error:" in new_content
            ):
                logger.error(
                    f"Invalid new content generation result: '{new_content}' for node {node.sequence}"
                )
                return None, False
            new_content = new_content.strip()

            await self.llm.progress(
                f"Expanding Node {node.sequence} (Generating tags)..."
            )
            new_tags = await self._generate_tags_for_node(new_content)
            if debug:
                logger.debug(f"Node {node.sequence+1} Generated Tags: {new_tags}")
            thought_entry += f"... Generated Tags: {new_tags}\n"
            is_surprising, surprise_explanation = self._check_surprise(
                node, new_content, approach_type, approach_family
            )
            if is_surprising:
                thought_entry += f"**SURPRISE DETECTED!**\n{surprise_explanation}\n"

            # Create child node
            initial_alpha = max(1e-9, cfg["beta_prior_alpha"])
            initial_beta = max(1e-9, cfg["beta_prior_beta"])
            child = Node(
                content=new_content,
                parent=node,
                sequence=self.get_next_sequence(),
                is_surprising=is_surprising,
                surprise_explanation=surprise_explanation,
                approach_type=approach_type,
                approach_family=approach_family,
                thought=thought,
                max_children=cfg["max_children"],
                use_bayesian_evaluation=cfg["use_bayesian_evaluation"],
                alpha=initial_alpha,
                beta=initial_beta,
                descriptive_tags=new_tags,
            )
            node.add_child(child)
            if is_surprising:
                self.surprising_nodes.append(child)

            thought_entry += f"--> New Analysis {child.sequence} (Tags: {child.descriptive_tags}): {truncate_text(new_content, 100)}\n"
            self.thought_history.append(thought_entry)

            if len(node.children) > 1:
                self.memory["branches"] = self.memory.get("branches", 0) + 1

            if debug:
                logger.debug(
                    f"Successfully expanded Node {node.sequence} into Child {child.sequence}"
                )
            return child, is_surprising
        except Exception as e:
            logger.error(f"Expand error on Node {node.sequence}: {e}", exc_info=debug)
            return None, False

    async def _generate_tags_for_node(self, analysis_text: str) -> List[str]:
        # (implementation unchanged)
        cfg = self.config
        debug = self.debug_logging
        if not analysis_text:
            return []
        max_tags_to_keep = 3
        try:
            # Ensure the llm object (Pipe instance) is available
            if not self.llm:
                logger.error("LLM object not available in MCTS for tag generation.")
                return []

            tag_string_raw = await self.llm.get_completion(
                self.llm.resolve_model(body={"model": self.llm.__model__}),
                messages=[
                    {
                        "role": "user",
                        "content": tag_generation_prompt.format(
                            analysis_text=analysis_text
                        ),
                    }
                ],
            )
            if not tag_string_raw or "Error:" in tag_string_raw:
                logger.warning(f"Tag generation failed: {tag_string_raw}")
                return []

            cleaned_tags = []
            phrases_to_remove = [
                "instruction>",
                "tags:",
                "json",
                "list:",
                "`",
                "output only",
                "here are",
                "keywords:",
                "output:",
                "text_to_tag>",
                "response:",
                "tags",
            ]
            cleaned_text = tag_string_raw
            # Remove potential prefixes like "Tags:", "Keywords:", etc. more robustly
            cleaned_text = re.sub(
                r"^\s*[\w\s]*?[:\-]\s*", "", cleaned_text, count=1
            ).strip()
            for phrase in phrases_to_remove:
                cleaned_text = re.sub(
                    re.escape(phrase), "", cleaned_text, flags=re.IGNORECASE
                )

            # Split by common delimiters and clean up each tag
            potential_tags = re.split(r"[,\n;]+", cleaned_text)
            for tag in potential_tags:
                tag = tag.strip().strip(
                    "'\"` M*[]{}:<>/().,-"
                )  # More aggressive stripping
                tag = re.sub(r"[*_`]", "", tag)  # Remove markdown emphasis
                tag = re.sub(r"\s+", " ", tag).strip()  # Normalize whitespace
                # Basic validation: not empty, length constraints, not just a number, not 'none'
                if (
                    tag
                    and 1 < len(tag) < 50
                    and not tag.isdigit()
                    and tag.lower() != "none"
                ):
                    # Check for duplicates (case-insensitive)
                    is_duplicate = False
                    for existing_tag in cleaned_tags:
                        if existing_tag.lower() == tag.lower():
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        cleaned_tags.append(tag)
                if len(cleaned_tags) >= max_tags_to_keep:
                    break  # Stop once max tags reached

            if debug:
                logger.debug(
                    f"Raw tags: '{tag_string_raw}'. Cleaned tags: {cleaned_tags}"
                )
            return cleaned_tags[:max_tags_to_keep]
        except Exception as e:
            logger.error(f"Tag generation/parsing error: {e}", exc_info=debug)
            return []

    async def simulate(self, node: Node) -> Optional[float]:
        # (implementation unchanged - uses context from get_context_for_node)
        cfg = self.config
        debug = self.debug_logging
        if debug:
            logger.debug(
                f"Simulating node {node.sequence} ('{truncate_text(node.content, 50)}') Tags: {node.descriptive_tags}"
            )
        score = None
        raw_score = 0
        try:
            # Ensure the llm object (Pipe instance) is available
            if not self.llm:
                logger.error(
                    f"LLM object not available in MCTS for simulation of node {node.sequence}"
                )
                return None

            await self.llm.progress(f"Evaluating Analysis Node {node.sequence}...")
            context = self.get_context_for_node(
                node
            )  # Context now includes previous state info
            node_content = str(node.content) if node.content else ""
            if not node_content:
                logger.warning(
                    f"Node {node.sequence} content is empty. Assigning score 1."
                )
                return 1.0

            # Call evaluation using context-aware prompt
            score_result = await self.llm.evaluate_answer(
                node_content, context, self.config
            )
            eval_type = "absolute"
            if not isinstance(score_result, int) or not (1 <= score_result <= 10):
                logger.error(
                    f"Evaluation for Node {node.sequence} returned invalid result: {score_result}. Defaulting to score 5."
                )
                score = 5.0
                eval_type = "absolute (failed)"
                raw_score = 5
            else:
                score = float(score_result)
                raw_score = score_result

            node.raw_scores.append(raw_score)
            approach = node.approach_type if node.approach_type else "unknown"

            # Update approach priors (using current alpha/beta)
            if cfg["use_bayesian_evaluation"]:
                # Use raw score for pseudo counts (scale 1-10)
                pseudo_successes = max(0, raw_score - 1)  # E.g., 10 -> 9 successes
                pseudo_failures = max(0, 10 - raw_score)  # E.g., 3 -> 7 failures
                current_alpha = self.approach_alphas.setdefault(
                    approach, cfg["beta_prior_alpha"]
                )
                current_beta = self.approach_betas.setdefault(
                    approach, cfg["beta_prior_beta"]
                )
                # Ensure they remain positive after update
                self.approach_alphas[approach] = max(
                    1e-9, current_alpha + pseudo_successes
                )
                self.approach_betas[approach] = max(
                    1e-9, current_beta + pseudo_failures
                )
            # Update simple average score tracker (Non-Bayesian)
            current_avg = self.approach_scores.get(approach, score)
            self.approach_scores[approach] = (
                0.7 * score + 0.3 * current_avg
            )  # EMA update

            if debug:
                logger.debug(
                    f"Node {node.sequence} eval: Type={eval_type}, Raw={raw_score}, Score={score:.1f}/10"
                )
            self.thought_history.append(
                f"### Evaluating Node {node.sequence} (Tags: {node.descriptive_tags})\n... Score: {score:.1f}/10 ({eval_type}, raw: {raw_score})\n"
            )

            # Update high score memory
            if score >= 7:  # Threshold for interesting node
                entry = (score, node.content, approach, node.thought)
                self.memory["high_scoring_nodes"].append(entry)
                self.memory["high_scoring_nodes"].sort(key=lambda x: x[0], reverse=True)
                self.memory["high_scoring_nodes"] = self.memory["high_scoring_nodes"][
                    : cfg["memory_cutoff"]
                ]

        except Exception as e:
            logger.error(
                f"Simulate error for node {node.sequence}: {e}", exc_info=debug
            )
            return None
        return score

    def backpropagate(self, node: Node, score: float):
        # (implementation unchanged)
        cfg = self.config
        debug = self.debug_logging
        if debug:
            logger.debug(f"Backpropagating score {score:.2f} from {node.sequence}...")
        backprop_path_nodes = []
        temp_node = node
        # Use score directly (normalized 1-10) for updates
        pseudo_successes = max(0, score - 1)
        pseudo_failures = max(0, 10 - score)

        while temp_node:
            backprop_path_nodes.append(f"Node {temp_node.sequence}")
            temp_node.visits += 1
            if cfg["use_bayesian_evaluation"]:
                # Ensure alpha/beta exist before updating
                if temp_node.alpha is not None and temp_node.beta is not None:
                    temp_node.alpha = max(1e-9, temp_node.alpha + pseudo_successes)
                    temp_node.beta = max(1e-9, temp_node.beta + pseudo_failures)
                else:
                    logger.warning(
                        f"Node {temp_node.sequence} missing alpha/beta during backprop. Cannot update Bayesian values."
                    )
                    # Initialize if missing? Or just skip update? Skipping for now.
                    # temp_node.alpha = max(1e-9, cfg["beta_prior_alpha"] + pseudo_successes)
                    # temp_node.beta = max(1e-9, cfg["beta_prior_beta"] + pseudo_failures)
            else:  # Non-Bayesian update (cumulative score)
                if temp_node.value is not None:
                    temp_node.value += (
                        score  # Add the raw score (1-10) to cumulative value
                    )
                else:
                    logger.warning(
                        f"Node {temp_node.sequence} missing value during non-Bayesian backprop. Initializing."
                    )
                    temp_node.value = score  # Initialize if missing
            temp_node = temp_node.parent

        path_str = " → ".join(reversed(backprop_path_nodes))
        self.thought_history.append(
            f"### Backpropagating Score {score:.1f}\n... Path: {path_str}\n"
        )
        if debug:
            logger.debug(f"Backprop complete: {path_str}")

    async def search(self, simulations_per_iteration: int):
        # (implementation unchanged)
        cfg = self.config
        debug = self.debug_logging
        show_chat_sim_details = cfg.get("show_processing_details", False)

        if not self.llm:
            logger.error("LLM object not available in MCTS for search. Cannot proceed.")
            return None  # Cannot run search without LLM

        if debug:
            logger.info(
                f"Starting MCTS Iteration {self.iterations_completed + 1} ({simulations_per_iteration} simulations)..."
            )
        nodes_simulated = 0
        for i in range(simulations_per_iteration):
            self.simulations_completed += 1
            self.current_simulation_in_iteration = i + 1
            sim_entry_log = f"### Iter {self.iterations_completed + 1} - Sim {i+1}/{simulations_per_iteration}\n"
            self.thought_history.append(sim_entry_log)
            if debug:
                logger.debug(
                    f"--- Starting Sim {self.current_simulation_in_iteration}/{simulations_per_iteration} ---"
                )

            best_score_before_sim = self.best_score
            leaf = await self.select()
            self.selected = leaf
            node_to_simulate = leaf
            expanded_in_this_sim = False
            expansion_result_tuple = None
            thought_leading_to_expansion = None

            if leaf and not leaf.fully_expanded() and leaf.content:
                if debug:
                    logger.debug(
                        f"Sim {i+1}: Attempting expansion from Node {leaf.sequence}."
                    )
                expansion_result_tuple = await self.expand(leaf)
                if expansion_result_tuple and expansion_result_tuple[0]:
                    node_to_simulate = expansion_result_tuple[0]
                    expanded_in_this_sim = True
                    thought_leading_to_expansion = node_to_simulate.thought
                    if debug:
                        logger.debug(
                            f"Sim {i+1}: Expanded {leaf.sequence} -> {node_to_simulate.sequence}."
                        )
                else:
                    if debug:
                        logger.warning(
                            f"Sim {i+1}: Expansion failed for {leaf.sequence}. Simulating original leaf."
                        )
                    node_to_simulate = leaf
                    expanded_in_this_sim = False
            elif not leaf:
                logger.error(f"Sim {i+1}: Selection returned None. Cannot proceed.")
                continue
            elif leaf:  # Leaf exists but might be fully expanded or contentless
                if debug:
                    logger.debug(
                        f"Sim {i+1}: Leaf node {leaf.sequence} is fully expanded or has no content. Simulating it directly."
                    )
                node_to_simulate = leaf
                expanded_in_this_sim = False

            score = None
            if node_to_simulate and node_to_simulate.content:
                score = await self.simulate(node_to_simulate)
                nodes_simulated += 1
                if debug:
                    logger.debug(
                        f"Sim {i+1}: Node {node_to_simulate.sequence} simulated. Score={score}"
                    )
            elif node_to_simulate:
                if debug:
                    logger.warning(
                        f"Sim {i+1}: Skipping simulation for {node_to_simulate.sequence} (no content)."
                    )
            else:
                logger.error(
                    f"Sim {i+1}: Cannot simulate, node_to_simulate is None (Should not happen after selection/expansion logic)."
                )
                continue

            if score is not None:
                self.backpropagate(node_to_simulate, score)
                new_best_overall = score > self.best_score
                if new_best_overall:
                    self.best_score = score
                    self.best_solution = str(node_to_simulate.content)
                    node_info = f"Node {node_to_simulate.sequence} ({node_to_simulate.approach_type}) Tags: {node_to_simulate.descriptive_tags}"
                    self.thought_history.append(
                        f"### New Best! Score: {score:.1f}/10 ({node_info})\n"
                    )
                    if debug:
                        logger.info(
                            f"Sim {i+1}: New best! Score: {score:.1f}, {node_info}"
                        )
                    self.high_score_counter = (
                        0  # Reset stability counter on improvement
                    )
                elif score == self.best_score:
                    # Score matched best, don't reset counter (allows stability check)
                    pass

                # --- Conditional Simulation Detail Emission to CHAT ---
                if (
                    show_chat_sim_details and node_to_simulate and self.llm
                ):  # Check self.llm again
                    sim_detail_msg = f"--- Iter {self.iterations_completed + 1} / Sim {self.current_simulation_in_iteration} ---\n"
                    sim_detail_msg += f"Selected Node: {leaf.sequence} (Visits: {leaf.visits}, Score: {leaf.get_average_score():.1f}, Tags: {leaf.descriptive_tags})\n"

                    if expanded_in_this_sim and thought_leading_to_expansion:
                        sim_detail_msg += f'Based on thought: "{str(thought_leading_to_expansion).strip()}"\n'
                        sim_detail_msg += f"--> Expanded to New Node: {node_to_simulate.sequence} ({node_to_simulate.approach_type})\n"
                        sim_detail_msg += (
                            f"    Tags: {node_to_simulate.descriptive_tags}\n"
                        )
                        # Optionally show snippet of new content:
                        # sim_detail_msg += f"    Content Snippet: {truncate_text(str(node_to_simulate.content), 100)}\n"
                    else:
                        sim_detail_msg += f"--> Re-evaluating Node: {node_to_simulate.sequence} (Visits: {node_to_simulate.visits})\n"
                        sim_detail_msg += (
                            f"    Tags: {node_to_simulate.descriptive_tags}\n"
                        )

                    sim_detail_msg += f"Evaluated Score: {score:.1f}/10"
                    if score > best_score_before_sim:
                        sim_detail_msg += " ✨"  # Indicate improvement within iteration
                    if new_best_overall:
                        sim_detail_msg += " 🏆 (New Overall Best!)"
                    sim_detail_msg += "\n"

                    # Use the llm (Pipe instance) to emit the message
                    await self.llm.emit_message(sim_detail_msg)
                    await asyncio.sleep(0.05)  # Small delay for readability
                # --- End Conditional Emission ---

                # Early Stopping Check
                if cfg["early_stopping"]:
                    if self.best_score >= cfg["early_stopping_threshold"]:
                        self.high_score_counter += 1
                        if debug:
                            logger.debug(
                                f"Sim {i+1}: Best score ({self.best_score:.1f}) >= threshold. Stability counter: {self.high_score_counter}/{cfg['early_stopping_stability']}"
                            )
                        if self.high_score_counter >= cfg["early_stopping_stability"]:
                            if debug:
                                logger.info(
                                    f"Early stopping criteria met after sim {i+1}, iter {self.iterations_completed + 1}."
                                )
                            # Report early stop to user if verbose
                            if show_chat_sim_details and self.llm:
                                await self.llm.emit_message(
                                    f"**Stopping early:** Analysis score ({self.best_score:.1f}/10) reached threshold and stability."
                                )
                            self._store_iteration_snapshot(
                                "Early Stopping (High Score Stability)"
                            )
                            return None  # Signal early stop
                    else:
                        self.high_score_counter = (
                            0  # Reset if score drops below threshold
                        )

            else:  # Score is None
                if node_to_simulate:
                    if debug:
                        logger.warning(
                            f"Sim {i+1}: Simulation failed or skipped for Node {node_to_simulate.sequence}. No score obtained."
                        )
                self.high_score_counter = (
                    0  # Reset stability counter if simulation fails
                )

        self._store_iteration_snapshot("End of Iteration")
        if debug:
            logger.info(f"Finished MCTS Iteration {self.iterations_completed + 1}.")
        return self.selected  # Return last selected node

    def _store_iteration_snapshot(self, reason: str):  # For debug/analysis
        # (implementation unchanged)
        cfg = self.config
        debug = self.debug_logging
        MAX_SNAPSHOTS = 10
        if len(self.iteration_json_snapshots) >= MAX_SNAPSHOTS:
            if debug:
                logger.warning(
                    f"Max snapshots ({MAX_SNAPSHOTS}) reached. Not storing for: {reason}"
                )
            return
        try:
            if debug:
                logger.debug(f"Storing tree snapshot: {reason}")
            # Use monotonic time for more reliable timing if available
            try:
                timestamp = asyncio.get_running_loop().time()
            except RuntimeError:
                timestamp = datetime.now().timestamp()  # Fallback

            snapshot = {
                "iteration": self.iterations_completed + 1,
                "simulation": self.current_simulation_in_iteration,
                "reason": reason,
                "timestamp": timestamp,
                "best_score_so_far": self.best_score,
                "tree_json": self.export_tree_as_json(),
            }
            self.iteration_json_snapshots.append(snapshot)
        except Exception as e:
            logger.error(f"Snapshot store failed: {e}", exc_info=debug)

    async def _report_tree_stats(self):  # For verbose output
        # (implementation unchanged)
        cfg = self.config
        debug = self.debug_logging
        try:
            total_nodes = self.node_sequence
            max_depth = self.memory.get("depth", 0)
            num_leaves = 0
            leaf_nodes = []
            self._collect_leaves(self.root, leaf_nodes)
            num_leaves = len(leaf_nodes)
            # Avoid division by zero if tree is just root or root + leaves
            num_internal_nodes = total_nodes - num_leaves
            avg_branching = (
                ((total_nodes - 1) / max(1, num_internal_nodes))
                if num_internal_nodes > 0
                else 0
            )
            stats_msg = f"### Tree Stats: Nodes={total_nodes}, Depth={max_depth}, Leaves={num_leaves}, Avg Branching={avg_branching:.2f}\n"
            if debug:
                self.debug_history.append(stats_msg)
                logger.debug(stats_msg)
            # If verbose, maybe emit to chat? Currently only logs.
            # if cfg.get("show_processing_details", False) and self.llm:
            #     await self.llm.emit_message(stats_msg)
        except Exception as e:
            logger.error(f"Error reporting tree stats: {e}", exc_info=debug)

    def _collect_leaves(self, node, leaf_nodes):  # Helper for stats
        # (implementation unchanged)
        if not node:
            return
        if not node.children:
            leaf_nodes.append(node)
        else:
            for child in node.children:
                if child is not None:
                    self._collect_leaves(child, leaf_nodes)

    async def analyze_iteration(self):  # Not currently used, keep stub
        # (implementation unchanged)
        RUN_ANALYSIS = False
        if not RUN_ANALYSIS:
            return None

    def formatted_output(
        self, highlighted_node=None, final_output=False
    ) -> str:  # Verbose mode output
        # (implementation unchanged)
        cfg = self.config
        debug = self.debug_logging
        result = ""
        try:
            if not final_output:
                return ""  # Only generate for final summary
            result = (
                f"# MCTS Final Analysis Summary (Verbose)\n"  # Title indicates verbose
            )
            result += f"The following summarizes the MCTS exploration process, highlighting the best analysis found and the key development steps (thoughts) that led to high-scoring nodes.\n\n"

            # 1. Best Solution (Full Analysis + Tags)
            if self.best_solution:
                best_node = self.find_best_final_node()  # Try to find the node object
                tags_str = (
                    f"Tags: {best_node.descriptive_tags}"
                    if best_node and best_node.descriptive_tags
                    else "Tags: []"
                )
                result += f"## Best Analysis Found (Score: {self.best_score:.1f}/10)\n"
                result += f"**{tags_str}**\n\n"
                analysis_text = str(self.best_solution)
                # Clean potential markdown code blocks from analysis text
                analysis_text = re.sub(
                    r"^```(json|markdown)?\s*",
                    "",
                    analysis_text,
                    flags=re.IGNORECASE | re.MULTILINE,
                )
                analysis_text = re.sub(
                    r"\s*```$", "", analysis_text, flags=re.MULTILINE
                )
                result += f"{analysis_text.strip()}\n"
            else:
                result += "## Best Analysis\nNo valid solution found.\n"

            # 2. Top Performing Nodes (Thought + Score Details)
            result += "\n## Top Performing Nodes & Driving Thoughts\n"
            all_nodes = []
            nodes_to_process = []
            processed_nodes = set()
            if self.root:
                nodes_to_process.append(self.root)
            while nodes_to_process:  # BFS to collect all visited nodes
                current = nodes_to_process.pop(0)
                if current is None or current.id in processed_nodes:
                    continue
                processed_nodes.add(current.id)
                if current.visits > 0:
                    all_nodes.append(current)  # Only include visited nodes
                valid_children = [
                    child for child in current.children if child is not None
                ]
                nodes_to_process.extend(valid_children)

            # Sort nodes by average score
            sorted_nodes = sorted(
                all_nodes, key=lambda n: n.get_average_score(), reverse=True
            )
            top_n = 5  # Show top 5 performing nodes
            if sorted_nodes:
                for i, node in enumerate(sorted_nodes[:top_n]):
                    score = node.get_average_score()
                    score_details = ""
                    if (
                        cfg["use_bayesian_evaluation"]
                        and node.alpha is not None
                        and node.beta is not None
                    ):
                        score_details = f"(α={node.alpha:.1f}, β={node.beta:.1f})"
                    elif not cfg["use_bayesian_evaluation"] and node.value is not None:
                        score_details = f"(value={node.value:.1f})"
                    tags_str = (
                        f"Tags: {node.descriptive_tags}"
                        if node.descriptive_tags
                        else "Tags: []"
                    )
                    result += f"### Node {node.sequence}: Score {score:.1f}/10 {score_details}\n"
                    result += f"- **Approach**: {node.approach_type} ({node.approach_family})\n"
                    result += f"- **Visits**: {node.visits}\n"
                    result += f"- **{tags_str}**\n"
                    if node.thought:
                        result += f"- **Thought**: {str(node.thought).strip()}\n"
                    else:
                        result += "- **Thought**: (N/A - Initial Node or Root)\n"
                    if node.is_surprising:
                        result += f"- **Surprising**: Yes ({truncate_text(node.surprise_explanation, 100)})\n"
                    result += "\n"
            else:
                result += "No nodes with visits found.\n"

            # 3. Most Explored Path (by Visits/Score)
            result += "\n## Most Explored Path\n"
            current = self.root
            path = []
            if current:
                path.append(current)
            while current and current.children:
                best_child_node = (
                    current.best_child()
                )  # best_child uses visits then score
                if not best_child_node or best_child_node.visits == 0:
                    if debug:
                        logger.debug(
                            f"Path exploration stopped at Node {current.sequence}. Best child '{getattr(best_child_node, 'sequence', 'N/A')}' is None or has 0 visits."
                        )
                    break
                path.append(best_child_node)
                current = best_child_node
            if len(path) > 1:
                result += "The search explored this primary path (by visits/score):\n\n"
                for i, node in enumerate(path):
                    prefix = "└─ " if i == len(path) - 1 else "├─ "
                    indent = "   " * i
                    score = node.get_average_score()
                    tags_str = (
                        f"Tags: {node.descriptive_tags}"
                        if node.descriptive_tags
                        else ""
                    )
                    result += f"{indent}{prefix}Node {node.sequence} ({node.approach_type}, Score: {score:.1f}, Visits: {node.visits}) {tags_str}\n"
            else:
                result += "Search did not explore significantly beyond the root node.\n"

            # 4. Surprising Nodes
            if self.surprising_nodes:
                result += "\n## Surprising Nodes\n"
                result += "Nodes that triggered surprise detection:\n\n"
                max_show = 5
                start = max(0, len(self.surprising_nodes) - max_show)
                for node in self.surprising_nodes[start:]:
                    if node is None:
                        continue
                    score = node.get_average_score()
                    tags_str = (
                        f"Tags: {node.descriptive_tags}"
                        if node.descriptive_tags
                        else "Tags: []"
                    )
                    result += f"- **Node {node.sequence}** ({node.approach_type}, Score: {score:.1f}, {tags_str}):\n  "
                    result += f"{truncate_text(node.surprise_explanation.splitlines()[0], 150)}\n"

            # 5. Approach Performance (using current run's priors)
            if self.approach_scores or self.approach_alphas:
                result += "\n## Thought Approach Performance (This Run)\n"
                approaches_data = []
                # Use current alpha/beta state for report
                all_apps = set(self.approach_alphas.keys()) | set(
                    self.approach_scores.keys()
                )
                for app in all_apps:
                    if app == "unknown":
                        continue
                    count = len(self.explored_approaches.get(app, []))
                    if count == 0 and app != "initial":
                        continue  # Skip if not used and not initial
                    score_str = "N/A"
                    sort_key = -1.0
                    if cfg["use_bayesian_evaluation"]:
                        alpha = self.approach_alphas.get(app, cfg["beta_prior_alpha"])
                        beta = self.approach_betas.get(app, cfg["beta_prior_beta"])
                        alpha = max(1e-9, alpha)
                        beta = max(1e-9, beta)  # Safety
                        if (alpha + beta) > 1e-9:
                            mean_score = alpha / (alpha + beta) * 10
                            score_str = f"Score: {mean_score:.2f}/10 (α={alpha:.1f}, β={beta:.1f})"
                            sort_key = mean_score
                        else:
                            score_str = "Score: N/A (Priors Error?)"
                            sort_key = -1.0
                    else:  # Non-Bayesian avg score
                        if app in self.approach_scores:
                            avg_score = self.approach_scores[app]
                            score_str = f"Score: {avg_score:.2f}/10"
                            sort_key = avg_score
                        elif count > 0 or app == "initial":
                            score_str = "Score: N/A"
                            sort_key = -1.0  # Should not happen if approach exists

                    # Add count if > 0
                    count_str = f" ({count} thoughts)" if count > 0 else ""
                    approaches_data.append(
                        {
                            "name": app,
                            "score_str": score_str,
                            "count_str": count_str,
                            "sort_key": sort_key,
                        }
                    )

                sorted_approaches = sorted(
                    approaches_data, key=lambda x: x["sort_key"], reverse=True
                )
                max_show = 7
                for data in sorted_approaches[:max_show]:
                    result += f"- **{data['name']}**: {data['score_str']}{data['count_str']}\n"
                if len(sorted_approaches) > max_show:
                    result += f"- ... ({len(sorted_approaches) - max_show} more)\n"

            # 6. Search Parameters Used
            result += f"\n## Search Parameters Used (This Run)\n"
            result += f"- **Iterations**: {self.iterations_completed}/{cfg['max_iterations']}\n"
            result += f"- **Simulations/Iter**: {cfg['simulations_per_iteration']}\n"
            result += f"- **Total Simulations**: {self.simulations_completed}\n"
            eval_str = (
                "Bayesian (Beta)"
                if cfg["use_bayesian_evaluation"]
                else "Traditional (Avg)"
            )
            select_str = (
                "Thompson"
                if cfg["use_bayesian_evaluation"] and cfg["use_thompson_sampling"]
                else "UCT"
            )
            result += f"- **Evaluation**: {eval_str}\n"
            result += f"- **Selection**: {select_str}\n"
            if cfg["use_bayesian_evaluation"]:
                result += f"- **Beta Priors (Initial)**: α={cfg['beta_prior_alpha']:.2f}, β={cfg['beta_prior_beta']:.2f} (Learned reflected in Approach Perf.)\n"  # Clarify these are initial, learned are in approach perf.
            result += f"- **Exploration Weight**: {cfg['exploration_weight']:.2f}\n"
            result += (
                f"- **Early Stopping**: {'On' if cfg['early_stopping'] else 'Off'}\n"
            )
            if cfg["early_stopping"]:
                result += f"  - Threshold: {cfg['early_stopping_threshold']:.1f}/10\n"
                result += f"  - Stability: {cfg['early_stopping_stability']}\n"
            result += f"- **Show Chat Details**: {'On' if cfg.get('show_processing_details', False) else 'Off'}\n"

            # 7. Conditional Debug Log Snippets
            if debug and self.debug_history:
                result += "\n## Debug Log Snippets (Last 3)\n\n"
                for entry in self.debug_history[-3:]:
                    cleaned_entry = re.sub(r"\n+", "\n", entry).strip()
                    result += truncate_text(cleaned_entry, 200) + "\n---\n"

            return result.strip()
        except Exception as e:
            logger.error(f"Error formatting final output: {e}", exc_info=debug)
            error_msg = (
                f"\n\n# Error generating final summary:\n{type(e).__name__}: {str(e)}\n"
            )
            result += error_msg
            return result

    def find_best_final_node(
        self,
    ) -> Optional[Node]:  # Finds node object matching best_solution content
        # (implementation unchanged)
        if not self.best_solution or not self.root:
            return None  # Need root and target solution
        queue = []
        visited = set()
        best_match_node = None
        min_score_diff = float("inf")

        queue.append(self.root)
        visited.add(self.root.id)

        # Clean the target best solution content once
        best_sol_content_cleaned = str(self.best_solution)
        best_sol_content_cleaned = re.sub(
            r"^```(json|markdown)?\s*",
            "",
            best_sol_content_cleaned,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        best_sol_content_cleaned = re.sub(
            r"\s*```$", "", best_sol_content_cleaned, flags=re.MULTILINE
        ).strip()

        while queue:
            current = queue.pop(0)
            if current is None:
                continue

            # Clean the node's content for comparison
            node_content_cleaned = str(current.content)
            node_content_cleaned = re.sub(
                r"^```(json|markdown)?\s*",
                "",
                node_content_cleaned,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            node_content_cleaned = re.sub(
                r"\s*```$", "", node_content_cleaned, flags=re.MULTILINE
            ).strip()

            # Check for exact content match
            if node_content_cleaned == best_sol_content_cleaned:
                score_diff = abs(current.get_average_score() - self.best_score)
                # If multiple nodes match content, prefer the one with score closest to self.best_score
                if best_match_node is None or score_diff < min_score_diff:
                    best_match_node = current
                    min_score_diff = score_diff
                # Optimization: If score difference is negligible, we can potentially stop early
                # if min_score_diff < 1e-3: break

            # Continue BFS
            valid_children = [
                child for child in current.children if child and child.id not in visited
            ]
            for child in valid_children:
                visited.add(child.id)
                queue.append(child)

        if not best_match_node:
            logger.warning("Could not find node object matching best solution content.")
        return best_match_node


# ==============================================================================


# Pipe class (MODIFIED for Statefulness and Intent Handling - Top Level)
class Pipe:
    # Ensure Valves class is defined *inside* Pipe or accessible globally
    class Valves(BaseModel):  # Configuration options
        # MCTS Core Parameters (Copied from default_config for UI visibility)
        MAX_ITERATIONS: int = Field(
            default=default_config["max_iterations"], title="Max Iterations", ge=1
        )
        SIMULATIONS_PER_ITERATION: int = Field(
            default=default_config["simulations_per_iteration"],
            title="Simulations / Iteration",
            ge=1,
        )
        MAX_CHILDREN: int = Field(
            default=default_config["max_children"], title="Max Children / Node", ge=1
        )
        EXPLORATION_WEIGHT: float = Field(
            default=default_config["exploration_weight"],
            title="Exploration Weight (UCT)",
            ge=0.0,
        )
        USE_THOMPSON_SAMPLING: bool = Field(
            default=default_config["use_thompson_sampling"],
            title="Use Thompson Sampling (if Bayesian)",
        )
        FORCE_EXPLORATION_INTERVAL: int = Field(
            default=default_config["force_exploration_interval"],
            title="Force Branch Explore Interval (0=off)",
            ge=0,
        )
        SCORE_DIVERSITY_BONUS: float = Field(
            default=default_config["score_diversity_bonus"],
            title="UCT Score Diversity Bonus",
            ge=0.0,
        )
        USE_BAYESIAN_EVALUATION: bool = Field(
            default=default_config["use_bayesian_evaluation"],
            title="Use Bayesian (Beta) Evaluation",
        )
        BETA_PRIOR_ALPHA: float = Field(
            default=default_config["beta_prior_alpha"],
            gt=0,
            title="Bayesian Prior Alpha (>0)",
        )
        BETA_PRIOR_BETA: float = Field(
            default=default_config["beta_prior_beta"],
            gt=0,
            title="Bayesian Prior Beta (>0)",
        )
        USE_SEMANTIC_DISTANCE: bool = Field(
            default=default_config["use_semantic_distance"],
            title="Use Semantic Distance (Surprise)",
        )
        SURPRISE_THRESHOLD: float = Field(
            default=default_config["surprise_threshold"],
            ge=0.0,
            le=1.0,
            title="Surprise Threshold (Semantic)",
        )
        SURPRISE_SEMANTIC_WEIGHT: float = Field(
            default=default_config["surprise_semantic_weight"],
            title="Surprise: Semantic Weight",
            ge=0.0,
            le=1.0,
        )
        SURPRISE_PHILOSOPHICAL_SHIFT_WEIGHT: float = Field(
            default=default_config["surprise_philosophical_shift_weight"],
            title="Surprise: Shift Weight (Thought)",
            ge=0.0,
            le=1.0,
        )
        SURPRISE_NOVELTY_WEIGHT: float = Field(
            default=default_config["surprise_novelty_weight"],
            title="Surprise: Novelty Weight (Thought)",
            ge=0.0,
            le=1.0,
        )
        SURPRISE_OVERALL_THRESHOLD: float = Field(
            default=default_config["surprise_overall_threshold"],
            ge=0.0,
            le=1.0,
            title="Surprise: Overall Threshold",
        )
        GLOBAL_CONTEXT_IN_PROMPTS: bool = Field(
            default=default_config["global_context_in_prompts"],
            title="Use Global Context in Prompts",
        )
        TRACK_EXPLORED_APPROACHES: bool = Field(
            default=default_config["track_explored_approaches"],
            title="Track Explored Thought Approaches",
        )
        SIBLING_AWARENESS: bool = Field(
            default=default_config["sibling_awareness"],
            title="Add Sibling Context to Prompts",
        )
        MEMORY_CUTOFF: int = Field(
            default=default_config["memory_cutoff"],
            title="Memory Cutoff (Top N High Scores)",
            ge=0,
        )
        EARLY_STOPPING: bool = Field(
            default=default_config["early_stopping"], title="Enable Early Stopping"
        )
        EARLY_STOPPING_THRESHOLD: float = Field(
            default=default_config["early_stopping_threshold"],
            ge=1.0,
            le=10.0,
            title="Early Stopping Score Threshold",
        )
        EARLY_STOPPING_STABILITY: int = Field(
            default=default_config["early_stopping_stability"],
            ge=1,
            title="Early Stopping Stability",
        )
        SHOW_PROCESSING_DETAILS: bool = Field(
            default=default_config["show_processing_details"],
            title="Show Detailed MCTS Steps in Chat",
        )
        DEBUG_LOGGING: bool = Field(
            default=default_config["debug_logging"],
            title="Enable Detailed Debug Logging (Console/Logs)",
        )
        # NEW Config for stateful behavior
        ENABLE_STATE_PERSISTENCE: bool = Field(
            default=True, title="Enable State Persistence (via DB)"
        )
        UNFIT_SCORE_THRESHOLD: float = Field(
            default=default_config["unfit_score_threshold"],
            ge=0.0,
            le=10.0,
            title="Unfit Marker Score Threshold",
        )
        UNFIT_VISIT_THRESHOLD: int = Field(
            default=default_config["unfit_visit_threshold"],
            ge=1,
            title="Unfit Marker Min Visits",
        )

    # <<< Methods correctly indented within the Pipe class >>>
    def __init__(self):
        self.type = "manifold"
        # Standard pipe type
        self.__current_event_emitter__ = None
        # Stores the emitter passed in pipe()
        self.__question__ = ""
        # Holds the primary user task/question for the current run
        self.__model__ = ""
        # Resolved model name
        self.__llm_client__ = None  # For potential persistent client (future opt.)
        self.valves = self.Valves()  # Initialize with defaults from Valves class
        # self.config = default_config.copy() # Initialize internal config state (can be updated by valves) - Removed, use current_config in pipe()

    def pipes(self) -> list[dict[str, str]]:
        # (implementation unchanged)
        try:
            # Try to refresh model list
            ollama.get_all_models()  # Assuming this updates app.state.OLLAMA_MODELS
            if hasattr(app.state, "OLLAMA_MODELS") and app.state.OLLAMA_MODELS:
                models = app.state.OLLAMA_MODELS
                valid_models = {
                    k: v
                    for k, v in models.items()
                    if isinstance(v, dict) and "name" in v
                }
                if not valid_models:
                    logger.warning("No valid Ollama models found in app state.")
                    return [{"id": f"{name}-error", "name": f"{name} (No models)"}]
                return [
                    {"id": f"{name}-{k}", "name": f"{name} ({v['name']})"}
                    for k, v in valid_models.items()
                ]
            else:
                logger.error(
                    "OLLAMA_MODELS not found or empty in app state after refresh."
                )
                return [{"id": f"{name}-error", "name": f"{name} (Model load error)"}]
        except Exception as e:
            logger.error(
                f"Failed to list pipes: {e}",
                exc_info=default_config.get("debug_logging", False),
            )
            return [{"id": f"{name}-error", "name": f"{name} (Error: {e})"}]

    def resolve_model(self, body: dict) -> str:
        # (implementation unchanged)
        model_id = body.get("model", "").strip()
        pipe_internal_name = name
        prefix_to_find = f"{pipe_internal_name}-"
        separator_index = model_id.rfind(prefix_to_find)
        if separator_index != -1:
            base_model_name = model_id[separator_index + len(prefix_to_find) :]
            if base_model_name:
                if ":" not in base_model_name:
                    logger.warning(
                        f"Resolved model '{base_model_name}' seems to be missing a tag (e.g., ':latest'). Using anyway."
                    )
                # else: logger.info(f"Resolved base model '{base_model_name}' from pipe model ID '{model_id}'") # Reduced logging verbosity
                return base_model_name
            else:
                logger.error(
                    f"Separator '{prefix_to_find}' found in '{model_id}' but no subsequent model name. Falling back to using full ID."
                )
                return model_id
        else:
            logger.warning(
                f"Pipe prefix separator '{prefix_to_find}' not found in model ID '{model_id}'. Assuming it's already the base model name."
            )
            return model_id

    def resolve_question(self, body: dict) -> str:  # Gets last user message
        # (implementation unchanged)
        msgs = body.get("messages", [])
        for msg in reversed(msgs):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                return content.strip() if isinstance(content, str) else ""
        return ""

    # <<< NEW: Database Helper Methods (Indented within Pipe) >>>
    def _get_db_connection(self):
        "Establishes connection to the SQLite DB and ensures table exists."
        conn = None
        try:
            # Ensure the directory exists
            db_dir = os.path.dirname(DB_FILE)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
                logger.info(f"Created directory for database: {db_dir}")

            conn = sqlite3.connect(DB_FILE)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS mcts_state (chat_id TEXT PRIMARY KEY, last_state_json TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)"
            )
            # Optional: Add index for faster lookup if table grows large
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mcts_state_timestamp ON mcts_state (timestamp);"
            )
            conn.commit()
            logger.debug(f"Connected to DB: {DB_FILE}")
            return conn
        except sqlite3.Error as e:
            logger.error(
                f"SQLite error connecting to or creating table in {DB_FILE}: {e}",
                exc_info=True,
            )
            if conn:
                conn.close()
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error during DB connection setup {DB_FILE}: {e}",
                exc_info=True,
            )
            if conn:
                conn.close()
            return None

    def _save_mcts_state(self, chat_id: str, state_json: str):
        "Saves the MCTS state JSON for a given chat_id."
        if not chat_id:
            logger.warning("Cannot save state: chat_id is missing.")
            return
        conn = self._get_db_connection()
        if not conn:
            logger.error("Cannot save state: Failed to get DB connection.")
            return
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO mcts_state (chat_id, last_state_json, timestamp) VALUES (?, ?, ?)",
                (chat_id, state_json, datetime.now()),
            )
            conn.commit()
            logger.info(f"Saved MCTS state for chat_id: {chat_id}")
        except sqlite3.Error as e:
            logger.error(
                f"SQLite error saving state for chat_id {chat_id}: {e}", exc_info=True
            )
        finally:
            if conn:
                conn.close()

    def _load_mcts_state(self, chat_id: str) -> Optional[str]:
        "Loads the most recent MCTS state JSON for a given chat_id."
        if not chat_id:
            logger.warning("Cannot load state: chat_id is missing.")
            return None
        conn = self._get_db_connection()
        if not conn:
            logger.error("Cannot load state: Failed to get DB connection.")
            return None
        state_json = None
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT last_state_json FROM mcts_state WHERE chat_id = ? ORDER BY timestamp DESC LIMIT 1",
                (chat_id,),
            )
            result = cursor.fetchone()
            if result:
                state_json = result[0]
                logger.info(f"Loaded MCTS state for chat_id: {chat_id}")
            else:
                logger.info(f"No previous MCTS state found for chat_id: {chat_id}")
        except sqlite3.Error as e:
            logger.error(
                f"SQLite error loading state for chat_id {chat_id}: {e}", exc_info=True
            )
        finally:
            if conn:
                conn.close()
        return state_json

    # <<< NEW: State Serialization Logic (Indented within Pipe) >>>
    def _serialize_mcts_state(self, mcts_instance: MCTS, config: Dict) -> str:
        "Extracts selective state from MCTS and returns as JSON string."
        if not mcts_instance or not mcts_instance.root:
            logger.warning("Attempted to serialize empty or invalid MCTS instance.")
            return "{}"

        state = {"version": "0.8.0"}  # Version state format

        try:
            # Basic info
            state["best_score"] = mcts_instance.best_score
            best_node = mcts_instance.find_best_final_node()
            state["best_solution_summary"] = truncate_text(
                mcts_instance.best_solution, 400
            )
            state["best_solution_content"] = str(
                mcts_instance.best_solution
            )  # Save full best content
            state["best_node_tags"] = best_node.descriptive_tags if best_node else []

            # Learned Priors (if Bayesian)
            if mcts_instance.config.get("use_bayesian_evaluation"):
                # Ensure alphas/betas exist and are dicts before accessing
                alphas = getattr(mcts_instance, "approach_alphas", {})
                betas = getattr(mcts_instance, "approach_betas", {})
                if isinstance(alphas, dict) and isinstance(betas, dict):
                    state["approach_priors"] = {
                        "alpha": {k: round(v, 4) for k, v in alphas.items()},
                        "beta": {k: round(v, 4) for k, v in betas.items()},
                    }
                else:
                    logger.warning(
                        "Approach priors (alpha/beta) were not dictionaries. Skipping serialization."
                    )
                    state["approach_priors"] = None  # Indicate missing priors

            # Top Nodes (e.g., top 3 based on score)
            all_nodes = []
            nodes_to_process = [mcts_instance.root]
            visited_ids = set()
            while nodes_to_process:
                current = nodes_to_process.pop(0)
                if not current or current.id in visited_ids:
                    continue
                visited_ids.add(current.id)
                if current.visits > 0:
                    all_nodes.append(current)
                nodes_to_process.extend(
                    [child for child in current.children if child]
                )  # Add valid children
            # Sort by score, descending
            sorted_nodes = sorted(
                all_nodes, key=lambda n: n.get_average_score(), reverse=True
            )
            state["top_nodes"] = [
                node.node_to_state_dict() for node in sorted_nodes[:3]
            ]  # Save state dict for top 3

            # Unfit Markers (simple approach: nodes below threshold score after enough visits)
            unfit_markers = []
            score_thresh = config.get("unfit_score_threshold", 4.0)
            visit_thresh = config.get("unfit_visit_threshold", 3)
            for node in all_nodes:  # Reuse collected nodes
                if (
                    node.visits >= visit_thresh
                    and node.get_average_score() < score_thresh
                ):
                    marker = {
                        "id": node.id,
                        "sequence": node.sequence,
                        "summary": truncate_text(node.thought or node.content, 80),
                        "reason": f"Low score ({node.get_average_score():.1f} < {score_thresh}) after {node.visits} visits",
                        "tags": node.descriptive_tags,
                    }
                    unfit_markers.append(marker)
            state["unfit_markers"] = unfit_markers[:10]  # Limit number saved

            return json.dumps(state)

        except Exception as e:
            logger.error(f"Error during MCTS state serialization: {e}", exc_info=True)
            return "{}"  # Return empty JSON on error

    # <<< NEW: Intent Classification (Indented within Pipe) >>>
    async def classify_intent(self, text_to_classify: str) -> str:
        "Uses the LLM to classify the user's intent."
        logger.debug(f"Classifying intent for: {truncate_text(text_to_classify)}")
        prompt = intent_classifier_prompt.format(raw_input_text=text_to_classify)
        default_intent = "ANALYZE_NEW"  # Default if classification fails or unclear
        try:
            # Make sure self.__model__ is resolved before calling get_completion
            if not self.__model__:
                logger.error("Intent classification called before model was resolved.")
                await self.emit_message(
                    "**Warning:** Internal error (model not set). Defaulting to new analysis."
                )
                return default_intent

            classification_result = await self.get_completion(
                self.__model__, [{"role": "user", "content": prompt}]
            )
            # Basic validation and cleaning
            valid_intents = [
                "ANALYZE_NEW",
                "CONTINUE_ANALYSIS",
                "ASK_LAST_RUN_SUMMARY",
                "ASK_PROCESS",
                "ASK_CONFIG",
                "GENERAL_CONVERSATION",
            ]
            # Handle potential multi-line/wordy responses from LLM
            clean_result = (
                classification_result.strip().upper().split()[0]
                if classification_result
                else ""
            )  # Take first word, upper case
            # Remove trailing punctuation if any
            clean_result = re.sub(r"[.,!?;:]$", "", clean_result)

            if clean_result in valid_intents:
                logger.info(f"Intent classified as: {clean_result}")
                return clean_result
            else:
                logger.warning(
                    f"Intent classification returned unexpected result: '{classification_result}'. Defaulting to {default_intent}."
                )
                # Maybe the input itself looks like a continuation keyword?
                if any(
                    keyword in text_to_classify.lower()
                    for keyword in [
                        "continue",
                        "elaborate",
                        "further",
                        "what about",
                        "refine",
                        "build on",
                    ]
                ):
                    logger.info(
                        "Input text suggests continuation despite classification. Setting intent to CONTINUE_ANALYSIS."
                    )
                    return "CONTINUE_ANALYSIS"
                return default_intent
        except Exception as e:
            logger.error(f"Intent classification LLM call failed: {e}", exc_info=True)
            # Check if emit_message is available before calling
            if hasattr(self, "emit_message") and callable(self.emit_message):
                await self.emit_message(
                    f"**Warning:** Could not determine intent due to error. Defaulting to new analysis."
                )
            return default_intent

    # <<< NEW: Intent Handlers (Indented within Pipe) >>>
    async def handle_ask_process(self, user_input: str):
        logger.info("Handling intent: ASK_PROCESS")
        # Simple predefined answer for now, could use LLM later
        explanation = "I use an Advanced Bayesian Monte Carlo Tree Search (MCTS) algorithm. Key aspects include:- **Exploration vs. Exploitation:** Balancing trying new ideas (exploration) with focusing on promising ones (exploitation) using UCT or Thompson Sampling.- **Bayesian Evaluation:** (Optional) Using Beta distributions to represent score uncertainty, allowing for more informed decisions under uncertainty.- **Node Expansion:** Generating new 'thoughts' (critiques/alternatives/connections) via LLM calls to expand the analysis tree in diverse ways.- **Simulation:** Evaluating the quality of analysis nodes using LLM calls based on criteria like insight, novelty, relevance, and depth compared to previous bests.- **Backpropagation:** Updating scores (or Beta distribution parameters) and visit counts up the tree path.- **State Persistence:** (Optional) Saving key results (best analysis, score, tags) and learned approach preferences between your turns within this chat using a local database (`{db_file_name}`).- **Intent Handling:** Trying to understand if you want a new analysis, to continue the last one, or ask about results/process/config.You can adjust parameters like exploration weight, iterations, priors, surprise sensitivity, etc., using the Valves settings in the UI.".format(
            db_file_name=os.path.basename(DB_FILE)
        )  # Show only filename
        await self.emit_message(f"**About My Process:**\n{explanation}")
        await self.done()
        return None  # Signal completion

    async def handle_ask_config(self, current_config: dict):
        logger.info("Handling intent: ASK_CONFIG")
        try:
            # Filter config for display (optional, shows all for now)
            # Maybe exclude very low-level or redundant items if needed
            config_to_display = current_config.copy()
            # Convert to JSON string for clean formatting
            config_str = json.dumps(config_to_display, indent=2)
            await self.emit_message(
                f"**Current Run Configuration:**\n```json\n{config_str}\n```"
            )
        except Exception as e:
            logger.error(f"Failed to format/emit config: {e}")
            await self.emit_message("**Error:** Could not display configuration.")
        await self.done()
        return None  # Signal completion

    async def handle_ask_last_run_summary(self, loaded_state: Optional[dict]):
        logger.info("Handling intent: ASK_LAST_RUN_SUMMARY")
        if not loaded_state:
            await self.emit_message(
                "I don't have any saved results from a previous analysis run in this chat session."
            )
        else:
            try:
                summary = "**Summary of Last Analysis Run:**\n"
                summary += f"- **Best Score:** {loaded_state.get('best_score', 'N/A'):.1f}/10\n"
                summary += f"- **Best Analysis Tags:** {', '.join(loaded_state.get('best_node_tags', [])) or 'N/A'}\n"
                summary += f"- **Best Analysis Summary:** {loaded_state.get('best_solution_summary', 'N/A')}\n"

                priors = loaded_state.get("approach_priors")
                if priors and "alpha" in priors and "beta" in priors:
                    means = {}
                    alphas = priors.get("alpha", {})
                    betas = priors.get("beta", {})
                    for app, alpha in alphas.items():
                        beta = betas.get(app, 1.0)  # Default beta if missing
                        alpha = max(1e-9, alpha)
                        beta = max(1e-9, beta)  # Safety
                        if alpha + beta > 1e-9:
                            means[app] = (alpha / (alpha + beta)) * 10
                    if means:
                        sorted_means = sorted(
                            means.items(), key=lambda item: item[1], reverse=True
                        )
                        top_approaches = [
                            f"{app} ({score:.1f})" for app, score in sorted_means[:3]
                        ]
                        summary += f"- **Learned Approach Preferences:** Favored {', '.join(top_approaches)}...\n"
                    else:
                        summary += f"- **Learned Approach Preferences:** (No valid priors loaded)\n"

                unfit = loaded_state.get("unfit_markers", [])
                if unfit:
                    # Show summary of first unfit marker found
                    first_unfit = unfit[0]
                    summary += f"- **Potential Unfit Areas Noted:** {len(unfit)} (e.g., '{first_unfit.get('summary','...')}' due to {first_unfit.get('reason','...')})\n"
                else:
                    summary += f"- **Potential Unfit Areas Noted:** None\n"

                await self.emit_message(summary)
            except Exception as e:
                logger.error(f"Error formatting last run summary: {e}")
                await self.emit_message(
                    "**Error:** Could not display summary of last run."
                )

        await self.done()
        return None  # Signal completion

    # Terrible way to handle this!! Why do we have a hardcoded response rather than letting the LLM decide what to say?
    async def handle_general_conversation(self):
        logger.info("Handling intent: GENERAL_CONVERSATION")
        # Slightly more helpful response
        response = "I understand you might be saying hello or thank you! My primary role is to perform analysis using Monte Carlo Tree Search. You can: 1.  Provide text or a question for a **new analysis**. 2.  Ask me to **continue** or **elaborate on** the previous analysis run (if one exists in this chat). 3.  Ask about the **results of the last run**. 4.  Ask **how I work** or about my **current settings**. How can I help you with an analysis task?"
        await self.emit_message(response)
        await self.done()
        return None  # Signal completion

    # <<< MODIFIED: Main pipe method with intent handling and state logic (Indented within Pipe) >>>
    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        __task__=None,
        __model__=None,  # Note: __model__ might be None if called directly
    ) -> Union[str, AsyncGenerator[str, None], None]:

        self.__current_event_emitter__ = (
            __event_emitter__  # Store emitter for use by helpers
        )
        mcts_instance = None
        current_config = default_config.copy()  # Start with defaults for this run
        initial_analysis_text = ""
        loaded_state_dict = None  # Holds deserialized state from DB
        debug_this_run = False
        show_chat_details = False
        state_persistence_enabled = True  # Default based on new valve

        conn = None  # DB connection managed within scope

        try:
            # --- 1. Initial Setup & Resolve Input ---
            pipe_model_id = body.get("model", "")  # Model ID used to call the pipe
            resolved_model = self.resolve_model(body)  # Actual base model name
            if not resolved_model:
                # Use emit_message IF emitter is set, otherwise log
                if self.__current_event_emitter__:
                    await self.emit_message("**Error:** No model identified.")
                else:
                    logger.error("Error: No model identified.")
                if self.__current_event_emitter__:
                    await self.done()
                return "Error: Model not identified."
            self.__model__ = resolved_model  # Store resolved model for internal use

            raw_input_text = self.resolve_question(body)
            if not raw_input_text:
                if self.__current_event_emitter__:
                    await self.emit_message("**Error:** No input text provided.")
                else:
                    logger.error("Error: No input text provided.")
                if self.__current_event_emitter__:
                    await self.done()
                return "Error: No input text."
            self.__question__ = raw_input_text  # Store raw input

            chat_id = body.get("chat_id")
            if not chat_id:
                logger.warning(
                    "chat_id not found in body. State persistence will be disabled for this request."
                )
                # Don't emit to chat here, it's a backend detail mostly.
                # state_persistence_enabled = False # Handled below after valve check

            # --- 2. Apply Valve Settings ---
            logger.debug("Applying Valve settings...")
            try:
                # Update self.valves instance based on request body if present
                # This assumes OpenWebUI might pass 'valves' in the 'body' dictionary
                request_valves_data = body.get("valves")
                if request_valves_data and isinstance(request_valves_data, dict):
                    try:
                        # Validate and create a new Valves instance from request data
                        self.valves = self.Valves(**request_valves_data)
                        logger.info(
                            "Loaded and validated Valves settings from request body."
                        )
                    except (
                        Exception
                    ) as validation_error:  # Catch Pydantic validation errors etc.
                        logger.error(
                            f"Failed to validate Valves from request: {validation_error}. Using defaults.",
                            exc_info=True,
                        )
                        self.valves = self.Valves()  # Revert to default Valves
                        if self.__current_event_emitter__:
                            await self.emit_message(
                                f"**Warning:** Invalid settings provided. Using defaults."
                            )
                else:
                    # If no valves in request, use the existing self.valves (initialized in __init__)
                    logger.info(
                        "No Valves settings in request body, using current self.valves."
                    )

                # Apply the final self.valves settings to the current_config for this run
                valve_dict = self.valves.model_dump()
                for key_upper, value in valve_dict.items():
                    key_lower = key_upper.lower()  # Config uses lower case keys
                    if key_lower in current_config:
                        current_config[key_lower] = value
                    # else: logger.warning(f"Valve key '{key_upper}' not found in default_config.") # Optional: warn about extra valves

                # Update specific runtime flags based on config
                debug_this_run = current_config["debug_logging"]
                show_chat_details = current_config["show_processing_details"]
                state_persistence_enabled = current_config.get(
                    "enable_state_persistence", True
                )
                setup_logger(
                    logging.DEBUG if debug_this_run else logging.INFO
                )  # Re-setup logger level

                # Validate/sanitize crucial numeric config values AFTER applying valves
                current_config["beta_prior_alpha"] = max(
                    1e-9, current_config["beta_prior_alpha"]
                )
                current_config["beta_prior_beta"] = max(
                    1e-9, current_config["beta_prior_beta"]
                )
                current_config["exploration_weight"] = max(
                    0.0, current_config["exploration_weight"]
                )
                current_config["early_stopping_threshold"] = max(
                    1.0, min(10.0, current_config["early_stopping_threshold"])
                )
                current_config["early_stopping_stability"] = max(
                    1, current_config["early_stopping_stability"]
                )
                current_config["unfit_score_threshold"] = max(
                    0.0, min(10.0, current_config.get("unfit_score_threshold", 4.0))
                )
                current_config["unfit_visit_threshold"] = max(
                    1, current_config.get("unfit_visit_threshold", 3)
                )

                logger.info("Valve settings applied to current run config.")
                # Disable state persistence if chat_id is missing, even if enabled by Valve
                if not chat_id and state_persistence_enabled:
                    logger.warning(
                        "State persistence enabled by Valve, but chat_id missing. Disabled for this run."
                    )
                    state_persistence_enabled = False

            except Exception as e:
                logger.error(
                    f"Error applying Valve settings: {e}. Using default configuration.",
                    exc_info=True,
                )
                if self.__current_event_emitter__:
                    await self.emit_message(
                        f"**Warning:** Error applying settings ({e}). Using defaults."
                    )
                current_config = default_config.copy()
                # Revert to defaults on error
                # Reset flags based on reverted defaults
                debug_this_run = current_config["debug_logging"]
                show_chat_details = current_config["show_processing_details"]
                state_persistence_enabled = current_config.get(
                    "enable_state_persistence", True
                )
                if not chat_id:
                    state_persistence_enabled = False  # Still disable if no chat_id
                setup_logger(logging.DEBUG if debug_this_run else logging.INFO)

            # --- 3. Classify Intent ---
            # Pass the raw input text for classification
            intent = await self.classify_intent(raw_input_text)

            # --- 4. Load State (if relevant intent and enabled) ---
            if (
                state_persistence_enabled
                and chat_id
                and intent in ["CONTINUE_ANALYSIS", "ASK_LAST_RUN_SUMMARY"]
            ):
                try:
                    state_json = self._load_mcts_state(chat_id)
                    if state_json:
                        loaded_state_dict = json.loads(state_json)
                        # Basic validation of loaded state (optional)
                        if (
                            not isinstance(loaded_state_dict, dict)
                            or loaded_state_dict.get("version") != "0.8.0"
                        ):
                            logger.warning(
                                f"Loaded state for {chat_id} is invalid or wrong version. Discarding."
                            )
                            loaded_state_dict = None  # Discard invalid state
                            if self.__current_event_emitter__:
                                await self.emit_message(
                                    "**Warning:** Previous state was incompatible. Starting fresh."
                                )
                            if intent == "CONTINUE_ANALYSIS":
                                intent = "ANALYZE_NEW"  # Treat as new if state invalid
                        else:
                            logger.info(
                                "Successfully deserialized and validated loaded state."
                            )
                    else:
                        # If trying to continue but no state found, treat as new analysis
                        if intent == "CONTINUE_ANALYSIS":
                            if self.__current_event_emitter__:
                                await self.emit_message(
                                    "**Info:** No previous analysis state found. Starting a new analysis."
                                )
                            intent = "ANALYZE_NEW"
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Error decoding loaded state JSON for chat_id {chat_id}: {e}"
                    )
                    if self.__current_event_emitter__:
                        await self.emit_message(
                            "**Warning:** Could not load previous state (corrupted data?). Starting fresh."
                        )
                    if intent == "CONTINUE_ANALYSIS":
                        intent = "ANALYZE_NEW"
                    loaded_state_dict = None
                except Exception as e:
                    logger.error(
                        f"Unexpected error loading state for chat_id {chat_id}: {e}",
                        exc_info=True,
                    )
                    if self.__current_event_emitter__:
                        await self.emit_message(
                            "**Warning:** Error loading previous state. Starting fresh."
                        )
                    if intent == "CONTINUE_ANALYSIS":
                        intent = "ANALYZE_NEW"
                    loaded_state_dict = None

            # --- 5. Dispatch Based on Intent ---
            # Ensure emitter is available before calling handlers that use it
            if not self.__current_event_emitter__:
                logger.error(
                    "Event emitter is not set. Cannot proceed with intent handling."
                )
                return "Error: Internal setup error (emitter missing)."

            if intent == "ASK_PROCESS":
                return await self.handle_ask_process(
                    raw_input_text
                )  # Returns None on success
            elif intent == "ASK_CONFIG":
                return await self.handle_ask_config(
                    current_config
                )  # Returns None on success
            elif intent == "ASK_LAST_RUN_SUMMARY":
                # Pass the loaded state dictionary
                return await self.handle_ask_last_run_summary(
                    loaded_state_dict
                )  # Returns None on success
            elif intent == "GENERAL_CONVERSATION":
                return (
                    await self.handle_general_conversation()
                )  # Returns None on success
            elif intent in ["ANALYZE_NEW", "CONTINUE_ANALYSIS"]:
                # --- Proceed with MCTS Analysis ---
                analysis_target_text = raw_input_text  # The user's core request text

                # a. Handle Title Generation Task (Check *after* intent)
                if __task__ == TASKS.TITLE_GENERATION:
                    logger.info(
                        f"Handling TITLE_GENERATION task for: {truncate_text(analysis_target_text)}"
                    )
                    # Use get_completion for a simple, non-streaming task
                    completion = await self.get_completion(
                        self.__model__,
                        [
                            {
                                "role": "user",
                                "content": f"Generate a concise title for the following text: {analysis_target_text}",
                            }
                        ],
                    )
                    await self.done()
                    # Format title nicely
                    title_result = (
                        truncate_text(completion, 70).replace("\n", " ").strip()
                    )
                    return (
                        f"Title Suggestion: {title_result}"
                        if title_result
                        else "Could not generate title."
                    )

                # b. Emit startup messages
                await self.emit_message(
                    f'# {name} v0.8.0\n*Analyzing:* "{truncate_text(analysis_target_text, 100)}" *using model* `{self.__model__}`.'
                )
                if intent == "CONTINUE_ANALYSIS" and loaded_state_dict:
                    await self.emit_message(
                        "🚀 **Continuing MCTS Analysis...** (Loading previous state)"
                    )
                else:
                    await self.emit_message("🚀 **Starting MCTS Analysis...**")
                if show_chat_details:
                    await self.emit_message(
                        "*(Verbose mode enabled: Showing intermediate steps)*"
                    )
                # Log key parameters for the run
                log_params = {
                    k: v
                    for k, v in current_config.items()
                    if k
                    in [
                        "max_iterations",
                        "simulations_per_iteration",
                        "use_bayesian_evaluation",
                        "early_stopping",
                        "exploration_weight",
                    ]
                }
                logger.info(
                    f"--- Run Parameters ---\nIntent: {intent}\nState Loaded: {bool(loaded_state_dict)}\nChat ID: {chat_id}\nConfig: {json.dumps(log_params)}"
                )

                # c. Generate Initial Analysis (Needed for root node content)
                #    Run this even if continuing, as it sets the initial interpretation frame.
                #    The loaded state influences the *subsequent* exploration.
                await self.progress("Generating initial analysis...")
                # Use get_completion for initial analysis (non-streaming better here)
                initial_analysis_returned = await self.get_completion(
                    self.__model__,
                    [
                        {
                            "role": "user",
                            "content": initial_prompt.format(
                                question=analysis_target_text
                            ),
                        }
                    ],
                )
                if (
                    not isinstance(initial_analysis_returned, str)
                    or "Error:" in initial_analysis_returned
                ):
                    logger.error(
                        f"Initial analysis failed: {initial_analysis_returned}"
                    )
                    await self.emit_message(
                        f"**Error:** Initial analysis generation failed: {initial_analysis_returned}"
                    )
                    await self.done()
                    return f"Error: {initial_analysis_returned}"
                initial_analysis_text = initial_analysis_returned.strip()
                # Clean text just in case
                initial_analysis_text = re.sub(
                    r"^```(json|markdown)?\s*",
                    "",
                    initial_analysis_text,
                    flags=re.IGNORECASE | re.MULTILINE,
                )
                initial_analysis_text = re.sub(
                    r"\s*```$", "", initial_analysis_text, flags=re.MULTILINE
                ).strip()
                if not initial_analysis_text:
                    logger.error("Initial analysis result was empty after cleaning.")
                    await self.emit_message(
                        "**Error:** Initial analysis generated empty content."
                    )
                    await self.done()
                    return "Error: Empty initial analysis."
                # Display initial analysis clearly
                await self.emit_message(
                    "\n## Initial Analysis\n" + initial_analysis_text
                )
                await self.emit_message("\n***\n")
                await asyncio.sleep(0.1)  # Separator

                # d. Initialize MCTS Instance
                await self.progress("Initializing MCTS...")
                mcts_init_args = {
                    "llm": self,  # Pass the Pipe instance itself as the 'llm' for MCTS
                    "question": analysis_target_text,
                    "mcts_config": current_config,
                    "initial_analysis_content": initial_analysis_text,
                }
                # Pass loaded state dictionary if continuing
                if intent == "CONTINUE_ANALYSIS" and loaded_state_dict:
                    mcts_init_args["initial_state"] = loaded_state_dict

                mcts_instance = MCTS(**mcts_init_args)
                logger.info("MCTS instance created. Starting search iterations...")

                # e. Run MCTS Loop
                for i in range(current_config["max_iterations"]):
                    iteration_num = i + 1
                    if debug_this_run:
                        logger.info(
                            f"--- Starting Iteration {iteration_num}/{current_config['max_iterations']} ---"
                        )
                    await self.progress(
                        f"Running MCTS Iteration {iteration_num}/{current_config['max_iterations']}..."
                    )
                    best_score_before_iter = mcts_instance.best_score

                    # Run one iteration of search (multiple simulations)
                    search_result_node = await mcts_instance.search(
                        current_config["simulations_per_iteration"]
                    )
                    mcts_instance.iterations_completed += (
                        1  # Increment after iteration finishes
                    )

                    # Conditional Iteration Summary (Verbose Mode)
                    if show_chat_details:
                        iter_best_node = mcts_instance.find_best_final_node()
                        iter_summary_msg = (
                            f"\n**--- Iteration {iteration_num} Summary ---**\n"
                        )
                        iter_summary_msg += f"- Overall Best Score So Far: {mcts_instance.best_score:.1f}/10"
                        if mcts_instance.best_score > best_score_before_iter:
                            iter_summary_msg += " (✨ New best found this iteration!)"
                        else:
                            iter_summary_msg += " (Best score unchanged)"
                        if iter_best_node:
                            tags_str = (
                                f"Tags: {iter_best_node.descriptive_tags}"
                                if iter_best_node.descriptive_tags
                                else "Tags: []"
                            )
                            iter_summary_msg += f"\n- Current Best Node: {iter_best_node.sequence} ({tags_str})"
                        iter_summary_msg += "\n-------------------------------\n"
                        await self.emit_message(iter_summary_msg)

                    # Check for early stopping condition *after* the iteration completes
                    # Check if search itself signaled an early stop (returned None)
                    if search_result_node is None:
                        logger.info(
                            f"Search signaled early stopping after iteration {iteration_num}."
                        )
                        # Message already emitted by search if verbose
                        break  # Exit MCTS loop

                    # Check stability condition if search didn't stop early
                    stability_met = (
                        mcts_instance.high_score_counter
                        >= current_config["early_stopping_stability"]
                    )
                    threshold_met = (
                        mcts_instance.best_score
                        >= current_config["early_stopping_threshold"]
                    )
                    if (
                        current_config["early_stopping"]
                        and threshold_met
                        and stability_met
                    ):
                        logger.info(
                            f"Early stopping stability criteria met after iteration {iteration_num}."
                        )
                        await self.emit_message(
                            f"**Stopping early:** Analysis score ({mcts_instance.best_score:.1f}/10) reached threshold and maintained stability."
                        )
                        break  # Exit MCTS loop

                    await asyncio.sleep(
                        0.05
                    )  # Small delay between iterations for UI responsiveness

                # --- End of MCTS Loop ---
                logger.info("MCTS iterations finished.")
                await self.emit_message(
                    "\n🏁 **MCTS Exploration Finished.** Generating final output..."
                )

                # Determine final best analysis content safely
                final_best_analysis_text = (
                    initial_analysis_text  # Fallback to initial if MCTS failed badly
                )
                if (
                    mcts_instance
                    and isinstance(mcts_instance.best_solution, str)
                    and mcts_instance.best_solution.strip()
                ):
                    final_best_analysis_text = mcts_instance.best_solution
                else:
                    logger.warning(
                        "MCTS best_solution was invalid or empty. Using initial analysis as final result."
                    )

                # f. Generate Final Output (Quiet/Verbose)
                if show_chat_details:  # Verbose Mode
                    await self.progress("Generating verbose summary...")
                    # Ensure mcts_instance exists before calling its methods
                    final_summary_output = (
                        mcts_instance.formatted_output(final_output=True)
                        if mcts_instance
                        else "Error: MCTS instance lost."
                    )
                    await self.emit_message(final_summary_output)
                else:  # Quiet Mode (Default)
                    await self.progress("Extracting best analysis...")
                    # Ensure mcts_instance exists
                    if mcts_instance:
                        best_node = mcts_instance.find_best_final_node()
                        tags_str = (
                            f"Tags: {best_node.descriptive_tags}"
                            if best_node and best_node.descriptive_tags
                            else "Tags: []"
                        )
                        quiet_summary = f"## Best Analysis Found (Score: {mcts_instance.best_score:.1f}/10)\n"
                        quiet_summary += f"**{tags_str}**\n\n"
                        best_solution_cleaned = str(
                            final_best_analysis_text
                        )  # Use the determined best text
                        # Clean again just in case
                        best_solution_cleaned = re.sub(
                            r"^```(json|markdown)?\s*",
                            "",
                            best_solution_cleaned,
                            flags=re.IGNORECASE | re.MULTILINE,
                        )
                        best_solution_cleaned = re.sub(
                            r"\s*```$", "", best_solution_cleaned, flags=re.MULTILINE
                        ).strip()
                        quiet_summary += f"{best_solution_cleaned}\n"
                        await self.emit_message(quiet_summary)
                    else:
                        await self.emit_message(
                            "## Best Analysis Found\nError: Could not retrieve final analysis."
                        )

                # g. Generate Final Synthesis Step
                await self.progress("Generating final synthesis...")
                synthesis_text_generated = ""
                # Ensure mcts_instance exists before synthesis
                if mcts_instance:
                    try:
                        best_node_final = mcts_instance.find_best_final_node()
                        path_thoughts_list = []
                        path_to_best = []
                        temp_node = best_node_final
                        # Traverse up from the best node to the root
                        while temp_node:
                            path_to_best.append(temp_node)
                            temp_node = temp_node.parent
                        path_to_best.reverse()  # Reverse to get path from root to best

                        # Collect thoughts along the path
                        for node in path_to_best:
                            # Include thought only if it exists and node isn't the root (root has no preceding thought)
                            if node.thought and node.parent:
                                path_thoughts_list.append(
                                    f"- (Node {node.sequence} based on {node.parent.sequence}): {node.thought.strip()}"
                                )

                        path_thoughts_str = (
                            "\n".join(path_thoughts_list)
                            if path_thoughts_list
                            else "No significant development path identified."
                        )

                        synthesis_context = {
                            "question_summary": mcts_instance.question_summary,  # Summary of the analyzed text
                            "initial_analysis_summary": truncate_text(
                                initial_analysis_text, 300
                            ),
                            "best_score": f"{mcts_instance.best_score:.1f}",
                            "path_thoughts": path_thoughts_str,
                            "final_best_analysis_summary": truncate_text(
                                final_best_analysis_text, 400
                            ),
                        }
                        # Use get_completion for synthesis (non-streaming usually better)
                        synthesis_text_generated = await self.get_completion(
                            self.__model__,
                            [
                                {
                                    "role": "user",
                                    "content": final_synthesis_prompt.format(
                                        **synthesis_context
                                    ),
                                }
                            ],
                        )
                        if "Error:" in synthesis_text_generated:
                            logger.error(
                                f"Synthesis generation failed: {synthesis_text_generated}"
                            )
                            await self.emit_message(
                                f"**Warning:** Synthesis generation failed: {synthesis_text_generated}"
                            )
                        else:
                            synthesis_text_cleaned = synthesis_text_generated.strip()
                            synthesis_text_cleaned = re.sub(
                                r"^```(json|markdown)?\s*",
                                "",
                                synthesis_text_cleaned,
                                flags=re.IGNORECASE | re.MULTILINE,
                            )
                            synthesis_text_cleaned = re.sub(
                                r"\s*```$",
                                "",
                                synthesis_text_cleaned,
                                flags=re.MULTILINE,
                            ).strip()
                            if synthesis_text_cleaned:
                                await self.emit_message(
                                    "\n***\n## Final Synthesis\n"
                                    + synthesis_text_cleaned
                                )
                            else:
                                await self.emit_message(
                                    "\n***\n## Final Synthesis\n(Synthesis generation resulted in empty content.)"
                                )
                    except Exception as synth_err:
                        logger.error(
                            f"Final synthesis step failed: {synth_err}",
                            exc_info=debug_this_run,
                        )
                        await self.emit_message(
                            "\n***\n## Final Synthesis\n**Error:** Failed to generate final synthesis."
                        )
                else:
                    await self.emit_message(
                        "\n***\n## Final Synthesis\n**Error:** Could not generate synthesis (MCTS instance missing)."
                    )

                # h. Serialize & Save State (if enabled and chat_id exists)
                if state_persistence_enabled and chat_id and mcts_instance:
                    await self.progress("Saving analysis state...")
                    try:
                        state_json = self._serialize_mcts_state(
                            mcts_instance, current_config
                        )
                        if state_json != "{}":
                            self._save_mcts_state(chat_id, state_json)
                        else:
                            logger.warning(
                                "Serialization produced empty state, not saving."
                            )
                    except Exception as save_err:
                        logger.error(
                            f"Failed to save MCTS state for chat_id {chat_id}: {save_err}",
                            exc_info=True,
                        )
                        await self.emit_message(
                            "**Warning:** Failed to save analysis state."
                        )

                # i. Return None (pipe completes successfully)
                await self.done()
                logger.info(f"Pipe '{name}' finished analysis intent successfully.")
                return (
                    None  # Indicate successful completion of the generator/async task
                )

            else:  # Should not happen if classify_intent covers all cases
                logger.error(f"Unhandled intent after classification: {intent}")
                await self.emit_message(
                    "**Error:** Internal error handling input intent."
                )
                await self.done()
                return "Error: Unhandled intent"
            # --- End Dispatch ---

        except Exception as e:
            # --- Fatal Error Handling ---
            logger.error(f"FATAL Pipe Error in '{name}': {e}", exc_info=True)
            error_message = f"\n\n**FATAL ERROR:**\n```\n{type(e).__name__}: {str(e)}\n```\nProcessing stopped unexpectedly. Please check the application logs."
            # Try to emit the error message if the emitter is still available
            if self.__current_event_emitter__:
                try:
                    await self.emit_message(error_message)
                except Exception as emit_err:
                    logger.error(f"Failed to emit fatal error message: {emit_err}")
            # Ensure done is called in fatal error path too
            if self.__current_event_emitter__:
                try:
                    await self.done()
                except Exception:
                    pass  # Ignore errors during cleanup done call
            # Return an error string for the UI
            return f"Error: Pipe failed unexpectedly. Check logs. ({type(e).__name__})"
        finally:
            # --- Cleanup (ensure DB conn closed if opened directly - handled by context manager if used) ---
            # No explicit conn.close() needed if using _get_db_connection which handles it.
            # Call standard cleanup if needed
            await self.cleanup()

    # --- LLM Interaction & Helper Methods (Indented within Pipe) ---
    # These methods rely on self.__current_event_emitter__ and self.__model__ set in pipe()

    async def progress(self, message: str):
        # Safely check if emitter exists and is callable
        if self.__current_event_emitter__ and callable(self.__current_event_emitter__):
            debug = default_config.get(
                "debug_logging", False
            )  # Use default config as fallback if instance config not set yet
            if hasattr(self, "valves"):
                debug = self.valves.DEBUG_LOGGING  # Check instance valves if available
            try:
                if debug:
                    logger.debug(f"Progress Update: {message}")
                await self.__current_event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "level": "info",
                            "description": str(message),
                            "done": False,
                        },
                    }
                )
            except Exception as e:
                logger.error(f"Emit progress error: {e}")
        # else: logger.warning("Progress called but emitter not set.") # Optional: log if called too early

    async def done(self):
        if self.__current_event_emitter__ and callable(self.__current_event_emitter__):
            debug = default_config.get("debug_logging", False)
            if hasattr(self, "valves"):
                debug = self.valves.DEBUG_LOGGING
            try:
                if debug:
                    logger.debug("Sending 'done' status event.")
                await self.__current_event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "level": "info",
                            "description": "Processing Complete",
                            "done": True,
                        },
                    }
                )
            except Exception as e:
                logger.error(f"Emit done error: {e}")
            # Clear emitter ref after sending done? Not strictly necessary if instance is short-lived.
            # self.__current_event_emitter__ = None
        # else: logger.warning("Done called but emitter not set.")

    async def emit_message(self, message: str):
        if self.__current_event_emitter__ and callable(self.__current_event_emitter__):
            try:
                # Ensure message is a string
                await self.__current_event_emitter__(
                    {"type": "message", "data": {"content": str(message)}}
                )
            except Exception as e:
                logger.error(f"Emit message error: {e} (Msg: {str(message)[:100]}...)")
        # else: logger.warning("Emit_message called but emitter not set.")

    async def emit_replace(self, message: str):  # Not used currently but keep
        if self.__current_event_emitter__ and callable(self.__current_event_emitter__):
            try:
                await self.__current_event_emitter__(
                    {"type": "replace", "data": {"content": str(message)}}
                )
            except Exception as e:
                logger.error(f"Emit replace error: {e}")
        # else: logger.warning("Emit_replace called but emitter not set.")

    def get_chunk_content(self, chunk_bytes: bytes) -> Generator[str, None, None]:
        # (implementation unchanged)
        debug = default_config.get("debug_logging", False)
        if hasattr(self, "valves"):
            debug = self.valves.DEBUG_LOGGING
        try:
            chunk_str = chunk_bytes.decode("utf-8")
            for line in chunk_str.splitlines():
                line = line.strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    json_data_str = line[
                        len("data: ") :
                    ]  # Use slicing instead of fixed index
                    try:
                        chunk_data = json.loads(json_data_str)
                        # Check structure more carefully
                        if (
                            isinstance(chunk_data, dict)
                            and "choices" in chunk_data
                            and isinstance(chunk_data["choices"], list)
                            and chunk_data["choices"]  # Ensure list is not empty
                            and isinstance(
                                chunk_data["choices"][0], dict
                            )  # Check choice is dict
                            and isinstance(chunk_data["choices"][0].get("delta"), dict)
                        ):
                            content = chunk_data["choices"][0]["delta"].get("content")
                            # Yield content only if it's a non-empty string
                            if isinstance(content, str) and content:
                                yield content
                    except json.JSONDecodeError:
                        logger.warning(
                            f"JSON decode error in stream chunk: {json_data_str}"
                        )
                    except (IndexError, KeyError, TypeError) as e:
                        logger.error(
                            f"Error processing stream chunk structure: {e} - Data: {json_data_str}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Unexpected error processing stream chunk data: {e}"
                        )
        except UnicodeDecodeError:
            logger.error(
                f"Unicode decode error in stream chunk: {chunk_bytes[:100]}..."
            )
        except Exception as e:
            logger.error(f"Error decoding/splitting stream chunk: {e}", exc_info=debug)
        # Ensure generator finishes cleanly
        return

    async def get_streaming_completion(
        self, model: str, messages: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        # (implementation unchanged)
        response = None
        debug = default_config.get("debug_logging", False)
        if hasattr(self, "valves"):
            debug = self.valves.DEBUG_LOGGING
        try:
            # Ensure model is specified
            if not model:
                logger.error("get_streaming_completion called with empty model name.")
                yield "Error: Model name not specified for LLM call."
                return

            response = await self.call_ollama_endpoint_function(
                {"model": model, "messages": messages, "stream": True}
            )

            # Handle explicit error response from endpoint function
            if isinstance(response, dict) and response.get("error"):
                err_msg = self.get_response_content(response)  # Extract error message
                logger.error(f"LLM stream initiation failed: {err_msg}")
                yield f"Error: {err_msg}"
                # Yield the error message
                return

            # Check if response has the expected streaming iterator
            if hasattr(response, "body_iterator"):
                async for chunk_bytes in response.body_iterator:
                    # Yield content parts from the chunk parser
                    for part in self.get_chunk_content(chunk_bytes):
                        if part:
                            yield part
            # Handle non-streaming response if received unexpectedly
            elif isinstance(response, dict):
                content = self.get_response_content(response)
                if content:
                    logger.warning(
                        "Expected stream response, but received a single dictionary. Yielding full content."
                    )
                    yield content
                else:
                    logger.error(
                        f"Expected stream, but got invalid dict response: {str(response)[:200]}"
                    )
                    yield "Error: Invalid LLM response dictionary."
            else:
                # Handle unexpected response type
                logger.error(
                    f"Expected streaming response or dict, but got type: {type(response)}."
                )
                yield f"Error: Unexpected LLM response type ({type(response)})."

        except AttributeError as ae:
            # More specific error for attribute errors (often indicates response object issues)
            logger.error(
                f"AttributeError during streaming (likely response object issue): {ae}",
                exc_info=debug,
            )
            yield f"Error during streaming: {str(ae)}"
        except Exception as e:
            logger.error(f"LLM stream processing error: {e}", exc_info=debug)
            yield f"Error during streaming: {str(e)}"
        finally:
            # Ensure response resources are released if applicable
            if (
                response is not None
                and hasattr(response, "release")
                and callable(response.release)
            ):
                try:
                    await response.release()
                except Exception as release_err:
                    logger.error(f"Error releasing stream response: {release_err}")

    async def get_message_completion(
        self, model: str, content: str
    ) -> AsyncGenerator[str, None]:  # Helper for single message stream
        # (implementation unchanged)
        debug = default_config.get("debug_logging", False)
        if hasattr(self, "valves"):
            debug = self.valves.DEBUG_LOGGING
        try:
            # Use async for to iterate through the streaming completion
            async for chunk in self.get_streaming_completion(
                model, [{"role": "user", "content": str(content)}]
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Error in get_message_completion: {e}", exc_info=debug)
            yield f"Error: {str(e)}"

    async def get_completion(
        self, model: str, messages: List[Dict[str, str]]
    ) -> str:  # Non-streaming call
        # (implementation unchanged)
        response = None
        debug = default_config.get("debug_logging", False)
        if hasattr(self, "valves"):
            debug = self.valves.DEBUG_LOGGING
        try:
            # Ensure model is specified
            if not model:
                logger.error("get_completion called with empty model name.")
                return "Error: Model name not specified for LLM call."

            response = await self.call_ollama_endpoint_function(
                {"model": model, "messages": messages, "stream": False}
            )
            # Extract content using the helper function, which handles error structure
            content = self.get_response_content(response)

            # Check if the extracted content indicates an error
            if isinstance(response, dict) and response.get("error"):
                logger.error(f"Non-streaming LLM call failed: {content}")
                # Return the error message extracted by get_response_content
                return f"Error: {content}" if content else "Error: Unknown LLM failure."

            # Return the successful content
            return (
                content if content else ""
            )  # Return empty string if content is None or empty

        except Exception as e:
            logger.error(f"Error in get_completion: {e}", exc_info=debug)
            return f"Error: LLM call failed ({str(e)})."
        finally:
            # Ensure response resources are released if applicable
            if (
                response is not None
                and hasattr(response, "release")
                and callable(response.release)
            ):
                try:
                    await response.release()
                except Exception as release_err:
                    logger.error(f"Error releasing non-stream response: {release_err}")

    async def call_ollama_endpoint_function(
        self, payload: Dict[str, Any]
    ):  # Internal call to Ollama endpoint
        # (implementation unchanged)
        debug = default_config.get("debug_logging", False)
        if hasattr(self, "valves"):
            debug = self.valves.DEBUG_LOGGING
        try:
            # Mock a FastAPI request object
            async def receive():
                return {
                    "type": "http.request",
                    "body": json.dumps(payload).encode("utf-8"),
                }

            mock_request = Request(
                scope={
                    "type": "http",
                    "headers": [],
                    "method": "POST",
                    "scheme": "http",
                    "server": ("local", 80),
                    "path": "/api/ollama/generate",
                    "query_string": b"",
                    "client": ("127.0.0.1", 8080),
                    "app": app,
                },
                receive=receive,
            )

            if debug:
                logger.debug(
                    f"Calling internal ollama endpoint: {str(payload)[:200]}..."
                )
            # Call the actual endpoint function from open_webui.routers.ollama
            # Ensure the function signature matches what's expected (request, form_data, user)
            response = await ollama.generate_openai_chat_completion(
                request=mock_request, form_data=payload, user=admin
            )

            # Debug log the response type if not easily identifiable
            if (
                debug
                and not isinstance(response, dict)
                and not hasattr(response, "body_iterator")
            ):
                logger.debug(f"Internal endpoint response type: {type(response)}")

            return response
        except Exception as e:
            logger.error(f"Ollama internal call error: {str(e)}", exc_info=debug)
            # Return an error structure consistent with Ollama API for non-streaming errors
            error_content = (
                f"Error: LLM internal call failed ({str(e)[:100]}...). See logs."
            )
            return {
                "error": True,
                "choices": [
                    {"message": {"role": "assistant", "content": error_content}}
                ],
            }

    # <<< MODIFIED: stream_prompt_completion handles new context vars >>>
    async def stream_prompt_completion(
        self, prompt: str, emit_chunks: bool = True, **format_args
    ) -> str:
        "Streams completion for a prompt, optionally emitting chunks via emit_message, and always returns the final aggregated and cleaned string. Handles potential missing keys in format_args gracefully."
        debug = default_config.get("debug_logging", False)
        if hasattr(self, "valves"):
            debug = self.valves.DEBUG_LOGGING
        complete_response = ""
        error_occurred = False

        # Safely format the prompt, providing defaults for potentially missing keys
        # Convert all values to strings first
        safe_format_args = {
            k: str(v) if v is not None else "" for k, v in format_args.items()
        }
        # Ensure ALL keys potentially used in ANY prompt have defaults
        keys_needed = [
            "question",
            "question_summary",
            "best_answer",
            "best_score",
            "current_answer",
            "current_sequence",
            "current_tags",
            "answer",
            "improvements",
            "answer_to_evaluate",
            "analysis_text",
            "initial_analysis_summary",
            "path_thoughts",
            "final_best_analysis_summary",
            "previous_best_summary",
            "unfit_markers_summary",
            "learned_approach_summary",
            "tree_depth",
            "branches",
            "approach_types",
            "explored_approaches",
            "high_scoring_examples",
            "sibling_approaches",  # Add any other keys used in prompts
        ]
        for key in keys_needed:
            safe_format_args.setdefault(key, "N/A")  # Provide a default like N/A

        try:
            formatted_prompt = prompt.format(**safe_format_args)
        except KeyError as e:
            err_msg = f"Error: Prompt formatting key error: '{e}'. Check prompt structure and default keys."
            logger.error(
                f"{err_msg} Available keys subset: {list(safe_format_args.keys())[:15]}..."
            )
            if emit_chunks and self.__current_event_emitter__:
                await self.emit_message(f"**{err_msg}**")
            return err_msg  # Return error string
        except Exception as e:
            err_msg = f"Error: Prompt formatting failed: {e}."
            logger.error(err_msg, exc_info=debug)
            if emit_chunks and self.__current_event_emitter__:
                await self.emit_message(f"**{err_msg}**")
            return err_msg  # Return error string

        # Ensure model is set before streaming
        if not self.__model__:
            logger.error("stream_prompt_completion called before model was resolved.")
            return "Error: Internal setup error (model not set for streaming)."

        # Stream the response and conditionally emit chunks
        try:
            async for chunk in self.get_message_completion(
                self.__model__, formatted_prompt
            ):
                if chunk is not None:
                    chunk_str = str(chunk)
                    # Check for internal error messages from streaming first
                    if chunk_str.startswith(
                        "Error during streaming:"
                    ) or chunk_str.startswith("Error:"):
                        logger.error(f"LLM stream error received: {chunk_str}")
                        # Emit the error chunk itself if allowed and emitter exists
                        if emit_chunks and self.__current_event_emitter__:
                            await self.emit_message(f"**{chunk_str}**")
                        complete_response = chunk_str  # Store error as final result
                        error_occurred = True
                        break  # Stop processing more chunks

                    # Emit valid chunk to UI if enabled and emitter exists
                    if emit_chunks and self.__current_event_emitter__:
                        await self.emit_message(chunk_str)
                    # Always append to the full response
                    complete_response += chunk_str

            if error_occurred:
                return complete_response  # Return the captured error message

            # Clean the final aggregated response *after* streaming/aggregation
            clean_response = str(complete_response).strip()
            # Remove common conversational closing remarks more aggressively
            clean_response = re.sub(
                r"\n*\s*(?:Would you like me to.*?|Do you want to explore.*?|Is there anything else.*?|Let me know if.*?)\??\s*$",
                "",
                clean_response,
                flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
            ).strip()
            # Remove potential markdown code blocks
            clean_response = re.sub(
                r"^```(json|markdown|text)?\s*",
                "",
                clean_response,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            clean_response = re.sub(
                r"\s*```$", "", clean_response, flags=re.MULTILINE
            ).strip()

            return clean_response  # Return the full, cleaned response string

        except Exception as e:
            err_msg = f"Error: LLM stream processing failed: {e}."
            logger.error(err_msg, exc_info=debug)
            if emit_chunks and self.__current_event_emitter__:
                await self.emit_message(f"**{err_msg}**")
            return err_msg  # Return error message

    # --- MCTS Specific LLM Call Wrappers (Now use context-aware helpers - Indented within Pipe) ---
    async def generate_thought(
        self, current_analysis: str, context: Dict, config: Dict
    ) -> str:
        # Use get_completion for thoughts - less need for streaming here
        format_args = context.copy()
        # Already contains loaded state context from get_context_for_node
        result = await self.get_completion(
            self.__model__,
            [{"role": "user", "content": thoughts_prompt.format(**format_args)}],
        )
        # Return thought or error string, basic cleaning
        return (
            str(result).strip()
            if isinstance(result, str)
            else "Error: Invalid thought format."
        )

    async def update_approach(
        self, original_analysis: str, critique: str, context: Dict, config: Dict
    ) -> str:
        # Use get_completion for updates - less need for streaming here
        format_args = context.copy()
        format_args["answer"] = original_analysis
        format_args["improvements"] = critique.strip()
        # Ensure all keys are present with defaults
        format_args.setdefault("question_summary", "N/A")
        format_args.setdefault("best_answer", "N/A")
        format_args.setdefault("best_score", "N/A")
        format_args.setdefault("current_tags", "None")
        format_args.setdefault("previous_best_summary", "N/A")
        format_args.setdefault("unfit_markers_summary", "None")

        result = await self.get_completion(
            self.__model__,
            [{"role": "user", "content": update_prompt.format(**format_args)}],
        )

        if not isinstance(result, str) or "Error:" in result:
            logger.error(f"Update approach failed: {result}")
            return str(original_analysis)  # Fallback to original on error
        if result.strip():
            clean_result = re.sub(
                r"^```(json|markdown)?\s*",
                "",
                result,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            clean_result = re.sub(
                r"\s*```$", "", clean_result, flags=re.MULTILINE
            ).strip()
            return (
                clean_result if clean_result else str(original_analysis)
            )  # Fallback if empty after cleaning
        logger.warning("Update approach result was empty or invalid. Falling back.")
        return str(original_analysis)

    async def evaluate_answer(
        self, analysis_to_evaluate: str, context: Dict, config: Dict
    ) -> int:
        # Use get_completion for evaluation - expecting a single number
        format_args = context.copy()
        format_args["answer_to_evaluate"] = analysis_to_evaluate
        # Ensure all keys are present with defaults
        format_args.setdefault("question_summary", "N/A")
        format_args.setdefault("best_answer", "N/A")
        format_args.setdefault("best_score", "N/A")
        format_args.setdefault("current_tags", "None")
        format_args.setdefault("previous_best_summary", "N/A")
        format_args.setdefault("unfit_markers_summary", "None")

        result = await self.get_completion(
            self.__model__,
            [{"role": "user", "content": eval_answer_prompt.format(**format_args)}],
        )

        if not isinstance(result, str) or "Error:" in result:
            logger.warning(
                f"Evaluation call failed or returned error: {result}. Defaulting to score 5."
            )
            return 5

        # Try strict match first (just the number, potentially surrounded by whitespace)
        score_match = re.search(r"^\s*([1-9]|10)\s*$", result.strip())
        if score_match:
            try:
                return int(score_match.group(1))
            except ValueError:
                logger.warning(
                    f"Eval parse error (strict): '{result}'. Defaulting to 5."
                )
                return 5
        else:
            # Try relaxed match (find first number 1-10 in the string)
            logger.warning(
                f"Eval strict score not found in: '{result}'. Trying relaxed match."
            )
            relaxed_match = re.search(r"\b([1-9]|10)\b", result.strip())
            if relaxed_match:
                try:
                    score_val = int(relaxed_match.group(1))
                    logger.info(
                        f"Evaluation relaxed match found score: {score_val} in '{result}'"
                    )
                    return score_val
                except ValueError:
                    logger.warning(
                        f"Eval parse error (relaxed): '{result}'. Defaulting to 5."
                    )
                    return 5
            else:
                # Final fallback if no number found
                logger.warning(
                    f"Eval score not found even with relaxed match: '{result}'. Defaulting to 5."
                )
                return 5

    def get_response_content(
        self, response: Union[Dict, Any]
    ) -> str:  # Extracts content from non-streaming response
        # (implementation unchanged)
        try:
            if isinstance(response, dict):
                # Check for explicit error key first
                if response.get("error"):
                    # Try to find a detailed message within choices if available
                    try:
                        # Check nested structure carefully
                        if (
                            "choices" in response
                            and isinstance(response["choices"], list)
                            and response["choices"]
                            and isinstance(response["choices"][0].get("message"), dict)
                        ):
                            return str(
                                response["choices"][0]["message"].get(
                                    "content", "Unknown LLM Error content"
                                )
                            )
                    except (IndexError, KeyError, TypeError):
                        # If structure is unexpected, fall back to generic error
                        pass
                    # Return the value of the 'error' key if it's a string, or a generic message
                    error_detail = response.get("error")
                    return (
                        f"LLM Error: {error_detail}"
                        if isinstance(error_detail, str)
                        else "Unknown LLM Error (structure)"
                    )

                # Check for standard successful OpenAI-like response structure
                elif (
                    "choices" in response
                    and isinstance(response["choices"], list)
                    and response["choices"]
                    and isinstance(response["choices"][0].get("message"), dict)
                ):
                    return str(
                        response["choices"][0]["message"].get("content", "")
                    )  # Return content or empty string
                else:
                    # Log if structure is neither known error nor known success
                    logger.warning(
                        f"Unexpected dictionary structure in get_response_content: {str(response)[:200]}"
                    )
                    return ""  # Return empty for unexpected dict structures

            # Log if response is not a dictionary at all
            logger.warning(
                f"Unexpected response type in get_response_content: {type(response)}"
            )
            return ""  # Return empty for non-dict types

        except Exception as e:
            logger.error(f"Response content extraction error: {str(e)}", exc_info=True)
            return ""  # Return empty on any extraction error

    async def cleanup(self):  # Optional cleanup hook (Indented within Pipe)
        debug = default_config.get("debug_logging", False)
        if hasattr(self, "valves"):
            debug = self.valves.DEBUG_LOGGING
        if debug:
            logger.info("Pipe cleanup initiated...")
        self.__current_event_emitter__ = None
        # Clear emitter reference

        # Close persistent aiohttp client if used (example)
        if (
            self.__llm_client__
            and hasattr(self.__llm_client__, "close")
            and callable(self.__llm_client__.close)
        ):
            try:
                if not self.__llm_client__.closed:
                    await self.__llm_client__.close()
                    logger.info("Closed persistent aiohttp client session.")
            except Exception as e:
                logger.error(f"Error closing client session: {e}")
        self.__llm_client__ = None

        # Optional: Trigger garbage collection
        try:
            gc.collect()
        except Exception as e:
            logger.error(f"GC error during cleanup: {e}")
        if debug:
            logger.info("Pipe cleanup complete.")


# ==============================================================================
# FILE END
# ==============================================================================
