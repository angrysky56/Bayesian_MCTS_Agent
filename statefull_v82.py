
# -*- coding: utf-8 -*-
"""
title: advanced_mcts_stateful (Single File)
version: 0.8.6 # <<< This will be dynamically handled by SCRIPT_VERSION constant now

author: angrysky56
author_url: https://github.com/angrysky56
Project Link: https://github.com/angrysky56/Bayesian_MCTS_Agent

Where I found my stored functions, replace ty with your user name:
/home/ty/.open-webui/cache/functions

The way I launch openweb-ui:
DATA_DIR=~/.open-webui uvx --python 3.11 open-webui@latest serve
http://localhost:8080

Function url:

description: >
  Stateful Advanced Bayesian MCTS v{SCRIPT_VERSION} (Single File Version):
  - Corrected pipe signature parameter names (__user__, __event_emitter__, __task__)
    to match expected OpenWebUI injection pattern, fixing missing emitter issue.
  - Implemented 'inlet' method for standard chat_id retrieval.
  - Removed truncation from synthesis context generation.
  - Removed truncation from verbose simulation summary thought output.
  - Corrected user object handling in internal Ollama calls.
  - Combines all logic into one script for Open WebUI pipe compatibility.
  - Integrates SQLite database for state persistence across user turns within a chat session.
  - Uses LLM-based intent classification to handle different user requests.
  - Implements Selective State Persistence: Saves/Loads learned approach priors, best results, and basic "unfit" markers.
  - Injects loaded state context into MCTS prompts to guide exploration.
  - Maintains core MCTS logic, quiet/verbose modes, and configuration via Valves.
  - Refactored internally for better readability while remaining a single file.
  - Enhanced error handling, logging, comments, and reduced code duplication.
  - **Added centralized SCRIPT_VERSION constant.**
  - **Improved logging in resolve_model and _resolve_question.**
  - **Improved robustness in pipes model listing.**
  - **Revised resolve_model function for better parsing and fallback.**


Requires:
- User to configure DB_FILE path correctly.
- Standard Python libs (sqlite3, json, datetime, re, logging, asyncio, math, random).
- Optional: scikit-learn for improved semantic distance calculation.
- Running within Open WebUI environment for pipe functionality.

################################################################################
# This was added and never needed before so bad code was introduced here-
# IMPORTANT: For Open WebUI integration to work correctly, this file MUST    #
#            be named exactly `advanced_mcts_stateful.py`                    #
#            in the `pipes` directory.                                       #
################################################################################
"""

# ==============================================================================
# Core Imports
# ==============================================================================
import asyncio
import gc
import json
import logging
import math
import os
import random
import re
import sqlite3
from collections import Counter
from datetime import datetime
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np  # Used for stats, random sampling
from numpy.random import beta as beta_sample
from pydantic import BaseModel, Field, field_validator, ValidationError

# from scipy import stats # Not currently used, commented out

# --- Optional Dependency: scikit-learn ---
try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
    # Add common MCTS/analysis terms to stop words for better semantic distance
    CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS) + [
        "analysis",
        "however",
        "therefore",
        "furthermore",
        "perspective",
        "node",
        "mcts",
        "score",
        "approach",
        "concept",
        "system",
        "model",
        "text",
        "data",
        "result",
        "based",
        "consider",
        "provide",
        "evaluate",
        "generate",
        "update",
    ]
    _temp_logger_skl = logging.getLogger(__name__)  # Use temp logger before setup
    _temp_logger_skl.info("scikit-learn found. TF-IDF semantic distance enabled.")
except ImportError:
    SKLEARN_AVAILABLE = False
    TfidfVectorizer, cosine_similarity, CUSTOM_STOP_WORDS = None, None, None
    _temp_logger_skl = logging.getLogger(__name__)
    _temp_logger_skl.warning(
        "scikit-learn not found. Using Jaccard similarity for semantic distance."
    )


# --- Open WebUI Specific Imports (Assumed available in the environment) ---
# Use try-except for robustness, although they are expected in the environment
try:
    from fastapi import Request, Response
    import open_webui.routers.ollama as ollama
    from open_webui.constants import TASKS  # For title generation task check
    from open_webui.main import app

    OPENWEBUI_IMPORTS_AVAILABLE = True
except ImportError as e:
    OPENWEBUI_IMPORTS_AVAILABLE = False
    _temp_logger_owui = logging.getLogger(__name__)
    _temp_logger_owui.error(
        f"Failed to import Open WebUI components: {e}. Pipe will likely fail."
    )
    # Define dummy classes/vars if needed for script to load without crashing immediately
    Request, Response, ollama, TASKS, app = None, None, None, None, None


# ==============================================================================
# Configuration Constants
# ==============================================================================
# This was added and not needed when working before.
# !! Filename Check !! Ensure this file is named advanced_mcts_stateful.py
PIPE_NAME = "advanced_mcts_stateful"  # Name for identification and logging
SCRIPT_VERSION = "0.8.6" # <<< CENTRALIZED VERSION CONSTANT >>>


# --- Database Configuration ---
# !!! IMPORTANT: Set this path to a writable location for the backend process !!!
# ----->>>>>>>> CHANGE THIS PATH <<<<<<<<<<-----
DB_FILE = "/home/ty/Repositories/sqlite-db/NEXUS_PRIME.db" # !!! User specific path !!!
# ----->>>>>>>> CHANGE THIS PATH <<<<<<<<<<-----


# --- Default MCTS Configuration ---
DEFAULT_CONFIG = {
    # Core MCTS
    "max_iterations": 1,
    "simulations_per_iteration": 10,
    "max_children": 10,
    "exploration_weight": 3.0,
    "use_thompson_sampling": True,
    "force_exploration_interval": 4,  # 0=off
    "score_diversity_bonus": 0.7,  # UCT diversity bonus based on sibling scores
    # Evaluation
    "use_bayesian_evaluation": True,
    "beta_prior_alpha": 1.0,
    "beta_prior_beta": 1.0,
    "relative_evaluation": False,  # Note: Relative eval not fully implemented
    "unfit_score_threshold": 4.0,  # Score below which nodes might be marked unfit (if stateful)
    "unfit_visit_threshold": 3,  # Min visits before marking unfit by score (if stateful)
    # Surprise Mechanism
    "use_semantic_distance": True,  # Use TF-IDF (or future embeddings) for surprise
    "surprise_threshold": 0.66,  # Semantic distance threshold for surprise
    "surprise_semantic_weight": 0.6,
    "surprise_philosophical_shift_weight": 0.3,  # Weight for change in approach family
    "surprise_novelty_weight": 0.3,  # Weight for how rare the approach family is
    "surprise_overall_threshold": 0.9,  # Combined weighted threshold to trigger surprise
    # Context & Prompts
    "global_context_in_prompts": True,  # Include global context (best score, etc.)
    "track_explored_approaches": True,  # Include summary of explored thoughts/approaches
    "sibling_awareness": True,  # Include sibling context in prompts
    # Performance & State
    "memory_cutoff": 5,  # Max number of high-scoring nodes to remember
    "early_stopping": True,
    "early_stopping_threshold": 10.0,  # Score threshold (1-10)
    "early_stopping_stability": 2,  # Iterations score must stay >= threshold
    "enable_state_persistence": True,  # Master switch for using SQLite DB
    # Output & Logging
    "show_processing_details": True,  # Show verbose MCTS steps in chat output
    "debug_logging": False,  # Enable detailed DEBUG level logs (console/log file)
}

# --- Approach Taxonomy & Metadata ---
APPROACH_TAXONOMY = {
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
APPROACH_METADATA = {
    "empirical": {"family": "epistemology"}, "rational": {"family": "epistemology"},
    "phenomenological": {"family": "epistemology"}, "hermeneutic": {"family": "epistemology"},
    "reductionist": {"family": "ontology"}, "holistic": {"family": "ontology"},
    "materialist": {"family": "ontology"}, "idealist": {"family": "ontology"},
    "analytical": {"family": "methodology"}, "synthetic": {"family": "methodology"},
    "dialectical": {"family": "methodology"}, "comparative": {"family": "methodology"},
    "critical": {"family": "perspective"}, "constructive": {"family": "perspective"},
    "pragmatic": {"family": "perspective"}, "normative": {"family": "perspective"},
    "structural": {"family": "general"}, "alternative": {"family": "general"},
    "complementary": {"family": "general"}, "variant": {"family": "general"},
    "initial": {"family": "general"},
}

# --- Logging Configuration ---
LOG_LEVEL = logging.DEBUG if DEFAULT_CONFIG["debug_logging"] else logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s [%(levelname)s] %(message)s"
PIPE_LOG_NAME = f"pipe.{PIPE_NAME}"
MCTS_LOG_NAME = f"mcts.{PIPE_NAME}"

# ==============================================================================
# Prompts
# ==============================================================================
INITIAL_ANALYSIS_PROMPT = """ Utilize INTENT_CLASSIFIER_PROMPT when appropriate.
<instruction>Provide an initial analysis and interpretation of the themes, arguments, and potential implications presented by the user suitable for the further MCTS analysis.</instruction>
<question>{question}</question>
"""
GENERATE_THOUGHT_PROMPT = """
<instruction>Critically examine the current analysis below using the following approaches:
1.  **Challenge Core Assumption:** Identify a major underlying assumption or weakness in the current analysis ({current_answer}) and propose a significant course correction.
2.  **Novel Connection/Analogy:** Propose a novel connection to another domain, concept, or analogy that reframes the analysis in a surprising way.
3.  **Perspective Shift:** Adopt the perspective of an expert (e.g., Philosopher, Cognitive Scientist, Sociologist) and identify the most critical missing angle or challenge.
4.  **Refinement Focus ('Coherence'):** Suggest a way to improve the analysis's internal `Coherence` – its logical integrity, alignment with the original text, or the flow between ideas.
5.  **Exploration Focus ('Curiosity'):** If the analysis seems stuck or lacks depth, suggest a fundamentally different direction or question to explore.

**Previous Run Context (If Available):**
- Previous Best Analysis Summary: {previous_best_summary}
- Previously Marked Unfit Concepts/Areas: {unfit_markers_summary}
- Learned Approach Preferences: {learned_approach_summary}

**Current Search Context:**
- Original Text Summary: {question_summary}
- Best Overall Analysis (Score {best_score}/10): {best_answer}
- Current Analysis (Node {current_sequence}): {current_answer}
- Current Analysis Tags: {current_tags}
- Explored Approaches Summary: {explored_approaches}
- Sibling Node Thoughts: {sibling_approaches}

**Constraints:**
- Ensure your critique/suggestion is *directly relevant* to the original text and current analysis.
- Do *not* simply rephrase existing ideas. Push the thinking significantly forward.
- Do *not* explore areas previously marked as unfit: {unfit_markers_summary} unless your suggestion offers a truly novel angle on them.
- Respond with the critique/suggestion itself.</instruction>
"""
UPDATE_ANALYSIS_PROMPT = """
<instruction>Substantially revise the draft analysis below to incorporate the core idea from the critique. Develop the analysis further based on this new direction. Output the revised analysis text itself.

**Previous Run Context (If Available):**
- Previous Best Analysis Summary: {previous_best_summary}
- Previously Marked Unfit Concepts/Areas: {unfit_markers_summary}

Ensure the revision considers past findings and avoids known unproductive paths unless the critique justifies revisiting them. Ensure the revision is well-grounded in the original text and does not introduce unsupported information.

**Current Search Context:**
- Original Text Summary: {question_summary}
- Best Overall Analysis (Score {best_score}/10): {best_answer}
- Current Analysis Tags: {current_tags}

**Inputs for Revision:**
<draft>{answer}</draft>
<critique>{improvements}</critique>

Write the new, revised analysis text ONLY.</instruction>
"""
EVALUATE_ANALYSIS_PROMPT = """
<instruction>Evaluate the intellectual quality of the analysis below (1-10) regarding the original text. Consider multiple factors:

1.  **Insight & Novelty (+/- 1-5):** How deep, insightful, relevant, and novel is this analysis compared to the best so far ({best_score}/10)? Does it significantly advance understanding? (Reserve 9-10 for truly exceptional, transformative insights).
2.  **Grounding & Reliability (Yes/No - implicit in score):** Is the analysis well-grounded in the original text ({question_summary})? Does it avoid unsupported claims or potential hallucinations?
3.  **Coherence & Structure (+/- 1-5):** How well-structured and internally consistent is the analysis? Does it exhibit strong `Coherence` (integrity, alignment, flow)? Or does it feel disjointed (`Entropy`)?
4.  **Perspective (Considered - implicit in score):** Does the analysis demonstrate a comprehensive perspective, potentially integrating different angles, or does it feel narrow?

**Previous Run Context (If Available):**
- Previous Best Analysis Summary: {previous_best_summary}
- Previously Marked Unfit Concepts/Areas: {unfit_markers_summary}

**Current Search Context:**
- Original Text Summary: {question_summary}
- Best Overall Analysis (Score {best_score}/10): {best_answer}
- Analysis Tags: {current_tags}

**Analysis to Evaluate:**
<answer_to_evaluate>{answer_to_evaluate}</answer_to_evaluate>

Based primarily on **Insight & Novelty**, but informed by Grounding, Coherence, and Perspective, provide a single overall score from 1 to 10. Respond ONLY with the number.</instruction>
"""
GENERATE_TAGS_PROMPT = """
<instruction>Generate concise keyword tags summarizing the main concepts in the following text. Output ONLY the tags, separated by commas. DO NOT add any other text.</instruction>
<text_to_tag>{analysis_text}</text_to_tag>
"""
FINAL_SYNTHESIS_PROMPT = """
<instruction>Synthesize the key insights developed along the primary path of analysis below into a concise, conclusive statement addressing the original question/text. Focus on the progression of ideas represented by the sequence of best scoring nodes and 'thoughts'. Respond with clear, natural language text ONLY. DO NOT use JSON, markdown, lists, or ask questions.</instruction>
<original_question_summary>{question_summary}</original_question_summary>
<initial_analysis_summary>{initial_analysis_summary}</initial_analysis_summary>
<best_analysis_score>{best_score}/10</best_analysis_score>
<development_path>
{path_thoughts}
</development_path>
<final_best_analysis_summary>{final_best_analysis_summary}</final_best_analysis_summary>
Synthesize the journey of thoughts into a final conclusion:</instruction>
"""
INTENT_CLASSIFIER_PROMPT = """
Determine the primary purpose of the user's input. Choose the *single best* category from the list. Respond with the category name (e.g., ANALYZE_NEW).

Categories:
- ANALYZE_NEW: User wants a fresh MCTS analysis on the provided text/task, ignoring any previous runs in this chat.
- CONTINUE_ANALYSIS: User wants to continue, refine, or build upon the previous MCTS analysis run in this chat (e.g., "elaborate", "explore X further", "what about Y?").
- ASK_LAST_RUN_SUMMARY: User is asking specifically about the results (score, tags, summary) of the analysis that was just completed.
- ASK_PROCESS: User is asking how the tool works, about its algorithms, or its general capabilities.
- ASK_CONFIG: User is asking about the current operational settings or configuration parameters.
- GENERAL_CONVERSATION: The input is conversational.

User Input:
"{raw_input_text}"

Classification:
"""
ASK_PROCESS_EXPLANATION = """ Answer adaptively to the user's question about how the tool works.
I use an Advanced Bayesian Monte Carlo Tree Search (MCTS) algorithm to analyze text or questions. Here's a breakdown:
- **Exploration vs. Exploitation:** I balance trying new interpretations (exploration) with refining promising ones (exploitation) using techniques like UCT or Thompson Sampling.
- **Bayesian Evaluation (Optional):** I can use Beta distributions to model the uncertainty in analysis scores, leading to potentially more robust exploration.
- **Node Expansion:** I generate new 'thoughts' (critiques, alternative angles, connections) using LLM calls to branch out the analysis tree. These thoughts are generated based on advanced prompts encouraging diverse perspectives, critical thinking, and coherence.
- **Simulation (Evaluation):** I assess the quality of each analysis node using LLM calls, judging insight, novelty, relevance, coherence, and grounding compared to the best analysis found so far.
- **Backpropagation:** Scores (or Bayesian parameters) and visit counts are updated back up the tree path after each evaluation.
- **State Persistence (Optional):** Within a single chat session, I can save key results (best analysis, score, tags) and learned preferences for different analytical approaches to a local database (`{db_file_name}`). This requires a 'chat_id' provided by the framework. If no 'chat_id' is received, this feature is automatically disabled for safety.
- **Intent Handling:** I try to figure out if you want a completely new analysis, want to build on the last one, or are asking about results, my process, or settings.

You can fine-tune parameters like exploration intensity, iterations, evaluation methods, and more using the 'Valves' settings in the UI.
"""
GENERAL_CONVERSATION_PROMPT = """
The user has provided input classified as general conversation. Respond in a friendly and engaging manner, maintaining an appropriate tone for the conversation.

User Input: "{user_input}"

Your Response:
"""

# ==============================================================================
# Utility Functions
# ==============================================================================
# --- Logger Setup ---
loggers: Dict[str, logging.Logger] = {}

def setup_logger(name: str, level: int, log_format: str) -> logging.Logger:
    global loggers
    if name in loggers:
        if loggers[name].level != level: loggers[name].setLevel(level);
        for handler in loggers[name].handlers: handler.setLevel(level)
        return loggers[name]
    logger_instance = logging.getLogger(name); logger_instance.setLevel(level)
    handler_name = f"{name}_handler"
    if not any(h.get_name() == handler_name for h in logger_instance.handlers):
        handler = logging.StreamHandler(); handler.set_name(handler_name)
        formatter = logging.Formatter(log_format); handler.setFormatter(formatter)
        handler.setLevel(level); logger_instance.addHandler(handler)
        logger_instance.propagate = False
    loggers[name] = logger_instance
    return logger_instance

logger = setup_logger(PIPE_LOG_NAME, LOG_LEVEL, LOG_FORMAT)
mcts_logger = setup_logger(MCTS_LOG_NAME, LOG_LEVEL, LOG_FORMAT)

# --- Text Processing ---
def truncate_text(text: Optional[str], max_length: int = 8196) -> str:
    if not text: return ""
    text = str(text).strip(); text = re.sub(r"^\s*```[\s\S]*?```\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*```(json|markdown|text)?\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE).strip()
    if len(text) <= max_length: return text
    last_space = text.rfind(" ", 0, max_length)
    return text[:last_space] + "..." if last_space != -1 else text[:max_length] + "..."

def calculate_semantic_distance(text1: Optional[str], text2: Optional[str], logger_instance: logging.Logger, use_tfidf: bool = SKLEARN_AVAILABLE) -> float:
    if not text1 or not text2: return 1.0
    text1, text2 = str(text1), str(text2)
    if use_tfidf and SKLEARN_AVAILABLE and TfidfVectorizer and cosine_similarity:
        try:
            vectorizer = TfidfVectorizer(stop_words=CUSTOM_STOP_WORDS, max_df=0.9, min_df=1)
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            if tfidf_matrix.shape[0] < 2 or tfidf_matrix.shape[1] == 0: raise ValueError("TF-IDF matrix issue.")
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return 1.0 - max(0.0, min(1.0, similarity))
        except ValueError as ve: logger_instance.warning(f"TF-IDF ValueError: {ve}. Fallback Jaccard.")
        except Exception as e: logger_instance.warning(f"TF-IDF error: {e}. Fallback Jaccard.", exc_info=True)
    try:
        words1 = set(re.findall(r"\b\w+\b", text1.lower())); words2 = set(re.findall(r"\b\w+\b", text2.lower()))
        if not words1 and not words2: return 0.0;
        if not words1 or not words2: return 1.0
        intersection = len(words1.intersection(words2)); union = len(words1.union(words2))
        if union == 0: return 0.0
        return 1.0 - (intersection / union)
    except Exception as fallback_e: logger_instance.error(f"Jaccard fallback failed: {fallback_e}", exc_info=True); return 1.0

# --- Approach Classification ---
def classify_approach(thought: Optional[str], taxonomy: Dict[str, List[str]], metadata: Dict[str, Dict[str, str]], random_state: Any, logger_instance: logging.Logger) -> Tuple[str, str]:
    approach_type, approach_family = "variant", "general"
    if not thought or not isinstance(thought, str): logger_instance.debug("Cannot classify empty thought."); return approach_type, approach_family
    thought_lower = thought.lower(); scores = {app: sum(1 for kw in kws if kw in thought_lower) for app, kws in taxonomy.items()}
    positive_scores = {app: score for app, score in scores.items() if score > 0}
    if positive_scores: max_score = max(positive_scores.values()); best_types = [app for app, score in positive_scores.items() if score == max_score]; approach_type = random_state.choice(best_types)
    approach_family = metadata.get(approach_type, {}).get("family", "general")
    logger_instance.debug(f"Classified '{truncate_text(thought, 50)}' as: {approach_type} ({approach_family})")
    return approach_type, approach_family

# --- Mock Admin User ---
class AdminUserMock:
    def __init__(self, role: str = "admin"): self.role = role
ADMIN_USER = AdminUserMock()

# --- Database Utilities ---
def get_db_connection(db_file_path: str, logger_instance: logging.Logger) -> Optional[sqlite3.Connection]:
    conn = None
    try:
        db_dir = os.path.dirname(db_file_path);
        if db_dir: os.makedirs(db_dir, exist_ok=True)
        conn = sqlite3.connect(db_file_path, timeout=10); conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""CREATE TABLE IF NOT EXISTS mcts_state ( chat_id TEXT PRIMARY KEY, last_state_json TEXT NOT NULL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mcts_state_timestamp ON mcts_state (timestamp);"); conn.commit()
        logger_instance.debug(f"DB connected: {db_file_path}"); return conn
    except sqlite3.Error as e: logger_instance.error(f"SQLite connection/setup error {db_file_path}: {e}", exc_info=True)
    except OSError as e: logger_instance.error(f"OS error DB access '{db_file_path}': {e}", exc_info=True)
    except Exception as e: logger_instance.error(f"Unexpected DB connection error {db_file_path}: {e}", exc_info=True)
    if conn: conn.close(); return None

def save_mcts_state(db_file_path: str, chat_id: str, state: Dict[str, Any], logger_instance: logging.Logger):
    if not chat_id: logger_instance.warning("Cannot save state: chat_id missing."); return
    if not isinstance(state, dict) or not state: logger_instance.warning(f"Attempted save invalid state chat_id {chat_id}."); return
    conn = None
    try:
        state_json = json.dumps(state); conn = get_db_connection(db_file_path, logger_instance)
        if not conn: logger_instance.error("Cannot save state: DB connection failed."); return
        with conn: conn.execute("INSERT OR REPLACE INTO mcts_state (chat_id, last_state_json, timestamp) VALUES (?, ?, ?)", (chat_id, state_json, datetime.now()))
        logger_instance.info(f"Saved MCTS state for chat_id: {chat_id}")
    except json.JSONDecodeError as json_err: logger_instance.error(f"Failed serialize state JSON chat_id '{chat_id}': {json_err}", exc_info=True)
    except sqlite3.Error as e: logger_instance.error(f"SQLite save state error chat_id {chat_id}: {e}", exc_info=True)
    except Exception as e: logger_instance.error(f"Unexpected save state error chat_id '{chat_id}': {e}", exc_info=True)
    finally:
        if conn: conn.close()

def load_mcts_state(db_file_path: str, chat_id: str, logger_instance: logging.Logger) -> Optional[Dict[str, Any]]:
    if not chat_id: logger_instance.warning("Cannot load state: chat_id missing."); return None
    conn = None; state_dict = None
    try:
        conn = get_db_connection(db_file_path, logger_instance)
        if not conn: logger_instance.error("Cannot load state: DB connection failed."); return None
        with conn:
            cursor = conn.cursor(); cursor.execute("SELECT last_state_json FROM mcts_state WHERE chat_id = ? ORDER BY timestamp DESC LIMIT 1", (chat_id,))
            result = cursor.fetchone()
            if result and result[0]:
                try:
                    loaded_data = json.loads(result[0])
                    if isinstance(loaded_data, dict): state_dict = loaded_data; logger_instance.info(f"Loaded MCTS state for chat_id: {chat_id}")
                    else: logger_instance.warning(f"Loaded state chat_id {chat_id} not dict (type: {type(loaded_data)}). Discarding."); state_dict = None
                except json.JSONDecodeError as json_err: logger_instance.error(f"Error decoding state JSON chat_id {chat_id}: {json_err}", exc_info=True); state_dict = None
            else: logger_instance.info(f"No previous MCTS state found chat_id: {chat_id}")
    except sqlite3.Error as e: logger_instance.error(f"SQLite load state error chat_id {chat_id}: {e}", exc_info=True); state_dict = None
    except Exception as e: logger_instance.error(f"Unexpected load state error chat_id {chat_id}: {e}", exc_info=True); state_dict = None
    finally:
        if conn: conn.close()
    return state_dict

# --- LLM Interaction Utilities ---
async def call_ollama_endpoint(payload: Dict[str, Any], logger_instance: logging.Logger, user_object: Optional[Union[Dict, AdminUserMock]], debug_logging: bool = False) -> Union[Dict, Any]:
    """Calls the internal OpenWebUI Ollama endpoint, ensuring user object compatibility."""
    if not OPENWEBUI_IMPORTS_AVAILABLE or not ollama or not app:
        logger_instance.critical("Cannot call Ollama: Open WebUI components missing."); return {"error": True, "choices": [{"message": {"role": "assistant", "content": "Internal Error: Missing OpenWebUI components."}}]}

    try:
        async def receive(): return {"type": "http.request", "body": json.dumps(payload).encode("utf-8")}
        mock_request = Request(scope={"type": "http", "headers": [], "method": "POST", "scheme": "http", "server": ("localhost", 8080), "path": "/api/ollama/generate", "query_string": b"", "client": ("127.0.0.1", 8080), "app": app}, receive=receive)

        if debug_logging:
            log_payload = {k: (f"{str(v)[:100]}..." if len(str(v)) > 100 else v) for k, v in payload.items() if k != "messages"}
            log_payload["messages_count"] = len(payload.get("messages", []))
            logger_instance.debug(f"Calling internal ollama. Payload summary: {log_payload}")

        # --- User Object Handling (FIXED) ---
        final_user_object = AdminUserMock() # Default to mock admin
        if isinstance(user_object, dict):
            user_role = user_object.get('role', 'admin') # Default to 'admin' if role key is missing
            final_user_object.role = user_role
            logger_instance.debug(f"Using framework-provided user dict, role set to: {user_role}")
        elif hasattr(user_object, 'role'):
             final_user_object = user_object # Assume it's already a compatible object (like AdminUserMock)
             logger_instance.debug(f"Using provided user object with role: {final_user_object.role}")
        else:
             # This case should ideally not happen if __user__ is always set in pipe()
             logger_instance.warning("User object was None or unexpected type, using default AdminUserMock.")
        # --- End User Object Handling ---

        response = await ollama.generate_openai_chat_completion(
            request=mock_request, form_data=payload, user=final_user_object # Pass the compatible object
        )

        if debug_logging and not isinstance(response, dict) and not hasattr(response, "body_iterator"): logger_instance.debug(f"Internal Ollama response type: {type(response)}")
        return response
    except AttributeError as ae: logger_instance.error(f"Ollama internal call AttributeError: {ae}. Compatible?", exc_info=debug_logging); return {"error": True, "choices": [{"message": {"role": "assistant", "content": f"Error: LLM comms failed (AttributeError: {ae})."}}]}
    except Exception as e:
        err_str = str(e)
        if "400" in err_str and "Model" in err_str and "not found" in err_str: logger_instance.error(f"Ollama internal call reported: {err_str}", exc_info=debug_logging); return {"error": True, "choices": [{"message": {"role": "assistant", "content": f"Error: LLM comms failed ({err_str}). Check logs."}}]}
        logger_instance.error(f"Ollama internal call unexpected error: {err_str}", exc_info=debug_logging); return {"error": True, "choices": [{"message": {"role": "assistant", "content": f"Error: LLM comms failed ({str(e)[:100]}...). Check logs."}}]}

def get_chunk_content(chunk_bytes: bytes, logger_instance: logging.Logger, debug_logging: bool = False) -> List[str]:
    parts = []
    try:
        chunk_str = chunk_bytes.decode("utf-8")
        for line in chunk_str.splitlines():
            line = line.strip();
            if not line or line == "data: [DONE]": continue
            if line.startswith("data: "):
                json_str = line[len("data: ") :];
                try:
                    data = json.loads(json_str); content = data.get("choices", [{}])[0].get("delta", {}).get("content")
                    if isinstance(content, str) and content: parts.append(content)
                except (json.JSONDecodeError, IndexError, KeyError, TypeError) as e:
                    if debug_logging: logger_instance.warning(f"Stream chunk JSON parse error: {e} - Data: {json_str[:100]}...")
                except Exception as e: logger_instance.error(f"Unexpected error parsing stream chunk JSON: {e}", exc_info=debug_logging)
            elif debug_logging: logger_instance.debug(f"Ignoring non-data line in stream chunk: {line[:100]}...")
    except UnicodeDecodeError: logger_instance.error(f"Unicode decode error in stream chunk: {chunk_bytes[:100]}...")
    except Exception as e: logger_instance.error(f"Error processing stream chunk bytes: {e}", exc_info=debug_logging)
    return parts

def get_response_content(response: Union[Dict, Any], logger_instance: logging.Logger) -> str:
    try:
        if not isinstance(response, dict): logger_instance.warning(f"Cannot extract content: Response not dict (type: {type(response)})."); return ""
        if response.get("error"):
            try: error_content = str(response.get("choices", [{}])[0].get("message", {}).get("content", "Unknown LLM Error"))
            except (IndexError, KeyError, TypeError): err_detail = response.get("error"); error_content = (f"{err_detail}" if isinstance(err_detail, (str, bool, int, float)) else "Unknown LLM Error (Invalid Format)")
            return error_content if "Error:" in error_content else f"Error: {error_content}"
        elif "choices" in response and isinstance(response["choices"], list) and response["choices"]:
            try: message = response["choices"][0].get("message", {}); content = message.get("content", ""); return str(content)
            except (IndexError, KeyError, TypeError): logger_instance.warning(f"Unexpected structure within 'choices': {str(response)[:200]}"); return ""
            except Exception as e: logger_instance.error(f"Unexpected error extracting content: {e}", exc_info=True); return ""
        else: logger_instance.warning(f"Unexpected dict structure (no choices/error): {str(response)[:200]}"); return ""
    except Exception as e: logger_instance.error(f"General response content extraction error: {str(e)}", exc_info=True); return ""

# ==============================================================================
# Node Class Definition
# ==============================================================================
class Node(BaseModel):
    id: str = Field(default_factory=lambda: f"node_{random.randbytes(4).hex()}")
    content: str = ""
    parent: Optional["Node"] = Field(default=None, exclude=True)
    children: List["Node"] = Field(default_factory=list)
    visits: int = 0
    raw_scores: List[Union[int, float]] = Field(default_factory=list)
    sequence: int = 0
    is_surprising: bool = False
    surprise_explanation: str = ""
    approach_type: str = "initial"
    approach_family: str = "general"
    thought: str = ""
    max_children: int = DEFAULT_CONFIG["max_children"]
    use_bayesian_evaluation: bool = DEFAULT_CONFIG["use_bayesian_evaluation"]
    alpha: Optional[float] = None
    beta: Optional[float] = None
    value: Optional[float] = None
    descriptive_tags: List[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("parent", "children", mode="before")
    @classmethod
    def _validate_optional_fields(cls, v): return v

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.max_children = data.get("max_children", DEFAULT_CONFIG["max_children"])
        self.use_bayesian_evaluation = data.get("use_bayesian_evaluation", DEFAULT_CONFIG["use_bayesian_evaluation"])
        if self.use_bayesian_evaluation:
            prior_alpha = data.get("alpha", DEFAULT_CONFIG["beta_prior_alpha"])
            prior_beta = data.get("beta", DEFAULT_CONFIG["beta_prior_beta"])
            self.alpha = max(1e-9, float(prior_alpha)); self.beta = max(1e-9, float(prior_beta)); self.value = None
        else:
            self.value = float(data.get("value", 0.0)); self.alpha = None; self.beta = None

    def add_child(self, child: "Node"):
        if child not in self.children: self.children.append(child); child.parent = self

    def fully_expanded(self) -> bool:
        valid_children_count = sum(1 for c in self.children if isinstance(c, Node)); return valid_children_count >= self.max_children

    def get_bayesian_mean(self) -> float:
        if self.use_bayesian_evaluation and self.alpha is not None and self.beta is not None:
            alpha_safe, beta_safe = max(1e-9, self.alpha), max(1e-9, self.beta); denominator = alpha_safe + beta_safe
            return (alpha_safe / denominator) if denominator > 1e-18 else 0.5
        return 0.5

    def get_average_score(self) -> float:
        if self.use_bayesian_evaluation: return self.get_bayesian_mean() * 10.0
        else: return self.value / self.visits if self.visits > 0 and self.value is not None else 5.0

    def thompson_sample(self) -> float:
        if self.use_bayesian_evaluation and self.alpha is not None and self.beta is not None:
            alpha_safe, beta_safe = max(1e-9, self.alpha), max(1e-9, self.beta)
            try: sample = float(beta_sample(alpha_safe, beta_safe)); return max(0.0, min(1.0, sample))
            except Exception as e: logger = logging.getLogger(MCTS_LOG_NAME) or logging.getLogger(); logger.warning(f"TS failed node {self.sequence} (α={alpha_safe:.3f}, β={beta_safe:.3f}): {e}. Using mean."); return self.get_bayesian_mean()
        return self.get_bayesian_mean()

    def best_child(self) -> Optional["Node"]:
        valid_children = [c for c in self.children if isinstance(c, Node)];
        if not valid_children: return None
        try: max_visits = max(child.visits for child in valid_children if isinstance(child.visits, int))
        except ValueError: return None # Happens if visits are somehow not ints or list empty
        most_visited_children = [child for child in valid_children if child.visits == max_visits]
        if len(most_visited_children) == 1: return most_visited_children[0]
        elif len(most_visited_children) > 1:
            # Tie-breaking: prioritize higher score (Bayesian mean or simple average)
            score_key_func = lambda c: c.get_bayesian_mean() if self.use_bayesian_evaluation else c.get_average_score()
            try: return max(most_visited_children, key=score_key_func)
            except Exception as e: logger = logging.getLogger(MCTS_LOG_NAME) or logging.getLogger(); logger.warning(f"Error best_child tie-breaking {len(most_visited_children)} nodes: {e}. Returning first."); return most_visited_children[0]
        # If max_visits was 0 or some other issue, return one of the valid children randomly? Should not happen if max_visits > 0.
        elif valid_children: logger = logging.getLogger(MCTS_LOG_NAME) or logging.getLogger(); logger.warning(f"best_child fallback: Random choice {len(valid_children)} valid children."); return random.choice(valid_children)
        else: return None # No valid children

    def node_to_json(self, max_depth: int = 3, current_depth: int = 0) -> Dict[str, Any]:
        score = self.get_average_score(); valid_children = [c for c in self.children if isinstance(c, Node)]
        node_dict: Dict[str, Any] = {"id": self.id, "sequence": self.sequence, "content_summary": truncate_text(self.content, 150), "visits": self.visits, "approach_type": self.approach_type, "approach_family": self.approach_family, "is_surprising": self.is_surprising, "thought_summary": truncate_text(self.thought, 100), "tags": self.descriptive_tags[:], "score": round(score, 2) if score is not None else None, "children_count": len(valid_children), "children": []}
        if self.use_bayesian_evaluation and self.alpha is not None and self.beta is not None: node_dict.update({"alpha": round(self.alpha, 3), "beta": round(self.beta, 3), "mean_score_0_1": round(self.get_bayesian_mean(), 3)})
        elif not self.use_bayesian_evaluation and self.value is not None: node_dict["cumulative_value"] = round(self.value, 2)
        if current_depth < max_depth: node_dict["children"] = [child.node_to_json(max_depth, current_depth + 1) for child in valid_children]
        return node_dict

    def node_to_state_dict(self) -> Dict[str, Any]:
        score = self.get_average_score(); state_dict: Dict[str, Any] = {"id": self.id, "sequence": self.sequence, "content_summary": truncate_text(self.content, 250), "visits": self.visits, "approach_type": self.approach_type, "approach_family": self.approach_family, "thought": self.thought, "tags": self.descriptive_tags[:], "score": round(score, 2) if score is not None else None, "is_surprising": self.is_surprising}
        if self.use_bayesian_evaluation and self.alpha is not None and self.beta is not None: state_dict.update({"alpha": round(self.alpha, 4), "beta": round(self.beta, 4)})
        elif not self.use_bayesian_evaluation and self.value is not None: state_dict["value"] = round(self.value, 2)
        return state_dict

# ==============================================================================
# LLM Interface Definition
# ==============================================================================
class LLMInterface:
    # Define methods that the MCTS class will expect the Pipe class to implement
    async def generate_thought(self, current_analysis: str, context: Dict, config: Dict) -> str: ...
    async def update_approach(self, original_analysis: str, critique: str, context: Dict, config: Dict) -> str: ...
    async def evaluate_answer(self, analysis_to_evaluate: str, context: Dict, config: Dict) -> Union[int, str]: ...
    async def get_completion(self, model: str, messages: List[Dict[str, str]]) -> str: ...
    async def progress(self, message: str): ...
    async def emit_message(self, message: str): ...
    def resolve_model(self, body: dict) -> str: ...


# ==============================================================================
# MCTS Class Definition
# ==============================================================================
class MCTS:
    def __init__(self, llm_interface: LLMInterface, question: str, mcts_config: Dict[str, Any], initial_analysis_content: str, initial_state: Optional[Dict[str, Any]] = None, model_body: Optional[Dict[str, Any]] = None):
        self.llm = llm_interface; self.config = mcts_config; self.question = question; self.question_summary = self._summarize_question(question)
        self.model_body = model_body or {}; self.debug_logging = self.config.get("debug_logging", False); self.show_chat_details = self.config.get("show_processing_details", True) # Default to True
        self.logger = logging.getLogger(MCTS_LOG_NAME); self.logger.setLevel(logging.DEBUG if self.debug_logging else logging.INFO)
        self.loaded_initial_state = initial_state; self.node_sequence = 0; self.iterations_completed = 0; self.simulations_completed = 0
        self.high_score_counter = 0; self.random_state = random.Random(); self.thought_history: List[str] = []; self.debug_history: List[str] = []
        self.surprising_nodes: List[Node] = []; self.explored_approaches: Dict[str, List[str]] = {}; self.explored_thoughts: Set[str] = set()
        self.memory: Dict[str, Any] = {"depth": 0, "branches": 0, "high_scoring_nodes": []}
        # Log MCTS Initialization using SCRIPT_VERSION
        mcts_start_log = f"# MCTS Init v{SCRIPT_VERSION}\nQ: {self.question_summary}\nState Loaded: {bool(initial_state)}\n"
        if initial_state:
            l_score = initial_state.get("best_score", "N/A"); l_tags = initial_state.get("best_node_tags", []); l_unfit_count = len(initial_state.get("unfit_markers", []))
            mcts_start_log += f" LScore: {l_score}, LTags: {l_tags}, LUnfit: {l_unfit_count}\n"
        self.thought_history.append(mcts_start_log); self.logger.info(f"MCTS Initializing. Question: '{self.question_summary}'. State Loaded: {bool(initial_state)}")
        # Initialize Approach Priors/Scores
        cfg = self.config; prior_alpha = max(1e-9, float(cfg.get("beta_prior_alpha", 1.0))); prior_beta = max(1e-9, float(cfg.get("beta_prior_beta", 1.0)))
        self.approach_alphas: Dict[str, float] = {}; self.approach_betas: Dict[str, float] = {}; self.approach_scores: Dict[str, float] = {}
        loaded_priors = initial_state.get("approach_priors") if initial_state and isinstance(initial_state, dict) else None
        if cfg.get("use_bayesian_evaluation") and loaded_priors and isinstance(loaded_priors.get("alpha"), dict) and isinstance(loaded_priors.get("beta"), dict):
            self.approach_alphas = {k: max(1e-9, float(v)) for k, v in loaded_priors["alpha"].items() if isinstance(v, (int, float))}
            self.approach_betas = {k: max(1e-9, float(v)) for k, v in loaded_priors["beta"].items() if isinstance(v, (int, float))}
            all_keys = set(APPROACH_TAXONOMY.keys()) | {"initial", "variant"}
            for k in all_keys: self.approach_alphas.setdefault(k, prior_alpha); self.approach_betas.setdefault(k, prior_beta)
            self.logger.info(f"Loaded and validated {len(self.approach_alphas)} Bayesian approach priors from state.")
        else:
            # Initialize defaults if not loading or not Bayesian
            all_keys = list(APPROACH_TAXONOMY.keys()) + ["initial", "variant"]
            for k in all_keys: self.approach_alphas[k] = prior_alpha; self.approach_betas[k] = prior_beta; self.approach_scores[k] = 5.0 # Default score
            if cfg.get("use_bayesian_evaluation"): self.logger.info("Initialized default Bayesian priors (no valid priors in loaded state).")
            else: self.logger.info("Initialized default approach scores (Bayesian mode off).")
        # Initialize Best Score/Solution Tracking
        self.best_score: float = 0.0; self.best_solution: str = initial_analysis_content; self.previous_best_solution_content: Optional[str] = None
        if initial_state and isinstance(initial_state, dict):
            try:
                loaded_score = initial_state.get("best_score"); self.best_score = float(loaded_score) if isinstance(loaded_score, (int, float)) else 0.0
                loaded_content = initial_state.get("best_solution_content"); self.previous_best_solution_content = loaded_content if isinstance(loaded_content, str) and loaded_content else None
                self.logger.info(f"Initialized best score tracker from state (Score: {self.best_score:.2f}). Previous best content stored.")
            except Exception as e: self.logger.error(f"Error processing loaded state for best score/content: {e}", exc_info=self.debug_logging); self.best_score = 0.0; self.previous_best_solution_content = None
        else: self.logger.info("Initialized best score tracker with defaults (Score: 0.0).")
        # Initialize Root Node
        try:
            self.root: Optional[Node] = Node(content=initial_analysis_content, sequence=self.get_next_sequence(), parent=None, max_children=cfg["max_children"], use_bayesian_evaluation=cfg["use_bayesian_evaluation"], alpha=prior_alpha, beta=prior_beta, value=0.0, approach_type="initial", approach_family="general")
            if not self.root: raise RuntimeError("Root node creation returned None unexpectedly")
        except Exception as e: self.logger.critical(f"FATAL: Root node initialization failed: {e}", exc_info=True); self.root = None; raise RuntimeError(f"MCTS Root Node creation failed: {e}") from e
        # Initialize Unfit Markers
        self.unfit_markers: List[Dict[str, Any]] = []
        if initial_state and isinstance(initial_state, dict):
            markers = initial_state.get("unfit_markers")
            if isinstance(markers, list): self.unfit_markers = markers; self.logger.info(f"Loaded {len(self.unfit_markers)} unfit markers from state.")
            elif markers is not None: self.logger.warning(f"Loaded 'unfit_markers' from state is not a list (type: {type(markers)}). Ignoring.")

    def get_next_sequence(self) -> int: self.node_sequence += 1; return self.node_sequence
    def _summarize_question(self, text: str, max_words: int = 50) -> str:
        if not text or not isinstance(text, str): return "N/A"
        try: words = re.findall(r"\b\w+\b", text); summary = " ".join(words[:max_words]); return (summary + ("..." if len(words) > max_words else "") if words else text.strip())
        except Exception as e: self.logger.error(f"Question summarization failed: {e}. Using raw text."); return text.strip()
    def export_tree_as_json(self, max_depth: int = 3) -> Dict[str, Any]:
        if not self.root: self.logger.error("Cannot export tree: Root node is missing."); return {"error": "MCTS export failed: Root node is None."}
        try: return self.root.node_to_json(max_depth=max_depth)
        except Exception as e: self.logger.error(f"Tree JSON export failed during recursion: {e}", exc_info=self.debug_logging); return {"error": f"Tree export failed: {type(e).__name__}: {e}"}

    def get_context_for_node(self, node: Node) -> Dict[str, str]:
        if not isinstance(node, Node): self.logger.error("get_context_for_node called with invalid node."); return {"error": "Invalid node provided"}
        cfg = self.config
        context: Dict[str, Any] = {
            "question_summary": self.question_summary,
            "best_answer": truncate_text(str(self.best_solution), 300),
            "best_score": f"{self.best_score:.1f}",
            "current_answer": truncate_text(node.content, 300),
            "current_sequence": str(node.sequence),
            "current_approach": node.approach_type or "N/A",
            "current_tags": ", ".join(node.descriptive_tags) or "None",
            "tree_depth": str(self.memory.get("depth", 0)),
            "branches": str(self.memory.get("branches", 0)),
            "approach_types": "None explored", # Placeholder, populated below
            "previous_best_summary": "N/A", # Placeholder, populated below
            "unfit_markers_summary": "None", # Placeholder, populated below
            "learned_approach_summary": "Default priors/scores", # Placeholder, populated below
            "explored_approaches": "None yet.", # Placeholder, populated below
            "high_scoring_examples": "None yet.", # Placeholder, populated below
            "sibling_approaches": "None." # Placeholder, populated below
        }
        # Add context from loaded state if available
        if self.loaded_initial_state and isinstance(self.loaded_initial_state, dict):
            prev_summary = self.loaded_initial_state.get("best_solution_summary"); context["previous_best_summary"] = truncate_text(prev_summary, 200) if isinstance(prev_summary, str) and prev_summary else "N/A"
            unfit = self.loaded_initial_state.get("unfit_markers", []); context["unfit_markers_summary"] = ("; ".join([f"'{m.get('summary', '?')}' ({m.get('reason', '?')})" for m in unfit[:5] if isinstance(m, dict)]) + ("..." if len(unfit) > 5 else "")) if isinstance(unfit, list) and unfit else "None"
            priors = self.loaded_initial_state.get("approach_priors")
            if priors and isinstance(priors.get("alpha"), dict) and isinstance(priors.get("beta"), dict):
                means = {}; alphas = priors["alpha"]; betas = priors["beta"]
                for app, alpha in alphas.items():
                    beta = betas.get(app, 1.0)
                    try: a_f, b_f = max(1e-9, float(alpha)), max(1e-9, float(beta)); denominator = a_f + b_f; means[app] = (a_f / denominator * 10.0) if denominator > 1e-9 else -1
                    except (ValueError, TypeError): means[app] = -1
                valid_means = {k: v for k, v in means.items() if v >= 0 and k not in ["initial", "variant"]}; top = sorted(valid_means.items(), key=lambda item: item[1], reverse=True)[:3]
                context["learned_approach_summary"] = (f"Favors: {', '.join([f'{a} ({s:.1f})' for a, s in top])}" + ("..." if len(valid_means) > 3 else "")) if top else "Default priors"
            elif priors: self.logger.warning("Loaded 'approach_priors' has invalid format. Using default summary."); context["learned_approach_summary"] = "Default priors (invalid state)"

        # Add context from the current MCTS run
        try:
            if cfg.get("track_explored_approaches", True) and self.explored_approaches:
                lines = []
                # Sort approaches by performance (Bayesian mean or average score)
                sort_key_func = lambda k: (-self.approach_alphas.get(k, 1.0) / (self.approach_alphas.get(k, 1.0) + self.approach_betas.get(k, 1.0)) if cfg.get("use_bayesian_evaluation") else -self.approach_scores.get(k, -1.0))
                sorted_apps = sorted(self.explored_approaches.keys(), key=sort_key_func)
                for app in sorted_apps:
                    thoughts = self.explored_approaches[app]; count = len(thoughts);
                    if count == 0 or app == "initial": continue # Skip initial/empty
                    score_txt = ""
                    if cfg.get("use_bayesian_evaluation"): a = self.approach_alphas.get(app, cfg.get("beta_prior_alpha", 1.0)); b = self.approach_betas.get(app, cfg.get("beta_prior_beta", 1.0)); mean_score = (a / (a + b) * 10.0) if (a + b) > 1e-9 else -1; score_txt = f"(S:{mean_score:.1f}, α:{a:.1f}, β:{b:.1f}, N={count})" if mean_score >= 0 else f"(Err, N={count})"
                    else: avg_score = self.approach_scores.get(app, 5.0); score_txt = f"(AvgS:{avg_score:.1f}, N={count})"
                    samples = "; ".join([f"'{truncate_text(t, 40)}'" for t in thoughts[-min(2, count) :]]) # Show last 1 or 2 thoughts
                    lines.append(f"- {app} {score_txt}: {samples}")
                if lines: context["explored_approaches"] = "\n".join(["Explored Summary:"] + lines[:7]) # Limit length

            if self.memory.get("high_scoring_nodes"): context["high_scoring_examples"] = "\n".join(["Top Examples (Score, Approach, Thought -> Summary):"] + [f"- S:{s:.1f} ({a}): '{truncate_text(t, 50)}' -> '{truncate_text(c, 60)}'" for s, c, a, t in self.memory["high_scoring_nodes"]])

            if cfg.get("sibling_awareness", True) and node.parent and len(node.parent.children) > 1:
                siblings = [s for s in node.parent.children if isinstance(s, Node) and s != node and s.visits > 0]
                if siblings:
                    sorted_siblings = sorted(siblings, key=lambda x: x.sequence) # Sort by sequence for consistency
                    lines = ["Sibling Thoughts:"] + [f'- N{s.sequence} "{truncate_text(s.thought, 50)}" (S:{s.get_average_score():.1f}, T:{s.descriptive_tags})' for s in sorted_siblings]
                    context["sibling_approaches"] = "\n".join(lines[:5]) # Limit length

        except Exception as e: self.logger.error(f"Error generating MCTS run context for node {node.sequence}: {e}", exc_info=self.debug_logging); context.update({"explored_approaches": "Error generating summary.", "high_scoring_examples": "Error generating summary.", "sibling_approaches": "Error generating summary."})

        # Ensure all context values are strings
        return {k: str(v) if v is not None else "" for k, v in context.items()}

    def _calculate_uct(self, node: Node, parent_visits: int) -> float:
        cfg = self.config;
        if node.visits == 0: return float("inf") # Prioritize unvisited nodes
        # Exploitation term (normalized score 0-1)
        exploit_score_0_1 = node.get_bayesian_mean() if cfg.get("use_bayesian_evaluation") else max(0.0, min(1.0, (node.get_average_score() - 1.0) / 9.0)) # Normalize 1-10 score to 0-1 approx
        # Exploration term
        exploration_bonus = 0.0
        if parent_visits > 0 and node.visits > 0: exploration_bonus = cfg.get("exploration_weight", 1.41) * math.sqrt(math.log(parent_visits + 1e-6) / node.visits)
        elif cfg.get("exploration_weight", 1.41) > 0: exploration_bonus = cfg.get("exploration_weight", 1.41) * 1.5 # Boost if parent has 0 visits (e.g., root children)
        # Penalty for known unfit nodes (unless surprising)
        unfit_penalty = -100.0 if any(isinstance(m, dict) and (m.get("id") == node.id or m.get("seq") == node.sequence) for m in getattr(self, "unfit_markers", [])) and not node.is_surprising else 0.0
        # Bonus for surprising nodes
        surprise_bonus = 0.3 if node.is_surprising else 0.0
        # Bonus for score diversity among siblings
        diversity_bonus = 0.0; diversity_weight = cfg.get("score_diversity_bonus", 0.0)
        if diversity_weight > 0 and node.parent and len(node.parent.children) > 1:
            my_score_0_1 = exploit_score_0_1
            sibling_scores_0_1 = [(s.get_bayesian_mean() if cfg.get("use_bayesian_evaluation") else max(0.0, min(1.0, (s.get_average_score() - 1.0) / 9.0))) for s in node.parent.children if isinstance(s, Node) and s != node and s.visits > 0]
            if sibling_scores_0_1: avg_sibling_score_0_1 = sum(sibling_scores_0_1) / len(sibling_scores_0_1); diversity_bonus = diversity_weight * abs(my_score_0_1 - avg_sibling_score_0_1)

        uct_score = exploit_score_0_1 + exploration_bonus + surprise_bonus + diversity_bonus + unfit_penalty
        if not math.isfinite(uct_score): self.logger.warning(f"UCT calculation for Node {node.sequence} non-finite ({uct_score}). Returning 0."); return 0.0
        if self.debug_logging: self.logger.debug(f"UCT N{node.sequence}: Score={uct_score:.3f} = Exploit={exploit_score_0_1:.3f} + Explore={exploration_bonus:.3f} + Surprise={surprise_bonus:.2f} + Diversity={diversity_bonus:.2f} + Penalty={unfit_penalty:.1f}")
        return uct_score

    def _collect_non_leaf_nodes(self, node: Optional[Node], non_leaf_nodes: List[Node], max_depth: int, current_depth: int = 0):
        """Helper for forced exploration: collects expandable nodes up to max_depth."""
        if node is None or current_depth > max_depth: return
        # Add node if it has children BUT is not yet fully expanded
        if node.children and not node.fully_expanded(): non_leaf_nodes.append(node)
        # Recurse for children within depth limit
        for child in node.children:
            if isinstance(child, Node): self._collect_non_leaf_nodes(child, non_leaf_nodes, max_depth, current_depth + 1)

    async def select(self) -> Optional[Node]:
        cfg = self.config; node = self.root; path: List[Node] = []; select_log: List[str] = ["### Select Log:"]
        if not isinstance(node, Node): self.logger.error("Select Error: Root node invalid."); return None
        path.append(node); select_log.append(f"- Start at Root N{node.sequence} (Visits: {node.visits})")

        # --- Forced Exploration Mechanism ---
        force_interval = cfg.get("force_exploration_interval", 0)
        # Check if interval is met, we have depth, and it's not the very first simulation
        if force_interval > 0 and self.simulations_completed > 0 and self.simulations_completed % force_interval == 0 and self.memory.get("depth", 0) > 1:
            candidate_nodes: List[Node] = []
            max_force_depth = max(1, self.memory.get("depth", 0) // 2) # Explore upper half of tree
            self._collect_non_leaf_nodes(self.root, candidate_nodes, max_depth=max_force_depth)
            if candidate_nodes:
                selected_node = self.random_state.choice(candidate_nodes)
                select_log.append(f"- FORCE EXPLORE triggered (Sim {self.simulations_completed}).")
                select_log.append(f"- Found {len(candidate_nodes)} candidates up to depth {max_force_depth}.")
                select_log.append(f"- Forcing selection of N{selected_node.sequence}.")
                self.logger.info(f"Select: Forced exploration to Node {selected_node.sequence}.")
                # Build path string for logging
                forced_path_nodes: List[Node] = []; curr: Optional[Node] = selected_node
                while curr: forced_path_nodes.append(curr); curr = curr.parent
                forced_path_str = " -> ".join(f"N{n.sequence}" for n in reversed(forced_path_nodes))
                self.thought_history.append(f"### Select (Forced Exploration)\nPath: {forced_path_str}\n")
                return selected_node # Return the forced node directly
            else:
                select_log.append(f"- FORCE EXPLORE triggered, but no suitable non-leaf candidates found up to depth {max_force_depth}.")
                self.logger.debug("Select: Forced exploration triggered, but no suitable candidates found.")
        # --- End Forced Exploration ---

        # --- Standard Selection Loop (UCT / Thompson Sampling) ---
        while node.children:
            valid_children = [c for c in node.children if isinstance(c, Node)]
            if not valid_children: self.logger.warning(f"Select Warning: Node {node.sequence} has children list but no valid instances. Stopping."); select_log.append(f"- Stop at N{node.sequence}: No valid children."); break

            # Prioritize expanding unvisited children first
            unvisited_children = [c for c in valid_children if c.visits == 0]
            if unvisited_children:
                selected_child = self.random_state.choice(unvisited_children) # Randomly pick one unvisited child
                node = selected_child; path.append(node); select_log.append(f"- Select UNVISITED child N{node.sequence}."); break # Break to expand this node

            # If all children visited, use selection strategy
            parent_visits = node.visits; selected_child = None; strategy_used = "None"
            use_ts = cfg.get("use_bayesian_evaluation") and cfg.get("use_thompson_sampling", True)

            if use_ts:
                strategy_used = "Thompson Sampling"
                ts_samples = [(child, sample) for child in valid_children if math.isfinite(sample := child.thompson_sample())] # Python 3.8+ walrus
                if ts_samples:
                    selected_child, best_ts_sample = max(ts_samples, key=lambda item: item[1])
                    select_log.append(f"- TS ({len(ts_samples)} children): Best sample {best_ts_sample:.3f} from N{selected_child.sequence}")
                else:
                    self.logger.warning(f"Thompson Sampling failed for all children of Node {node.sequence}. Falling back to UCT."); select_log.append(f"- TS failed. Fallback UCT."); use_ts = False # Fallback to UCT

            if not use_ts: # Either UCT selected or TS failed
                strategy_used = "UCT"
                uct_scores = [(child, score) for child in valid_children if math.isfinite(score := self._calculate_uct(child, parent_visits))] # Python 3.8+ walrus
                if uct_scores:
                    selected_child, best_uct_score = max(uct_scores, key=lambda item: item[1])
                    select_log.append(f"- UCT ({len(uct_scores)} children): Best score {best_uct_score:.3f} from N{selected_child.sequence}")
                else:
                    # This should be rare if _calculate_uct handles infinities/NaNs
                    self.logger.error(f"CRITICAL SELECT FAILURE: UCT failed for all children of Node {node.sequence}. Cannot proceed down this path."); select_log.append(f"- !! UCT failed for all children. Stop."); break

            # Descend to the selected child
            if selected_child:
                node = selected_child; path.append(node); select_log.append(f"- Descend to N{node.sequence} (using {strategy_used}).")
            else:
                # Should not happen if UCT/TS selects a child
                select_log.append(f"- Selection loop terminated at N{node.sequence} (no child selected - unexpected)."); break

            # If the selected node is not fully expanded, stop here to expand it next
            if not node.fully_expanded():
                select_log.append(f"- Stop at N{node.sequence} (not fully expanded)."); break
        # --- End Standard Selection Loop ---

        path_str = " -> ".join([f"N{n.sequence}(V:{n.visits}, S:{n.get_average_score():.1f})" for n in path]); self.thought_history.append(f"### Select Path\n{path_str}\n"); self.memory["depth"] = max(self.memory.get("depth", 0), len(path) - 1)
        if self.debug_logging: self.debug_history.append("\n".join(select_log)); self.logger.debug(f"Select path: {path_str}\nSelection Log:\n" + "\n".join(select_log[1:]))
        return node # Return the leaf or node selected for expansion

    def _check_surprise(self, parent: Node, child_content: str, child_type: str, child_family: str) -> Tuple[bool, str]:
        cfg = self.config; is_surprising = False; explanation = ""; surprise_factors: List[Dict] = []
        # 1. Semantic Distance
        if cfg.get("use_semantic_distance", True) and SKLEARN_AVAILABLE and parent.content and child_content:
            try:
                distance = calculate_semantic_distance(parent.content, child_content, self.logger, use_tfidf=True)
                threshold = cfg.get("surprise_threshold", 0.66); weight = cfg.get("surprise_semantic_weight", 0.6)
                if distance > threshold and weight > 0: factor = {"type": "semantic", "value": distance, "weight": weight, "score": distance * weight, "desc": f"Semantic Distance ({distance:.2f} > {threshold:.2f})"}; surprise_factors.append(factor); self.logger.debug(f"Surprise Check (Semantic): N{parent.sequence} -> New. Dist={distance:.2f}")
            except Exception as e: self.logger.warning(f"Surprise check failed semantic distance: {e}", exc_info=self.debug_logging)
        # 2. Philosophical/Family Shift
        parent_family = getattr(parent, "approach_family", "general"); shift_weight = cfg.get("surprise_philosophical_shift_weight", 0.3)
        if shift_weight > 0 and parent_family != child_family and child_family not in ["general", "initial", "variant"]: factor = {"type": "family_shift", "value": 1.0, "weight": shift_weight, "score": 1.0 * shift_weight, "desc": f"Approach Family Shift ('{parent_family}' -> '{child_family}')"}; surprise_factors.append(factor); self.logger.debug(f"Surprise Check (Shift): N{parent.sequence} ('{parent_family}') -> New ('{child_family}')")
        # 3. Approach Novelty (Rarity in recent tree)
        novelty_weight = cfg.get("surprise_novelty_weight", 0.3)
        if novelty_weight > 0 and child_family not in ["general", "initial", "variant"]:
            try:
                family_counts = Counter(); queue: List[Tuple[Optional[Node], int]] = [(self.root, 0)]; visited_ids: Set[str] = set(); nodes_checked = 0; MAX_BFS_NODES, MAX_BFS_DEPTH = 100, 5 # Limit BFS scope
                while queue and nodes_checked < MAX_BFS_NODES:
                    current_node, depth = queue.pop(0)
                    if not current_node or current_node.id in visited_ids or depth > MAX_BFS_DEPTH: continue
                    visited_ids.add(current_node.id); nodes_checked += 1; family_counts[getattr(current_node, "approach_family", "general")] += 1
                    if depth < MAX_BFS_DEPTH: queue.extend((child, depth + 1) for child in current_node.children if isinstance(child, Node) and child.id not in visited_ids)
                child_family_count = family_counts.get(child_family, 0)
                # Consider novel if it's the first or second time seeing this family recently
                if child_family_count <= 1: factor = {"type": "novelty", "value": 1.0, "weight": novelty_weight, "score": 1.0 * novelty_weight, "desc": f"Novel Approach Family ('{child_family}', count={child_family_count} in {nodes_checked} nodes checked)"}; surprise_factors.append(factor); self.logger.debug(f"Surprise Check (Novelty): Family '{child_family}' count={child_family_count}/{nodes_checked} nodes.")
            except Exception as e: self.logger.warning(f"Surprise check failed novelty BFS: {e}", exc_info=self.debug_logging)

        # Combine factors and check threshold
        if surprise_factors:
            total_weighted_score = sum(f["score"] for f in surprise_factors); total_weight = sum(f["weight"] for f in surprise_factors)
            if total_weight > 1e-6: # Avoid division by zero
                overall_score = total_weighted_score / total_weight; overall_threshold = cfg.get("surprise_overall_threshold", 0.9)
                if overall_score >= overall_threshold:
                    is_surprising = True; factor_descs = [f"- {f['desc']} (Contribution: {f['score']:.2f})" for f in surprise_factors]; explanation = f"Surprise Triggered! (Overall Score: {overall_score:.2f} >= {overall_threshold:.2f})\nContributing Factors (Total Weight: {total_weight:.2f}):\n" + "\n".join(factor_descs); self.logger.info(f"Surprise DETECTED: N{parent.sequence} -> New Child. Score={overall_score:.2f}\n{explanation}")
            else: self.logger.debug("Surprise check: Total weight zero. No surprise.")
        else: self.logger.debug("Surprise check: No factors found.")
        return is_surprising, explanation

    async def expand(self, node: Node) -> Optional[Node]: # Modified return type
        cfg = self.config
        if not isinstance(node, Node): self.logger.error("Expand called with invalid node."); return None
        if node.fully_expanded(): self.logger.warning(f"Attempted expand Node {node.sequence}, but full ({len(node.children)} children)."); return None # Should not happen if select works right
        self.logger.debug(f"Expanding Node {node.sequence}..."); expand_log_entry = f"### Expand N{node.sequence}\n"
        try:
            # 1. Generate Thought/Critique
            await self.llm.progress(f"N{node.sequence}: Generating thought..."); context = self.get_context_for_node(node)
            # Basic check for essential context fields
            if not all(k in context for k in ["current_answer", "question_summary", "best_answer"]): self.logger.error(f"Expand N{node.sequence}: Critical context missing."); expand_log_entry += "... Error: Missing context.\n"; self.thought_history.append(expand_log_entry); return None
            thought_text = await self.llm.generate_thought(node.content, context, cfg)
            if not isinstance(thought_text, str) or not thought_text.strip() or thought_text.startswith("Error:"): self.logger.error(f"Expand N{node.sequence}: Thought gen failed: '{thought_text}'"); expand_log_entry += f"... Thought Error: {thought_text}\n"; self.thought_history.append(expand_log_entry); return None
            thought = thought_text.strip(); approach_type, approach_family = classify_approach(thought, APPROACH_TAXONOMY, APPROACH_METADATA, self.random_state, self.logger)
            self.logger.debug(f"N{node.sequence} -> Thought ({approach_type}/{approach_family}): '{truncate_text(thought, 100)}'"); expand_log_entry += f"... Thought ({approach_type}/{approach_family}): {thought}\n"
            self.explored_thoughts.add(thought); self.explored_approaches.setdefault(approach_type, []).append(thought)

            # 2. Update Analysis based on Thought
            await self.llm.progress(f"N{node.sequence}: Updating analysis..."); updated_content_text = await self.llm.update_approach(node.content, thought, context, cfg)
            if not isinstance(updated_content_text, str) or not updated_content_text.strip() or updated_content_text.startswith("Error:"): self.logger.error(f"Expand N{node.sequence}: Update failed: '{updated_content_text}'"); expand_log_entry += f"... Update Error: {updated_content_text}\n"; self.thought_history.append(expand_log_entry); return None
            child_content = updated_content_text.strip(); expand_log_entry += f"... Updated Analysis: {truncate_text(child_content, 150)}\n"

            # 3. Generate Tags for New Content
            await self.llm.progress(f"N{node.sequence}: Generating tags..."); child_tags: List[str] = []
            try: child_tags = await self._generate_tags_for_node(child_content); expand_log_entry += f"... Tags: {child_tags}\n"; self.logger.debug(f"N{node.sequence} -> Child Tags: {child_tags}")
            except Exception as tag_err: self.logger.error(f"Expand N{node.sequence}: Tag gen failed: {tag_err}", exc_info=self.debug_logging); expand_log_entry += f"... Tag Error: {tag_err}\n"

            # 4. Check for Surprise
            is_surprising, surprise_expl = False, ""
            try:
                is_surprising, surprise_expl = self._check_surprise(node, child_content, approach_type, approach_family)
                if is_surprising: expand_log_entry += f"**SURPRISE DETECTED!**\n{surprise_expl}\n"
            except Exception as surprise_err: self.logger.error(f"Expand N{node.sequence}: Surprise check failed: {surprise_err}", exc_info=self.debug_logging); expand_log_entry += f"... Surprise Error: {surprise_err}\n"

            # 5. Create Child Node
            child_sequence = self.get_next_sequence(); child_alpha = max(1e-9, float(cfg.get("beta_prior_alpha", 1.0))); child_beta = max(1e-9, float(cfg.get("beta_prior_beta", 1.0)))
            child_node = Node(content=child_content, sequence=child_sequence, parent=node, is_surprising=is_surprising, surprise_explanation=surprise_expl, approach_type=approach_type, approach_family=approach_family, thought=thought, max_children=cfg["max_children"], use_bayesian_evaluation=cfg["use_bayesian_evaluation"], alpha=child_alpha, beta=child_beta, value=0.0, descriptive_tags=child_tags)
            node.add_child(child_node)
            if is_surprising: self.surprising_nodes.append(child_node)
            # Track branching factor approx
            if len(node.children) == 2: self.memory["branches"] = self.memory.get("branches", 0) + 1

            expand_log_entry += f"--> Created Child N{child_sequence}\n"; self.thought_history.append(expand_log_entry); self.logger.info(f"Expanded N{node.sequence} -> N{child_sequence} (Approach: {approach_type}, Surprise: {is_surprising})")
            return child_node # Return the newly created child node
        except Exception as e: self.logger.error(f"Expand N{node.sequence} unexpected error: {e}", exc_info=self.debug_logging); expand_log_entry += f"... Unexpected Expansion Error: {type(e).__name__}: {e}\n"; self.thought_history.append(expand_log_entry); return None

    async def _generate_tags_for_node(self, text: str) -> List[str]:
        if not text or not isinstance(text, str): return []
        tags: List[str] = []
        try:
            model_name = self.llm.resolve_model(self.model_body)
            if not model_name: self.logger.error("Tag gen failed: No model name resolved."); return []
            # Truncate input for tag generation to avoid excessive token usage
            prompt = GENERATE_TAGS_PROMPT.format(analysis_text=truncate_text(text, 1000)) # Limit to ~1000 chars
            raw_response = await self.llm.get_completion(model_name, [{"role": "user", "content": prompt}])
            if not raw_response or raw_response.startswith("Error:"): self.logger.warning(f"Tag gen LLM failed: {raw_response}"); return []

            # Clean the LLM response robustly
            cleaned = re.sub(r"^\s*```.*?\s*$", "", raw_response, flags=re.DOTALL | re.MULTILINE).strip() # Remove markdown blocks
            cleaned = re.sub(r"^\s*(tags|keywords|output|response)[:\-]?\s*", "", cleaned, flags=re.IGNORECASE).strip() # Remove prefixes
            cleaned = re.sub(r"[`'*_]", "", cleaned) # Remove common markdown chars
            cleaned = re.sub(r"^\s*[-*+]\s*", "", cleaned, flags=re.MULTILINE).strip() # Remove list markers at line starts

            potential_tags = re.split(r"[,\n;]+", cleaned) # Split by common delimiters
            processed_tags: Set[str] = set()
            final_tags: List[str] = []
            for tag in potential_tags:
                # Clean individual tags: remove extra spaces, quotes, brackets, convert to lower for checking uniqueness
                t_lower = re.sub(r"\s+", " ", tag.strip().strip("'.\"-()[]{}").strip()).lower()
                # Basic validation: not empty, reasonable length, not just numbers, not 'none', not duplicate
                if t_lower and 1 < len(t_lower) < 50 and not t_lower.isdigit() and t_lower != "none" and t_lower not in processed_tags:
                    # Store the original-cased tag, but track uniqueness using lower case
                    original_casing_tag = re.sub(r"\s+", " ", tag.strip().strip("'.\"-()[]{}").strip())
                    final_tags.append(original_casing_tag)
                    processed_tags.add(t_lower)
                    if len(final_tags) >= 5: break # Limit to max 5 tags
            tags = final_tags
            if self.debug_logging: self.logger.debug(f"Tag Gen: Raw='{raw_response}', Cleaned='{cleaned}', Final={tags}")
        except Exception as e: self.logger.error(f"Tag gen process failed: {e}", exc_info=self.debug_logging); return []
        return tags

    async def simulate(self, node: Node) -> Optional[float]:
        cfg = self.config
        if not isinstance(node, Node): self.logger.error("Simulate called with invalid node."); return None
        self.logger.debug(f"Simulating Node {node.sequence} (Approach: {node.approach_type}, Tags: {node.descriptive_tags})..."); simulation_score: Optional[float] = None; raw_llm_response: Union[int, str] = 0; sim_log_entry = f"### Evaluate N{node.sequence} (Tags:{node.descriptive_tags})\n"
        try:
            # Handle empty node content
            if not node.content or not isinstance(node.content, str): self.logger.warning(f"Node {node.sequence} content is empty. Assigning score 1.0."); raw_llm_response = 1; simulation_score = 1.0; sim_log_entry += "... Score: 1.0/10 (Node empty)\n"
            else:
                # Call LLM for evaluation
                await self.llm.progress(f"Evaluating N{node.sequence}..."); context = self.get_context_for_node(node)
                # Ensure context for evaluation prompt is present
                eval_req_keys = ["answer_to_evaluate", "question_summary", "best_answer", "best_score"]
                if "answer_to_evaluate" not in context: context["answer_to_evaluate"] = node.content # Add the content to evaluate if missing
                if not all(k in context for k in eval_req_keys): self.logger.error(f"Simulate N{node.sequence}: Critical eval context missing: {[k for k in eval_req_keys if k not in context]}"); sim_log_entry += "... Error: Missing eval context.\n"; self.thought_history.append(sim_log_entry); return None

                eval_response = await self.llm.evaluate_answer(node.content, context, cfg)

                # Parse LLM evaluation response
                if isinstance(eval_response, int): # Direct integer response
                    raw_llm_response = eval_response; simulation_score = float(max(1, min(10, raw_llm_response))); sim_log_entry += f"... LLM Score: {simulation_score:.1f}/10 (Raw: {raw_llm_response})\n"
                elif isinstance(eval_response, str) and eval_response.startswith("Error:"): # Error string from LLM
                    self.logger.error(f"Simulate N{node.sequence}: Eval LLM failed: {eval_response}"); sim_log_entry += f"... Eval Error (LLM): {eval_response}\n"; simulation_score = None; raw_llm_response = eval_response
                else: # Unexpected response type
                    self.logger.error(f"Simulate N{node.sequence}: Unexpected eval response type: {type(eval_response)}. Response: '{eval_response}'"); sim_log_entry += f"... Eval Error: Unexpected type {type(eval_response)}.\n"; simulation_score = None; raw_llm_response = f"Error: Unexpected eval type {type(eval_response)}"

            # Update approach stats if evaluation was successful
            if simulation_score is not None:
                node.raw_scores.append(raw_llm_response if isinstance(raw_llm_response, (int, float)) else -1) # Store raw score
                approach = node.approach_type or "unknown"
                if approach != "unknown": # Don't track stats for 'unknown'
                    if cfg.get("use_bayesian_evaluation"):
                        # Convert 1-10 score to successes/failures for Beta update
                        successes = max(0.0, simulation_score - 1.0); failures = max(0.0, 10.0 - simulation_score)
                        current_alpha = self.approach_alphas.get(approach, cfg.get("beta_prior_alpha", 1.0)); current_beta = self.approach_betas.get(approach, cfg.get("beta_prior_beta", 1.0))
                        self.approach_alphas[approach] = max(1e-9, current_alpha + successes); self.approach_betas[approach] = max(1e-9, current_beta + failures)
                        if self.debug_logging: self.logger.debug(f"Updated Bayes '{approach}': α={self.approach_alphas[approach]:.2f}, β={self.approach_betas[approach]:.2f} (+S:{successes:.1f}, F:{failures:.1f})")
                    else:
                        # Update simple moving average for the approach score
                        current_avg = self.approach_scores.get(approach, simulation_score); smoothing_factor = 0.3; self.approach_scores[approach] = (smoothing_factor * simulation_score) + ((1 - smoothing_factor) * current_avg)
                        if self.debug_logging: self.logger.debug(f"Updated Avg score '{approach}': {self.approach_scores[approach]:.2f}")

                # Update high-score memory
                memory_threshold = 7.0 # Example threshold
                if simulation_score >= memory_threshold:
                    memory_entry = (simulation_score, node.content, node.approach_type, node.thought); current_memory: List = self.memory.get("high_scoring_nodes", [])
                    current_memory.append(memory_entry); self.memory["high_scoring_nodes"] = sorted(current_memory, key=lambda x: x, reverse=True)[: cfg.get("memory_cutoff", 5)] # Sort by score (index 0), keep top N

            # Log final simulation result
            if simulation_score is not None: self.logger.info(f"Simulated N{node.sequence}: Score = {simulation_score:.1f}/10 (Raw LLM: {raw_llm_response})")
            else: self.logger.warning(f"Simulation FAILED for Node {node.sequence}.")
            self.thought_history.append(sim_log_entry) # Log simulation attempt details

        except Exception as e: self.logger.error(f"Simulate N{node.sequence} unexpected error: {e}", exc_info=self.debug_logging); sim_log_entry += f"... Unexpected Sim Error: {type(e).__name__}: {e}\n"; self.thought_history.append(sim_log_entry); return None # Return None on unexpected error
        return simulation_score # Return score or None if failed

    def backpropagate(self, node: Node, score: float):
        cfg = self.config
        if not isinstance(node, Node): self.logger.error("Backpropagate called with invalid node."); return
        if not isinstance(score, (int, float)) or not math.isfinite(score): self.logger.error(f"Backpropagate received invalid score: {score}. Aborting backpropagation for this path."); return
        self.logger.debug(f"Backpropagating score {score:.2f} from N{node.sequence}..."); backprop_path_nodes: List[str] = []
        current_node: Optional[Node] = node
        # For Bayesian: calculate successes/failures once
        successes = max(0.0, score - 1.0); failures = max(0.0, 10.0 - score)

        while current_node:
            node_id = current_node.sequence; backprop_path_nodes.append(f"N{node_id}"); current_node.visits += 1
            # Update node statistics based on config
            if cfg.get("use_bayesian_evaluation"):
                # Ensure alpha/beta exist before updating
                if current_node.alpha is not None and current_node.beta is not None: current_node.alpha = max(1e-9, current_node.alpha + successes); current_node.beta = max(1e-9, current_node.beta + failures)
                else: self.logger.warning(f"Backprop Warning: N{node_id} missing alpha/beta during update. Initializing with this score."); prior_alpha = max(1e-9, cfg.get("beta_prior_alpha", 1.0)); prior_beta = max(1e-9, cfg.get("beta_prior_beta", 1.0)); current_node.alpha = prior_alpha + successes; current_node.beta = prior_beta + failures
            else: # Use traditional value accumulation
                if current_node.value is not None: current_node.value += score
                else: self.logger.warning(f"Backprop Warning: N{node_id} missing value during update. Initializing with this score."); current_node.value = score

            # Debug logging for the update
            if self.debug_logging:
                if cfg.get("use_bayesian_evaluation") and current_node.alpha is not None: details = f"α={current_node.alpha:.2f}, β={current_node.beta:.2f} (Mean: {current_node.get_bayesian_mean():.3f})"
                elif not cfg.get("use_bayesian_evaluation") and current_node.value is not None: details = f"Value={current_node.value:.2f}, AvgScore={current_node.get_average_score():.2f}"
                else: details = "Params Missing!"
                self.logger.debug(f"  Backprop updated N{node_id}: Visits={current_node.visits}, {details}")

            # Move to parent
            current_node = current_node.parent

        # Log the path taken for backpropagation
        final_path_str = " -> ".join(reversed(backprop_path_nodes)); self.thought_history.append(f"### Backpropagate Score {score:.1f}\n... Path: {final_path_str}\n")
        if self.debug_logging: self.logger.debug(f"Backprop complete for score {score:.1f}. Path: {final_path_str}")


    async def search(self, sims_per_iter: int) -> bool:
        """Performs one iteration of MCTS simulations (Selection, Expansion, Simulation, Backpropagation)."""
        cfg = self.config; debug = cfg.get("debug_logging", False); show_sim_details = cfg.get("show_processing_details", False)
        current_iteration = self.iterations_completed + 1
        self.logger.info(f"--- Starting MCTS Iteration {current_iteration}/{cfg.get('max_iterations', 0)} ({sims_per_iter} sims) ---")

        for i in range(sims_per_iter):
            self.simulations_completed += 1; current_simulation = i + 1; simulation_id = f"Iter {current_iteration}.{current_simulation}"
            self.thought_history.append(f"\n### Simulation {simulation_id} (Total: {self.simulations_completed})\n")
            if debug: self.logger.debug(f"--- Starting Sim {simulation_id} ---")

            sim_summary = ""; node_to_evaluate: Optional[Node] = None; evaluated_score: Optional[float] = None; selected_node: Optional[Node] = None

            # 1. Selection
            try:
                selected_node = await self.select()
                if not selected_node:
                    self.logger.error(f"Sim {simulation_id}: Selection failed. Skipping simulation."); sim_summary += "Select: FAILED.\n"; self.thought_history.append(f"... {sim_summary}"); continue # Skip to next sim
                sim_summary += f"Select: N{selected_node.sequence} (V:{selected_node.visits}, S:{selected_node.get_average_score():.1f}, T:{selected_node.descriptive_tags})\n"; node_to_evaluate = selected_node # Tentatively set node_to_evaluate
            except Exception as e:
                self.logger.error(f"Sim {simulation_id}: Selection error: {e}", exc_info=debug); sim_summary += f"Select: Error ({type(e).__name__}).\n"; self.thought_history.append(f"... {sim_summary}"); continue # Skip to next sim

            # 2. Expansion (if node not fully expanded and has content)
            if not selected_node.fully_expanded() and selected_node.content:
                if debug: self.logger.debug(f"Sim {simulation_id}: Expanding N{selected_node.sequence}.")
                sim_summary += "Expand: Attempting...\n"; expanded_node: Optional[Node] = None
                try:
                    expanded_node = await self.expand(selected_node) # expand returns the new Node or None
                    if expanded_node:
                        node_to_evaluate = expanded_node # Evaluate the newly expanded node
                        thought_str = str(expanded_node.thought).strip() if expanded_node.thought else "(Thought N/A)"
                        # Verbose summary includes full thought, no longer truncated here
                        sim_summary += f'  Expand Thought: "{thought_str}"\n'
                        sim_summary += f"  Expand Result: --> N{expanded_node.sequence} ({expanded_node.approach_type}, S:{expanded_node.get_average_score():.1f}, T:{expanded_node.descriptive_tags})\n"
                        if debug: self.logger.debug(f"Sim {simulation_id}: Expanded {selected_node.sequence} -> {expanded_node.sequence}.")
                    else:
                        # Expansion failed, evaluate the original selected node
                        self.logger.warning(f"Sim {simulation_id}: Expansion failed for N{selected_node.sequence}. Will evaluate original node."); sim_summary += f"  Expand Result: FAILED. Evaluating N{selected_node.sequence}.\n"
                except Exception as e:
                    self.logger.error(f"Sim {simulation_id}: Expansion error: {e}", exc_info=debug); sim_summary += f"  Expand: Error ({type(e).__name__}). Evaluating N{selected_node.sequence}.\n"
            else:
                # Node was already fully expanded or had no content to expand from
                expand_skip_reason = "Node full" if selected_node.fully_expanded() else "Node no content"; sim_summary += f"Expand: Skipped ({expand_skip_reason}). Evaluating N{selected_node.sequence}.\n"; self.logger.debug(f"Sim {simulation_id}: Skip expansion N{selected_node.sequence} ({expand_skip_reason}).")

            # 3. Simulation (Evaluation)
            # Ensure we have a valid node with content to evaluate
            if node_to_evaluate and node_to_evaluate.content:
                if debug: self.logger.debug(f"Sim {simulation_id}: Evaluating N{node_to_evaluate.sequence}.")
                sim_summary += f"Evaluate: N{node_to_evaluate.sequence}...\n"
                try:
                    evaluated_score = await self.simulate(node_to_evaluate)
                    if evaluated_score is not None: sim_summary += f"  Evaluate Score: {evaluated_score:.1f}/10\n"
                    else: self.logger.warning(f"Sim {simulation_id}: Evaluation failed for N{node_to_evaluate.sequence} (returned None)."); sim_summary += f"  Evaluate Score: FAILED.\n"; evaluated_score = None # Ensure score is None
                except Exception as e: self.logger.error(f"Sim {simulation_id}: Simulation error during evaluation: {e}", exc_info=debug); sim_summary += f"  Evaluate: Error ({type(e).__name__}).\n"; evaluated_score = None
            elif node_to_evaluate: # Node exists but has no content
                sim_summary += f"Evaluate: Skipped N{node_to_evaluate.sequence} (no content).\n"; self.logger.debug(f"Sim {simulation_id}: Skip eval N{node_to_evaluate.sequence} (no content).")
            else: # Should not happen if selection worked
                sim_summary += f"Evaluate: Skipped (no valid node selected/expanded).\n"; self.logger.error(f"Sim {simulation_id}: No node available for evaluation.")

            # 4. Backpropagation (if evaluation successful)
            if evaluated_score is not None and node_to_evaluate:
                if debug: self.logger.debug(f"Sim {simulation_id}: Backpropagating score {evaluated_score:.1f} from N{node_to_evaluate.sequence}.")
                sim_summary += f"Backpropagate: Score {evaluated_score:.1f} from N{node_to_evaluate.sequence}...\n"
                try: self.backpropagate(node_to_evaluate, evaluated_score); sim_summary += "  Backpropagate: OK.\n"
                except Exception as e: self.logger.error(f"Sim {simulation_id}: Backpropagation error: {e}", exc_info=debug); sim_summary += f"  Backpropagate: Error ({type(e).__name__}).\n"

                # Check for new best score
                if evaluated_score > self.best_score:
                    self.best_score = evaluated_score; self.best_solution = str(node_to_evaluate.content) if node_to_evaluate.content else self.best_solution; self.high_score_counter = 0; # Reset stability counter
                    sim_summary += f"🏆 New Best Score Found! {self.best_score:.1f}/10\n"; node_info = f"N{node_to_evaluate.sequence} ({node_to_evaluate.approach_type}) Tags:{node_to_evaluate.descriptive_tags}"; self.thought_history.append(f"### New Best! Score: {self.best_score:.1f} ({node_info})\n"); self.logger.info(f"Sim {simulation_id}: New best! S:{self.best_score:.1f}, Node: {node_info}")
                # Check for early stopping stability
                elif cfg.get("early_stopping", True) and evaluated_score >= cfg.get("early_stopping_threshold", 10.0):
                    self.high_score_counter += 1; stability_required = cfg.get("early_stopping_stability", 2); sim_summary += f"  Stability Check: Score {evaluated_score:.1f} >= Thresh. Stability: {self.high_score_counter}/{stability_required}\n"
                    if self.high_score_counter >= stability_required:
                        self.logger.info(f"Sim {simulation_id}: Early stopping condition met (Score {evaluated_score:.1f} stable for {self.high_score_counter} simulations)."); await self.llm.emit_message(f"**Stopping early:** Score ({self.best_score:.1f}/10) reached threshold and remained stable.")
                        if show_sim_details: await self.llm.emit_message(f"--- Sim {simulation_id} Summary ---\n{sim_summary}") # Show final sim summary
                        self.thought_history.append(f"... {sim_summary}"); return False # Signal to stop searching
                else: # Score was not high enough or dropped below threshold
                    self.high_score_counter = 0
            else: # Backpropagation skipped (no score or no node)
                self.high_score_counter = 0; # Reset stability counter if eval failed
                sim_summary += "Backpropagate: Skipped (No score or invalid node).\n"

            # Emit per-simulation details if enabled
            if show_sim_details:
                await self.llm.emit_message(f"--- Sim {simulation_id} Summary ---\n{sim_summary}")
                await asyncio.sleep(0.05) # Small delay to allow UI update

            # Store summary in history regardless of show_details setting
            self.thought_history.append(f"... {sim_summary}")
            # --- End of Simulation Loop ---

        # Iteration finished
        # Moved incrementing iterations_completed outside the loop, to be done after MCTS finishes
        self.logger.info(f"--- Finished Iteration {current_iteration}. Current Best Score: {self.best_score:.1f} ---")
        return True # Signal to continue searching unless early stop happened


    def find_best_final_node(self) -> Optional[Node]:
        """Finds the node associated with the best recorded solution content."""
        if not self.root: return None;
        if not self.best_solution: return self.root # Fallback to root if no best solution content recorded

        best_match_node: Optional[Node] = None; min_score_diff = float("inf")
        # Clean the target content once
        target_content_cleaned = re.sub(r"^\s*```[\s\S]*?```\s*$", "", str(self.best_solution), flags=re.MULTILINE).strip()

        # Breadth-first search to find the exact content match
        queue: List[Node] = [self.root]; visited_ids: Set[str] = {self.root.id}
        nodes_with_matching_content: List[Node] = []

        while queue:
            current_node = queue.pop(0)
            # Clean node content for comparison
            node_content_cleaned = re.sub(r"^\s*```[\s\S]*?```\s*$", "", str(current_node.content), flags=re.MULTILINE).strip()

            if node_content_cleaned == target_content_cleaned:
                 nodes_with_matching_content.append(current_node)
                 self.logger.debug(f"Found potential best node (content match): N{current_node.sequence} (Score: {current_node.get_average_score():.2f})")

            # Add children to queue
            for child in current_node.children:
                if isinstance(child, Node) and child.id not in visited_ids:
                    visited_ids.add(child.id); queue.append(child)

        # Select the best node among those with matching content
        if nodes_with_matching_content:
            # Choose the one with the score closest to the recorded best_score
            best_match_node = min(nodes_with_matching_content, key=lambda n: abs(n.get_average_score() - self.best_score))
            min_score_diff = abs(best_match_node.get_average_score() - self.best_score)
            self.logger.info(f"Found best final node by content match: N{best_match_node.sequence} (Score Diff: {min_score_diff:.3f})")
            return best_match_node
        else:
             # Fallback: Find the node with the highest average score if no content match
            self.logger.warning(f"Could not find node by exact content match for best_solution. Searching for highest score node...")
            all_nodes: List[Node] = []; q: List[Node] = [self.root]; visited: Set[str] = {self.root.id}
            while q:
                curr = q.pop(0)
                # Only consider nodes that were actually visited/evaluated
                if curr.visits > 0: all_nodes.append(curr)
                for child in curr.children:
                    if isinstance(child, Node) and child.id not in visited: visited.add(child.id); q.append(child)

            if all_nodes:
                highest_score_node = max(all_nodes, key=lambda n: n.get_average_score())
                self.logger.info(f"Falling back to highest score node: N{highest_score_node.sequence} (Score: {highest_score_node.get_average_score():.2f})")
                return highest_score_node
            else:
                 self.logger.warning("Fallback failed: No visited nodes found. Returning root.")
                 return self.root # Ultimate fallback


    def get_state_for_persistence(self) -> Optional[Dict[str, Any]]:
        """Gathers key MCTS results into a dictionary for saving."""
        if not self.root: self.logger.error("Cannot get state for persistence: Root node is missing."); return None
        # Use SCRIPT_VERSION when creating state dict
        state: Dict[str, Any] = {"version": SCRIPT_VERSION, "timestamp": datetime.now().isoformat()}
        try:
            state["best_score"] = round(self.best_score, 3); best_node = self.find_best_final_node()
            state["best_solution_content"] = str(self.best_solution) # Store full best content
            state["best_solution_summary"] = truncate_text(self.best_solution, 400) # Store summary
            state["best_node_tags"] = best_node.descriptive_tags[:] if best_node else []
            state["best_node_sequence"] = best_node.sequence if best_node else None

            # Save approach priors if using Bayesian evaluation
            if self.config.get("use_bayesian_evaluation"):
                if isinstance(self.approach_alphas, dict) and self.approach_alphas and isinstance(self.approach_betas, dict) and self.approach_betas: state["approach_priors"] = {"alpha": {k: round(v, 4) for k, v in self.approach_alphas.items()}, "beta": {k: round(v, 4) for k, v in self.approach_betas.items()}}
                else: self.logger.warning("Saving state, but Bayesian approach priors are invalid or empty. Saving null for priors."); state["approach_priors"] = None

            # Find top nodes and unfit markers by traversing the tree
            all_nodes: List[Node] = []; queue: List[Node] = [self.root]; visited_ids: Set[str] = {self.root.id}
            while queue:
                current_node = queue.pop(0)
                # Only include nodes that were actually visited/simulated
                if current_node.visits > 0: all_nodes.append(current_node)
                for child in current_node.children:
                    if isinstance(child, Node) and child.id not in visited_ids: visited_ids.add(child.id); queue.append(child)

            # Store top N nodes based on score
            sorted_nodes = sorted(all_nodes, key=lambda n: n.get_average_score(), reverse=True); state["top_nodes"] = [node.node_to_state_dict() for node in sorted_nodes[:3]] # Save top 3

            # Identify unfit nodes based on score and visits
            unfit_markers = []; score_threshold = self.config.get("unfit_score_threshold", 4.0); visit_threshold = self.config.get("unfit_visit_threshold", 3)
            for node in all_nodes:
                avg_score = node.get_average_score()
                if node.visits >= visit_threshold and avg_score < score_threshold: unfit_markers.append({"id": node.id, "seq": node.sequence, "summary": truncate_text(node.thought or node.content, 80), "reason": f"Low score ({avg_score:.1f} < {score_threshold}) after {node.visits} visits", "tags": node.descriptive_tags[:]})
            state["unfit_markers"] = unfit_markers[:10] # Limit stored unfit markers

            self.logger.info(f"Generated state for persistence. Best Score: {state['best_score']:.2f}, Top Nodes: {len(state['top_nodes'])}, Unfit Markers: {len(state['unfit_markers'])}"); return state
        except Exception as e: self.logger.error(f"Failed to generate state persistence dictionary: {e}", exc_info=self.debug_logging); return None

    def get_final_synthesis_context(self) -> Optional[Dict[str, str]]:
        """Prepares context specifically for the final synthesis LLM prompt (Uses FULL content, no truncation)."""
        if not self.root or not self.best_solution: self.logger.error("Cannot generate synthesis context: Root node or best_solution missing."); return None

        best_node = self.find_best_final_node()
        if not best_node: self.logger.warning("Synthesis context: Best final node not found, falling back to root."); best_node = self.root

        # Reconstruct the path from root to the best node
        path_to_best: List[Node] = []; current_node: Optional[Node] = best_node
        while current_node: path_to_best.append(current_node); current_node = current_node.parent
        path_to_best.reverse() # Order from root to best

        # Format the path thoughts
        path_thoughts_lines = []
        for i, node in enumerate(path_to_best):
            if i > 0 and node.thought: # Skip root's "thought" (it has none)
                parent_seq = node.parent.sequence if node.parent else "?"
                path_thoughts_lines.append(f"- N{node.sequence} (From N{parent_seq}, Approach: {node.approach_type}): {node.thought.strip()}") # Keep full thought
            elif i == 0: # Root node
                path_thoughts_lines.append(f"- N{node.sequence} (Initial Root)")
        path_thoughts_str = "\n".join(path_thoughts_lines) if path_thoughts_lines else "No development path recorded."

        try:
            synthesis_context = {
                "question_summary": self.question_summary, # Keep question summary short
                "initial_analysis_summary": str(self.root.content) if self.root else "N/A", # Use FULL initial content
                "best_score": f"{self.best_score:.1f}",
                "path_thoughts": path_thoughts_str, # Full path thoughts
                "final_best_analysis_summary": str(self.best_solution) if self.best_solution else "N/A", # Use FULL best content
            }
            # Log the lengths for debugging potential context size issues
            if self.debug_logging:
                for key, value in synthesis_context.items(): self.logger.debug(f"Synth Context '{key}' length: {len(value)}")
            return synthesis_context
        except Exception as e: self.logger.error(f"Error assembling final synthesis context: {e}", exc_info=self.debug_logging); return None

    def formatted_output(self) -> str:
        """Generates a formatted markdown string summarizing the MCTS run (Verbose - FULL thoughts)."""
        cfg = self.config
        # Use SCRIPT_VERSION in header
        output_lines = [f"# MCTS Summary (Verbose) v{SCRIPT_VERSION}", f"*Run Completed: {datetime.now():%Y-%m-%d %H:%M:%S}*"]
        try:
            # Best Analysis Section
            best_node = self.find_best_final_node(); best_tags = f"Tags: {best_node.descriptive_tags}" if best_node and best_node.descriptive_tags else "Tags: []"; output_lines.append(f"\n## Best Analysis Found (Score: {self.best_score:.1f}/10)\n**{best_tags}**")
            # Clean markdown from the final solution before displaying in text block
            cleaned_best_solution = re.sub(r"^\s*```[\s\S]*?```\s*$", "", str(self.best_solution), flags=re.MULTILINE).strip(); output_lines.append(f"\n```text\n{cleaned_best_solution}\n```\n")

            # Top Nodes Section
            output_lines.append("\n## Top Performing Nodes & Driving Thoughts (Max 5)")
            all_nodes: List[Node] = []; queue = [self.root] if self.root else []; visited_ids = {self.root.id} if self.root else set()
            while queue:
                current_node = queue.pop(0)
                if current_node.visits > 0: all_nodes.append(current_node) # Only include visited nodes
                for child in current_node.children:
                    if isinstance(child, Node) and child.id not in visited_ids: visited_ids.add(child.id); queue.append(child)

            if not all_nodes: output_lines.append("*No nodes visited.*")
            else:
                sorted_nodes = sorted(all_nodes, key=lambda n: n.get_average_score(), reverse=True)
                for i, node in enumerate(sorted_nodes[:5]):
                    score = node.get_average_score(); score_details = ""
                    if cfg.get("use_bayesian_evaluation") and node.alpha is not None and node.beta is not None: score_details = f"(α={node.alpha:.1f}, β={node.beta:.1f})"
                    elif not cfg.get("use_bayesian_evaluation") and node.value is not None: score_details = f"(Value={node.value:.1f})"
                    # Display FULL thought here
                    output_lines.append(f"### {i+1}. Node {node.sequence}: Score {score:.1f}/10 {score_details}\n- **Info**: Approach={node.approach_type}({node.approach_family}), Visits={node.visits}, Tags={node.descriptive_tags}\n- **Thought**: {node.thought or '(Root Node)'}")
                    if node.is_surprising: output_lines.append(f"- **Surprising**: Yes ({truncate_text(node.surprise_explanation, 100)})") # Keep surprise explanation truncated

            # Most Explored Path Section
            output_lines.append("\n## Most Explored Path (Based on Visits)")
            exploration_path: List[Node] = []; current_node = self.root
            if current_node:
                exploration_path.append(current_node)
                while current_node and current_node.children:
                    most_visited_child = current_node.best_child() # Use best_child which handles ties
                    if not most_visited_child or most_visited_child.visits == 0: break # Stop if no visited child
                    exploration_path.append(most_visited_child); current_node = most_visited_child
            if len(exploration_path) > 1:
                output_lines.append("```"); # Start code block for tree structure
                for i, node in enumerate(exploration_path): prefix = "  " * i + ("└─ " if i == len(exploration_path) - 1 else "├─ "); output_lines.append(f"{prefix}N{node.sequence} ({node.approach_type}, S:{node.get_average_score():.1f}, V:{node.visits}, T:{node.descriptive_tags})")
                output_lines.append("```\n") # End code block
            elif self.root: output_lines.append(f"*Only Root Node N{self.root.sequence} explored.*")
            else: output_lines.append("*No exploration path found (root missing?).*")

            # Surprising Nodes Section
            output_lines.append("\n## Surprising Nodes Found (Max 5)")
            if self.surprising_nodes:
                 # Show last 5 surprising nodes encountered
                 for n in self.surprising_nodes[-5:]: output_lines.append(f"- **N{n.sequence}** ({n.approach_type}, S:{n.get_average_score():.1f}, T:{n.descriptive_tags}): {truncate_text(n.surprise_explanation, 150)}") # Truncate explanation here
            else: output_lines.append("*No surprising nodes detected.*")

            # Approach Performance Section
            output_lines.append("\n## Approach Performance Summary")
            approach_perf_data = []; all_approach_keys = set(self.approach_alphas.keys()) | set(self.approach_scores.keys()) | set(self.explored_approaches.keys()); valid_approach_keys = [a for a in all_approach_keys if a not in ["unknown", "initial", "variant"]] # Exclude meta-types
            for app_key in sorted(valid_approach_keys):
                thought_count = len(self.explored_approaches.get(app_key, []))
                if thought_count == 0: continue # Skip approaches not actually used
                score_str, sort_key = "N/A", -1.0
                if cfg.get("use_bayesian_evaluation"): alpha = self.approach_alphas.get(app_key, 1.0); beta = self.approach_betas.get(app_key, 1.0); mean_score = (alpha / (alpha + beta) * 10.0) if (alpha + beta) > 1e-9 else -1; score_str = f"Bayesian Score: {mean_score:.2f} (α={alpha:.1f}, β={beta:.1f})" if mean_score >= 0 else "Score Error"; sort_key = mean_score
                else: avg_score = self.approach_scores.get(app_key); score_str = f"Average Score: {avg_score:.2f}" if avg_score is not None else "N/A"; sort_key = avg_score if avg_score is not None else -1.0
                approach_perf_data.append({"name": app_key, "score_info": score_str, "count_info": f"(Thoughts: {thought_count})", "sort_key": sort_key})
            sorted_perf_data = sorted(approach_perf_data, key=lambda x: x["sort_key"], reverse=True) # Sort by best score
            if not sorted_perf_data: output_lines.append("*No specific approaches tracked or used.*")
            else:
                for item in sorted_perf_data[:7]: output_lines.append(f"- **{item['name']}**: {item['score_info']} {item['count_info']}")
                if len(sorted_perf_data) > 7: output_lines.append(f"- *... ({len(sorted_perf_data) - 7} more)*")

            # Search Parameters Section
            output_lines.append(f"\n## Search Parameters Used")
            output_lines.append(f"- Iterations: {self.iterations_completed}/{cfg.get('max_iterations')}, Sims/Iter: {cfg.get('simulations_per_iteration')}, Total Sims: {self.simulations_completed}")
            eval_mode = "Bayesian (Beta)" if cfg.get("use_bayesian_evaluation") else "Traditional (Average)"; select_mode = ("Thompson Sampling" if cfg.get("use_thompson_sampling") else "UCT") if cfg.get("use_bayesian_evaluation") else "UCT"; output_lines.append(f"- Evaluation: {eval_mode}, Selection: {select_mode}, Exploration Weight: {cfg.get('exploration_weight'):.2f}")
            if cfg.get("use_bayesian_evaluation"): output_lines.append(f"- Bayesian Priors (Initial): α={cfg.get('beta_prior_alpha'):.2f}, β={cfg.get('beta_prior_beta'):.2f}")
            output_lines.append(f"- Early Stopping: {cfg.get('early_stopping')} (Threshold: {cfg.get('early_stopping_threshold'):.1f}, Stability: {cfg.get('early_stopping_stability')})")
            output_lines.append(f"- State Persistence: {cfg.get('enable_state_persistence')}, Verbose Output: {cfg.get('show_processing_details')}, Debug Logging: {cfg.get('debug_logging')}")

            # Debug Log Snippets (Optional)
            if self.debug_logging and self.debug_history:
                output_lines.append("\n## Recent Debug Log Snippets (Max 3)")
                for entry in self.debug_history[-3:]: cleaned_entry = re.sub(r"\n+", "\n", entry).strip(); output_lines.append(f"\n```\n{truncate_text(cleaned_entry, 250)}\n```") # Keep debug snippets truncated

            return "\n".join(output_lines).strip()
        except Exception as e: self.logger.error(f"Error formatting verbose output: {e}", exc_info=self.debug_logging); return ("\n".join(output_lines).strip() + f"\n\n**# ERROR generating verbose summary:**\n{type(e).__name__}: {str(e)}\n")


# ==============================================================================
# Main Pipe Class Definition
# ==============================================================================
class Pipe(LLMInterface):

    # Pydantic model for Valves
    class Valves(BaseModel):
        MAX_ITERATIONS: int = Field(default=DEFAULT_CONFIG["max_iterations"], title="Max MCTS Iterations", ge=1, le=100)
        SIMULATIONS_PER_ITERATION: int = Field(default=DEFAULT_CONFIG["simulations_per_iteration"], title="Simulations / Iteration", ge=1, le=50)
        MAX_CHILDREN: int = Field(default=DEFAULT_CONFIG["max_children"], title="Max Children / Node", ge=1, le=20)
        EXPLORATION_WEIGHT: float = Field(default=DEFAULT_CONFIG["exploration_weight"], title="Exploration Weight (UCT)", ge=0.0, le=10.0)
        USE_THOMPSON_SAMPLING: bool = Field(default=DEFAULT_CONFIG["use_thompson_sampling"], title="Use Thompson Sampling (if Bayesian)")
        FORCE_EXPLORATION_INTERVAL: int = Field(default=DEFAULT_CONFIG["force_exploration_interval"], title="Force Branch Explore Interval (0=off)", ge=0, le=20)
        SCORE_DIVERSITY_BONUS: float = Field(default=DEFAULT_CONFIG["score_diversity_bonus"], title="UCT Score Diversity Bonus Weight", ge=0.0, le=1.0)
        USE_BAYESIAN_EVALUATION: bool = Field(default=DEFAULT_CONFIG["use_bayesian_evaluation"], title="Use Bayesian (Beta) Evaluation")
        BETA_PRIOR_ALPHA: float = Field(default=DEFAULT_CONFIG["beta_prior_alpha"], title="Bayesian Prior Alpha (>0)", gt=0, le=100.0)
        BETA_PRIOR_BETA: float = Field(default=DEFAULT_CONFIG["beta_prior_beta"], title="Bayesian Prior Beta (>0)", gt=0, le=100.0)
        USE_SEMANTIC_DISTANCE: bool = Field(default=DEFAULT_CONFIG["use_semantic_distance"], title="Use Semantic Distance (Surprise)", json_schema_extra={"disabled": not SKLEARN_AVAILABLE})
        SURPRISE_THRESHOLD: float = Field(default=DEFAULT_CONFIG["surprise_threshold"], title="Surprise Threshold (Semantic)", ge=0.0, le=1.0)
        SURPRISE_SEMANTIC_WEIGHT: float = Field(default=DEFAULT_CONFIG["surprise_semantic_weight"], title="Surprise: Semantic Weight", ge=0.0, le=1.0)
        SURPRISE_PHILOSOPHICAL_SHIFT_WEIGHT: float = Field(default=DEFAULT_CONFIG["surprise_philosophical_shift_weight"], title="Surprise: Approach Shift Weight", ge=0.0, le=1.0)
        SURPRISE_NOVELTY_WEIGHT: float = Field(default=DEFAULT_CONFIG["surprise_novelty_weight"], title="Surprise: Approach Novelty Weight", ge=0.0, le=1.0)
        SURPRISE_OVERALL_THRESHOLD: float = Field(default=DEFAULT_CONFIG["surprise_overall_threshold"], title="Surprise: Overall Threshold", ge=0.0, le=1.0)
        GLOBAL_CONTEXT_IN_PROMPTS: bool = Field(default=DEFAULT_CONFIG["global_context_in_prompts"], title="Use Global Context in Prompts")
        TRACK_EXPLORED_APPROACHES: bool = Field(default=DEFAULT_CONFIG["track_explored_approaches"], title="Track Explored Thought Approaches")
        SIBLING_AWARENESS: bool = Field(default=DEFAULT_CONFIG["sibling_awareness"], title="Add Sibling Context to Prompts")
        MEMORY_CUTOFF: int = Field(default=DEFAULT_CONFIG["memory_cutoff"], title="Memory Cutoff (Top N High Scores)", ge=0, le=20)
        EARLY_STOPPING: bool = Field(default=DEFAULT_CONFIG["early_stopping"], title="Enable Early Stopping")
        EARLY_STOPPING_THRESHOLD: float = Field(default=DEFAULT_CONFIG["early_stopping_threshold"], title="Early Stopping Score Threshold", ge=1.0, le=10.0)
        EARLY_STOPPING_STABILITY: int = Field(default=DEFAULT_CONFIG["early_stopping_stability"], title="Early Stopping Stability Count", ge=1, le=10)
        ENABLE_STATE_PERSISTENCE: bool = Field(default=DEFAULT_CONFIG["enable_state_persistence"], title="Enable State Persistence (DB)")
        UNFIT_SCORE_THRESHOLD: float = Field(default=DEFAULT_CONFIG["unfit_score_threshold"], title="Unfit Marker Score Threshold", ge=0.0, le=10.0)
        UNFIT_VISIT_THRESHOLD: int = Field(default=DEFAULT_CONFIG["unfit_visit_threshold"], title="Unfit Marker Min Visits", ge=1, le=20)
        SHOW_PROCESSING_DETAILS: bool = Field(default=DEFAULT_CONFIG["show_processing_details"], title="Show Detailed MCTS Steps (Chat)")
        DEBUG_LOGGING: bool = Field(default=DEFAULT_CONFIG["debug_logging"], title="Enable Detailed Debug Logging")

    def __init__(self):
        """Initializes the Pipe instance."""
        self.type = "manifold"; self.name = PIPE_NAME; self.valves = self.Valves()
        self.current_config: Dict[str, Any] = {}; self.debug_logging: bool = False
        self.__current_event_emitter__: Optional[Callable] = None; self.__model__: str = ""
        self.__chat_id__: Optional[str] = None # Initialize chat_id attribute
        self.__request_body__: Dict[str, Any] = {}
        self.__user__: Optional[Union[Dict, AdminUserMock]] = None # Instance variable for user
        self.logger = logging.getLogger(PIPE_LOG_NAME)
        # Use SCRIPT_VERSION in init log
        self.logger.info(f"Pipe '{self.name}' initialized. Version: {SCRIPT_VERSION}")
        if not OPENWEBUI_IMPORTS_AVAILABLE: self.logger.error("OpenWebUI components failed init.")
        if not SKLEARN_AVAILABLE: self.logger.warning("scikit-learn not found init.")

    # --- Inlet Method (NEW) ---
    async def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """
        Preprocessing step called before the main pipe method.
        Captures chat_id and stores request body.
        Note: __user__ is also passed here but usually handled in pipe.
        """
        self.__request_body__ = body
        self.__chat_id__ = body.get("chat_id")
        temp_logger = getattr(self, 'logger', logging.getLogger(PIPE_LOG_NAME)) # Safe logger access
        if self.__chat_id__: temp_logger.debug(f"inlet: Stored chat_id: {self.__chat_id__}")
        else: temp_logger.debug("inlet: chat_id not found in body.")
        return body

    # --- Pipe Interface Methods ---
    def pipes(self) -> List[Dict[str, str]]:
        """Lists available models combined with this pipe's name."""
        # Use logger safely, might be called before full init sometimes
        list_logger = getattr(self, 'logger', logging.getLogger(PIPE_LOG_NAME))

        # Check for essential OpenWebUI components (app is needed for state)
        if not OPENWEBUI_IMPORTS_AVAILABLE or not app:
            list_logger.error("Cannot list pipes: OpenWebUI components missing (app or imports).")
            return [{"id": f"{self.name}-error-imports", "name": "(Error: OpenWebUI Missing)"}]

        try:
            # --- Rely *only* on app.state for listing models ---
            list_logger.debug("Attempting to list models using app.state.OLLAMA_MODELS...")
            models_in_state = getattr(app.state, "OLLAMA_MODELS", {})

            if not models_in_state or not isinstance(models_in_state, dict):
                list_logger.error("app.state.OLLAMA_MODELS missing or is not a dictionary.")
                return [{"id": f"{self.name}-error-state", "name": "(Error: No Models State)"}]

            # Process models found in state, ensuring they are valid entries
            valid_models = {}
            for model_key, model_info in models_in_state.items():
                # Check if model_info is a dict and has a 'name' field
                if isinstance(model_info, dict) and model_info.get("name"):
                    valid_models[model_key] = model_info
                else:
                    list_logger.warning(f"Ignoring invalid model entry in app.state: Key='{model_key}', Value='{model_info}'")

            if not valid_models:
                list_logger.warning("No valid models found in app.state.OLLAMA_MODELS.")
                return [{"id": f"{self.name}-no-valid-models", "name": "(No Valid Models)"}]

            # Format the valid models for the pipe list
            formatted_pipes = []
            for model_key, model_info in valid_models.items():
                 # <<< ADDED Check for empty model_key >>>
                 if not model_key:
                      list_logger.warning(f"Skipping model entry with empty key. Info: {model_info}")
                      continue

                 # Use the actual model name for display, fallback to key if needed
                 model_display_name = model_info.get("name", model_key)
                 # Construct the unique ID for the pipe+model combination
                 pipe_id = f"{self.name}-{model_key}"
                 # Construct the display name shown in the UI
                 formatted_pipes.append({"id": pipe_id, "name": f"{model_display_name}"})

            list_logger.info(f"Exposing {len(formatted_pipes)} models via pipe '{self.name}' using app.state.")
            return formatted_pipes

        except AttributeError as ae:
             # Catch error if app.state itself doesn't exist
             list_logger.error(f"AttributeError accessing app.state: {ae}", exc_info=True)
             return [{"id": f"{self.name}-error-state-attr", "name": "(Error: App State Access)"}]
        except Exception as e:
            # General catch-all for unexpected errors during model listing
            list_logger.error(f"Unexpected pipe list error: {e}", exc_info=True)
            return [{"id": f"{self.name}-list-error", "name": "(Error Listing Models)"}]

    # <<< REVISED resolve_model Function >>>
    def resolve_model(self, body: Optional[Dict[str, Any]] = None) -> str:
        """
        Resolves the base Ollama model name from the combined pipe ID
        (e.g., expected format 'pipe_name-model:tag').
        """
        body_to_use = body or self.__request_body__
        # Use .get() with default to prevent KeyError if 'model' is missing
        pipe_model_id = body_to_use.get("model", "").strip()

        self.logger.debug(f"Attempting to resolve base model. Received 'model' field value: '{pipe_model_id}'")

        if not pipe_model_id:
            self.logger.error("resolve_model failed: 'model' field was empty or missing in the request body.")
            # Adding body dump only if debug logging is enabled to avoid overly verbose logs
            if self.debug_logging:
                self.logger.debug(f"Request body content (at resolve_model): {body_to_use}")
            return ""

        # --- Standard Pipe Format Check ---
        # Expected format: f"{self.name}-{base_model_name}" (e.g., "advanced_mcts_stateful-cogito:latest")
        expected_prefix = f"{self.name}-"

        if pipe_model_id.startswith(expected_prefix):
            # Extract the part after the prefix
            base_model_name = pipe_model_id[len(expected_prefix):]

            if base_model_name:
                # Basic validation: Check if it looks like a valid model name (contains ':' or just letters/numbers)
                # This allows "model:tag" or just "model"
                if ":" in base_model_name or re.match(r"^[a-zA-Z0-9_./-]+$", base_model_name):
                     self.logger.info(f"Resolved base model '{base_model_name}' using standard prefix stripping from ID '{pipe_model_id}'.")
                     return base_model_name
                else:
                     # Found prefix, but the remainder looks odd. Log warning but proceed.
                     self.logger.warning(f"Stripped prefix '{expected_prefix}' from '{pipe_model_id}', but remainder '{base_model_name}' seems unusual. Proceeding with it cautiously.")
                     return base_model_name
            else:
                # Starts with prefix but nothing follows (e.g., "advanced_mcts_stateful-")
                self.logger.error(f"resolve_model failed: Pipe ID '{pipe_model_id}' consists only of the expected prefix '{expected_prefix}'. Invalid format.")
                return ""
        else:
            # --- Fallback / Direct Model Name / Incorrect Format Handling ---
            # The ID doesn't start with the expected pipe prefix.
            # This could mean it's a direct call with just the base model name,
            # OR it could be the mangled format seen in the UI screenshot ('pipenamemodel:tag').

            self.logger.warning(f"Pipe ID '{pipe_model_id}' does not start with the expected prefix '{expected_prefix}'. This might be a direct model call or an unexpected ID format.")

            # Check if the pipe_model_id *contains* the pipe name at the start, even without the hyphen
            # (Attempt to handle the observed 'pipenamemodel:tag' case)
            if pipe_model_id.startswith(self.name):
                potential_base_model = pipe_model_id[len(self.name):]
                if potential_base_model and (":" in potential_base_model or re.match(r"^[a-zA-Z0-9_./-]+$", potential_base_model)):
                     self.logger.warning(f"Attempting recovery: Assuming '{pipe_model_id}' is a mangled ID. Extracted potential base model: '{potential_base_model}'")
                     return potential_base_model

            # If it doesn't start with the prefix (with or without hyphen)
            # check if it looks like a plausible direct model name (e.g., contains ':').
            if ":" in pipe_model_id or re.match(r"^[a-zA-Z0-9_./-]+$", pipe_model_id):
                self.logger.warning(f"Assuming '{pipe_model_id}' is a direct base model name (or unable to parse otherwise).")
                return pipe_model_id
            else:
                # If it doesn't start with the prefix AND doesn't look like a model name, it's likely an error.
                self.logger.error(f"resolve_model failed: Pipe ID '{pipe_model_id}' does not match expected format ('{expected_prefix}...') and doesn't appear to be a direct model name or recoverable mangled ID.")
                return ""

    # <<< REVISED _resolve_question Function with Logging >>>
    def _resolve_question(self, body: Dict[str, Any]) -> str:
        """Extracts the last user message content from the request body."""
        messages = body.get("messages", [])

        # <<< ADDED Debug Logging >>>
        if self.debug_logging:
            # Show first 2 and last 2 messages for context if list is long enough
            msg_preview = f"{str(messages[:2])}"
            if len(messages) > 4:
                 msg_preview += f"...{str(messages[-2:])}"
            elif len(messages) > 2:
                 msg_preview += f"{str(messages[2:])}"
            self.logger.debug(f"_resolve_question: Received messages structure: {msg_preview}")


        if isinstance(messages, list):
            # Iterate backwards to find the most recent user message
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content")
                    extracted_content = str(content).strip() if content is not None else ""
                    # <<< ADDED Debug Logging >>>
                    if self.debug_logging:
                         self.logger.debug(f"_resolve_question: Found last user message. Extracted content (truncated): '{truncate_text(extracted_content, 100)}'")
                    return extracted_content
        # <<< ADDED Debug Logging >>>
        if self.debug_logging:
             self.logger.debug("_resolve_question: No user message found in messages list.")
        # Return empty string if no user message found
        return ""


    # --- Event Emitter Helpers ---
    async def progress(self, message: str):
        """Sends a progress/status update back to the UI if emitter is available."""
        if self.__current_event_emitter__:
            try:
                if self.debug_logging: self.logger.debug(f"Emit Progress: '{message}'")
                # Emit a status update event
                await self.__current_event_emitter__({"type": "status", "data": {"level": "info", "description": str(message), "done": False}})
            except Exception as e: self.logger.error(f"Failed to emit progress update: {e}", exc_info=self.debug_logging)
        elif self.debug_logging: self.logger.warning(f"Progress update skipped (no emitter): '{message}'")

    async def done(self):
        """Sends a final 'done' status update back to the UI."""
        if self.__current_event_emitter__:
            try:
                if self.debug_logging: self.logger.debug("Emit Done Status.")
                # Emit a final status update event indicating completion
                await self.__current_event_emitter__({"type": "status", "data": {"level": "info", "description": "Processing Complete.", "done": True}})
            except Exception as e: self.logger.error(f"Failed to emit done status: {e}", exc_info=self.debug_logging)
        elif self.debug_logging: self.logger.warning("Done status skipped (no emitter).")

    async def emit_message(self, message: str):
        """Sends a regular message chunk back to the UI chat."""
        if self.__current_event_emitter__:
            try:
                 # Emit a message chunk event
                 await self.__current_event_emitter__({"type": "message", "data": {"content": str(message)}})
            except Exception as e: self.logger.error(f"Failed to emit message ('{truncate_text(message, 50)}...'): {e}", exc_info=self.debug_logging)
        elif self.debug_logging: self.logger.warning(f"Emit message skipped (no emitter): '{truncate_text(message, 50)}...'")

    # --- LLMInterface Implementation ---
    async def _call_llm_base(self, payload: Dict[str, Any]) -> Tuple[Optional[Any], Optional[str]]:
        """Base helper to call the internal Ollama endpoint and handle basic errors."""
        response = None
        error_message = None
        try:
            # Pass the stored user object (__user__) to the endpoint call function
            response = await call_ollama_endpoint(payload, self.logger, self.__user__, self.debug_logging)

            # Check if the response indicates an error
            if isinstance(response, dict) and response.get("error"):
                error_msg = get_response_content(response, self.logger) # Extract error details
                self.logger.error(f"LLM call failed: {error_msg}. Payload Summary: {str({k:v for k,v in payload.items() if k != 'messages'})[:200]}...");
                error_message = error_msg # Store the error message
                response = None # Set response to None as it was an error

            # Return the response (or None if error) and the error message (or None)
            return response, error_message

        except Exception as e:
            # Catch any unexpected exceptions during the call
            self.logger.error(f"Unhandled exception in _call_llm_base: {e}", exc_info=self.debug_logging)
            # Format an error message to return
            error_message = f"Error: LLM call exception ({type(e).__name__})."
            return None, error_message # Return None response and the error message

    async def get_completion(self, model: str, messages: List[Dict[str, str]]) -> str:
        """Gets a non-streaming completion from the LLM."""
        response = None; error_message = None
        try:
            model_to_use = model or self.__model__; # Use provided model or the one stored for the pipe
            if not model_to_use: self.logger.error("LLM get_completion: No model name available."); return "Error: LLM model name missing."

            # Construct payload for non-streaming request
            payload = {"model": model_to_use, "messages": messages, "stream": False}
            response, error_message = await self._call_llm_base(payload)

            if error_message: return error_message # Return error if the call failed
            if response is None: self.logger.error("LLM get_completion: _call_llm_base returned None without error message."); return "Error: LLM communication failure (None response)."

            # Extract content from the successful response
            content = get_response_content(response, self.logger)
            if content.startswith("Error:"): self.logger.warning(f"get_response_content extracted an error message: {content}"); return content

            return content if content else "" # Return content or empty string if extraction failed silently

        except Exception as e:
            self.logger.error(f"Unexpected error in get_completion: {e}", exc_info=self.debug_logging)
            return f"Error: get_completion failed ({type(e).__name__})."
        finally:
            # Ensure response resources are released if applicable (though less common for non-streaming)
            if response and hasattr(response, "release") and callable(response.release):
                try:
                    if asyncio.iscoroutinefunction(response.release): await response.release()
                    else: response.release()
                except Exception as release_err: self.logger.error(f"Error releasing non-streaming response resources: {release_err}")

    async def get_streaming_completion(self, model: str, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """Gets a streaming completion from the LLM, yielding chunks."""
        response = None; error_message = None
        try:
            model_to_use = model or self.__model__
            if not model_to_use: self.logger.error("LLM get_streaming_completion: No model name available."); yield "Error: LLM model name missing."; return

            # Construct payload for streaming request
            payload = {"model": model_to_use, "messages": messages, "stream": True}
            response, error_message = await self._call_llm_base(payload)

            if error_message: yield error_message; return # Yield error if call failed
            if response is None: self.logger.error("LLM get_streaming_completion: _call_llm_base returned None without error message."); yield "Error: LLM communication failure (None response)."; return

            # Process the streaming response body
            if hasattr(response, "body_iterator"):
                async for chunk_bytes in response.body_iterator:
                    content_parts = get_chunk_content(chunk_bytes, self.logger, self.debug_logging);
                    for part in content_parts: yield part # Yield each valid content part
            # Handle case where stream=True was requested but a non-streaming dict was returned
            elif isinstance(response, dict):
                self.logger.warning("Expected streaming response, but received a dictionary. Attempting to extract content.")
                content = get_response_content(response, self.logger)
                yield (content if content and not content.startswith("Error:") else (content or "Error: Invalid dictionary received instead of stream."))
            else: # Unexpected response type
                self.logger.error(f"LLM streaming call returned unexpected type: {type(response)}."); yield f"Error: Unexpected LLM response type ({type(response).__name__})."

        except Exception as e:
            self.logger.error(f"Error during streaming completion: {e}", exc_info=self.debug_logging)
            yield f"Error: Streaming failed ({type(e).__name__})."
        finally:
            # Ensure streaming response resources are released
            if response and hasattr(response, "release") and callable(response.release):
                try:
                    if asyncio.iscoroutinefunction(response.release): await response.release()
                    else: response.release()
                except Exception as release_err: self.logger.error(f"Error releasing streaming response resources: {release_err}")

    async def generate_thought(self, current_analysis: str, context: Dict, config_dict: Dict) -> str:
        """Generates a critique or new direction ('thought') using the LLM."""
        try:
            # Ensure all necessary context keys are present, providing defaults
            required_keys = ["current_answer", "question_summary", "best_answer", "best_score", "previous_best_summary", "unfit_markers_summary", "learned_approach_summary", "explored_approaches", "sibling_approaches", "current_sequence", "current_tags"]
            formatted_context = {key: context.get(key, "N/A") for key in required_keys}
            # Explicitly set current_answer from context, don't rely on default N/A if it exists
            if "current_answer" in context: formatted_context["current_answer"] = context["current_answer"]
            else: formatted_context["current_answer"] = "N/A" # Fallback if somehow missing

            # Format the prompt
            prompt = GENERATE_THOUGHT_PROMPT.format(**formatted_context)
            # Call the LLM for a non-streaming completion
            return await self.get_completion(self.__model__, [{"role": "user", "content": prompt}])
        except KeyError as e: self.logger.error(f"Generate thought prompt formatting failed. Missing key: {e}", exc_info=self.debug_logging); return f"Error: Prompt formatting failed (key: {e})."
        except Exception as e: self.logger.error(f"Unexpected error during generate_thought: {e}", exc_info=self.debug_logging); return f"Error: Thought generation failed ({type(e).__name__})."

    async def update_approach(self, original_analysis: str, critique: str, context: Dict, config_dict: Dict) -> str:
        """Updates the analysis based on the provided critique/thought using the LLM."""
        # Prepare arguments for the prompt, copying context and adding specific inputs
        prompt_args = context.copy(); prompt_args["answer"] = original_analysis; prompt_args["improvements"] = critique.strip()
        # Ensure required keys exist, defaulting to "N/A"
        required_keys = ["question_summary", "best_answer", "best_score", "current_tags", "previous_best_summary", "unfit_markers_summary", "answer", "improvements"]
        for key in required_keys: prompt_args.setdefault(key, "N/A")
        try:
            # Format the prompt
            prompt = UPDATE_ANALYSIS_PROMPT.format(**prompt_args)
            # Get the updated analysis from the LLM
            llm_result = await self.get_completion(self.__model__, [{"role": "user", "content": prompt}])
            # If LLM returned an error, log it and return the original analysis unchanged
            if llm_result.startswith("Error:"): self.logger.error(f"Analysis update LLM call failed: {llm_result}. Returning original analysis."); return str(original_analysis)
            # Clean the result (remove markdown code blocks)
            cleaned_result = re.sub(r"^\s*```[\s\S]*?```\s*$", "", llm_result, flags=re.MULTILINE).strip()
            # Return the cleaned result, or the original if cleaning resulted in empty string
            return cleaned_result if cleaned_result else str(original_analysis)
        except KeyError as e: self.logger.error(f"Update analysis prompt formatting failed. Missing key: {e}", exc_info=self.debug_logging); return str(original_analysis) # Return original on error
        except Exception as e: self.logger.error(f"Unexpected error during update_approach: {e}", exc_info=self.debug_logging); return str(original_analysis) # Return original on error

    async def evaluate_answer(self, analysis_to_evaluate: str, context: Dict, config_dict: Dict) -> Union[int, str]:
        """Evaluates the quality of an analysis node using the LLM, expecting a score (1-10)."""
        # Prepare arguments for the prompt
        prompt_args = context.copy(); prompt_args["answer_to_evaluate"] = analysis_to_evaluate
        # Ensure required keys exist
        required_keys = ["question_summary", "best_answer", "best_score", "current_tags", "previous_best_summary", "unfit_markers_summary", "answer_to_evaluate"]
        for key in required_keys: prompt_args.setdefault(key, "N/A")
        try:
            # Format the prompt
            prompt = EVALUATE_ANALYSIS_PROMPT.format(**prompt_args)
            # Get the evaluation from the LLM
            result_str = await self.get_completion(self.__model__, [{"role": "user", "content": prompt}])
            # Handle LLM errors
            if result_str.startswith("Error:"): self.logger.warning(f"Evaluation LLM call failed: {result_str}."); return result_str # Return error string

            # Attempt to parse the score from the response
            cleaned_str = result_str.strip()
            # Try strict parsing first (only the number 1-10)
            strict_match = re.search(r"^\s*([1-9]|10)\s*$", cleaned_str)
            if strict_match:
                score = int(strict_match.group(1)); self.logger.debug(f"Parsed evaluation score (strict regex): {score}"); return score
            # Try relaxed parsing (find number 1-10 within the text)
            relaxed_match = re.search(r"\b([1-9]|10)\b", cleaned_str)
            if relaxed_match:
                score = int(relaxed_match.group(1)); self.logger.info(f"Parsed evaluation score (relaxed regex): {score} from response: '{cleaned_str}'"); return score

            # If parsing failed
            self.logger.warning(f"Could not parse evaluation score from LLM response: '{cleaned_str}'."); return "Error: Failed to parse score from LLM response."
        except KeyError as e: self.logger.error(f"Evaluate analysis prompt formatting failed. Missing key: {e}", exc_info=self.debug_logging); return f"Error: Prompt formatting failed (key: {e})."
        except Exception as e: self.logger.error(f"Unexpected error during evaluate_answer: {e}", exc_info=self.debug_logging); return f"Error: Evaluation failed ({type(e).__name__})."

    # --- Intent Handling ---
    async def _classify_intent(self, text: str) -> str:
        """Uses LLM to classify the user's intent, with heuristic fallback."""
        if not text: return "ANALYZE_NEW" # Default for empty input
        self.logger.debug(f"Classifying intent for input: '{truncate_text(text, 100)}...'"); default_intent = "ANALYZE_NEW"
        try:
            # Format prompt and call LLM
            prompt = INTENT_CLASSIFIER_PROMPT.format(raw_input_text=text)
            response = await self.get_completion(self.__model__, [{"role": "user", "content": prompt}])

            # Handle LLM error
            if response.startswith("Error:"): self.logger.error(f"Intent classification LLM call failed: {response}"); await self.emit_message("Warning: Intent classification failed. Assuming new analysis."); return default_intent

            # Process valid LLM response
            valid_intents = {"ANALYZE_NEW", "CONTINUE_ANALYSIS", "ASK_LAST_RUN_SUMMARY", "ASK_PROCESS", "ASK_CONFIG", "GENERAL_CONVERSATION"}
            cleaned_response = ""
            if response and isinstance(response, str):
                 # Extract the first word/potential intent category, clean it
                 parts = response.strip().upper().split(maxsplit=1)
                 if parts: cleaned_response = re.sub(r"[.,!?;:]$", "", parts) # Remove trailing punctuation from the first part only

            if cleaned_response in valid_intents:
                 self.logger.info(f"Intent classified via LLM: {cleaned_response}"); return cleaned_response
            else:
                 # Fallback to heuristics if LLM response is unexpected
                 self.logger.warning(f"LLM intent classification returned unexpected response: '{response}'. Using heuristic fallback."); text_lower = text.lower()
                 if any(kw in text_lower for kw in ["continue", "elaborate", "further", "more", "what about", "build on", "last time", "refine"]): self.logger.info("Heuristic fallback classification: CONTINUE_ANALYSIS"); return "CONTINUE_ANALYSIS"
                 elif any(kw in text_lower for kw in ["how", "explain", "process", "algorithm", "mcts", "work", "function"]): self.logger.info("Heuristic fallback classification: ASK_PROCESS"); return "ASK_PROCESS"
                 elif any(kw in text_lower for kw in ["config", "setting", "parameter", "valve", "option", "tune"]): self.logger.info("Heuristic fallback classification: ASK_CONFIG"); return "ASK_CONFIG"
                 elif any(kw in text_lower for kw in ["last run", "summary", "previous result", "score", "result", "what did you find", "did it work"]): self.logger.info("Heuristic fallback classification: ASK_LAST_RUN_SUMMARY"); return "ASK_LAST_RUN_SUMMARY"
                 # If no heuristics match, default
                 self.logger.info(f"Heuristic fallback failed. Defaulting intent to: {default_intent}"); return default_intent
        except Exception as e:
            # Handle unexpected errors during classification
            self.logger.error(f"Intent classification process encountered an unexpected error: {e}", exc_info=self.debug_logging); await self.emit_message("Warning: Error during intent classification. Assuming new analysis."); return default_intent

    async def _handle_ask_process(self):
        """Handles ASK_PROCESS intent by sending predefined explanation."""
        self.logger.info("Handling Intent: ASK_PROCESS");
        try:
            # Get database filename for the explanation template
            db_file_name = os.path.basename(DB_FILE) if DB_FILE else "N/A"
            explanation = ASK_PROCESS_EXPLANATION.format(db_file_name=db_file_name)
            # Use SCRIPT_VERSION in message
            await self.emit_message(f"**About My Process (Advanced MCTS v{SCRIPT_VERSION}):**\n{explanation}")
        except Exception as e: self.logger.error(f"Error handling process explanation request: {e}", exc_info=self.debug_logging); await self.emit_message("Error: There was an issue explaining my process.")
        finally: await self.done() # Ensure done status is sent

    async def _handle_ask_config(self):
        """Handles ASK_CONFIG intent by showing current MCTS parameters."""
        self.logger.info("Handling Intent: ASK_CONFIG");
        try:
            # Show the currently active configuration derived from Valves/Defaults
            config_to_show = self.current_config.copy()
            config_str = json.dumps(config_to_show, indent=2) # Pretty-print JSON
            await self.emit_message(f"**Current MCTS Configuration:**\n\n```json\n{config_str}\n```\n")
        except TypeError as e: # Handles cases where config might not be JSON serializable
            self.logger.error(f"Configuration serialization failed: {e}", exc_info=True); await self.emit_message("Error: Could not format the current configuration for display.")
        except Exception as e: self.logger.error(f"Error handling configuration request: {e}", exc_info=self.debug_logging); await self.emit_message("Error: There was an issue retrieving the current configuration.")
        finally: await self.done() # Ensure done status is sent

    async def _handle_ask_last_run_summary(self, state: Optional[Dict]):
        """Handles ASK_LAST_RUN_SUMMARY intent using loaded state."""
        self.logger.info("Handling Intent: ASK_LAST_RUN_SUMMARY")
        if not state or not isinstance(state, dict):
            await self.emit_message("Sorry, I don't have a saved summary for the last MCTS run in this chat session."); await self.done(); return

        lines = ["**Summary of Last Saved MCTS Run:**"]
        try:
            # Extract and format key info from the loaded state dictionary
            score = state.get("best_score"); lines.append(f"- **Best Score**: {score:.1f}/10" if isinstance(score, (int, float)) else f"- **Best Score**: {score or 'N/A'}")
            tags = state.get("best_node_tags"); lines.append(f"- **Best Node Tags**: {tags}" if isinstance(tags, list) and tags else "- **Best Node Tags**: N/A")
            summary = state.get("best_solution_summary"); lines.append(f"- **Best Solution Summary**: {summary or 'N/A'}")

            # Format approach preferences if available
            priors = state.get("approach_priors")
            if priors and isinstance(priors.get("alpha"), dict) and isinstance(priors.get("beta"), dict):
                means = {}; alphas = priors["alpha"]; betas = priors["beta"]
                for app, a_val in alphas.items():
                    b_val = betas.get(app, 1.0);
                    try: a_f, b_f = max(1e-9, float(a_val)), max(1e-9, float(b_val)); denominator = a_f + b_f; means[app] = ((a_f / denominator * 10.0) if denominator > 1e-9 else -1)
                    except (ValueError, TypeError): means[app] = -1 # Handle potential non-numeric values
                valid_means = {k: v for k, v in means.items() if v >= 0 and k not in ["initial", "variant", "unknown"]} # Filter out invalid/meta approaches
                top_approaches = sorted(valid_means.items(), key=lambda item: item, reverse=True)[:3] # Get top 3 by score
                prefs_str = (f"{', '.join([f'{a}({s:.1f})' for a, s in top_approaches])}" + ("..." if len(valid_means) > 3 else "")) if top_approaches else "None learned yet"; lines.append(f"- **Learned Approach Preference**: {prefs_str}")
            else: lines.append("- **Learned Approach Preference**: N/A (Bayesian priors not saved or mode not used)")

            # Summarize unfit markers
            unfit_markers = state.get("unfit_markers", [])
            if isinstance(unfit_markers, list) and unfit_markers:
                first_marker = unfit_markers # Show details of the first one as an example
                example_str = f"'{first_marker.get('summary','?')}' ({first_marker.get('reason','?')})" if isinstance(first_marker, dict) else "Details N/A"
                lines.append(f"- **Marked Unfit Areas**: {len(unfit_markers)} found (e.g., {example_str})")
            else: lines.append("- **Marked Unfit Areas**: None")

            # List top nodes from the saved state
            top_nodes = state.get("top_nodes", [])
            if isinstance(top_nodes, list) and top_nodes:
                lines.append("- **Top Nodes from Last Run**:")
                for i, node_state in enumerate(top_nodes):
                    if isinstance(node_state, dict): # Ensure node_state is a dict
                        seq = node_state.get("sequence", "?"); scr = node_state.get("score", "?"); scr_fmt = f"{scr:.1f}" if isinstance(scr, (int, float)) else scr; tgs = node_state.get("tags", []); sumry = node_state.get("content_summary", "?")
                        lines.append(f"  {i+1}. N{seq} (Score: {scr_fmt}, Tags: {tgs}): '{sumry}'")
                    else: lines.append(f"  {i+1}. Invalid node state data.")
            else: lines.append("- **Top Nodes from Last Run**: N/A")

            # Send the formatted summary
            await self.emit_message("\n".join(lines))
        except Exception as e: self.logger.error(f"Error formatting last run summary from state: {e}", exc_info=self.debug_logging); await self.emit_message("\n".join(lines) + "\n\nError: Issue encountered while formatting the summary.");
        finally: await self.done() # Ensure done status is sent

    async def _handle_general_conversation(self, user_input: str):
        """Handles GENERAL_CONVERSATION intent by providing a simple LLM response."""
        self.logger.info("Handling Intent: GENERAL_CONVERSATION");
        try:
            prompt = GENERAL_CONVERSATION_PROMPT.format(user_input=user_input)
            response = await self.get_completion(self.__model__, [{"role": "user", "content": prompt}])
            fallback = "That's interesting. How can I help you with an MCTS analysis today?"
            # Send LLM response or fallback if error occurred
            await self.emit_message(response if not response.startswith("Error:") else fallback)
        except Exception as e: self.logger.error(f"Error handling general conversation: {e}", exc_info=self.debug_logging); await self.emit_message("Sorry, I encountered an issue responding to that.")
        finally: await self.done() # Ensure done status is sent

    # --- Main Execution Logic ---
    async def _initialize_run(self, body: Dict) -> Tuple[bool, str, Optional[str]]:
        """Initializes pipe state, config, logging, and extracts basic info."""
        # NOTE: self.__chat_id__ and self.__request_body__ should be set by 'inlet' before this runs
        # Use SCRIPT_VERSION in log
        self.logger.info(f"--- Initializing Pipe Run: {self.name} v{SCRIPT_VERSION} ---")

        # 1. Reset Instance State Variables
        self.current_config = DEFAULT_CONFIG.copy(); self.__model__ = "";
        # User and Emitter are handled in the main pipe method

        # 2. Resolve Base Model Name
        self.__model__ = self.resolve_model(self.__request_body__) # Use stored body
        if not self.__model__:
            err_msg = "Initialization failed: Could not resolve a valid base model name from the request."; self.logger.error(err_msg); await self.emit_message(f"Error: {err_msg}"); await self.done(); return (False, "", None)
        self.logger.info(f"Using base model for LLM calls: {self.__model__}")

        # 3. Extract User Input
        user_input_text = self._resolve_question(self.__request_body__) # Use stored body
        # Check if input is empty, but allow if it's a title generation task
        is_title_task = OPENWEBUI_IMPORTS_AVAILABLE and TASKS and self.__request_body__.get("task") == TASKS.TITLE_GENERATION
        if not user_input_text and not is_title_task:
            err_msg = "Initialization failed: No user input text found in the messages."; self.logger.error(err_msg); await self.emit_message(f"Error: {err_msg}"); await self.done(); return False, "", None
        elif not user_input_text and is_title_task:
            # For title generation, the 'prompt' field might contain the text instead of 'messages'
             user_input_text = self.__request_body__.get("prompt", "")
             if not user_input_text:
                 err_msg = "Initialization failed: No user input text found for title generation task."; self.logger.error(err_msg); await self.emit_message(f"Error: {err_msg}"); await self.done(); return False, "", None
             self.logger.info("Proceeding with title generation task using text from 'prompt' field.")
        self.logger.debug(f"Received Input: '{truncate_text(user_input_text, 100)}...'")

        # 4. Log Chat ID (already set by inlet)
        if self.__chat_id__: self.logger.info(f"Running within chat session: {self.__chat_id__}.")
        else: self.logger.warning("No chat_id found in request (set via inlet). State persistence will be disabled for this run.")

        # 5. Apply Configuration from Valves
        self.logger.debug("Applying configuration from Valves...")
        try:
            request_valves = self.__request_body__.get("valves"); # Use stored body from inlet
            if request_valves and isinstance(request_valves, dict):
                # Validate and apply valves using the Pydantic model
                self.valves = self.Valves(**request_valves); validated_valve_dict = self.valves.model_dump()
                # Update current_config (lowercase keys) with validated valve values (uppercase keys)
                for key_upper, value in validated_valve_dict.items():
                    key_lower = key_upper.lower()
                    if key_lower in self.current_config: self.current_config[key_lower] = value
                    else: self.logger.warning(f"Valve key '{key_upper}' from request is not recognized in default config. Ignoring.")
                self.logger.info("Applied configuration overrides from request Valves.")
            else:
                # No valves in request or invalid format, use defaults
                self.valves = self.Valves(); default_valve_dict = self.valves.model_dump()
                for key_upper, value in default_valve_dict.items():
                     key_lower = key_upper.lower()
                     if key_lower in self.current_config: self.current_config[key_lower] = value
                self.logger.info("Using default configuration (no valid Valves in request).")

            # Ensure specific float priors are valid numbers >= 1e-9
            self.current_config["beta_prior_alpha"] = max(1e-9, float(self.current_config.get("beta_prior_alpha", 1.0)))
            self.current_config["beta_prior_beta"] = max(1e-9, float(self.current_config.get("beta_prior_beta", 1.0)))

            # Set logging level based on config
            self.debug_logging = self.current_config.get("debug_logging", False); new_log_level = logging.DEBUG if self.debug_logging else logging.INFO
            setup_logger(PIPE_LOG_NAME, new_log_level, LOG_FORMAT); setup_logger(MCTS_LOG_NAME, new_log_level, LOG_FORMAT)
            self.logger.info(f"Logging level set to: {'DEBUG' if self.debug_logging else 'INFO'}")

        except ValidationError as e:
            # Handle Pydantic validation errors for Valves
            self.logger.error(f"Configuration validation failed due to invalid Valve values: {e}. Using default configuration.", exc_info=True); await self.emit_message(f"Warning: Invalid configuration values provided. Using defaults. Details: {e}")
            # Reset to defaults on validation error
            self.valves = self.Valves(); default_valve_dict = self.valves.model_dump()
            for key_upper, value in default_valve_dict.items():
                 key_lower = key_upper.lower()
                 if key_lower in self.current_config: self.current_config[key_lower] = value
            self.debug_logging = self.current_config.get("debug_logging", False); new_log_level = logging.DEBUG if self.debug_logging else logging.INFO
            setup_logger(PIPE_LOG_NAME, new_log_level, LOG_FORMAT); setup_logger(MCTS_LOG_NAME, new_log_level, LOG_FORMAT)
        except Exception as e:
            # Handle other unexpected errors during config application
            self.logger.error(f"Unexpected error applying configuration from Valves: {e}. Using default configuration.", exc_info=True); await self.emit_message(f"Warning: Error applying configuration ({type(e).__name__}). Using defaults.")
            # Reset to defaults on other errors
            self.valves = self.Valves(); default_valve_dict = self.valves.model_dump()
            for key_upper, value in default_valve_dict.items():
                 key_lower = key_upper.lower()
                 if key_lower in self.current_config: self.current_config[key_lower] = value
            self.debug_logging = self.current_config.get("debug_logging", False); new_log_level = logging.DEBUG if self.debug_logging else logging.INFO
            setup_logger(PIPE_LOG_NAME, new_log_level, LOG_FORMAT); setup_logger(MCTS_LOG_NAME, new_log_level, LOG_FORMAT)

        self.logger.info("Pipe initialization complete.")
        # Return success status, the extracted user input, and the chat ID
        return True, user_input_text, self.__chat_id__

    async def _determine_intent_and_load_state(self, user_input_text: str) -> Tuple[str, Optional[Dict], bool]:
        """Determines user intent and loads MCTS state if applicable."""
        # 1. Classify Intent
        intent = await self._classify_intent(user_input_text); self.logger.info(f"Determined Intent: {intent}")

        # 2. Check if State Persistence is Enabled and Possible
        state_persistence_enabled_by_config = self.current_config.get("enable_state_persistence", True)
        state_is_enabled_for_run: bool = False; loaded_state: Optional[Dict] = None

        if not state_persistence_enabled_by_config:
             self.logger.info("State persistence is disabled by configuration (ENABLE_STATE_PERSISTENCE=False).")
        elif not self.__chat_id__:
             self.logger.info("State persistence cannot be used for this run: chat_id was not provided in the request.")
             # Provide feedback to user only if they *intended* to use state persistence
             if state_persistence_enabled_by_config:
                 await self.emit_message("*(Info: State persistence is enabled in config, but no chat ID was received from the UI. State cannot be saved or loaded for this message.)*")
        else:
             # State persistence is enabled by config AND we have a chat_id
             state_is_enabled_for_run = True; self.logger.info(f"State persistence is active for this run (Chat ID: {self.__chat_id__}).")

        # 3. Load State if Needed (Continue or Ask Summary) and Possible
        if state_is_enabled_for_run and self.__chat_id__ and intent in ["CONTINUE_ANALYSIS", "ASK_LAST_RUN_SUMMARY"]:
            self.logger.info(f"Attempting to load previous MCTS state for chat '{self.__chat_id__}' (Intent: {intent})")
            try:
                loaded_state = load_mcts_state(DB_FILE, self.__chat_id__, self.logger)
                if loaded_state:
                    # Version Check
                    state_version = loaded_state.get("version", "?.?.?")
                    # Use SCRIPT_VERSION for comparison
                    required_version = SCRIPT_VERSION
                    if state_version != required_version:
                         self.logger.warning(f"Loaded state version '{state_version}' does not match required version '{required_version}'. Discarding loaded state."); await self.emit_message(f"Warning: Found incompatible saved state (version {state_version}, need {required_version}). Starting a fresh analysis."); loaded_state = None
                         # If trying to continue with incompatible state, switch intent to new
                         if intent == "CONTINUE_ANALYSIS": intent = "ANALYZE_NEW"; self.logger.info("Switching intent to ANALYZE_NEW due to incompatible state version.")
                    else:
                         self.logger.info(f"Successfully loaded compatible state (version {state_version}).")
                elif intent == "CONTINUE_ANALYSIS":
                     # Tried to continue, but no state was found in DB
                     self.logger.info("Intent is CONTINUE_ANALYSIS, but no previous state found in database. Switching to ANALYZE_NEW."); await self.emit_message("Info: No previous analysis state found for this chat. Starting a new analysis."); intent = "ANALYZE_NEW"
            except Exception as e:
                 # Handle errors during state loading
                 self.logger.error(f"Failed to load MCTS state for chat '{self.__chat_id__}': {e}", exc_info=True); await self.emit_message("Warning: An error occurred while loading previous analysis state. Starting a fresh analysis."); loaded_state = None
                 # If trying to continue failed due to error, switch intent to new
                 if intent == "CONTINUE_ANALYSIS": intent = "ANALYZE_NEW"; self.logger.info("Switching intent to ANALYZE_NEW due to state loading error.")

        # Final consistency check: If intent is still CONTINUE but state is None, switch to ANALYZE_NEW
        if intent == "CONTINUE_ANALYSIS" and not loaded_state:
            self.logger.info("Consistency Check: Intent is CONTINUE_ANALYSIS, but no valid state loaded. Final intent set to ANALYZE_NEW.")
            intent = "ANALYZE_NEW"
            # No need to message user again here, previous messages covered why

        return intent, loaded_state, state_is_enabled_for_run

    async def _handle_intent(self, intent: str, user_input: str, loaded_state: Optional[Dict]) -> bool:
        """Routes execution based on intent. Returns True if intent was handled directly (no MCTS needed)."""
        self.logger.debug(f"Routing based on determined intent: {intent}")
        # Define handlers for intents that don't require MCTS
        intent_handlers = {
            "ASK_PROCESS": self._handle_ask_process,
            "ASK_CONFIG": self._handle_ask_config,
             # Use lambda to pass state to the summary handler
            "ASK_LAST_RUN_SUMMARY": lambda: self._handle_ask_last_run_summary(loaded_state),
             # Use lambda to pass input to the conversation handler
            "GENERAL_CONVERSATION": lambda: self._handle_general_conversation(user_input)
        }

        if intent in intent_handlers:
            handler_func = intent_handlers[intent]
            await handler_func() # Execute the handler
            self.logger.info(f"Intent '{intent}' was handled directly without MCTS run.")
            return True # Signal that the intent was fully handled
        elif intent in ["ANALYZE_NEW", "CONTINUE_ANALYSIS"]:
            # These intents require running the MCTS process
            self.logger.info(f"Intent '{intent}' requires MCTS analysis run.")
            return False # Signal that MCTS should proceed
        else:
            # Fallback for unknown/unhandled intents
            self.logger.error(f"Unhandled intent encountered: {intent}"); await self.emit_message(f"Error: Internal error - cannot handle the determined intent '{intent}'."); await self.done();
            return True # Signal handled (by erroring out) to prevent MCTS

    async def _run_mcts_analysis(self, intent: str, user_input: str, loaded_state: Optional[Dict]) -> Optional[MCTS]:
        """Coordinates and executes the main MCTS analysis process."""
        run_type_msg = "Continuing previous analysis" if intent == "CONTINUE_ANALYSIS" and loaded_state else "Starting new analysis"
        # Initial user feedback message using SCRIPT_VERSION
        await self.emit_message(f'# {self.name} v{SCRIPT_VERSION}\n*Analyzing:* "{truncate_text(user_input, 100)}" *using* `{self.__model__}`.\n🚀 **{run_type_msg}...**')
        if self.current_config.get("show_processing_details"): await self.emit_message("*(Verbose processing details enabled. Expect frequent updates...)*\n")

        # Log start parameters
        log_params = {k: v for k, v in self.current_config.items() if k in ["max_iterations", "simulations_per_iteration", "use_bayesian_evaluation", "early_stopping", "enable_state_persistence"]}; self.logger.info(f"--- Starting MCTS Run --- Intent: {intent}, State Loaded: {bool(loaded_state)}, Chat ID: {self.__chat_id__ or 'N/A'}, Config Summary: {json.dumps(log_params)}")

        initial_analysis_text = ""
        try:
            # 1. Generate Initial Analysis (always needed for the root node)
            await self.progress("Generating initial analysis..."); initial_prompt = INITIAL_ANALYSIS_PROMPT.format(question=user_input)
            initial_response = await self.get_completion(self.__model__, [{"role": "user", "content": initial_prompt}])
            if initial_response.startswith("Error:"): self.logger.error(f"Failed to generate initial analysis: {initial_response}"); await self.emit_message(f"Error: Could not start MCTS because the initial analysis generation failed.\nDetails: {initial_response}"); await self.done(); return None
            # Clean initial analysis text
            initial_analysis_text = re.sub(r"^\s*```[\s\S]*?```\s*$", "", initial_response, flags=re.MULTILINE).strip()
            if not initial_analysis_text: self.logger.error("Initial analysis generation resulted in empty text."); await self.emit_message("Error: The initial analysis generated by the LLM was empty. Cannot proceed."); await self.done(); return None

            # Show initial analysis to user before starting iterations
            await self.emit_message(f"\n## Initial Analysis\n```text\n{initial_analysis_text}\n```\n\n{'-'*30}\n*Starting MCTS iterations...*\n"); await asyncio.sleep(0.1) # Small delay for UI

        except Exception as e: self.logger.error(f"Error during initial analysis generation: {e}", exc_info=self.debug_logging); await self.emit_message(f"Error generating initial analysis: {type(e).__name__}. Cannot start MCTS."); await self.done(); return None

        # 2. Initialize MCTS Instance
        mcts_instance: Optional[MCTS] = None
        try:
            await self.progress("Initializing MCTS engine..."); state_to_pass = loaded_state if intent == "CONTINUE_ANALYSIS" and loaded_state else None
            mcts_instance = MCTS(llm_interface=self, question=user_input, mcts_config=self.current_config, initial_analysis_content=initial_analysis_text, initial_state=state_to_pass, model_body=self.__request_body__)
            if not mcts_instance.root: raise RuntimeError("MCTS initialization failed - Root node is None after creation.") # Should not happen with checks in MCTS.__init__
        except Exception as e:
            self.logger.critical(f"MCTS engine initialization critically failed: {e}", exc_info=self.debug_logging); await self.emit_message(f"**FATAL ERROR:** Failed to initialize the MCTS engine.\nDetails: {type(e).__name__}: {e}"); await self.done();
            if mcts_instance: del mcts_instance; # Try cleanup
            return None # Cannot proceed

        # 3. Run MCTS Search Iterations
        should_continue_search = True; iterations_run = 0
        max_iterations = self.current_config.get("max_iterations", DEFAULT_CONFIG["max_iterations"]); sims_per_iteration = self.current_config.get("simulations_per_iteration", DEFAULT_CONFIG["simulations_per_iteration"])
        for iteration in range(max_iterations):
            if not should_continue_search: self.logger.info(f"Stopping MCTS search early before iteration {iteration + 1}."); break
            iterations_run = iteration + 1; await self.progress(f"Running MCTS Iteration {iterations_run}/{max_iterations}...")
            # Store score before iteration for potential delta reporting (optional)
            # score_before_iter = mcts_instance.best_score
            try:
                # Perform one iteration (select, expand, simulate, backpropagate * sims_per_iteration)
                should_continue_search = await mcts_instance.search(sims_per_iteration)
            except Exception as e:
                 self.logger.error(f"Error during MCTS Search Iteration {iterations_run}: {e}", exc_info=self.debug_logging); await self.emit_message(f"Warning: An error occurred during MCTS iteration {iterations_run}. Stopping search."); should_continue_search = False # Stop search on error
            # Optional: Report score change after iteration (can be noisy)
            # score_after_iter = mcts_instance.best_score
            # if score_after_iter > score_before_iter: await self.emit_message(f"*Iter {iterations_run} finished. Best score improved to {score_after_iter:.1f}*")

        # 4. Post-Search Processing
        mcts_instance.iterations_completed = iterations_run # Store actual iterations completed
        self.logger.info(f"MCTS search loop finished. Iterations Run: {mcts_instance.iterations_completed}, Total Sims: {mcts_instance.simulations_completed}, Final Best Score: {mcts_instance.best_score:.1f}")
        await self.emit_message("\n🏁 **MCTS search complete.** Generating final output...")
        return mcts_instance # Return the completed MCTS instance

    async def _finalize_run(self, mcts_instance: Optional[MCTS], initial_analysis_text: str, state_is_enabled: bool):
        """Generates final output, synthesis, and saves state if applicable."""
        self.logger.debug(f"Finalizing MCTS run. MCTS instance present: {bool(mcts_instance)}. State saving enabled: {state_is_enabled}. Emitter present: {bool(self.__current_event_emitter__)}")

        if not mcts_instance:
             # This should ideally not be reached if _run_mcts_analysis returns None correctly
             self.logger.error("Finalize called, but MCTS instance is missing. Cannot generate output."); await self.emit_message("Error: MCTS analysis process was incomplete. Cannot provide results."); await self.done(); return

        try:
            # 1. Determine Final Analysis Text (Best solution or fallback)
            final_analysis_text = initial_analysis_text # Default to initial
            if mcts_instance.best_solution:
                # Clean markdown from the best solution found
                cleaned_best = re.sub(r"^\s*```[\s\S]*?```\s*$", "", str(mcts_instance.best_solution), flags=re.MULTILINE).strip();
                if cleaned_best: final_analysis_text = cleaned_best
                else: self.logger.warning("MCTS best_solution content was empty after cleaning. Using initial analysis as fallback.")
            else: self.logger.warning("MCTS best_solution attribute was empty. Using initial analysis as fallback.")

            # 2. Generate and Emit Main Summary (Verbose or Simple)
            self.logger.debug(f"Generating final summary. Verbose mode: {self.current_config.get('show_processing_details')}")
            if self.current_config.get("show_processing_details"):
                await self.progress("Generating verbose summary..."); self.logger.debug("Calling mcts_instance.formatted_output()...")
                verbose_summary = mcts_instance.formatted_output() # Get verbose summary from MCTS
                self.logger.debug(f"Emitting verbose summary (length: {len(verbose_summary)})...")
                await self.emit_message(verbose_summary) # Emit the verbose summary
                self.logger.debug("Finished emitting verbose summary.")
            else:
                # Generate simple summary
                await self.progress("Extracting best analysis summary..."); best_node_final = mcts_instance.find_best_final_node(); final_tags = best_node_final.descriptive_tags if best_node_final else []
                simple_summary = f"## Best Analysis Found (Score: {mcts_instance.best_score:.1f}/10)\n**Tags: {final_tags}**\n\n```text\n{final_analysis_text}\n```\n"; self.logger.debug(f"Emitting simple summary (length: {len(simple_summary)})...")
                await self.emit_message(simple_summary)
                self.logger.debug("Finished emitting simple summary.")

            # 3. Generate and Emit Final Synthesis
            await self.progress("Generating final synthesis..."); self.logger.debug("Getting synthesis context...")
            synthesis_context = mcts_instance.get_final_synthesis_context() # Gets full content context
            if synthesis_context:
                try:
                    synthesis_prompt = FINAL_SYNTHESIS_PROMPT.format(**synthesis_context); self.logger.debug("Calling LLM for final synthesis...")
                    synthesis_response = await self.get_completion(self.__model__, [{"role": "user", "content": synthesis_prompt}])
                    if synthesis_response.startswith("Error:"): self.logger.error(f"Final synthesis LLM call failed: {synthesis_response}"); await self.emit_message("\n***\n## Final Synthesis\nWarning: Could not generate final synthesis (LLM error).")
                    else:
                         # Clean and emit synthesis
                         cleaned_synthesis = re.sub(r"^\s*```[\s\S]*?```\s*$", "", synthesis_response, flags=re.MULTILINE).strip(); self.logger.debug(f"Emitting final synthesis (length: {len(cleaned_synthesis)})..."); await self.emit_message(f"\n***\n## Final Synthesis\n{cleaned_synthesis or '(Synthesis generation resulted in empty text.)'}") ; self.logger.debug("Finished emitting final synthesis.")
                except KeyError as e: self.logger.error(f"Synthesis prompt formatting failed. Missing key: {e}", exc_info=self.debug_logging); await self.emit_message("\n***\n## Final Synthesis\nError: Internal error formatting synthesis prompt.")
                except Exception as e: self.logger.error(f"Unexpected error during final synthesis generation: {e}", exc_info=self.debug_logging); await self.emit_message("\n***\n## Final Synthesis\nError: Unexpected error generating synthesis.")
            else: self.logger.error("Could not get context for final synthesis."); await self.emit_message("\n***\n## Final Synthesis\nError: Internal error preparing context for synthesis.")

            # 4. Save State if Enabled and Possible
            self.logger.debug(f"Checking state saving conditions. Enabled: {state_is_enabled}, ChatID: {self.__chat_id__}")
            if state_is_enabled and self.__chat_id__:
                await self.progress("Saving final MCTS state...")
                try:
                    state_to_save = mcts_instance.get_state_for_persistence()
                    if state_to_save and isinstance(state_to_save, dict):
                         save_mcts_state(DB_FILE, self.__chat_id__, state_to_save, self.logger) # Call the save function
                    elif state_to_save is None: self.logger.error("State generation for persistence failed. State not saved.")
                    # else: # state_to_save might be empty dict, which is valid but maybe not useful
                    #    self.logger.warning("Generated state for persistence was empty. State not saved.")
                except Exception as e: self.logger.error(f"Failed to save final MCTS state for chat '{self.__chat_id__}': {e}", exc_info=True); await self.emit_message("Warning: Failed to save the analysis state to the database.")
            else: self.logger.info("State persistence is disabled or chat_id is missing. Skipping state saving.")

            # 5. Signal Completion
            self.logger.debug("Finalizing run by emitting done status.")
            await self.done(); self.logger.info(f"Pipe '{self.name}' successfully finalized run for chat '{self.__chat_id__ or 'N/A'}'.")

        except Exception as e:
             # Catch critical errors during the finalization process itself
             self.logger.critical(f"Critical error during run finalization: {e}", exc_info=True)
             try:
                 # Try to inform the user about the finalization error
                 if self.__current_event_emitter__: await self.emit_message(f"\n\n**CRITICAL ERROR:** An error occurred while finalizing the results.\nDetails: {type(e).__name__}. Please check server logs."); await self.done()
             except Exception as emit_err: self.logger.error(f"Failed to emit critical finalization error message: {emit_err}")

    # --- Main Pipe Entry Point ---
    async def pipe(
        self,
        body: Dict,
        # Use double underscores for framework-injected parameters
        __user__: Optional[Dict] = None,
        __event_emitter__: Optional[Callable] = None,
        __task__: Optional[str] = None,
    ) -> Union[str, None, AsyncGenerator[str, None]]:
        """Main entry point for the pipe, using framework-expected parameter names."""

        # --- Setup: Store context safely using double-underscore params ---
        # Use logger safely during setup
        init_logger = getattr(self, 'logger', logging.getLogger(PIPE_LOG_NAME))
        if __event_emitter__:
            self.__current_event_emitter__ = __event_emitter__
            init_logger.debug("Emitter received and stored.")
        else:
            init_logger.warning("Pipe started without __event_emitter__. Output disabled.")
            self.__current_event_emitter__ = None

        # Assign user, defaulting to AdminUserMock if None is provided by framework
        self.__user__ = __user__ if __user__ is not None else ADMIN_USER
        if __user__ is None:
             init_logger.warning("Pipe started without __user__ object. Using mock admin user.")
        else:
            # Log the type of user object received for debugging framework integration
             init_logger.debug(f"Pipe started with __user__ object of type: {type(__user__)}")


        # --- Implicit Inlet Call by Framework ---
        # The 'body' passed here should have been processed by 'inlet',
        # which should have set self.__chat_id__ and self.__request_body__

        # --- Variable Initialization ---
        mcts_run_instance: Optional[MCTS] = None; initial_analysis_text: str = ""; state_enabled_for_run: bool = False
        try:
            # === Step 1: Initialize ===
            # Initialize config, model name, extract input, etc.
            init_success, user_input, chat_id_from_init = await self._initialize_run(body) # Uses self.__chat_id__ set by inlet
            if not init_success:
                # Errors already logged and emitted by _initialize_run
                return None # Stop execution

            # === Step 2: Determine Intent & Load State ===
            # Classify intent and load state if needed and possible
            intent, loaded_state, state_enabled_for_run = await self._determine_intent_and_load_state(user_input) # Uses self.__chat_id__

            # === Step 3: Handle Non-Analysis Intents ===
            # If intent is ASK_*, GENERAL_*, handle it directly and exit
            intent_handled_directly = await self._handle_intent(intent, user_input, loaded_state)
            if intent_handled_directly:
                # Output/done status handled within _handle_intent methods
                return None # Stop execution

            # === Step 4: Handle Specific Tasks (e.g., Title Generation) ===
            # Check __task__ provided by the framework
            is_title_task = OPENWEBUI_IMPORTS_AVAILABLE and TASKS and __task__ == TASKS.TITLE_GENERATION
            if is_title_task:
                self.logger.info(f"Handling specific task: TITLE_GENERATION")
                # Generate title without full MCTS
                title_prompt = f"Generate a concise title (maximum 10 words, ideally fewer) for the following text: {user_input}"
                title_response = await self.get_completion(self.__model__, [{"role": "user", "content": title_prompt}])
                # Clean up potential quotes or markdown
                cleaned_title = title_response.strip().strip('"\'`').strip()
                final_title = truncate_text(cleaned_title, 70) if not title_response.startswith("Error:") else "Generated Title Error"
                self.logger.info(f"Generated Title: '{final_title}'")
                # Title generation usually expects a direct string return, not emitted messages
                await self.done() # Signal completion via status emitter
                return final_title # Return the generated title string

            # === Step 5: Run MCTS Analysis (Only if not handled above) ===
            # If intent is ANALYZE_NEW or CONTINUE_ANALYSIS
            mcts_run_instance = await self._run_mcts_analysis(intent, user_input, loaded_state)

            # Extract initial analysis text *if* MCTS ran successfully and has a root
            # This is needed for fallback in finalize if best_solution is empty
            if mcts_run_instance and mcts_run_instance.root:
                 initial_analysis_text = mcts_run_instance.root.content

            # If MCTS failed to start/run (_run_mcts_analysis returned None), errors were already emitted.
            # Prevent proceeding to finalize if mcts_run_instance is None.
            if not mcts_run_instance and not intent_handled_directly and not is_title_task:
                 self.logger.error("MCTS analysis did not run successfully (_run_mcts_analysis returned None). Skipping finalization.")
                 # Ensure done is called if MCTS failed to start, unless it was handled directly
                 if not intent_handled_directly: await self.done()
                 return None

            # === Step 6: Finalize Run (Only if MCTS ran successfully) ===
            if mcts_run_instance:
                 await self._finalize_run(mcts_run_instance, initial_analysis_text, state_enabled_for_run)
            else:
                 # This case should ideally be caught above, but as a safeguard:
                 self.logger.warning("Finalization skipped as MCTS instance was not available (might be OK if intent handled directly or was title task).")
                 # Ensure 'done' is called if not already handled
                 if not intent_handled_directly and not is_title_task:
                      await self.done()

            # Manifold pipes typically control output via the emitter and return None
            return None

        # --- Error Handling & Cleanup ---
        except Exception as e:
            final_logger = getattr(self, 'logger', logging.getLogger(PIPE_LOG_NAME))
            final_logger.critical(f"FATAL UNHANDLED ERROR in pipe '{self.name}': {e}", exc_info=True)
            error_message = f"\n\n**FATAL PIPE ERROR:**\n```\n{type(e).__name__}: {e}\n```\nCheck server logs."
            try:
                # Try to emit the fatal error message
                if self.__current_event_emitter__:
                    await self.emit_message(error_message)
                    await self.done() # Signal completion even on error
            except Exception as emit_err:
                final_logger.error(f"Failed to emit fatal error message: {emit_err}")
            # Return an error string (framework might ignore it, but good practice)
            return f"Error: Pipe execution failed critically ({type(e).__name__})."
        finally:
            # --- Cleanup ---
            final_logger = getattr(self, 'logger', logging.getLogger(PIPE_LOG_NAME))
            if hasattr(self, 'debug_logging') and self.debug_logging:
                 final_logger.debug(f"Pipe '{self.name}' cleaning up request state.")
            # Reset stateful variables to prepare for the next request
            self.__current_event_emitter__ = None
            self.__user__ = None
            self.__request_body__ = {}
            self.__model__ = ""
            self.__chat_id__ = None
            self.current_config = {} # Reset config back to default for next run
            # Attempt to explicitly delete the potentially large MCTS instance
            if 'mcts_run_instance' in locals() and mcts_run_instance:
                del mcts_run_instance
                final_logger.debug("Deleted MCTS instance reference.")
            # Trigger garbage collection to free up memory
            collected_count = gc.collect()
            final_logger.debug(f"Pipe '{self.name}' request cleanup finished. GC collected {collected_count} objects.")

# ==============================================================================
# END OF SCRIPT
# ==============================================================================
