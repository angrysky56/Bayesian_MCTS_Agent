# -*- coding: utf-8 -*-
"""
title: advanced_mcts_stateful (Single File)
version: 0.8.1

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
  Stateful Advanced Bayesian MCTS v0.8.1 (Single File Version):
  - Combines all logic into one script for Open WebUI pipe compatibility.
  - Integrates SQLite database for state persistence across user turns within a chat session.
  - Uses LLM-based intent classification to handle different user requests.
  - Implements Selective State Persistence: Saves/Loads learned approach priors, best results, and basic "unfit" markers.
  - Injects loaded state context into MCTS prompts to guide exploration.
  - Maintains core MCTS logic, quiet/verbose modes, and configuration via Valves.
  - Refactored internally for better readability while remaining a single file.

Requires:
- User to configure DB_FILE path correctly.
- Standard Python libs (sqlite3, json, datetime, re, logging, asyncio, math, random).
- Optional: scikit-learn for improved semantic distance calculation.
- Running within Open WebUI environment for pipe functionality.
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
from typing import (Any, AsyncGenerator, Awaitable, Callable, Dict, Generator,
                    List, Optional, Set, Tuple, Union)

import numpy as np  # Used for stats, random sampling
from numpy.random import beta as beta_sample
from pydantic import BaseModel, Field, field_validator
from scipy import stats  # Used potentially for future analysis? (Not currently used)

# --- Optional Dependency: scikit-learn ---
try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
    # Add common MCTS/analysis terms to stop words for better semantic distance
    CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS) + [
        "analysis", "however", "therefore", "furthermore", "perspective", "node",
        "mcts", "score", "approach", "concept", "system", "model", "text", "data"
    ]
    logger = logging.getLogger(__name__)
    logger.info("scikit-learn found. TF-IDF semantic distance enabled.")
except ImportError:
    SKLEARN_AVAILABLE = False
    TfidfVectorizer, cosine_similarity, CUSTOM_STOP_WORDS = None, None, None
    logger = logging.getLogger(__name__)
    logger.warning("scikit-learn not found. Using Jaccard similarity for semantic distance.")


# --- Open WebUI Specific Imports (Assumed available in the environment) ---
from fastapi import Request, Response
import open_webui.routers.ollama as ollama
from open_webui.constants import TASKS # For title generation task check
from open_webui.main import app


# ==============================================================================
# Configuration Constants (Formerly config.py)
# ==============================================================================
PIPE_NAME = "advanced_mcts_stateful" # Name for identification and logging

# --- Database Configuration ---
# !!! IMPORTANT: Set this path to a writable location for the backend process !!!
# ----->>>>>>>> CHANGE THIS PATH <<<<<<<<<<-----
DB_FILE = "/home/ty/Repositories/sqlite-db/mcts_state.db" # !!! REPLACE !!! with your own path.
# ----->>>>>>>> CHANGE THIS PATH <<<<<<<<<<-----


# --- Default MCTS Configuration ---
DEFAULT_CONFIG = {
    # Core MCTS
    "max_iterations": 5,
    "simulations_per_iteration": 5,
    "max_children": 10,
    "exploration_weight": 3.0,
    "use_thompson_sampling": True,
    "force_exploration_interval": 4, # 0=off
    "score_diversity_bonus": 0.7, # UCT diversity bonus based on sibling scores
    # Evaluation
    "use_bayesian_evaluation": True,
    "beta_prior_alpha": 1.0,
    "beta_prior_beta": 1.0,
    "relative_evaluation": False, # Note: Relative eval not fully implemented
    "unfit_score_threshold": 4.0, # Score below which nodes might be marked unfit (if stateful)
    "unfit_visit_threshold": 3,   # Min visits before marking unfit by score (if stateful)
    # Surprise Mechanism
    "use_semantic_distance": True, # Use TF-IDF (or future embeddings) for surprise
    "surprise_threshold": 0.66, # Semantic distance threshold for surprise
    "surprise_semantic_weight": 0.6,
    "surprise_philosophical_shift_weight": 0.3, # Weight for change in approach family
    "surprise_novelty_weight": 0.3, # Weight for how rare the approach family is
    "surprise_overall_threshold": 0.9, # Combined weighted threshold to trigger surprise
    # Context & Prompts
    "global_context_in_prompts": True, # Include global context (best score, etc.)
    "track_explored_approaches": True, # Include summary of explored thoughts/approaches
    "sibling_awareness": True, # Include sibling context in prompts
    # Performance & State
    "memory_cutoff": 5, # Max number of high-scoring nodes to remember
    "early_stopping": True,
    "early_stopping_threshold": 10.0, # Score threshold (1-10)
    "early_stopping_stability": 2, # Iterations score must stay >= threshold
    "enable_state_persistence": True, # Master switch for using SQLite DB
    # Output & Logging
    "show_processing_details": True, # Show verbose MCTS steps in chat output
    "debug_logging": False, # Enable detailed DEBUG level logs (console/log file)
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
    # 'variant' is dynamically assigned if no other type matches well
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
LOG_LEVEL = logging.DEBUG if DEFAULT_CONFIG['debug_logging'] else logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s [%(levelname)s] %(message)s"
PIPE_LOG_NAME = f"pipe.{PIPE_NAME}" # Specific logger name for the pipe
MCTS_LOG_NAME = f"mcts.{PIPE_NAME}" # Specific logger name for MCTS logic


# ==============================================================================
# Prompts (Formerly prompts.py)
# ==============================================================================

# --- Core MCTS Prompts ---
INITIAL_ANALYSIS_PROMPT = """ Utilize INTENT_CLASSIFIER_PROMPT when appropriate.
<instruction>Provide an initial analysis and interpretation of the themes, arguments, and potential implications presented by the user suitable for the further MCTS analysis.</instruction>
<question>{question}</question>
"""

# <<< MODIFIED: Enhanced Generate Thought Prompt >>>
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

# <<< MODIFIED: Enhanced Evaluate Analysis Prompt >>>
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
<instruction>Synthesize the key insights developed along the primary path of analysis below into a concise, conclusive statement addressing the original question/text. Focus on the progression of ideas represented by the sequence of best scoring nodes and 'thoughts'.</instruction>

<original_question_summary>{question_summary}</original_question_summary>
<initial_analysis_summary>{initial_analysis_summary}</initial_analysis_summary>
<best_analysis_score>{best_score}/10</best_analysis_score>

<development_path>
{path_thoughts}
</development_path>

<final_best_analysis_summary>{final_best_analysis_summary}</final_best_analysis_summary>

Synthesize the journey of thoughts into a final conclusion:</instruction>
"""

# --- Intent Classification Prompt ---
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

# --- Intent Handling Prompts ---
ASK_PROCESS_EXPLANATION = """ Answer adaptively to the user's question about how the tool works.
I use an Advanced Bayesian Monte Carlo Tree Search (MCTS) algorithm to analyze text or questions. Here's a breakdown:
- **Exploration vs. Exploitation:** I balance trying new interpretations (exploration) with refining promising ones (exploitation) using techniques like UCT or Thompson Sampling.
- **Bayesian Evaluation (Optional):** I can use Beta distributions to model the uncertainty in analysis scores, leading to potentially more robust exploration.
- **Node Expansion:** I generate new 'thoughts' (critiques, alternative angles, connections) using LLM calls to branch out the analysis tree. These thoughts are generated based on advanced prompts encouraging diverse perspectives, critical thinking, and coherence.
- **Simulation (Evaluation):** I assess the quality of each analysis node using LLM calls, judging insight, novelty, relevance, coherence, and grounding compared to the best analysis found so far.
- **Backpropagation:** Scores (or Bayesian parameters) and visit counts are updated back up the tree path after each evaluation.
- **State Persistence (Optional):** Within a single chat session, I can save key results (best analysis, score, tags) and learned preferences for different analytical approaches to a local database (`{db_file_name}`). This allows me to 'continue' an analysis, avoiding known unproductive paths.
- **Intent Handling:** I try to figure out if you want a completely new analysis, want to build on the last one, or are asking about results, my process, or settings.

You can fine-tune parameters like exploration intensity, iterations, evaluation methods, and more using the 'Valves' settings in the UI.
"""

GENERAL_CONVERSATION_PROMPT = """
The user has provided input classified as general conversation. Respond in a friendly and engaging manner, maintaining an appropriate tone for the conversation.

User Input: "{user_input}"

Your Response:
"""

# ==============================================================================
# Utility Functions (Formerly utils.py)
# ==============================================================================

# --- Logger Setup ---
# Store loggers globally to avoid reconfiguration within the single script
loggers: Dict[str, logging.Logger] = {}

def setup_logger(name: str, level: int, log_format: str) -> logging.Logger:
    """Sets up or retrieves a logger with the specified configuration."""
    global loggers
    if name in loggers:
        loggers[name].setLevel(level)
        for handler in loggers[name].handlers: handler.setLevel(level)
        return loggers[name]

    logger_instance = logging.getLogger(name)
    logger_instance.setLevel(level)
    handler_name = f"{name}_handler"

    if not any(handler.get_name() == handler_name for handler in logger_instance.handlers):
        handler = logging.StreamHandler()
        handler.set_name(handler_name)
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)
        logger_instance.addHandler(handler)
        logger_instance.propagate = False

    loggers[name] = logger_instance
    return logger_instance

# --- Initialize Loggers ---
# Initialize loggers defined in config section *after* setup_logger is defined
logger = setup_logger(PIPE_LOG_NAME, LOG_LEVEL, LOG_FORMAT)
mcts_logger = setup_logger(MCTS_LOG_NAME, LOG_LEVEL, LOG_FORMAT)


# --- Text Processing ---
def truncate_text(text: Optional[str], max_length: int = 200) -> str:
    """Truncates text, trying to end at a word boundary."""
    if not text: return ""
    text = str(text).strip()
    text = re.sub(r"^\s*```(json|markdown|text)?\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE).strip()
    if len(text) <= max_length: return text
    last_space = text.rfind(" ", 0, max_length)
    return text[:last_space] + "..." if last_space != -1 else text[:max_length] + "..."

def calculate_semantic_distance(
    text1: Optional[str],
    text2: Optional[str],
    logger_instance: logging.Logger, # Pass logger explicitly
    use_tfidf: bool = SKLEARN_AVAILABLE
) -> float:
    """Calculates semantic distance (0=identical, 1=different) using TF-IDF or Jaccard."""
    if not text1 or not text2: return 1.0
    text1, text2 = str(text1), str(text2)

    if use_tfidf and SKLEARN_AVAILABLE and TfidfVectorizer and cosine_similarity:
        try:
            vectorizer = TfidfVectorizer(stop_words=CUSTOM_STOP_WORDS, max_df=0.9, min_df=1)
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            if tfidf_matrix.shape[0] < 2 or tfidf_matrix.shape[1] == 0:
                raise ValueError(f"TF-IDF matrix shape issue: {tfidf_matrix.shape}")
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return 1.0 - max(0.0, min(1.0, similarity))
        except ValueError as ve:
             logger_instance.warning(f"TF-IDF semantic distance value error: {ve}. Using Jaccard.")
        except Exception as e:
            logger_instance.warning(f"TF-IDF semantic distance failed: {e}. Using Jaccard.")

    try: # Jaccard Fallback
        words1 = set(re.findall(r"\b\w+\b", text1.lower()))
        words2 = set(re.findall(r"\b\w+\b", text2.lower()))
        if not words1 or not words2: return 1.0
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        if union == 0: return 0.0
        return 1.0 - (intersection / union)
    except Exception as fallback_e:
        logger_instance.error(f"Jaccard similarity fallback failed: {fallback_e}")
        return 1.0

# --- Approach Classification ---
def classify_approach(
    thought: Optional[str],
    taxonomy: Dict[str, List[str]],
    metadata: Dict[str, Dict[str, str]],
    random_state: Any, # Typically random.Random instance
    logger_instance: logging.Logger # Pass logger explicitly
) -> Tuple[str, str]:
    """Classifies a thought based on keywords."""
    approach_type, approach_family = "variant", "general"
    if not thought or not isinstance(thought, str): return approach_type, approach_family
    thought_lower = thought.lower()
    scores = {app: sum(1 for kw in kws if kw in thought_lower) for app, kws in taxonomy.items()}
    positive_scores = {app: score for app, score in scores.items() if score > 0}
    if positive_scores:
        max_score = max(positive_scores.values())
        best = [app for app, score in positive_scores.items() if score == max_score]
        approach_type = random_state.choice(best)
    if approach_type in metadata:
        approach_family = metadata[approach_type].get("family", "general")
    logger_instance.debug(f"Classified '{truncate_text(thought, 50)}' as: {approach_type} ({approach_family})")
    return approach_type, approach_family

# --- Mock Admin User ---
class AdminUserMock:
    def __init__(self): self.role = "admin"
ADMIN_USER = AdminUserMock() # Fallback user if real one isn't available


# ==============================================================================
# Database Utilities (Formerly database_utils.py)
# ==============================================================================

def get_db_connection(db_file_path: str, logger_instance: logging.Logger) -> Optional[sqlite3.Connection]:
    """Establishes SQLite connection and ensures table exists."""
    conn = None
    try:
        db_dir = os.path.dirname(db_file_path)
        if db_dir: os.makedirs(db_dir, exist_ok=True) # Ensure directory exists
        conn = sqlite3.connect(db_file_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL;") # Use WAL for better concurrency
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mcts_state (
                chat_id TEXT PRIMARY KEY,
                last_state_json TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mcts_state_timestamp ON mcts_state (timestamp);")
        conn.commit()
        logger_instance.debug(f"DB connected: {db_file_path}")
        return conn
    except sqlite3.Error as e:
        logger_instance.error(f"SQLite connection/setup error for {db_file_path}: {e}", exc_info=True)
        if conn: conn.close()
        return None
    except Exception as e:
        logger_instance.error(f"Unexpected DB connection error {db_file_path}: {e}", exc_info=True)
        if conn: conn.close()
        return None

def save_mcts_state(db_file_path: str, chat_id: str, state: Dict[str, Any], logger_instance: logging.Logger):
    """Saves the MCTS state dict for a chat_id."""
    if not chat_id:
        logger_instance.warning("Cannot save state: chat_id missing.")
        return
    if not isinstance(state, dict) or not state:
         logger_instance.warning(f"Attempted to save invalid state for chat_id {chat_id}.")
         return
    state_json = json.dumps(state)
    conn = get_db_connection(db_file_path, logger_instance)
    if not conn:
        logger_instance.error("Cannot save state: DB connection failed.")
        return
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO mcts_state (chat_id, last_state_json, timestamp) VALUES (?, ?, ?)",
                (chat_id, state_json, datetime.now()),
            )
        logger_instance.info(f"Saved MCTS state for chat_id: {chat_id}")
    except sqlite3.Error as e:
        logger_instance.error(f"SQLite save state error for chat_id {chat_id}: {e}", exc_info=True)
    finally:
        if conn: conn.close()

def load_mcts_state(db_file_path: str, chat_id: str, logger_instance: logging.Logger) -> Optional[Dict[str, Any]]:
    """Loads the most recent MCTS state dict for a chat_id."""
    if not chat_id:
        logger_instance.warning("Cannot load state: chat_id missing.")
        return None
    conn = get_db_connection(db_file_path, logger_instance)
    if not conn:
        logger_instance.error("Cannot load state: DB connection failed.")
        return None
    state_dict = None
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT last_state_json FROM mcts_state WHERE chat_id = ? ORDER BY timestamp DESC LIMIT 1",
                (chat_id,),
            )
            result = cursor.fetchone()
            if result and result[0]:
                try:
                    state_dict = json.loads(result[0])
                    if isinstance(state_dict, dict):
                         logger_instance.info(f"Loaded MCTS state for chat_id: {chat_id}")
                    else:
                         logger_instance.warning(f"Loaded state for {chat_id} not a dict. Discarding.")
                         state_dict = None
                except json.JSONDecodeError as json_err:
                    logger_instance.error(f"Error decoding state JSON for {chat_id}: {json_err}")
                    state_dict = None
            else:
                logger_instance.info(f"No previous MCTS state found for chat_id: {chat_id}")
    except sqlite3.Error as e:
        logger_instance.error(f"SQLite load state error for {chat_id}: {e}", exc_info=True)
    finally:
        if conn: conn.close()
    return state_dict


# ==============================================================================
# LLM Interaction Utilities (Formerly llm_interaction.py)
# ==============================================================================

# --- Ollama Endpoint Interaction ---
# <<< MODIFIED: Added user_object parameter and use it >>>
async def call_ollama_endpoint(
    payload: Dict[str, Any],
    logger_instance: logging.Logger, # Pass logger
    user_object: Optional[Dict],    # <--- ADDED: To pass the real user object
    debug_logging: bool = False
) -> Union[Dict, Any]:
    """Calls the internal Open WebUI Ollama endpoint function."""
    try:
        async def receive(): return {"type": "http.request", "body": json.dumps(payload).encode("utf-8")}
        mock_request = Request(
            scope={"type": "http", "headers": [], "method": "POST", "scheme": "http",
                   "server": ("local", 80), "path": "/api/ollama/generate",
                   "query_string": b"", "client": ("127.0.0.1", 8080), "app": app},
            receive=receive,
        )
        if debug_logging: logger_instance.debug(f"Calling internal ollama: {str(payload)[:200]}...")

        # <<< MODIFIED: Determine user to use >>>
        # Use the actual user object if provided, otherwise fallback to the mock Admin user
        user_to_use = user_object if user_object is not None else ADMIN_USER

        # <<< MODIFIED: Pass the determined user object >>>
        response = await ollama.generate_openai_chat_completion(request=mock_request, form_data=payload, user=user_to_use)

        if debug_logging and not isinstance(response, dict) and not hasattr(response, 'body_iterator'):
             logger_instance.debug(f"Internal endpoint response type: {type(response)}")
        return response
    except Exception as e:
        logger_instance.error(f"Ollama internal call error: {str(e)}", exc_info=debug_logging)
        err_content = f"Error: LLM internal call failed ({str(e)[:100]}...). See logs."
        return {"error": True, "choices": [{"message": {"role": "assistant", "content": err_content}}]}

# --- Response Parsing ---
def get_chunk_content(chunk_bytes: bytes, logger_instance: logging.Logger, debug_logging: bool = False) -> List[str]:
    """Parses a streaming chunk (bytes) and yields content parts (str)."""
    parts = []
    try:
        chunk_str = chunk_bytes.decode("utf-8")
        for line in chunk_str.splitlines():
            line = line.strip()
            if not line or line == "data: [DONE]": continue
            if line.startswith("data: "):
                json_str = line[len("data: "):]
                try:
                    data = json.loads(json_str)
                    content = data.get("choices", [{}])[0].get("delta", {}).get("content")
                    if isinstance(content, str) and content: parts.append(content)
                except (json.JSONDecodeError, IndexError, KeyError, TypeError) as e:
                     if debug_logging: logger_instance.warning(f"Stream chunk parse error: {e} - Data: {json_str[:100]}...")
                except Exception as e: logger_instance.error(f"Unexpected stream chunk error: {e}")
    except UnicodeDecodeError: logger_instance.error(f"Unicode decode error in stream chunk: {chunk_bytes[:100]}...")
    except Exception as e: logger_instance.error(f"Stream chunk decode/split error: {e}", exc_info=debug_logging)
    return parts

def get_response_content(response: Union[Dict, Any], logger_instance: logging.Logger) -> str:
    """Extracts content from a non-streaming Ollama response dict."""
    try:
        if isinstance(response, dict):
            if response.get("error"):
                try: return str(response.get("choices", [{}])[0].get("message", {}).get("content", "Unknown LLM Error"))
                except: err_detail = response.get("error"); return f"LLM Error: {err_detail}" if isinstance(err_detail, str) else "Unknown LLM Error"
            elif "choices" in response and isinstance(response["choices"], list) and response["choices"]:
                 try: return str(response["choices"][0].get("message", {}).get("content", ""))
                 except: logger_instance.warning(f"Unexpected choice structure: {str(response)[:200]}"); return ""
            else: logger_instance.warning(f"Unexpected dict structure: {str(response)[:200]}"); return ""
        else: logger_instance.warning(f"Unexpected response type: {type(response)}"); return ""
    except Exception as e: logger_instance.error(f"Response content extraction error: {str(e)}", exc_info=True); return ""

# --- High-Level LLM Interaction Functions ---
# These functions are now part of the Pipe class, implemented further down.
# async def get_streaming_completion(...) -> This logic is now within Pipe.get_streaming_completion
# async def get_completion(...) -> This logic is now within Pipe.get_completion


# ==============================================================================
# Node Class Definition (Formerly node.py)
# ==============================================================================
class Node(BaseModel):
    """Represents a node in the Monte Carlo Tree Search tree."""
    id: str = Field(default_factory=lambda: f"node_{random.randbytes(4).hex()}")
    content: str = ""
    parent: Optional["Node"] = Field(default=None, exclude=True) # Avoid circular refs in default serialization
    children: List["Node"] = Field(default_factory=list)
    visits: int = 0
    raw_scores: List[Union[int, float]] = Field(default_factory=list)
    sequence: int = 0
    is_surprising: bool = False
    surprise_explanation: str = ""
    approach_type: str = "initial"
    approach_family: str = "general"
    thought: str = ""
    max_children: int = DEFAULT_CONFIG["max_children"] # Default, can be overridden
    use_bayesian_evaluation: bool = DEFAULT_CONFIG["use_bayesian_evaluation"] # Default

    alpha: Optional[float] = None # Bayesian state
    beta: Optional[float] = None  # Bayesian state
    value: Optional[float] = None # Traditional state (cumulative score)
    descriptive_tags: List[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("parent", "children", mode="before")
    @classmethod
    def _validate_optional_fields(cls, v): return v # Pydantic v1 compatibility if needed

    def __init__(self, **data: Any):
        """Initializes the Node, setting up evaluation state."""
        super().__init__(**data)
        self.max_children = data.get("max_children", self.max_children)
        self.use_bayesian_evaluation = data.get("use_bayesian_evaluation", self.use_bayesian_evaluation)

        if self.use_bayesian_evaluation:
            prior_alpha = data.get("alpha", DEFAULT_CONFIG["beta_prior_alpha"])
            prior_beta = data.get("beta", DEFAULT_CONFIG["beta_prior_beta"])
            self.alpha = max(1e-9, float(prior_alpha))
            self.beta = max(1e-9, float(prior_beta))
            self.value = None
        else:
            self.value = float(data.get("value", 0.0))
            self.alpha = None
            self.beta = None

    def add_child(self, child: "Node"): self.children.append(child); child.parent = self
    def fully_expanded(self) -> bool: return len(self.children) >= self.max_children

    def get_bayesian_mean(self) -> float:
        if self.use_bayesian_evaluation and self.alpha is not None and self.beta is not None:
            alpha_safe, beta_safe = max(1e-9, self.alpha), max(1e-9, self.beta)
            denominator = alpha_safe + beta_safe
            return (alpha_safe / denominator) if denominator > 1e-18 else 0.5
        return 0.5

    def get_average_score(self) -> float:
        if self.use_bayesian_evaluation: return self.get_bayesian_mean() * 10.0
        else: return (self.value / self.visits) if self.visits > 0 and self.value is not None else 5.0

    def thompson_sample(self) -> float:
        if self.use_bayesian_evaluation and self.alpha is not None and self.beta is not None:
            alpha_safe, beta_safe = max(1e-9, self.alpha), max(1e-9, self.beta)
            try: return float(beta_sample(alpha_safe, beta_safe))
            except Exception as e:
                mcts_logger.warning(f"TS failed node {self.sequence} (α={alpha_safe}, β={beta_safe}): {e}. Using mean.")
                return self.get_bayesian_mean()
        return 0.5

    def best_child(self) -> Optional["Node"]:
        valid_children = [c for c in self.children if c is not None]
        if not valid_children: return None
        max_visits = max(child.visits for child in valid_children)
        most_visited = [child for child in valid_children if child.visits == max_visits]
        if len(most_visited) == 1: return most_visited[0]
        elif not most_visited: return random.choice(valid_children) # Fallback
        # Tie-break by score
        key_func = lambda c: c.get_bayesian_mean() if self.use_bayesian_evaluation else c.get_average_score()
        return max(most_visited, key=key_func)

    def node_to_json(self) -> Dict[str, Any]:
        score = self.get_average_score()
        valid_children = [child for child in self.children if child is not None]
        base: Dict[str, Any] = {
            "id": self.id, "sequence": self.sequence, "content_summary": truncate_text(self.content, 150),
            "visits": self.visits, "approach_type": self.approach_type, "approach_family": self.approach_family,
            "is_surprising": self.is_surprising, "thought_summary": truncate_text(self.thought, 100),
            "tags": self.descriptive_tags, "score": round(score, 2), "children_count": len(valid_children),
            "children": [child.node_to_json() for child in valid_children], # Recursive for debug views
        }
        if self.use_bayesian_evaluation and self.alpha is not None and self.beta is not None:
             base.update({"alpha": round(self.alpha, 3), "beta": round(self.beta, 3), "mean": round(self.get_bayesian_mean(), 3)})
        elif not self.use_bayesian_evaluation and self.value is not None:
             base["value_cum"] = round(self.value, 2)
        return base

    def node_to_state_dict(self) -> Dict[str, Any]:
        score = self.get_average_score()
        state: Dict[str, Any] = {
            "id": self.id, "sequence": self.sequence, "content_summary": truncate_text(self.content, 250),
            "visits": self.visits, "approach_type": self.approach_type, "approach_family": self.approach_family,
            "thought": self.thought, "tags": self.descriptive_tags, "score": round(score, 2),
            "is_surprising": self.is_surprising,
        }
        if self.use_bayesian_evaluation and self.alpha is not None and self.beta is not None:
            state.update({"alpha": round(self.alpha, 4), "beta": round(self.beta, 4)})
        elif not self.use_bayesian_evaluation and self.value is not None:
            state["value"] = round(self.value, 2)
        return state


# ==============================================================================
# LLM Interface Definition (Protocol for MCTS)
# ==============================================================================
class LLMInterface:
    """Defines the methods MCTS needs to interact with an LLM."""
    async def generate_thought(self, current_analysis: str, context: Dict, config: Dict) -> str: ...
    async def update_approach(self, original_analysis: str, critique: str, context: Dict, config: Dict) -> str: ...
    async def evaluate_answer(self, analysis_to_evaluate: str, context: Dict, config: Dict) -> int: ...
    async def get_completion(self, model: str, messages: List[Dict[str, str]]) -> str: ...
    async def progress(self, message: str): ...
    async def emit_message(self, message: str): ...
    def resolve_model(self, body: dict) -> str: ...


# ==============================================================================
# MCTS Class Definition (Formerly mcts.py)
# ==============================================================================
class MCTS:
    """Implements the Monte Carlo Tree Search algorithm for text analysis."""

    def __init__(
        self,
        llm_interface: LLMInterface,
        question: str,
        mcts_config: Dict[str, Any],
        initial_analysis_content: str,
        initial_state: Optional[Dict[str, Any]] = None,
        model_body: Optional[Dict[str, Any]] = None
    ):
        self.llm = llm_interface
        self.config = mcts_config
        self.question = question
        self.question_summary = self._summarize_question(question)
        self.model_body = model_body or {} # For model resolution within MCTS

        self.debug_logging = self.config.get("debug_logging", False)
        self.show_chat_details = self.config.get("show_processing_details", False)
        # Use MCTS specific logger instance
        self.logger = logging.getLogger(MCTS_LOG_NAME)
        self.logger.setLevel(logging.DEBUG if self.debug_logging else logging.INFO)

        self.loaded_initial_state = initial_state
        self.node_sequence = 0
        self.iterations_completed = 0
        self.simulations_completed = 0
        self.high_score_counter = 0
        self.random_state = random.Random()

        self.thought_history: List[str] = []
        self.debug_history: List[str] = []
        self.surprising_nodes: List[Node] = []
        self.explored_approaches: Dict[str, List[str]] = {}
        self.explored_thoughts: Set[str] = set()
        self.memory: Dict[str, Any] = {"depth": 0, "branches": 0, "high_scoring_nodes": []}

        mcts_start_log = f"# MCTS Start\nQ Summary: {self.question_summary}\n"
        if self.loaded_initial_state:
            mcts_start_log += f"Loaded state (Best score: {self.loaded_initial_state.get('best_score', 'N/A')}).\n"
        self.thought_history.append(mcts_start_log)

        cfg = self.config
        prior_alpha = max(1e-9, cfg["beta_prior_alpha"])
        prior_beta = max(1e-9, cfg["beta_prior_beta"])

        # Init Approach Priors/Scores
        self.approach_alphas: Dict[str, float] = {}
        self.approach_betas: Dict[str, float] = {}
        self.approach_scores: Dict[str, float] = {}
        loaded_priors = self.loaded_initial_state.get("approach_priors") if self.loaded_initial_state else None
        if cfg["use_bayesian_evaluation"] and loaded_priors and isinstance(loaded_priors.get("alpha"), dict):
            self.approach_alphas = {k: max(1e-9, v) for k, v in loaded_priors["alpha"].items()}
            self.approach_betas = {k: max(1e-9, v) for k, v in loaded_priors.get("beta", {}).items()} # Handle if beta missing
            self.logger.info("Loaded approach priors (Bayesian) from state.")
        else:
            keys = list(APPROACH_TAXONOMY.keys()) + ["initial", "variant"]
            for app in keys:
                self.approach_alphas[app] = prior_alpha
                self.approach_betas[app] = prior_beta
                self.approach_scores[app] = 5.0 # Default avg score
            if loaded_priors: self.logger.warning("Could not load Bayesian priors, using defaults.")
            else: self.logger.info("Initialized default approach priors.")

        # Init Best Solution Tracking
        self.best_score: float = 0.0
        self.best_solution: str = initial_analysis_content
        if self.loaded_initial_state:
            self.best_score = float(self.loaded_initial_state.get("best_score", 0.0))
            self.previous_best_solution_content = self.loaded_initial_state.get("best_solution_content")
            self.logger.info(f"Initialized best score tracker ({self.best_score}) from state.")
        else: self.previous_best_solution_content = None

        # Init Root Node
        self.root = Node(
            content=initial_analysis_content, sequence=self.get_next_sequence(),
            parent=None, max_children=cfg["max_children"],
            use_bayesian_evaluation=cfg["use_bayesian_evaluation"],
            alpha=prior_alpha, beta=prior_beta, # Root starts with default priors
            approach_type="initial", approach_family="general",
        )

        # Load Unfit Markers
        self.unfit_markers: List[Dict[str, Any]] = []
        if self.loaded_initial_state:
            loaded_markers = self.loaded_initial_state.get("unfit_markers", [])
            if isinstance(loaded_markers, list):
                 self.unfit_markers = loaded_markers
                 self.logger.info(f"Loaded {len(self.unfit_markers)} unfit markers from state.")
            else: self.logger.warning("Loaded unfit markers not a list. Ignoring.")

    def get_next_sequence(self) -> int: self.node_sequence += 1; return self.node_sequence

    def _summarize_question(self, text: str, max_words=50) -> str:
        if not text: return "N/A"
        words = re.findall(r'\b\w+\b', text)
        return " ".join(words[:max_words]) + "..." if len(words) > max_words else text.strip()

    def export_tree_as_json(self) -> Dict[str, Any]:
        try: return self.root.node_to_json() if self.root else {"error": "No root node"}
        except Exception as e: self.logger.error(f"Tree JSON export error: {e}", exc_info=self.debug_logging); return {"error": f"Export failed: {e}"}

    def get_context_for_node(self, node: Node) -> Dict[str, str]:
        """Gathers context for prompts, including loaded state."""
        cfg = self.config
        context: Dict[str, Any] = {
            "question_summary": self.question_summary, "best_answer": truncate_text(str(self.best_solution), 300),
            "best_score": f"{self.best_score:.1f}", "current_answer": truncate_text(node.content, 300),
            "current_sequence": str(node.sequence), "current_approach": node.approach_type,
            "current_tags": ", ".join(node.descriptive_tags) or "None", "tree_depth": str(self.memory.get("depth", 0)),
            "branches": str(self.memory.get("branches", 0)), "approach_types": ", ".join(self.explored_approaches.keys()),
            "previous_best_summary": "N/A", "unfit_markers_summary": "None", "learned_approach_summary": "Default priors",
            "explored_approaches": "None yet.", "high_scoring_examples": "None yet.", "sibling_approaches": "None.",
        }
        # Add context from loaded state
        if self.loaded_initial_state:
            context["previous_best_summary"] = self.loaded_initial_state.get("best_solution_summary", "N/A")
            unfit = self.loaded_initial_state.get("unfit_markers", [])
            if unfit:
                markers = "; ".join([f"'{m.get('summary', '?')}' ({m.get('reason', '?')})" for m in unfit[:5]])
                context["unfit_markers_summary"] = markers + ("..." if len(unfit) > 5 else "")
            priors = self.loaded_initial_state.get("approach_priors")
            if priors and isinstance(priors.get("alpha"), dict):
                means = {}
                for app, alpha in priors["alpha"].items():
                    beta = priors.get("beta", {}).get(app, 1.0)
                    alpha, beta = max(1e-9, alpha), max(1e-9, beta)
                    if alpha + beta > 1e-9: means[app] = (alpha / (alpha + beta)) * 10
                if means:
                    top = sorted(means.items(), key=lambda i: i[1], reverse=True)[:3]
                    context["learned_approach_summary"] = f"Favors: {', '.join([f'{a} ({s:.1f})' for a, s in top])}" + ("..." if len(means)>3 else "")
        # Add dynamic context
        try: # Explored Approaches
            if cfg["track_explored_approaches"] and self.explored_approaches:
                texts = []
                for app, thoughts in sorted(self.explored_approaches.items()):
                    if not thoughts: continue
                    count = len(thoughts); score_txt = ""
                    if cfg["use_bayesian_evaluation"]:
                        alpha = self.approach_alphas.get(app, 1.0); beta = self.approach_betas.get(app, 1.0)
                        if (alpha+beta)>1e-9: score_txt = f"(βM:{alpha/(alpha+beta):.2f}, N={count})"
                        else: score_txt = f"(N={count}, Err)"
                    else: score_txt = f"(Avg:{self.approach_scores.get(app, 5.0):.1f}, N={count})"
                    samples = '; '.join([f"'{truncate_text(t, 40)}'" for t in thoughts[-min(2, count):]])
                    texts.append(f"- {app} {score_txt}: {samples}")
                if texts: context["explored_approaches"] = "\n".join(texts)
        except Exception as e: self.logger.error(f"Ctx err (approaches): {e}", exc_info=self.debug_logging); context["explored_approaches"] = "Error."
        try: # High Scores
            if self.memory["high_scoring_nodes"]:
                 context["high_scoring_examples"] = "\n".join(["Top Examples:"] + [f"- S:{s:.1f} ({a}): {truncate_text(c, 60)}" for s,c,a,t in self.memory["high_scoring_nodes"]])
        except Exception as e: self.logger.error(f"Ctx err (high scores): {e}", exc_info=self.debug_logging); context["high_scoring_examples"] = "Error."
        try: # Siblings
            if cfg["sibling_awareness"] and node.parent and len(node.parent.children) > 1:
                sibs = [s for s in node.parent.children if s is not None and s != node and s.thought and s.visits > 0]
                if sibs:
                    texts = [f'- "{truncate_text(s.thought, 50)}" (S:{s.get_average_score():.1f}, Tags:{s.descriptive_tags})' for s in sorted(sibs, key=lambda x: x.sequence)]
                    if texts: context["sibling_approaches"] = "\n".join(["Siblings:"] + texts)
        except Exception as e: self.logger.error(f"Ctx err (siblings): {e}", exc_info=self.debug_logging); context["sibling_approaches"] = "Error."
        # Final conversion to string
        return {k: str(v) if v is not None else "" for k, v in context.items()}

    def _calculate_uct(self, node: Node, parent_visits: int) -> float:
        """Calculates UCT score, considering exploitation, exploration, penalties, and bonuses."""
        cfg = self.config
        if node.visits == 0: return float('inf')
        exploit = (node.get_bayesian_mean() if cfg["use_bayesian_evaluation"] else node.get_average_score() / 10.0)
        explore = cfg["exploration_weight"] * math.sqrt(math.log(max(1, parent_visits)) / node.visits) if parent_visits > 0 and node.visits > 0 else cfg["exploration_weight"]

        # Penalty for unfit nodes
        is_unfit = False
        if hasattr(self, 'unfit_markers') and self.unfit_markers:
            for marker in self.unfit_markers:
                if marker.get("id") == node.id or marker.get("sequence") == node.sequence: is_unfit = True; break
        if is_unfit and not node.is_surprising: return 1e-6 # Penalize heavily

        surprise_bonus = 0.3 if node.is_surprising else 0.0
        diversity_bonus = 0.0
        if node.parent and len(node.parent.children) > 1 and cfg["score_diversity_bonus"] > 0:
            my_score = node.get_average_score() / 10.0
            sib_scores = [s.get_average_score() / 10.0 for s in node.parent.children if s and s != node and s.visits > 0]
            if sib_scores: diversity_bonus = cfg["score_diversity_bonus"] * abs(my_score - (sum(sib_scores) / len(sib_scores)))

        uct = exploit + explore + surprise_bonus + diversity_bonus
        return uct if math.isfinite(uct) else 0.0

    def _collect_non_leaf_nodes(self, node: Node, non_leaf_nodes: List[Node], max_depth: int, current_depth: int = 0):
        """Helper to find expandable nodes up to max_depth."""
        if node is None or current_depth > max_depth: return
        if node.children and not node.fully_expanded(): non_leaf_nodes.append(node)
        for child in node.children:
            if child: self._collect_non_leaf_nodes(child, non_leaf_nodes, max_depth, current_depth + 1)

    async def select(self) -> Node:
        """Selects a node using UCT or Thompson Sampling."""
        cfg = self.config; node = self.root; path = [node]; debug_info = "### Selection Path:\n"
        if node is None: raise RuntimeError("MCTS Select Error: Root node missing.")

        # Forced Exploration
        force_interval = cfg["force_exploration_interval"]
        if (force_interval > 0 and self.simulations_completed > 0 and
            self.simulations_completed % force_interval == 0 and self.memory.get("depth", 0) > 1):
            candidates: List[Node] = []
            self._collect_non_leaf_nodes(self.root, candidates, max_depth=max(1, self.memory["depth"] // 2))
            if candidates:
                selected = self.random_state.choice(candidates)
                debug_info += f"- Branch Enhance: Forced selection of Node {selected.sequence}\n"
                self.logger.debug(f"BRANCH ENHANCE: Selected Node {selected.sequence}")
                curr = selected; p_str_list = []
                while curr: p_str_list.append(f"N{curr.sequence}"); curr = curr.parent
                self.thought_history.append(f"### Selection Path (Forced)\n{' -> '.join(reversed(p_str_list))}\n")
                if self.debug_logging: self.debug_history.append(debug_info)
                return selected

        # Standard Selection
        while node.children:
            valid_children = [child for child in node.children if child]
            if not valid_children: self.logger.warning(f"Node {node.sequence} has invalid children. Stopping select."); break
            unvisited = [child for child in valid_children if child.visits == 0]
            if unvisited: node = self.random_state.choice(unvisited); debug_info += f"- Selected unvisited Node {node.sequence}\n"; path.append(node); break

            parent_visits = node.visits; use_ts = cfg["use_bayesian_evaluation"] and cfg["use_thompson_sampling"]
            selected_child = None

            if use_ts:
                samples = [(c, c.thompson_sample()) for c in valid_children if math.isfinite(c.thompson_sample())]
                if samples: selected_child, best_val = max(samples, key=lambda x: x[1]); debug_info += f"- TS selected Node {selected_child.sequence} ({best_val:.3f})\n"
                else: self.logger.warning(f"No valid TS samples for children of {node.sequence}. Falling back to UCT."); use_ts = False # Fallback flag

            if not use_ts: # UCT Selection (or TS fallback)
                uct_vals = [(c, self._calculate_uct(c, parent_visits)) for c in valid_children if math.isfinite(self._calculate_uct(c, parent_visits))]
                if not uct_vals: self.logger.error(f"No valid UCT values for children of {node.sequence}. Random choice."); selected_child = self.random_state.choice(valid_children) if valid_children else None
                else: selected_child, best_val = max(uct_vals, key=lambda x: x[1]); debug_info += f"- UCT selected Node {selected_child.sequence} ({best_val:.3f})\n"

            if selected_child: node = selected_child; path.append(node)
            else: self.logger.error(f"Selection failed for node {node.sequence}, no valid child chosen."); break # Stop if selection fails

            if not node.fully_expanded(): break # Stop if expandable

        path_str = " -> ".join([f"N{n.sequence}(V:{n.visits},S:{n.get_average_score():.1f})" for n in path])
        self.thought_history.append(f"### Selection Path\n{path_str}\n")
        if self.debug_logging: self.debug_history.append(debug_info); self.logger.debug(f"Select path: {path_str}\n{debug_info.strip()}")
        self.memory["depth"] = max(self.memory.get("depth", 0), len(path) - 1)
        return node

    def _check_surprise(self, parent_node: Node, new_content: str, new_approach_type: str, new_approach_family: str) -> Tuple[bool, str]:
        """Checks if a new node is 'surprising'."""
        cfg = self.config; factors: List[Dict] = []; explanation = ""
        if cfg["use_semantic_distance"]:
            try:
                dist = calculate_semantic_distance(parent_node.content, new_content, self.logger)
                if dist > cfg["surprise_threshold"]:
                    factors.append({"type": "semantic", "value": dist, "weight": cfg["surprise_semantic_weight"], "desc": f"Semantic dist ({dist:.2f})"})
            except Exception as e:
                self.logger.warning(f"Surprise semantic dist check failed: {e}")
        parent_family = getattr(parent_node, "approach_family", "general")
        if parent_family != new_approach_family and new_approach_family != "general":
            factors.append({"type": "family_shift", "value": 1.0, "weight": cfg["surprise_philosophical_shift_weight"], "desc": f"Shift '{parent_family}'->'{new_approach_family}'"})
        try: # Novelty check via BFS
            counts = Counter(); q: List[Tuple[Node, int]] = [(self.root, 0)] if self.root else []; visited_bfs = set(); nodes_bfs = 0
            MAX_BFS = 100; MAX_DEPTH = 5
            while q and nodes_bfs < MAX_BFS:
                 curr, depth = q.pop(0)
                 if not curr or curr.id in visited_bfs or depth > MAX_DEPTH: continue
                 visited_bfs.add(curr.id); nodes_bfs += 1
                 counts[getattr(curr, 'approach_family', 'general')] += 1
                 for child in curr.children:
                     if child and child.id not in visited_bfs and depth + 1 <= MAX_DEPTH: q.append((child, depth + 1))
            if counts.get(new_approach_family, 0) <= 1 and new_approach_family != "general":
                factors.append({"type": "novelty", "value": 0.8, "weight": cfg["surprise_novelty_weight"], "desc": f"Novel family ('{new_approach_family}')"})
        except Exception as e: self.logger.warning(f"Surprise novelty BFS failed: {e}", exc_info=self.debug_logging)

        if factors:
            total_w_score = sum(f["value"] * f["weight"] for f in factors); total_w = sum(f["weight"] for f in factors)
            score = (total_w_score / total_w) if total_w > 1e-6 else 0.0
            if score >= cfg["surprise_overall_threshold"]:
                descs = [f"- {f['desc']} (V:{f['value']:.2f}*W:{f['weight']:.1f})" for f in factors]
                explanation = f"Surprise ({score:.2f} >= {cfg['surprise_overall_threshold']:.2f}):\n" + "\n".join(descs)
                self.logger.debug(f"Surprise DETECTED (potential node {parent_node.sequence+1}): Score={score:.2f}\n{explanation}")
                return True, explanation
        return False, ""

    async def expand(self, node: Node) -> Optional[Node]:
        """Expands a node: generates thought, updates analysis, creates child."""
        cfg = self.config; self.logger.debug(f"Expanding node {node.sequence}...")
        try:
            await self.llm.progress(f"Node {node.sequence}: Generating thought...")
            context = self.get_context_for_node(node)
            thought = await self.llm.generate_thought(node.content, context, cfg)
            if not isinstance(thought, str) or not thought.strip() or thought.startswith("Error:"):
                self.logger.error(f"Invalid thought for node {node.sequence}: '{thought}'"); self.thought_history.append(f"### Expand {node.sequence}\n... Thought Failed: {thought}\n"); return None
            thought = thought.strip(); self.logger.debug(f"Node {node.sequence} Thought: '{thought}'")
            approach_type, approach_family = classify_approach(thought, APPROACH_TAXONOMY, APPROACH_METADATA, self.random_state, self.logger)
            t_entry = f"### Expand Node {node.sequence}\n... Thought: {thought}\n... Approach: {approach_type} ({approach_family})\n"
            self.explored_thoughts.add(thought); self.explored_approaches.setdefault(approach_type, []).append(thought)

            await self.llm.progress(f"Node {node.sequence}: Updating analysis...")
            new_content = await self.llm.update_approach(node.content, thought, context, cfg)
            if not isinstance(new_content, str) or not new_content.strip() or new_content.startswith("Error:"):
                self.logger.error(f"Update failed for node {node.sequence}: '{new_content}'"); self.thought_history.append(f"{t_entry}... Update Failed: {new_content}\n"); return None
            new_content = new_content.strip()

            await self.llm.progress(f"Node {node.sequence}: Generating tags...")
            new_tags = await self._generate_tags_for_node(new_content); t_entry += f"... Tags: {new_tags}\n"
            self.logger.debug(f"Potential Node {node.sequence+1} Tags: {new_tags}")
            is_surprising, explanation = self._check_surprise(node, new_content, approach_type, approach_family)
            if is_surprising: t_entry += f"**SURPRISE!**\n{explanation}\n"

            child = Node(content=new_content, sequence=self.get_next_sequence(), is_surprising=is_surprising, surprise_explanation=explanation,
                         approach_type=approach_type, approach_family=approach_family, thought=thought, max_children=cfg["max_children"],
                         use_bayesian_evaluation=cfg["use_bayesian_evaluation"], alpha=max(1e-9, cfg["beta_prior_alpha"]),
                         beta=max(1e-9, cfg["beta_prior_beta"]), descriptive_tags=new_tags)
            node.add_child(child)
            if is_surprising: self.surprising_nodes.append(child)
            t_entry += f"--> New Analysis {child.sequence} (Tags: {child.descriptive_tags}): {truncate_text(new_content, 100)}\n"
            self.thought_history.append(t_entry); self.memory["branches"] = self.memory.get("branches", 0) + (len(node.children) > 1)
            self.logger.debug(f"Expanded Node {node.sequence} -> Child {child.sequence}")
            return child
        except Exception as e: self.logger.error(f"Expand error on Node {node.sequence}: {e}", exc_info=self.debug_logging); self.thought_history.append(f"### Expand {node.sequence}\n... Error: {e}\n"); return None

    async def _generate_tags_for_node(self, text: str) -> List[str]:
        """Generates keyword tags for analysis text via LLM."""
        if not text: return []
        try:
            model = self.llm.resolve_model(self.model_body)
            if not model: self.logger.error("Tag gen failed: model resolve failed."); return []
            prompt = GENERATE_TAGS_PROMPT.format(analysis_text=text)
            raw = await self.llm.get_completion(model, [{"role": "user", "content": prompt}]) # LLM call delegated to interface
            if not raw or raw.startswith("Error:"): self.logger.warning(f"Tag gen LLM failed: {raw}"); return []

            cleaned = re.sub(r"^\s*<.*?>\s*", "", raw, flags=re.DOTALL | re.IGNORECASE) # Remove instructions
            cleaned = re.sub(r"^\s*(tags|keywords|output|response)[:\-]?\s*", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"[`'*_]", "", cleaned) # Remove markdown
            tags = re.split(r"[,\n;]+", cleaned); final_tags = []; seen = set()
            for tag in tags:
                t = tag.strip().strip("'.\"-()[]{}").strip()
                t = re.sub(r"\s+", " ", t)
                if t and 1 < len(t) < 50 and not t.isdigit() and t.lower() != "none" and t.lower() not in seen:
                    final_tags.append(t); seen.add(t.lower())
                if len(final_tags) >= 3: break
            if self.debug_logging: self.logger.debug(f"Raw tags: '{raw}'. Cleaned: {final_tags}")
            return final_tags
        except Exception as e: self.logger.error(f"Tag generation error: {e}", exc_info=self.debug_logging); return []

    async def simulate(self, node: Node) -> Optional[float]:
        """Evaluates a node's analysis using LLM, returns score (1-10) or None."""
        cfg = self.config; self.logger.debug(f"Simulating node {node.sequence}...")
        score: Optional[float] = None; raw_score = 0
        try:
            await self.llm.progress(f"Evaluating Node {node.sequence}...")
            context = self.get_context_for_node(node)
            if not node.content: self.logger.warning(f"Node {node.sequence} empty. Score 1."); raw_score = 1; score = 1.0
            else: raw_score = await self.llm.evaluate_answer(node.content, context, cfg); score = float(max(1, min(10, raw_score)))
            node.raw_scores.append(raw_score); approach = node.approach_type or "unknown"

            if score is not None: # Update approach scores/priors
                if cfg["use_bayesian_evaluation"]:
                    s, f = max(0, score - 1), max(0, 10 - score) # Pseudo counts
                    alpha, beta = self.approach_alphas.get(approach, cfg["beta_prior_alpha"]), self.approach_betas.get(approach, cfg["beta_prior_beta"])
                    self.approach_alphas[approach] = max(1e-9, alpha + s); self.approach_betas[approach] = max(1e-9, beta + f)
                    if self.debug_logging: self.logger.debug(f"Bayesian update '{approach}': α={self.approach_alphas[approach]:.2f}, β={self.approach_betas[approach]:.2f}")
                else:
                    current = self.approach_scores.get(approach, score); self.approach_scores[approach] = (0.7 * score) + (0.3 * current) # EMA
                    if self.debug_logging: self.logger.debug(f"Traditional update '{approach}': Avg={self.approach_scores[approach]:.2f}")

            self.logger.debug(f"Node {node.sequence} eval: Raw={raw_score}, Score={score:.1f}/10")
            self.thought_history.append(f"### Evaluate Node {node.sequence} (Tags:{node.descriptive_tags})\n... Score: {score:.1f}/10 (raw: {raw_score})\n")

            if score is not None and score >= 7.0: # Update high score memory
                entry = (score, node.content, approach, node.thought)
                self.memory["high_scoring_nodes"] = sorted(self.memory["high_scoring_nodes"] + [entry], key=lambda x: x[0], reverse=True)[:cfg["memory_cutoff"]]
        except Exception as e: self.logger.error(f"Simulate error node {node.sequence}: {e}", exc_info=self.debug_logging); self.thought_history.append(f"### Evaluate {node.sequence}\n... Error: {e}\n"); return None
        return score

    def backpropagate(self, node: Node, score: float):
        """Backpropagates score up the tree."""
        cfg = self.config; self.logger.debug(f"Backpropagating score {score:.2f} from {node.sequence}...")
        path = []; curr: Optional[Node] = node
        s, f = max(0, score - 1), max(0, 10 - score) # Pseudo counts for Bayesian
        while curr:
            nid = curr.sequence; path.append(f"N{nid}")
            curr.visits += 1
            if cfg["use_bayesian_evaluation"]:
                if curr.alpha is not None and curr.beta is not None: curr.alpha = max(1e-9, curr.alpha + s); curr.beta = max(1e-9, curr.beta + f)
                else: self.logger.warning(f"Node {nid} missing alpha/beta in BP."); curr.alpha = max(1e-9, cfg["beta_prior_alpha"] + s); curr.beta = max(1e-9, cfg["beta_prior_beta"] + f) # Re-init?
                if self.debug_logging: self.logger.debug(f"  BP N{nid}: α={curr.alpha:.2f}, β={curr.beta:.2f}")
            else:
                if curr.value is not None: curr.value += score
                else: self.logger.warning(f"Node {nid} missing value in BP."); curr.value = score # Init
                if self.debug_logging: self.logger.debug(f"  BP N{nid}: V={curr.value:.2f}, Avg={curr.get_average_score():.2f}")
            curr = curr.parent
        path_str = " -> ".join(reversed(path))
        self.thought_history.append(f"### Backprop Score {score:.1f}\n... Path: {path_str}\n")
        if self.debug_logging: self.logger.debug(f"Backprop complete: {path_str}")

    async def search(self, sims_per_iter: int) -> bool:
        """Performs one MCTS iteration. Returns True to continue, False to stop early."""
        cfg = self.config; iter_num = self.iterations_completed + 1
        self.logger.info(f"--- MCTS Iteration {iter_num} ({sims_per_iter} sims) ---")
        for i in range(sims_per_iter):
            sim_num = i + 1; self.simulations_completed += 1
            self.thought_history.append(f"### Iter {iter_num} - Sim {sim_num}/{sims_per_iter}\n"); self.logger.debug(f"--- Sim {sim_num} ---")
            sim_msg = ""; score = None; node_to_sim = None
            try: selected = await self.select()
            except Exception as e: self.logger.error(f"Sim {sim_num} Select Error: {e}", exc_info=self.debug_logging); continue
            if not selected: self.logger.error(f"Sim {sim_num}: Selection returned None!"); break
            sim_msg += f"Selected: N{selected.sequence}(V:{selected.visits}, S:{selected.get_average_score():.1f})\n"; node_to_sim = selected

            if not selected.fully_expanded() and selected.content: # Try Expansion
                self.logger.debug(f"Sim {sim_num}: Expanding N{selected.sequence}.")
                try: expanded = await self.expand(selected)
                except Exception as e: self.logger.error(f"Sim {sim_num} Expand Error: {e}", exc_info=self.debug_logging); sim_msg += "--> Expand Error.\n"
                if expanded: node_to_sim = expanded; sim_msg += f'Thought:"{truncate_text(expanded.thought, 50)}"\n--> Expanded to N{expanded.sequence}({expanded.approach_type}, T:{expanded.descriptive_tags})\n'; self.logger.debug(f"Sim {sim_num}: Expanded {selected.sequence} -> {expanded.sequence}.")
                else: self.logger.warning(f"Sim {sim_num}: Expansion failed N{selected.sequence}."); sim_msg += f"--> Expand Failed. Re-eval N{selected.sequence}.\n"
            else: sim_msg += f"--> Re-eval N{selected.sequence} (T:{selected.descriptive_tags}).\n" # Simulate selected

            if node_to_sim and node_to_sim.content: # Simulate
                try: score = await self.simulate(node_to_sim)
                except Exception as e: self.logger.error(f"Sim {sim_num} Simulate Error: {e}", exc_info=self.debug_logging); sim_msg += "Eval Error."
                if score is not None: sim_msg += f"Eval Score: {score:.1f}/10"
                else: sim_msg += "Eval Failed."
            else: sim_msg += "Skipped Eval (No Content)."

            if score is not None: # Backpropagate & Check Best/Stop
                try: self.backpropagate(node_to_sim, score)
                except Exception as e: self.logger.error(f"Sim {sim_num} Backprop Error: {e}", exc_info=self.debug_logging)
                new_best = score > self.best_score
                if new_best: self.best_score = score; self.best_solution = str(node_to_sim.content); self.high_score_counter = 0; sim_msg += " 🏆 (New Best!)"; info=f"N{node_to_sim.sequence}({node_to_sim.approach_type}) T:{node_to_sim.descriptive_tags}"; self.thought_history.append(f"### New Best! S:{score:.1f} ({info})\n"); self.logger.info(f"Sim {sim_num}: New best! S:{score:.1f}, {info}")
                else: self.high_score_counter = 0 if score < self.best_score else self.high_score_counter # Reset if score drops
                if cfg["early_stopping"] and self.best_score >= cfg["early_stopping_threshold"]:
                    self.high_score_counter += 1
                    if self.high_score_counter >= cfg["early_stopping_stability"]:
                        self.logger.info(f"Early stopping criteria met iter {iter_num}."); await self.llm.emit_message(f"**Stopping early:** Score ({self.best_score:.1f}) stable.")
                        return False # Stop search
            else: self.high_score_counter = 0 # Reset if sim failed

            if self.show_chat_details: await self.llm.emit_message(f"--- Sim {iter_num}.{sim_num} ---\n{sim_msg}\n"); await asyncio.sleep(0.05)

        self.iterations_completed += 1; self.logger.info(f"Finished Iteration {iter_num}.")
        return True # Continue search

    def find_best_final_node(self) -> Optional[Node]:
        """Finds node object matching best_solution content using BFS."""
        if not self.best_solution or not self.root: return None
        q: List[Node] = [self.root]; visited: Set[str] = {self.root.id}
        best_match: Optional[Node] = None; min_diff = float('inf')
        target = str(self.best_solution); target = re.sub(r"^\s*```.*\s*$", "", target, flags=re.DOTALL|re.MULTILINE).strip()
        while q:
            curr = q.pop(0); node_content = str(curr.content); node_content = re.sub(r"^\s*```.*\s*$", "", node_content, flags=re.DOTALL|re.MULTILINE).strip()
            if node_content == target:
                 score_diff = abs(curr.get_average_score() - self.best_score)
                 if best_match is None or score_diff < min_diff: best_match = curr; min_diff = score_diff; self.logger.debug(f"Best node match: N{curr.sequence} (Diff:{score_diff:.3f})");
                 if min_diff < 1e-4: break # Optimization
            for child in curr.children:
                 if child and child.id not in visited: visited.add(child.id); q.append(child)
        if not best_match: self.logger.warning("Could not find node matching best_solution content."); return self.root # Fallback
        return best_match

    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Extracts relevant state for database persistence."""
        if not self.root: self.logger.warning("Get state failed: No root."); return {}
        state = {"version": "0.8.1"} # Update version if format changes
        try:
            state["best_score"] = self.best_score; best_node = self.find_best_final_node()
            state["best_solution_content"] = str(self.best_solution); state["best_solution_summary"] = truncate_text(self.best_solution, 400)
            state["best_node_tags"] = best_node.descriptive_tags if best_node else []
            if self.config.get("use_bayesian_evaluation"): # Priors
                 if isinstance(self.approach_alphas, dict) and isinstance(self.approach_betas, dict):
                     state["approach_priors"] = {"alpha": {k: round(v, 4) for k,v in self.approach_alphas.items()}, "beta": {k: round(v, 4) for k,v in self.approach_betas.items()}}
                 else: state["approach_priors"] = None
            # Collect visited nodes for Top Nodes / Unfit Markers
            nodes: List[Node] = []; q: List[Node] = [self.root]; visited: Set[str] = {self.root.id}
            while q: curr = q.pop(0); nodes.append(curr);
            for child in curr.children:
                if child and child.id not in visited: visited.add(child.id); q.append(child)
            visited_nodes = [n for n in nodes if n.visits > 0]
            # Top Nodes (by score)
            sorted_nodes = sorted(visited_nodes, key=lambda n: n.get_average_score(), reverse=True)
            state["top_nodes"] = [node.node_to_state_dict() for node in sorted_nodes[:3]]
            # Unfit Markers
            thresh_s = self.config.get("unfit_score_threshold", 4.0); thresh_v = self.config.get("unfit_visit_threshold", 3)
            unfit = [{'id':n.id, 'seq':n.sequence, 'summary':truncate_text(n.thought or n.content, 80), 'reason':f"Low score ({n.get_average_score():.1f}<{thresh_s}) / {n.visits} visits", 'tags':n.descriptive_tags}
                     for n in visited_nodes if n.visits >= thresh_v and n.get_average_score() < thresh_s]
            state["unfit_markers"] = unfit[:10] # Limit count
            return state
        except Exception as e: self.logger.error(f"State serialization error: {e}", exc_info=self.debug_logging); return {}

    def get_final_synthesis_context(self) -> Optional[Dict[str, str]]:
         """Prepares context for the final synthesis prompt."""
         if not self.root or not self.best_solution: return None
         best_node = self.find_best_final_node(); path_nodes: List[Node] = []
         curr = best_node;
         while curr: path_nodes.append(curr); curr = curr.parent
         thoughts = [f"- (N{n.sequence} from N{n.parent.sequence}): {n.thought.strip()}" for n in reversed(path_nodes) if n.thought and n.parent]
         path_str = "\n".join(thoughts) or "No development path."
         return { "question_summary": self.question_summary, "initial_analysis_summary": truncate_text(self.root.content, 300),
             "best_score": f"{self.best_score:.1f}", "path_thoughts": path_str,
             "final_best_analysis_summary": truncate_text(self.best_solution, 400), }

    def formatted_output(self) -> str:
        """Generates a detailed, formatted summary for verbose output."""
        cfg = self.config; result = f"# MCTS Final Analysis Summary (Verbose)\n\n"
        try: # Wrap main formatting in try/except
            best_node = self.find_best_final_node(); tags = f"Tags: {best_node.descriptive_tags}" if best_node and best_node.descriptive_tags else "Tags: []"
            result += f"## Best Analysis (Score: {self.best_score:.1f}/10)\n**{tags}**\n\n{str(self.best_solution).strip()}\n"
            # Top Nodes
            result += "\n## Top Nodes & Thoughts\n"; all_nodes: List[Node] = []; q=[self.root]; visited={self.root.id}
            while q: curr=q.pop(0); all_nodes.append(curr);
            for c in curr.children:
                 if c and c.id not in visited: visited.add(c.id); q.append(c)
            visited_nodes = [n for n in all_nodes if n.visits > 0]
            sorted_nodes = sorted(visited_nodes, key=lambda n: n.get_average_score(), reverse=True)
            for node in sorted_nodes[:5]: # Show top 5
                score = node.get_average_score(); score_det = ""
                if cfg["use_bayesian_evaluation"] and node.alpha: score_det=f"(α={node.alpha:.1f},β={node.beta:.1f})"
                elif node.value: score_det=f"(V={node.value:.1f})"
                tags = f"Tags:{node.descriptive_tags}" if node.descriptive_tags else "Tags:[]"
                result += f"### N{node.sequence}: S:{score:.1f}/10 {score_det}\n- **Approach**: {node.approach_type}({node.approach_family}), **Visits**: {node.visits}, **{tags}**\n- **Thought**: {node.thought or '(Root)'}\n"
                if node.is_surprising: result += f"- **Surprising**: Yes ({truncate_text(node.surprise_explanation, 100)})\n"; result += "\n"
            # Explored Path
            result += "\n## Most Explored Path\n"; curr = self.root; path = [curr] if curr else []
            while curr and curr.children:
                best_c = curr.best_child()
                if not best_c or best_c.visits == 0:
                    break
                path.append(best_c)
                curr = best_c
            if len(path) > 1:
                for i, n in enumerate(path): result += f"{'  '*i}{'└─ ' if i==len(path)-1 else '├─ '}N{n.sequence}({n.approach_type}, S:{n.get_average_score():.1f}, V:{n.visits}, T:{n.descriptive_tags})\n"
            else: result += "No significant path explored.\n"
            # Surprising Nodes
            if self.surprising_nodes: result += "\n## Surprising Nodes\n";
            for node in self.surprising_nodes[-5:]: result += f"- **N{node.sequence}** ({node.approach_type}, S:{node.get_average_score():.1f}, T:{node.descriptive_tags}): {truncate_text(node.surprise_explanation.splitlines()[0], 150)}\n"
            # Approach Performance
            result += "\n## Approach Performance\n"; app_data = []
            all_apps = set(self.approach_alphas.keys()) | set(self.approach_scores.keys())
            for app in all_apps:
                if app == "unknown": continue; count = len(self.explored_approaches.get(app,[])); score_str, sort_key = "N/A", -1.0
                if count == 0 and app != "initial": continue
                if cfg["use_bayesian_evaluation"]: alpha, beta = self.approach_alphas.get(app,1.0), self.approach_betas.get(app,1.0);
                if alpha+beta>1e-9: mean_s=alpha/(alpha+beta)*10; score_str=f"S:{mean_s:.2f}(α={alpha:.1f},β={beta:.1f})"; sort_key=mean_s
                else: avg_s = self.approach_scores.get(app); score_str=f"S:{avg_s:.2f}" if avg_s else "S:N/A"; sort_key=avg_s or -1.0
                app_data.append({"n": app, "s": score_str, "c": f" ({count}T)" if count > 0 else "", "k": sort_key})
            sorted_apps = sorted(app_data, key=lambda x: x["k"], reverse=True)
            for d in sorted_apps[:7]: result += f"- **{d['n']}**: {d['s']}{d['c']}\n"
            if len(sorted_apps) > 7: result += f"- ... ({len(sorted_apps)-7} more)\n"
            # Config Used
            result += f"\n## Search Params\n- Iter:{self.iterations_completed}/{cfg['max_iterations']}, Sim/Iter:{cfg['simulations_per_iteration']}, Total Sim:{self.simulations_completed}\n"
            eval_s = "Bayesian" if cfg['use_bayesian_evaluation'] else "Traditional"; sel_s = "Thompson" if cfg['use_bayesian_evaluation'] and cfg['use_thompson_sampling'] else "UCT"
            result += f"- Eval:{eval_s}, Sel:{sel_s}, Exp W:{cfg['exploration_weight']:.2f}\n"
            if cfg['use_bayesian_evaluation']: result += f"- Beta Priors(Init): α={cfg['beta_prior_alpha']:.2f}, β={cfg['beta_prior_beta']:.2f}\n"
            result += f"- EarlyStop:{cfg['early_stopping']}(Th:{cfg['early_stopping_threshold']:.1f}, St:{cfg['early_stopping_stability']})\n- State:{cfg['enable_state_persistence']}, Verbose:{cfg['show_processing_details']}\n"
            # Debug History
            if self.debug_logging and self.debug_history: result += "\n## Debug Log Snippets\n";
            for entry in self.debug_history[-3:]: result += truncate_text(re.sub(r"\n+", "\n", entry).strip(), 200) + "\n---\n"
            return result.strip()
        except Exception as e: self.logger.error(f"Verbose output format error: {e}", exc_info=self.debug_logging); return result + f"\n\n# Error in Verbose Summary:\n{type(e).__name__}: {str(e)}\n"

# ==============================================================================
# Main Pipe Class Definition
# ==============================================================================
class Pipe(LLMInterface): # Implement the interface MCTS needs
    """
    Main Pipe class for Stateful MCTS Analysis in Open WebUI.
    """
    class Valves(BaseModel):
        """Configurable parameters via Open WebUI Valves."""
        MAX_ITERATIONS: int = Field(default=DEFAULT_CONFIG["max_iterations"], title="Max Iterations", ge=1)
        SIMULATIONS_PER_ITERATION: int = Field(default=DEFAULT_CONFIG["simulations_per_iteration"], title="Simulations / Iteration", ge=1)
        MAX_CHILDREN: int = Field(default=DEFAULT_CONFIG["max_children"], title="Max Children / Node", ge=1)
        EXPLORATION_WEIGHT: float = Field(default=DEFAULT_CONFIG["exploration_weight"], title="Exploration Weight (UCT)", ge=0.0)
        USE_THOMPSON_SAMPLING: bool = Field(default=DEFAULT_CONFIG["use_thompson_sampling"], title="Use Thompson Sampling (if Bayesian)")
        FORCE_EXPLORATION_INTERVAL: int = Field(default=DEFAULT_CONFIG["force_exploration_interval"], title="Force Branch Explore Interval (0=off)", ge=0)
        SCORE_DIVERSITY_BONUS: float = Field(default=DEFAULT_CONFIG["score_diversity_bonus"], title="UCT Score Diversity Bonus", ge=0.0)
        USE_BAYESIAN_EVALUATION: bool = Field(default=DEFAULT_CONFIG["use_bayesian_evaluation"], title="Use Bayesian (Beta) Evaluation")
        BETA_PRIOR_ALPHA: float = Field(default=DEFAULT_CONFIG["beta_prior_alpha"], gt=0, title="Bayesian Prior Alpha (>0)")
        BETA_PRIOR_BETA: float = Field(default=DEFAULT_CONFIG["beta_prior_beta"], gt=0, title="Bayesian Prior Beta (>0)")
        USE_SEMANTIC_DISTANCE: bool = Field(default=DEFAULT_CONFIG["use_semantic_distance"], title="Use Semantic Distance (Surprise)")
        SURPRISE_THRESHOLD: float = Field(default=DEFAULT_CONFIG["surprise_threshold"], ge=0.0, le=1.0, title="Surprise Threshold (Semantic)")
        SURPRISE_SEMANTIC_WEIGHT: float = Field(default=DEFAULT_CONFIG["surprise_semantic_weight"], title="Surprise: Semantic Weight", ge=0.0, le=1.0)
        SURPRISE_PHILOSOPHICAL_SHIFT_WEIGHT: float = Field(default=DEFAULT_CONFIG["surprise_philosophical_shift_weight"], title="Surprise: Shift Weight (Thought)", ge=0.0, le=1.0)
        SURPRISE_NOVELTY_WEIGHT: float = Field(default=DEFAULT_CONFIG["surprise_novelty_weight"], title="Surprise: Novelty Weight (Thought)", ge=0.0, le=1.0)
        SURPRISE_OVERALL_THRESHOLD: float = Field(default=DEFAULT_CONFIG["surprise_overall_threshold"], ge=0.0, le=1.0, title="Surprise: Overall Threshold")
        GLOBAL_CONTEXT_IN_PROMPTS: bool = Field(default=DEFAULT_CONFIG["global_context_in_prompts"], title="Use Global Context in Prompts")
        TRACK_EXPLORED_APPROACHES: bool = Field(default=DEFAULT_CONFIG["track_explored_approaches"], title="Track Explored Thought Approaches")
        SIBLING_AWARENESS: bool = Field(default=DEFAULT_CONFIG["sibling_awareness"], title="Add Sibling Context to Prompts")
        MEMORY_CUTOFF: int = Field(default=DEFAULT_CONFIG["memory_cutoff"], title="Memory Cutoff (Top N High Scores)", ge=0)
        EARLY_STOPPING: bool = Field(default=DEFAULT_CONFIG["early_stopping"], title="Enable Early Stopping")
        EARLY_STOPPING_THRESHOLD: float = Field(default=DEFAULT_CONFIG["early_stopping_threshold"], ge=1.0, le=10.0, title="Early Stopping Score Threshold")
        EARLY_STOPPING_STABILITY: int = Field(default=DEFAULT_CONFIG["early_stopping_stability"], ge=1, title="Early Stopping Stability")
        ENABLE_STATE_PERSISTENCE: bool = Field(default=DEFAULT_CONFIG["enable_state_persistence"], title="Enable State Persistence (via DB)")
        UNFIT_SCORE_THRESHOLD: float = Field(default=DEFAULT_CONFIG["unfit_score_threshold"], ge=0.0, le=10.0, title="Unfit Marker Score Threshold")
        UNFIT_VISIT_THRESHOLD: int = Field(default=DEFAULT_CONFIG["unfit_visit_threshold"], ge=1, title="Unfit Marker Min Visits")
        SHOW_PROCESSING_DETAILS: bool = Field(default=DEFAULT_CONFIG["show_processing_details"], title="Show Detailed MCTS Steps in Chat")
        DEBUG_LOGGING: bool = Field(default=DEFAULT_CONFIG["debug_logging"], title="Enable Detailed Debug Logging (Console/Logs)")

    def __init__(self):
        self.type = "manifold"
        self.name = PIPE_NAME
        self.valves = self.Valves()
        self.current_config: Dict[str, Any] = {}
        self.debug_logging: bool = False
        self.__current_event_emitter__: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
        self.__model__: str = ""
        self.__chat_id__: Optional[str] = None
        self.__request_body__: Dict[str, Any] = {}
        # <<< MODIFIED: Added placeholder for the user object >>>
        self.__user__: Optional[Dict] = None # Will store the user dict passed to pipe()
        # Use the pipe-specific logger instance
        self.logger = logging.getLogger(PIPE_LOG_NAME)

    # --- Pipe Interface Methods ---
    def pipes(self) -> List[Dict[str, str]]:
        """Lists available models for Open WebUI."""
        try:
            if hasattr(ollama, 'get_all_models'): ollama.get_all_models() # Refresh if possible
            models = getattr(app.state, "OLLAMA_MODELS", {})
            valid = {k:v for k,v in models.items() if isinstance(v,dict) and "name" in v}
            if not valid: self.logger.warning("No valid Ollama models found."); return [{"id": f"{self.name}-error", "name": f"{self.name} (No models)"}]
            return [{"id": f"{self.name}-{k}", "name": f"{self.name} ({v['name']})"} for k,v in valid.items()]
        except Exception as e: self.logger.error(f"List pipes failed: {e}", exc_info=True); return [{"id": f"{self.name}-error", "name": f"{self.name} (Error: {e})"}]

    def resolve_model(self, body: Optional[Dict[str, Any]] = None) -> str:
        """Resolves base model name from pipe model ID."""
        body_to_use = body or self.__request_body__
        model_id = body_to_use.get("model", "").strip()
        prefix = f"{self.name}-"
        idx = model_id.rfind(prefix)
        if idx != -1: base = model_id[idx + len(prefix):]; return base if base else model_id # Fallback
        return model_id # Assume already base name

    def _resolve_question(self, body: Dict[str, Any]) -> str:
        """Extracts last user message."""
        messages = body.get("messages", [])
        if isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user": return str(msg.get("content", "")).strip()
        return ""

    # --- Event Emitter Helpers ---
    async def progress(self, message: str):
        if self.__current_event_emitter__:
            try:
                 if self.debug_logging: self.logger.debug(f"Progress: {message}")
                 await self.__current_event_emitter__({"type":"status", "data":{"level":"info", "description":str(message), "done":False}})
            except Exception as e: self.logger.error(f"Emit progress error: {e}")
    async def done(self):
        if self.__current_event_emitter__:
            try:
                 if self.debug_logging: self.logger.debug("Sending 'done' status.")
                 await self.__current_event_emitter__({"type":"status", "data":{"level":"info", "description":"Complete", "done":True}})
            except Exception as e: self.logger.error(f"Emit done error: {e}")
    async def emit_message(self, message: str):
        if self.__current_event_emitter__:
            try: await self.__current_event_emitter__({"type": "message", "data": {"content": str(message)}})
            except Exception as e: self.logger.error(f"Emit message error: {e}")

    # --- LLMInterface Implementation (Delegates to llm_interaction utils) ---
    # <<< MODIFIED: Now passes self.__user__ to call_ollama_endpoint >>>
    async def get_completion(self, model: str, messages: List[Dict[str, str]]) -> str:
        """Gets a non-streaming completion from the LLM."""
        response = None
        try:
            model_to_use = model or self.__model__
            if not model_to_use: self.logger.error("LLMIface.get_completion: No model."); return "Error: Model missing."
            payload = {"model": model_to_use, "messages": messages, "stream": False}
            # Pass the stored user object: self.__user__
            response = await call_ollama_endpoint(payload, self.logger, self.__user__, self.debug_logging)
            content = get_response_content(response, self.logger)
            if isinstance(response, dict) and response.get("error"):
                 self.logger.error(f"Non-streaming LLM call failed: {content}")
                 return f"Error: {content}" if content else "Error: Unknown LLM failure."
            return content if content else ""
        except Exception as e: self.logger.error(f"Completion error: {e}", exc_info=self.debug_logging); return f"Error: LLM call failed ({str(e)})."
        finally:
            if response is not None and hasattr(response, 'release') and callable(response.release):
                try: await response.release()
                except Exception as release_err: self.logger.error(f"Non-stream release error: {release_err}")

    # <<< MODIFIED: Now passes self.__user__ to call_ollama_endpoint >>>
    async def get_streaming_completion(self, model: str, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """Gets a streaming completion from the LLM."""
        response = None
        try:
            model_to_use = model or self.__model__
            if not model_to_use: self.logger.error("LLMIface.stream: No model."); yield "Error: Model missing."; return
            payload = {"model": model_to_use, "messages": messages, "stream": True}
            # Pass the stored user object: self.__user__
            response = await call_ollama_endpoint(payload, self.logger, self.__user__, self.debug_logging)
            if isinstance(response, dict) and response.get("error"):
                err_msg = get_response_content(response, self.logger)
                self.logger.error(f"LLM stream init failed: {err_msg}"); yield f"Error: {err_msg}"; return
            if hasattr(response, 'body_iterator'):
                async for chunk_bytes in response.body_iterator:
                    for part in get_chunk_content(chunk_bytes, self.logger, self.debug_logging): yield part
            elif isinstance(response, dict): # Unexpected non-stream
                content = get_response_content(response, self.logger)
                if content: self.logger.warning("Expected stream, got dict. Yielding content."); yield content
                else: self.logger.error(f"Expected stream, got invalid dict: {str(response)[:200]}"); yield "Error: Invalid LLM dict."
            else: self.logger.error(f"Expected stream/dict, got type: {type(response)}."); yield f"Error: Unexpected LLM type ({type(response)})."
        except Exception as e: self.logger.error(f"LLM stream error: {e}", exc_info=self.debug_logging); yield f"Error: {str(e)}"
        finally:
            if response is not None and hasattr(response, 'release') and callable(response.release):
                try: await response.release()
                except Exception as release_err: self.logger.error(f"Stream release error: {release_err}")

    # --- MCTS Specific LLM Wrappers ---
    # These methods internally call self.get_completion or self.get_streaming_completion,
    # which have already been modified to pass the correct user object down.
    async def generate_thought(self, current_analysis: str, context: Dict, config_dict: Dict) -> str:
        try: prompt = GENERATE_THOUGHT_PROMPT.format(**context); return await self.get_completion(self.__model__, [{"role":"user", "content":prompt}])
        except KeyError as e: self.logger.error(f"Gen thought fmt error - Key: {e}. Keys: {list(context.keys())}"); return f"Error: Prompt fmt key {e}."
        except Exception as e: self.logger.error(f"Gen thought error: {e}", exc_info=self.debug_logging); return f"Error: Gen thought failed ({type(e).__name__})."
    async def update_approach(self, original_analysis: str, critique: str, context: Dict, config_dict: Dict) -> str:
        args = context.copy(); args["answer"] = original_analysis; args["improvements"] = critique.strip()
        keys = ["question_summary", "best_answer", "best_score", "current_tags", "previous_best_summary", "unfit_markers_summary"]
        for k in keys: args.setdefault(k, "N/A")
        try:
            prompt = UPDATE_ANALYSIS_PROMPT.format(**args); result = await self.get_completion(self.__model__, [{"role":"user", "content":prompt}])
            if result.startswith("Error:"): self.logger.error(f"Update LLM failed: {result}"); return str(original_analysis)
            clean = re.sub(r"^\s*```.*\s*$", "", result, flags=re.DOTALL|re.MULTILINE).strip() # Clean code blocks
            return clean if clean else str(original_analysis)
        except KeyError as e: self.logger.error(f"Update fmt error - Key: {e}. Keys: {list(args.keys())}"); return str(original_analysis)
        except Exception as e: self.logger.error(f"Update error: {e}", exc_info=self.debug_logging); return str(original_analysis)
    async def evaluate_answer(self, analysis_to_evaluate: str, context: Dict, config_dict: Dict) -> int:
        args = context.copy(); args["answer_to_evaluate"] = analysis_to_evaluate
        keys = ["question_summary", "best_answer", "best_score", "current_tags", "previous_best_summary", "unfit_markers_summary"]
        for k in keys: args.setdefault(k, "N/A")
        try:
            prompt = EVALUATE_ANALYSIS_PROMPT.format(**args); result = await self.get_completion(self.__model__, [{"role":"user", "content":prompt}])
            if result.startswith("Error:"): self.logger.warning(f"Eval LLM failed: {result}. Default 5."); return 5
            m_strict = re.search(r"^\s*([1-9]|10)\s*$", result.strip())
            if m_strict: return int(m_strict.group(1))
            m_relax = re.search(r"\b([1-9]|10)\b", result)
            if m_relax: self.logger.info(f"Eval relaxed parse: {m_relax.group(1)} from '{result}'"); return int(m_relax.group(1))
            self.logger.warning(f"Could not parse eval score: '{result}'. Default 5."); return 5
        except KeyError as e: self.logger.error(f"Eval fmt error - Key: {e}. Keys: {list(args.keys())}"); return 5
        except Exception as e: self.logger.error(f"Eval error: {e}", exc_info=self.debug_logging); return 5

    # --- Intent Handling ---
    async def _classify_intent(self, text: str) -> str:
        self.logger.debug(f"Classifying intent: {truncate_text(text)}")
        default = "ANALYZE_NEW"
        try:
            prompt = INTENT_CLASSIFIER_PROMPT.format(raw_input_text=text)
            res = await self.get_completion(self.__model__, [{"role":"user", "content":prompt}])
            if res.startswith("Error:"): self.logger.error(f"Intent LLM failed: {res}"); await self.emit_message("Warn: Intent failed."); return default
            valid = ["ANALYZE_NEW", "CONTINUE_ANALYSIS", "ASK_LAST_RUN_SUMMARY", "ASK_PROCESS", "ASK_CONFIG", "GENERAL_CONVERSATION"]
            clean = re.sub(r"[.,!?;:]$", "", res.strip().upper().split(maxsplit=1)[0])
            if clean in valid: self.logger.info(f"Intent: {clean}"); return clean
            else: # Heuristic fallback
                self.logger.warning(f"Intent unexpected: '{res}'. Using heuristic.")
                if any(kw in text.lower() for kw in ["continue", "elaborate", "further", "what about"]): return "CONTINUE_ANALYSIS"
                return default
        except Exception as e: self.logger.error(f"Intent classify failed: {e}", exc_info=self.debug_logging); await self.emit_message("Warn: Intent error."); return default
    async def _handle_ask_process(self): self.logger.info("Intent: ASK_PROCESS"); explanation = ASK_PROCESS_EXPLANATION.format(db_file_name=os.path.basename(DB_FILE)); await self.emit_message(f"**About My Process:**\n{explanation}"); await self.done()
    async def _handle_ask_config(self):
        self.logger.info("Intent: ASK_CONFIG")
        try:
            config_str = json.dumps(self.current_config, indent=2)
            await self.emit_message(f"**Current Config:**\n\n{config_str}\n")
        except Exception as e:
            self.logger.error(f"Config emit error: {e}")
            await self.emit_message("Err: Could not show config.")
        await self.done()
    async def _handle_ask_last_run_summary(self, state: Optional[Dict]):
        self.logger.info("Intent: ASK_LAST_RUN_SUMMARY")
        if not state:
            await self.emit_message("No saved results found for this chat.")
            await self.done()
            return
        try:
            summary = f"**Last Run Summary:**\n- Score: {state.get('best_score','N/A'):.1f}/10\n- Tags: {', '.join(state.get('best_node_tags',[])) or 'N/A'}\n- Summary: {state.get('best_solution_summary','N/A')}\n"
        except Exception as e:
            self.logger.error(f"Basic summary format error: {e}")
            await self.emit_message("Error: Could not format basic summary.")
            await self.done()
            return
        try:
            priors = state.get("approach_priors");
            if priors and isinstance(priors.get("alpha"),dict):
                means={}; alphas=priors['alpha']; betas=priors.get('beta',{})
                for app,a in alphas.items(): b=betas.get(app,1.0); a,b=max(1e-9,a),max(1e-9,b);
                if a+b>1e-9: means[app]=(a/(a+b))*10
                if means: top=sorted(means.items(),key=lambda i:i[1],reverse=True)[:3]; summary+=f"- Prefers: {', '.join([f'{a}({s:.1f})' for a,s in top])}...\n"
            unfit=state.get("unfit_markers",[]); summary+=f"- Unfit Areas: {len(unfit)} (e.g., '{unfit[0]['summary']}' due to {unfit[0]['reason']})\n" if unfit else "- Unfit Areas: None\n"
            await self.emit_message(summary)
        except Exception as e:
            self.logger.error(f"Summary format error: {e}")
            await self.emit_message("Err: Could not show summary.")
        await self.done()
    async def _handle_general_conversation(self, user_input: str):
        self.logger.info("Intent: GENERAL_CONVERSATION")
        try:
            prompt = GENERAL_CONVERSATION_PROMPT.format(user_input=user_input)
            response = await self.get_completion(self.__model__, [{"role":"user","content":prompt}])
            fallback = "Noted. How can I help with analysis?"
            await self.emit_message(response if not response.startswith("Error:") else fallback)
        except Exception as e:
            self.logger.error(f"General conv error: {e}")
            await self.emit_message("Error processing that. How can I help?")
        await self.done()
    # --- Main Pipe Execution Logic ---
    async def _initialize_run(self, body: Dict) -> Tuple[bool, str, Optional[str]]:
        """Initializes pipe state, config, model, input."""
        self.__request_body__ = body; self.current_config = DEFAULT_CONFIG.copy()
        self.__model__ = self.resolve_model(body)
        if not self.__model__: await self.emit_message("Err: Model not identified."); await self.done(); return False, "Err: No model.", None
        text = self._resolve_question(body)
        if not text: await self.emit_message("Err: No input text."); await self.done(); return False, "Err: No input.", None
        self.__chat_id__ = body.get("chat_id");
        if not self.__chat_id__: self.logger.warning("chat_id missing. State persistence disabled.")
        self.logger.debug("Applying Valve settings...")
        try: # Apply Valves
            req_valves = body.get("valves"); self.valves = self.Valves(**req_valves) if req_valves and isinstance(req_valves, dict) else self.Valves()
            valve_dict = self.valves.model_dump();
            for k_up, v in valve_dict.items():
                 k_low = k_up.lower();
                 if k_low in self.current_config: self.current_config[k_low] = v
            self.debug_logging = self.current_config["debug_logging"]; log_level = logging.DEBUG if self.debug_logging else logging.INFO
            setup_logger(PIPE_LOG_NAME, log_level, LOG_FORMAT); setup_logger(MCTS_LOG_NAME, log_level, LOG_FORMAT) # Update levels
            # Sanitize config values
            self.current_config["beta_prior_alpha"]=max(1e-9, self.current_config["beta_prior_alpha"])
            self.current_config["beta_prior_beta"]=max(1e-9, self.current_config["beta_prior_beta"])
            self.logger.info("Valve settings applied.")
        except Exception as e: self.logger.error(f"Valve error: {e}. Using defaults.", exc_info=True); await self.emit_message(f"Warn: Valve error ({e}). Using defaults."); self.current_config=DEFAULT_CONFIG.copy(); self.debug_logging=self.current_config["debug_logging"]
        return True, text, self.__chat_id__

    async def _determine_intent_and_load_state(self, text: str) -> Tuple[str, Optional[Dict], bool]:
        """Classifies intent and loads state if needed."""
        intent = await self._classify_intent(text); loaded_state = None
        enabled = self.current_config.get("enable_state_persistence", True)
        if not self.__chat_id__ and enabled: self.logger.warning("State enabled but chat_id missing. Disabling."); enabled = False
        if enabled and self.__chat_id__ and intent in ["CONTINUE_ANALYSIS", "ASK_LAST_RUN_SUMMARY"]:
            try:
                loaded_state = load_mcts_state(DB_FILE, self.__chat_id__, self.logger)
                if loaded_state and loaded_state.get("version") != "0.8.1": # VERSION CHECK
                    self.logger.warning(f"State version mismatch ({loaded_state.get('version')}). Discarding."); await self.emit_message("Warn: State incompatible."); loaded_state = None;
                    if intent=="CONTINUE_ANALYSIS": intent="ANALYZE_NEW"
                elif not loaded_state and intent=="CONTINUE_ANALYSIS": await self.emit_message("Info: No prior state. Starting new."); intent="ANALYZE_NEW"
            except Exception as e: self.logger.error(f"State load error {self.__chat_id__}: {e}", exc_info=True); await self.emit_message("Warn: State load error."); loaded_state=None;
            if intent=="CONTINUE_ANALYSIS": intent="ANALYZE_NEW"
        return intent, loaded_state, enabled

    async def _handle_intent(self, intent: str, text: str, state: Optional[Dict]) -> bool:
        """Dispatches to intent handlers. Returns True if handled, False if MCTS should run."""
        handlers = { "ASK_PROCESS": self._handle_ask_process, "ASK_CONFIG": self._handle_ask_config,
                     "ASK_LAST_RUN_SUMMARY": lambda s=self: s._handle_ask_last_run_summary(state),
                     "GENERAL_CONVERSATION": lambda s=self: s._handle_general_conversation(text) }
        if intent in handlers: await handlers[intent](); return True
        elif intent in ["ANALYZE_NEW", "CONTINUE_ANALYSIS"]: return False # Proceed to MCTS
        else: self.logger.error(f"Unhandled intent: {intent}"); await self.emit_message("Err: Unhandled intent."); await self.done(); return True

    async def _run_mcts_analysis(self, intent: str, text: str, state: Optional[Dict]) -> Optional[MCTS]:
        """Runs the MCTS analysis loop."""
        await self.emit_message(f'# {self.name} v0.8.1\n*Analyzing:* "{truncate_text(text, 100)}" *using* `{self.__model__}`.')
        await self.emit_message("🚀 Continuing Analysis..." if intent == "CONTINUE_ANALYSIS" and state else "🚀 Starting Analysis...")
        if self.current_config.get("show_processing_details"): await self.emit_message("*(Verbose mode on)*")
        log_params = {k:v for k,v in self.current_config.items() if k in ["max_iterations","simulations_per_iteration","use_bayesian_evaluation","early_stopping"]}
        self.logger.info(f"--- Run Params ---\nIntent:{intent}, State:{bool(state)}, Chat:{self.__chat_id__}, Cfg:{json.dumps(log_params)}")

        await self.progress("Generating initial analysis..."); initial_text = await self.get_completion(self.__model__, [{"role":"user", "content":INITIAL_ANALYSIS_PROMPT.format(question=text)}])
        if initial_text.startswith("Error:"): self.logger.error(f"Initial analysis failed: {initial_text}"); await self.emit_message(f"Err: Initial analysis failed: {initial_text}"); await self.done(); return None
        initial_text = re.sub(r"^\s*```.*\s*$", "", initial_text, flags=re.DOTALL|re.MULTILINE).strip() # Clean
        if not initial_text: self.logger.error("Initial analysis empty."); await self.emit_message("Err: Initial analysis empty."); await self.done(); return None
        await self.emit_message(f"\n## Initial Analysis\n{initial_text}\n\n***\n"); await asyncio.sleep(0.1)

        await self.progress("Initializing MCTS...")
        try: mcts_inst = MCTS(llm_interface=self, question=text, mcts_config=self.current_config, initial_analysis_content=initial_text, initial_state=state if intent=="CONTINUE_ANALYSIS" else None, model_body=self.__request_body__)
        except Exception as e: self.logger.error(f"MCTS init failed: {e}", exc_info=self.debug_logging); await self.emit_message(f"Err: MCTS init failed: {e}"); await self.done(); return None

        keep_searching = True; max_iter = self.current_config["max_iterations"]; sims_p_iter = self.current_config["simulations_per_iteration"]
        for i in range(max_iter):
             if not keep_searching: break
             iter_num = i + 1; await self.progress(f"Running Iteration {iter_num}/{max_iter}...")
             score_before = mcts_inst.best_score
             try: keep_searching = await mcts_inst.search(sims_p_iter)
             except Exception as e: self.logger.error(f"MCTS Search Iter {iter_num} Error: {e}", exc_info=self.debug_logging); await self.emit_message(f"Warn: Search error iter {iter_num}."); keep_searching=False
             if self.current_config.get("show_processing_details"): # Iteration Summary
                 best_n = mcts_inst.find_best_final_node(); summary = f"\n**--- Iter {iter_num} Summary ---**\n- Best Score: {mcts_inst.best_score:.1f}/10{' ✨' if mcts_inst.best_score > score_before else ''}\n";
                 if best_n: summary += f"- Best Node: {best_n.sequence} (T:{best_n.descriptive_tags})\n"; summary += "-------------------------------\n"; await self.emit_message(summary)
             await asyncio.sleep(0.05)
        self.logger.info("MCTS iterations finished."); await self.emit_message("\n🏁 **MCTS Finished.** Generating output..."); return mcts_inst

    async def _finalize_run(self, mcts_inst: Optional[MCTS], init_text: str, state_enabled: bool):
        """Generates final output, synthesis, saves state."""
        if not mcts_inst: self.logger.error("Finalize: MCTS instance missing."); await self.emit_message("Err: MCTS failed."); await self.done(); return
        final_text = str(mcts_inst.best_solution) if mcts_inst.best_solution else init_text
        final_text = re.sub(r"^\s*.*\s*$", "", final_text, flags=re.DOTALL|re.MULTILINE).strip()
        if not final_text: final_text = init_text # Ensure not empty

        if self.current_config.get("show_processing_details"): # Verbose Output
            await self.progress("Generating verbose summary..."); summary = mcts_inst.formatted_output(); await self.emit_message(summary)
        else: # Quiet Output
            await self.progress("Extracting best analysis..."); best_node = mcts_inst.find_best_final_node(); tags = f"Tags: {best_node.descriptive_tags}" if best_node and best_node.descriptive_tags else "Tags: []"; summary = f"## Best Analysis (Score: {mcts_inst.best_score:.1f}/10)\n**{tags}**\n\n{final_text}\n"; await self.emit_message(summary)

        await self.progress("Generating final synthesis..."); synth_ctx = mcts_inst.get_final_synthesis_context()
        if synth_ctx:
            try:
                synth_text = await self.get_completion(self.__model__, [{"role":"user", "content":FINAL_SYNTHESIS_PROMPT.format(**synth_ctx)}])
                if synth_text.startswith("Error:"):
                    self.logger.error(f"Synthesis failed: {synth_text}")
                    await self.emit_message("\n***\n## Synthesis\nWarn: Synthesis failed.")
                else:
                    synth_text = re.sub(r"^\s*.*\s*$", "", synth_text, flags=re.DOTALL|re.MULTILINE).strip()
                    await self.emit_message(f"\n***\n## Final Synthesis\n{synth_text or '(Empty synthesis.)'}")
            except KeyError as e: self.logger.error(f"Synthesis fmt error - Key: {e}. Keys: {list(synth_ctx.keys())}"); await self.emit_message("\n***\n## Synthesis\nErr: Synth fmt failed.")
            except Exception as e: self.logger.error(f"Synthesis error: {e}", exc_info=self.debug_logging); await self.emit_message("\n***\n## Synthesis\nErr: Synth failed.")
        else: await self.emit_message("\n***\n## Synthesis\nErr: Synth context failed.")

        if state_enabled and self.__chat_id__: # Save State
            await self.progress("Saving state...")
            try:
                state_to_save = mcts_inst.get_state_for_persistence()
                if state_to_save:
                    save_mcts_state(DB_FILE, self.__chat_id__, state_to_save, self.logger)
                else:
                    self.logger.warning("Empty state generated, not saving.")
            except Exception as e:
                self.logger.error(f"Save state error {self.__chat_id__}: {e}", exc_info=True)
                await self.emit_message("Warn: Save state failed.")
        await self.done(); self.logger.info(f"Pipe '{self.name}' finished analysis intent.")

    # --- Main Pipe Entry Point ---
    # <<< MODIFIED: Now captures the 'user' object into self.__user__ >>>
    async def pipe(
        self, body: Dict, user: Optional[Dict], emitter: Callable, task=None
    ) -> Union[str, None, AsyncGenerator[str, None]]:
        self.__current_event_emitter__ = emitter
        self.__user__ = user # <<< ADDED: Store the user object passed by the framework
        mcts_inst: Optional[MCTS] = None; init_text: str = ""
        try:
            success, text, chat_id = await self._initialize_run(body)
            if not success: return text # Return error string
            intent, state, state_enabled = await self._determine_intent_and_load_state(text)
            intent_handled = await self._handle_intent(intent, text, state)
            if intent_handled: return None # Handlers call done()

            # Title Gen Task (Simple Example)
            if task == TASKS.TITLE_GENERATION:
                 self.logger.info(f"Handling TITLE task: {truncate_text(text)}"); title = await self.get_completion(self.__model__, [{"role":"user", "content": f"Title for: {text}"}]); await self.done(); return f"Title: {truncate_text(title, 70)}" if not title.startswith("Error:") else title

            # Run MCTS Analysis
            mcts_inst = await self._run_mcts_analysis(intent, text, state)
            init_text = mcts_inst.root.content if mcts_inst and mcts_inst.root else ""
            await self._finalize_run(mcts_inst, init_text, state_enabled)
            return None # Success
        except Exception as e:
            self.logger.error(f"FATAL Pipe Error: {e}", exc_info=True); err_msg=f"\n\n**FATAL ERROR:**\n```\n{type(e).__name__}: {e}\n```\nCheck logs.";
            try: await self.emit_message(err_msg)
            except: pass;
            try: await self.done()
            except: pass;
            return f"Error: Pipe failed ({type(e).__name__}). Check logs."
        finally:
            if self.debug_logging: self.logger.debug("Pipe cleanup...");
            self.__current_event_emitter__=None;
            self.__user__ = None # Clear stored user
            gc.collect();
            if self.debug_logging: self.logger.debug("Pipe cleanup complete.")

# ==============================================================================
# END OF SCRIPT
# ==============================================================================