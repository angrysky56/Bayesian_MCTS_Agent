
# -*- coding: utf-8 -*-
"""
title: advanced_mcts_stateful (Single File - Pipe Class Structure - Corrected Calls)
version: 0.9.5
author: angrysky56
# ... (keep other frontmatter) ...

description: >
  Stateful Advanced Bayesian MCTS v{SCRIPT_VERSION} (Single File Version):
  - **Fix**: Renamed main class back to `Pipe` to match previously working v0.8.0 structure, overriding previous assumption based on error logs.
  - **Fix**: Corrected calls to top-level utility functions (DB, LLM, text utils) that were refactored from class methods, ensuring arguments like loggers, config, etc., are passed correctly.
  - Maintaining single-file structure.
  - Structure: Imports -> Logging -> Constants -> Utility Functions -> Core Logic Classes (Node, MCTS) -> Prompts -> Main `Pipe` class.

Requires:
 numpy, pydantic, scikit-learn

################################################################################
# IMPORTANT: For Open WebUI integration to work correctly:                   #
#            1. This file MUST be named exactly `advanced_mcts_stateful.py`  #
#            2. It MUST be placed in the `pipes` directory.                  #
#            3. The main class MUST be named `Pipe`.                         # <<-- Updated Requirement based on v0.8.0
#            4. Ensure DB_FILE path below is correct and writable.           #
################################################################################
"""

# ==============================================================================
# Core Imports & Basic Config
# ==============================================================================
import logging
import sys
import os
import asyncio
import gc
import json
import math
import random
import re
import sqlite3
from collections import Counter
from datetime import datetime
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

# --- Basic Logging Setup ---
log_format = "%(asctime)s - %(name)s [%(levelname)s] %(message)s"
pipe_log_name = "pipe.advanced_mcts_stateful"
mcts_log_name = "mcts.advanced_mcts_stateful"
logger = logging.getLogger(pipe_log_name)
mcts_logger = logging.getLogger(mcts_log_name)
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    handler_pipe = logging.StreamHandler(sys.stderr)
    formatter_pipe = logging.Formatter(log_format)
    handler_pipe.setFormatter(formatter_pipe)
    logger.addHandler(handler_pipe)
    logger.propagate = False
    logger.info(f"Basic handler configured for {pipe_log_name}")
if not mcts_logger.hasHandlers():
    mcts_logger.setLevel(logging.INFO)
    handler_mcts = logging.StreamHandler(sys.stderr)
    formatter_mcts = logging.Formatter(log_format)
    handler_mcts.setFormatter(formatter_mcts)
    mcts_logger.addHandler(handler_mcts)
    mcts_logger.propagate = False
    logger.info(f"Basic handler configured for {mcts_log_name}")
logger.info(f"Script loading {__name__}...")

# --- Required External Dependency Imports ---
try:
    import numpy as np
    from numpy.random import beta as beta_sample

    logger.info("Successfully imported numpy.")
except ImportError as e:
    logger.critical(f"FATAL: numpy import failed: {e}. Pipe cannot load.")
    raise
try:
    from pydantic import BaseModel, Field, field_validator, ValidationError

    logger.info("Successfully imported pydantic.")
except ImportError as e:
    logger.critical(f"FATAL: pydantic import failed: {e}. Pipe cannot load.")
    raise

# --- Optional Dependency: scikit-learn ---
try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
    CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS) + [
        "analysis", "however", "therefore", "furthermore", "perspective", "node", "mcts",
        "score", "approach", "concept", "system", "model", "text", "data", "result",
        "based", "consider", "provide", "evaluate", "generate", "update",
    ]
    logger.info("Optional: scikit-learn found. TF-IDF enabled.")
except ImportError:
    SKLEARN_AVAILABLE = False
    TfidfVectorizer, cosine_similarity, CUSTOM_STOP_WORDS = None, None, None
    logger.warning("Optional: scikit-learn not found. Using Jaccard similarity.")

# --- Open WebUI Specific Imports ---
try:
    from fastapi import Request, Response  # Needed for call_ollama_endpoint structure
    from open_webui.constants import TASKS  # Needed for title generation task check
    # from open_webui.utils.auth import AdminUser # Use this if not using mock

    OPENWEBUI_IMPORTS_AVAILABLE = True
    logger.info("Checked Open WebUI components (fastapi Request/Response, TASKS).")
except ImportError as e:
    OPENWEBUI_IMPORTS_AVAILABLE = False
    Request, Response, TASKS = None, None, None
    logger.error(f"Failed to import Open WebUI components: {e}. Functionality limited.")


# ==============================================================================
# Configuration Constants (Define Early)
# ==============================================================================
PIPE_NAME = "advanced_mcts_stateful"
SCRIPT_VERSION = "0.9.5" # <<< Updated version

# --- Database Configuration ---
# !!! IMPORTANT: Set this path correctly for your system !!!
DB_FILE = "/home/ty/Repositories/sqlite-db/NEXUS_PRIME.db" # <<< YOUR ACTUAL PATH HERE
# !!! IMPORTANT: Set this path correctly for your system !!!

# --- Safety Check for DB_FILE Path ---
if "/path/to/your/database" in DB_FILE: # Default path check
    logger.warning(f"DB_FILE path ('{DB_FILE}') seems default. Please update.")
else:
    db_dir = os.path.dirname(DB_FILE)
    if not os.path.exists(db_dir):
        try:
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"Created database directory: {db_dir}")
        except OSError as e:
            logger.error(f"Failed to create DB directory '{db_dir}': {e}. State persistence WILL fail.")
    elif not os.access(db_dir, os.W_OK):
        logger.warning(f"DB dir '{db_dir}' may not be writable. State persistence might fail.")

# --- Default MCTS Configuration ---
DEFAULT_CONFIG = {
    "max_iterations": 1,
    "simulations_per_iteration": 10,
    "max_children": 10,
    "exploration_weight": 3.0,
    "use_thompson_sampling": True,
    "force_exploration_interval": 4,
    "score_diversity_bonus": 0.7,
    "use_bayesian_evaluation": True,
    "beta_prior_alpha": 1.0,
    "beta_prior_beta": 1.0,
    "relative_evaluation": False,
    "unfit_score_threshold": 4.0,
    "unfit_visit_threshold": 3,
    "use_semantic_distance": True,
    "surprise_threshold": 0.66,
    "surprise_semantic_weight": 0.6,
    "surprise_philosophical_shift_weight": 0.3,
    "surprise_novelty_weight": 0.3,
    "surprise_overall_threshold": 0.9,
    "global_context_in_prompts": True,
    "track_explored_approaches": True,
    "sibling_awareness": True,
    "memory_cutoff": 5,
    "early_stopping": True,
    "early_stopping_threshold": 10.0,
    "early_stopping_stability": 2,
    "enable_state_persistence": True,
    "show_processing_details": True,
    "debug_logging": False,
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
    "variant": [], "initial": [], "unknown": [],
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
    "initial": {"family": "general"}, "unknown": {"family": "general"},
}

# ==============================================================================
# Utility Functions & Classes (Define before Pipe, Node, MCTS)
# ==============================================================================

# --- Text Processing Utility ---
def truncate_text(text: Optional[str], max_length: int = 8196) -> str:
    """Truncates text cleanly, removing markdown code blocks and trying to break at spaces."""
    if not text:
        return ""
    text = str(text).strip()
    # More robust removal of code blocks (handles various languages and ``` variations)
    text = re.sub(r"^\s*```[\s\S]*?```\s*$", "", text, flags=re.MULTILINE) # Full block removal
    text = re.sub(r"^\s*```(.*?)\n", "", text, flags=re.IGNORECASE | re.MULTILINE) # Leading ``` line
    text = re.sub(r"\n```\s*$", "", text, flags=re.MULTILINE).strip() # Trailing ``` line

    if len(text) <= max_length:
        return text
    # Find the last space within the limit
    last_space = text.rfind(" ", 0, max_length)
    return text[:last_space] + "..." if last_space != -1 else text[:max_length] + "..."

# --- Semantic Distance Utility ---
def calculate_semantic_distance(
    text1: Optional[str],
    text2: Optional[str],
    logger_instance: logging.Logger, # Needs logger passed
    use_tfidf: bool = SKLEARN_AVAILABLE,
) -> float:
    """Calculates semantic distance (1 - similarity) using TF-IDF or Jaccard."""
    if not text1 or not text2:
        return 1.0
    text1, text2 = str(text1), str(text2)

    if use_tfidf and SKLEARN_AVAILABLE and TfidfVectorizer and cosine_similarity:
        try:
            vectorizer = TfidfVectorizer(stop_words=CUSTOM_STOP_WORDS, max_df=0.9, min_df=1)
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            if tfidf_matrix.shape < 2 or tfidf_matrix.shape == 0:
                raise ValueError(f"TF-IDF matrix gen failed (shape {tfidf_matrix.shape}).")
            # Ensure indices are valid before accessing
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return 1.0 - max(0.0, min(1.0, similarity)) # Clamp similarity
        except Exception as e:
            logger_instance.warning(f"TF-IDF failed ({e}), fallback Jaccard.", exc_info=False) # Less verbose on common error

    # Jaccard Fallback
    try:
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        if not words1 and not words2: return 0.0 # Both empty
        if not words1 or not words2: return 1.0  # One empty
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        if union == 0: return 0.0 # Should be caught by above checks
        jaccard_similarity = intersection / union
        return 1.0 - jaccard_similarity
    except Exception as fallback_e:
        logger_instance.error(f"Jaccard similarity fallback failed: {fallback_e}")
        return 1.0 # Default to max distance on error


# --- Approach Classification Utility ---
def classify_approach(
    thought: Optional[str],
    taxonomy: Dict[str, List[str]],
    metadata: Dict[str, Dict[str, str]],
    random_state: Any, # Needs random state passed
    logger_instance: logging.Logger, # Needs logger passed
) -> Tuple[str, str]:
    """Classifies a thought text based on keywords against a taxonomy."""
    approach_type, approach_family = "variant", "general" # Default
    if not thought or not isinstance(thought, str):
        return approach_type, approach_family

    thought_lower = thought.lower()
    scores = {app: sum(1 for kw in kws if kw in thought_lower) for app, kws in taxonomy.items()}
    positive_scores = {app: score for app, score in scores.items() if score > 0}

    if positive_scores:
        max_score = max(positive_scores.values())
        # Handle ties by random choice among best
        best_types = [app for app, score in positive_scores.items() if score == max_score]
        approach_type = random_state.choice(best_types)

    # Get family from metadata, default to general
    approach_family = metadata.get(approach_type, {}).get("family", "general")

    logger_instance.debug(f"Classified thought '{truncate_text(thought, 50)}...' as: {approach_type} (Family: {approach_family})")
    return approach_type, approach_family


# --- Mock Admin User Class ---
class AdminUserMock:
    """Mocks the AdminUser structure expected by some internal OWB functions."""
    def __init__(self, role: str = "admin"):
        self.role = role


# --- Database Utilities ---
def get_db_connection(db_file_path: str, logger_instance: logging.Logger) -> Optional[sqlite3.Connection]:
    """Establishes connection to the SQLite DB and ensures table exists."""
    conn = None
    try:
        db_dir = os.path.dirname(db_file_path)
        if db_dir: os.makedirs(db_dir, exist_ok=True) # Ensure dir exists

        conn = sqlite3.connect(db_file_path, timeout=10) # Increased timeout
        conn.execute("PRAGMA journal_mode=WAL;") # Recommended for concurrency
        # Ensure table schema matches expected fields
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mcts_state (
                chat_id TEXT PRIMARY KEY,
                last_state_json TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mcts_state_timestamp ON mcts_state (timestamp);")
        conn.commit()
        logger_instance.debug(f"SQLite DB connection OK: {db_file_path}")
        return conn
    except sqlite3.Error as e:
        logger_instance.error(f"SQLite connection/setup error for '{db_file_path}': {e}", exc_info=True)
    except Exception as e:
        logger_instance.error(f"Unexpected DB connection error for '{db_file_path}': {e}", exc_info=True)

    # Ensure connection is closed if acquisition failed after opening
    if conn: conn.close()
    return None

def save_mcts_state(db_file_path: str, chat_id: str, state: Dict[str, Any], logger_instance: logging.Logger):
    """Saves the MCTS state JSON for a given chat_id."""
    if not chat_id:
        logger_instance.warning("Save MCTS state failed: chat_id missing.")
        return
    if not isinstance(state, dict) or not state:
        logger_instance.warning(f"Save MCTS state failed: invalid/empty state provided for chat '{chat_id}'.")
        return

    conn = None
    try:
        state_json = json.dumps(state) # Serialize state first
        conn = get_db_connection(db_file_path, logger_instance)
        if not conn:
            logger_instance.error(f"Save state failed for chat '{chat_id}': DB connection could not be established.")
            return

        with conn: # Use context manager for commit/rollback
            conn.execute(
                "INSERT OR REPLACE INTO mcts_state (chat_id, last_state_json, timestamp) VALUES (?, ?, ?)",
                (chat_id, state_json, datetime.now())
            )
        logger_instance.info(f"Saved MCTS state for chat_id: {chat_id}")

    except json.JSONDecodeError as e:
         logger_instance.error(f"Error serializing MCTS state for chat '{chat_id}': {e}", exc_info=True)
    except sqlite3.Error as e:
        logger_instance.error(f"SQLite error saving state for chat '{chat_id}': {e}", exc_info=True)
    except Exception as e:
        logger_instance.error(f"Unexpected error saving MCTS state for chat '{chat_id}': {e}", exc_info=True)
    finally:
        if conn: conn.close()


def load_mcts_state(db_file_path: str, chat_id: str, logger_instance: logging.Logger) -> Optional[Dict[str, Any]]:
    """Loads the most recent MCTS state JSON for a given chat_id."""
    if not chat_id:
        logger_instance.warning("Load MCTS state failed: chat_id missing.")
        return None

    conn = None
    state_dict = None
    try:
        conn = get_db_connection(db_file_path, logger_instance)
        if not conn:
            logger_instance.error(f"Load state failed for chat '{chat_id}': DB connection could not be established.")
            return None

        with conn: # Use context manager
            cursor = conn.cursor()
            cursor.execute(
                "SELECT last_state_json FROM mcts_state WHERE chat_id = ? ORDER BY timestamp DESC LIMIT 1",
                (chat_id,)
            )
            result = cursor.fetchone()

            if result and result:
                try:
                    loaded_data = json.loads(result)
                    if isinstance(loaded_data, dict):
                        state_dict = loaded_data # Assign if valid dict
                        logger_instance.info(f"Loaded MCTS state for chat_id: {chat_id}")
                    else:
                        logger_instance.warning(f"Loaded state for chat '{chat_id}' is not a dictionary (type: {type(loaded_data)}). Discarding.")
                except json.JSONDecodeError as json_err:
                    logger_instance.error(f"Error decoding loaded state JSON for chat '{chat_id}': {json_err}")
            else:
                logger_instance.info(f"No previous MCTS state found for chat_id: {chat_id}")

    except sqlite3.Error as e:
        logger_instance.error(f"SQLite error loading state for chat '{chat_id}': {e}", exc_info=True)
    except Exception as e:
        logger_instance.error(f"Unexpected error loading MCTS state for chat '{chat_id}': {e}", exc_info=True)
    finally:
        if conn: conn.close()

    return state_dict


# --- LLM Interaction Utilities ---
async def call_ollama_endpoint(
    payload: Dict[str, Any],
    logger_instance: logging.Logger,
    user_object: Optional[Union[Dict, "AdminUserMock"]], # Needs user object
    debug_logging: bool = False
) -> Union[Dict, Any]:
    """Calls the internal OWB ollama endpoint."""
    try: # Defer imports until needed
        from open_webui.main import app
        import open_webui.routers.ollama as ollama_router
        if not app or not ollama_router:
            raise ImportError("OWB app/ollama router missing or not initialized")
    except Exception as e:
        logger_instance.critical(f"Ollama call failed: Missing required Open WebUI components: {e}")
        # Return an error structure consistent with the API
        return {"error": True, "choices": [{"message": {"role": "assistant", "content": "Internal Error: Missing required components."}}]}

    try:
        # Mock FastAPI request
        async def receive():
            return {"type": "http.request", "body": json.dumps(payload).encode("utf-8")}

        if Request is None: raise ImportError("FastAPI Request class missing")
        mock_request = Request(
            scope={
                "type": "http", "headers": [], "method": "POST", "scheme": "http",
                "server": ("localhost", 8080), "path": "/api/ollama/generate", "query_string": b"",
                "client": ("127.0.0.1", 8080), "app": app, # Pass the actual app instance
            },
            receive=receive,
        )

        if debug_logging:
            # Log payload summary carefully, avoid logging sensitive data if any
            log_payload = payload.copy()
            if 'messages' in log_payload:
                 log_payload['messages'] = f"<{len(log_payload['messages'])} messages>" # Summarize messages
            logger_instance.debug(f"Calling internal ollama. Payload summary: {log_payload}")

        # Ensure user_object is suitable (mock or potentially real OWB user)
        final_user_object = AdminUserMock() # Default to mock if needed
        if isinstance(user_object, dict):
             # If a dict is passed, assume it's compatible or create mock from it
             final_user_object = AdminUserMock(role=user_object.get("role", "admin"))
        elif isinstance(user_object, AdminUserMock): # Or if already a mock
             final_user_object = user_object
        # else: logger_instance.warning(f"Unexpected user_object type: {type(user_object)}. Using default mock.")

        # Make the actual call to the OWB router function
        response = await ollama_router.generate_openai_chat_completion(
            request=mock_request, form_data=payload, user=final_user_object
        )
        return response

    except Exception as e:
        err_str = str(e)
        err_type = type(e).__name__
        # Provide more specific error messages if possible
        if "400" in err_str and "Model" in err_str and "not found" in err_str:
            error_msg = f"Error: LLM Model '{payload.get('model')}' not found."
        elif "Connection refused" in err_str or "Connect call failed" in err_str:
             error_msg = f"Error: Connection to Ollama backend failed."
        else:
            error_msg = f"Error: LLM communication failed ({err_type}: {str(err_str)[:100]}...)."

        logger_instance.error(f"Ollama internal call error: {err_type}: {err_str}", exc_info=debug_logging)
        # Return consistent error structure
        return {"error": True, "choices": [{"message": {"role": "assistant", "content": error_msg}}]}


def get_chunk_content(
    chunk_bytes: bytes,
    logger_instance: logging.Logger, # Needs logger
    debug_logging: bool = False
) -> List[str]:
    """Parses content from a streaming chunk."""
    parts = []
    try:
        chunk_str = chunk_bytes.decode("utf-8")
        for line in chunk_str.splitlines():
            line = line.strip()
            # Check for data line and exclude DONE signal
            if line.startswith("data: ") and line != "data: [DONE]":
                json_str = line[len("data: "):]
                try:
                    data = json.loads(json_str)
                    # Safely extract content
                    content = data.get("choices", [{}]).get("delta", {}).get("content")
                    # Ensure content is a non-empty string before appending
                    if isinstance(content, str) and content:
                        parts.append(content)
                except (json.JSONDecodeError, IndexError, KeyError, TypeError) as e:
                    if debug_logging:
                        logger_instance.warning(f"Stream chunk JSON parse/structure error: {e} - Data: {json_str[:100]}...")
    except UnicodeDecodeError as e:
         logger_instance.error(f"Stream chunk decode error: {e} - Bytes: {chunk_bytes[:100]}...")
    except Exception as e:
        logger_instance.error(f"Error processing stream chunk bytes: {e}", exc_info=debug_logging)
    return parts


def get_response_content(
    response: Union[Dict, Any],
    logger_instance: logging.Logger # Needs logger
) -> str:
    """Extracts content from a non-streaming response, handling errors."""
    try:
        if not isinstance(response, dict):
            logger_instance.warning(f"get_response_content received non-dict type: {type(response)}")
            return ""

        # Check for explicit error structure first
        if response.get("error"):
            try:
                # Attempt to get detailed error message from standard structure
                error_content = str(response.get("choices", [{}]).get("message", {}).get("content", "Unknown LLM Error"))
            except (IndexError, KeyError, TypeError):
                # Fallback if structure is unexpected
                err_detail = response.get("error")
                error_content = f"{err_detail}" if isinstance(err_detail, (str, bool, int, float)) else "Unknown LLM Error Detail"
            # Ensure error message starts with "Error:" for clarity
            return error_content if error_content.startswith("Error:") else f"Error: {error_content}"

        # Check for standard successful response structure
        elif "choices" in response and isinstance(response["choices"], list) and response["choices"]:
            try:
                message = response["choices"].get("message", {})
                content = message.get("content", "")
                return str(content) # Return content or empty string
            except (IndexError, KeyError, TypeError) as e:
                logger_instance.warning(f"Error extracting content from success structure: {e} - Response: {str(response)[:200]}")
                return ""
        else:
            # If structure is neither known error nor known success
            logger_instance.warning(f"Unexpected response dictionary structure: {str(response)[:200]}")
            return ""

    except Exception as e:
        logger_instance.error(f"Response content extraction error: {e}", exc_info=True)
        return ""


# ==============================================================================
# Core Logic Classes (LLMInterface, Node, MCTS)
# ==============================================================================

# --- Abstract LLM Interface ---
class LLMInterface:
    """Defines the interface expected by MCTS for interacting with the LLM."""
    async def generate_thought( self, current_analysis: str, context: Dict, config: Dict ) -> str: raise NotImplementedError
    async def update_approach( self, original_analysis: str, critique: str, context: Dict, config: Dict ) -> str: raise NotImplementedError
    async def evaluate_answer( self, analysis_to_evaluate: str, context: Dict, config: Dict ) -> Union[int, str]: raise NotImplementedError
    async def get_completion(self, model: str, messages: List[Dict[str, str]]) -> str: raise NotImplementedError
    async def progress(self, message: str): raise NotImplementedError
    async def emit_message(self, message: str): raise NotImplementedError
    def resolve_model(self, body: dict) -> str: raise NotImplementedError


# --- Node Class ---
class Node(BaseModel):
    id: str = Field(default_factory=lambda: f"node_{random.randbytes(4).hex()}")
    content: str = ""
    parent: Optional["Node"] = Field(default=None, exclude=True) # Avoid serializing parent backref
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
    alpha: Optional[float] = None # Bayesian alpha
    beta: Optional[float] = None # Bayesian beta
    value: Optional[float] = None # Non-Bayesian cumulative score
    descriptive_tags: List[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    # Validator to handle optional fields correctly during initialization
    @field_validator("parent", "children", mode="before")
    @classmethod
    def _validate_optional_fields(cls, v): return v

    def __init__(self, **data: Any):
        super().__init__(**data)
        # Set max_children and use_bayesian from instance data or defaults
        self.max_children = data.get("max_children", DEFAULT_CONFIG["max_children"])
        self.use_bayesian_evaluation = data.get("use_bayesian_evaluation", DEFAULT_CONFIG["use_bayesian_evaluation"])

        # Initialize Bayesian or standard scoring parameters
        if self.use_bayesian_evaluation:
            prior_alpha = data.get("alpha", DEFAULT_CONFIG["beta_prior_alpha"])
            prior_beta = data.get("beta", DEFAULT_CONFIG["beta_prior_beta"])
            self.alpha = max(1e-9, float(prior_alpha)) # Ensure positive
            self.beta = max(1e-9, float(prior_beta))  # Ensure positive
            self.value = None # Ensure value is None if Bayesian
        else:
            self.value = float(data.get("value", 0.0)) # Init cumulative score
            self.alpha = None
            self.beta = None

    def add_child(self, child: "Node"):
        if child not in self.children:
            self.children.append(child)
            child.parent = self # Set parent reference

    def fully_expanded(self) -> bool:
        """Checks if the node has reached its maximum number of children."""
        # Count only non-None children if list might contain Nones temporarily (shouldn't usually)
        valid_children_count = sum(1 for c in self.children if isinstance(c, Node))
        return valid_children_count >= self.max_children

    def get_bayesian_mean(self) -> float:
        """Calculates the mean of the Beta distribution (score between 0 and 1)."""
        if self.use_bayesian_evaluation and self.alpha is not None and self.beta is not None:
            alpha_safe, beta_safe = max(1e-9, self.alpha), max(1e-9, self.beta)
            denominator = alpha_safe + beta_safe
            return (alpha_safe / denominator) if denominator > 1e-18 else 0.5 # Default to 0.5 if sum is near zero
        return 0.5 # Default if not Bayesian or invalid state

    def get_average_score(self) -> float:
        """Returns the node's score (1-10 scale). Uses Bayesian mean or simple average."""
        if self.use_bayesian_evaluation:
            # Scale Bayesian mean to score (simple linear scaling)
            # Mean 0 -> Score 1, Mean 0.5 -> Score 5.5, Mean 1 -> Score 10
            # Score = 1 + Mean * 9
            # return 1.0 + self.get_bayesian_mean() * 9.0
            # OR simpler: Scale to
            return self.get_bayesian_mean() * 10.0
        else:
            # Standard average score calculation
            return (self.value / max(1, self.visits)) if self.visits > 0 and self.value is not None else 5.0 # Default to midpoint 5.0

    def thompson_sample(self) -> float:
        """Samples from the Beta distribution for Thompson Sampling (returns value between 0 and 1)."""
        if self.use_bayesian_evaluation and self.alpha is not None and self.beta is not None:
            alpha_safe, beta_safe = max(1e-9, self.alpha), max(1e-9, self.beta)
            try:
                sample = float(beta_sample(alpha_safe, beta_safe))
                return max(0.0, min(1.0, sample)) # Clamp sample to
            except Exception as e:
                # Use MCTS logger if available, fallback to default logger
                (logging.getLogger(mcts_log_name) or logging.getLogger()).warning(
                    f"Thompson Sampling failed N{self.sequence} (α={alpha_safe:.2f}, β={beta_safe:.2f}): {e}. Fallback mean."
                )
                return self.get_bayesian_mean() # Fallback to mean on error
        # If not Bayesian, Thompson Sampling isn't applicable in this way. Return mean as proxy.
        return self.get_bayesian_mean() # Or maybe return 0.5? Mean seems better.

    def best_child(self) -> Optional["Node"]:
        """Selects the best child based on visits (primary) and score (tie-breaker)."""
        valid_children = [c for c in self.children if isinstance(c, Node)]
        if not valid_children: return None

        try:
            # Find max visits among valid children
            max_visits = max(child.visits for child in valid_children if isinstance(child.visits, int))
        except ValueError: # Handles case where maybe no children have visits yet or list is empty after filter
            return None # Or random.choice(valid_children)? None seems safer.

        # Get all children with max visits
        most_visited_children = [child for child in valid_children if child.visits == max_visits]

        if len(most_visited_children) == 1:
            return most_visited_children
        elif len(most_visited_children) > 1:
            # Tie-break using average score (higher is better)
            try:
                return max(most_visited_children, key=lambda c: c.get_average_score())
            except Exception as e:
                 (logging.getLogger(mcts_log_name) or logging.getLogger()).warning(f"Best child tie-break error: {e}.")
                 # Fallback to first most visited on error
                 return most_visited_children
        elif valid_children: # If no most visited (e.g., all 0 visits), pick one randomly? Seems unlikely here.
             return random.choice(valid_children) # Fallback? Or return None?
        else:
            return None

    def node_to_json(self, max_depth: int = 3, current_depth: int = 0) -> Dict[str, Any]:
        """Creates a nested dictionary representation for debugging/visualization."""
        score = self.get_average_score()
        valid_children = [c for c in self.children if isinstance(c, Node)]

        node_dict: Dict[str, Any] = {
            "id": self.id,
            "sequence": self.sequence,
            "content_summary": truncate_text(self.content, 150),
            "visits": self.visits,
            "approach_type": self.approach_type,
            "approach_family": self.approach_family,
            "is_surprising": self.is_surprising,
            "thought_summary": truncate_text(self.thought, 100),
            "tags": self.descriptive_tags[:], # Copy tags list
            "score": round(score, 2) if score is not None else None,
            "children_count": len(valid_children),
            "children": [],
        }
        # Add scoring details based on mode
        if self.use_bayesian_evaluation and self.alpha is not None and self.beta is not None:
            node_dict.update({
                "alpha": round(self.alpha, 3),
                "beta": round(self.beta, 3),
                "mean_score_0_1": round(self.get_bayesian_mean(), 3),
            })
        elif not self.use_bayesian_evaluation and self.value is not None:
            node_dict["cumulative_value"] = round(self.value, 2)

        # Recursively add children up to max_depth
        if current_depth < max_depth:
            node_dict["children"] = [child.node_to_json(max_depth, current_depth + 1) for child in valid_children]

        return node_dict

    def node_to_state_dict(self) -> Dict[str, Any]:
        """Creates a flat dictionary representation suitable for state persistence."""
        score = self.get_average_score()
        state_dict: Dict[str, Any] = {
            "id": self.id,
            "sequence": self.sequence,
            "content_summary": truncate_text(self.content, 250), # Slightly more context for state
            "visits": self.visits,
            "approach_type": self.approach_type,
            "approach_family": self.approach_family,
            "thought": self.thought, # Include full thought
            "tags": self.descriptive_tags[:], # Copy tags
            "score": round(score, 2) if score is not None else None,
            "is_surprising": self.is_surprising,
        }
        # Include priors/value based on mode
        if self.use_bayesian_evaluation and self.alpha is not None and self.beta is not None:
            state_dict.update({
                "alpha": round(self.alpha, 4), # More precision for state
                "beta": round(self.beta, 4),
            })
        elif not self.use_bayesian_evaluation and self.value is not None:
            state_dict["value"] = round(self.value, 2) # Save cumulative value

        return state_dict


# --- MCTS Class ---
class MCTS:
    """Manages the Monte Carlo Tree Search process."""
    def __init__(
        self,
        llm_interface: LLMInterface, # This will be the Pipe instance
        question: str,
        mcts_config: Dict[str, Any],
        initial_analysis_content: str,
        initial_state: Optional[Dict[str, Any]] = None,
        model_body: Optional[Dict[str, Any]] = None, # Pass the request body for model resolution
    ):
        self.llm = llm_interface
        self.config = mcts_config
        self.question = question
        self.question_summary = self._summarize_question(question)
        self.model_body = model_body or {} # Store for use in _generate_tags_for_node
        self.debug_logging = self.config.get("debug_logging", False)
        self.show_chat_details = self.config.get("show_processing_details", True)

        # Use the dedicated MCTS logger
        self.logger = logging.getLogger(mcts_log_name)
        self.logger.setLevel(logging.DEBUG if self.debug_logging else logging.INFO)
        # Get pipe logger too if needed (e.g., for DB calls done via pipe instance)
        self.pipe_logger = logging.getLogger(pipe_log_name)
        self.pipe_logger.setLevel(logging.DEBUG if self.debug_logging else logging.INFO)

        self.loaded_initial_state = initial_state
        self.node_sequence = 0 # Reset sequence for each MCTS instance
        self.iterations_completed = 0
        self.simulations_completed = 0
        self.high_score_counter = 0 # For early stopping stability
        self.random_state = random.Random() # Instance specific random state

        # Tracking structures
        self.thought_history: List[str] = [] # Log of thoughts/actions
        self.debug_history: List[str] = [] # More detailed logs if debug enabled
        self.surprising_nodes: List[Node] = [] # Nodes marked as surprising
        self.explored_approaches: Dict[str, List[str]] = {} # Track thoughts per approach type
        self.explored_thoughts: Set[str] = set() # Track unique thoughts generated
        self.memory: Dict[str, Any] = {"depth": 0, "branches": 0, "high_scoring_nodes": []} # Simple memory

        # Log MCTS start
        mcts_start_log = f"# MCTS Init v{SCRIPT_VERSION}\nQ: {self.question_summary}\nState Loaded: {bool(initial_state)}\n"
        self.thought_history.append(mcts_start_log)

        # Init Approach Priors/Scores
        cfg = self.config
        prior_alpha = max(1e-9, float(cfg.get("beta_prior_alpha", 1.0)))
        prior_beta = max(1e-9, float(cfg.get("beta_prior_beta", 1.0)))
        self.approach_alphas: Dict[str, float] = {} # For Bayesian mode
        self.approach_betas: Dict[str, float] = {} # For Bayesian mode
        self.approach_scores: Dict[str, float] = {} # For non-Bayesian mode (simple avg)

        # Load priors from state if available and valid
        loaded_priors = initial_state.get("approach_priors") if initial_state else None
        all_approach_keys = list(APPROACH_TAXONOMY.keys()) # Includes initial, variant, unknown

        if cfg.get("use_bayesian_evaluation") and loaded_priors and isinstance(loaded_priors.get("alpha"), dict) and isinstance(loaded_priors.get("beta"), dict):
            try:
                # Load from state, ensuring values are valid floats > 0
                self.approach_alphas = { k: max(1e-9, float(v)) for k, v in loaded_priors["alpha"].items() if isinstance(v, (int, float)) }
                self.approach_betas = { k: max(1e-9, float(v)) for k, v in loaded_priors["beta"].items() if isinstance(v, (int, float)) }
                # Ensure all known approaches have an entry, defaulting if missing in state
                for k in all_approach_keys:
                    self.approach_alphas.setdefault(k, prior_alpha)
                    self.approach_betas.setdefault(k, prior_beta)
                self.logger.info(f"Loaded {len(self.approach_alphas)} Bayesian approach priors from state.")
            except (ValueError, TypeError) as e:
                self.logger.error(f"Error loading state priors: {e}. Using defaults.")
                # Fallback to defaults if loading failed
                for k in all_approach_keys:
                    self.approach_alphas[k] = prior_alpha
                    self.approach_betas[k] = prior_beta
        else:
             # Initialize with defaults if not Bayesian or no valid state priors
            for k in all_approach_keys:
                self.approach_alphas[k] = prior_alpha # Still init even if not used, simpler logic
                self.approach_betas[k] = prior_beta
                self.approach_scores[k] = 5.0 # Default average score

        # Init Best Score Tracking
        self.best_score: float = 0.0
        self.best_solution: str = initial_analysis_content # Start with initial analysis
        self.previous_best_solution_content: Optional[str] = None # Track previous best for context

        if initial_state:
            try:
                loaded_score = initial_state.get("best_score")
                # Load score if valid number
                self.best_score = float(loaded_score) if isinstance(loaded_score, (int, float)) else 0.0
                # Load previous best content if available (used for context)
                loaded_content = initial_state.get("best_solution_content")
                self.previous_best_solution_content = loaded_content if isinstance(loaded_content, str) and loaded_content else None
                self.logger.info(f"Initialized best score ({self.best_score:.2f}) and previous content tracker from state.")
            except Exception as e:
                self.logger.error(f"Error processing state best score/content: {e}. Using defaults.")
                self.best_score = 0.0 # Reset on error

        # Create Root Node
        try:
            self.root: Optional[Node] = Node(
                content=initial_analysis_content,
                sequence=self.get_next_sequence(),
                parent=None,
                max_children=cfg["max_children"],
                use_bayesian_evaluation=cfg["use_bayesian_evaluation"],
                alpha=prior_alpha, # Root starts with default priors
                beta=prior_beta,
                value=0.0, # Initial value for non-Bayesian
                approach_type="initial",
                approach_family="general",
            )
            if not self.root: raise RuntimeError("Root node creation failed unexpectedly.")
        except Exception as e:
            self.logger.critical(f"FATAL: Root node initialization failed: {e}", exc_info=True)
            raise # Cannot proceed without root

        # Load Unfit Markers
        self.unfit_markers: List[Dict[str, Any]] = []
        if initial_state:
            markers = initial_state.get("unfit_markers")
            if isinstance(markers, list):
                # Basic validation: ensure markers are dictionaries
                self.unfit_markers = [m for m in markers if isinstance(m, dict)]
                self.logger.info(f"Loaded {len(self.unfit_markers)} unfit markers from state.")
            elif markers is not None: # Log if key exists but isn't a list
                 self.logger.warning(f"Loaded 'unfit_markers' is not a list (type: {type(markers)}). Ignoring.")

        # Ensure logger levels are consistent after config load
        new_log_level = logging.DEBUG if self.debug_logging else logging.INFO
        self.logger.setLevel(new_log_level)
        self.pipe_logger.setLevel(new_log_level)
        for handler in self.logger.handlers + self.pipe_logger.handlers:
             handler.setLevel(new_log_level)

        self.logger.info("MCTS instance initialized.")


    def get_next_sequence(self) -> int:
        """Increments and returns the next node sequence ID."""
        self.node_sequence += 1
        return self.node_sequence

    def _summarize_question(self, text: str, max_words: int = 50) -> str:
        """Generates a simple summary of the input question/text."""
        if not text or not isinstance(text, str): return "N/A"
        try:
            words = re.findall(r'\b\w+\b', text)
            summary = " ".join(words[:max_words])
            return summary + ("..." if len(words) > max_words else "") if words else text.strip() # Handle empty text case
        except Exception as e:
            self.logger.error(f"Question summary generation failed: {e}.")
            # Fallback to simple truncation if regex fails
            return truncate_text(text.strip(), max_words * 7) # Approx chars

    def export_tree_as_json(self, max_depth: int = 3) -> Dict[str, Any]:
        """Exports the current MCTS tree structure as JSON (up to max_depth)."""
        if not self.root:
            return {"error": "MCTS export failed: Root node missing."}
        try:
            return self.root.node_to_json(max_depth=max_depth)
        except Exception as e:
            self.logger.error(f"Tree JSON export failed: {e}", exc_info=self.debug_logging)
            return {"error": f"Tree export failed: {type(e).__name__}"}

    def get_context_for_node(self, node: Node) -> Dict[str, str]:
        """Assembles context information for LLM prompts related to a specific node."""
        if not isinstance(node, Node):
            self.logger.error("get_context_for_node called with invalid node.")
            return {"error": "Invalid node provided"}

        cfg = self.config
        context: Dict[str, Any] = {
            "question_summary": self.question_summary,
            "best_answer": truncate_text(str(self.best_solution), 300), # Use current best solution
            "best_score": f"{self.best_score:.1f}",
            "current_answer": truncate_text(node.content, 300),
            "current_sequence": str(node.sequence),
            "current_approach": node.approach_type or "N/A",
            "current_tags": ", ".join(node.descriptive_tags) or "None",
            "tree_depth": str(self.memory.get("depth", 0)),
            "branches": str(self.memory.get("branches", 0)),
            # Placeholders for loaded state context
            "previous_best_summary": "N/A",
            "unfit_markers_summary": "None",
            "learned_approach_summary": "Default priors/scores",
            # Placeholders for dynamic context
            "explored_approaches": "None yet.",
            "high_scoring_examples": "None yet.",
            "sibling_approaches": "None.",
        }

        # --- Add Context from Loaded State (if available) ---
        if self.loaded_initial_state and isinstance(self.loaded_initial_state, dict):
            # Previous best summary
            prev_summary = self.loaded_initial_state.get("best_solution_summary")
            context["previous_best_summary"] = truncate_text(prev_summary, 200) if isinstance(prev_summary, str) and prev_summary else "N/A"

            # Unfit markers summary
            unfit = self.loaded_initial_state.get("unfit_markers", [])
            context["unfit_markers_summary"] = ('; '.join([f"'{m.get('summary', '?')}' ({m.get('reason', '?')})" for m in unfit[:5] if isinstance(m, dict)]) + ("..." if len(unfit) > 5 else "")) if isinstance(unfit, list) and unfit else "None"

            # Learned approach summary (Bayesian)
            priors = self.loaded_initial_state.get("approach_priors")
            if priors and isinstance(priors.get("alpha"), dict) and isinstance(priors.get("beta"), dict):
                means = {}
                alphas = priors["alpha"]
                betas = priors["beta"]
                for app, alpha in alphas.items():
                    beta = betas.get(app, 1.0)
                    try:
                        a_f, b_f = max(1e-9, float(alpha)), max(1e-9, float(beta))
                        denominator = a_f + b_f
                        # Calculate mean score 0-10
                        means[app] = (a_f / denominator * 10.0) if denominator > 1e-9 else -1 # Mark invalid as -1
                    except (ValueError, TypeError):
                        means[app] = -1 # Mark error as -1
                # Filter out invalid scores and non-informative types
                valid_means = { k: v for k, v in means.items() if v >= 0 and k not in ["initial", "variant", "unknown"] }
                # Get top 3 based on score
                top = sorted(valid_means.items(), key=lambda item: item, reverse=True)[:3]
                context["learned_approach_summary"] = (f"Favors: {', '.join([f'{a} ({s:.1f})' for a, s in top])}" + ("..." if len(valid_means) > 3 else "")) if top else "Default priors"
            elif priors: # If priors key exists but format is wrong
                context["learned_approach_summary"] = "Default priors (invalid state format)"

        # --- Add Dynamic Context from Current Run ---
        try:
            # Explored Approaches Summary
            if cfg.get("track_explored_approaches", True) and self.explored_approaches:
                lines = []
                # Sort approaches based on current performance (Bayesian mean or avg score)
                def sort_key_func(k):
                    if cfg.get("use_bayesian_evaluation"):
                        a = self.approach_alphas.get(k, 1.0)
                        b = self.approach_betas.get(k, 1.0)
                        return -(a / (a + b + 1e-9)) # Sort descending by mean
                    else:
                        return -self.approach_scores.get(k, -1.0) # Sort descending by score

                sorted_apps = sorted((k for k in self.explored_approaches.keys() if k not in ["initial", "variant", "unknown"]), key=sort_key_func)

                for app in sorted_apps:
                    thoughts = self.explored_approaches[app]
                    count = len(thoughts)
                    if count == 0: continue # Skip if no thoughts recorded
                    score_txt = ""
                    # Use current run's scores/priors for context
                    if cfg.get("use_bayesian_evaluation"):
                        a = self.approach_alphas.get(app, cfg.get("beta_prior_alpha", 1.0))
                        b = self.approach_betas.get(app, cfg.get("beta_prior_beta", 1.0))
                        mean_score = (a / (a + b) * 10.0) if (a + b) > 1e-9 else -1
                        score_txt = f"(S:{mean_score:.1f}, α:{a:.1f}, β:{b:.1f}, N={count})" if mean_score >= 0 else f"(Err, N={count})"
                    else: # Non-Bayesian
                        avg_score = self.approach_scores.get(app, 5.0) # Use current avg score
                        score_txt = f"(AvgS:{avg_score:.1f}, N={count})"
                    # Get last 1 or 2 thought examples
                    samples = "; ".join([f"'{truncate_text(t, 40)}'" for t in thoughts[-min(2, count):]])
                    lines.append(f"- {app} {score_txt}: {samples}")

                if lines: context["explored_approaches"] = "\n".join(["Explored Summary:"] + lines[:7]) # Limit length

            # High Scoring Examples Summary
            if self.memory.get("high_scoring_nodes"):
                # Format: Score, Content Summary, Approach, Thought Summary
                context["high_scoring_examples"] = "\n".join(
                    ["Top Examples (Score, Approach, Thought -> Summary):"] +
                    [f"- S:{s:.1f} ({a}): '{truncate_text(t, 50)}' -> '{truncate_text(c, 60)}'"
                     for s, c, a, t in self.memory["high_scoring_nodes"]]
                )

            # Sibling Awareness Context
            if cfg.get("sibling_awareness", True) and node.parent and len(node.parent.children) > 1:
                siblings = [s for s in node.parent.children if isinstance(s, Node) and s != node and s.visits > 0]
                if siblings:
                    # Sort siblings by sequence for consistent order
                    sorted_siblings = sorted(siblings, key=lambda x: x.sequence)
                    lines = ["Sibling Thoughts:"] + [
                        f'- N{s.sequence} "{truncate_text(s.thought, 50)}" (S:{s.get_average_score():.1f}, T:{s.descriptive_tags})'
                        for s in sorted_siblings
                    ]
                    context["sibling_approaches"] = "\n".join(lines[:5]) # Limit length

        except Exception as e:
            self.logger.error(f"Error generating dynamic MCTS context for N{node.sequence}: {e}", exc_info=self.debug_logging)
            # Set error messages for affected context parts
            context.update({ k: "Error generating summary." for k in ["explored_approaches", "high_scoring_examples", "sibling_approaches"] if k not in context })

        # Ensure all context values are strings before returning
        return {k: str(v) if v is not None else "" for k, v in context.items()}

    def _calculate_uct(self, node: Node, parent_visits: int) -> float:
        """Calculates the UCT score for a node, incorporating bonuses and penalties."""
        cfg = self.config
        if node.visits == 0: return float("inf") # Prioritize unvisited

        # Exploitation Term (normalized score 0-1)
        exploit_score_0_1 = node.get_bayesian_mean() if cfg.get("use_bayesian_evaluation") else max(0.0, min(1.0, (node.get_average_score() - 1.0) / 9.0))

        # Exploration Term
        exploration_bonus = 0.0
        if parent_visits > 0 and node.visits > 0:
            # Standard UCT exploration formula
             exploration_bonus = cfg.get("exploration_weight", 1.41) * math.sqrt(math.log(parent_visits + 1e-6) / node.visits)
        elif cfg.get("exploration_weight", 1.41) > 0: # Boost if parent_visits is 0 but exploration is on
             exploration_bonus = cfg.get("exploration_weight", 1.41) * 1.5 # Initial exploration boost

        # Unfit Penalty (based on markers from loaded state or current run)
        unfit_penalty = 0.0
        # Check if this node matches any known unfit markers (unless it's surprising)
        if any(isinstance(m, dict) and (m.get("id") == node.id or m.get("seq") == node.sequence) for m in getattr(self, "unfit_markers", [])) and not node.is_surprising:
             unfit_penalty = -100.0 # Strong penalty to avoid re-exploring known bad paths
             if self.debug_logging: self.logger.debug(f"Applying unfit penalty to N{node.sequence}")

        # Surprise Bonus
        surprise_bonus = 0.3 if node.is_surprising else 0.0 # Simple fixed bonus

        # Score Diversity Bonus (relative to siblings)
        diversity_bonus = 0.0
        diversity_weight = cfg.get("score_diversity_bonus", 0.0)
        if diversity_weight > 0 and node.parent and len(node.parent.children) > 1:
            my_score_0_1 = exploit_score_0_1 # Use normalized score
            sibling_scores_0_1 = [
                (s.get_bayesian_mean() if cfg.get("use_bayesian_evaluation") else max(0.0, min(1.0, (s.get_average_score() - 1.0) / 9.0)))
                for s in node.parent.children if isinstance(s, Node) and s != node and s.visits > 0
            ]
            if sibling_scores_0_1:
                avg_sibling_score_0_1 = sum(sibling_scores_0_1) / len(sibling_scores_0_1)
                # Bonus proportional to distance from average sibling score
                diversity_bonus = diversity_weight * abs(my_score_0_1 - avg_sibling_score_0_1)

        # --- Final UCT Calculation ---
        uct_score = exploit_score_0_1 + exploration_bonus + surprise_bonus + diversity_bonus + unfit_penalty

        # Safety check for non-finite results
        if not math.isfinite(uct_score):
            self.logger.warning(f"UCT calculation for N{node.sequence} resulted in non-finite value ({uct_score}). Returning 0.")
            return 0.0

        if self.debug_logging:
            self.logger.debug(
                f"UCT N{node.sequence}: Score={uct_score:.3f} = "
                f"Epl={exploit_score_0_1:.3f} + Exp={exploration_bonus:.3f} + "
                f"Sr={surprise_bonus:.2f} + Div={diversity_bonus:.2f} + Pen={unfit_penalty:.1f}"
            )
        return uct_score

    def _collect_non_leaf_nodes(self, node: Optional[Node], non_leaf_nodes: List[Node], max_depth: int, current_depth: int = 0):
        """Helper to recursively find nodes that are not leaves and not fully expanded within a depth limit."""
        if node is None or current_depth > max_depth: return
        # A node is a candidate for forced exploration if it has children but isn't full
        if node.children and not node.fully_expanded():
            non_leaf_nodes.append(node)
        # Recurse into children
        for child in node.children:
            if isinstance(child, Node):
                self._collect_non_leaf_nodes(child, non_leaf_nodes, max_depth, current_depth + 1)

    async def select(self) -> Optional[Node]:
        """Selects a node for expansion or simulation using UCT or Thompson Sampling."""
        cfg = self.config
        node = self.root
        path: List[Node] = []
        select_log: List[str] = ["### Select Log:"]

        if not isinstance(node, Node):
            self.logger.error("Select Error: Root node is invalid.")
            return None
        path.append(node)
        select_log.append(f"- Start Root N{node.sequence} (V:{node.visits})")

        # --- Forced Exploration Branch ---
        force_interval = cfg.get("force_exploration_interval", 0)
        if force_interval > 0 and self.simulations_completed > 0 and self.simulations_completed % force_interval == 0 and self.memory.get("depth", 0) > 1:
            candidate_nodes: List[Node] = []
            # Limit search depth for candidates to avoid always picking shallow nodes
            max_force_depth = max(1, self.memory.get("depth", 0) // 2)
            self._collect_non_leaf_nodes(self.root, candidate_nodes, max_depth=max_force_depth)
            if candidate_nodes:
                selected_node = self.random_state.choice(candidate_nodes)
                select_log.append(f"- FORCE EXPLORE (Sim {self.simulations_completed}) -> N{selected_node.sequence}.")
                self.logger.info(f"Select: Forced exploration to node N{selected_node.sequence}.")
                # Log the forced path
                forced_path_nodes: List[Node] = []
                curr: Optional[Node] = selected_node
                while curr: forced_path_nodes.append(curr); curr = curr.parent
                forced_path_str = " -> ".join(f"N{n.sequence}" for n in reversed(forced_path_nodes))
                self.thought_history.append(f"### Select (Forced)\nPath: {forced_path_str}\n")
                return selected_node # Return the randomly chosen non-leaf node
            else:
                select_log.append(f"- FORCE EXPLORE triggered, but no suitable candidates found up to depth {max_force_depth}.")

        # --- Standard Selection Loop ---
        while True:
            # If current node has no children, it's a leaf - select it.
            if not node.children:
                select_log.append(f"- Stop N{node.sequence}: Leaf node.")
                break

            valid_children = [c for c in node.children if isinstance(c, Node)]
            if not valid_children:
                select_log.append(f"- Stop N{node.sequence}: Node has empty children list or only None entries.")
                break # Cannot proceed

            # If there are unvisited children, select one randomly.
            unvisited_children = [c for c in valid_children if c.visits == 0]
            if unvisited_children:
                selected_child = self.random_state.choice(unvisited_children)
                node = selected_child
                path.append(node)
                select_log.append(f"- Select UNVISITED N{node.sequence}.")
                break # Stop selection, this node will be simulated/expanded

            # If all children visited, use selection strategy (Thompson or UCT).
            parent_visits = node.visits
            selected_child = None
            strategy_used = "None"

            # --- Thompson Sampling (if enabled and Bayesian) ---
            use_ts = cfg.get("use_bayesian_evaluation") and cfg.get("use_thompson_sampling", True)
            if use_ts:
                strategy_used = "Thompson Sampling"
                # Get valid samples from children
                ts_samples = [ (child, sample) for child in valid_children if math.isfinite(sample := child.thompson_sample()) ]
                if ts_samples:
                    # Select child with the highest sample value
                    selected_child, best_ts_sample = max(ts_samples, key=lambda item: item)
                    select_log.append(f"- TS ({len(ts_samples)} children): N{selected_child.sequence} selected (Sample:{best_ts_sample:.3f})")
                else:
                    self.logger.warning(f"Thompson Sampling failed for all children of N{node.sequence}. Fallback to UCT.")
                    select_log.append(f"- TS failed for all children. Fallback UCT.")
                    use_ts = False # Force UCT fallback

            # --- UCT (if not Thompson or TS failed) ---
            if not use_ts:
                strategy_used = "UCT"
                # Get valid UCT scores for children
                uct_scores = [ (child, score) for child in valid_children if math.isfinite(score := self._calculate_uct(child, parent_visits)) ]
                if uct_scores:
                     # Select child with the highest UCT score
                    selected_child, best_uct_score = max(uct_scores, key=lambda item: item)
                    select_log.append(f"- UCT ({len(uct_scores)} children): N{selected_child.sequence} selected (Score:{best_uct_score:.3f})")
                else:
                    # This should ideally not happen if children exist
                    self.logger.error(f"CRITICAL SELECT FAILURE: UCT failed for all children of N{node.sequence}.")
                    select_log.append(f"- !! UCT failed for all children. Selection cannot proceed.")
                    # Stay at the current node; maybe it can be expanded? Otherwise, simulation will happen here.
                    break # Stop selection loop

            # Descend to the selected child
            if selected_child:
                node = selected_child
                path.append(node)
                select_log.append(f"- Descend to N{node.sequence} using {strategy_used}.")
            else:
                # Should be caught by UCT/TS failure checks above
                select_log.append(f"- Selection loop terminated at N{node.sequence} (no child selected - unexpected).")
                break

            # If the selected node isn't fully expanded, stop selection here (it will be expanded next).
            if not node.fully_expanded():
                select_log.append(f"- Stop at N{node.sequence} (node is not fully expanded).")
                break
            # Otherwise, continue the loop from the newly selected node

        # --- Log Final Selection Path ---
        path_str = " -> ".join([f"N{n.sequence}(V:{n.visits}, S:{n.get_average_score():.1f})" for n in path])
        self.thought_history.append(f"### Select Path\n{path_str}\n")
        # Update max depth reached
        self.memory["depth"] = max(self.memory.get("depth", 0), len(path) - 1)

        if self.debug_logging:
            self.debug_history.append("\n".join(select_log))
            self.logger.debug(f"Select path: {path_str}\n" + "\n".join(select_log[1:]))

        return node # Return the selected node


    def _check_surprise( self, parent: Node, child_content: str, child_type: str, child_family: str ) -> Tuple[bool, str]:
        """Checks if a new child node is surprising based on configured factors."""
        cfg = self.config
        is_surprising = False
        explanation = ""
        surprise_factors: List[Dict] = [] # Store factors contributing to surprise

        # 1. Semantic Distance Check (using top-level function)
        if cfg.get("use_semantic_distance", True) and SKLEARN_AVAILABLE and parent.content and child_content:
            try:
                # Pass the MCTS logger instance
                distance = calculate_semantic_distance( parent.content, child_content, self.logger, use_tfidf=True )
                threshold = cfg.get("surprise_threshold", 0.66)
                weight = cfg.get("surprise_semantic_weight", 0.6)
                if distance > threshold and weight > 0:
                    surprise_factors.append({
                        "type": "semantic", "value": distance, "weight": weight, "score": distance * weight,
                        "desc": f"Semantic Distance ({distance:.2f} > {threshold:.2f})"
                    })
            except Exception as e:
                 self.logger.warning(f"Surprise check failed (semantic distance): {e}")

        # 2. Philosophical Shift Check
        parent_family = getattr(parent, "approach_family", "general")
        shift_weight = cfg.get("surprise_philosophical_shift_weight", 0.3)
        # Consider shift surprising if not 'general' and different from parent
        if shift_weight > 0 and parent_family != child_family and child_family not in ["general", "initial", "variant", "unknown"]:
            surprise_factors.append({
                "type": "family_shift", "value": 1.0, "weight": shift_weight, "score": 1.0 * shift_weight,
                "desc": f"Approach Family Shift ('{parent_family}' -> '{child_family}')"
            })

        # 3. Novelty Check (using BFS on current tree)
        novelty_weight = cfg.get("surprise_novelty_weight", 0.3)
        if novelty_weight > 0 and child_family not in ["general", "initial", "variant", "unknown"]:
            try:
                family_counts = Counter()
                queue: List[Tuple[Optional[Node], int]] = [(self.root, 0)] # BFS queue (node, depth)
                visited_ids: Set[str] = set() # Track visited node IDs
                nodes_checked = 0
                MAX_BFS_NODES, MAX_BFS_DEPTH = 100, 5 # Limits to prevent excessive search

                while queue and nodes_checked < MAX_BFS_NODES:
                    current_node, depth = queue.pop(0)
                    # Skip if node invalid, already visited, or too deep
                    if not current_node or current_node.id in visited_ids or depth > MAX_BFS_DEPTH: continue

                    visited_ids.add(current_node.id)
                    nodes_checked += 1
                    family_counts[getattr(current_node, "approach_family", "general")] += 1

                    # Add valid children to queue if within depth limit
                    if depth < MAX_BFS_DEPTH:
                        queue.extend( (child, depth + 1) for child in current_node.children if isinstance(child, Node) and child.id not in visited_ids )

                # Check if the new family is rare (count <= 1)
                child_family_count = family_counts.get(child_family, 0)
                if child_family_count <= 1:
                    surprise_factors.append({
                        "type": "novelty", "value": 1.0, "weight": novelty_weight, "score": 1.0 * novelty_weight,
                        "desc": f"Novel Approach Family ('{child_family}', seen {child_family_count}x in near tree)"
                    })
            except Exception as e:
                self.logger.warning(f"Surprise check failed (novelty BFS): {e}", exc_info=self.debug_logging)

        # --- Calculate Overall Surprise Score ---
        if surprise_factors:
            total_weighted_score = sum(f["score"] for f in surprise_factors)
            total_weight = sum(f["weight"] for f in surprise_factors)
            if total_weight > 1e-6: # Avoid division by zero
                overall_score = total_weighted_score / total_weight
                overall_threshold = cfg.get("surprise_overall_threshold", 0.9)
                # Check if overall score meets the threshold
                if overall_score >= overall_threshold:
                    is_surprising = True
                    factor_descs = [f"- {f['desc']} (Contributes: {f['score']:.2f})" for f in surprise_factors]
                    explanation = ( f"Surprise! (Overall Score: {overall_score:.2f} >= {overall_threshold:.2f})\nFactors:\n" + "\n".join(factor_descs) )
                    self.logger.info( f"Surprise DETECTED: N{parent.sequence} -> New Child. Score={overall_score:.2f}\nFactors: {factor_descs}" )

        return is_surprising, explanation

    async def expand(self, node: Node) -> Optional[Node]:
        """Expands a node by generating a thought, updating content, and creating a child."""
        cfg = self.config
        if not isinstance(node, Node):
            self.logger.error("Expand called with invalid node.")
            return None
        if node.fully_expanded():
            self.logger.warning(f"Attempted to expand already full node N{node.sequence}. Aborting expansion.")
            return None # Cannot expand if already full

        self.logger.debug(f"Expanding Node N{node.sequence}...")
        expand_log_entry = f"### Expand N{node.sequence}\n"

        try:
            # 1. Generate Thought
            await self.llm.progress(f"N{node.sequence}: Generating thought...")
            context = self.get_context_for_node(node) # Get context for the parent node
            # Ensure required context keys are present
            required_context_keys = ["current_answer", "question_summary", "best_answer"]
            if not all(k in context for k in required_context_keys):
                 self.logger.error(f"Expand N{node.sequence}: Missing required context keys for thought generation. Keys: {list(context.keys())}")
                 expand_log_entry += "... Error: Missing context for thought generation.\n"
                 self.thought_history.append(expand_log_entry)
                 return None

            thought_text = await self.llm.generate_thought(node.content, context, cfg)
            # Validate thought
            if not isinstance(thought_text, str) or not thought_text.strip() or thought_text.startswith("Error:"):
                self.logger.error(f"Expand N{node.sequence}: Thought generation failed or returned error: '{thought_text}'")
                expand_log_entry += f"... Thought Generation Error: {thought_text}\n"
                self.thought_history.append(expand_log_entry)
                return None
            thought = thought_text.strip()

            # 2. Classify Thought (using top-level function)
            # Pass self.random_state and self.logger
            approach_type, approach_family = classify_approach( thought, APPROACH_TAXONOMY, APPROACH_METADATA, self.random_state, self.logger )
            expand_log_entry += f"... Thought ({approach_type}/{approach_family}): {thought}\n"

            # Track explored thought/approach
            self.explored_thoughts.add(thought)
            self.explored_approaches.setdefault(approach_type, []).append(thought)

            # 3. Update Analysis Content
            await self.llm.progress(f"N{node.sequence}: Updating analysis based on thought...")
            # Context for update might be same as thought context here
            updated_content_text = await self.llm.update_approach( node.content, thought, context, cfg )
             # Validate updated content
            if not isinstance(updated_content_text, str) or not updated_content_text.strip() or updated_content_text.startswith("Error:"):
                self.logger.error(f"Expand N{node.sequence}: Content update failed or returned error: '{updated_content_text}'")
                expand_log_entry += f"... Content Update Error: {updated_content_text}\n"
                self.thought_history.append(expand_log_entry)
                return None
            child_content = updated_content_text.strip()
            expand_log_entry += f"... Updated Analysis Snippet: {truncate_text(child_content, 150)}\n"

            # 4. Generate Tags (using internal MCTS method now)
            await self.llm.progress(f"N{node.sequence}: Generating tags...")
            child_tags: List[str] = []
            try:
                child_tags = await self._generate_tags_for_node(child_content)
                expand_log_entry += f"... Generated Tags: {child_tags}\n"
            except Exception as tag_err:
                self.logger.error(f"Expand N{node.sequence}: Tag generation failed: {tag_err}", exc_info=self.debug_logging)
                expand_log_entry += f"... Tag Generation Error: {tag_err}\n"

            # 5. Check for Surprise (using internal method)
            is_surprising, surprise_expl = False, ""
            try:
                is_surprising, surprise_expl = self._check_surprise( node, child_content, approach_type, approach_family )
                if is_surprising:
                    expand_log_entry += f"**SURPRISE DETECTED!**\n{surprise_expl}\n"
            except Exception as surprise_err:
                self.logger.error(f"Expand N{node.sequence}: Surprise check failed: {surprise_err}", exc_info=self.debug_logging)
                expand_log_entry += f"... Surprise Check Error: {surprise_err}\n"

            # 6. Create Child Node
            child_sequence = self.get_next_sequence()
            # Inherit config settings for child node initialization
            child_alpha = max(1e-9, float(cfg.get("beta_prior_alpha", 1.0))) # Start with default priors
            child_beta = max(1e-9, float(cfg.get("beta_prior_beta", 1.0)))

            child_node = Node(
                content=child_content,
                sequence=child_sequence,
                parent=node, # Set parent reference
                is_surprising=is_surprising,
                surprise_explanation=surprise_expl,
                approach_type=approach_type,
                approach_family=approach_family,
                thought=thought,
                max_children=cfg["max_children"], # Pass config down
                use_bayesian_evaluation=cfg["use_bayesian_evaluation"], # Pass config down
                alpha=child_alpha,
                beta=child_beta,
                value=0.0, # Initial value for non-Bayesian
                descriptive_tags=child_tags,
            )
            node.add_child(child_node) # Add child to parent's list

            # Track surprising nodes
            if is_surprising: self.surprising_nodes.append(child_node)
            # Update branch count memory
            if len(node.children) == 2: # If this is the second child
                 self.memory["branches"] = self.memory.get("branches", 0) + 1

            expand_log_entry += f"--> Created Child N{child_sequence}\n"
            self.thought_history.append(expand_log_entry)
            self.logger.info(f"Expanded N{node.sequence} -> N{child_sequence} (Approach: {approach_type}, Surprise: {is_surprising})")
            return child_node

        except Exception as e:
            self.logger.error(f"Expand N{node.sequence} encountered an unexpected error: {e}", exc_info=self.debug_logging)
            expand_log_entry += f"... Unexpected Expansion Error: {type(e).__name__}\n"
            self.thought_history.append(expand_log_entry)
            return None

    async def _generate_tags_for_node(self, text: str) -> List[str]:
        """Generates descriptive tags for a given text using the LLM."""
        if not text or not isinstance(text, str): return []
        tags: List[str] = []
        try:
            # Ensure LLM interface is available
            if not self.llm:
                 self.logger.error("Tag generation failed: LLM interface not available in MCTS.")
                 return []

            # Resolve model using the llm interface and stored request body
            model_name = self.llm.resolve_model(self.model_body)
            if not model_name:
                self.logger.error("Tag gen failed: Could not resolve model name.")
                return []

            # Create prompt and call LLM
            prompt = GENERATE_TAGS_PROMPT.format(analysis_text=truncate_text(text, 1000))
            raw_response = await self.llm.get_completion( model_name, [{"role": "user", "content": prompt}] )

            # Validate response
            if not raw_response or raw_response.startswith("Error:"):
                self.logger.warning(f"Tag generation LLM call failed or returned error: {raw_response}")
                return []

            # --- Tag Parsing Logic ---
            # Remove potential code blocks or markdown first
            cleaned = re.sub(r"^\s*```[\s\S]*?```\s*$", "", raw_response, flags=re.DOTALL | re.MULTILINE).strip()
            # Remove common prefixes like "Tags:", "Keywords:" etc.
            cleaned = re.sub(r"^\s*(tags|keywords|output|response)[:\-]?\s*", "", cleaned, flags=re.IGNORECASE).strip()
            # Remove specific formatting characters
            cleaned = re.sub(r"[`'*_]", "", cleaned)
            # Remove leading list markers like '-', '*', '+'
            cleaned = re.sub(r"^\s*[-*+]\s*", "", cleaned, flags=re.MULTILINE).strip()

            potential_tags = re.split(r'[,\n;]+', cleaned) # Split by common delimiters
            processed_tags_lower: Set[str] = set() # Track lower-case tags to avoid duplicates
            final_tags: List[str] = []

            for tag in potential_tags:
                # Clean individual tag: strip whitespace, remove surrounding quotes/brackets, normalize internal space
                t_clean = re.sub(r"\s+", " ", tag.strip().strip("'.\"-()[]{}").strip())
                t_lower = t_clean.lower()

                # Validate tag: non-empty, reasonable length, not just digits, not 'none', not duplicate
                if ( t_lower and 1 < len(t_lower) < 50 and not t_lower.isdigit() and t_lower != "none" and t_lower not in processed_tags_lower ):
                    final_tags.append(t_clean) # Keep original casing
                    processed_tags_lower.add(t_lower)

                if len(final_tags) >= 5: break # Limit to max 5 tags

            tags = final_tags
            if self.debug_logging:
                self.logger.debug(f"Tag Gen: Raw='{raw_response[:100]}...', Cleaned='{cleaned[:100]}...', Final={tags}")

        except Exception as e:
            self.logger.error(f"Tag generation failed: {e}", exc_info=self.debug_logging)
            return [] # Return empty list on error
        return tags

    async def simulate(self, node: Node) -> Optional[float]:
        """Simulates (evaluates) a node's content using the LLM."""
        cfg = self.config
        if not isinstance(node, Node):
            self.logger.error("Simulate called with invalid node.")
            return None

        self.logger.debug(f"Simulating N{node.sequence} (Approach:{node.approach_type}, Tags:{node.descriptive_tags})...")
        simulation_score: Optional[float] = None
        raw_llm_response: Union[int, str] = 0 # Store raw response for score calculation
        sim_log_entry = f"### Evaluate N{node.sequence} (Tags:{node.descriptive_tags})\n"

        try:
            # Handle nodes with empty content
            if not node.content or not isinstance(node.content, str):
                 self.logger.warning(f"Simulating N{node.sequence} with empty content. Assigning score 1.0.")
                 raw_llm_response = 1
                 simulation_score = 1.0
                 sim_log_entry += "... Score: 1.0/10 (Node content was empty)\n"
            else:
                # Proceed with LLM evaluation
                await self.llm.progress(f"Evaluating N{node.sequence}...")
                context = self.get_context_for_node(node)
                # Ensure required context keys for evaluation prompt are present
                eval_req_keys = ["answer_to_evaluate", "question_summary", "best_answer", "best_score"]
                context["answer_to_evaluate"] = node.content # Add the content to evaluate
                if not all(k in context for k in eval_req_keys):
                    self.logger.error(f"Simulate N{node.sequence}: Missing required context keys for evaluation. Keys: {list(context.keys())}")
                    sim_log_entry += "... Error: Missing required context for evaluation.\n"
                    self.thought_history.append(sim_log_entry)
                    return None # Cannot proceed without required context

                # Call LLM for evaluation
                eval_response = await self.llm.evaluate_answer( node.content, context, cfg )

                # Process evaluation response
                if isinstance(eval_response, int) and 1 <= eval_response <= 10:
                    raw_llm_response = eval_response
                    simulation_score = float(raw_llm_response)
                    sim_log_entry += f"... LLM Score: {simulation_score:.1f}/10\n"
                elif isinstance(eval_response, str) and eval_response.startswith("Error:"):
                    # Handle specific error message from evaluate_answer
                    self.logger.error(f"Simulate N{node.sequence}: Evaluation LLM call failed: {eval_response}")
                    sim_log_entry += f"... Evaluation Error (LLM): {eval_response}\n"
                    simulation_score = None # Indicate failure
                    raw_llm_response = eval_response # Store error string
                else: # Handle unexpected response type (e.g., string that's not an error)
                    self.logger.error(f"Simulate N{node.sequence}: Unexpected evaluation response type: {type(eval_response)}. Response: '{eval_response}'")
                    sim_log_entry += f"... Evaluation Error: Unexpected response type {type(eval_response)}.\n"
                    simulation_score = None # Indicate failure
                    raw_llm_response = f"Error: Unexpected eval type {type(eval_response)}"

            # --- Post-Evaluation Updates (if score was obtained) ---
            if simulation_score is not None:
                # Store raw score (might be numeric or error string)
                node.raw_scores.append(raw_llm_response if isinstance(raw_llm_response, (int, float)) else -1) # Store -1 for errors

                # Update Approach Priors/Scores
                approach = node.approach_type or "unknown"
                if approach != "unknown":
                    if cfg.get("use_bayesian_evaluation"):
                        # Convert score 1-10 to successes/failures for Beta update
                        # Score 1 -> 0 successes, 9 failures
                        # Score 10 -> 9 successes, 0 failures
                        successes = max(0.0, simulation_score - 1.0)
                        failures = max(0.0, 10.0 - simulation_score)
                        current_alpha = self.approach_alphas.get(approach, cfg.get("beta_prior_alpha", 1.0))
                        current_beta = self.approach_betas.get(approach, cfg.get("beta_prior_beta", 1.0))
                        # Update priors, ensuring they stay positive
                        self.approach_alphas[approach] = max(1e-9, current_alpha + successes)
                        self.approach_betas[approach] = max(1e-9, current_beta + failures)
                    else: # Non-Bayesian: Update simple average score
                        current_avg = self.approach_scores.get(approach, simulation_score) # Init with current score if first time
                        smoothing_factor = 0.3 # Simple exponential moving average factor
                        self.approach_scores[approach] = (smoothing_factor * simulation_score) + ((1 - smoothing_factor) * current_avg)

                # Update High Score Memory
                memory_threshold = 7.0 # Threshold score to be considered for memory
                if simulation_score >= memory_threshold:
                    memory_entry = (simulation_score, node.content, node.approach_type, node.thought)
                    current_memory: List = self.memory.get("high_scoring_nodes", [])
                    current_memory.append(memory_entry)
                    # Keep memory sorted and limited
                    self.memory["high_scoring_nodes"] = sorted(current_memory, key=lambda x: x, reverse=True)[:cfg.get("memory_cutoff", 5)]

                self.logger.info(f"Simulated N{node.sequence}: Score = {simulation_score:.1f}/10 (Raw Eval: {raw_llm_response})")
            else:
                self.logger.warning(f"Simulation FAILED for N{node.sequence}. No score obtained.")

            self.thought_history.append(sim_log_entry) # Log simulation details

        except Exception as e:
            self.logger.error(f"Simulate N{node.sequence} encountered an unexpected error: {e}", exc_info=self.debug_logging)
            sim_log_entry += f"... Unexpected Simulation Error: {type(e).__name__}\n"
            self.thought_history.append(sim_log_entry)
            return None # Return None on unexpected error

        return simulation_score # Return the numeric score or None if failed

    def backpropagate(self, node: Node, score: float):
        """Updates statistics up the tree from the simulated node."""
        cfg = self.config
        if not isinstance(node, Node):
            self.logger.error("Backpropagate called with invalid node.")
            return
        # Validate score
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            self.logger.error(f"Backpropagate called with invalid score: {score}. Aborting.")
            return

        self.logger.debug(f"Backpropagating score {score:.2f} from N{node.sequence}...")
        backprop_path_nodes: List[str] = [] # Track path for logging
        current_node: Optional[Node] = node

        # Convert score 1-10 to successes/failures for Bayesian update
        successes = max(0.0, score - 1.0)
        failures = max(0.0, 10.0 - score)

        while current_node:
            node_id = current_node.sequence
            backprop_path_nodes.append(f"N{node_id}")
            current_node.visits += 1

            # Update node values based on scoring mode
            if cfg.get("use_bayesian_evaluation"):
                if current_node.alpha is not None and current_node.beta is not None:
                    current_node.alpha = max(1e-9, current_node.alpha + successes)
                    current_node.beta = max(1e-9, current_node.beta + failures)
                else: # Initialize priors if somehow missing (shouldn't happen with __init__)
                     self.logger.warning(f"Node N{node_id} missing alpha/beta during backprop. Initializing.")
                     prior_alpha = max(1e-9, cfg.get("beta_prior_alpha", 1.0))
                     prior_beta = max(1e-9, cfg.get("beta_prior_beta", 1.0))
                     current_node.alpha = prior_alpha + successes
                     current_node.beta = prior_beta + failures
            else: # Non-Bayesian: Update cumulative score
                if current_node.value is not None:
                    current_node.value += score # Add the 1-10 score directly
                else: # Initialize value if somehow missing
                    self.logger.warning(f"Node N{node_id} missing value during non-Bayesian backprop. Initializing.")
                    current_node.value = score

            # Debug Logging for node state after update
            if self.debug_logging:
                if cfg.get("use_bayesian_evaluation") and current_node.alpha is not None and current_node.beta is not None:
                     details = f"α={current_node.alpha:.2f}, β={current_node.beta:.2f} (Mean: {current_node.get_bayesian_mean():.3f})"
                elif not cfg.get("use_bayesian_evaluation") and current_node.value is not None:
                     details = f"Value={current_node.value:.2f}, AvgScore={current_node.get_average_score():.2f}"
                else: details = "Params Missing!"
                self.logger.debug(f"  Backprop Updated N{node_id}: V={current_node.visits}, {details}")

            # Move to parent
            current_node = current_node.parent

        # Log the backpropagation path
        final_path_str = " -> ".join(reversed(backprop_path_nodes))
        self.thought_history.append(f"### Backprop Score {score:.1f}\nPath: {final_path_str}\n")


    async def search(self, sims_per_iter: int) -> bool:
        """Runs the main MCTS loop for a specified number of simulations."""
        cfg = self.config
        debug = self.debug_logging
        show_sim_details = self.show_chat_details # Use instance flag
        current_iteration = self.iterations_completed + 1

        self.logger.info(f"--- MCTS Iteration {current_iteration}/{cfg.get('max_iterations', 0)} ({sims_per_iter} sims) ---")

        for i in range(sims_per_iter):
            self.simulations_completed += 1
            current_simulation = i + 1
            simulation_id = f"Iter {current_iteration}.{current_simulation}" # Unique ID for logging/reporting
            self.thought_history.append(f"\n### Simulation {simulation_id} (Total: {self.simulations_completed})\n")
            sim_summary = "" # For chat reporting if enabled
            node_to_evaluate: Optional[Node] = None
            evaluated_score: Optional[float] = None
            selected_node: Optional[Node] = None # Node chosen by select phase

            # --- 1. Selection Phase ---
            try:
                selected_node = await self.select()
                if not selected_node:
                    sim_summary += "Select: FAILED (No node selected).\n"
                    self.thought_history.append(f"... {sim_summary}")
                    self.logger.error(f"Sim {simulation_id}: Select phase failed.")
                    continue # Skip to next simulation if selection fails

                sim_summary += f"Select: N{selected_node.sequence} (V:{selected_node.visits}, S:{selected_node.get_average_score():.1f}, T:{selected_node.descriptive_tags})\n"
                node_to_evaluate = selected_node # Initially, simulate the selected node

            except Exception as e:
                sim_summary += f"Select: Error ({type(e).__name__}).\n"
                self.logger.error(f"Sim {simulation_id}: Error during Select phase: {e}", exc_info=debug)
                self.thought_history.append(f"... {sim_summary}")
                continue # Skip simulation on error

            # --- 2. Expansion Phase ---
            # Expand if the selected node is not fully expanded and has content
            if not selected_node.fully_expanded() and selected_node.content:
                sim_summary += "Expand: Attempting...\n"
                expanded_node: Optional[Node] = None
                try:
                    expanded_node = await self.expand(selected_node) # Attempt expansion
                    if expanded_node:
                        node_to_evaluate = expanded_node # Simulate the *newly expanded* node
                        thought_str = str(expanded_node.thought).strip() if expanded_node.thought else "(N/A)"
                        sim_summary += f'  Expand Thought: "{thought_str}"\n'
                        sim_summary += f"  Expand Result: --> New Node N{expanded_node.sequence} ({expanded_node.approach_type}, S:{expanded_node.get_average_score():.1f}, T:{expanded_node.descriptive_tags})\n"
                    else: # Expansion failed
                        sim_summary += f"  Expand Result: FAILED. Evaluating originally selected N{selected_node.sequence}.\n"
                        # node_to_evaluate remains selected_node
                except Exception as e:
                     sim_summary += f"  Expand: Error ({type(e).__name__}). Evaluating N{selected_node.sequence}.\n"
                     self.logger.error(f"Sim {simulation_id}: Error during Expand phase: {e}", exc_info=debug)
                     # node_to_evaluate remains selected_node
            else: # Expansion skipped (node full or no content)
                 sim_summary += f"Expand: Skipped ({( 'Node full' if selected_node.fully_expanded() else 'Node has no content')}).\n"

            # --- 3. Simulation Phase ---
            # Simulate the node determined by select/expand phase (if valid)
            if node_to_evaluate and node_to_evaluate.content:
                sim_summary += f"Evaluate: N{node_to_evaluate.sequence}...\n"
                try:
                    evaluated_score = await self.simulate(node_to_evaluate)
                    if evaluated_score is not None:
                         sim_summary += f"  Evaluate Score: {evaluated_score:.1f}/10\n"
                    else:
                        sim_summary += f"  Evaluate Score: FAILED.\n"
                        evaluated_score = None # Ensure it's None if sim fails
                except Exception as e:
                    sim_summary += f"  Evaluate: Error ({type(e).__name__}).\n"
                    self.logger.error(f"Sim {simulation_id}: Error during Simulate phase: {e}", exc_info=debug)
                    evaluated_score = None
            elif node_to_evaluate: # Node exists but has no content
                 sim_summary += f"Evaluate: Skipped N{node_to_evaluate.sequence} (no content).\n"
            else: # Should not happen if selection worked
                 sim_summary += f"Evaluate: Skipped (no valid node selected/expanded).\n"

            # --- 4. Backpropagation Phase ---
            if evaluated_score is not None and node_to_evaluate:
                sim_summary += f"Backpropagate: Score {evaluated_score:.1f} from N{node_to_evaluate.sequence}...\n"
                try:
                    self.backpropagate(node_to_evaluate, evaluated_score)
                    sim_summary += "  Backpropagate: OK.\n"
                except Exception as e:
                    sim_summary += f"  Backpropagate: Error ({type(e).__name__}).\n"
                    self.logger.error(f"Sim {simulation_id}: Error during Backpropagate phase: {e}", exc_info=debug)

                # --- Update Best Score & Check Early Stopping ---
                if evaluated_score > self.best_score:
                    old_best = self.best_score
                    self.best_score = evaluated_score
                    self.best_solution = str(node_to_evaluate.content) if node_to_evaluate.content else self.best_solution # Update best content
                    self.high_score_counter = 0 # Reset stability counter on improvement
                    sim_summary += f"🏆 New Best Overall Score! {self.best_score:.1f}/10 (Prev: {old_best:.1f})\n"
                    node_info = f"N{node_to_evaluate.sequence} ({node_to_evaluate.approach_type}) Tags:{node_to_evaluate.descriptive_tags}"
                    self.thought_history.append(f"### New Best! Score: {self.best_score:.1f} ({node_info})\n")
                    self.logger.info(f"Sim {simulation_id}: New best found! Score: {self.best_score:.1f}, Node: {node_info}")

                # Early Stopping Check (only if score wasn't a new best this sim)
                elif cfg.get("early_stopping", True) and evaluated_score >= cfg.get("early_stopping_threshold", 10.0):
                    self.high_score_counter += 1
                    stability_required = cfg.get("early_stopping_stability", 2)
                    sim_summary += f"  Stability Check: Score {evaluated_score:.1f} >= Threshold. Count:{self.high_score_counter}/{stability_required}\n"
                    if self.high_score_counter >= stability_required:
                        self.logger.info(f"Sim {simulation_id}: Early stopping criteria met (Score {evaluated_score:.1f} reached threshold and stability).")
                        # Report early stop to user if verbose
                        if show_sim_details and self.llm:
                            await self.llm.emit_message(f"**Stopping early:** Analysis score ({self.best_score:.1f}/10) stable at/above threshold.")
                        self.thought_history.append(f"... {sim_summary}") # Log final summary before stop
                        return False # Signal to stop MCTS loop

                else: # Reset counter if score drops below threshold
                    self.high_score_counter = 0

            else: # Backpropagation skipped (no score or no node)
                 self.high_score_counter = 0 # Reset stability counter if sim/eval failed
                 sim_summary += "Backpropagate: Skipped (No valid score/node).\n"

            # --- Emit Simulation Summary (if verbose) ---
            if show_sim_details:
                await self.llm.emit_message(f"--- Sim {simulation_id} Summary ---\n{sim_summary}")
                await asyncio.sleep(0.05) # Small delay for chat readability

            # Append summary to internal history regardless of verbosity
            self.thought_history.append(f"... {sim_summary}")

        # --- End of Simulation Loop for Iteration ---
        self.logger.info(f"--- Finished Iteration {current_iteration}. Current Best Score: {self.best_score:.1f} ---")
        return True # Signal to continue MCTS loop


    def find_best_final_node(self) -> Optional[Node]:
        """Finds the node object that corresponds to the stored best_solution content."""
        if not self.root: return None
        if not self.best_solution: # If no best solution recorded, return root? Or highest score node?
             self.logger.warning("find_best_final_node called but self.best_solution is empty. Trying highest score fallback.")
             # Fallback: Find node with highest average score overall
             return self._find_highest_score_node()

        # Normalize target content for comparison
        target_content_cleaned = re.sub(r'\s+', ' ', str(self.best_solution).strip(), flags=re.MULTILINE)

        queue: List[Node] = [self.root]
        visited_ids: Set[str] = {self.root.id}
        nodes_with_matching_content: List[Node] = []

        # BFS to find nodes with matching content
        while queue:
            current_node = queue.pop(0)
            # Normalize node content
            node_content_cleaned = re.sub(r'\s+', ' ', str(current_node.content).strip(), flags=re.MULTILINE)
            # Compare cleaned content
            if node_content_cleaned == target_content_cleaned:
                nodes_with_matching_content.append(current_node)

            # Add valid children to queue
            for child in current_node.children:
                if isinstance(child, Node) and child.id not in visited_ids:
                    visited_ids.add(child.id)
                    queue.append(child)

        if nodes_with_matching_content:
            # If multiple nodes match content, choose the one with score closest to self.best_score
            best_match_node = min( nodes_with_matching_content, key=lambda n: abs(n.get_average_score() - self.best_score) )
            self.logger.debug(f"Found best final node N{best_match_node.sequence} matching content.")
            return best_match_node
        else:
            # Fallback if no exact content match found
            self.logger.warning(f"Could not find node via content match for best score {self.best_score:.1f}. Trying highest score fallback.")
            return self._find_highest_score_node()

    def _find_highest_score_node(self) -> Optional[Node]:
        """Helper: Finds the node with the highest average score in the entire tree."""
        if not self.root: return None
        all_nodes: List[Node] = []
        queue: List[Node] = [self.root]
        visited: Set[str] = {self.root.id}
        while queue:
            curr = queue.pop(0)
            if curr.visits > 0: # Only consider nodes that were actually visited/evaluated
                 all_nodes.append(curr)
            # Add children to queue
            for child in curr.children:
                if isinstance(child, Node) and child.id not in visited:
                    visited.add(child.id)
                    queue.append(child)
        if all_nodes:
            # Return node with the maximum average score
            highest_score_node = max(all_nodes, key=lambda n: n.get_average_score())
            self.logger.debug(f"Highest score node fallback: N{highest_score_node.sequence} (Score: {highest_score_node.get_average_score():.1f})")
            return highest_score_node
        else: # If no visited nodes (only root?)
            return self.root


    def get_state_for_persistence(self) -> Optional[Dict[str, Any]]:
        """Extracts the relevant state from the MCTS process for saving."""
        if not self.root:
            self.logger.error("Cannot get state for persistence: Root node is missing.")
            return None

        state: Dict[str, Any] = {
            "version": SCRIPT_VERSION, # Include script version in state
            "timestamp": datetime.now().isoformat(), # Timestamp of saving
        }
        try:
            # --- Core State Information ---
            state["best_score"] = round(self.best_score, 3)
            best_node = self.find_best_final_node() # Get the node corresponding to the best score/content
            state["best_solution_content"] = str(self.best_solution) # Store full best content
            state["best_solution_summary"] = truncate_text(self.best_solution, 400) # Store summary
            state["best_node_tags"] = best_node.descriptive_tags[:] if best_node else [] # Tags of best node
            state["best_node_sequence"] = best_node.sequence if best_node else None

            # --- Approach Priors/Scores ---
            if self.config.get("use_bayesian_evaluation"):
                state["approach_priors"] = {
                    "alpha": {k: round(v, 4) for k, v in self.approach_alphas.items()},
                    "beta": {k: round(v, 4) for k, v in self.approach_betas.items()},
                }
            else: # Store non-Bayesian scores if used
                state["approach_scores"] = {k: round(v, 3) for k, v in self.approach_scores.items()}

            # --- Representative Nodes (e.g., Top 3 by score) ---
            all_nodes: List[Node] = []
            queue: List[Node] = [self.root]
            visited_ids: Set[str] = {self.root.id}
            while queue:
                current_node = queue.pop(0)
                if current_node.visits > 0: # Only include nodes that were actually visited
                    all_nodes.append(current_node)
                # Add valid children to queue
                for child in current_node.children:
                    if isinstance(child, Node) and child.id not in visited_ids:
                        visited_ids.add(child.id)
                        queue.append(child)
            # Sort visited nodes by score
            sorted_nodes = sorted(all_nodes, key=lambda n: n.get_average_score(), reverse=True)
            # Store state dict for top N nodes
            state["top_nodes"] = [node.node_to_state_dict() for node in sorted_nodes[:3]]

            # --- Unfit Markers (Nodes below score threshold after sufficient visits) ---
            unfit_markers = []
            score_threshold = self.config.get("unfit_score_threshold", 4.0)
            visit_threshold = self.config.get("unfit_visit_threshold", 3)
            for node in all_nodes: # Reuse collected nodes
                avg_score = node.get_average_score()
                if node.visits >= visit_threshold and avg_score < score_threshold:
                    unfit_markers.append({
                        "id": node.id, # Use node ID for potential future matching
                        "seq": node.sequence,
                        "summary": truncate_text(node.thought or node.content, 80), # Summary of thought/content
                        "reason": f"Low score ({avg_score:.1f})",
                        "tags": node.descriptive_tags[:], # Include tags for context
                    })
            state["unfit_markers"] = unfit_markers[:10] # Limit number of markers saved

            self.logger.info(f"Generated state for persistence. Best Score: {state['best_score']:.2f}, Top Nodes: {len(state['top_nodes'])}, Unfit: {len(state['unfit_markers'])}")
            return state

        except Exception as e:
            self.logger.error(f"Failed to generate state for persistence: {e}", exc_info=self.debug_logging)
            return None


    def get_final_synthesis_context(self) -> Optional[Dict[str, str]]:
        """Assembles context specifically for the final synthesis LLM call."""
        if not self.root or not self.best_solution:
            self.logger.warning("Cannot get synthesis context: Root or best solution missing.")
            return None

        best_node = self.find_best_final_node()
        if not best_node: best_node = self.root # Fallback to root if best node not found

        # --- Reconstruct Path to Best Node ---
        path_to_best: List[Node] = []
        current_node: Optional[Node] = best_node
        while current_node:
            path_to_best.append(current_node)
            current_node = current_node.parent
        path_to_best.reverse() # Order from root to best

        # --- Extract Thoughts Along the Path ---
        path_thoughts_lines = []
        for i, node in enumerate(path_to_best):
            if i > 0 and node.thought: # Include thought if not root and thought exists
                parent_seq = node.parent.sequence if node.parent else "?"
                path_thoughts_lines.append(f"- N{node.sequence} ({node.approach_type}): {node.thought.strip()}")
            elif i == 0: # Identify root node
                 path_thoughts_lines.append(f"- N{node.sequence} (Initial Root)")
        path_thoughts_str = "\n".join(path_thoughts_lines) if path_thoughts_lines else "N/A."

        try:
            # Prepare context dictionary
            synthesis_context = {
                "question_summary": self.question_summary,
                "initial_analysis_summary": truncate_text(str(self.root.content), 500) if self.root else "N/A",
                "best_score": f"{self.best_score:.1f}",
                "path_thoughts": path_thoughts_str,
                "final_best_analysis_summary": truncate_text(str(self.best_solution), 1500), # More context for final analysis
            }
            return synthesis_context
        except Exception as e:
            self.logger.error(f"Error assembling final synthesis context: {e}", exc_info=self.debug_logging)
            return None


    def formatted_output(self) -> str:
        """Generates a formatted summary of the MCTS results (verbose mode)."""
        cfg = self.config
        output_lines = [f"# MCTS Summary v{SCRIPT_VERSION}", f"*Completed: {datetime.now():%Y-%m-%d %H:%M:%S}*"]

        try:
            # --- 1. Best Analysis ---
            best_node = self.find_best_final_node()
            best_tags = f"Tags: {best_node.descriptive_tags}" if best_node and best_node.descriptive_tags else "Tags: []"
            output_lines.append(f"\n## Best Analysis (Score: {self.best_score:.1f}/10)\n**{best_tags}**")
            # Clean best solution text for output
            cleaned_best_solution = re.sub(r'\s+', ' ', str(self.best_solution).strip(), flags=re.MULTILINE) if self.best_solution else "(N/A)"
            output_lines.append(f"\n\n{cleaned_best_solution}\n\n")

            # --- 2. Top Performing Nodes ---
            output_lines.append("\n## Top Performing Nodes (Max 5)")
            all_nodes: List[Node] = []
            queue = [self.root] if self.root else []
            visited_ids = {self.root.id} if self.root else set()
            while queue: # BFS to collect all visited nodes
                current_node = queue.pop(0)
                if current_node.visits > 0: all_nodes.append(current_node)
                for child in current_node.children:
                    if isinstance(child, Node) and child.id not in visited_ids:
                        visited_ids.add(child.id); queue.append(child)

            if not all_nodes: output_lines.append("*No nodes visited.*")
            else:
                sorted_nodes = sorted(all_nodes, key=lambda n: n.get_average_score(), reverse=True)
                for i, node in enumerate(sorted_nodes[:5]):
                    score = node.get_average_score()
                    score_details = ""
                    # Add scoring details based on mode
                    if cfg.get("use_bayesian_evaluation") and node.alpha is not None and node.beta is not None:
                        score_details = f"(α={node.alpha:.1f},β={node.beta:.1f})"
                    elif not cfg.get("use_bayesian_evaluation") and node.value is not None:
                        score_details = f"(Val={node.value:.1f})"
                    output_lines.append( f"### {i+1}. N{node.sequence}: {score:.1f}/10 {score_details}\n"
                                         f"- Info: {node.approach_type}({node.approach_family}), V={node.visits}, T={node.descriptive_tags}\n"
                                         f"- Thought: {node.thought or '(Root)'}" )
                    if node.is_surprising: output_lines.append(f"- Surprise: Yes ({truncate_text(node.surprise_explanation, 100)})")

            # --- 3. Most Explored Path ---
            output_lines.append("\n## Most Explored Path")
            exploration_path: List[Node] = []
            current_node = self.root
            if current_node: exploration_path.append(current_node)
            while current_node and current_node.children:
                most_visited_child = current_node.best_child() # Uses visits then score
                if not most_visited_child or most_visited_child.visits == 0: break # Stop if no visited child
                exploration_path.append(most_visited_child)
                current_node = most_visited_child
            # Format path output
            if len(exploration_path) > 1:
                output_lines.append("") # Add newline
                for i, node in enumerate(exploration_path):
                    prefix = "  " * i + ("└─ " if i == len(exploration_path) - 1 else "├─ ")
                    output_lines.append(f"{prefix}N{node.sequence} ({node.approach_type}, S:{node.get_average_score():.1f}, V:{node.visits}, T:{node.descriptive_tags})")
            elif self.root: output_lines.append(f"*Root N{self.root.sequence} only.*")
            else: output_lines.append("*No path (root missing?).*")

            # --- 4. Surprising Nodes ---
            output_lines.append("\n## Surprising Nodes (Max 5)")
            if self.surprising_nodes:
                for n in self.surprising_nodes[-5:]: # Show last 5 surprising
                    output_lines.append(f"- N{n.sequence} ({n.approach_type}, S:{n.get_average_score():.1f}, T:{n.descriptive_tags}): {truncate_text(n.surprise_explanation, 150)}")
            else: output_lines.append("*None detected.*")

            # --- 5. Approach Performance ---
            output_lines.append("\n## Approach Performance")
            approach_perf_data = []
            # Include all approaches that have scores/priors OR were explored
            all_approach_keys = (set(self.approach_alphas.keys()) | set(self.approach_scores.keys()) | set(self.explored_approaches.keys()))
            valid_approach_keys = [ a for a in all_approach_keys if a not in ["unknown", "initial", "variant"] ]

            for app_key in sorted(valid_approach_keys):
                thought_count = len(self.explored_approaches.get(app_key, []))
                if thought_count == 0: continue # Skip if never used

                score_str, sort_key = "N/A", -1.0
                if cfg.get("use_bayesian_evaluation"):
                    alpha = self.approach_alphas.get(app_key, 1.0)
                    beta = self.approach_betas.get(app_key, 1.0)
                    mean_score = (alpha / (alpha + beta) * 10.0) if (alpha + beta) > 1e-9 else -1
                    score_str = f"Bayes Score: {mean_score:.2f} (α={alpha:.1f},β={beta:.1f})" if mean_score >= 0 else "Error"
                    sort_key = mean_score
                else: # Non-Bayesian
                    avg_score = self.approach_scores.get(app_key)
                    score_str = f"Avg Score: {avg_score:.2f}" if avg_score is not None else "N/A"
                    sort_key = avg_score if avg_score is not None else -1.0
                approach_perf_data.append({ "name": app_key, "score_info": score_str, "count_info": f"(N={thought_count})", "sort_key": sort_key })

            sorted_perf_data = sorted(approach_perf_data, key=lambda x: x['sort_key'], reverse=True)
            if not sorted_perf_data: output_lines.append("*None tracked.*")
            else:
                for item in sorted_perf_data[:7]: # Show top 7
                    output_lines.append(f"- {item['name']}: {item['score_info']} {item['count_info']}")
                if len(sorted_perf_data) > 7: output_lines.append(f"- ... ({len(sorted_perf_data) - 7} more)")

            # --- 6. Search Parameters ---
            output_lines.append(f"\n## Search Parameters")
            output_lines.append(f"- Iter/Sims: {self.iterations_completed}/{cfg.get('max_iterations')} / {cfg.get('simulations_per_iteration')} (Total Sims: {self.simulations_completed})")
            eval_mode = "Bayesian" if cfg.get("use_bayesian_evaluation") else "Avg Score"
            select_mode = ("Thompson" if cfg.get("use_thompson_sampling") else "UCT") if cfg.get("use_bayesian_evaluation") else "UCT"
            output_lines.append(f"- Eval: {eval_mode}, Select: {select_mode}, Explore W: {cfg.get('exploration_weight'):.2f}")
            if cfg.get("use_bayesian_evaluation"): output_lines.append(f"- Priors (Initial): α={cfg.get('beta_prior_alpha'):.2f}, β={cfg.get('beta_prior_beta'):.2f}")
            output_lines.append(f"- Early Stop: {cfg.get('early_stopping')} (Thresh:{cfg.get('early_stopping_threshold'):.1f}, Stab:{cfg.get('early_stopping_stability')})")
            output_lines.append(f"- State:{cfg.get('enable_state_persistence')}, Verbose:{cfg.get('show_processing_details')}, Debug:{cfg.get('debug_logging')}")

            # --- 7. Debug Log Snippets (Optional) ---
            if self.debug_logging and self.debug_history:
                output_lines.append("\n## Debug Log Snippets (Max 3)")
                for entry in self.debug_history[-3:]:
                    cleaned_entry = re.sub(r"\n+", "\n", entry).strip()
                    output_lines.append(f"\n```\n{truncate_text(cleaned_entry, 250)}\n```\n")

            return "\n".join(output_lines).strip()

        except Exception as e:
            self.logger.error(f"Error formatting output: {e}", exc_info=self.debug_logging)
            # Append error to partial output if possible
            return "\n".join(output_lines).strip() + f"\n\n**ERROR generating summary:** {type(e).__name__}"


# ==============================================================================
# Prompts (Define Before Main Class)
# ==============================================================================
INITIAL_ANALYSIS_PROMPT = """<instruction>Provide an initial analysis and interpretation of the themes, arguments, and potential implications presented by the user suitable for the further MCTS analysis.</instruction> <question>{question}</question>"""
GENERATE_THOUGHT_PROMPT = """<instruction>Critically examine the current analysis below using diverse approaches: Challenge assumptions, propose novel connections/analogies, shift perspective (e.g., expert view), refine coherence, or explore new directions if stuck. Consider previous context if available. Avoid rephrasing or exploring marked unfit areas unless novel. Respond with the critique/suggestion ONLY. Previous Context: - Prev Best Summary: {previous_best_summary} - Unfit Areas: {unfit_markers_summary} - Learned Prefs: {learned_approach_summary} Current Context: - Q Summary: {question_summary} - Best Analysis (Score {best_score}/10): {best_answer} - Current Analysis (N{current_sequence}, Tags:{current_tags}): {current_answer} - Explored Approaches: {explored_approaches} - Sibling Thoughts: {sibling_approaches}</instruction>"""
UPDATE_ANALYSIS_PROMPT = """<instruction>Substantially revise the draft analysis below to incorporate the critique's core idea. Develop the analysis further, grounded in the original text and avoiding known unfit paths unless justified. Output the revised analysis text ONLY. Previous Context: - Prev Best Summary: {previous_best_summary} - Unfit Areas: {unfit_markers_summary} Current Context: - Q Summary: {question_summary} - Best Analysis (Score {best_score}/10): {best_answer} - Current Tags: {current_tags} Inputs: <draft>{answer}</draft> <critique>{improvements}</critique></instruction>"""
EVALUATE_ANALYSIS_PROMPT = """<instruction>Evaluate the analysis below (1-10) vs the original text ({question_summary}). Consider: 1. Insight/Novelty (vs best {best_score}/10). 2. Grounding/Reliability. 3. Coherence/Structure. 4. Perspective. Avoid unfit areas ({unfit_markers_summary}). Previous Best Summary: {previous_best_summary}. Analysis Tags: {current_tags}. Analysis to Evaluate: <answer_to_evaluate>{answer_to_evaluate}</answer_to_evaluate> Respond ONLY with the number (1-10).</instruction>"""
GENERATE_TAGS_PROMPT = """<instruction>Generate concise keyword tags (max 5) summarizing the text's main concepts. Output ONLY comma-separated tags.</instruction> <text_to_tag>{analysis_text}</text_to_tag>"""
FINAL_SYNTHESIS_PROMPT = """<instruction>Synthesize the key insights from the analysis path into a concise, conclusive statement addressing the original question ({question_summary}). Focus on the progression from initial analysis ({initial_analysis_summary}) through the best path ({path_thoughts}) to the final best analysis ({final_best_analysis_summary}, Score: {best_score}/10). Respond with natural language text ONLY.</instruction>"""
INTENT_CLASSIFIER_PROMPT = """Determine the primary purpose: ANALYZE_NEW (fresh analysis), CONTINUE_ANALYSIS (build on last run), ASK_LAST_RUN_SUMMARY (results of last run), ASK_PROCESS (how it works), ASK_CONFIG (settings), GENERAL_CONVERSATION. Respond ONLY with the category name. Input: "{raw_input_text}" Classification:"""
ASK_PROCESS_EXPLANATION = """I use Advanced Bayesian MCTS. Key steps: Balance exploration (new ideas) & exploitation (refining good ones) via UCT/Thompson Sampling. Optionally use Bayesian Beta distributions for score uncertainty. Expand tree with LLM-generated 'thoughts' (critiques, angles). Evaluate nodes via LLM for insight, grounding, coherence vs best. Update scores/priors via backpropagation. Optionally persist state (best analysis, score, tags, learned preferences) to `{db_file_name}` per chat session (requires `chat_id`). Try to understand intent (new analysis, continue, ask about process/config/results). Valves in UI tune parameters."""
GENERAL_CONVERSATION_PROMPT = """User input was classified as general conversation. Respond appropriately and conversationally. User Input: "{user_input}" Your Response:"""


# ==============================================================================
# Main Plugin Class Definition (Pipe) # <<< RENAMED HERE
# ==============================================================================
class Pipe(LLMInterface): # <<< RENAMED HERE from Pipeline/Function to Pipe

    id = PIPE_NAME # Keep this as the pipeline identifier

    # Valves class for user configuration in the UI
    class Valves(BaseModel):
        MAX_ITERATIONS: int = Field( default=DEFAULT_CONFIG["max_iterations"], title="Max MCTS Iterations", ge=1, le=100)
        SIMULATIONS_PER_ITERATION: int = Field( default=DEFAULT_CONFIG["simulations_per_iteration"], title="Simulations / Iteration", ge=1, le=50)
        MAX_CHILDREN: int = Field( default=DEFAULT_CONFIG["max_children"], title="Max Children / Node", ge=1, le=20)
        EXPLORATION_WEIGHT: float = Field( default=DEFAULT_CONFIG["exploration_weight"], title="Exploration Weight (UCT)", ge=0.0, le=10.0)
        USE_THOMPSON_SAMPLING: bool = Field( default=DEFAULT_CONFIG["use_thompson_sampling"], title="Use Thompson Sampling (if Bayesian)")
        FORCE_EXPLORATION_INTERVAL: int = Field( default=DEFAULT_CONFIG["force_exploration_interval"], title="Force Branch Explore Interval (0=off)", ge=0, le=20)
        SCORE_DIVERSITY_BONUS: float = Field( default=DEFAULT_CONFIG["score_diversity_bonus"], title="UCT Score Diversity Bonus Weight", ge=0.0, le=1.0)
        USE_BAYESIAN_EVALUATION: bool = Field( default=DEFAULT_CONFIG["use_bayesian_evaluation"], title="Use Bayesian (Beta) Evaluation")
        BETA_PRIOR_ALPHA: float = Field( default=DEFAULT_CONFIG["beta_prior_alpha"], title="Bayesian Prior Alpha (>0)", gt=0, le=100.0)
        BETA_PRIOR_BETA: float = Field( default=DEFAULT_CONFIG["beta_prior_beta"], title="Bayesian Prior Beta (>0)", gt=0, le=100.0)
        USE_SEMANTIC_DISTANCE: bool = Field( default=DEFAULT_CONFIG["use_semantic_distance"], title="Use Semantic Distance (Surprise)", json_schema_extra={"disabled": not SKLEARN_AVAILABLE})
        SURPRISE_THRESHOLD: float = Field( default=DEFAULT_CONFIG["surprise_threshold"], title="Surprise Threshold (Semantic)", ge=0.0, le=1.0)
        SURPRISE_SEMANTIC_WEIGHT: float = Field( default=DEFAULT_CONFIG["surprise_semantic_weight"], title="Surprise: Semantic Weight", ge=0.0, le=1.0)
        SURPRISE_PHILOSOPHICAL_SHIFT_WEIGHT: float = Field( default=DEFAULT_CONFIG["surprise_philosophical_shift_weight"], title="Surprise: Approach Shift Weight", ge=0.0, le=1.0)
        SURPRISE_NOVELTY_WEIGHT: float = Field( default=DEFAULT_CONFIG["surprise_novelty_weight"], title="Surprise: Approach Novelty Weight", ge=0.0, le=1.0)
        SURPRISE_OVERALL_THRESHOLD: float = Field( default=DEFAULT_CONFIG["surprise_overall_threshold"], title="Surprise: Overall Threshold", ge=0.0, le=1.0)
        GLOBAL_CONTEXT_IN_PROMPTS: bool = Field( default=DEFAULT_CONFIG["global_context_in_prompts"], title="Use Global Context in Prompts")
        TRACK_EXPLORED_APPROACHES: bool = Field( default=DEFAULT_CONFIG["track_explored_approaches"], title="Track Explored Thought Approaches")
        SIBLING_AWARENESS: bool = Field( default=DEFAULT_CONFIG["sibling_awareness"], title="Add Sibling Context to Prompts")
        MEMORY_CUTOFF: int = Field( default=DEFAULT_CONFIG["memory_cutoff"], title="Memory Cutoff (Top N High Scores)", ge=0, le=20)
        EARLY_STOPPING: bool = Field( default=DEFAULT_CONFIG["early_stopping"], title="Enable Early Stopping")
        EARLY_STOPPING_THRESHOLD: float = Field( default=DEFAULT_CONFIG["early_stopping_threshold"], title="Early Stopping Score Threshold", ge=1.0, le=10.0)
        EARLY_STOPPING_STABILITY: int = Field( default=DEFAULT_CONFIG["early_stopping_stability"], title="Early Stopping Stability Count", ge=1, le=10)
        ENABLE_STATE_PERSISTENCE: bool = Field( default=DEFAULT_CONFIG["enable_state_persistence"], title="Enable State Persistence (DB)")
        UNFIT_SCORE_THRESHOLD: float = Field( default=DEFAULT_CONFIG["unfit_score_threshold"], title="Unfit Marker Score Threshold", ge=0.0, le=10.0)
        UNFIT_VISIT_THRESHOLD: int = Field( default=DEFAULT_CONFIG["unfit_visit_threshold"], title="Unfit Marker Min Visits", ge=1, le=20)
        SHOW_PROCESSING_DETAILS: bool = Field( default=DEFAULT_CONFIG["show_processing_details"], title="Show Detailed MCTS Steps (Chat)")
        DEBUG_LOGGING: bool = Field( default=DEFAULT_CONFIG["debug_logging"], title="Enable Detailed Debug Logging")


    # Pipe metadata
    name: str = PIPE_NAME
    type: str = "pipe" # Standard pipe type

    def __init__(self):
        # Use PIPE logger
        self.logger = logging.getLogger(pipe_log_name)
        self.mcts_logger = logging.getLogger(mcts_log_name) # Init MCTS logger too
        self.valves = self.Valves() # Initialize Valves with defaults
        # Internal state variables for a single run
        self.current_config: Dict[str, Any] = {}
        self.debug_logging: bool = False
        self.__current_event_emitter__: Optional[Callable] = None
        self.__model__: str = ""
        self.__chat_id__: Optional[str] = None
        self.__request_body__: Dict[str, Any] = {}
        self.__user__: Optional[Union[Dict, "AdminUserMock"]] = None # User info from request
        self.logger.info( f"Pipe '{self.id}' INSTANCE initialized. Version: {SCRIPT_VERSION}" )
        if not OPENWEBUI_IMPORTS_AVAILABLE: self.logger.error("Instance Init Warning: OWB components missing.")
        if not SKLEARN_AVAILABLE: self.logger.warning("Instance Init Info: scikit-learn not found.")

    async def on_startup(self):
        self.logger.info(f"Pipe '{self.id}' on_startup.")
        pass # Placeholder

    async def on_shutdown(self):
        self.logger.info(f"Pipe '{self.id}' on_shutdown.")
        pass # Placeholder

    # --- Core Pipe Methods (pipes, pipe) ---
    def pipes(self) -> List[Dict[str, str]]:
        """Lists available models combined with this pipe's ID."""
        list_logger = self.logger
        list_logger.info(f"Pipe '{self.id}' generating pipes list...")
        formatted_pipes = []
        try:
            from open_webui.main import app # Defer import
            if not app: raise ImportError("app object not available")
        except Exception as e:
            list_logger.error(f"Pipes list fail: OWB 'app' missing: {e}")
            return [{"id": f"{self.id}-error-app-missing", "name": f"{self.name} (Error: OWB App Missing)"}]
        try:
            if not hasattr(app, "state"):
                 list_logger.warning("Cannot access app.state.")
                 return [{"id": f"{self.id}-error-app-state", "name": f"{self.name} (Error: App State Missing)"}]

            # Get models from app state (assuming it's populated elsewhere)
            models_in_state = getattr(app.state, "OLLAMA_MODELS", {})
            if not models_in_state or not isinstance(models_in_state, dict):
                list_logger.warning("No models in app.state.OLLAMA_MODELS")
                return [{"id": f"{self.id}-error-no-models", "name": f"{self.name} (No Models Found)"}]

            # Filter for valid models with names
            valid_models = { k: v for k, v in models_in_state.items() if isinstance(v, dict) and v.get("name") }
            if not valid_models:
                list_logger.warning("No valid models found after filter.")
                return [{"id": f"{self.id}-no-valid-models", "name": f"{self.name} (No Valid Models Found)"}]

            # Format pipe IDs like 'pipe_id-model_id'
            for model_key, model_info in valid_models.items():
                if not model_key: continue # Skip if key is empty
                model_display_name = model_info.get("name", model_key) # Use model name for display
                pipe_id = f"{self.id}-{model_key}" # Combine pipe ID and model key
                formatted_pipes.append({"id": pipe_id, "name": model_display_name}) # Use model name directly

            list_logger.info(f"Generated {len(formatted_pipes)} pipes for Pipe '{self.id}'.")
            return formatted_pipes
        except Exception as e:
            list_logger.error(f"Pipes generation error: {e}", exc_info=True)
            return [{"id": f"{self.id}-list-error", "name": f"{self.name} (Error Listing Models)"}]


    # --- LLMInterface Implementation & Helper Methods ---

    def resolve_model(self, body: Optional[Dict[str, Any]] = None) -> str:
        """Resolves the base LLM model name from the combined pipe model ID."""
        body_to_use = body or self.__request_body__ # Use passed body or stored request body
        if not body_to_use:
            self.logger.error("resolve_model: Request body missing.")
            return ""

        pipe_model_id = body_to_use.get("model", "").strip()
        if not pipe_model_id:
            self.logger.error("resolve_model: 'model' field empty/missing in request body.")
            return ""

        expected_prefix = f"{self.id}-" # e.g., "advanced_mcts_stateful-"
        if pipe_model_id.startswith(expected_prefix):
            base_model_name = pipe_model_id[len(expected_prefix):]
            # Basic validation for base model name format
            if base_model_name and re.match(r"^[a-zA-Z0-9_./:-]+$", base_model_name):
                self.logger.debug(f"Resolved base model '{base_model_name}' from pipe ID '{pipe_model_id}'.")
                return base_model_name
            elif base_model_name: # Allow potentially unusual names but warn
                self.logger.warning(f"Resolved model name '{base_model_name}' has unusual characters. Using anyway.")
                return base_model_name
            else: # Prefix found but nothing after it
                self.logger.error(f"resolve_model: Pipe ID '{pipe_model_id}' has no model name after prefix '{expected_prefix}'.")
                return ""
        else:
            # If prefix not found, assume the ID is the base model directly (warn)
            if re.match(r"^[a-zA-Z0-9_./:-]+$", pipe_model_id):
                 self.logger.warning(f"resolve_model: Pipe ID '{pipe_model_id}' does not contain prefix '{expected_prefix}'. Assuming it's the direct base model name.")
                 return pipe_model_id
            else:
                 self.logger.error(f"resolve_model: Pipe ID '{pipe_model_id}' has invalid format and no prefix.")
                 return ""

    def _resolve_question(self, body: Dict[str, Any]) -> str:
        """Extracts the last user message content from the request body."""
        messages = body.get("messages", [])
        if not isinstance(messages, list): return "" # Return empty if messages format is wrong

        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                extracted_content = str(content).strip() if content is not None else ""
                if self.debug_logging:
                    self.logger.debug(f"_resolve_question: Found user message: '{truncate_text(extracted_content, 100)}'")
                return extracted_content
        return "" # Return empty if no user message found

    async def progress(self, message: str):
        """Sends a progress update event if the emitter is available."""
        if self.__current_event_emitter__ and callable(self.__current_event_emitter__):
            try:
                await self.__current_event_emitter__({
                    "type": "status",
                    "data": {"level": "info", "description": str(message), "done": False},
                })
            except Exception as e:
                self.logger.error(f"Emit progress failed: {e}", exc_info=self.debug_logging)

    async def done(self):
        """Sends a completion status event."""
        if self.__current_event_emitter__ and callable(self.__current_event_emitter__):
            try:
                await self.__current_event_emitter__({
                    "type": "status",
                    "data": {"level": "info", "description": "Processing Complete.", "done": True},
                })
            except Exception as e:
                self.logger.error(f"Emit done failed: {e}", exc_info=self.debug_logging)

    async def emit_message(self, message: str):
        """Sends a message chunk event."""
        if self.__current_event_emitter__ and callable(self.__current_event_emitter__):
            try:
                await self.__current_event_emitter__({
                    "type": "message", "data": {"content": str(message)}
                })
            except Exception as e:
                self.logger.error(f"Emit message failed ('{truncate_text(message, 50)}'): {e}", exc_info=self.debug_logging)

    async def _call_llm_base( self, payload: Dict[str, Any] ) -> Tuple[Optional[Any], Optional[str]]:
        """Internal helper to call the top-level ollama endpoint function."""
        response = None
        error_message = None
        try:
            # Call the TOP-LEVEL utility function, passing necessary context
            response = await call_ollama_endpoint( payload, self.logger, self.__user__, self.debug_logging )

            # Check response for errors (utility handles structure)
            if isinstance(response, dict) and response.get("error"):
                # Call the TOP-LEVEL utility function to extract error
                error_msg = get_response_content(response, self.logger)
                self.logger.error(f"LLM call failed: {error_msg}.")
                error_message = error_msg
                response = None # Nullify response on error

            return response, error_message
        except Exception as e:
            self.logger.error(f"_call_llm_base error: {e}", exc_info=self.debug_logging)
            return None, f"Error: LLM call exception ({type(e).__name__})."


    async def get_completion(self, model: str, messages: List[Dict[str, str]]) -> str:
        """Gets a non-streaming completion from the LLM."""
        response = None
        error_message = None
        try:
            model_to_use = model or self.__model__ # Use specified model or instance default
            if not model_to_use: return "Error: LLM model name missing."

            payload = {"model": model_to_use, "messages": messages, "stream": False}
            # Call internal base method which calls top-level endpoint function
            response, error_message = await self._call_llm_base(payload)

            if error_message: return error_message # Return error if call failed
            if response is None: return "Error: LLM communication failure (no response)."

            # Call TOP-LEVEL utility function to extract content
            content = get_response_content(response, self.logger)
            # Return content if valid, otherwise empty string (get_response_content handles internal errors)
            return content if content and not content.startswith("Error:") else (content or "")

        except Exception as e:
             self.logger.error(f"get_completion error: {e}", exc_info=self.debug_logging)
             return f"Error: get_completion failed ({type(e).__name__})."
        finally:
             # Resource cleanup (e.g., close response if needed, though less common for non-streaming)
            if response and hasattr(response, 'aclose') and callable(response.aclose):
                 try: await response.aclose()
                 except Exception as e: self.logger.error(f"Error aclosing response: {e}")
            elif response and hasattr(response, 'release') and callable(response.release):
                 try:
                      await response.release() if asyncio.iscoroutinefunction(response.release) else response.release()
                 except Exception as e: self.logger.error(f"Error releasing response: {e}")


    async def get_streaming_completion( self, model: str, messages: List[Dict[str, str]] ) -> AsyncGenerator[str, None]:
        """Gets a streaming completion from the LLM."""
        response = None
        error_message = None
        try:
            model_to_use = model or self.__model__
            if not model_to_use: yield "Error: LLM model name missing."; return

            payload = {"model": model_to_use, "messages": messages, "stream": True}
            # Call internal base method which calls top-level endpoint function
            response, error_message = await self._call_llm_base(payload)

            if error_message: yield error_message; return
            if response is None: yield "Error: LLM communication failure (no response)."; return

            # Check if response is streamable
            if hasattr(response, "body_iterator"):
                async for chunk_bytes in response.body_iterator:
                    # Call TOP-LEVEL utility function to parse chunk
                    for part in get_chunk_content( chunk_bytes, self.logger, self.debug_logging ):
                        yield part
            elif isinstance(response, dict): # Handle non-stream dict response (e.g., error?)
                # Call TOP-LEVEL utility function
                content = get_response_content(response, self.logger)
                yield content if content and not content.startswith("Error:") else (content or "Error: Invalid dict response")
            else:
                yield f"Error: Unexpected LLM response type ({type(response).__name__})."

        except Exception as e:
            self.logger.error(f"Streaming error: {e}", exc_info=self.debug_logging)
            yield f"Error: Streaming failed ({type(e).__name__})."
        finally:
            # Ensure stream resources are released
            if response and hasattr(response, 'aclose') and callable(response.aclose):
                 try: await response.aclose()
                 except Exception as e: self.logger.error(f"Error aclosing stream: {e}")
            elif response and hasattr(response, 'release') and callable(response.release):
                 try:
                      await response.release() if asyncio.iscoroutinefunction(response.release) else response.release()
                 except Exception as e: self.logger.error(f"Error releasing stream: {e}")


    # --- MCTS Interface methods (implemented by Pipe) ---
    async def generate_thought( self, current_analysis: str, context: Dict, config_dict: Dict ) -> str:
        """Generates a thought/critique using the LLM."""
        try:
            # Prepare context, ensuring all keys used by the prompt format string are present
            required_keys = [
                "current_answer", "question_summary", "best_answer", "best_score",
                "previous_best_summary", "unfit_markers_summary", "learned_approach_summary",
                "explored_approaches", "sibling_approaches", "current_sequence", "current_tags"
            ]
            # Use context directly, assuming get_context_for_node provided defaults
            formatted_context = {key: context.get(key, "N/A") for key in required_keys}
            formatted_context.update({ k: v for k, v in context.items() if k in required_keys }) # Override defaults

            prompt = GENERATE_THOUGHT_PROMPT.format(**formatted_context)
            # Use internal get_completion method
            return await self.get_completion( self.__model__, [{"role": "user", "content": prompt}] )
        except KeyError as e:
             self.logger.error(f"Generate thought prompt format fail (KeyError: {e}). Context keys: {list(context.keys())}")
             return f"Error: Prompt format fail (key: {e})."
        except Exception as e:
             self.logger.error(f"Generate thought failed: {e}", exc_info=self.debug_logging)
             return f"Error: Thought generation failed ({type(e).__name__})."

    async def update_approach( self, original_analysis: str, critique: str, context: Dict, config_dict: Dict ) -> str:
        """Updates analysis based on a critique using the LLM."""
        prompt_args = context.copy()
        prompt_args["answer"] = original_analysis
        prompt_args["improvements"] = critique.strip()
        # Ensure required keys have defaults
        required_keys = [
            "question_summary", "best_answer", "best_score", "current_tags",
            "previous_best_summary", "unfit_markers_summary", "answer", "improvements"
        ]
        for key in required_keys: prompt_args.setdefault(key, "N/A")

        try:
            prompt = UPDATE_ANALYSIS_PROMPT.format(**prompt_args)
            # Use internal get_completion
            llm_result = await self.get_completion( self.__model__, [{"role": "user", "content": prompt}] )

            # Fallback to original analysis on error
            if llm_result.startswith("Error:"):
                self.logger.warning(f"Update approach LLM call failed: {llm_result}. Falling back to original.")
                return str(original_analysis)

            # Clean result (remove potential markdown blocks)
            cleaned_result = re.sub(r"^\s*```[\s\S]*?```\s*$", "", llm_result, flags=re.MULTILINE).strip()
            cleaned_result = re.sub(r"^\s*```(.*?)\n", "", cleaned_result, flags=re.IGNORECASE | re.MULTILINE).strip()
            cleaned_result = re.sub(r"\n```\s*$", "", cleaned_result, flags=re.MULTILINE).strip()

            # Return cleaned result or fallback if empty
            return cleaned_result if cleaned_result else str(original_analysis)
        except KeyError as e:
             self.logger.error(f"Update approach prompt format fail (KeyError: {e}). Args: {list(prompt_args.keys())}")
             return str(original_analysis) # Fallback
        except Exception as e:
             self.logger.error(f"Update approach failed: {e}", exc_info=self.debug_logging)
             return str(original_analysis) # Fallback

    async def evaluate_answer( self, analysis_to_evaluate: str, context: Dict, config_dict: Dict ) -> Union[int, str]:
        """Evaluates analysis content using the LLM, returning a score (1-10) or error string."""
        prompt_args = context.copy()
        prompt_args["answer_to_evaluate"] = analysis_to_evaluate
        # Ensure required keys have defaults
        required_keys = [
            "question_summary", "best_answer", "best_score", "current_tags",
            "previous_best_summary", "unfit_markers_summary", "answer_to_evaluate"
        ]
        for key in required_keys: prompt_args.setdefault(key, "N/A")

        try:
            prompt = EVALUATE_ANALYSIS_PROMPT.format(**prompt_args)
            # Use internal get_completion
            result_str = await self.get_completion( self.__model__, [{"role": "user", "content": prompt}] )

            # Handle LLM error first
            if result_str.startswith("Error:"): return result_str

            # --- Score Parsing Logic ---
            cleaned_str = result_str.strip()
            # 1. Try strict match (just the number 1-10)
            strict_match = re.search(r"^\s*([1-9]|10)\s*$", cleaned_str)
            if strict_match:
                try: return int(strict_match.group(1))
                except ValueError: pass # Should not happen with regex, but handle anyway

            # 2. Try relaxed match (find first number 1-10 in the string)
            relaxed_match = re.search(r"\b([1-9]|10)\b", cleaned_str)
            if relaxed_match:
                 try:
                      score = int(relaxed_match.group(1))
                      self.logger.warning(f"Evaluator used relaxed parsing for score '{score}' from response: '{cleaned_str[:100]}...'")
                      return score
                 except ValueError: pass

            # 3. Failed to parse
            err_msg = f"Error: Failed to parse score from LLM response: '{cleaned_str[:100]}...'"
            self.logger.error(err_msg)
            return err_msg # Return error string if parsing failed

        except KeyError as e:
            self.logger.error(f"Evaluate answer prompt format fail (KeyError: {e}). Args: {list(prompt_args.keys())}")
            return f"Error: Prompt format fail (key: {e})."
        except Exception as e:
             self.logger.error(f"Evaluate answer failed: {e}", exc_info=self.debug_logging)
             return f"Error: Evaluation failed ({type(e).__name__})."


    # --- Intent Handling Methods ---
    async def _classify_intent(self, text: str) -> str:
        """Classifies user intent using LLM."""
        default_intent = "ANALYZE_NEW"
        if not text: return default_intent
        try:
            prompt = INTENT_CLASSIFIER_PROMPT.format(raw_input_text=text)
            # Use internal get_completion
            response = await self.get_completion( self.__model__, [{"role": "user", "content": prompt}] )

            if response.startswith("Error:"):
                 self.logger.warning(f"Intent classification LLM failed: {response}. Defaulting.")
                 return default_intent # Default on LLM error

            # Basic validation and cleaning
            valid_intents = {"ANALYZE_NEW", "CONTINUE_ANALYSIS", "ASK_LAST_RUN_SUMMARY", "ASK_PROCESS", "ASK_CONFIG", "GENERAL_CONVERSATION"}
            cleaned_response = ""
            # Take first word of response, uppercase, remove trailing punctuation
            if response and isinstance(response, str):
                parts = response.strip().upper().split(maxsplit=1)
                if parts: cleaned_response = re.sub(r"[.,!?;:]$", "", parts).strip()

            if cleaned_response in valid_intents:
                 self.logger.info(f"Intent classified as: {cleaned_response}")
                 return cleaned_response
            else: # Fallback heuristics if LLM response is unclear
                self.logger.warning(f"Intent classification returned unexpected: '{response}'. Using fallback heuristics.")
                text_lower = text.lower()
                if any(kw in text_lower for kw in ["how", "explain", "process", "mcts", "work"]): return "ASK_PROCESS"
                if any(kw in text_lower for kw in ["config", "setting", "parameter", "valve"]): return "ASK_CONFIG"
                if any(kw in text_lower for kw in ["last run", "summary", "result", "score"]): return "ASK_LAST_RUN_SUMMARY"
                if any(kw in text_lower for kw in ["continue", "elaborate", "further", "more", "refine", "what about"]): return "CONTINUE_ANALYSIS"
                return default_intent # Default if no heuristics match

        except Exception as e:
            self.logger.error(f"Intent classification error: {e}", exc_info=self.debug_logging)
            return default_intent # Default on any exception


    async def _handle_ask_process(self):
        """Handles the ASK_PROCESS intent."""
        try:
            explanation = ASK_PROCESS_EXPLANATION.format(db_file_name=(os.path.basename(DB_FILE) if DB_FILE else "N/A"))
            await self.emit_message(f"**About My Process (MCTS v{SCRIPT_VERSION}):**\n{explanation}")
        except Exception as e:
            self.logger.error(f"Error handling ASK_PROCESS: {e}")
            await self.emit_message("**Error:** Could not explain process.")
        finally:
            await self.done() # Ensure done is called

    async def _handle_ask_config(self):
        """Handles the ASK_CONFIG intent."""
        try:
            # Use the config applied for the *current* run
            config_str = json.dumps(self.current_config, indent=2)
            await self.emit_message(f"**Current Config:**\n```json\n{config_str}\n```\n")
        except Exception as e:
            self.logger.error(f"Error handling ASK_CONFIG: {e}")
            await self.emit_message("**Error:** Could not retrieve configuration.")
        finally:
            await self.done() # Ensure done is called

    async def _handle_ask_last_run_summary(self, state: Optional[Dict]):
        """Handles the ASK_LAST_RUN_SUMMARY intent."""
        if not state or not isinstance(state, dict):
            await self.emit_message("No saved summary found for this chat session.")
            await self.done()
            return

        lines = ["**Summary of Last Saved Run:**"]
        try:
            # Extract and format info safely from state dict
            score = state.get("best_score")
            lines.append(f"- Best Score: {score:.1f}/10" if isinstance(score, (int, float)) else f"- Best Score: {score or 'N/A'}")
            tags = state.get("best_node_tags")
            lines.append(f"- Best Tags: {tags}" if tags else "- Best Tags: N/A")
            summary = state.get("best_solution_summary")
            lines.append(f"- Best Summary: {summary or 'N/A'}")

            # Format learned preferences if available (Bayesian example)
            priors = state.get("approach_priors")
            if priors and isinstance(priors.get("alpha"), dict) and isinstance(priors.get("beta"), dict):
                means = {}
                alphas = priors["alpha"]
                betas = priors["beta"]
                for app, a_val in alphas.items():
                    b_val = betas.get(app, 1.0)
                    try:
                        a_f, b_f = max(1e-9, float(a_val)), max(1e-9, float(b_val))
                        denominator = a_f + b_f
                        means[app] = (a_f / denominator * 10.0) if denominator > 1e-9 else -1
                    except (ValueError, TypeError): means[app] = -1
                valid = { k: v for k, v in means.items() if v >= 0 and k not in ["initial", "variant", "unknown"] }
                top = sorted(valid.items(), key=lambda i: i, reverse=True)[:3]
                prefs = (f"{', '.join([f'{a}({s:.1f})' for a,s in top])}" + ("..." if len(valid) > 3 else "")) if top else "None"
                lines.append(f"- Learned Prefs: {prefs}")
            # TODO: Add handling for non-Bayesian scores if needed
            else: lines.append("- Learned Prefs: N/A")

            unfit = state.get("unfit_markers", [])
            lines.append(f"- Unfit Markers: {len(unfit)} found" if unfit else "- Unfit Markers: None")

            top_nodes = state.get("top_nodes", [])
            if top_nodes: lines.append("- Top Nodes:")
            for i, ns in enumerate(top_nodes): # Format top node summaries
                seq = ns.get("sequence", "?")
                scr = ns.get("score", "?")
                scr_f = f"{scr:.1f}" if isinstance(scr, (int, float)) else scr
                tgs = ns.get("tags", [])
                sumry = ns.get("content_summary", "?")
                lines.append(f"  {i+1}. N{seq} (S:{scr_f}, T:{tgs}): '{sumry}'")

            await self.emit_message("\n".join(lines))
        except Exception as e:
            self.logger.error(f"Error formatting last run summary: {e}", exc_info=self.debug_logging)
            await self.emit_message("\n".join(lines) + "\n\n**Error:** Could not format summary.")
        finally:
            await self.done() # Ensure done is called

    async def _handle_general_conversation( self, user_input: str ):
        """Handles the GENERAL_CONVERSATION intent."""
        try:
            prompt = GENERAL_CONVERSATION_PROMPT.format(user_input=user_input)
            # Use internal get_completion
            response = await self.get_completion( self.__model__, [{"role": "user", "content": prompt}] )
            await self.emit_message( response if response and not response.startswith("Error:") else "How can I help with analysis?" )
        except Exception as e:
            self.logger.error(f"Error handling general conversation: {e}", exc_info=self.debug_logging)
            await self.emit_message("Sorry, there was an issue responding.")
        finally:
            await self.done() # Ensure done is called

    # --- Main Pipe Logic Sections (as internal methods) ---
    async def _initialize_run(self) -> Tuple[bool, str, Optional[str]]:
        """Initializes state for a new pipe run (config, model, input, chat_id)."""
        self.logger.info(f"--- Initializing Run: {self.id} v{SCRIPT_VERSION} ---")
        # Reset run-specific state
        self.current_config = {}
        self.__model__ = ""
        self.__chat_id__ = None
        self.debug_logging = False # Reset debug flag

        # Store user/emitter from pipe call
        # self.__current_event_emitter__ and self.__user__ are set in pipe() entry

        # Resolve Chat ID
        self.__chat_id__ = self.__request_body__.get("chat_id")
        self.logger.info(f"Chat ID for this run: {self.__chat_id__ or 'N/A'}.")

        # Resolve Model
        self.__model__ = self.resolve_model(self.__request_body__)
        if not self.__model__:
            await self.emit_message("Error: Could not resolve base LLM model name.")
            await self.done()
            return (False, "", None) # Indicate failure
        self.logger.info(f"Using base model: {self.__model__}")

        # Resolve User Input (using internal helper)
        user_input_text = self._resolve_question(self.__request_body__)

        # Check if it's a title generation task (often has minimal messages)
        is_title_task = OPENWEBUI_IMPORTS_AVAILABLE and TASKS and self.__request_body__.get("task") == TASKS.TITLE_GENERATION
        if not user_input_text and is_title_task:
            # Try getting prompt directly for title task if messages are empty
            user_input_text = str(self.__request_body__.get("prompt", "")).strip()
            self.logger.info("Resolved input from 'prompt' field for Title Generation task.")

        # Final check for input
        if not user_input_text:
            await self.emit_message("Error: No user input found in messages or prompt.")
            await self.done()
            return False, "", self.__chat_id__

        # Apply Valve Settings to current_config
        self.current_config = DEFAULT_CONFIG.copy() # Start with defaults
        try:
            request_valves = self.__request_body__.get("valves")
            if request_valves and isinstance(request_valves, dict):
                # Validate and apply valves
                self.valves = self.Valves(**request_valves) # Validate against Valve model
                validated_valve_dict = self.valves.model_dump()
                for key_upper, value in validated_valve_dict.items():
                    key_lower = key_upper.lower()
                    if key_lower in self.current_config:
                        self.current_config[key_lower] = value
                self.logger.info("Applied valve settings from request.")
            else:
                self.valves = self.Valves() # Use default valves if none provided
                self.current_config = DEFAULT_CONFIG.copy() # Ensure config matches default valves
                self.logger.info("No valves in request, using default settings.")

            # Sanitize critical numeric configs after applying valves
            self.current_config["beta_prior_alpha"] = max(1e-9, float(self.current_config.get("beta_prior_alpha", 1.0)))
            self.current_config["beta_prior_beta"] = max(1e-9, float(self.current_config.get("beta_prior_beta", 1.0)))
            # Add other numeric validations if needed

            # Set debug logging flag for this run
            self.debug_logging = self.current_config.get("debug_logging", False)
            new_log_level = logging.DEBUG if self.debug_logging else logging.INFO
            self.logger.setLevel(new_log_level)
            self.mcts_logger.setLevel(new_log_level)
            for handler in self.logger.handlers + self.mcts_logger.handlers:
                 handler.setLevel(new_log_level)
            self.logger.info(f"Log level set for run: {'DEBUG' if self.debug_logging else 'INFO'}")

        except ValidationError as e:
             self.logger.error(f"Valve validation error: {e}. Using defaults.", exc_info=True)
             await self.emit_message(f"Warning: Invalid settings provided. Using defaults.\nDetails: {e}")
             self.valves = self.Valves() # Revert to default Valves
             self.current_config = DEFAULT_CONFIG.copy() # Revert to default config
             self.debug_logging = False
             self.logger.setLevel(logging.INFO); self.mcts_logger.setLevel(logging.INFO)
        except Exception as e:
            self.logger.error(f"Error applying valve settings: {e}. Using defaults.", exc_info=True)
            await self.emit_message(f"Warning: Error applying settings ({type(e).__name__}). Using defaults.")
            self.valves = self.Valves(); self.current_config = DEFAULT_CONFIG.copy(); self.debug_logging = False
            self.logger.setLevel(logging.INFO); self.mcts_logger.setLevel(logging.INFO)

        self.logger.info("Pipe initialization complete.")
        return True, user_input_text, self.__chat_id__

    async def _determine_intent_and_load_state( self, user_input_text: str ) -> Tuple[str, Optional[Dict], bool]:
        """Determines user intent and loads state if applicable."""
        # 1. Classify Intent
        intent = await self._classify_intent(user_input_text)
        self.logger.info(f"Determined Intent: {intent}")

        # 2. Check if State Persistence is Enabled for this Run
        state_persistence_enabled_by_config = self.current_config.get("enable_state_persistence", True)
        # Actual state enabled status depends on config AND chat_id presence
        state_is_enabled_for_run: bool = state_persistence_enabled_by_config and bool(self.__chat_id__)

        # 3. Load State if Intent Requires it and State is Enabled
        loaded_state: Optional[Dict] = None
        if state_is_enabled_for_run and intent in ["CONTINUE_ANALYSIS", "ASK_LAST_RUN_SUMMARY"]:
            self.logger.info(f"Attempting to load state for chat '{self.__chat_id__}' for intent '{intent}'.")
            try:
                # Call top-level DB function
                loaded_state = load_mcts_state( DB_FILE, self.__chat_id__, self.logger )

                # Validate loaded state (version check)
                if loaded_state:
                    state_version = loaded_state.get("version", "0")
                    required_version = SCRIPT_VERSION # Compare against current script version
                    if state_version != required_version:
                        self.logger.warning(f"Loaded state version '{state_version}' incompatible with script version '{required_version}'. Discarding state.")
                        await self.emit_message(f"**Warning:** Previous state (v{state_version}) is incompatible with current script (v{required_version}). Starting fresh analysis.")
                        loaded_state = None # Discard incompatible state
                        # If intent was to continue, force new analysis
                        if intent == "CONTINUE_ANALYSIS": intent = "ANALYZE_NEW"
                elif intent == "CONTINUE_ANALYSIS":
                    # If trying to continue but no state was found
                    self.logger.info("Intent was CONTINUE_ANALYSIS, but no state found. Switching to ANALYZE_NEW.")
                    await self.emit_message("**Info:** No previous analysis state found. Starting a new analysis.")
                    intent = "ANALYZE_NEW"

            except Exception as e: # Catch errors during load/validation
                self.logger.error(f"Error loading/validating state for chat '{self.__chat_id__}': {e}", exc_info=self.debug_logging)
                await self.emit_message("**Warning:** Error loading previous state. Starting fresh analysis.")
                loaded_state = None # Ensure state is None on error
                if intent == "CONTINUE_ANALYSIS": intent = "ANALYZE_NEW" # Force new if load failed

        # If intent ended up as CONTINUE but state is None, switch to NEW
        if intent == "CONTINUE_ANALYSIS" and not loaded_state:
             self.logger.info("Correcting intent to ANALYZE_NEW as CONTINUE_ANALYSIS requires loaded state.")
             intent = "ANALYZE_NEW"

        return intent, loaded_state, state_is_enabled_for_run


    async def _handle_intent( self, intent: str, user_input: str, loaded_state: Optional[Dict] ) -> bool:
        """Dispatches to specific handlers for non-analysis intents."""
        intent_handlers = {
            "ASK_PROCESS": self._handle_ask_process,
            "ASK_CONFIG": self._handle_ask_config,
            # Lambda needed to pass loaded_state correctly
            "ASK_LAST_RUN_SUMMARY": lambda: self._handle_ask_last_run_summary(loaded_state),
            # Lambda needed to pass user_input
            "GENERAL_CONVERSATION": lambda: self._handle_general_conversation(user_input),
        }

        if intent in intent_handlers:
            await intent_handlers[intent]() # Call the appropriate handler
            return True # Indicate intent was handled directly
        elif intent in ["ANALYZE_NEW", "CONTINUE_ANALYSIS"]:
            return False # Indicate MCTS analysis should proceed
        else: # Should not happen if classify_intent is correct
            self.logger.error(f"Unhandled intent encountered in _handle_intent: {intent}")
            await self.emit_message(f"**Error:** Internal error processing intent '{intent}'.")
            await self.done()
            return True # Indicate processing should stop


    async def _run_mcts_analysis( self, intent: str, user_input: str, loaded_state: Optional[Dict] ) -> Optional["MCTS"]:
        """Runs the core MCTS analysis process."""
        run_type_msg = "Continuing analysis (using loaded state)" if intent == "CONTINUE_ANALYSIS" and loaded_state else "Starting new analysis"
        await self.emit_message( f'# {self.name} v{SCRIPT_VERSION}\n*Analyzing:* "{truncate_text(user_input, 100)}" *using* `{self.__model__}`.\n🚀 **{run_type_msg}...**' )
        if self.current_config.get("show_processing_details"): await self.emit_message("*(Verbose details enabled...)*\n")

        initial_analysis_text = ""
        mcts_instance: Optional["MCTS"] = None

        # 1. Generate Initial Analysis (Always run to set root node content)
        try:
            await self.progress("Generating initial analysis...")
            initial_prompt_formatted = INITIAL_ANALYSIS_PROMPT.format(question=user_input)
            initial_response = await self.get_completion( self.__model__, [{"role": "user", "content": initial_prompt_formatted}] )

            if initial_response.startswith("Error:"):
                await self.emit_message(f"Error: MCTS start failed (initial analysis generation).\nDetails: {initial_response}")
                await self.done(); return None # Stop if initial analysis fails

            # Clean initial analysis text
            initial_analysis_text = re.sub(r"^\s*```[\s\S]*?```\s*$", "", initial_response, flags=re.MULTILINE).strip()
            initial_analysis_text = re.sub(r"^\s*```(.*?)\n", "", initial_analysis_text, flags=re.IGNORECASE | re.MULTILINE).strip()
            initial_analysis_text = re.sub(r"\n```\s*$", "", initial_analysis_text, flags=re.MULTILINE).strip()

            if not initial_analysis_text:
                await self.emit_message("Error: Initial analysis generated empty content."); await self.done(); return None

            await self.emit_message(f"\n## Initial Analysis\n{initial_analysis_text}\n\n*{'-'*20} Starting MCTS... {'-'*20}*\n")
            await asyncio.sleep(0.1) # Small delay for readability

        except Exception as e:
            self.logger.error(f"Error during initial analysis generation: {e}", exc_info=self.debug_logging)
            await self.emit_message(f"Error generating initial analysis: {type(e).__name__}.")
            await self.done(); return None

        # 2. Initialize MCTS Instance
        try:
            await self.progress("Initializing MCTS engine...")
            # Pass loaded state only if continuing
            state_to_pass = loaded_state if intent == "CONTINUE_ANALYSIS" else None
            mcts_instance = MCTS(
                llm_interface=self, # Pass self (Pipe instance)
                question=user_input,
                mcts_config=self.current_config, # Pass current run config
                initial_analysis_content=initial_analysis_text, # Root node content
                initial_state=state_to_pass, # Loaded state dict or None
                model_body=self.__request_body__, # Pass request body for model resolution
            )
            if not mcts_instance.root: raise RuntimeError("MCTS init failed - Root node is None.") # Critical check
            self.logger.info("MCTS instance created successfully.")

        except Exception as e:
            self.logger.critical(f"MCTS Initialization failed: {e}", exc_info=True)
            await self.emit_message(f"**FATAL ERROR:** Failed to initialize MCTS analysis engine.\nDetails: {type(e).__name__}: {e}")
            await self.done(); return None

        # 3. Run MCTS Search Loop
        iterations_run = 0
        try:
            should_continue_search = True
            max_iterations = self.current_config.get("max_iterations", 1)
            sims_per_iteration = self.current_config.get("simulations_per_iteration", 10)

            for iteration in range(max_iterations):
                if not should_continue_search: break # Check if search signaled early stop

                iterations_run = iteration + 1
                await self.progress(f"Running MCTS Iteration {iterations_run}/{max_iterations}...")
                # Run the search for one iteration
                should_continue_search = await mcts_instance.search(sims_per_iteration)
                # Optional: Add a small delay between iterations if needed
                # await asyncio.sleep(0.05)

            mcts_instance.iterations_completed = iterations_run # Record actual iterations run
            self.logger.info(f"MCTS search finished. Ran {iterations_run} iterations, {mcts_instance.simulations_completed} total simulations. Best Score: {mcts_instance.best_score:.1f}")
            await self.emit_message("\n🏁 **MCTS search complete.** Finalizing...")
            return mcts_instance # Return the completed MCTS instance

        except Exception as e:
            self.logger.error(f"Error during MCTS Search (Iteration {iterations_run}): {e}", exc_info=self.debug_logging)
            await self.emit_message(f"**Warning:** Error occurred during MCTS search (iteration {iterations_run}). Results may be incomplete.")
            # Return the instance even if search failed partway, for partial results/state saving
            return mcts_instance


    async def _finalize_run( self, mcts_instance: Optional["MCTS"], initial_analysis_text: str, state_is_enabled: bool ):
        """Generates final output, synthesis, and saves state."""
        if not mcts_instance:
            await self.emit_message("Error: MCTS analysis instance is missing. Cannot finalize.")
            await self.done()
            return

        try:
            # --- Determine Final Analysis Text ---
            final_analysis_text = initial_analysis_text # Default fallback
            cleaned_best = ""
            if mcts_instance.best_solution: # Check if a best solution was found
                # Clean the best solution text
                cleaned_best = re.sub(r'\s+', ' ', str(mcts_instance.best_solution).strip(), flags=re.MULTILINE)
            if cleaned_best: final_analysis_text = cleaned_best # Use cleaned best solution if valid

            # --- Generate Summary (Verbose or Quiet) ---
            if self.current_config.get("show_processing_details"):
                await self.progress("Generating verbose summary...")
                verbose_summary = mcts_instance.formatted_output() # MCTS method
                await self.emit_message(verbose_summary)
            else: # Quiet mode: Just show best analysis
                await self.progress("Extracting best analysis...")
                best_node_final = mcts_instance.find_best_final_node()
                final_tags = best_node_final.descriptive_tags[:] if best_node_final else []
                simple_summary = f"## Best Analysis (Score: {mcts_instance.best_score:.1f}/10)\n**Tags: {final_tags}**\n\n{final_analysis_text}\n"
                await self.emit_message(simple_summary)

            # --- Generate Final Synthesis ---
            await self.progress("Generating final synthesis...")
            synthesis_context = mcts_instance.get_final_synthesis_context() # MCTS method
            if synthesis_context:
                try:
                    synthesis_prompt = FINAL_SYNTHESIS_PROMPT.format(**synthesis_context)
                    # Use internal get_completion
                    synthesis_response = await self.get_completion( self.__model__, [{"role": "user", "content": synthesis_prompt}] )

                    if synthesis_response.startswith("Error:"):
                         await self.emit_message(f"\n***\n## Final Synthesis\n**Warning:** Could not generate synthesis ({synthesis_response}).")
                    else: # Clean and emit synthesis
                        cleaned_synthesis = re.sub(r"^\s*```[\s\S]*?```\s*$", "", synthesis_response, flags=re.MULTILINE).strip()
                        cleaned_synthesis = re.sub(r"^\s*```(.*?)\n", "", cleaned_synthesis, flags=re.IGNORECASE | re.MULTILINE).strip()
                        cleaned_synthesis = re.sub(r"\n```\s*$", "", cleaned_synthesis, flags=re.MULTILINE).strip()
                        await self.emit_message(f"\n***\n## Final Synthesis\n{cleaned_synthesis or '(Empty.)'}")
                except Exception as e:
                    self.logger.error(f"Final synthesis LLM call failed: {e}", exc_info=self.debug_logging)
                    await self.emit_message(f"\n***\n## Final Synthesis\n**Error:** Failed to generate synthesis ({type(e).__name__}).")
            else:
                await self.emit_message("\n***\n## Final Synthesis\n**Error:** Internal error preparing context for synthesis.")

            # --- Save State ---
            if state_is_enabled and self.__chat_id__:
                await self.progress("Saving final MCTS state...")
                try:
                    state_to_save = mcts_instance.get_state_for_persistence() # MCTS method
                    if state_to_save and isinstance(state_to_save, dict):
                        # Call top-level DB function
                        save_mcts_state( DB_FILE, self.__chat_id__, state_to_save, self.logger )
                    else:
                        self.logger.error("State generation failed or returned invalid type. State not saved.")
                except Exception as e:
                     self.logger.error(f"Error saving state for chat {self.__chat_id__}: {e}", exc_info=self.debug_logging)
                     await self.emit_message("**Warning:** Failed to save analysis state.")

            # --- Signal Completion ---
            await self.done()

        except Exception as e:
            self.logger.critical(f"Finalization error: {e}", exc_info=True)
            await self.emit_message(f"\n**CRITICAL ERROR:** Finalizing run failed ({type(e).__name__}). Check logs.")
            await self.done() # Still call done on error


    # --- Main Pipe Entry Point ---
    async def pipe(
        self,
        # Using the newer signature - ensure OWB provides these
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict, # Keep body for now for chat_id, valves etc.
        __user__: Optional[Dict] = None,
        __event_emitter__: Optional[Callable] = None,
        __task__: Optional[str] = None,
    ) -> Union[str, None, AsyncGenerator[str, None]]:
        """Main entry point for the pipeline."""
        self.__current_event_emitter__ = __event_emitter__
        self.__user__ = __user__ if __user__ is not None else AdminUserMock()
        self.__request_body__ = body # Store full body
        mcts_instance: Optional["MCTS"] = None
        initial_analysis_text_for_finalize: str = ""
        state_enabled_for_run: bool = False
        pipe_id_log = self.id # For logging context

        try:
            # 1. Initialize Run (sets config, model, chat_id, gets input)
            # Note: _initialize_run uses self._resolve_question(body) internally.
            # If user_message arg should be primary, _initialize_run needs adjustment.
            init_success, user_input, chat_id = await self._initialize_run()
            if not init_success: return None # Error already emitted

            # 2. Determine Intent & Load State
            intent, loaded_state, state_enabled_for_run = await self._determine_intent_and_load_state(user_input)

            # 3. Handle Non-Analysis Intents
            intent_handled_directly = await self._handle_intent(intent, user_input, loaded_state)
            if intent_handled_directly: return None # Response handled by emitter

            # 4. Handle Title Generation Task (if applicable)
            is_title_task = OPENWEBUI_IMPORTS_AVAILABLE and TASKS and __task__ == TASKS.TITLE_GENERATION
            if is_title_task:
                self.logger.info(f"Handling TITLE_GENERATION task for: {truncate_text(user_input)}")
                title_prompt = f"Generate concise title (max 10 words): {user_input}"
                title_response = await self.get_completion(self.__model__, [{"role": "user", "content": title_prompt}])
                # Clean title response
                cleaned_title = re.sub(r"^\s*[\"\'`*#\-]+\s*|\s*[\"\'`*#\-]+\s*$", "", title_response.strip())
                final_title = truncate_text(cleaned_title, 70) if not title_response.startswith("Error:") else "Title Error"
                await self.done()
                return final_title # Return title string directly

            # 5. Run MCTS Analysis (for relevant intents)
            if intent in ["ANALYZE_NEW", "CONTINUE_ANALYSIS"]:
                mcts_instance = await self._run_mcts_analysis(intent, user_input, loaded_state)
                # Store initial analysis text for potential fallback in finalize
                if mcts_instance and mcts_instance.root:
                    initial_analysis_text_for_finalize = mcts_instance.root.content
                elif not mcts_instance: # If MCTS failed to even initialize
                    # Error should have been emitted by _run_mcts_analysis
                    return None # Stop processing
            else: # Should be caught by _handle_intent, but as safety net
                self.logger.error(f"Unhandled intent '{intent}' reached main execution logic.")
                await self.emit_message(f"Internal Error: Unexpected intent '{intent}'.")
                await self.done()
                return None

            # 6. Finalize Run (if MCTS instance exists)
            if mcts_instance:
                await self._finalize_run( mcts_instance, initial_analysis_text_for_finalize, state_enabled_for_run )
            else: # Ensure 'done' is called if MCTS failed/skipped
                await self.done()

            return None # Indicate successful completion (output via emitter)

        except Exception as e:
            self.logger.critical(f"FATAL pipe error '{pipe_id_log}': {e}", exc_info=True)
            error_message = f"\n**FATAL ERROR in {pipe_id_log}:**\n{type(e).__name__}: {e}"
            try: # Safely try to emit error
                if self.__current_event_emitter__:
                    await self.emit_message(error_message); await self.done()
            except: pass # Ignore errors during error reporting
            return None # Suppress default OWB error handling if we emitted our own
        finally:
            # --- Cleanup ---
            self.__current_event_emitter__ = None
            self.__user__ = None
            self.__request_body__ = {}
            self.__model__ = ""
            self.__chat_id__ = None
            self.current_config = {}
            # Explicitly delete large objects and collect garbage
            if 'mcts_instance' in locals() and mcts_instance:
                try:
                    if hasattr(mcts_instance, 'root'): mcts_instance.root = None
                    if hasattr(mcts_instance, 'thought_history'): mcts_instance.thought_history = []
                    del mcts_instance
                except Exception as gc_err: self.logger.warning(f"Error during MCTS instance cleanup: {gc_err}")
            gc.collect()
            self.logger.debug(f"Pipe '{pipe_id_log}' request cleanup finished.")


# ==============================================================================
# Final Script Confirmation
# ==============================================================================
logger.info(f"Script {__name__} version {SCRIPT_VERSION} fully loaded successfully.")

# END OF SCRIPT
# ==============================================================================
