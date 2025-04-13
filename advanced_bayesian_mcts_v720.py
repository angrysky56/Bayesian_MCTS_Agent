# -*- coding: utf-8 -*-
"""
title: advanced_bayesian_mcts
version: 0.7.20

author: angrysky56
author_url: https://github.com/angrysky56
Project Link: https://github.com/angrysky56/Bayesian_MCTS_Agent

Where I found my stored functions, replace ty with your user name:
/home/ty/.open-webui/cache/functions

The way I launch openweb-ui:
DATA_DIR=~/.open-webui uvx --python 3.11 open-webui@latest serve
http://localhost:8080

description: >
  Advanced Bayesian MCTS v0.7.20: Defaults to 'quiet' mode, hiding intermediate MCTS steps
  (simulations, iteration summaries) from chat. Shows a simple "Processing..." message.
  Final analysis and synthesis are always shown. A new Valve setting allows enabling
  verbose logging to the chat for explainability.

Key improvements in v0.7.20:
- Quiet Default Mode: By default, only the initial "Processing..." message and the
  final analysis/synthesis are shown in chat.
- Optional Verbose Mode: Added 'Show Detailed MCTS Steps in Chat' Valve (default: False).
  When enabled, intermediate simulation details and iteration summaries are shown in chat.
- Clearer User Expectation: Initial message informs the user processing may take time.
- Maintained features: Iteration Summaries (in verbose mode), Live View Focus (in verbose mode),
  Final Summary Clarity, Synthesis, Strict prompts, robustness fixes, Bayesian options.
- Fix: Returns None from pipe to prevent UI overwrite.

# ... previous versions ...
"""

from fastapi import Request, Response
import logging
import random
import math
import asyncio
import json
import re
import gc
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

# Import aiohttp for potential future session management fix
import aiohttp

from typing import (
    List, Optional, AsyncGenerator, Callable, Awaitable, Generator, Iterator,
    Dict, Any, Tuple, Set, Union
)
from pydantic import BaseModel, Field, field_validator
from open_webui.constants import TASKS
import open_webui.routers.ollama as ollama
from open_webui.main import app

# ==============================================================================

name = "advanced_mcts"

# --- DEFAULT Global Configuration ---
# Changed 'show_simulation_details' to 'show_processing_details' and default to False
default_config = {
    "max_children": 10,
    "exploration_weight": 3.0,
    "max_iterations": 5,
    "simulations_per_iteration": 5,
    "surprise_threshold": 0.66,
    "use_semantic_distance": True,
    "relative_evaluation": True,
    "score_diversity_bonus": 0.7,
    "force_exploration_interval": 4,
    "debug_logging": False, # Console/file logging, independent of chat verbosity
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
    "show_processing_details": False, # Default OFF - Controls chat verbosity during run
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
# ==============================================================================

# --- Prompts (Unchanged) ---
initial_prompt = """<instruction>Provide an initial analysis and interpretation of the core themes, arguments, and potential implications presented in the following text. Identify key concepts. Respond with clear, natural language text ONLY. DO NOT use JSON, markdown code blocks, lists, bullet points, or any other structured formatting. DO NOT ask follow-up questions or engage in conversation.</instruction><question>{question}</question>"""
thoughts_prompt = """<instruction>Critically examine the current analysis below. Suggest a SIGNIFICANTLY DIFFERENT interpretation, identify a MAJOR underlying assumption or weakness, or propose a novel connection to another domain or concept. Push the thinking in a new direction.</instruction><context>Original Text Summary: {question_summary}\nBest Overall Analysis (Score {best_score}/10): {best_answer}\nCurrent Analysis (Node {current_sequence}): {current_answer}\nCurrent Analysis Tags: {current_tags}</context>Generate your critique or alternative direction as a concise natural language sentence or two. Respond ONLY with the critique/suggestion itself. DO NOT add conversational text or questions.</instruction>"""
update_prompt = """<instruction>Substantially revise the draft analysis below to incorporate the core idea from the critique. Develop the analysis further based on this new direction. Output ONLY the revised analysis text itself, as plain natural language. DO NOT include headers, explanations, introductions, conclusions, conversational elements, questions, JSON, markdown, lists, or any structured formatting.</instruction><context>Original Text Summary: {question_summary}\nBest Overall Analysis (Score {best_score}/10): {best_answer}\nCurrent Analysis Tags: {current_tags}</context><draft>{answer}</draft><critique>{improvements}</critique>Write the new, revised analysis text ONLY."""
eval_answer_prompt = """<instruction>Evaluate the intellectual quality and insightfulness of the analysis below (1-10) concerning the original text. Higher scores for depth, novelty, and relevance. Use the full 1-10 scale. Reserve 9-10 for truly exceptional analyses that significantly surpass previous best analysis ({best_score}/10).</instruction><context>Original Text Summary: {question_summary}\nBest Overall Analysis (Score {best_score}/10): {best_answer}\nAnalysis Tags: {current_tags}</context><answer_to_evaluate>{answer_to_evaluate}</answer_to_evaluate>How insightful, deep, relevant, and well-developed is this analysis compared to the best so far? Does it offer a genuinely novel perspective? Rate 1-10 based purely on intellectual merit. Respond ONLY with a single number from 1 to 10.</instruction>"""
tag_generation_prompt = """<instruction>Generate 1-3 concise keyword tags summarizing the main concepts in the following text. Output ONLY the tags, separated by commas (e.g., Tag1, Tag2, Tag3). DO NOT add any other text.</instruction>\n<text_to_tag>{analysis_text}</text_to_tag>"""
final_synthesis_prompt = """<instruction>Synthesize the key insights developed along the primary path of analysis below into a concise, conclusive statement addressing the original question. Focus on the progression of ideas represented by the sequence of 'thoughts'. Respond with clear, natural language text ONLY. DO NOT use JSON, markdown, lists, or ask questions.</instruction>
<original_question_summary>{question_summary}</original_question_summary>
<initial_analysis>{initial_analysis_summary}</initial_analysis>
<best_analysis_score>{best_score}/10</best_analysis_score>
<development_path>
{path_thoughts}
</development_path>
<final_best_analysis>{final_best_analysis_summary}</final_best_analysis>
Synthesize the journey of thoughts into a final conclusion:"""
# ==============================================================================

# Logger Setup Function (Unchanged)
def setup_logger(level=None):
    logger = logging.getLogger(__name__)
    # Use the config value for debug_logging, not show_processing_details
    log_level = level if level is not None else (logging.DEBUG if default_config.get("debug_logging", False) else logging.INFO)
    logger.setLevel(log_level)
    handler_name = f"{name}_handler"
    if not any(handler.get_name() == handler_name for handler in logger.handlers):
        handler = logging.StreamHandler(); handler.set_name(handler_name)
        formatter = logging.Formatter("%(asctime)s - %(name)s [%(levelname)s] %(message)s")
        handler.setFormatter(formatter); logger.addHandler(handler); logger.propagate = False
    else:
        # Ensure existing handler level is updated if config changes
        for handler in logger.handlers:
            if handler.get_name() == handler_name: handler.setLevel(log_level); break
    return logger
logger = setup_logger()

# Admin User Mock (Unchanged)
class AdminUserMock:
    def __init__(self): self.role = "admin"
admin = AdminUserMock()
# ==============================================================================

# Text processing functions (Unchanged)
def truncate_text(text, max_length=200):
    if not text: return ""
    text = str(text).strip()
    text = re.sub(r"^```(json|markdown)?\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE).strip()
    if len(text) <= max_length: return text
    last_space = text.rfind(' ', 0, max_length)
    return text[:last_space] + "..." if last_space != -1 else text[:max_length] + "..."

def calculate_semantic_distance(text1, text2, llm=None, current_config=None):
    # Use debug_logging for internal debug messages
    debug = current_config.get("debug_logging", False) if current_config else False
    if not text1 or not text2: return 1.0
    text1, text2 = str(text1), str(text2)
    if SKLEARN_AVAILABLE:
        try:
            custom_stop_words = list(ENGLISH_STOP_WORDS) + ["analysis", "however", "therefore", "furthermore", "perspective"]
            vectorizer = TfidfVectorizer(stop_words=custom_stop_words, max_df=0.9, min_df=1);
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            if tfidf_matrix.shape[0] < 2 or tfidf_matrix.shape[1] == 0: raise ValueError("TF-IDF matrix issue.")
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]; similarity = max(0.0, min(1.0, similarity)); return 1.0 - similarity
        except Exception as e: logger.warning(f"TF-IDF semantic distance error: {e}. Falling back to Jaccard.")
    try:
        words1 = set(re.findall(r"\w+", text1.lower())); words2 = set(re.findall(r"\w+", text2.lower()))
        if not words1 or not words2: return 1.0
        intersection = len(words1.intersection(words2)); union = len(words1.union(words2))
        if union == 0: return 0.0
        jaccard_similarity = intersection / union; return 1.0 - jaccard_similarity
    except Exception as fallback_e: logger.error(f"Jaccard similarity fallback failed: {fallback_e}"); return 1.0
# ==============================================================================
# Node class (Unchanged)
class Node(BaseModel):
    id: str = Field(default_factory=lambda: "node_" + "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=4)))
    content: str = ""
    parent: Optional["Node"] = None
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
    value: Optional[float] = None
    descriptive_tags: List[str] = Field(default_factory=list)
    model_config = { "arbitrary_types_allowed": True }
    @field_validator('parent', 'children', mode='before')
    @classmethod
    def _validate_optional_fields(cls, v): return v
    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.use_bayesian_evaluation:
            self.alpha = max(1e-9, float(data.get("alpha", default_config["beta_prior_alpha"])))
            self.beta = max(1e-9, float(data.get("beta", default_config["beta_prior_beta"])))
            self.value = None
        else:
            self.value = float(data.get("value", 0.0)); self.alpha = None; self.beta = None
    def add_child(self, child: "Node") -> "Node":
        child.parent = self; self.children.append(child); return child
    def fully_expanded(self) -> bool: return len(self.children) >= self.max_children
    def get_bayesian_mean(self) -> float:
        if self.use_bayesian_evaluation and self.alpha is not None and self.beta is not None:
            alpha_safe = max(1e-9, self.alpha); beta_safe = max(1e-9, self.beta)
            # Avoid division by zero if both are extremely small
            if alpha_safe + beta_safe < 1e-18: return 0.5
            return alpha_safe / (alpha_safe + beta_safe)
        return 0.5 # Default mean if not Bayesian
    def get_average_score(self) -> float:
        if self.use_bayesian_evaluation: return self.get_bayesian_mean() * 10
        else: return (self.value / max(1, self.visits)) if self.visits > 0 and self.value is not None else 5.0 # Default score if no visits/value
    def thompson_sample(self) -> float:
        if self.use_bayesian_evaluation and self.alpha is not None and self.beta is not None:
            alpha_safe = max(1e-9, self.alpha); beta_safe = max(1e-9, self.beta)
            try: return float(beta_sample(alpha_safe, beta_safe))
            except Exception as e: logger.warning(f"Thompson sample failed for node {self.sequence} (α={alpha_safe}, β={beta_safe}): {e}. Using mean."); return self.get_bayesian_mean()
        return 0.5 # Default sample if not Bayesian
    def best_child(self):
        if not self.children: return None
        valid_children = [c for c in self.children if c is not None]
        if not valid_children: return None

        # Prioritize by visits first to find the most explored branches
        max_visits = -1
        most_visited_children = []
        for child in valid_children:
            if child.visits > max_visits:
                max_visits = child.visits
                most_visited_children = [child]
            elif child.visits == max_visits:
                most_visited_children.append(child)

        if not most_visited_children: return None # Should not happen if valid_children exists
        if len(most_visited_children) == 1: return most_visited_children[0]

        # If multiple children have the same max visits, use score as tie-breaker
        if self.use_bayesian_evaluation:
            return max(most_visited_children, key=lambda c: c.get_bayesian_mean())
        else:
            # Handle potential division by zero for non-Bayesian score
            return max(most_visited_children, key=lambda c: (c.value / max(1, c.visits)) if c.visits > 0 and c.value is not None else 0.0)

    def node_to_json(self) -> Dict:
        score = self.get_average_score()
        valid_children = [child for child in self.children if child is not None]
        base_json = { "id": self.id, "sequence": self.sequence, "content_summary": truncate_text(self.content, 150),
                      "visits": self.visits, "approach_type": self.approach_type, "approach_family": self.approach_family,
                      "is_surprising": self.is_surprising, "thought_summary": truncate_text(self.thought, 100),
                      "descriptive_tags": self.descriptive_tags, "score": round(score, 2),
                      "children": [child.node_to_json() for child in valid_children] }
        if self.use_bayesian_evaluation and self.alpha is not None and self.beta is not None:
            base_json["value_alpha"] = round(self.alpha, 3); base_json["value_beta"] = round(self.beta, 3); base_json["value_mean"] = round(self.get_bayesian_mean(), 3)
        elif not self.use_bayesian_evaluation and self.value is not None: base_json["value_cumulative"] = round(self.value, 2)
        return base_json
# ==============================================================================

# MCTS class (Methods adapted for conditional logging)
class MCTS:
    """Monte Carlo Tree Search for exploring and refining analyses."""
    def __init__(self, **kwargs):
        self.config = kwargs.get("mcts_config", default_config.copy())
        self.llm = kwargs.get("llm"); self.question = kwargs.get("question")
        self.question_summary = self._summarize_question(self.question)
        self.root_node_content = kwargs.get("root").content
        self.node_sequence = 0; self.selected = None; self.current_simulation_in_iteration = 0
        self.thought_history = []; self.debug_history = []; self.surprising_nodes = []
        self.best_solution = str(self.root_node_content); self.best_score = 0.0
        self.iterations_completed = 0; self.simulations_completed = 0; self.high_score_counter = 0
        self.random_state = random.Random(); self.approach_types = ["initial"]; self.explored_approaches = {}
        self.explored_thoughts = set(); self.approach_scores = {}; self.memory = {"depth": 0, "branches": 0, "high_scoring_nodes": []}
        self.iteration_json_snapshots = []
        # Internal debug logging uses 'debug_logging', chat verbosity uses 'show_processing_details'
        self.debug_logging = self.config.get("debug_logging", False)
        self.show_chat_details = self.config.get("show_processing_details", False)

        # Always add MCTS start to internal history, only show in chat if verbose
        mcts_start_log = f"# MCTS Analysis Start\nQ Summary: {self.question_summary}\n"
        self.thought_history.append(mcts_start_log)
        # Don't emit this to chat here, Pipe handles initial message

        cfg = self.config; prior_alpha = max(1e-9, cfg["beta_prior_alpha"]); prior_beta = max(1e-9, cfg["beta_prior_beta"])
        self.root = Node( content=self.root_node_content, sequence=self.get_next_sequence(), parent=None,
                          max_children=cfg["max_children"], use_bayesian_evaluation=cfg["use_bayesian_evaluation"],
                          alpha=prior_alpha, beta=prior_beta, approach_type="initial", approach_family="general" )
        self.selected = self.root
        self.approach_alphas = {approach: prior_alpha for approach in approach_taxonomy.keys()}; self.approach_alphas.update({"initial": prior_alpha, "variant": prior_alpha})
        self.approach_betas = {approach: prior_beta for approach in approach_taxonomy.keys()}; self.approach_betas.update({"initial": prior_beta, "variant": prior_beta})

    def _summarize_question(self, question_text: str, max_words=50) -> str:
        # (implementation unchanged)
        if not question_text: return "N/A"
        words = re.findall(r'\w+', question_text)
        if len(words) <= max_words: return question_text.strip()
        try:
            if not SKLEARN_AVAILABLE: raise ImportError("Scikit-learn not available.")
            sentences = re.split(r'[.!?]+\s*', question_text); sentences = [s for s in sentences if len(s.split()) > 3]
            if not sentences: return " ".join(words[:max_words]) + "..."
            vectorizer = TfidfVectorizer(stop_words='english'); tfidf_matrix = vectorizer.fit_transform(sentences)
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten(); num_summary_sentences = max(1, min(3, len(sentences) // 5))
            top_sentence_indices = sentence_scores.argsort()[-num_summary_sentences:][::-1]; top_sentence_indices.sort()
            summary = " ".join([sentences[i] for i in top_sentence_indices]); summary_words = summary.split()
            if len(summary_words) > max_words * 1.2: return " ".join(summary_words[:max_words]) + "..."
            return summary + "..."
        except Exception as e: logger.warning(f"TF-IDF summary failed ({e}). Truncating."); return " ".join(words[:max_words]) + "..."

    def get_next_sequence(self) -> int: self.node_sequence += 1; return self.node_sequence

    def export_tree_as_json(self) -> Dict:
        try: return self.root.node_to_json()
        except Exception as e: logger.error(f"JSON export error: {e}", exc_info=self.debug_logging); return {"error": f"Export failed: {e}"}

    def _calculate_uct(self, node: Node, parent_visits: int) -> float:
        # (implementation unchanged)
        cfg = self.config;
        if node.visits == 0: return float("inf")
        exploitation = node.get_bayesian_mean() if cfg["use_bayesian_evaluation"] else (node.value / node.visits if node.value is not None else 0.5)
        log_parent_visits = math.log(max(1, parent_visits)); exploration = cfg["exploration_weight"] * math.sqrt(log_parent_visits / node.visits)
        surprise_bonus = 0.3 if node.is_surprising else 0; diversity_bonus = 0.0
        if node.parent and len(node.parent.children) > 1 and cfg["score_diversity_bonus"] > 0:
            my_score = exploitation; sibling_scores = []
            for sibling in node.parent.children:
                if sibling is not None and sibling != node and sibling.visits > 0:
                     sibling_scores.append(sibling.get_average_score() / 10.0)
            if sibling_scores: sibling_avg = sum(sibling_scores) / len(sibling_scores); diversity_bonus = cfg["score_diversity_bonus"] * abs(my_score - sibling_avg)
        uct_value = exploitation + exploration + surprise_bonus + diversity_bonus
        return uct_value if math.isfinite(uct_value) else 0.0 # Return 0 if calculation leads to infinity/NaN

    def get_context_for_node(self, node: Node) -> Dict[str, str]:
        # (implementation unchanged)
        cfg = self.config; debug = self.debug_logging
        best_answer_str = str(self.best_solution) if self.best_solution else "N/A"
        context = { "question_summary": self.question_summary, "best_answer": truncate_text(best_answer_str, 300),
                    "best_score": f"{self.best_score:.1f}", "current_answer": truncate_text(node.content, 300),
                    "current_sequence": str(node.sequence), "current_approach": node.approach_type,
                    "current_tags": ", ".join(node.descriptive_tags) if node.descriptive_tags else "None",
                    "tree_depth": str(self.memory.get("depth", 0)), "branches": str(self.memory.get("branches", 0)),
                    "approach_types": ", ".join(self.approach_types) }
        try: # Explored Thought Types
            if cfg["track_explored_approaches"]:
                exp_app_text = []
                sorted_approach_keys = sorted(self.explored_approaches.keys())
                for app in sorted_approach_keys:
                    thoughts = self.explored_approaches.get(app, []) # Use get for safety
                    if thoughts:
                        count = len(thoughts); score_text = ""
                        if cfg["use_bayesian_evaluation"]: alpha = self.approach_alphas.get(app, 1); beta = self.approach_betas.get(app, 1); score_text = f"(β-Mean: {max(1e-9, alpha) / (max(1e-9, alpha) + max(1e-9, beta)):.2f}, N={count})" if (alpha + beta) > 1e-9 else "(N/A)"
                        else: score = self.approach_scores.get(app, 0); score_text = f"(Avg: {score:.1f}, N={count})"
                        sample = thoughts[-min(2, len(thoughts)) :]; exp_app_text.append(f"- {app} {score_text}: {'; '.join([f'{truncate_text(str(t), 50)}' for t in sample])}")
                context["explored_approaches"] = "\n".join(exp_app_text) if exp_app_text else "None yet."
        except Exception as e: logger.error(f"Ctx err (approaches): {e}"); context["explored_approaches"] = "Error."
        try: # High Scoring Examples
            if self.memory["high_scoring_nodes"]: high_score_text = [f"- Score {score:.1f} ({app}): {truncate_text(content, 70)}" for score, content, app, thought in self.memory["high_scoring_nodes"]]; context["high_scoring_examples"] = "\n".join(["Top Examples:"] + high_score_text)
            else: context["high_scoring_examples"] = "None yet."
        except Exception as e: logger.error(f"Ctx err (high scores): {e}")
        try: # Sibling Context
            if cfg["sibling_awareness"] and node.parent and len(node.parent.children) > 1:
                siblings = [c for c in node.parent.children if c is not None and c != node];
                if siblings:
                    sib_app_text = []; sorted_siblings = sorted(siblings, key=lambda s: s.sequence)
                    for s in sorted_siblings:
                         if s.thought and s.visits > 0: score = s.get_average_score(); tags_str = f"Tags: [{', '.join(s.descriptive_tags)}]" if s.descriptive_tags else ""; sib_app_text.append(f'"{truncate_text(str(s.thought), 50)}" -> (Score: {score:.1f} {tags_str})')
                    if sib_app_text: context["sibling_approaches"] = "\n".join(["Siblings:"] + [f"- {sa}" for sa in sib_app_text])
        except Exception as e: logger.error(f"Ctx err (siblings): {e}")
        safe_context = {k: str(v) if v is not None else "" for k, v in context.items()}; return safe_context

    def _collect_non_leaf_nodes(self, node, non_leaf_nodes, max_depth, current_depth=0):
        # (implementation unchanged)
        if current_depth > max_depth: return
        if node is None: return
        if node.children and not node.fully_expanded(): non_leaf_nodes.append(node)
        for child in node.children:
            if child is not None:
                self._collect_non_leaf_nodes(child, non_leaf_nodes, max_depth, current_depth + 1)

    async def select(self) -> Node:
        # (implementation unchanged - internal logging uses self.debug_logging)
        cfg = self.config; debug = self.debug_logging
        if debug: logger.debug("Selecting node...")
        node = self.root; selection_path = [node]; debug_info = "### Selection Path Decisions:\n"; force_interval = cfg["force_exploration_interval"]
        curr_path_node = None
        # Forced exploration logic (unchanged)
        if force_interval > 0 and self.simulations_completed > 0 and self.simulations_completed % force_interval == 0 and self.memory.get("depth", 0) > 1:
            candidate_nodes = []; self._collect_non_leaf_nodes(self.root, candidate_nodes, max_depth=max(1, self.memory["depth"] // 2)); expandable_candidates = [n for n in candidate_nodes if not n.fully_expanded()]
            if expandable_candidates:
                selected_node = self.random_state.choice(expandable_candidates);
                debug_info += f"BRANCH ENHANCE: Forcing selection {selected_node.sequence}\n";
                if debug: logger.debug(f"BRANCH ENHANCE: Selected {selected_node.sequence}");
                temp_path = [];
                curr_path_node = selected_node
                while curr_path_node:
                    temp_path.append(f"Node {curr_path_node.sequence}");
                    curr_path_node = curr_path_node.parent
                path_str = " → ".join(reversed(temp_path)); self.thought_history.append(f"### Selection Path (Forced)\n{path_str}\n");
                if debug: self.debug_history.append(debug_info);
                return selected_node
        # Regular selection loop (unchanged)
        while node.children:
            valid_children = [child for child in node.children if child is not None]
            if not valid_children:
                if debug: logger.warning(f"Node {node.sequence} has children list but contains only None. Stopping selection.")
                break
            parent_visits = node.visits; unvisited = [child for child in valid_children if child.visits == 0]
            if unvisited: selected_child = self.random_state.choice(unvisited); debug_info += f"Selected unvisited {selected_child.sequence}\n"; node = selected_child; break
            if cfg["use_thompson_sampling"] and cfg["use_bayesian_evaluation"]:
                samples = [];
                for child in valid_children:
                    try:
                        sample_val = child.thompson_sample()
                        if math.isfinite(sample_val):
                            samples.append((child, sample_val))
                        else:
                             if debug: logger.warning(f"Node {child.sequence} TS returned non-finite value ({sample_val}). Skipping.")
                    except Exception as ts_err:
                        logger.error(f"Thompson Sampling error for node {child.sequence}: {ts_err}")
                if not samples:
                    if debug: logger.warning(f"No valid Thompson samples for children of {node.sequence}. Selecting randomly.")
                    selected_child = self.random_state.choice(valid_children)
                else: selected_child, best_sample = max(samples, key=lambda x: x[1]); debug_info += f"TS: Node {selected_child.sequence} ({best_sample:.3f})\n"
                node = selected_child
            else: # UCT
                uct_values = [];
                for child in valid_children:
                    try: uct = self._calculate_uct(child, parent_visits); uct_values.append((child, uct))
                    except Exception as uct_err: logger.error(f"UCT calculation error for node {child.sequence}: {uct_err}")
                if not uct_values:
                    if debug: logger.warning(f"No valid UCT values for children of {node.sequence}. Selecting randomly.")
                    selected_child = self.random_state.choice(valid_children)
                else: uct_values.sort(key=lambda x: x[1], reverse=True); selected_child = uct_values[0][0]; debug_info += f"UCT: Node {selected_child.sequence} ({uct_values[0][1]:.3f})\n"
                node = selected_child
            selection_path.append(node);
            # Break if we reached a node that isn't fully expanded or has no children
            if not node.children or not node.fully_expanded(): break
        # Log selection path internally
        path_str = " → ".join([f"Node {n.sequence} (Tags: {', '.join(n.descriptive_tags) if n.descriptive_tags else '[]'})" for n in selection_path]); self.thought_history.append(f"### Selection Path\n{path_str}\n")
        if debug: self.debug_history.append(debug_info); logger.debug(f"Selection path: {path_str}\n{debug_info}")
        current_depth = len(selection_path) - 1; self.memory["depth"] = max(self.memory.get("depth", 0), current_depth); return node

    def _classify_approach(self, thought: str) -> Tuple[str, str]:
        # (implementation unchanged - internal logging uses self.debug_logging)
        approach_type = "variant"; approach_family = "general";
        if not thought or not isinstance(thought, str): return approach_type, approach_family
        thought_lower = thought.lower(); approach_scores = {app: sum(1 for kw in kws if kw in thought_lower) for app, kws in approach_taxonomy.items()}; positive_scores = {app: score for app, score in approach_scores.items() if score > 0}
        if positive_scores: max_score = max(positive_scores.values()); best_approaches = [app for app, score in positive_scores.items() if score == max_score]; approach_type = self.random_state.choice(best_approaches);
        if approach_type in approach_metadata: approach_family = approach_metadata[approach_type].get("family", "general")
        if self.debug_logging: logger.debug(f"Classified thought '{truncate_text(thought, 50)}' as: {approach_type} ({approach_family})")
        return approach_type, approach_family

    def _check_surprise(self, parent_node, new_content, new_approach_type, new_approach_family) -> Tuple[bool, str]:
        # (implementation unchanged - internal logging uses self.debug_logging)
        cfg = self.config; debug = self.debug_logging
        surprise_factors = []; is_surprising = False; surprise_explanation = ""
        if cfg["use_semantic_distance"]: # Semantic Distance Check
            try:
                parent_content_str = str(parent_node.content) if parent_node.content else ""; new_content_str = str(new_content) if new_content else ""
                if parent_content_str and new_content_str:
                    dist = calculate_semantic_distance(parent_content_str, new_content_str, self.llm, cfg)
                    if dist > cfg["surprise_threshold"]: surprise_factors.append({"type": "semantic", "value": dist, "weight": cfg["surprise_semantic_weight"], "desc": f"Semantic dist ({dist:.2f})"})
            except Exception as e: logger.warning(f"Semantic distance check failed: {e}")
        parent_family = getattr(parent_node, "approach_family", "general"); # Shift in Thought Approach Family
        if parent_family != new_approach_family and new_approach_family != "general": surprise_factors.append({"type": "family_shift", "value": 1.0, "weight": cfg["surprise_philosophical_shift_weight"], "desc": f"Shift '{parent_family}'->'{new_approach_family}'"})
        try: # Novelty of Thought Approach Family (BFS)
            family_counts = Counter(); queue = []; nodes_visited = 0; MAX_NODES = 100; MAX_DEPTH = 5
            if self.root: queue.append((self.root, 0))
            else: logger.error("Novelty check cannot start: Root node is None.")

            while queue and nodes_visited < MAX_NODES:
                curr_node_in_loop = None
                try:
                    if not queue: break
                    curr_node_in_loop, depth = queue.pop(0)
                    if curr_node_in_loop is None: logger.warning("Popped None node during novelty check BFS. Skipping."); continue
                    if depth > MAX_DEPTH: continue
                    nodes_visited += 1
                    fam = getattr(curr_node_in_loop, "approach_family", "general"); family_counts[fam] += 1
                    if depth + 1 <= MAX_DEPTH:
                        nodes_to_add_to_queue = []
                        for child in curr_node_in_loop.children:
                            if child is not None: nodes_to_add_to_queue.append((child, depth + 1))
                            else:
                                parent_id = getattr(curr_node_in_loop, 'id', 'UNK')
                                logger.warning(f"Node {parent_id} contains a None child reference during novelty check BFS.")
                        if nodes_to_add_to_queue: queue.extend(nodes_to_add_to_queue)
                except Exception as node_proc_err:
                    node_id = getattr(curr_node_in_loop, 'id', 'UNK') if curr_node_in_loop else 'UNK_None'
                    logger.error(f"Err processing node {node_id} in novelty check BFS: {node_proc_err}", exc_info=debug);
                    continue
            # Check if the new family is rare (count <= 1 means it's either new or only the parent had it)
            if family_counts.get(new_approach_family, 0) <= 1 and new_approach_family != "general":
                 surprise_factors.append({"type": "novelty", "value": 0.8, "weight": cfg["surprise_novelty_weight"], "desc": f"Novel approach family ('{new_approach_family}')"})
        except Exception as e: logger.warning(f"Novelty check BFS failed overall: {e}", exc_info=debug)

        if surprise_factors:
            total_weighted_score = sum(f["value"] * f["weight"] for f in surprise_factors); total_weight = sum(f["weight"] for f in surprise_factors)
            combined_score = (total_weighted_score / total_weight) if total_weight > 1e-6 else 0.0
            if combined_score >= cfg["surprise_overall_threshold"]:
                is_surprising = True; factor_descs = [f"- {f['desc']} (Val: {f['value']:.2f}, W: {f['weight']:.1f})" for f in surprise_factors]
                surprise_explanation = f"Combined surprise ({combined_score:.2f} >= {cfg['surprise_overall_threshold']}):\n" + "\n".join(factor_descs)
                if debug: logger.debug(f"Surprise DETECTED for node sequence {parent_node.sequence+1}: Score={combined_score:.2f}\n{surprise_explanation}") # Log sequence of the *new* node
        return is_surprising, surprise_explanation

    async def expand(self, node: Node) -> Tuple[Optional[Node], bool]:
        # (implementation unchanged - internal logging uses self.debug_logging)
        cfg = self.config; debug = self.debug_logging
        if debug: logger.debug(f"Expanding node {node.sequence} ('{truncate_text(node.content, 50)}') Tags: {node.descriptive_tags}")
        try:
            # Progress updates always sent regardless of chat verbosity
            await self.llm.progress(f"Expanding Node {node.sequence} (Generating thought)...")
            context = self.get_context_for_node(node)
            # Ensure required keys exist even if empty
            context.setdefault("current_answer", truncate_text(node.content, 300)); context.setdefault("current_sequence", str(node.sequence)); context.setdefault("current_tags", ", ".join(node.descriptive_tags) if node.descriptive_tags else "None")

            thought = await self.llm.generate_thought(node.content, context, self.config)
            if not isinstance(thought, str) or not thought.strip() or "Error:" in thought: logger.error(f"Invalid thought generation result: '{thought}' for node {node.sequence}"); return None, False
            thought = thought.strip();
            if debug: logger.debug(f"Node {node.sequence} Thought: '{thought}'")

            thought_entry = f"### Expanding Node {node.sequence}\n... Thought: {thought}\n"; approach_type, approach_family = self._classify_approach(thought); thought_entry += f"... Approach: {approach_type} (Family: {approach_family})\n"; self.explored_thoughts.add(thought);
            if approach_type not in self.approach_types: self.approach_types.append(approach_type)
            if approach_type not in self.explored_approaches: self.explored_approaches[approach_type] = []
            self.explored_approaches[approach_type].append(thought);

            await self.llm.progress(f"Expanding Node {node.sequence} (Updating analysis)...")
            context_for_update = context.copy(); context_for_update["answer"] = node.content; context_for_update.pop("current_answer", None); context_for_update.pop("current_sequence", None)
            new_content = await self.llm.update_approach(node.content, thought, context_for_update, self.config)
            if not isinstance(new_content, str) or not new_content.strip() or "Error:" in new_content: logger.error(f"Invalid new content generation result: '{new_content}' for node {node.sequence}"); return None, False
            new_content = new_content.strip();

            await self.llm.progress(f"Expanding Node {node.sequence} (Generating tags)...")
            new_tags = await self._generate_tags_for_node(new_content);
            if debug: logger.debug(f"Node {node.sequence+1} Generated Tags: {new_tags}") # Log sequence of the *new* node
            thought_entry += f"... Generated Tags: {new_tags}\n"; is_surprising, surprise_explanation = self._check_surprise(node, new_content, approach_type, approach_family)
            if is_surprising: thought_entry += f"**SURPRISE DETECTED!**\n{surprise_explanation}\n"

            initial_alpha = max(1e-9, cfg["beta_prior_alpha"]); initial_beta = max(1e-9, cfg["beta_prior_beta"]);
            child = Node( content=new_content, parent=node, sequence=self.get_next_sequence(), is_surprising=is_surprising,
                          surprise_explanation=surprise_explanation, approach_type=approach_type, approach_family=approach_family, thought=thought,
                          max_children=cfg["max_children"], use_bayesian_evaluation=cfg["use_bayesian_evaluation"], alpha=initial_alpha, beta=initial_beta,
                          descriptive_tags=new_tags )
            node.add_child(child);
            if is_surprising: self.surprising_nodes.append(child)

            # Log expansion details internally
            thought_entry += f"--> New Analysis {child.sequence} (Tags: {child.descriptive_tags}): {truncate_text(new_content, 100)}\n"; self.thought_history.append(thought_entry);

            # Update branch count
            if len(node.children) > 1: self.memory["branches"] = self.memory.get("branches", 0) + 1

            if debug: logger.debug(f"Successfully expanded Node {node.sequence} into Child {child.sequence}")
            return child, is_surprising
        except Exception as e: logger.error(f"Expand error on Node {node.sequence}: {e}", exc_info=debug); return None, False

    async def _generate_tags_for_node(self, analysis_text: str) -> List[str]:
        # (implementation unchanged - internal logging uses self.debug_logging)
        cfg = self.config; debug = self.debug_logging
        if not analysis_text: return []
        max_tags_to_keep = 3
        try:
            tag_string_raw = await self.llm.get_completion( self.llm.resolve_model(body={"model": self.llm.__model__}), messages=[{"role": "user", "content": tag_generation_prompt.format(analysis_text=analysis_text)}] )
            if not tag_string_raw or "Error:" in tag_string_raw: logger.warning(f"Tag generation failed: {tag_string_raw}"); return []

            cleaned_tags = []; phrases_to_remove = ["instruction>", "tags:", "json", "list:", "`", "output only", "here are", "keywords:", "output:", "text_to_tag>", "response:", "tags"]
            cleaned_text = tag_string_raw
            # Clean common instruction/prefix patterns
            cleaned_text = re.sub(r"^[<\w\s>:]*:", "", cleaned_text, count=1).strip() # Remove initial label like "Tags:"
            for phrase in phrases_to_remove: cleaned_text = re.sub(re.escape(phrase), "", cleaned_text, flags=re.IGNORECASE)

            potential_tags = re.split(r'[,\n;]+', cleaned_text)
            for tag in potential_tags:
                # More aggressive cleaning of unwanted characters
                tag = tag.strip().strip('\'"` M*[]{}:<>/().,-')
                tag = re.sub(r'[*_`]', '', tag) # Remove markdown emphasis
                tag = re.sub(r'\s+', ' ', tag).strip() # Consolidate whitespace
                if tag and len(tag) > 1 and not tag.isdigit() and len(tag) < 50 and tag.lower() != "none":
                    # Avoid adding variations of the same tag (case-insensitive)
                    is_duplicate = False
                    for existing_tag in cleaned_tags:
                        if existing_tag.lower() == tag.lower():
                            is_duplicate = True
                            break
                    if not is_duplicate:
                         cleaned_tags.append(tag)
                if len(cleaned_tags) >= max_tags_to_keep: break

            if debug: logger.debug(f"Raw tags: '{tag_string_raw}'. Cleaned tags: {cleaned_tags}")
            return cleaned_tags[:max_tags_to_keep]
        except Exception as e: logger.error(f"Tag generation/parsing error: {e}", exc_info=debug); return []

    async def simulate(self, node: Node) -> Optional[float]:
        # (implementation unchanged - internal logging uses self.debug_logging)
        cfg = self.config; debug = self.debug_logging
        if debug: logger.debug(f"Simulating node {node.sequence} ('{truncate_text(node.content, 50)}') Tags: {node.descriptive_tags}")
        score = None; raw_score = 0
        try:
            # Progress updates always sent
            await self.llm.progress(f"Evaluating Analysis Node {node.sequence}..."); context = self.get_context_for_node(node); node_content = str(node.content) if node.content else ""
            if not node_content: logger.warning(f"Node {node.sequence} content is empty. Assigning score 1."); return 1.0

            score_result = await self.llm.evaluate_answer(node_content, context, self.config); eval_type = "absolute"
            # evaluate_answer now returns 5 on failure, no need to check for None explicitly
            if not isinstance(score_result, int) or not (1 <= score_result <= 10):
                logger.error(f"Evaluation for Node {node.sequence} returned invalid result: {score_result}. Defaulting to score 5."); score = 5.0; eval_type = "absolute (failed)"; raw_score = 5
            else: score = float(score_result); raw_score = score_result

            # Store raw score and update Bayesian/approach stats
            node.raw_scores.append(raw_score); approach = node.approach_type if node.approach_type else "unknown"
            if cfg["use_bayesian_evaluation"]:
                 pseudo_successes = max(0, score - 1); pseudo_failures = max(0, 10 - score)
                 current_alpha = self.approach_alphas.setdefault(approach, cfg["beta_prior_alpha"]); current_beta = self.approach_betas.setdefault(approach, cfg["beta_prior_beta"])
                 self.approach_alphas[approach] = max(1e-9, current_alpha + pseudo_successes); self.approach_betas[approach] = max(1e-9, current_beta + pseudo_failures)
            # Update simple average score tracking as well
            current_avg = self.approach_scores.get(approach, score); self.approach_scores[approach] = 0.7 * score + 0.3 * current_avg # Use weighted average

            if debug: logger.debug(f"Node {node.sequence} eval: Type={eval_type}, Raw={raw_score}, Score={score:.1f}/10")
            # Log evaluation internally
            self.thought_history.append(f"### Evaluating Node {node.sequence} (Tags: {node.descriptive_tags})\n... Score: {score:.1f}/10 ({eval_type}, raw: {raw_score})\n")

            # Update high score memory
            if score >= 7: # Consider thresholding this?
                 entry = (score, node.content, approach, node.thought); self.memory["high_scoring_nodes"].append(entry); self.memory["high_scoring_nodes"].sort(key=lambda x: x[0], reverse=True); self.memory["high_scoring_nodes"] = self.memory["high_scoring_nodes"][: cfg["memory_cutoff"]]

        except Exception as e: logger.error(f"Simulate error for node {node.sequence}: {e}", exc_info=debug); return None # Return None on exception
        return score

    def backpropagate(self, node: Node, score: float):
        # (implementation unchanged - internal logging uses self.debug_logging)
        cfg = self.config; debug = self.debug_logging;
        if debug: logger.debug(f"Backpropagating score {score:.2f} from {node.sequence}...")
        backprop_path_nodes = []; temp_node = node; pseudo_successes = max(0, score - 1); pseudo_failures = max(0, 10 - score)
        while temp_node:
            backprop_path_nodes.append(f"Node {temp_node.sequence}"); temp_node.visits += 1
            if cfg["use_bayesian_evaluation"]:
                if temp_node.alpha is not None and temp_node.beta is not None: temp_node.alpha = max(1e-9, temp_node.alpha + pseudo_successes); temp_node.beta = max(1e-9, temp_node.beta + pseudo_failures)
                else: logger.warning(f"Node {temp_node.sequence} missing alpha/beta during backprop.")
            else:
                if temp_node.value is not None: temp_node.value += score
                else: logger.warning(f"Node {temp_node.sequence} missing value during backprop.")
            temp_node = temp_node.parent
        path_str = " → ".join(reversed(backprop_path_nodes)); self.thought_history.append(f"### Backpropagating Score {score:.1f}\n... Path: {path_str}\n")
        if debug: logger.debug(f"Backprop complete: {path_str}")

    ### MODIFIED: search - Emit simulation details conditionally ###
    async def search(self, simulations_per_iteration: int):
        """Performs MCTS simulations, conditionally emitting details to chat."""
        cfg = self.config; debug = self.debug_logging;
        # Use the specific config flag for chat verbosity
        show_chat_sim_details = cfg.get("show_processing_details", False)

        if debug: logger.info(f"Starting MCTS Iteration {self.iterations_completed + 1} ({simulations_per_iteration} simulations)...")
        nodes_simulated = 0
        for i in range(simulations_per_iteration):
            self.simulations_completed += 1; self.current_simulation_in_iteration = i + 1;
            # Log simulation start internally
            sim_entry_log = f"### Iter {self.iterations_completed + 1} - Sim {i+1}/{simulations_per_iteration}\n"; self.thought_history.append(sim_entry_log)
            if debug: logger.debug(f"--- Starting Sim {self.current_simulation_in_iteration}/{simulations_per_iteration} ---")

            best_score_before_sim = self.best_score
            leaf = await self.select(); self.selected = leaf; node_to_simulate = leaf
            expanded_in_this_sim = False
            expansion_result_tuple = None
            thought_leading_to_expansion = None

            if leaf and not leaf.fully_expanded() and leaf.content:
                if debug: logger.debug(f"Sim {i+1}: Attempting expansion from Node {leaf.sequence}.")
                expansion_result_tuple = await self.expand(leaf)
                if expansion_result_tuple and expansion_result_tuple[0]:
                    node_to_simulate = expansion_result_tuple[0]; expanded_in_this_sim = True
                    thought_leading_to_expansion = node_to_simulate.thought
                    if debug: logger.debug(f"Sim {i+1}: Expanded {leaf.sequence} -> {node_to_simulate.sequence}.")
                else:
                    if debug: logger.warning(f"Sim {i+1}: Expansion failed for {leaf.sequence}. Simulating original leaf.")
                    node_to_simulate = leaf # Simulate the leaf that failed to expand
                    expanded_in_this_sim = False # Ensure this is false
            elif not leaf:
                 logger.error(f"Sim {i+1}: Selection returned None. Cannot proceed."); continue
            # If leaf is fully expanded or has no content, simulate the leaf itself
            elif leaf:
                 if debug: logger.debug(f"Sim {i+1}: Leaf node {leaf.sequence} is fully expanded or has no content. Simulating it directly.")
                 node_to_simulate = leaf
                 expanded_in_this_sim = False

            score = None
            if node_to_simulate and node_to_simulate.content:
                score = await self.simulate(node_to_simulate); nodes_simulated += 1;
                if debug: logger.debug(f"Sim {i+1}: Node {node_to_simulate.sequence} simulated. Score={score}")
            elif node_to_simulate:
                 if debug: logger.warning(f"Sim {i+1}: Skipping simulation for {node_to_simulate.sequence} (no content).")
            else:
                 # This case should be less likely now with the checks above
                 logger.error(f"Sim {i+1}: Cannot simulate, node_to_simulate is None (Should not happen after selection/expansion logic).")
                 continue # Skip to next simulation if node is somehow None

            if score is not None:
                self.backpropagate(node_to_simulate, score)
                new_best_overall = score > self.best_score
                if new_best_overall:
                     self.best_score = score; self.best_solution = str(node_to_simulate.content); node_info = f"Node {node_to_simulate.sequence} ({node_to_simulate.approach_type}) Tags: {node_to_simulate.descriptive_tags}";
                     # Log new best internally
                     self.thought_history.append(f"### New Best! Score: {score:.1f}/10 ({node_info})\n");
                     if debug: logger.info(f"Sim {i+1}: New best! Score: {score:.1f}, {node_info}")
                     self.high_score_counter = 0 # Reset stability counter on new best
                elif score == self.best_score:
                    # If score matches the best, still update best_solution if content is different?
                    # Or maybe only update if score is strictly greater. Let's stick to strictly greater.
                    pass

                # --- Conditional Simulation Detail Emission to CHAT ---
                # Check the specific flag for chat verbosity
                if show_chat_sim_details and node_to_simulate:
                    sim_detail_msg = f"--- Iter {self.iterations_completed + 1} / Sim {self.current_simulation_in_iteration} ---\n"
                    sim_detail_msg += f"Selected Node: {leaf.sequence} (Visits: {leaf.visits}, Score: {leaf.get_average_score():.1f}, Tags: {leaf.descriptive_tags})\n"

                    if expanded_in_this_sim and thought_leading_to_expansion:
                        sim_detail_msg += f"Based on thought: \"{str(thought_leading_to_expansion).strip()}\"\n"
                        sim_detail_msg += f"--> Expanded to New Node: {node_to_simulate.sequence} ({node_to_simulate.approach_type})\n"
                        sim_detail_msg += f"    Tags: {node_to_simulate.descriptive_tags}\n"
                        # Show analysis only for first expansion for brevity in chat
                        if node_to_simulate.sequence == 2:
                             sim_detail_msg += f"    Initial Expanded Analysis:\n{truncate_text(str(node_to_simulate.content), 300)}\n" # Truncate for chat
                    else: # Node was re-evaluated or expansion failed
                        sim_detail_msg += f"--> Re-evaluating Node: {node_to_simulate.sequence} (Visits: {node_to_simulate.visits})\n"
                        sim_detail_msg += f"    Tags: {node_to_simulate.descriptive_tags}\n"

                    # Always show the score obtained in this simulation step
                    sim_detail_msg += f"Evaluated Score: {score:.1f}/10"
                    if score > best_score_before_sim: sim_detail_msg += " ✨" # Improvement this sim
                    if new_best_overall: sim_detail_msg += " 🏆 (New Overall Best!)"
                    sim_detail_msg += "\n"

                    await self.llm.emit_message(sim_detail_msg) # Append to chat
                    await asyncio.sleep(0.05) # Small delay for readability
                # --- End Conditional Emission ---

                # Early Stopping Check
                if cfg["early_stopping"]:
                    # Check if the *current best score* meets the threshold
                    if self.best_score >= cfg['early_stopping_threshold']:
                        self.high_score_counter += 1 # Increment counter if best score is high
                        if debug: logger.debug(f"Sim {i+1}: Best score ({self.best_score:.1f}) >= threshold. Stability counter: {self.high_score_counter}/{cfg['early_stopping_stability']}")
                        if self.high_score_counter >= cfg['early_stopping_stability']:
                            if debug: logger.info(f"Early stopping criteria met after sim {i+1}, iter {self.iterations_completed + 1}.")
                            self._store_iteration_snapshot("Early Stopping (High Score Stability)")
                            # Need to break out of the inner simulation loop
                            return self.selected # Return selected node to signal loop break
                    else:
                        self.high_score_counter = 0 # Reset counter if best score drops below threshold

            else: # Score is None (simulation failed)
                if node_to_simulate:
                    if debug: logger.warning(f"Sim {i+1}: Simulation failed or skipped for Node {node_to_simulate.sequence}. No score obtained.")
                self.high_score_counter = 0 # Reset stability counter on simulation failure

        self._store_iteration_snapshot("End of Iteration")
        if debug: logger.info(f"Finished MCTS Iteration {self.iterations_completed + 1}.")
        return self.selected # Return last selected node after iteration finishes normally


    def _store_iteration_snapshot(self, reason: str):
        # (implementation unchanged - internal logging uses self.debug_logging)
        cfg = self.config; debug = self.debug_logging; MAX_SNAPSHOTS = 10;
        # Store snapshots only if debug logging is enabled? Or always? Let's keep always for now.
        if len(self.iteration_json_snapshots) >= MAX_SNAPSHOTS:
            if debug: logger.warning(f"Max snapshots ({MAX_SNAPSHOTS}) reached. Not storing for: {reason}")
            return
        try:
            if debug: logger.debug(f"Storing tree snapshot: {reason}")
            snapshot = {"iteration": self.iterations_completed + 1, "simulation": self.current_simulation_in_iteration, "reason": reason, "timestamp": asyncio.get_event_loop().time(), "best_score_so_far": self.best_score, "tree_json": self.export_tree_as_json()}; self.iteration_json_snapshots.append(snapshot)
        except Exception as e: logger.error(f"Snapshot store failed: {e}", exc_info=debug)

    async def _report_tree_stats(self):
        # (implementation unchanged - internal logging uses self.debug_logging)
        cfg = self.config; debug = self.debug_logging
        try:
            total_nodes = self.node_sequence; max_depth = self.memory.get("depth", 0); num_leaves = 0; leaf_nodes = []
            self._collect_leaves(self.root, leaf_nodes); num_leaves = len(leaf_nodes)
            # Avoid division by zero if total_nodes=1 or num_leaves=total_nodes
            avg_branching = ((total_nodes - 1) / max(1, total_nodes - num_leaves)) if total_nodes > 1 and num_leaves < total_nodes else 0
            stats_msg = f"### Tree Stats: Nodes={total_nodes}, Depth={max_depth}, Leaves={num_leaves}, Avg Branching={avg_branching:.2f}\n"
            if debug: self.debug_history.append(stats_msg); logger.debug(stats_msg)
        except Exception as e: logger.error(f"Error reporting tree stats: {e}", exc_info=debug)

    def _collect_leaves(self, node, leaf_nodes):
        # (implementation unchanged)
        if not node: return
        if not node.children:
            leaf_nodes.append(node)
        else:
            for child in node.children:
                if child is not None:
                    self._collect_leaves(child, leaf_nodes)

    async def analyze_iteration(self):
        # (implementation unchanged - kept disabled)
        RUN_ANALYSIS = False;
        if not RUN_ANALYSIS: return None

    def formatted_output(self, highlighted_node=None, final_output=False) -> str:
        # (implementation unchanged - generates final summary regardless of verbosity)
        """Generates summary output, showing full thoughts for top nodes."""
        cfg = self.config; debug = self.debug_logging;
        result = ""
        try:
            # This function is only called for the final output now
            if not final_output: return ""
            result = f"# MCTS Final Analysis Summary\n"
            result += f"The following summarizes the MCTS exploration process, highlighting the best analysis found and the key development steps (thoughts) that led to high-scoring nodes.\n\n"

            # 1. Best Solution (Full Analysis + Tags)
            if self.best_solution:
                best_node = self.find_best_final_node()
                tags_str = f"Tags: {best_node.descriptive_tags}" if best_node and best_node.descriptive_tags else "Tags: []"
                result += f"## Best Analysis Found (Score: {self.best_score:.1f}/10)\n"
                result += f"**{tags_str}**\n\n"
                analysis_text = str(self.best_solution)
                analysis_text = re.sub(r"^```(json|markdown)?\s*", "", analysis_text, flags=re.IGNORECASE | re.MULTILINE)
                analysis_text = re.sub(r"\s*```$", "", analysis_text, flags=re.MULTILINE)
                result += f"{analysis_text.strip()}\n"
            else:
                result += "## Best Analysis\nNo valid solution found.\n"

            # 2. Top Performing Nodes (FULL Thought + NO Analysis Snippet)
            result += "\n## Top Performing Nodes & Driving Thoughts\n"; all_nodes = []; nodes_to_process = []; processed_nodes = set()
            if self.root: nodes_to_process.append(self.root)
            while nodes_to_process:
                current = nodes_to_process.pop(0)
                if current is None: continue
                if current.id not in processed_nodes:
                    processed_nodes.add(current.id)
                    if current.visits > 0: all_nodes.append(current)
                    valid_children = [child for child in current.children if child is not None]
                    nodes_to_process.extend(valid_children)
            sorted_nodes = sorted(all_nodes, key=lambda n: n.get_average_score(), reverse=True)
            top_n = 5
            if sorted_nodes:
                for i, node in enumerate(sorted_nodes[:top_n]):
                    score = node.get_average_score(); score_details = ""
                    if cfg["use_bayesian_evaluation"] and node.alpha is not None and node.beta is not None: score_details = f"(α={node.alpha:.1f}, β={node.beta:.1f})"
                    elif not cfg["use_bayesian_evaluation"] and node.value is not None: score_details = f"(value={node.value:.1f})"
                    tags_str = f"Tags: {node.descriptive_tags}" if node.descriptive_tags else "Tags: []"
                    result += f"### Node {node.sequence}: Score {score:.1f}/10 {score_details}\n"
                    result += f"- **Approach**: {node.approach_type} ({node.approach_family})\n"
                    result += f"- **Visits**: {node.visits}\n"; result += f"- **{tags_str}**\n"
                    if node.thought: result += f"- **Thought**: {str(node.thought).strip()}\n" # FULL THOUGHT
                    else: result += "- **Thought**: (N/A - Initial Node)\n"
                    if node.is_surprising: result += f"- **Surprising**: Yes ({truncate_text(node.surprise_explanation, 100)})\n"
                    # REMOVED Analysis Snippet Line
                    result += "\n" # Add newline for spacing between nodes
            else: result += "No nodes with visits found.\n"

            # 3. Most Explored Path
            result += "\n## Most Explored Path\n"; current = self.root; path = []
            if current: path.append(current)
            while current and current.children:
                best_child_node = current.best_child()
                # Stop if best child has no visits or is None
                if not best_child_node or best_child_node.visits == 0:
                    if debug: logger.warning(f"Path exploration stopped at Node {current.sequence}. Best child '{getattr(best_child_node, 'sequence', 'N/A')}' is None or has 0 visits.")
                    break
                path.append(best_child_node); current = best_child_node
            if len(path) > 1:
                result += "The search explored this primary path (by visits/score):\n\n"
                for i, node in enumerate(path):
                    prefix = "└─ " if i == len(path) - 1 else "├─ "; indent = "   " * i
                    score = node.get_average_score(); tags_str = f"Tags: {node.descriptive_tags}" if node.descriptive_tags else ""
                    result += f"{indent}{prefix}Node {node.sequence} ({node.approach_type}, Score: {score:.1f}, Visits: {node.visits}) {tags_str}\n"
            else: result += "Search did not explore significantly beyond the root node.\n"

            # 4. Surprising Nodes
            if self.surprising_nodes:
                result += "\n## Surprising Nodes\n"; result += "Nodes that triggered surprise detection:\n\n"; max_show = 5; start = max(0, len(self.surprising_nodes) - max_show);
                for node in self.surprising_nodes[start:]:
                    if node is None: continue
                    score = node.get_average_score(); tags_str = f"Tags: {node.descriptive_tags}" if node.descriptive_tags else "Tags: []"
                    result += f"- **Node {node.sequence}** ({node.approach_type}, Score: {score:.1f}, {tags_str}):\n  "; result += f"{truncate_text(node.surprise_explanation.splitlines()[0], 150)}\n"

            # 5. Approach Performance
            if self.approach_scores or self.approach_alphas:
                result += "\n## Thought Approach Performance\n"; approaches_data = []
                all_apps = set(self.approach_alphas.keys()) | set(self.approach_scores.keys())
                for app in all_apps:
                    if app == "unknown": continue # Skip generic unknown type
                    count = len(self.explored_approaches.get(app, []));
                    # Only show approaches that were actually used or the initial one
                    if count == 0 and app != "initial": continue
                    score_str = "N/A"; sort_key = -1.0
                    if cfg["use_bayesian_evaluation"]:
                        alpha = self.approach_alphas.get(app, cfg["beta_prior_alpha"]); beta = self.approach_betas.get(app, cfg["beta_prior_beta"])
                        # Ensure priors weren't reset to 0 somehow
                        if (alpha + beta) > 1e-9:
                            mean_score = alpha / (alpha + beta) * 10; score_str = f"Score: {mean_score:.2f}/10 (α={alpha:.1f}, β={beta:.1f})"; sort_key = mean_score
                        else: score_str = "Score: N/A (Priors Error?)"; sort_key = -1.0
                    else:
                        if app in self.approach_scores: avg_score = self.approach_scores[app]; score_str = f"Score: {avg_score:.2f}/10"; sort_key = avg_score
                        elif count > 0 or app == "initial": score_str = "Score: N/A"; sort_key = -1.0 # Show initial even if no score?
                    approaches_data.append({"name": app, "score_str": score_str, "count": count, "sort_key": sort_key})
                sorted_approaches = sorted(approaches_data, key=lambda x: x["sort_key"], reverse=True); max_show = 7
                for data in sorted_approaches[:max_show]: result += f"- **{data['name']}**: {data['score_str']} ({data.get('count', 0)} thoughts)\n"
                if len(sorted_approaches) > max_show: result += f"- ... ({len(sorted_approaches) - max_show} more)\n"

            # 6. Search Parameters
            result += f"\n## Search Parameters Used\n"; result += f"- **Iterations**: {self.iterations_completed}/{cfg['max_iterations']}\n"; result += f"- **Simulations/Iter**: {cfg['simulations_per_iteration']}\n"; result += f"- **Total Simulations**: {self.simulations_completed}\n"; eval_str = "Bayesian (Beta)" if cfg["use_bayesian_evaluation"] else "Traditional (Avg)"; select_str = "Thompson" if cfg["use_bayesian_evaluation"] and cfg["use_thompson_sampling"] else "UCT"; result += f"- **Evaluation**: {eval_str}\n"; result += f"- **Selection**: {select_str}\n"
            if cfg["use_bayesian_evaluation"]: result += f"- **Beta Priors**: α={cfg['beta_prior_alpha']:.2f}, β={cfg['beta_prior_beta']:.2f}\n"
            result += f"- **Exploration Weight**: {cfg['exploration_weight']:.2f}\n"; result += f"- **Early Stopping**: {'On' if cfg['early_stopping'] else 'Off'}\n"
            if cfg["early_stopping"]: result += f"  - Threshold: {cfg['early_stopping_threshold']:.1f}/10\n"; result += f"  - Stability: {cfg['early_stopping_stability']}\n"
            # Use the correct config key name here
            result += f"- **Show Chat Details**: {'On' if cfg.get('show_processing_details', False) else 'Off'}\n"

            # 7. Conditional Debug (Uses debug_logging, not chat verbosity)
            if debug and self.debug_history:
                result += "\n## Debug Log Snippets (Last 3)\n\n"
                for entry in self.debug_history[-3:]:
                    cleaned_entry = re.sub(r"\n+", "\n", entry).strip()
                    result += truncate_text(cleaned_entry, 200) + "\n---\n"

            return result.strip()
        except Exception as e:
            logger.error(f"Error formatting final output: {e}", exc_info=debug)
            error_msg = f"\n\n# Error generating final summary:\n{type(e).__name__}: {str(e)}\n"
            # Append error to whatever result was generated so far
            result += error_msg
            return result # Return partial result + error

    def find_best_final_node(self) -> Optional[Node]:
        # (implementation unchanged)
        if not self.best_solution: return None
        queue = []; visited = set(); best_match_node = None; min_score_diff = float('inf')
        if self.root:
            queue.append(self.root)
            visited.add(self.root.id)

        # Clean the target solution once upfront
        best_sol_content_cleaned = str(self.best_solution)
        best_sol_content_cleaned = re.sub(r"^```(json|markdown)?\s*", "", best_sol_content_cleaned, flags=re.IGNORECASE | re.MULTILINE)
        best_sol_content_cleaned = re.sub(r"\s*```$", "", best_sol_content_cleaned, flags=re.MULTILINE).strip()

        while queue:
            current = queue.pop(0)
            if current is None: continue

            # Clean current node content for comparison
            node_content_cleaned = str(current.content)
            node_content_cleaned = re.sub(r"^```(json|markdown)?\s*", "", node_content_cleaned, flags=re.IGNORECASE | re.MULTILINE)
            node_content_cleaned = re.sub(r"\s*```$", "", node_content_cleaned, flags=re.MULTILINE).strip()

            if node_content_cleaned == best_sol_content_cleaned:
                 score_diff = abs(current.get_average_score() - self.best_score)
                 # Prioritize exact content match; if multiple, pick closest score
                 if best_match_node is None or score_diff < min_score_diff:
                    best_match_node = current
                    min_score_diff = score_diff

            # Add valid children to queue
            valid_children = [child for child in current.children if child and child.id not in visited]
            for child in valid_children:
                 visited.add(child.id); queue.append(child)

        if not best_match_node: logger.warning("Could not find node object matching best solution content.")
        return best_match_node

# ==============================================================================

# Pipe class
class Pipe:
    """Interface with Open WebUI, running MCTS for analysis exploration."""
    ### MODIFIED: Valves - Added SHOW_PROCESSING_DETAILS ###
    class Valves(BaseModel):
        # MCTS Core Parameters
        MAX_ITERATIONS: int = Field(default=default_config["max_iterations"], title="Max Iterations", ge=1)
        SIMULATIONS_PER_ITERATION: int = Field(default=default_config["simulations_per_iteration"], title="Simulations / Iteration", ge=1)
        MAX_CHILDREN: int = Field(default=default_config["max_children"], title="Max Children / Node", ge=1)

        # Selection Strategy
        EXPLORATION_WEIGHT: float = Field(default=default_config["exploration_weight"], title="Exploration Weight (UCT)", ge=0.0)
        USE_THOMPSON_SAMPLING: bool = Field(default=default_config["use_thompson_sampling"], title="Use Thompson Sampling (if Bayesian)")
        FORCE_EXPLORATION_INTERVAL: int = Field(default=default_config["force_exploration_interval"], title="Force Branch Explore Interval (0=off)", ge=0)
        SCORE_DIVERSITY_BONUS: float = Field(default=default_config["score_diversity_bonus"], title="UCT Score Diversity Bonus", ge=0.0)

        # Evaluation Strategy
        USE_BAYESIAN_EVALUATION: bool = Field(default=default_config["use_bayesian_evaluation"], title="Use Bayesian (Beta) Evaluation")
        BETA_PRIOR_ALPHA: float = Field(default=default_config["beta_prior_alpha"], gt=0, title="Bayesian Prior Alpha (>0)")
        BETA_PRIOR_BETA: float = Field(default=default_config["beta_prior_beta"], gt=0, title="Bayesian Prior Beta (>0)")

        # Surprise Mechanism
        USE_SEMANTIC_DISTANCE: bool = Field(default=default_config["use_semantic_distance"], title="Use Semantic Distance (Surprise)")
        SURPRISE_THRESHOLD: float = Field(default=default_config["surprise_threshold"], ge=0.0, le=1.0, title="Surprise Threshold (Semantic)")
        SURPRISE_SEMANTIC_WEIGHT: float = Field(default=default_config["surprise_semantic_weight"], title="Surprise: Semantic Weight", ge=0.0, le=1.0)
        SURPRISE_PHILOSOPHICAL_SHIFT_WEIGHT: float = Field(default=default_config["surprise_philosophical_shift_weight"], title="Surprise: Shift Weight (Thought)", ge=0.0, le=1.0)
        SURPRISE_NOVELTY_WEIGHT: float = Field(default=default_config["surprise_novelty_weight"], title="Surprise: Novelty Weight (Thought)", ge=0.0, le=1.0)
        SURPRISE_OVERALL_THRESHOLD: float = Field(default=default_config["surprise_overall_threshold"], ge=0.0, le=1.0, title="Surprise: Overall Threshold")

        # Context & Memory
        GLOBAL_CONTEXT_IN_PROMPTS: bool = Field(default=default_config["global_context_in_prompts"], title="Use Global Context in Prompts")
        TRACK_EXPLORED_APPROACHES: bool = Field(default=default_config["track_explored_approaches"], title="Track Explored Thought Approaches")
        SIBLING_AWARENESS: bool = Field(default=default_config["sibling_awareness"], title="Add Sibling Context to Prompts")
        MEMORY_CUTOFF: int = Field(default=default_config["memory_cutoff"], title="Memory Cutoff (Top N High Scores)", ge=0)

        # Termination & Output
        EARLY_STOPPING: bool = Field(default=default_config["early_stopping"], title="Enable Early Stopping")
        EARLY_STOPPING_THRESHOLD: float = Field(default=default_config["early_stopping_threshold"], ge=1.0, le=10.0, title="Early Stopping Score Threshold")
        EARLY_STOPPING_STABILITY: int = Field(default=default_config["early_stopping_stability"], ge=1, title="Early Stopping Stability")
        SHOW_PROCESSING_DETAILS: bool = Field( default=default_config["show_processing_details"], title="Show Detailed MCTS Steps in Chat" ) # Default False
        DEBUG_LOGGING: bool = Field(default=default_config["debug_logging"], title="Enable Detailed Debug Logging (Console/Logs)") # Independent of chat

    def __init__(self):
        self.type = "manifold";
        self.__current_event_emitter__ = None;
        self.__question__ = "";
        self.__model__ = "";
        self.__llm_client__ = None # Placeholder for potential session management

    def pipes(self) -> list[dict[str, str]]:
        # (implementation unchanged)
        try:
            # Force refresh of models if state might be stale
            ollama.get_all_models()
            if hasattr(app.state, "OLLAMA_MODELS") and app.state.OLLAMA_MODELS:
                models = app.state.OLLAMA_MODELS; valid_models = {k: v for k, v in models.items() if isinstance(v, dict) and "name" in v}
                if not valid_models: logger.warning("No valid Ollama models found in app state."); return [{"id": f"{name}-error", "name": f"{name} (No models)"}]
                return [{"id": f"{name}-{k}", "name": f"{name} ({v['name']})"} for k, v in valid_models.items()]
            else: logger.error("OLLAMA_MODELS not found or empty in app state after refresh."); return [{"id": f"{name}-error", "name": f"{name} (Model load error)"}]
        except Exception as e: logger.error(f"Failed to list pipes: {e}", exc_info=default_config.get("debug_logging", False)); return [{"id": f"{name}-error", "name": f"{name} (Error: {e})"}]

    def resolve_model(self, body: dict) -> str:
        # (implementation unchanged)
        model_id = body.get("model", "").strip(); pipe_internal_name = name; prefix_to_find = f"{pipe_internal_name}-";
        separator_index = model_id.rfind(prefix_to_find)
        if separator_index != -1:
            base_model_name = model_id[separator_index + len(prefix_to_find) :]
            if base_model_name:
                if ":" not in base_model_name: logger.warning(f"Resolved model '{base_model_name}' seems to be missing a tag (e.g., ':latest'). Using anyway.")
                else: logger.info(f"Resolved base model '{base_model_name}' from pipe model ID '{model_id}'")
                return base_model_name
            else: logger.error(f"Separator '{prefix_to_find}' found in '{model_id}' but no subsequent model name. Falling back to using full ID."); return model_id
        else: logger.warning(f"Pipe prefix separator '{prefix_to_find}' not found in model ID '{model_id}'. Assuming it's already the base model name."); return model_id

    def resolve_question(self, body: dict) -> str:
        # (implementation unchanged)
        msgs = body.get("messages", []);
        for msg in reversed(msgs):
             if isinstance(msg, dict) and msg.get("role") == "user": content = msg.get("content", ""); return content.strip() if isinstance(content, str) else ""
        return ""

    ### MODIFIED: pipe - Handles quiet/verbose mode ###
    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        __task__=None,
        __model__=None,
    ) -> Union[str, AsyncGenerator[str, None], None]:
        self.__current_event_emitter__ = __event_emitter__; mcts_instance = None; current_config = default_config.copy()
        initial_analysis_text = ""
        debug_this_run = current_config["debug_logging"] # Initial debug state
        show_chat_details = current_config["show_processing_details"] # Initial chat verbosity

        # Placeholder for potential aiohttp session management
        # http_session = None

        try:
            # --- Initial Setup & Validation ---
            model = self.resolve_model(body); input_text = self.resolve_question(body)
            if not input_text: await self.emit_message("**Error:** No input text provided."); await self.done(); return "Error: No input text."
            if not model: await self.emit_message("**Error:** No model identified."); await self.done(); return "Error: Model not identified."
            self.__model__ = model; self.__question__ = input_text

            # Handle Title Generation Task (if applicable)
            if __task__ == TASKS.TITLE_GENERATION:
                logger.info(f"Handling TITLE_GENERATION task.");
                completion = await self.get_completion(model, [{"role": "user", "content": f"Generate title for: {input_text}"}])
                await self.done();
                return f"{name}: {truncate_text(completion, 50)}"

            logger.info(f"Pipe '{name}' v0.7.20 starting. Model: {model}, Input: '{truncate_text(input_text)}'")

            # --- Apply Valve Settings ---
            if hasattr(self, "valves") and isinstance(self.valves, BaseModel):
                logger.info("Applying Valve settings...")
                try:
                    valve_dict = self.valves.model_dump();
                    for key_upper, value in valve_dict.items():
                        key_lower = key_upper.lower()
                        if key_lower in current_config:
                            current_config[key_lower] = value
                            # Update runtime flags if relevant keys change
                            if key_lower == "debug_logging": debug_this_run = value
                            if key_lower == "show_processing_details": show_chat_details = value
                    # Update logger level based on final debug_logging setting
                    setup_logger(logging.DEBUG if debug_this_run else logging.INFO)
                    # Validate/clamp critical values
                    current_config["beta_prior_alpha"] = max(1e-9, current_config["beta_prior_alpha"]); current_config["beta_prior_beta"] = max(1e-9, current_config["beta_prior_beta"])
                    current_config["exploration_weight"] = max(0.0, current_config["exploration_weight"]); current_config["early_stopping_threshold"] = max(1.0, min(10.0, current_config["early_stopping_threshold"]))
                    current_config["early_stopping_stability"] = max(1, current_config["early_stopping_stability"])
                    logger.info("Valve settings applied.")
                except Exception as e:
                     logger.error(f"Error applying Valve settings: {e}. Using default configuration.", exc_info=True);
                     await self.emit_message(f"**Warning:** Error applying settings ({e}). Using defaults.");
                     current_config = default_config.copy(); # Reset to defaults on error
                     debug_this_run = current_config["debug_logging"]
                     show_chat_details = current_config["show_processing_details"]
                     setup_logger(logging.DEBUG if debug_this_run else logging.INFO) # Ensure logger matches default
            else: logger.info("No valves instance found. Using default configuration.")

            # --- Initial User Messages ---
            await self.emit_message(f"# Advanced Bayesian MCTS v0.7.20\n*Analyzing:* \"{truncate_text(input_text, 100)}\" *using model* `{model}`.")
            # Provide context about processing time
            await self.emit_message("🚀 **Starting MCTS Analysis...** This may take some time depending on the complexity and settings.\n")
            if show_chat_details:
                await self.emit_message("*(Verbose mode enabled: Showing detailed MCTS steps in chat)*")
            # Log key parameters used
            params_to_log = {k: v for k, v in current_config.items() if k in ["max_iterations", "simulations_per_iteration", "exploration_weight", "early_stopping", "show_processing_details", "use_bayesian_evaluation", "use_thompson_sampling"]}
            logger.info(f"--- Key Parameters ---\n{json.dumps(params_to_log, indent=2)}")
            if debug_this_run: logger.debug(f"--- Full Configuration ---\n{json.dumps(current_config, indent=2)}")

            # --- Initial Analysis ---
            await self.progress("Generating initial analysis...")
            initial_analysis_text = await self.stream_prompt_completion( initial_prompt, question=input_text )
            if not isinstance(initial_analysis_text, str) or "Error:" in initial_analysis_text:
                 logger.error(f"Initial analysis failed: {initial_analysis_text}"); await self.emit_message(f"**Error:** Initial analysis failed: {initial_analysis_text}"); await self.done(); return f"Error: {initial_analysis_text}"
            initial_analysis_text = initial_analysis_text.strip()
            initial_analysis_text = re.sub(r"^```(json|markdown)?\s*", "", initial_analysis_text, flags=re.IGNORECASE | re.MULTILINE)
            initial_analysis_text = re.sub(r"\s*```$", "", initial_analysis_text, flags=re.MULTILINE).strip()
            if not initial_analysis_text: logger.error("Initial analysis result was empty."); await self.emit_message("**Error:** Initial analysis generated empty content."); await self.done(); return "Error: Empty initial analysis."
            # Emit initial analysis ONLY AFTER it's complete and cleaned
            await self.emit_message("\n## Initial Analysis\n" + initial_analysis_text); await self.emit_message("\n---\n"); await asyncio.sleep(0.1)

            # --- MCTS Initialization ---
            await self.progress("Initializing MCTS...");
            root_stub = Node(content=initial_analysis_text) # Use cleaned initial analysis for root
            mcts_instance = MCTS(root=root_stub, llm=self, question=input_text, mcts_config=current_config)
            logger.info("Starting MCTS iterations...")

            # --- MCTS Iteration Loop ---
            for i in range(current_config["max_iterations"]):
                iteration_num = i + 1
                if debug_this_run: logger.info(f"--- Starting Iteration {iteration_num}/{current_config['max_iterations']} ---")
                await self.progress(f"Running MCTS Iteration {iteration_num}/{current_config['max_iterations']}...")
                best_score_before_iter = mcts_instance.best_score

                # Run the search (which conditionally emits sim details based on config)
                search_result_node = await mcts_instance.search(current_config["simulations_per_iteration"])
                mcts_instance.iterations_completed += 1

                # --- Conditional Iteration Summary Emission ---
                if show_chat_details:
                    iter_best_node = mcts_instance.find_best_final_node()
                    iter_summary_msg = f"\n**--- Iteration {iteration_num} Summary ---**\n"
                    iter_summary_msg += f"- Overall Best Score So Far: {mcts_instance.best_score:.1f}/10"
                    if mcts_instance.best_score > best_score_before_iter:
                        iter_summary_msg += " (✨ New best found this iteration!)"
                    else:
                        iter_summary_msg += " (Best score unchanged this iteration)"
                    if iter_best_node:
                        tags_str = f"Tags: {iter_best_node.descriptive_tags}" if iter_best_node.descriptive_tags else "Tags: []"
                        iter_summary_msg += f"\n- Current Best Node: {iter_best_node.sequence} ({tags_str})"
                    else:
                        iter_summary_msg += "\n- (Could not identify current best node object)"
                    iter_summary_msg += "\n-------------------------------\n"
                    await self.emit_message(iter_summary_msg)
                # --- End Conditional Iteration Summary ---

                # Check for early stopping *after* potentially emitting summary
                if (current_config["early_stopping"] and
                    mcts_instance.best_score >= current_config['early_stopping_threshold'] and
                    mcts_instance.high_score_counter >= current_config["early_stopping_stability"]):
                    logger.info(f"Early stopping criteria met after iteration {iteration_num}.")
                    await self.emit_message(f"**Stopping early:** Analysis score ({mcts_instance.best_score:.1f}/10) reached threshold and stability.")
                    break # Exit the main iteration loop

                # Check if search signaled an early stop within the iteration itself
                # (This might happen if the stability counter hits the limit mid-iteration)
                # We check the condition again here for robustness.
                if search_result_node is None and current_config["early_stopping"] and mcts_instance.high_score_counter >= current_config["early_stopping_stability"]:
                     if debug_this_run: logger.info("Search returned None potentially due to mid-iteration early stop signal.")
                     # Message about stopping was likely already sent by search's check
                     break # Exit the main iteration loop

                await asyncio.sleep(0.1) # Small delay between iterations if not stopping
            # --- End of Iteration Loop ---

            # --- Final Output Generation ---
            logger.info("MCTS iterations finished.")
            await self.emit_message("\n🏁 **MCTS Exploration Finished.** Preparing final analysis summary...")

            # Determine final best analysis content safely
            final_best_analysis = initial_analysis_text # Fallback
            if isinstance(mcts_instance.best_solution, str) and mcts_instance.best_solution.strip():
                final_best_analysis = mcts_instance.best_solution
            else:
                logger.warning("MCTS best_solution was invalid or empty. Using initial analysis as final result.");

            # Emit the main formatted summary (always shown)
            final_summary_output = mcts_instance.formatted_output(final_output=True)
            await self.emit_message(final_summary_output)

            # --- Final Synthesis Step (always shown) ---
            await self.progress("Generating final synthesis...")
            await self.emit_message("\n---\n## Final Synthesis\n")
            try:
                best_node_final = mcts_instance.find_best_final_node()
                path_thoughts_list = []
                path_to_best = []
                temp_node = best_node_final
                while temp_node:
                    path_to_best.append(temp_node)
                    temp_node = temp_node.parent
                path_to_best.reverse() # Root first

                for node in path_to_best:
                    if node.thought and node.parent: # Exclude root's non-existent thought
                        path_thoughts_list.append(f"- (From Node {node.parent.sequence} -> Node {node.sequence}): {node.thought.strip()}")

                if not path_thoughts_list and len(path_to_best) <= 1:
                    logger.warning("No development path thoughts found for synthesis.")
                    await self.emit_message("(Could not retrieve development path for synthesis.)")
                else:
                    path_thoughts_str = "\n".join(path_thoughts_list)
                    synthesis_context = {
                        "question_summary": mcts_instance.question_summary,
                        "initial_analysis_summary": truncate_text(initial_analysis_text, 300),
                        "best_score": f"{mcts_instance.best_score:.1f}",
                        "path_thoughts": path_thoughts_str if path_thoughts_str else "No significant development path found.",
                        "final_best_analysis_summary": truncate_text(final_best_analysis, 400)
                    }
                    # stream_prompt_completion handles emitting the synthesis to chat
                    synthesis_text = await self.stream_prompt_completion(final_synthesis_prompt, **synthesis_context)
                    if "Error:" in synthesis_text:
                        # Error already logged/emitted by stream_prompt_completion
                        pass # No extra message needed unless desired
            except Exception as synth_err:
                logger.error(f"Final synthesis step failed: {synth_err}", exc_info=debug_this_run)
                await self.emit_message("**Error:** Failed to generate final synthesis.")
            # --- End Final Synthesis Step ---

            # Optional Debug: Export Snapshots (uses debug_logging, not chat verbosity)
            EXPORT_SNAPSHOTS = False # Keep disabled unless needed for debugging
            if EXPORT_SNAPSHOTS and debug_this_run:
                logger.debug("--- JSON Tree Snapshots ---")
                if mcts_instance.iteration_json_snapshots:
                     for i, snapshot in enumerate(mcts_instance.iteration_json_snapshots):
                          try: json_str = json.dumps(snapshot["tree_json"], indent=2); logger.debug(f"Snapshot {i+1} (Iter {snapshot['iteration']}, Sim {snapshot['simulation']}, Reason: {snapshot['reason']}):\n{truncate_text(json_str, 5000)}")
                          except Exception as json_err: logger.error(f"Error logging JSON snapshot {i+1}: {json_err}")
                else: logger.debug("No MCTS snapshots were stored.")

            # --- Final Return ---
            # Return None because all desired output was sent via emit_message
            await self.done();
            logger.info(f"Pipe '{name}' finished successfully. Returning None.")
            return None

        except Exception as e:
            # --- Fatal Error Handling ---
            logger.error(f"FATAL Pipe Error in '{name}': {e}", exc_info=True);
            try: await self.emit_message(f"\n\n**FATAL ERROR:**\n```\n{type(e).__name__}: {str(e)}\n```\nProcessing stopped unexpectedly. Please check the application logs.")
            except Exception as emit_err: logger.error(f"Failed to emit fatal error message: {emit_err}")
            # Return an error string in case of fatal exceptions
            return f"Error: Pipe failed unexpectedly. Check logs. ({type(e).__name__})"
        finally:
            # --- Cleanup ---
            # Ensure 'done' is always called and cleanup runs
            await self.done()
            await self.cleanup()


    # --- LLM Interaction & Helper Methods (Unchanged from previous version) ---
    async def progress(self, message: str):
        # Progress updates are generally lightweight and useful, send regardless of chat verbosity
        debug = self.config.get("debug_logging", False) if hasattr(self, 'config') else False # Safely access config
        if self.__current_event_emitter__:
            try:
                if debug: logger.debug(f"Progress Update: {message}")
                await self.__current_event_emitter__({"type": "status", "data": {"level": "info", "description": str(message), "done": False}})
            except Exception as e: logger.error(f"Emit progress error: {e}")

    async def done(self):
        debug = self.config.get("debug_logging", False) if hasattr(self, 'config') else False
        if self.__current_event_emitter__:
            try:
                if debug: logger.debug("Sending 'done' status event.")
                await self.__current_event_emitter__({"type": "status", "data": {"level": "info", "description": "Processing Complete", "done": True}})
            except Exception as e: logger.error(f"Emit done error: {e}")
            # Clear emitter reference *after* sending done
            self.__current_event_emitter__ = None

    async def emit_message(self, message: str):
        # Ensure message is a string before sending
        if self.__current_event_emitter__:
            try: await self.__current_event_emitter__({"type": "message", "data": {"content": str(message)}})
            except Exception as e: logger.error(f"Emit message error: {e} (Msg: {str(message)[:100]}...)")

    async def emit_replace(self, message: str):
         # Function exists but is not used in the current workflow.
        if self.__current_event_emitter__:
            try: await self.__current_event_emitter__({"type": "replace", "data": {"content": str(message)}})
            except Exception as e: logger.error(f"Emit replace error: {e}")

    def get_chunk_content(self, chunk_bytes: bytes) -> Generator[str, None, None]:
        # (implementation unchanged)
        debug = self.config.get("debug_logging", False) if hasattr(self, 'config') else False
        try:
            chunk_str = chunk_bytes.decode("utf-8")
            for line in chunk_str.splitlines():
                line = line.strip()
                if not line or line == "data: [DONE]": continue
                if line.startswith("data: "):
                    json_data_str = line[6:]
                    try:
                        chunk_data = json.loads(json_data_str)
                        # Check structure for OpenAI compatible streaming format
                        if (isinstance(chunk_data, dict) and
                            "choices" in chunk_data and
                            isinstance(chunk_data["choices"], list) and
                            chunk_data["choices"] and
                            isinstance(chunk_data["choices"][0].get("delta"), dict)):
                            content = chunk_data["choices"][0]["delta"].get("content")
                            if isinstance(content, str) and content: # Ensure content is non-empty string
                                yield content
                    except json.JSONDecodeError: logger.warning(f"JSON decode error in stream chunk: {json_data_str}")
                    except Exception as e: logger.error(f"Error processing stream chunk data: {e}")
        except UnicodeDecodeError: logger.error(f"Unicode decode error in stream chunk: {chunk_bytes[:100]}...")
        except Exception as e: logger.error(f"Error decoding/splitting stream chunk: {e}", exc_info=debug)
        return

    async def get_streaming_completion(self, model: str, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        # (implementation unchanged)
        response = None; debug = self.config.get("debug_logging", False) if hasattr(self, 'config') else False
        try:
            response = await self.call_ollama_endpoint_function({"model": model, "messages": messages, "stream": True});

            if (isinstance(response, dict) and response.get("error")):
                 err_msg = self.get_response_content(response); logger.error(f"LLM stream initiation failed: {err_msg}"); yield err_msg; return;

            if hasattr(response, "body_iterator"):
                async for chunk_bytes in response.body_iterator:
                    for part in self.get_chunk_content(chunk_bytes):
                        if part: yield part
            elif isinstance(response, dict):
                content = self.get_response_content(response)
                if content:
                    logger.warning("Expected stream response, but received a single dictionary. Yielding full content.")
                    yield content
                else:
                    logger.error(f"Expected stream, but got invalid dict response: {str(response)[:200]}")
                    yield "Error: Invalid LLM response dictionary."
            else:
                logger.error(f"Expected streaming response or dict, but got type: {type(response)}.")
                yield f"Error: Unexpected LLM response type ({type(response)})."

        except AttributeError as ae:
            logger.error(f"AttributeError during streaming (likely response object issue): {ae}", exc_info=debug)
            yield f"Error during streaming: {str(ae)}"
        except Exception as e:
            logger.error(f"LLM stream processing error: {e}", exc_info=debug);
            yield f"Error during streaming: {str(e)}";
        finally:
            if response is not None and hasattr(response, 'release') and callable(response.release):
                try: await response.release();
                except Exception as release_err: logger.error(f"Error releasing stream response: {release_err}");

    async def get_message_completion(self, model: str, content: str) -> AsyncGenerator[str, None]:
        # (implementation unchanged)
        debug = self.config.get("debug_logging", False) if hasattr(self, 'config') else False
        try:
            async for chunk in self.get_streaming_completion(model, [{"role": "user", "content": str(content)}]): yield chunk
        except Exception as e: logger.error(f"Error in get_message_completion: {e}", exc_info=debug); yield f"Error: {str(e)}"

    async def get_completion(self, model: str, messages: List[Dict[str, str]]) -> str:
        # (implementation unchanged)
        response = None; debug = self.config.get("debug_logging", False) if hasattr(self, 'config') else False
        try:
            response = await self.call_ollama_endpoint_function({"model": model, "messages": messages, "stream": False})
            content = self.get_response_content(response)
            if isinstance(response, dict) and response.get("error"):
                return f"Error: {content}"
            return content
        except Exception as e:
            logger.error(f"Error in get_completion: {e}", exc_info=debug);
            return f"Error: LLM call failed ({str(e)})."
        finally:
             if response is not None and hasattr(response, 'release') and callable(response.release):
                try: await response.release();
                except Exception as release_err: logger.error(f"Error releasing non-stream response: {release_err}");

    async def call_ollama_endpoint_function(self, payload: Dict[str, Any]):
        # (implementation unchanged - still potential source of aiohttp errors)
        debug = self.config.get("debug_logging", False) if hasattr(self, 'config') else False
        try:
            async def receive(): return {"type": "http.request", "body": json.dumps(payload).encode("utf-8")}
            mock_request = Request(scope={"type": "http", "headers": [], "method": "POST", "scheme": "http", "server": ("local", 80), "path": "/api/ollama/generate", "query_string": b"", "client": ("127.0.0.1", 8080), "app": app}, receive=receive)
            if debug: logger.debug(f"Calling internal ollama endpoint: {str(payload)[:200]}...")
            response = await ollama.generate_openai_chat_completion(request=mock_request, form_data=payload, user=admin)
            if debug and not isinstance(response, dict) and not hasattr(response, 'body_iterator'): logger.debug(f"Internal endpoint response type: {type(response)}")
            return response
        except Exception as e:
            logger.error(f"Ollama internal call error: {str(e)}", exc_info=debug);
            return {"error": True, "choices": [{"message": {"role": "assistant", "content": f"Error: LLM call failed ({str(e)[:100]}...). See logs."}}]}

    async def stream_prompt_completion(self, prompt: str, **format_args) -> str:
        # (implementation unchanged - streams to chat, returns final string)
        debug = self.config.get("debug_logging", False) if hasattr(self, 'config') else False
        complete_response = ""; error_occurred = False
        safe_format_args = {k: str(v) if v is not None else "" for k, v in format_args.items()}
        try: formatted_prompt = prompt.format(**safe_format_args)
        except KeyError as e:
            err_msg = f"Error: Prompt formatting key error: '{e}'. Check prompt variables."
            logger.error(f"{err_msg} Available keys: {list(safe_format_args.keys())}")
            await self.emit_message(f"**{err_msg}**")
            return err_msg
        except Exception as e:
            err_msg = f"Error: Prompt formatting failed: {e}."
            logger.error(err_msg, exc_info=debug)
            await self.emit_message(f"**{err_msg}**")
            return err_msg

        try:
            async for chunk in self.get_message_completion(self.__model__, formatted_prompt):
                if chunk is not None:
                    chunk_str = str(chunk)
                    if chunk_str.startswith("Error during streaming:") or chunk_str.startswith("Error:"):
                        logger.error(f"LLM stream error received: {chunk_str}")
                        await self.emit_message(f"**{chunk_str}**")
                        complete_response = chunk_str; error_occurred = True; break

                    # Emit valid chunk to UI and append
                    await self.emit_message(chunk_str)
                    complete_response += chunk_str

            if error_occurred: return complete_response

            clean_response = str(complete_response).strip()
            clean_response = re.sub(r'\n*(?:Would you like me to.*?|Do you want to explore.*?|Is there anything else.*?)\??\s*$', '', clean_response, flags=re.IGNORECASE | re.DOTALL).strip()
            clean_response = re.sub(r"^```(json|markdown)?\s*", "", clean_response, flags=re.IGNORECASE | re.MULTILINE)
            clean_response = re.sub(r"\s*```$", "", clean_response, flags=re.MULTILINE).strip()
            return clean_response

        except Exception as e:
             err_msg = f"Error: LLM stream processing failed: {e}."
             logger.error(err_msg, exc_info=debug);
             await self.emit_message(f"**{err_msg}**")
             return err_msg

    async def generate_thought(self, current_analysis: str, context: Dict, config: Dict) -> str:
        # (implementation unchanged - uses stream_prompt_completion)
        format_args = context.copy(); format_args.setdefault("question_summary", "N/A"); format_args.setdefault("best_answer", "N/A"); format_args.setdefault("best_score", "0.0"); format_args.setdefault("current_answer", truncate_text(current_analysis, 300)); format_args.setdefault("current_sequence", "N/A"); format_args.setdefault("current_tags", "None")
        result = await self.stream_prompt_completion(thoughts_prompt, **format_args);
        return result

    async def update_approach(self, original_analysis: str, critique: str, context: Dict, config: Dict) -> str:
        # (implementation unchanged - uses stream_prompt_completion)
        format_args = context.copy(); format_args["answer"] = original_analysis; format_args["improvements"] = critique.strip(); format_args.setdefault("question_summary", "N/A"); format_args.setdefault("best_answer", "N/A"); format_args.setdefault("best_score", "0.0"); format_args.setdefault("current_tags", "None")
        result = await self.stream_prompt_completion(update_prompt, **format_args);
        if isinstance(result, str) and "Error:" in result:
            logger.error(f"Update approach failed: {result}")
            return str(original_analysis) # Fallback
        if isinstance(result, str) and result.strip():
            clean_result = re.sub(r"^```(json|markdown)?\s*", "", result, flags=re.IGNORECASE | re.MULTILINE)
            clean_result = re.sub(r"\s*```$", "", clean_result, flags=re.MULTILINE).strip()
            return clean_result if clean_result else str(original_analysis)
        logger.warning("Update approach result was empty or invalid. Falling back.")
        return str(original_analysis)

    async def evaluate_answer(self, analysis_to_evaluate: str, context: Dict, config: Dict) -> int:
        # (implementation unchanged - uses stream_prompt_completion)
        format_args = context.copy(); format_args["answer_to_evaluate"] = analysis_to_evaluate; format_args.setdefault("question_summary", "N/A"); format_args.setdefault("best_answer", "N/A"); format_args.setdefault("best_score", "0.0"); format_args.setdefault("current_tags", "None")
        result = await self.stream_prompt_completion(eval_answer_prompt, **format_args);
        if not isinstance(result, str) or "Error:" in result:
             logger.warning(f"Evaluation call failed or returned error: {result}. Defaulting to score 5."); return 5
        score_match = re.search(r"^\s*([1-9]|10)\s*$", result.strip());
        if score_match:
            try: return int(score_match.group(1))
            except ValueError: logger.warning(f"Eval parse error (strict): '{result}'. Defaulting to 5."); return 5
        else:
            logger.warning(f"Eval strict score not found in: '{result}'. Trying relaxed match.");
            relaxed_match = re.search(r"\b([1-9]|10)\b", result.strip());
            if relaxed_match:
                try: return int(relaxed_match.group(1))
                except ValueError: logger.warning(f"Eval parse error (relaxed): '{result}'. Defaulting to 5."); return 5
            else:
                 logger.warning(f"Eval score not found even with relaxed match: '{result}'. Defaulting to 5."); return 5

    async def evaluate_relative(self, parent_answer: str, answer: str, context: Optional[Dict] = None, config: Optional[Dict] = None) -> int:
        # (implementation unchanged - placeholder)
        return 3

    def get_response_content(self, response: Union[Dict, Any]) -> str:
        # (implementation unchanged)
        try:
            if isinstance(response, dict):
                if response.get("error"):
                    if "choices" in response and response["choices"] and isinstance(response["choices"][0].get("message"), dict):
                        return str(response["choices"][0]["message"].get("content", "Unknown LLM Error content"))
                    return f"Error: {response.get('error', 'Unknown LLM Error')}"
                elif ("choices" in response and isinstance(response["choices"], list) and
                      response["choices"] and isinstance(response["choices"][0].get("message"), dict)):
                    return str(response["choices"][0]["message"].get("content", ""))
            logger.warning(f"Unexpected response structure in get_response_content: {type(response)}")
            return ""
        except Exception as e:
            logger.error(f"Response content extraction error: {str(e)}", exc_info=True)
            return ""

    async def cleanup(self):
        # (implementation unchanged - includes placeholder client cleanup)
        debug = self.config.get("debug_logging", False) if hasattr(self, 'config') else False
        if debug: logger.info("Pipe cleanup initiated...")
        self.__current_event_emitter__ = None;
        if self.__llm_client__ and hasattr(self.__llm_client__, "close"):
            try:
                if not self.__llm_client__.closed:
                    await self.__llm_client__.close();
                    logger.info("Closed persistent aiohttp client session.")
                # else: logger.info("Persistent aiohttp client session was already closed.") # Optional log
            except Exception as e: logger.error(f"Error closing client session: {e}")
        self.__llm_client__ = None
        try: gc.collect()
        except Exception as e: logger.error(f"GC error during cleanup: {e}")
        if debug: logger.info("Pipe cleanup complete.")

# ==============================================================================
# FILE END
# ==============================================================================
