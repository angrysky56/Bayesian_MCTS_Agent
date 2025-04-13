# -*- coding: utf-8 -*-
"""
title: advanced_bayesian_mcts
version: 0.7.19

author: angrysky56
author_url: https://github.com/angrysky56
Project Link: https://github.com/angrysky56/Bayesian_MCTS_Agent

Where I found my stored functions, replace ty with your user name:
/home/ty/.open-webui/cache/functions

The way I launch openweb-ui:
DATA_DIR=~/.open-webui uvx --python 3.11 open-webui@latest serve
http://localhost:8080

description: >
  Advanced Bayesian MCTS v0.7.19: Adds Iteration Summaries to chat history. Live view still shows
  thoughts for expansions. Final summary includes full thoughts for top nodes & synthesis.

Key improvements in v0.7.19:
- Iteration Summaries: After each iteration (group of simulations), a summary message is posted
  to the chat, showing progress and the current best score. This persists across iterations.
- Live View Focus: Continues showing thoughts for expansions, hides most analysis text during run.
- Final Summary Clarity: Shows full thoughts for top nodes, no analysis snippets there. Includes synthesis.
- Maintained features: Strict prompts, robustness fixes, full best analysis display, scoring tweak.
# ... previous versions ...

Key improvements in v0.7.18:
- Live View Focus: Shows driving thoughts for expansions, hides redundant analysis.
- Final Summary Clarity: "Top Performing Nodes" shows full 'thought' but no analysis snippet.
- Final Synthesis: Added concluding step synthesizing the best path's thoughts.

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

name = "advanced_mcts"

# --- DEFAULT Global Configuration ---
default_config = {
    "max_children": 10,
    "exploration_weight": 3,
    "max_iterations": 1,
    "simulations_per_iteration": 10,
    "surprise_threshold": 0.62,
    "use_semantic_distance": True,
    "relative_evaluation": False,
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
    "surprise_semantic_weight": 0.5,
    "surprise_philosophical_shift_weight": 0.4,
    "surprise_novelty_weight": 0.4,
    "surprise_overall_threshold": 0.9,
    "use_bayesian_evaluation": True,
    "use_thompson_sampling": True,
    "beta_prior_alpha": 1.0,
    "beta_prior_beta": 1.0,
    "show_simulation_details": True,  # Default ON
}
# ==============================================================================
# Approach Taxonomy & Metadata (Unchanged)
# ... (taxonomy and metadata dictionaries remain the same) ...
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

# --- Prompts (Unchanged from 0.7.18) ---
# ... (initial_prompt, thoughts_prompt, update_prompt, eval_answer_prompt, tag_generation_prompt, final_synthesis_prompt remain the same) ...
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
# ... (setup_logger remains the same) ...
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
    if not any(handler.get_name() == handler_name for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.set_name(handler_name)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s [%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    else:
        for handler in logger.handlers:
            if handler.get_name() == handler_name:
                handler.setLevel(log_level)
                break
    return logger


logger = setup_logger()


# Admin User Mock (Unchanged)
# ... (AdminUserMock remains the same) ...
class AdminUserMock:
    def __init__(self):
        self.role = "admin"


admin = AdminUserMock()
# ==============================================================================


# Text processing functions (Unchanged)
# ... (truncate_text and calculate_semantic_distance remain the same) ...
def truncate_text(text, max_length=200):
    if not text:
        return ""
    text = str(text).strip()
    text = re.sub(
        r"^```(json|markdown)?\s*", "", text, flags=re.IGNORECASE | re.MULTILINE
    )
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE).strip()
    if len(text) <= max_length:
        return text
    last_space = text.rfind(" ", 0, max_length)
    return text[:last_space] + "..." if last_space != -1 else text[:max_length] + "..."


def calculate_semantic_distance(text1, text2, llm=None, current_config=None):
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
            if tfidf_matrix.shape[0] < 2 or tfidf_matrix.shape[1] == 0:
                raise ValueError("TF-IDF matrix issue.")
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            similarity = max(0.0, min(1.0, similarity))
            return 1.0 - similarity
        except Exception as e:
            logger.warning(
                f"TF-IDF semantic distance error: {e}. Falling back to Jaccard."
            )
    try:
        words1 = set(re.findall(r"\w+", text1.lower()))
        words2 = set(re.findall(r"\w+", text2.lower()))
        if not words1 or not words2:
            return 1.0
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        if union == 0:
            return 0.0
        jaccard_similarity = intersection / union
        return 1.0 - jaccard_similarity
    except Exception as fallback_e:
        logger.error(f"Jaccard similarity fallback failed: {fallback_e}")
        return 1.0


# ==============================================================================
# (Node class - unchanged from 0.7.12)
# ... (Node class remains the same) ...
class Node(BaseModel):
    id: str = Field(
        default_factory=lambda: "node_"
        + "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=4))
    )
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
            self.value = None
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
            return alpha_safe / (alpha_safe + beta_safe)
        return 0.5

    def get_average_score(self) -> float:
        if self.use_bayesian_evaluation:
            return self.get_bayesian_mean() * 10
        else:
            return (
                (self.value / max(1, self.visits))
                if self.visits > 0 and self.value is not None
                else 5.0
            )

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
                logger.warning(f"Thompson sample failed: {e}. Mean.")
                return self.get_bayesian_mean()
        return 0.5

    def best_child(self):
        if not self.children:
            return None
        max_visits = -1
        most_visited_children = []
        for child in self.children:
            if child is not None and child.visits > max_visits:
                max_visits = child.visits
                most_visited_children = [child]
            elif child is not None and child.visits == max_visits:
                most_visited_children.append(child)
        if not most_visited_children:
            return None
        if len(most_visited_children) == 1:
            return most_visited_children[0]
        valid_children_for_max = [c for c in most_visited_children if c is not None]
        if not valid_children_for_max:
            return None
        if self.use_bayesian_evaluation:
            return max(valid_children_for_max, key=lambda c: c.get_bayesian_mean())
        else:
            return max(
                valid_children_for_max,
                key=lambda c: (
                    (c.value / max(1, c.visits))
                    if c.visits > 0 and c.value is not None
                    else 0.0
                ),
            )

    def node_to_json(self) -> Dict:
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
            "children": [child.node_to_json() for child in valid_children],
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


# ==============================================================================


# (MCTS class - init, context etc unchanged from 0.7.12)
class MCTS:
    """Monte Carlo Tree Search for exploring and refining analyses."""

    def __init__(self, **kwargs):
        self.config = kwargs.get("mcts_config", default_config.copy())
        self.llm = kwargs.get("llm")
        self.question = kwargs.get("question")
        self.question_summary = self._summarize_question(self.question)
        self.root_node_content = kwargs.get("root").content
        self.node_sequence = 0
        self.selected = None
        self.current_simulation_in_iteration = 0
        self.thought_history = []
        self.debug_history = []
        self.surprising_nodes = []
        self.best_solution = str(self.root_node_content)
        self.best_score = 0.0
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
        self.thought_history.append(
            f"# MCTS Analysis Start\nQ Summary: {self.question_summary}\n"
        )
        cfg = self.config
        prior_alpha = max(1e-9, cfg["beta_prior_alpha"])
        prior_beta = max(1e-9, cfg["beta_prior_beta"])
        self.root = Node(
            content=self.root_node_content,
            sequence=self.get_next_sequence(),
            parent=None,
            max_children=cfg["max_children"],
            use_bayesian_evaluation=cfg["use_bayesian_evaluation"],
            alpha=prior_alpha,
            beta=prior_beta,
            approach_type="initial",
            approach_family="general",
        )
        self.selected = self.root
        self.approach_alphas = {
            approach: prior_alpha for approach in approach_taxonomy.keys()
        }
        self.approach_alphas.update({"initial": prior_alpha, "variant": prior_alpha})
        self.approach_betas = {
            approach: prior_beta for approach in approach_taxonomy.keys()
        }
        self.approach_betas.update({"initial": prior_beta, "variant": prior_beta})

    def _summarize_question(self, question_text: str, max_words=50) -> str:
        # ... (implementation unchanged) ...
        if not question_text:
            return "N/A"
        words = re.findall(r"\w+", question_text)
        if len(words) <= max_words:
            return question_text.strip()
        try:
            if not SKLEARN_AVAILABLE:
                raise ImportError("Scikit-learn not available.")
            sentences = re.split(r"[.!?]+\s*", question_text)
            sentences = [s for s in sentences if len(s.split()) > 3]
            if not sentences:
                return " ".join(words[:max_words]) + "..."
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(sentences)
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            num_summary_sentences = max(1, min(3, len(sentences) // 5))
            top_sentence_indices = sentence_scores.argsort()[-num_summary_sentences:][
                ::-1
            ]
            top_sentence_indices.sort()
            summary = " ".join([sentences[i] for i in top_sentence_indices])
            summary_words = summary.split()
            if len(summary_words) > max_words * 1.2:
                return " ".join(summary_words[:max_words]) + "..."
            return summary + "..."
        except Exception as e:
            logger.warning(f"TF-IDF summary failed ({e}). Truncating.")
            return " ".join(words[:max_words]) + "..."

    def get_next_sequence(self) -> int:
        self.node_sequence += 1
        return self.node_sequence

    def export_tree_as_json(self) -> Dict:
        try:
            return self.root.node_to_json()
        except Exception as e:
            logger.error(
                f"JSON export error: {e}",
                exc_info=self.config.get("debug_logging", False),
            )
            return {"error": f"Export failed: {e}"}

    def _calculate_uct(self, node: Node, parent_visits: int) -> float:
        # ... (implementation unchanged) ...
        cfg = self.config
        if node.visits == 0:
            return float("inf")
        exploitation = (
            node.get_bayesian_mean()
            if cfg["use_bayesian_evaluation"]
            else (node.value / node.visits if node.value is not None else 0.5)
        )
        log_parent_visits = math.log(max(1, parent_visits))
        exploration = cfg["exploration_weight"] * math.sqrt(
            log_parent_visits / node.visits
        )
        surprise_bonus = 0.3 if node.is_surprising else 0
        diversity_bonus = 0.0
        if (
            node.parent
            and len(node.parent.children) > 1
            and cfg["score_diversity_bonus"] > 0
        ):
            my_score = exploitation
            sibling_scores = []
            for sibling in node.parent.children:
                if sibling is not None and sibling != node and sibling.visits > 0:
                    sibling_scores.append(sibling.get_average_score() / 10.0)
            if sibling_scores:
                sibling_avg = sum(sibling_scores) / len(sibling_scores)
                diversity_bonus = cfg["score_diversity_bonus"] * abs(
                    my_score - sibling_avg
                )
        uct_value = exploitation + exploration + surprise_bonus + diversity_bonus
        return uct_value if math.isfinite(uct_value) else 0.0

    def get_context_for_node(self, node: Node) -> Dict[str, str]:
        # ... (implementation unchanged) ...
        cfg = self.config
        debug = cfg.get("debug_logging", False)
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
        }
        try:  # Explored Thought Types
            if cfg["track_explored_approaches"]:
                exp_app_text = []
                sorted_approach_keys = sorted(self.explored_approaches.keys())
                for app in sorted_approach_keys:
                    thoughts = self.explored_approaches.get(
                        app, []
                    )  # Use get for safety
                    if thoughts:
                        count = len(thoughts)
                        score_text = ""
                        if cfg["use_bayesian_evaluation"]:
                            alpha = self.approach_alphas.get(app, 1)
                            beta = self.approach_betas.get(app, 1)
                            score_text = (
                                f"(β-Mean: {max(1e-9, alpha) / (max(1e-9, alpha) + max(1e-9, beta)):.2f}, N={count})"
                                if (alpha + beta) > 1e-9
                                else "(N/A)"
                            )
                        else:
                            score = self.approach_scores.get(app, 0)
                            score_text = f"(Avg: {score:.1f}, N={count})"
                        sample = thoughts[-min(2, len(thoughts)) :]
                        exp_app_text.append(
                            f"- {app} {score_text}: {'; '.join([f'{truncate_text(str(t), 50)}' for t in sample])}"
                        )
                context["explored_approaches"] = (
                    "\n".join(exp_app_text) if exp_app_text else "None yet."
                )
        except Exception as e:
            logger.error(f"Ctx err (approaches): {e}")
            context["explored_approaches"] = "Error."
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
                        if s.thought and s.visits > 0:
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
        except Exception as e:
            logger.error(f"Ctx err (siblings): {e}")
        safe_context = {k: str(v) if v is not None else "" for k, v in context.items()}
        return safe_context

    # (_collect_non_leaf_nodes - unchanged from 0.7.12)
    def _collect_non_leaf_nodes(self, node, non_leaf_nodes, max_depth, current_depth=0):
        # ... (implementation unchanged) ...
        if current_depth > max_depth:
            return
        if node is None:
            return
        if node.children and not node.fully_expanded():
            non_leaf_nodes.append(node)
        for child in node.children:
            if child is not None:
                self._collect_non_leaf_nodes(
                    child, non_leaf_nodes, max_depth, current_depth + 1
                )

    # (select - unchanged from 0.7.15)
    async def select(self) -> Node:
        # ... (implementation unchanged) ...
        cfg = self.config
        debug = cfg.get("debug_logging", False)
        if debug:
            logger.debug("Selecting node...")
        node = self.root
        selection_path = [node]
        debug_info = "### Selection Path Decisions:\n"
        force_interval = cfg["force_exploration_interval"]
        curr_path_node = None
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
                return selected_node
        while node.children:
            valid_children = [child for child in node.children if child is not None]
            if not valid_children:
                (
                    logger.warning(
                        f"Node {node.sequence} has children list but contains only None. Stopping selection."
                    )
                    if debug
                    else None
                )
                break
            parent_visits = node.visits
            unvisited = [child for child in valid_children if child.visits == 0]
            if unvisited:
                selected_child = self.random_state.choice(unvisited)
                debug_info += f"Selected unvisited {selected_child.sequence}\n"
                node = selected_child
                break
            if cfg["use_thompson_sampling"] and cfg["use_bayesian_evaluation"]:
                samples = []
                for child in valid_children:
                    try:
                        sample_val = child.thompson_sample()
                        if math.isfinite(sample_val):
                            samples.append((child, sample_val))
                        else:
                            (
                                logger.warning(f"Node {child.sequence} TS not finite.")
                                if debug
                                else None
                            )
                    except Exception as ts_err:
                        logger.error(f"TS error: {ts_err}")
                if not samples:
                    selected_child = self.random_state.choice(valid_children)
                    logger.warning("No valid TS samples. Random.") if debug else None
                else:
                    selected_child, best_sample = max(samples, key=lambda x: x[1])
                    debug_info += (
                        f"TS: Node {selected_child.sequence} ({best_sample:.3f})\n"
                    )
                node = selected_child
            else:  # UCT
                uct_values = []
                for child in valid_children:
                    try:
                        uct = self._calculate_uct(child, parent_visits)
                        uct_values.append((child, uct))
                    except Exception as uct_err:
                        logger.error(f"UCT error: {uct_err}")
                if not uct_values:
                    selected_child = self.random_state.choice(valid_children)
                    logger.warning("No valid UCT. Random.") if debug else None
                else:
                    uct_values.sort(key=lambda x: x[1], reverse=True)
                    selected_child = uct_values[0][0]
                    debug_info += f"UCT: Node {selected_child.sequence} ({uct_values[0][1]:.3f})\n"
                node = selected_child
            selection_path.append(node)
            if not node.children or not node.fully_expanded():
                break
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
        return node

    # (_classify_approach - unchanged from 0.7.12)
    def _classify_approach(self, thought: str) -> Tuple[str, str]:
        # ... (implementation unchanged) ...
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
        if self.config.get("debug_logging", False):
            logger.debug(
                f"Classified thought '{truncate_text(thought, 50)}' as: {approach_type} ({approach_family})"
            )
        return approach_type, approach_family

    # (_check_surprise - unchanged from 0.7.12)
    def _check_surprise(
        self, parent_node, new_content, new_approach_type, new_approach_family
    ) -> Tuple[bool, str]:
        # ... (implementation unchanged) ...
        cfg = self.config
        debug = cfg.get("debug_logging", False)
        surprise_factors = []
        is_surprising = False
        surprise_explanation = ""
        if cfg["use_semantic_distance"]:  # Semantic Distance Check
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
        parent_family = getattr(parent_node, "approach_family", "general")
        # Shift in Thought Approach Family
        if parent_family != new_approach_family and new_approach_family != "general":
            surprise_factors.append(
                {
                    "type": "family_shift",
                    "value": 1.0,
                    "weight": cfg["surprise_philosophical_shift_weight"],
                    "desc": f"Shift '{parent_family}'->'{new_approach_family}'",
                }
            )
        try:  # Novelty of Thought Approach Family (BFS)
            family_counts = Counter()
            queue = []
            nodes_visited = 0
            MAX_NODES = 100
            MAX_DEPTH = 5
            if self.root:
                queue.append((self.root, 0))
            else:
                logger.error("Novelty check cannot start: Root node is None.")

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
                    if depth > MAX_DEPTH:
                        continue
                    nodes_visited += 1
                    fam = getattr(curr_node_in_loop, "approach_family", "general")
                    family_counts[fam] += 1
                    if depth + 1 <= MAX_DEPTH:
                        nodes_to_add_to_queue = []
                        for child in curr_node_in_loop.children:
                            if child is not None:
                                nodes_to_add_to_queue.append((child, depth + 1))
                            else:
                                parent_id = getattr(curr_node_in_loop, "id", "UNK")
                                logger.warning(
                                    f"Node {parent_id} contains a None child reference during novelty check BFS."
                                )
                        if nodes_to_add_to_queue:
                            queue.extend(nodes_to_add_to_queue)
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
                    continue
            if (
                family_counts.get(new_approach_family, 0) <= 1
                and new_approach_family != "general"
            ):
                surprise_factors.append(
                    {
                        "type": "novelty",
                        "value": 0.8,
                        "weight": cfg["surprise_novelty_weight"],
                        "desc": f"Novel approach ('{new_approach_family}')",
                    }
                )
        except Exception as e:
            logger.warning(f"Novelty check BFS failed overall: {e}", exc_info=debug)
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
                        f"Surprise DETECTED: Score={combined_score:.2f}\n{surprise_explanation}"
                    )
        return is_surprising, surprise_explanation

    # (expand - unchanged from 0.7.12)
    async def expand(self, node: Node) -> Tuple[Optional[Node], bool]:
        # ... (implementation unchanged) ...
        cfg = self.config
        debug = cfg.get("debug_logging", False)
        if debug:
            logger.debug(
                f"Expanding node {node.sequence} ('{truncate_text(node.content, 50)}') Tags: {node.descriptive_tags}"
            )
        try:
            await self.llm.progress(
                f"Expanding Node {node.sequence} (Generating thought)..."
            )
            context = self.get_context_for_node(node)
            context.setdefault("current_answer", truncate_text(node.content, 300))
            context.setdefault("current_sequence", str(node.sequence))
            context.setdefault(
                "current_tags",
                ", ".join(node.descriptive_tags) if node.descriptive_tags else "None",
            )
            thought = await self.llm.generate_thought(
                node.content, context, self.config
            )
            if (
                not isinstance(thought, str)
                or not thought.strip()
                or "Error:" in thought
            ):
                logger.error(f"Invalid thought: '{thought}'")
                return None, False
            thought = thought.strip()
            if debug:
                logger.debug(f"Node {node.sequence} Thought: '{thought}'")
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
            new_content = await self.llm.update_approach(
                node.content, thought, context_for_update, self.config
            )
            if (
                not isinstance(new_content, str)
                or not new_content.strip()
                or "Error:" in new_content
            ):
                logger.error(f"Invalid new content: '{new_content}'")
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

    # (_generate_tags_for_node - unchanged from 0.7.12)
    async def _generate_tags_for_node(self, analysis_text: str) -> List[str]:
        # ... (implementation unchanged) ...
        cfg = self.config
        debug = cfg.get("debug_logging", False)
        if not analysis_text:
            return []
        max_tags_to_keep = 3
        try:
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
            ]
            cleaned_text = tag_string_raw
            for phrase in phrases_to_remove:
                cleaned_text = re.sub(
                    re.escape(phrase), "", cleaned_text, flags=re.IGNORECASE
                )
            potential_tags = re.split(r"[,\n;]+", cleaned_text)
            for tag in potential_tags:
                tag = tag.strip().strip("'\"` M*[]{}:<>/")
                tag = re.sub(r"[*_`]", "", tag)
                if tag and len(tag) > 1 and not tag.isdigit() and len(tag) < 50:
                    cleaned_tags.append(tag)
                if len(cleaned_tags) >= max_tags_to_keep:
                    break
            if debug:
                logger.debug(
                    f"Raw tags: '{tag_string_raw}'. Cleaned tags: {cleaned_tags}"
                )
            return cleaned_tags[:max_tags_to_keep]
        except Exception as e:
            logger.error(f"Tag generation/parsing error: {e}", exc_info=debug)
            return []

    # (simulate, backpropagate unchanged from 0.7.12)
    async def simulate(self, node: Node) -> Optional[float]:
        # ... (implementation unchanged) ...
        cfg = self.config
        debug = cfg.get("debug_logging", False)
        if debug:
            logger.debug(
                f"Simulating node {node.sequence} ('{truncate_text(node.content, 50)}') Tags: {node.descriptive_tags}"
            )
        score = None
        raw_score = 0
        try:
            await self.llm.progress(f"Evaluating Analysis Node {node.sequence}...")
            context = self.get_context_for_node(node)
            node_content = str(node.content) if node.content else ""
            if not node_content:
                logger.warning(f"Node {node.sequence} empty. Score 1.")
                return 1.0
            score_result = await self.llm.evaluate_answer(
                node_content, context, self.config
            )
            eval_type = "absolute"
            if not isinstance(score_result, int) or not (1 <= score_result <= 10):
                logger.error(f"Eval failed ({score_result}). Score 5.")
                score = 5.0
                eval_type = "absolute (failed)"
                raw_score = 5
            else:
                score = float(score_result)
                raw_score = score_result
            if score is not None:
                node.raw_scores.append(raw_score)
                approach = node.approach_type if node.approach_type else "unknown"
                if cfg["use_bayesian_evaluation"]:
                    pseudo_successes = max(0, score - 1)
                    pseudo_failures = max(0, 10 - score)
                    current_alpha = self.approach_alphas.setdefault(
                        approach, cfg["beta_prior_alpha"]
                    )
                    current_beta = self.approach_betas.setdefault(
                        approach, cfg["beta_prior_beta"]
                    )
                    self.approach_alphas[approach] = max(
                        1e-9, current_alpha + pseudo_successes
                    )
                    self.approach_betas[approach] = max(
                        1e-9, current_beta + pseudo_failures
                    )
                current_avg = self.approach_scores.get(approach, score)
                self.approach_scores[approach] = 0.7 * score + 0.3 * current_avg
                if debug:
                    logger.debug(
                        f"Node {node.sequence} eval: Type={eval_type}, Raw={raw_score}, Score={score:.1f}/10"
                    )
                self.thought_history.append(
                    f"### Evaluating Node {node.sequence} (Tags: {node.descriptive_tags})\n... Score: {score:.1f}/10 ({eval_type}, raw: {raw_score})\n"
                )
                if score >= 7:
                    entry = (score, node.content, approach, node.thought)
                    self.memory["high_scoring_nodes"].append(entry)
                    self.memory["high_scoring_nodes"].sort(
                        key=lambda x: x[0], reverse=True
                    )
                    self.memory["high_scoring_nodes"] = self.memory[
                        "high_scoring_nodes"
                    ][: cfg["memory_cutoff"]]
            else:
                logger.error(f"Simulate None score.")
                return None
        except Exception as e:
            logger.error(f"Simulate error: {e}", exc_info=debug)
            return None
        return score

    def backpropagate(self, node: Node, score: float):
        # ... (implementation unchanged) ...
        cfg = self.config
        debug = cfg.get("debug_logging", False)
        if debug:
            logger.debug(f"Backpropagating score {score:.2f} from {node.sequence}...")
        backprop_path_nodes = []
        temp_node = node
        pseudo_successes = max(0, score - 1)
        pseudo_failures = max(0, 10 - score)
        while temp_node:
            backprop_path_nodes.append(f"Node {temp_node.sequence}")
            temp_node.visits += 1
            if cfg["use_bayesian_evaluation"]:
                if temp_node.alpha is not None and temp_node.beta is not None:
                    temp_node.alpha = max(1e-9, temp_node.alpha + pseudo_successes)
                    temp_node.beta = max(1e-9, temp_node.beta + pseudo_failures)
                else:
                    logger.warning(f"Node {temp_node.sequence} missing alpha/beta.")
            else:
                if temp_node.value is not None:
                    temp_node.value += score
                else:
                    logger.warning(f"Node {temp_node.sequence} missing value.")
            temp_node = temp_node.parent
        path_str = " → ".join(reversed(backprop_path_nodes))
        self.thought_history.append(
            f"### Backpropagating Score {score:.1f}\n... Path: {path_str}\n"
        )
        if debug:
            logger.debug(f"Backprop complete: {path_str}")

    ### MODIFIED: search - Refined live view content again ###
    async def search(self, simulations_per_iteration: int):
        """Performs MCTS simulations, emitting MCTS process steps if details enabled."""
        cfg = self.config
        debug = cfg.get("debug_logging", False)
        show_sim_details = cfg.get("show_simulation_details", False)

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
                    (
                        logger.debug(
                            f"Sim {i+1}: Expanded {leaf.sequence} -> {node_to_simulate.sequence}."
                        )
                        if debug
                        else None
                    )
                else:
                    (
                        logger.warning(
                            f"Sim {i+1}: Expansion failed for {leaf.sequence}. Simulating original leaf."
                        )
                        if debug
                        else None
                    )
                    node_to_simulate = leaf
                    expanded_in_this_sim = False
            elif not leaf:
                logger.error(f"Sim {i+1}: Selection returned None. Cannot proceed.")
                continue

            score = None
            if node_to_simulate and node_to_simulate.content:
                score = await self.simulate(node_to_simulate)
                nodes_simulated += 1
                (
                    logger.debug(
                        f"Sim {i+1}: Node {node_to_simulate.sequence} simulated. Score={score}"
                    )
                    if debug
                    else None
                )
            elif node_to_simulate:
                (
                    logger.warning(
                        f"Sim {i+1}: Skipping simulation for {node_to_simulate.sequence} (no content)."
                    )
                    if debug
                    else None
                )
            else:
                (
                    logger.error(
                        f"Sim {i+1}: Cannot simulate, node_to_simulate is None."
                    )
                    if debug
                    else None
                )

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
                    (
                        logger.info(
                            f"Sim {i+1}: New best! Score: {score:.1f}, {node_info}"
                        )
                        if debug
                        else None
                    )
                    self.high_score_counter = 0

                # --- Conditional Simulation Detail Emission (Focus on Thought/Action) ---
                if show_sim_details and node_to_simulate:
                    sim_detail_msg = f"--- Iter {self.iterations_completed + 1} / Sim {self.current_simulation_in_iteration} ---\n"
                    sim_detail_msg += f"Selected Node: {leaf.sequence} (Visits: {leaf.visits}, Score: {leaf.get_average_score():.1f}, Tags: {leaf.descriptive_tags})\n"

                    if expanded_in_this_sim:
                        # Show the thought and the resulting NEW node info + score
                        sim_detail_msg += f'Based on thought: "{str(thought_leading_to_expansion).strip()}"\n'  # FULL THOUGHT
                        sim_detail_msg += f"--> Expanded to New Node: {node_to_simulate.sequence} ({node_to_simulate.approach_type})\n"
                        sim_detail_msg += (
                            f"    Tags: {node_to_simulate.descriptive_tags}\n"
                        )
                        # Show full analysis only for the very first expansion (Node 2)
                        if node_to_simulate.sequence == 2:
                            sim_detail_msg += f"    Initial Expanded Analysis:\n{str(node_to_simulate.content)}\n"
                    else:
                        # Describe re-evaluation, show node info + score
                        sim_detail_msg += f"--> Re-evaluating Node: {node_to_simulate.sequence} (Visits: {node_to_simulate.visits})\n"
                        sim_detail_msg += (
                            f"    Tags: {node_to_simulate.descriptive_tags}\n"
                        )

                    # Always show the score obtained in this simulation step
                    sim_detail_msg += f"Evaluated Score: {score:.1f}/10"
                    if score > best_score_before_sim:
                        sim_detail_msg += " ✨"
                    if new_best_overall:
                        sim_detail_msg += " 🏆 (New Overall Best!)"
                    sim_detail_msg += "\n"

                    await self.llm.emit_message(sim_detail_msg)  # Append to chat
                    await asyncio.sleep(0.05)
                # --- End Conditional Emission ---

                # Early Stopping Check
                # ... (early stopping logic remains the same) ...
                if cfg["early_stopping"]:
                    if score >= cfg["early_stopping_threshold"]:
                        self.high_score_counter += 1
                        if debug:
                            logger.debug(
                                f"Sim {i+1}: High score ({score:.1f}). Stability: {self.high_score_counter}/{cfg['early_stopping_stability']}"
                            )
                        if self.high_score_counter >= cfg["early_stopping_stability"]:
                            if debug:
                                logger.info(
                                    f"Early stopping criteria met after sim {i+1}, iter {self.iterations_completed + 1}."
                                )
                            self._store_iteration_snapshot(
                                "Early Stopping (High Score Stability)"
                            )
                            return self.selected
                    else:
                        self.high_score_counter = 0

            else:
                if node_to_simulate:
                    (
                        logger.warning(
                            f"Sim {i+1}: Sim failed/skipped for {node_to_simulate.sequence}. No score."
                        )
                        if debug
                        else None
                    )
                self.high_score_counter = 0

        self._store_iteration_snapshot("End of Iteration")
        if debug:
            logger.info(f"Finished Iteration {self.iterations_completed + 1}.")
        return self.selected

    # (_store_iteration_snapshot, _report_tree_stats, _collect_leaves, analyze_iteration unchanged from 0.7.12)
    def _store_iteration_snapshot(self, reason: str):
        cfg = self.config
        debug = cfg.get("debug_logging", False)
        MAX_SNAPSHOTS = 10
        if len(self.iteration_json_snapshots) >= MAX_SNAPSHOTS:
            (
                logger.warning(f"Max snapshots ({MAX_SNAPSHOTS}) reached.")
                if debug
                else None
            )
            return
        try:
            if debug:
                logger.debug(f"Storing tree snapshot: {reason}")
            snapshot = {
                "iteration": self.iterations_completed + 1,
                "simulation": self.current_simulation_in_iteration,
                "reason": reason,
                "timestamp": asyncio.get_event_loop().time(),
                "best_score_so_far": self.best_score,
                "tree_json": self.export_tree_as_json(),
            }
            self.iteration_json_snapshots.append(snapshot)
        except Exception as e:
            logger.error(f"Snapshot store failed: {e}", exc_info=debug)

    async def _report_tree_stats(self):
        cfg = self.config
        debug = cfg.get("debug_logging", False)
        try:
            total_nodes = self.node_sequence
            max_depth = self.memory.get("depth", 0)
            num_leaves = 0
            leaf_nodes = []
            self._collect_leaves(self.root, leaf_nodes)
            num_leaves = len(leaf_nodes)
            avg_branching = (
                ((total_nodes - 1) / max(1, total_nodes - num_leaves))
                if total_nodes > 1 and num_leaves < total_nodes
                else 0
            )
            stats_msg = f"### Tree Stats: Nodes={total_nodes}, Depth={max_depth}, Leaves={num_leaves}, Avg Branching={avg_branching:.2f}\n"
            if debug:
                self.debug_history.append(stats_msg)
                logger.debug(stats_msg)
        except Exception as e:
            logger.error(f"Error reporting tree stats: {e}", exc_info=debug)

    def _collect_leaves(self, node, leaf_nodes):
        if not node:
            return
        if not node.children:
            leaf_nodes.append(node)
        else:
            for child in node.children:
                if child is not None:
                    self._collect_leaves(child, leaf_nodes)

    async def analyze_iteration(self):
        RUN_ANALYSIS = False
        if not RUN_ANALYSIS:
            return None

    # (formatted_output - unchanged from 0.7.17)
    def formatted_output(self, highlighted_node=None, final_output=False) -> str:
        """Generates summary output, showing full thoughts for top nodes."""
        cfg = self.config
        debug = cfg.get("debug_logging", False)
        result = ""
        try:
            if not final_output:
                return ""
            result = f"# MCTS Final Analysis Summary\n"
            result += f"The following summarizes the MCTS exploration process, highlighting the best analysis found and the key development steps (thoughts) that led to high-scoring nodes.\n\n"

            # 1. Best Solution (Full Analysis + Tags)
            if self.best_solution:
                best_node = self.find_best_final_node()
                tags_str = (
                    f"Tags: {best_node.descriptive_tags}"
                    if best_node and best_node.descriptive_tags
                    else "Tags: []"
                )
                result += f"## Best Analysis Found (Score: {self.best_score:.1f}/10)\n"
                result += f"**{tags_str}**\n\n"
                analysis_text = str(self.best_solution)
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

            # 2. Top Performing Nodes (FULL Thought + NO Analysis Snippet)
            result += "\n## Top Performing Nodes & Driving Thoughts\n"
            all_nodes = []
            nodes_to_process = []
            processed_nodes = set()
            if self.root:
                nodes_to_process.append(self.root)
            while nodes_to_process:
                current = nodes_to_process.pop(0)
                if current is None:
                    continue
                if current.id not in processed_nodes:
                    processed_nodes.add(current.id)
                    if current.visits > 0:
                        all_nodes.append(current)
                    valid_children = [
                        child for child in current.children if child is not None
                    ]
                    nodes_to_process.extend(valid_children)
            sorted_nodes = sorted(
                all_nodes, key=lambda n: n.get_average_score(), reverse=True
            )
            top_n = 5
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
                        result += f"- **Thought**: {str(node.thought).strip()}\n"  # FULL THOUGHT
                    else:
                        result += "- **Thought**: (N/A - Initial Node)\n"
                    if node.is_surprising:
                        result += f"- **Surprising**: Yes ({truncate_text(node.surprise_explanation, 100)})\n"
                    # REMOVED Analysis Snippet Line
                    result += "\n"  # Add newline for spacing between nodes
            else:
                result += "No nodes with visits found.\n"

            # 3. Most Explored Path (Unchanged from 0.7.12)
            result += "\n## Most Explored Path\n"
            current = self.root
            path = []
            if current:
                path.append(current)
            while current and current.children:
                best_child_node = current.best_child()
                if not best_child_node or best_child_node.visits == 0:
                    (
                        logger.warning(
                            f"Path explore stop: Node {current.sequence}, Child {getattr(best_child_node, 'sequence', 'N/A')} unvisited/missing."
                        )
                        if debug
                        else None
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

            # 4. Surprising Nodes (Unchanged from 0.7.12)
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

            # 5. Approach Performance (Unchanged from 0.7.12)
            if self.approach_scores or self.approach_alphas:
                result += "\n## Thought Approach Performance\n"
                approaches_data = []
                all_apps = set(self.approach_alphas.keys()) | set(
                    self.approach_scores.keys()
                )
                for app in all_apps:
                    if app == "unknown":
                        continue
                    count = len(self.explored_approaches.get(app, []))
                    if count == 0 and app != "initial":
                        continue
                    score_str = "N/A"
                    sort_key = -1.0
                    if cfg["use_bayesian_evaluation"]:
                        alpha = self.approach_alphas.get(app, cfg["beta_prior_alpha"])
                        beta = self.approach_betas.get(app, cfg["beta_prior_beta"])
                        if (alpha + beta) > 1e-9:
                            mean_score = alpha / (alpha + beta) * 10
                            score_str = f"Score: {mean_score:.2f}/10 (α={alpha:.1f}, β={beta:.1f})"
                            sort_key = mean_score
                        else:
                            score_str = "Score: N/A (Priors?)"
                            sort_key = -1.0
                    else:
                        if app in self.approach_scores:
                            avg_score = self.approach_scores[app]
                            score_str = f"Score: {avg_score:.2f}/10"
                            sort_key = avg_score
                        elif count > 0 or app == "initial":
                            score_str = "Score: N/A"
                            sort_key = -1.0
                    approaches_data.append(
                        {
                            "name": app,
                            "score_str": score_str,
                            "count": count,
                            "sort_key": sort_key,
                        }
                    )
                sorted_approaches = sorted(
                    approaches_data, key=lambda x: x["sort_key"], reverse=True
                )
                max_show = 7
                for data in sorted_approaches[:max_show]:
                    result += f"- **{data['name']}**: {data['score_str']} ({data.get('count', 0)} thoughts)\n"
                if len(sorted_approaches) > max_show:
                    result += f"- ... ({len(sorted_approaches) - max_show} more)\n"

            # 6. Search Parameters (Unchanged from 0.7.12)
            result += f"\n## Search Parameters Used\n"
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
                result += f"- **Beta Priors**: α={cfg['beta_prior_alpha']:.2f}, β={cfg['beta_prior_beta']:.2f}\n"
            result += f"- **Exploration Weight**: {cfg['exploration_weight']:.2f}\n"
            result += (
                f"- **Early Stopping**: {'On' if cfg['early_stopping'] else 'Off'}\n"
            )
            if cfg["early_stopping"]:
                result += f"  - Threshold: {cfg['early_stopping_threshold']:.1f}/10\n"
                result += f"  - Stability: {cfg['early_stopping_stability']}\n"
            result += f"- **Show Sim Details**: {'On' if cfg.get('show_simulation_details', False) else 'Off'}\n"

            # 7. Conditional Debug (Unchanged fix from 0.7.13)
            if debug and self.debug_history:
                result += "\n## Debug Log Snippets (Last 3)\n\n"
                for entry in self.debug_history[-3:]:
                    cleaned_entry = re.sub(r"\n+", "\n", entry).strip()
                    result += truncate_text(cleaned_entry, 200) + "\n---\n"

            return result.strip()
        except Exception as e:
            logger.error(f"Error formatting output: {e}", exc_info=debug)
            error_msg = (
                f"\n\n# Error generating final summary:\n{type(e).__name__}: {str(e)}\n"
            )
            result += error_msg
            return result

    # (find_best_final_node - unchanged from 0.7.14)
    def find_best_final_node(self) -> Optional[Node]:
        # ... (implementation unchanged) ...
        if not self.best_solution:
            return None
        queue = []
        visited = set()
        best_match_node = None
        min_score_diff = float("inf")
        if self.root:
            queue.append(self.root)
            visited.add(self.root.id)
        while queue:
            current = queue.pop(0)
            if current is None:
                continue
            node_content = str(current.content)
            node_content = re.sub(
                r"^```(json|markdown)?\s*",
                "",
                node_content,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            node_content = re.sub(
                r"\s*```$", "", node_content, flags=re.MULTILINE
            ).strip()
            best_sol_content = str(self.best_solution)
            best_sol_content = re.sub(
                r"^```(json|markdown)?\s*",
                "",
                best_sol_content,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            best_sol_content = re.sub(
                r"\s*```$", "", best_sol_content, flags=re.MULTILINE
            ).strip()
            if node_content == best_sol_content:
                score_diff = abs(current.get_average_score() - self.best_score)
                if score_diff < 0.1 or best_match_node is None:
                    best_match_node = current
                    min_score_diff = score_diff
                elif score_diff < min_score_diff:
                    best_match_node = current
                    min_score_diff = score_diff
            valid_children = [
                child for child in current.children if child and child.id not in visited
            ]
            for child in valid_children:
                visited.add(child.id)
                queue.append(child)
        if not best_match_node:
            logger.warning("Could not find node object for best solution content.")
        return best_match_node


# ==============================================================================


# (Pipe class - Valves, init, pipes, resolve_model, resolve_question unchanged from 0.7.12)
class Pipe:
    """Interface with Open WebUI, running MCTS for analysis exploration."""

    # ... (Valves remain the same) ...
    class Valves(BaseModel):
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
        SCORE_DIVERSITY_BONUS: float = Field(
            default=default_config["score_diversity_bonus"],
            title="UCT Score Diversity Bonus",
            ge=0.0,
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
        SHOW_SIMULATION_DETAILS: bool = Field(
            default=default_config["show_simulation_details"],
            title="Show Per-Simulation Details in Chat",
        )
        DEBUG_LOGGING: bool = Field(
            default=default_config["debug_logging"],
            title="Enable Detailed Debug Logging",
        )

    def __init__(self):
        self.type = "manifold"
        self.__current_event_emitter__ = None
        self.__question__ = ""
        self.__model__ = ""
        self.__llm_client__ = None

    def pipes(self) -> list[dict[str, str]]:
        # ... (implementation unchanged) ...
        try:
            if not hasattr(app.state, "OLLAMA_MODELS") or not app.state.OLLAMA_MODELS:
                logger.info("Loading Ollama models...")
                ollama.get_all_models()
            if hasattr(app.state, "OLLAMA_MODELS") and app.state.OLLAMA_MODELS:
                models = app.state.OLLAMA_MODELS
                valid_models = {
                    k: v
                    for k, v in models.items()
                    if isinstance(v, dict) and "name" in v
                }
                if not valid_models:
                    logger.warning("No valid models found.")
                    return [{"id": f"{name}-error", "name": f"{name} (No models)"}]
                return [
                    {"id": f"{name}-{k}", "name": f"{name} ({v['name']})"}
                    for k, v in valid_models.items()
                ]
            else:
                logger.error("OLLAMA_MODELS not found.")
                return [{"id": f"{name}-error", "name": f"{name} (Model load error)"}]
        except Exception as e:
            logger.error(
                f"Pipe list failed: {e}",
                exc_info=default_config.get("debug_logging", False),
            )
            return [{"id": f"{name}-error", "name": f"{name} (Error: {e})"}]

    def resolve_model(self, body: dict) -> str:
        # ... (implementation unchanged) ...
        model_id = body.get("model", "").strip()
        pipe_internal_name = name
        prefix_to_find = f"{pipe_internal_name}-"
        separator_index = model_id.rfind(prefix_to_find)
        if separator_index != -1:
            base_model_name = model_id[separator_index + len(prefix_to_find) :]
            if base_model_name:
                if ":" not in base_model_name:
                    logger.warning(
                        f"Resolved '{base_model_name}', missing tag? Using anyway."
                    )
                else:
                    logger.info(
                        f"Resolved base model '{base_model_name}' from '{model_id}'"
                    )
                return base_model_name
            else:
                logger.error(
                    f"Separator found in '{model_id}' but no model name. Falling back."
                )
                return model_id
        else:
            logger.warning(
                f"Separator '{prefix_to_find}' not found in '{model_id}'. Assuming base name."
            )
            return model_id

    def resolve_question(self, body: dict) -> str:
        # ... (implementation unchanged) ...
        msgs = body.get("messages", [])
        for msg in reversed(msgs):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                return content.strip() if isinstance(content, str) else ""
        return ""

    ### MODIFIED: pipe - Add iteration summary and final synthesis call ###
    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        __task__=None,
        __model__=None,
    ) -> Union[str, AsyncGenerator[str, None]]:
        self.__current_event_emitter__ = __event_emitter__
        mcts_instance = None
        current_config = default_config.copy()
        initial_analysis_text = ""
        try:
            model = self.resolve_model(body)
            input_text = self.resolve_question(body)
            if not input_text:
                await self.emit_message("**Error:** No input text provided.")
                await self.done()
                return "Error: No input text."
            if not model:
                await self.emit_message("**Error:** No model identified.")
                await self.done()
                return "Error: Model not identified."
            self.__model__ = model
            self.__question__ = input_text
            if __task__ == TASKS.TITLE_GENERATION:
                logger.info(f"Handling TITLE_GENERATION.")
                completion = await self.get_completion(
                    model,
                    [{"role": "user", "content": f"Generate title for: {input_text}"}],
                )
                await self.done()
                return f"{name}: {truncate_text(completion, 50)}"
            logger.info(
                f"Pipe '{name}' v0.7.18 starting. Model: {model}, Input: '{truncate_text(input_text)}'"
            )  # Updated version
            if hasattr(self, "valves") and isinstance(
                self.valves, BaseModel
            ):  # Apply Valves
                logger.info("Applying Valve settings...")
                try:
                    valve_dict = self.valves.model_dump()
                    for key_upper, value in valve_dict.items():
                        key_lower = key_upper.lower()
                        if key_lower in current_config:
                            current_config[key_lower] = value
                        if key_lower == "debug_logging":
                            setup_logger(logging.DEBUG if value else logging.INFO)
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
                    logger.info("Valve settings applied.")
                except Exception as e:
                    logger.error(f"Valve apply error: {e}. Defaults.", exc_info=True)
                    await self.emit_message(f"**Warn:** Valve error: {e}. Defaults.")
                    current_config = default_config.copy()
            else:
                logger.info("No valves. Using defaults.")
            setup_logger(
                logging.DEBUG if current_config["debug_logging"] else logging.INFO
            )
            debug_this_run = current_config["debug_logging"]
            await self.emit_message(
                f'# Advanced Bayesian MCTS v0.7.18\n*Exploring analysis for:* "{truncate_text(input_text, 100)}" *using model* `{model}`.\n'
            )  # Updated version
            params_to_show = {
                k: v
                for k, v in current_config.items()
                if k
                in [
                    "max_iterations",
                    "simulations_per_iteration",
                    "exploration_weight",
                    "early_stopping",
                    "show_simulation_details",
                ]
            }
            logger.info(f"--- Params ---\n{json.dumps(params_to_show, indent=2)}")
            if debug_this_run:
                logger.debug(
                    f"--- Full Config ---\n{json.dumps(current_config, indent=2)}"
                )
            await self.emit_message(
                "Params configured (logs). Starting initial analysis..."
            )
            await self.progress("Generating initial analysis...")  # Initial Analysis
            initial_analysis_text = await self.stream_prompt_completion(
                initial_prompt, question=input_text
            )
            if (
                not isinstance(initial_analysis_text, str)
                or "Error:" in initial_analysis_text
            ):
                logger.error(f"Initial analysis fail: {initial_analysis_text}")
                await self.emit_message(
                    f"**Error:** Initial analysis failed: {initial_analysis_text}"
                )
                await self.done()
                return f"Error: {initial_analysis_text}"
            initial_analysis_text = initial_analysis_text.strip()
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
                logger.error("Initial analysis empty.")
                await self.emit_message("**Error:** Initial analysis empty.")
                await self.done()
                return "Error: Empty initial analysis."
            await self.emit_message("\n## Initial Analysis\n" + initial_analysis_text)
            await self.emit_message("\n---\n")
            await asyncio.sleep(0.1)
            await self.progress("Initializing MCTS...")
            root_stub = Node(content=initial_analysis_text)  # MCTS Init
            mcts_instance = MCTS(
                root=root_stub,
                llm=self,
                question=input_text,
                mcts_config=current_config,
            )
            logger.info("Starting MCTS iterations...")  # MCTS Loop
            show_details = current_config.get("show_simulation_details", True)
            if show_details:
                await self.emit_message(
                    "🚀 **Starting MCTS Exploration...** (Showing MCTS process steps)"
                )
            else:
                await self.emit_message(
                    "🚀 **Starting MCTS Exploration...** (Simulation details hidden)"
                )

            final_best_analysis = initial_analysis_text
            for i in range(current_config["max_iterations"]):
                iteration_num = i + 1
                (
                    logger.info(
                        f"--- Starting Iteration {iteration_num}/{current_config['max_iterations']} ---"
                    )
                    if debug_this_run
                    else None
                )
                await self.progress(
                    f"Running MCTS Iteration {iteration_num}/{current_config['max_iterations']}..."
                )
                best_score_before_iter = mcts_instance.best_score

                # Run the simulations for this iteration
                await mcts_instance.search(current_config["simulations_per_iteration"])
                mcts_instance.iterations_completed += 1  # Increment after search

                # --- Generate and Emit Iteration Summary ---
                iter_best_node = (
                    mcts_instance.find_best_final_node()
                )  # Get current overall best
                iter_summary_msg = f"\n**--- Iteration {iteration_num} Summary ---**\n"
                iter_summary_msg += (
                    f"- Overall Best Score So Far: {mcts_instance.best_score:.1f}/10"
                )
                if mcts_instance.best_score > best_score_before_iter:
                    iter_summary_msg += " (✨ New best found this iteration!)"
                else:
                    iter_summary_msg += " (Best score unchanged this iteration)"
                if iter_best_node:
                    tags_str = (
                        f"Tags: {iter_best_node.descriptive_tags}"
                        if iter_best_node.descriptive_tags
                        else "Tags: []"
                    )
                    iter_summary_msg += (
                        f"\n- Current Best Node: {iter_best_node.sequence} ({tags_str})"
                    )
                else:
                    iter_summary_msg += (
                        "\n- (Could not identify current best node object)"
                    )
                iter_summary_msg += "\n-------------------------------\n"
                await self.emit_message(iter_summary_msg)  # Append summary to chat
                # --- End Iteration Summary ---

                # Early Stopping Check (check score *before* counter)
                if (
                    current_config["early_stopping"]
                    and mcts_instance.best_score
                    >= current_config["early_stopping_threshold"]
                    and mcts_instance.high_score_counter
                    >= current_config["early_stopping_stability"]
                ):
                    logger.info(
                        f"Early stopping criteria met after iter {iteration_num}."
                    )
                    await self.emit_message(
                        f"**Stopping early:** Analysis ({mcts_instance.best_score:.1f}/10) met stability criteria."
                    )
                    break

                await asyncio.sleep(0.1)  # Small delay between iterations

            # --- End of Iteration Loop ---

            logger.info("MCTS iterations finished.")  # Final Output
            await self.emit_message(
                "\n🏁 **MCTS Exploration Finished.** Preparing final analysis summary..."
            )
            if (
                isinstance(mcts_instance.best_solution, str)
                and mcts_instance.best_solution
            ):
                final_best_analysis = mcts_instance.best_solution
            else:
                logger.warning("MCTS best_solution invalid. Using initial.")
                final_best_analysis = initial_analysis_text
            # --- Emit the main formatted summary ---
            await self.emit_message(mcts_instance.formatted_output(final_output=True))

            # --- Final Synthesis Step ---
            await self.progress("Generating final synthesis...")
            await self.emit_message("\n---\n## Final Synthesis\n")
            try:
                best_node_final = mcts_instance.find_best_final_node()
                path_thoughts_list = []
                # Construct path from root to best node to get chronological thoughts
                path_to_best = []
                temp_node = best_node_final
                while temp_node:
                    path_to_best.append(temp_node)
                    temp_node = temp_node.parent
                path_to_best.reverse()  # Root first

                for node in path_to_best:
                    # Include thought only if it's not the root node's (which has no thought)
                    if node.thought and node.parent:
                        path_thoughts_list.append(
                            f"- (From Node {node.parent.sequence} -> Node {node.sequence}): {node.thought.strip()}"
                        )

                if (
                    not path_thoughts_list and len(path_to_best) <= 1
                ):  # Handle case where only root exists or path failed
                    logger.warning(
                        "No thoughts found on path to best node for synthesis."
                    )
                    await self.emit_message(
                        "Could not retrieve development path for synthesis."
                    )
                else:
                    path_thoughts_str = "\n".join(path_thoughts_list)
                    synthesis_context = {
                        "question_summary": mcts_instance.question_summary,
                        "initial_analysis_summary": truncate_text(
                            initial_analysis_text, 300
                        ),
                        "best_score": f"{mcts_instance.best_score:.1f}",
                        "path_thoughts": (
                            path_thoughts_str
                            if path_thoughts_str
                            else "No significant development path found."
                        ),
                        "final_best_analysis_summary": truncate_text(
                            final_best_analysis, 400
                        ),
                    }
                    synthesis_text = await self.stream_prompt_completion(
                        final_synthesis_prompt, **synthesis_context
                    )
                    if "Error:" in synthesis_text:
                        await self.emit_message(
                            f"**Warning:** Synthesis generation failed: {synthesis_text}"
                        )
                    else:
                        await self.emit_message(synthesis_text)
            except Exception as synth_err:
                logger.error(
                    f"Final synthesis failed: {synth_err}", exc_info=debug_this_run
                )
                await self.emit_message(
                    "**Error:** Failed to generate final synthesis."
                )
            # --- End Final Synthesis Step ---

            EXPORT_SNAPSHOTS = False
            if EXPORT_SNAPSHOTS and debug_this_run:
                logger.debug("--- JSON Snapshots ---")
            if mcts_instance.iteration_json_snapshots:
                for i, snapshot in enumerate(mcts_instance.iteration_json_snapshots):
                    try:
                        json_str = json.dumps(snapshot["tree_json"], indent=2)
                        logger.debug(
                            f"Snap {i+1} Iter {snapshot['iteration']} ({snapshot['reason']}):\n{truncate_text(json_str, 5000)}"
                        )
                    except Exception as json_err:
                        logger.error(f"JSON log err {i+1}: {json_err}")
            else:
                logger.debug("No snapshots.") if debug_this_run else None
            await self.done()
            logger.info(f"Pipe '{name}' finished.")
            final_analysis_cleaned = str(final_best_analysis)
            final_analysis_cleaned = re.sub(
                r"^```(json|markdown)?\s*",
                "",
                final_analysis_cleaned,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            final_analysis_cleaned = re.sub(
                r"\s*```$", "", final_analysis_cleaned, flags=re.MULTILINE
            ).strip()
            return final_analysis_cleaned

        except Exception as e:
            logger.error(f"FATAL Pipe Error: {e}", exc_info=True)
            try:
                await self.emit_message(
                    f"\n\n**FATAL ERROR:**\n```\n{type(e).__name__}: {str(e)}\n```\nPipe stopped."
                )
            except Exception as emit_err:
                logger.error(f"Emit fatal error failed: {emit_err}")
            return f"Error: Pipe failed unexpectedly. Check logs. ({type(e).__name__})"
        finally:
            await self.done()
            await self.cleanup()

    # --- LLM Interaction & Helper Methods (Unchanged from 0.7.17) ---
    async def progress(self, message: str):
        # ... (implementation unchanged) ...
        debug = default_config.get("debug_logging", False)
        if self.__current_event_emitter__:
            try:
                if debug:
                    logger.debug(f"Progress: {message}")
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

    async def done(self):
        # ... (implementation unchanged) ...
        debug = default_config.get("debug_logging", False)
        if self.__current_event_emitter__:
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
            finally:
                self.__current_event_emitter__ = None

    async def emit_message(self, message: str):
        # ... (implementation unchanged) ...
        if self.__current_event_emitter__:
            try:
                await self.__current_event_emitter__(
                    {"type": "message", "data": {"content": str(message)}}
                )
            except Exception as e:
                logger.error(f"Emit message error: {e} (Msg: {str(message)[:100]}...)")

    async def emit_replace(self, message: str):
        # ... (implementation unchanged) ...
        if self.__current_event_emitter__:
            try:
                await self.__current_event_emitter__(
                    {"type": "replace", "data": {"content": str(message)}}
                )
            except Exception as e:
                logger.error(f"Emit replace error: {e}")

    def get_chunk_content(self, chunk_bytes: bytes) -> Generator[str, None, None]:
        # ... (implementation unchanged) ...
        debug = default_config.get("debug_logging", False)
        try:
            chunk_str = chunk_bytes.decode("utf-8")
            for line in chunk_str.splitlines():
                line = line.strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    json_data_str = line[6:]
                    try:
                        chunk_data = json.loads(json_data_str)
                        if (
                            isinstance(chunk_data, dict)
                            and "choices" in chunk_data
                            and isinstance(chunk_data["choices"], list)
                            and chunk_data["choices"]
                            and isinstance(chunk_data["choices"][0].get("delta"), dict)
                        ):
                            content = chunk_data["choices"][0]["delta"].get("content")
                            if isinstance(content, str) and content:
                                yield content
                    except json.JSONDecodeError:
                        logger.warning(f"JSON decode error: {json_data_str}")
                    except Exception as e:
                        logger.error(f"Chunk processing error: {e}")
        except UnicodeDecodeError:
            logger.error(f"Decode error: {chunk_bytes[:100]}...")
        except Exception as e:
            logger.error(f"Error in get_chunk_content: {e}", exc_info=debug)
        return

    async def get_streaming_completion(
        self, model: str, messages: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        # ... (implementation unchanged) ...
        response = None
        debug = default_config.get("debug_logging", False)
        try:
            response = await self.call_ollama_endpoint_function(
                {"model": model, "messages": messages, "stream": True}
            )
            if isinstance(response, dict) and response.get("error"):
                err_msg = self.get_response_content(response)
                logger.error(f"LLM stream start failed: {err_msg}")
                yield err_msg
                return
            if hasattr(response, "body_iterator"):
                async for chunk_bytes in response.body_iterator:
                    for part in self.get_chunk_content(chunk_bytes):
                        if part:
                            yield part
            elif isinstance(response, dict):
                content = self.get_response_content(response)
                if content:
                    logger.warning("Expected stream, got dict.")
                    yield content
                else:
                    logger.error(f"Expected stream, bad dict: {str(response)[:200]}")
                    yield "Error: Invalid LLM dict."
            else:
                logger.error(f"Expected stream/dict, got {type(response)}.")
                yield f"Error: Bad LLM type ({type(response)})."
        except AttributeError as ae:
            logger.error(f"AttributeError during streaming: {ae}", exc_info=debug)
            yield f"Error during streaming: {str(ae)}"
        except Exception as e:
            logger.error(f"LLM stream processing error: {e}", exc_info=debug)
            yield f"Error during streaming: {str(e)}"
        finally:
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
    ) -> AsyncGenerator[str, None]:
        # ... (implementation unchanged) ...
        debug = default_config.get("debug_logging", False)
        try:
            async for chunk in self.get_streaming_completion(
                model, [{"role": "user", "content": str(content)}]
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Error in get_message_completion: {e}", exc_info=debug)
            yield f"Error: {str(e)}"

    async def get_completion(self, model: str, messages: List[Dict[str, str]]) -> str:
        # ... (implementation unchanged) ...
        debug = default_config.get("debug_logging", False)
        try:
            response = await self.call_ollama_endpoint_function(
                {"model": model, "messages": messages, "stream": False}
            )
            content = self.get_response_content(response)
            if isinstance(response, dict) and response.get("error"):
                return f"Error: {content}"
            return content
        except Exception as e:
            logger.error(f"Error in get_completion: {e}", exc_info=debug)
            return f"Error: {str(e)}"

    async def call_ollama_endpoint_function(self, payload: Dict[str, Any]):
        # ... (implementation unchanged) ...
        debug = default_config.get("debug_logging", False)
        try:

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
                logger.debug(f"Calling internal ollama: {str(payload)[:200]}...")
            response = await ollama.generate_openai_chat_completion(
                request=mock_request, form_data=payload, user=admin
            )
            if (
                debug
                and not isinstance(response, dict)
                and not hasattr(response, "body_iterator")
            ):
                logger.debug(f"Internal endpoint response type: {type(response)}")
            return response
        except Exception as e:
            logger.error(f"Ollama internal call error: {str(e)}", exc_info=debug)
            return {
                "error": True,
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"Error: LLM call failed ({str(e)[:100]}...). See logs.",
                        }
                    }
                ],
            }

    async def stream_prompt_completion(self, prompt: str, **format_args) -> str:
        # ... (implementation unchanged) ...
        debug = default_config.get("debug_logging", False)
        complete_response = ""
        error_occurred = False
        safe_format_args = {
            k: str(v) if v is not None else "" for k, v in format_args.items()
        }
        try:
            formatted_prompt = prompt.format(**safe_format_args)
        except KeyError as e:
            logger.error(
                f"Prompt fmt Key error: '{e}'. Keys: {list(safe_format_args.keys())}"
            )
            return f"Error: Prompt Key {e}."
        except Exception as e:
            logger.error(f"Prompt fmt error: {e}", exc_info=debug)
            return f"Error: Prompt format ({e})."
        try:
            async for chunk in self.get_message_completion(
                self.__model__, formatted_prompt
            ):
                if chunk is not None:
                    chunk_str = str(chunk)
                    if chunk_str.startswith("Error during streaming:"):
                        logger.error(f"LLM stream error reported: {chunk_str}")
                        complete_response = chunk_str
                        error_occurred = True
                        break
                    if chunk_str.startswith("Error:"):
                        logger.error(f"LLM stream error: {chunk_str}")
                        complete_response = chunk_str
                        error_occurred = True
                        break
                    complete_response += chunk_str
            if error_occurred:
                return complete_response
            clean_response = str(complete_response).strip()
            clean_response = re.sub(
                r"\n*(?:Would you like me to.*?|Do you want to explore.*?|Is there anything else.*?)\??\s*$",
                "",
                clean_response,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()
            clean_response = re.sub(
                r"^```(json|markdown)?\s*",
                "",
                clean_response,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            clean_response = re.sub(
                r"\s*```$", "", clean_response, flags=re.MULTILINE
            ).strip()
            return clean_response
        except Exception as e:
            logger.error(f"LLM stream failed: {e}", exc_info=debug)
            return f"Error: LLM stream failed ({e})."

    async def generate_thought(
        self, current_analysis: str, context: Dict, config: Dict
    ) -> str:
        # ... (implementation unchanged) ...
        format_args = context.copy()
        format_args.setdefault("question_summary", "N/A")
        format_args.setdefault("best_answer", "N/A")
        format_args.setdefault("best_score", "0.0")
        format_args.setdefault("current_answer", truncate_text(current_analysis, 300))
        format_args.setdefault("current_sequence", "N/A")
        format_args.setdefault("current_tags", "None")
        result = await self.stream_prompt_completion(thoughts_prompt, **format_args)
        return result if isinstance(result, str) and "Error:" not in result else ""

    async def update_approach(
        self, original_analysis: str, critique: str, context: Dict, config: Dict
    ) -> str:
        # ... (implementation unchanged) ...
        format_args = context.copy()
        format_args["answer"] = original_analysis
        format_args["improvements"] = critique.strip()
        format_args.setdefault("question_summary", "N/A")
        format_args.setdefault("best_answer", "N/A")
        format_args.setdefault("best_score", "0.0")
        format_args.setdefault("current_tags", "None")
        result = await self.stream_prompt_completion(update_prompt, **format_args)
        if isinstance(result, str) and result.strip() and "Error:" not in result:
            clean_result = re.sub(
                r"^```(json|markdown)?\s*",
                "",
                result,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            clean_result = re.sub(
                r"\s*```$", "", clean_result, flags=re.MULTILINE
            ).strip()
            return clean_result if clean_result else str(original_analysis)
        return str(original_analysis)

    async def evaluate_answer(
        self, analysis_to_evaluate: str, context: Dict, config: Dict
    ) -> int:
        # ... (implementation uses tweaked eval_answer_prompt) ...
        format_args = context.copy()
        format_args["answer_to_evaluate"] = analysis_to_evaluate
        format_args.setdefault("question_summary", "N/A")
        format_args.setdefault("best_answer", "N/A")
        format_args.setdefault("best_score", "0.0")
        format_args.setdefault("current_tags", "None")
        result = await self.stream_prompt_completion(eval_answer_prompt, **format_args)
        if not isinstance(result, str) or "Error:" in result:
            logger.warning(f"Eval failed: {result}")
            return 5
        score_match = re.search(r"^\s*([1-9]|10)\s*$", result.strip())
        if score_match:
            try:
                return int(score_match.group(1))
            except ValueError:
                logger.warning(f"Eval parse error (strict): '{result}'")
                return 5
        else:
            logger.warning(f"Eval strict score not found: '{result}'. Trying relaxed.")
            relaxed_match = re.search(r"\b([1-9]|10)\b", result.strip())
            if relaxed_match:
                try:
                    return int(relaxed_match.group(1))
                except ValueError:
                    logger.warning(f"Eval parse error (relaxed): '{result}'")
                    return 5
            else:
                logger.warning(f"Eval score not found (relaxed): '{result}'")
                return 5

    async def evaluate_relative(
        self,
        parent_answer: str,
        answer: str,
        context: Optional[Dict] = None,
        config: Optional[Dict] = None,
    ) -> int:
        # ... (implementation unchanged) ...
        return 3

    def get_response_content(self, response: Union[Dict, Any]) -> str:
        # ... (implementation unchanged) ...
        try:
            if isinstance(response, dict):
                if response.get("error"):
                    if (
                        "choices" in response
                        and response["choices"]
                        and isinstance(response["choices"][0].get("message"), dict)
                    ):
                        return str(response["choices"][0]["message"].get("content", ""))
                    return f"Error: {response.get('error', 'Unknown LLM Error')}"
                elif (
                    "choices" in response
                    and isinstance(response["choices"], list)
                    and response["choices"]
                    and isinstance(response["choices"][0].get("message"), dict)
                ):
                    return str(response["choices"][0]["message"].get("content", ""))
            logger.warning(
                f"Unexpected response structure in get_response_content: {type(response)}"
            )
            return ""
        except Exception as e:
            logger.error(f"Response content extraction error: {str(e)}", exc_info=True)
            return ""

    async def cleanup(self):
        # ... (implementation unchanged) ...
        debug = default_config.get("debug_logging", False)
        if debug:
            logger.info("Pipe cleanup initiated...")
        self.__current_event_emitter__ = None
        if self.__llm_client__ and hasattr(self.__llm_client__, "close"):
            try:
                await self.__llm_client__.close()
                logger.info("Closed persistent client.")
            except Exception as e:
                logger.error(f"Error closing client: {e}")
        self.__llm_client__ = None
        try:
            gc.collect()
        except Exception as e:
            logger.error(f"GC error: {e}")
        if debug:
            logger.info("Pipe cleanup complete.")


# ==============================================================================
# FILE END
# ==============================================================================
