"""Feature extraction module for CodeGhost Risk Scoring (Milestone 4).

Extracts a reproducible 15-dimensional numeric feature vector combining:
1. Retrieval evidence features (similarity metrics from vector store)
2. AST code structural & complexity features (LOC, nesting depth, branches, loops, cyclomatic complexity)
3. Historical file- and function-level bug frequency features
"""

import ast
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.ingestion.ast_parser import ASTParserManager
from backend.retrieval.schemas import RetrievedMatch
from backend.risk.schemas import RiskExample

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "top1_similarity",
    "top3_max_similarity",
    "topk_avg_similarity",
    "num_strong_matches",
    "similarity_std",
    "lines_of_code",
    "num_parameters",
    "max_nesting_depth",
    "ast_node_count",
    "num_branches",
    "num_loops",
    "has_exception_handling",
    "cyclomatic_complexity",
    "hist_bug_count_function",
    "hist_bug_count_file",
]


class CodeRiskFeatureExtractor:
    """Extracts numeric feature vectors from code, retrieval results, and repository history."""

    def __init__(self):
        self._ast_manager: Optional[ASTParserManager] = None

    @property
    def ast_manager(self) -> ASTParserManager:
        if self._ast_manager is None:
            self._ast_manager = ASTParserManager()
        return self._ast_manager

    def extract_features(
        self,
        code: str,
        language: str = "python",
        retrieved_matches: Optional[List[RetrievedMatch]] = None,
        function_name: Optional[str] = None,
        file_path: Optional[str] = None,
        historical_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Extract all 15 risk features for a given function and its retrieved historical matches.

        Args:
            code: Source code of the function.
            language: Language of the code (default: 'python').
            retrieved_matches: List of RetrievedMatch items from vector database.
            function_name: Name of the function.
            file_path: Path of the file.
            historical_stats: Optional lookup dict with prior bug counts.

        Returns:
            Dictionary mapping feature names to float values.
        """
        retrieved_matches = retrieved_matches or []
        features: Dict[str, float] = {}

        # ---------------------------------------------------------
        # 1. Retrieval Evidence Features
        # ---------------------------------------------------------
        sims = [m.similarity_score for m in retrieved_matches]
        if sims:
            features["top1_similarity"] = float(sims[0])
            features["top3_max_similarity"] = float(max(sims[:3]))
            features["topk_avg_similarity"] = float(sum(sims) / len(sims))
            features["num_strong_matches"] = float(sum(1 for s in sims if s >= 0.85))
            if len(sims) >= 2:
                mean = sum(sims) / len(sims)
                variance = sum((s - mean) ** 2 for s in sims) / len(sims)
                features["similarity_std"] = float(math.sqrt(variance))
            else:
                features["similarity_std"] = 0.0
        else:
            features["top1_similarity"] = 0.0
            features["top3_max_similarity"] = 0.0
            features["topk_avg_similarity"] = 0.0
            features["num_strong_matches"] = 0.0
            features["similarity_std"] = 0.0

        # ---------------------------------------------------------
        # 2. AST Code & Complexity Features
        # ---------------------------------------------------------
        code_features = self._compute_code_features(code, language)
        features.update(code_features)

        # ---------------------------------------------------------
        # 3. Historical Bug Frequency Features
        # ---------------------------------------------------------
        hist = historical_stats or {}
        fn_key = f"fn::{function_name}" if function_name else ""
        file_key = f"file::{file_path}" if file_path else ""

        features["hist_bug_count_function"] = float(hist.get(fn_key, 0))
        features["hist_bug_count_file"] = float(hist.get(file_key, 0))

        return features

    def extract_from_example(
        self,
        example: RiskExample,
        historical_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Convenience method to extract features directly from a RiskExample."""
        return self.extract_features(
            code=example.code,
            language=example.language,
            retrieved_matches=example.retrieved_matches,
            function_name=example.function_name,
            file_path=example.file_path,
            historical_stats=historical_stats,
        )

    def features_to_vector(self, features: Dict[str, float]) -> List[float]:
        """Convert a features dictionary into a flat ordered float vector."""
        return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]

    def extract_dataset_matrix(
        self,
        examples: List[RiskExample],
        historical_stats: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract feature matrix X and label vector y for a list of RiskExamples.

        Returns:
            (X, y, feature_names) where X is shape (N, 15) and y is shape (N,).
        """
        if not examples:
            return np.empty((0, len(FEATURE_NAMES))), np.empty((0,)), FEATURE_NAMES

        X_rows = []
        y_rows = []
        for ex in examples:
            feat_dict = self.extract_from_example(ex, historical_stats=historical_stats)
            X_rows.append(self.features_to_vector(feat_dict))
            y_rows.append(ex.label)

        return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32), FEATURE_NAMES

    def _compute_code_features(self, code: str, language: str) -> Dict[str, float]:
        """Compute AST and complexity features using Tree-sitter or Python fallback."""
        lines = [line for line in code.splitlines() if line.strip()]
        loc = len(lines)

        num_params = 0
        max_nesting = 0
        ast_nodes = 0
        branches = 0
        loops = 0
        has_exceptions = 0

        # Try Python native ast first for Python code for fast detailed analysis
        parsed_with_py_ast = False
        if language.lower() == "python":
            try:
                # If code is indented (e.g. inside a class), dedent it
                import textwrap
                dedented = textwrap.dedent(code)
                py_tree = ast.parse(dedented)
                parsed_with_py_ast = True

                for node in ast.walk(py_tree):
                    ast_nodes += 1
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        num_params = max(num_params, len(node.args.args))
                    elif isinstance(node, (ast.If, ast.IfExp)):
                        branches += 1
                    elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                        loops += 1
                    elif isinstance(node, (ast.Try, ast.TryStar)):
                        has_exceptions = 1

                # Calculate max nesting depth
                def _get_depth(node: ast.AST, current_depth: int) -> int:
                    max_d = current_depth
                    is_nesting_construct = isinstance(
                        node,
                        (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith),
                    )
                    next_depth = current_depth + 1 if is_nesting_construct else current_depth
                    for child in ast.iter_child_nodes(node):
                        max_d = max(max_d, _get_depth(child, next_depth))
                    return max_d

                max_nesting = _get_depth(py_tree, 0)
            except Exception:
                parsed_with_py_ast = False

        # Fallback to Tree-sitter if not Python or if python ast failed
        if not parsed_with_py_ast:
            try:
                lang = language.lower()
                if self.ast_manager.is_language_supported(lang):
                    tree = self.ast_manager.parse_source(code, lang)
                    cursor = tree.walk()

                    def traverse(c, depth: int):
                        nonlocal ast_nodes, branches, loops, has_exceptions, max_nesting
                        ast_nodes += 1
                        node_type = c.node.type

                        is_branch = node_type in (
                            "if_statement", "elif_clause", "conditional_expression",
                            "switch_statement", "case_statement", "match_statement"
                        )
                        is_loop = node_type in ("for_statement", "while_statement", "do_statement")
                        is_try = node_type in ("try_statement", "catch_clause", "except_clause")

                        if is_branch:
                            branches += 1
                        if is_loop:
                            loops += 1
                        if is_try:
                            has_exceptions = 1

                        new_depth = depth + 1 if (is_branch or is_loop or is_try) else depth
                        max_nesting = max(max_nesting, new_depth)

                        if c.goto_first_child():
                            traverse(c, new_depth)
                            while c.goto_next_sibling():
                                traverse(c, new_depth)
                            c.goto_parent()

                    traverse(cursor, 0)
                else:
                    ast_nodes = max(1, loc * 3)
            except Exception as e:
                logger.debug("Tree-sitter feature extraction fallback triggered: %s", e)
                ast_nodes = max(1, loc * 3)

        cyclomatic = 1 + branches + loops

        return {
            "lines_of_code": float(loc),
            "num_parameters": float(num_params),
            "max_nesting_depth": float(max_nesting),
            "ast_node_count": float(ast_nodes),
            "num_branches": float(branches),
            "num_loops": float(loops),
            "has_exception_handling": float(has_exceptions),
            "cyclomatic_complexity": float(cyclomatic),
        }
