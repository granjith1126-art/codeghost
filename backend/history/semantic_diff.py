"""AST Semantic Diff Analyzer for CodeGhost (Milestone 9).

Determines whether code changes between buggy and fixed code units represent
substantive semantic modifications (logic, control flow, arithmetic, calls)
or non-semantic changes (comments, formatting, docstrings, type annotations).
Eliminates label noise from mined bug-fix datasets.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from tree_sitter import Node, Tree

from backend.ingestion.ast_parser import ASTParserManager


@dataclass
class SemanticDiffResult:
    """Outcome of AST-level semantic comparison between two code snippets."""
    is_semantic: bool
    change_type: str  # SEMANTIC_CODE_CHANGE, COMMENT_ONLY, DOCSTRING_ONLY, FORMATTING_ONLY, IDENTICAL
    details: str
    ast_diff_summary: Optional[str] = None
    buggy_node_count: int = 0
    fixed_node_count: int = 0


class ASTSemanticDiffAnalyzer:
    """Analyzes whether differences between code units are semantic or non-semantic."""

    NON_SEMANTIC_NODE_TYPES: Set[str] = {
        "comment",
        "line_comment",
        "block_comment",
    }

    def __init__(self, parser_manager: Optional[ASTParserManager] = None):
        self.parser_manager = parser_manager or ASTParserManager()

    def analyze(self, code_a: str, code_b: str, language: str = "python") -> SemanticDiffResult:
        """Compare code_a (buggy) and code_b (fixed) at the AST level."""
        # 1. Exact text identity check
        if code_a == code_b:
            return SemanticDiffResult(
                is_semantic=False,
                change_type="IDENTICAL",
                details="Code units are identical in text.",
                buggy_node_count=0,
                fixed_node_count=0,
            )

        # 2. Whitespace-only difference check
        norm_a = re.sub(r"\s+", " ", code_a).strip()
        norm_b = re.sub(r"\s+", " ", code_b).strip()
        if norm_a == norm_b:
            return SemanticDiffResult(
                is_semantic=False,
                change_type="FORMATTING_ONLY",
                details="Code difference is whitespace / indentation only.",
                buggy_node_count=0,
                fixed_node_count=0,
            )

        # 3. Parse ASTs with Tree-sitter
        lang_key = language.lower()
        if lang_key in ("ts", "tsx"):
            lang_key = "typescript"
        elif lang_key in ("js", "jsx"):
            lang_key = "javascript"

        try:
            tree_a = self.parser_manager.parse_source(code_a, lang_key)
            tree_b = self.parser_manager.parse_source(code_b, lang_key)
        except Exception:
            return self._fallback_compare(code_a, code_b)

        # 4. Extract canonical semantic tokens from AST
        tokens_a, docstrings_a = self._extract_semantic_tokens(tree_a.root_node, code_a.encode("utf-8"), lang_key)
        tokens_b, docstrings_b = self._extract_semantic_tokens(tree_b.root_node, code_b.encode("utf-8"), lang_key)

        buggy_count = len(tokens_a)
        fixed_count = len(tokens_b)

        # 5. Check if tokens match exactly when comments and docstrings are excluded
        if tokens_a == tokens_b:
            if docstrings_a != docstrings_b:
                change_type = "DOCSTRING_ONLY"
                details = "Changes were confined strictly to docstrings / module documentation."
            else:
                change_type = "COMMENT_ONLY"
                details = "Changes were confined strictly to code comments."

            return SemanticDiffResult(
                is_semantic=False,
                change_type=change_type,
                details=details,
                buggy_node_count=buggy_count,
                fixed_node_count=fixed_count,
            )

        # 6. Check if difference is purely type annotations (e.g. Python param/return annotations)
        tokens_no_type_a = self._strip_type_annotations(tokens_a)
        tokens_no_type_b = self._strip_type_annotations(tokens_b)
        if tokens_no_type_a == tokens_no_type_b:
            return SemanticDiffResult(
                is_semantic=False,
                change_type="TYPE_ANNOTATION_ONLY",
                details="Changes were confined strictly to type annotations.",
                buggy_node_count=buggy_count,
                fixed_node_count=fixed_count,
            )

        # 7. Substantive semantic change detected
        diff_summary = self._summarize_ast_diff(tokens_a, tokens_b)
        return SemanticDiffResult(
            is_semantic=True,
            change_type="SEMANTIC_CODE_CHANGE",
            details="Substantive AST structure or operand changes detected.",
            ast_diff_summary=diff_summary,
            buggy_node_count=buggy_count,
            fixed_node_count=fixed_count,
        )

    def _extract_semantic_tokens(
        self, root: Node, source_bytes: bytes, language: str
    ) -> Tuple[List[Tuple[str, str]], List[str]]:
        """Extract sequence of (node_type, text) tokens, skipping comments and docstrings."""
        tokens: List[Tuple[str, str]] = []
        docstring_texts: List[str] = []

        # Helper to identify Python docstrings (expression statement with a string at top of body)
        docstring_byte_spans: Set[Tuple[int, int]] = set()
        if language == "python":
            docstring_byte_spans = self._find_python_docstrings(root, source_bytes)
            for start, end in sorted(list(docstring_byte_spans)):
                docstring_texts.append(source_bytes[start:end].decode("utf-8", errors="replace").strip())

        def walk(node: Node) -> None:
            # Skip comments
            if node.type in self.NON_SEMANTIC_NODE_TYPES:
                return

            # Skip docstrings
            if (node.start_byte, node.end_byte) in docstring_byte_spans:
                return

            # If it's a leaf node (no children), record its type and text
            if node.child_count == 0:
                text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace").strip()
                if text:
                    tokens.append((node.type, text))
            else:
                tokens.append((node.type, ""))
                for child in node.children:
                    walk(child)

        walk(root)
        return tokens, docstring_texts

    def _find_python_docstrings(self, root: Node, source_bytes: bytes) -> Set[Tuple[int, int]]:
        """Identify (start_byte, end_byte) of docstrings in Python functions, classes, and modules."""
        spans: Set[Tuple[int, int]] = set()

        def scan_body(node: Node) -> None:
            if node.type in ("function_definition", "class_definition", "module"):
                # Find body / block
                for child in node.children:
                    if child.type == "block":
                        # First non-comment child
                        for stmt in child.children:
                            if stmt.type == "comment":
                                continue
                            if stmt.type == "expression_statement":
                                for expr in stmt.children:
                                    if expr.type == "string":
                                        spans.add((stmt.start_byte, stmt.end_byte))
                            break

            for child in node.children:
                scan_body(child)

        scan_body(root)
        return spans

    def _strip_type_annotations(self, tokens: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Filter out type annotation tokens to detect annotation-only changes."""
        filtered = []
        skip_depth = 0
        for ntype, text in tokens:
            if ntype in ("type", "type_annotation"):
                continue
            if text in ("->", ":"):
                # Potential annotation indicator
                continue
            filtered.append((ntype, text))
        return filtered

    def _summarize_ast_diff(
        self, tokens_a: List[Tuple[str, str]], tokens_b: List[Tuple[str, str]]
    ) -> str:
        """Produce a brief summary of AST node additions or modifications."""
        types_a = [t[0] for t in tokens_a]
        types_b = [t[0] for t in tokens_b]

        added = set(types_b) - set(types_a)
        removed = set(types_a) - set(types_b)

        parts = []
        if added:
            parts.append(f"Added AST constructs: {', '.join(sorted(list(added))[:4])}")
        if removed:
            parts.append(f"Removed AST constructs: {', '.join(sorted(list(removed))[:4])}")
        if not parts:
            parts.append(f"Operand or expression mutation (Token count: {len(tokens_a)} -> {len(tokens_b)})")

        return "; ".join(parts)

    def _fallback_compare(self, code_a: str, code_b: str) -> SemanticDiffResult:
        """Fallback normalized comparison when Tree-sitter grammar is not available."""
        # Strip python comments
        def strip_comments(text: str) -> str:
            lines = []
            for line in text.splitlines():
                clean = re.sub(r"#.*$", "", line).strip()
                if clean:
                    lines.append(clean)
            return "\n".join(lines)

        sa = strip_comments(code_a)
        sb = strip_comments(code_b)

        if sa == sb:
            return SemanticDiffResult(
                is_semantic=False,
                change_type="COMMENT_ONLY",
                details="Fallback: code differences were comment or formatting only.",
            )

        return SemanticDiffResult(
            is_semantic=True,
            change_type="SEMANTIC_CODE_CHANGE",
            details="Fallback: code differences contain semantic changes.",
        )
