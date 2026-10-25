"""Tree-sitter AST parser for extracting structured code units (functions, classes, methods)."""

import bisect
import logging
from typing import Any, Dict, List, Optional, Tuple

import tree_sitter_cpp
import tree_sitter_go
import tree_sitter_java
import tree_sitter_javascript
import tree_sitter_python
import tree_sitter_rust
import tree_sitter_typescript
from tree_sitter import Language, Node, Parser, Tree

from backend.models.schemas import CodeUnit, CodeUnitType

logger = logging.getLogger(__name__)


class ASTParsingError(Exception):
    """Exception raised when an error occurs during AST parsing."""
    pass


class LineIndex:
    """Fast, accurate byte-offset to 1-indexed line number resolver."""

    def __init__(self, source_bytes: bytes):
        self.line_starts = [0]
        pos = 0
        while True:
            idx = source_bytes.find(b"\n", pos)
            if idx == -1:
                break
            self.line_starts.append(idx + 1)
            pos = idx + 1

    def byte_to_line(self, byte_offset: int) -> int:
        idx = bisect.bisect_right(self.line_starts, byte_offset)
        return max(1, idx)


class ASTParserManager:
    """Manages Tree-sitter parsers and performs AST extraction across multiple languages."""

    def __init__(self):
        self._languages: Dict[str, Language] = {}
        self._parsers: Dict[str, Parser] = {}
        self._init_languages()

    def _init_languages(self) -> None:
        """Initialize all supported Tree-sitter language grammars."""
        try:
            self._languages["python"] = Language(tree_sitter_python.language())
            self._languages["javascript"] = Language(tree_sitter_javascript.language())
            self._languages["typescript"] = Language(tree_sitter_typescript.language_typescript())
            self._languages["tsx"] = Language(tree_sitter_typescript.language_tsx())
            self._languages["go"] = Language(tree_sitter_go.language())
            self._languages["java"] = Language(tree_sitter_java.language())
            self._languages["rust"] = Language(tree_sitter_rust.language())
            self._languages["cpp"] = Language(tree_sitter_cpp.language())

            for lang_name, lang_obj in self._languages.items():
                self._parsers[lang_name] = Parser(lang_obj)
        except Exception as e:
            logger.error("Failed to initialize Tree-sitter language grammars: %s", e)
            raise ASTParsingError(f"Failed to initialize Tree-sitter: {e}") from e

    def is_language_supported(self, language: str) -> bool:
        return language in self._parsers

    def parse_source(self, source_code: str, language: str) -> Tree:
        """Parse source code string into a Tree-sitter Tree."""
        if language not in self._parsers:
            raise ASTParsingError(f"Unsupported language: {language}")

        source_bytes = source_code.encode("utf-8")
        parser = self._parsers[language]
        return parser.parse(source_bytes)

    def extract_units(
        self,
        source_code: str,
        language: str,
        file_path: str,
        repository: str,
        commit_hash: str,
    ) -> List[CodeUnit]:
        """
        Parse source code and extract all structured functions, classes, and methods.

        Args:
            source_code: Raw source code string of the file.
            language: Programming language identifier.
            file_path: Relative path of the file in the repository.
            repository: Repository identifier or URL.
            commit_hash: Git commit hash.

        Returns:
            List of structured CodeUnit instances.
        """
        if language not in self._parsers:
            logger.warning("No parser available for language '%s' (file: %s)", language, file_path)
            return []

        source_bytes = source_code.encode("utf-8")
        try:
            tree = self.parse_source(source_code, language)
        except Exception as e:
            logger.error("AST parsing failed for %s (%s): %s", file_path, language, e)
            raise ASTParsingError(f"Error parsing {file_path}: {e}") from e

        line_index = LineIndex(source_bytes)
        units: List[CodeUnit] = []

        if language == "python":
            units = self._extract_python_units(tree.root_node, source_bytes, line_index, file_path, repository, commit_hash)
        elif language in ("javascript", "typescript", "tsx"):
            units = self._extract_js_ts_units(tree.root_node, source_bytes, line_index, language, file_path, repository, commit_hash)
        elif language == "go":
            units = self._extract_go_units(tree.root_node, source_bytes, line_index, file_path, repository, commit_hash)
        elif language == "java":
            units = self._extract_java_units(tree.root_node, source_bytes, line_index, file_path, repository, commit_hash)
        elif language == "rust":
            units = self._extract_rust_units(tree.root_node, source_bytes, line_index, file_path, repository, commit_hash)
        elif language == "cpp":
            units = self._extract_cpp_units(tree.root_node, source_bytes, line_index, file_path, repository, commit_hash)
        else:
            logger.warning("No extractor rule for supported language '%s'", language)

        return units

    # ==========================================
    # Helper methods for node extraction
    # ==========================================

    @staticmethod
    def _node_text(node: Optional[Node], source_bytes: bytes) -> str:
        if node is None:
            return ""
        return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    @staticmethod
    def _get_line_range(node: Node, line_index: LineIndex) -> Tuple[int, int]:
        start = line_index.byte_to_line(node.start_byte)
        end_offset = max(node.start_byte, node.end_byte - 1) if node.end_byte > node.start_byte else node.end_byte
        end = line_index.byte_to_line(end_offset)
        return start, max(start, end)

    # ==========================================
    # Python Extraction
    # ==========================================

    def _extract_python_units(
        self,
        root_node: Node,
        source_bytes: bytes,
        line_index: LineIndex,
        file_path: str,
        repository: str,
        commit_hash: str,
    ) -> List[CodeUnit]:
        units: List[CodeUnit] = []

        def extract_docstring(body_node: Optional[Node]) -> Optional[str]:
            if not body_node:
                return None
            for child in body_node.children:
                if child.type == "expression_statement":
                    for sub in child.children:
                        if sub.type == "string":
                            text = self._node_text(sub, source_bytes).strip()
                            for q in ['"""', "'''", '"', "'"]:
                                if text.startswith(q) and text.endswith(q) and len(text) >= 2 * len(q):
                                    return text[len(q):-len(q)].strip()
                            return text
                    break
                elif child.type not in ("comment", "pass_statement"):
                    break
            return None

        def extract_params(params_node: Optional[Node]) -> List[str]:
            params: List[str] = []
            if not params_node:
                return params
            for child in params_node.children:
                if child.type in ("identifier",):
                    params.append(self._node_text(child, source_bytes))
                elif child.type in ("default_parameter", "typed_parameter", "typed_default_parameter"):
                    name_child = child.child_by_field_name("name") or (child.children[0] if child.children else None)
                    if name_child:
                        params.append(self._node_text(name_child, source_bytes))
                elif child.type in ("list_splat_pattern", "dictionary_splat_pattern"):
                    params.append(self._node_text(child, source_bytes))
            return params

        def visit(node: Node, parent_class: Optional[str] = None, outer_node: Optional[Node] = None):
            effective_node = outer_node if outer_node else node

            if node.type == "decorated_definition":
                inner = node.child_by_field_name("definition")
                if inner:
                    visit(inner, parent_class=parent_class, outer_node=node)
                else:
                    # In case grammar represents definition as last child
                    for child in reversed(node.children):
                        if child.type in ("function_definition", "async_function_definition", "class_definition"):
                            visit(child, parent_class=parent_class, outer_node=node)
                            break
                return

            if node.type in ("function_definition", "async_function_definition"):
                name_node = node.child_by_field_name("name")
                name = self._node_text(name_node, source_bytes) if name_node else "<anonymous>"
                is_async = node.type == "async_function_definition" or any(c.type == "async" for c in node.children)
                
                body_node = node.child_by_field_name("body")
                params_node = node.child_by_field_name("parameters")
                docstring = extract_docstring(body_node)
                parameters = extract_params(params_node)

                start_line, end_line = self._get_line_range(effective_node, line_index)
                source = self._node_text(effective_node, source_bytes)

                if parent_class:
                    unit_type = CodeUnitType.METHOD
                    full_name = f"{parent_class}.{name}"
                elif is_async:
                    unit_type = CodeUnitType.ASYNC_FUNCTION
                    full_name = name
                else:
                    unit_type = CodeUnitType.FUNCTION
                    full_name = name

                unit = CodeUnit(
                    repository=repository,
                    commit_hash=commit_hash,
                    file_path=file_path,
                    unit_type=unit_type,
                    name=full_name,
                    start_line=start_line,
                    end_line=end_line,
                    source_code=source,
                    language="python",
                    parent_name=parent_class,
                    parameters=parameters,
                    docstring=docstring,
                    metadata={"is_async": is_async},
                )
                units.append(unit)

                if body_node:
                    for child in body_node.children:
                        if child.type in ("function_definition", "async_function_definition", "class_definition", "decorated_definition"):
                            visit(child, parent_class=None)

            elif node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                class_name = self._node_text(name_node, source_bytes) if name_node else "<anonymous>"
                body_node = node.child_by_field_name("body")
                docstring = extract_docstring(body_node)

                start_line, end_line = self._get_line_range(effective_node, line_index)
                source = self._node_text(effective_node, source_bytes)

                unit = CodeUnit(
                    repository=repository,
                    commit_hash=commit_hash,
                    file_path=file_path,
                    unit_type=CodeUnitType.CLASS,
                    name=class_name,
                    start_line=start_line,
                    end_line=end_line,
                    source_code=source,
                    language="python",
                    parent_name=parent_class,
                    parameters=[],
                    docstring=docstring,
                    metadata={},
                )
                units.append(unit)

                if body_node:
                    for child in body_node.children:
                        visit(child, parent_class=class_name)

            else:
                for child in node.children:
                    if child.type in ("function_definition", "async_function_definition", "class_definition", "decorated_definition"):
                        visit(child, parent_class=parent_class)

        for top_child in root_node.children:
            visit(top_child)

        return units

    # ==========================================
    # JavaScript & TypeScript Extraction
    # ==========================================

    def _extract_js_ts_units(
        self,
        root_node: Node,
        source_bytes: bytes,
        line_index: LineIndex,
        language: str,
        file_path: str,
        repository: str,
        commit_hash: str,
    ) -> List[CodeUnit]:
        units: List[CodeUnit] = []

        def extract_params(params_node: Optional[Node]) -> List[str]:
            params: List[str] = []
            if not params_node:
                return params
            for child in params_node.children:
                if child.type in ("identifier", "property_identifier"):
                    params.append(self._node_text(child, source_bytes))
                elif child.type in ("required_parameter", "optional_parameter", "assignment_pattern", "rest_pattern"):
                    name_sub = child.child_by_field_name("name") or (child.children[0] if child.children else None)
                    if name_sub:
                        params.append(self._node_text(name_sub, source_bytes))
            return params

        def visit(node: Node, parent_class: Optional[str] = None):
            if node.type in ("function_declaration", "generator_function_declaration"):
                name_node = node.child_by_field_name("name")
                name = self._node_text(name_node, source_bytes) if name_node else "<anonymous>"
                params_node = node.child_by_field_name("parameters")
                start_line, end_line = self._get_line_range(node, line_index)
                is_async = any(c.type == "async" for c in node.children)

                unit = CodeUnit(
                    repository=repository,
                    commit_hash=commit_hash,
                    file_path=file_path,
                    unit_type=CodeUnitType.ASYNC_FUNCTION if is_async else CodeUnitType.FUNCTION,
                    name=name,
                    start_line=start_line,
                    end_line=end_line,
                    source_code=self._node_text(node, source_bytes),
                    language=language,
                    parent_name=parent_class,
                    parameters=extract_params(params_node),
                    metadata={"is_async": is_async},
                )
                units.append(unit)

            elif node.type == "method_definition":
                name_node = node.child_by_field_name("name")
                name = self._node_text(name_node, source_bytes) if name_node else "<anonymous>"
                params_node = node.child_by_field_name("parameters")
                start_line, end_line = self._get_line_range(node, line_index)
                full_name = f"{parent_class}.{name}" if parent_class else name
                is_async = any(c.type == "async" for c in node.children)

                unit = CodeUnit(
                    repository=repository,
                    commit_hash=commit_hash,
                    file_path=file_path,
                    unit_type=CodeUnitType.METHOD,
                    name=full_name,
                    start_line=start_line,
                    end_line=end_line,
                    source_code=self._node_text(node, source_bytes),
                    language=language,
                    parent_name=parent_class,
                    parameters=extract_params(params_node),
                    metadata={"is_async": is_async},
                )
                units.append(unit)

            elif node.type in ("class_declaration", "class"):
                name_node = node.child_by_field_name("name")
                class_name = self._node_text(name_node, source_bytes) if name_node else "<anonymous>"
                body_node = node.child_by_field_name("body")
                start_line, end_line = self._get_line_range(node, line_index)

                unit = CodeUnit(
                    repository=repository,
                    commit_hash=commit_hash,
                    file_path=file_path,
                    unit_type=CodeUnitType.CLASS,
                    name=class_name,
                    start_line=start_line,
                    end_line=end_line,
                    source_code=self._node_text(node, source_bytes),
                    language=language,
                    parent_name=parent_class,
                    parameters=[],
                    metadata={},
                )
                units.append(unit)

                if body_node:
                    for child in body_node.children:
                        visit(child, parent_class=class_name)

            elif node.type == "lexical_declaration":
                for decl in node.children:
                    if decl.type == "variable_declarator":
                        name_node = decl.child_by_field_name("name")
                        value_node = decl.child_by_field_name("value")
                        if name_node and value_node and value_node.type in ("arrow_function", "function_expression"):
                            fn_name = self._node_text(name_node, source_bytes)
                            params_node = value_node.child_by_field_name("parameters")
                            start_line, end_line = self._get_line_range(node, line_index)
                            is_async = any(c.type == "async" for c in value_node.children)

                            unit = CodeUnit(
                                repository=repository,
                                commit_hash=commit_hash,
                                file_path=file_path,
                                unit_type=CodeUnitType.ASYNC_FUNCTION if is_async else CodeUnitType.FUNCTION,
                                name=fn_name,
                                start_line=start_line,
                                end_line=end_line,
                                source_code=self._node_text(node, source_bytes),
                                language=language,
                                parent_name=parent_class,
                                parameters=extract_params(params_node),
                                metadata={"is_async": is_async, "is_arrow": value_node.type == "arrow_function"},
                            )
                            units.append(unit)

            elif node.type in ("export_statement", "expression_statement", "program", "statement_block"):
                for child in node.children:
                    visit(child, parent_class=parent_class)

        for child in root_node.children:
            visit(child)

        return units

    # ==========================================
    # Go Extraction
    # ==========================================

    def _extract_go_units(
        self,
        root_node: Node,
        source_bytes: bytes,
        line_index: LineIndex,
        file_path: str,
        repository: str,
        commit_hash: str,
    ) -> List[CodeUnit]:
        units: List[CodeUnit] = []

        def extract_params(params_node: Optional[Node]) -> List[str]:
            params: List[str] = []
            if not params_node:
                return params
            for child in params_node.children:
                if child.type == "parameter_declaration":
                    name_child = child.child_by_field_name("name")
                    if name_child:
                        params.append(self._node_text(name_child, source_bytes))
            return params

        for node in root_node.children:
            if node.type == "function_declaration":
                name_node = node.child_by_field_name("name")
                name = self._node_text(name_node, source_bytes) if name_node else "<anonymous>"
                params_node = node.child_by_field_name("parameters")
                start_line, end_line = self._get_line_range(node, line_index)

                units.append(
                    CodeUnit(
                        repository=repository,
                        commit_hash=commit_hash,
                        file_path=file_path,
                        unit_type=CodeUnitType.FUNCTION,
                        name=name,
                        start_line=start_line,
                        end_line=end_line,
                        source_code=self._node_text(node, source_bytes),
                        language="go",
                        parameters=extract_params(params_node),
                    )
                )

            elif node.type == "method_declaration":
                receiver_node = node.child_by_field_name("receiver")
                name_node = node.child_by_field_name("name")
                name = self._node_text(name_node, source_bytes) if name_node else "<anonymous>"
                receiver_text = self._node_text(receiver_node, source_bytes) if receiver_node else ""
                receiver_clean = receiver_text.replace("(", "").replace(")", "").replace("*", "").split()[-1] if receiver_text else ""
                full_name = f"{receiver_clean}.{name}" if receiver_clean else name

                params_node = node.child_by_field_name("parameters")
                start_line, end_line = self._get_line_range(node, line_index)

                units.append(
                    CodeUnit(
                        repository=repository,
                        commit_hash=commit_hash,
                        file_path=file_path,
                        unit_type=CodeUnitType.METHOD,
                        name=full_name,
                        start_line=start_line,
                        end_line=end_line,
                        source_code=self._node_text(node, source_bytes),
                        language="go",
                        parent_name=receiver_clean or None,
                        parameters=extract_params(params_node),
                    )
                )

            elif node.type == "type_declaration":
                for spec in node.children:
                    if spec.type == "type_spec":
                        name_node = spec.child_by_field_name("name")
                        name = self._node_text(name_node, source_bytes) if name_node else "<anonymous>"
                        type_node = spec.child_by_field_name("type")
                        if type_node and type_node.type in ("struct_type", "interface_type"):
                            start_line, end_line = self._get_line_range(node, line_index)
                            units.append(
                                CodeUnit(
                                    repository=repository,
                                    commit_hash=commit_hash,
                                    file_path=file_path,
                                    unit_type=CodeUnitType.CLASS,
                                    name=name,
                                    start_line=start_line,
                                    end_line=end_line,
                                    source_code=self._node_text(node, source_bytes),
                                    language="go",
                                    metadata={"type_kind": type_node.type},
                                )
                            )

        return units

    # ==========================================
    # Java Extraction
    # ==========================================

    def _extract_java_units(
        self,
        root_node: Node,
        source_bytes: bytes,
        line_index: LineIndex,
        file_path: str,
        repository: str,
        commit_hash: str,
    ) -> List[CodeUnit]:
        units: List[CodeUnit] = []

        def visit(node: Node, parent_class: Optional[str] = None):
            if node.type in ("class_declaration", "interface_declaration", "enum_declaration", "record_declaration"):
                name_node = node.child_by_field_name("name")
                class_name = self._node_text(name_node, source_bytes) if name_node else "<anonymous>"
                body_node = node.child_by_field_name("body")
                start_line, end_line = self._get_line_range(node, line_index)

                units.append(
                    CodeUnit(
                        repository=repository,
                        commit_hash=commit_hash,
                        file_path=file_path,
                        unit_type=CodeUnitType.CLASS,
                        name=class_name,
                        start_line=start_line,
                        end_line=end_line,
                        source_code=self._node_text(node, source_bytes),
                        language="java",
                        parent_name=parent_class,
                    )
                )

                if body_node:
                    for child in body_node.children:
                        visit(child, parent_class=class_name)

            elif node.type in ("method_declaration", "constructor_declaration"):
                name_node = node.child_by_field_name("name")
                name = self._node_text(name_node, source_bytes) if name_node else "<anonymous>"
                params_node = node.child_by_field_name("parameters")
                start_line, end_line = self._get_line_range(node, line_index)
                full_name = f"{parent_class}.{name}" if parent_class else name

                params: List[str] = []
                if params_node:
                    for child in params_node.children:
                        if child.type == "formal_parameter":
                            pname = child.child_by_field_name("name")
                            if pname:
                                params.append(self._node_text(pname, source_bytes))

                units.append(
                    CodeUnit(
                        repository=repository,
                        commit_hash=commit_hash,
                        file_path=file_path,
                        unit_type=CodeUnitType.METHOD if parent_class else CodeUnitType.FUNCTION,
                        name=full_name,
                        start_line=start_line,
                        end_line=end_line,
                        source_code=self._node_text(node, source_bytes),
                        language="java",
                        parent_name=parent_class,
                        parameters=params,
                    )
                )

            else:
                for child in node.children:
                    visit(child, parent_class=parent_class)

        for child in root_node.children:
            visit(child)

        return units

    # ==========================================
    # Rust Extraction
    # ==========================================

    def _extract_rust_units(
        self,
        root_node: Node,
        source_bytes: bytes,
        line_index: LineIndex,
        file_path: str,
        repository: str,
        commit_hash: str,
    ) -> List[CodeUnit]:
        units: List[CodeUnit] = []

        def extract_params(params_node: Optional[Node]) -> List[str]:
            params: List[str] = []
            if not params_node:
                return params
            for child in params_node.children:
                if child.type in ("parameter", "self_parameter"):
                    pattern = child.child_by_field_name("pattern") or child
                    params.append(self._node_text(pattern, source_bytes))
            return params

        def visit(node: Node, parent_impl: Optional[str] = None):
            if node.type == "function_item":
                name_node = node.child_by_field_name("name")
                name = self._node_text(name_node, source_bytes) if name_node else "<anonymous>"
                params_node = node.child_by_field_name("parameters")
                start_line, end_line = self._get_line_range(node, line_index)
                full_name = f"{parent_impl}.{name}" if parent_impl else name

                units.append(
                    CodeUnit(
                        repository=repository,
                        commit_hash=commit_hash,
                        file_path=file_path,
                        unit_type=CodeUnitType.METHOD if parent_impl else CodeUnitType.FUNCTION,
                        name=full_name,
                        start_line=start_line,
                        end_line=end_line,
                        source_code=self._node_text(node, source_bytes),
                        language="rust",
                        parent_name=parent_impl,
                        parameters=extract_params(params_node),
                    )
                )

            elif node.type in ("struct_item", "enum_item", "trait_item"):
                name_node = node.child_by_field_name("name")
                name = self._node_text(name_node, source_bytes) if name_node else "<anonymous>"
                start_line, end_line = self._get_line_range(node, line_index)

                units.append(
                    CodeUnit(
                        repository=repository,
                        commit_hash=commit_hash,
                        file_path=file_path,
                        unit_type=CodeUnitType.CLASS,
                        name=name,
                        start_line=start_line,
                        end_line=end_line,
                        source_code=self._node_text(node, source_bytes),
                        language="rust",
                        metadata={"kind": node.type},
                    )
                )

            elif node.type == "impl_item":
                type_node = node.child_by_field_name("type")
                impl_name = self._node_text(type_node, source_bytes) if type_node else "<impl>"
                body_node = node.child_by_field_name("body")
                if body_node:
                    for child in body_node.children:
                        visit(child, parent_impl=impl_name)

            else:
                for child in node.children:
                    visit(child, parent_impl=parent_impl)

        for child in root_node.children:
            visit(child)

        return units

    # ==========================================
    # C / C++ Extraction
    # ==========================================

    def _extract_cpp_units(
        self,
        root_node: Node,
        source_bytes: bytes,
        line_index: LineIndex,
        file_path: str,
        repository: str,
        commit_hash: str,
    ) -> List[CodeUnit]:
        units: List[CodeUnit] = []

        def visit(node: Node, parent_class: Optional[str] = None):
            if node.type == "function_definition":
                declarator = node.child_by_field_name("declarator")
                name = ""
                if declarator:
                    fn_decl = declarator
                    while fn_decl and fn_decl.type not in ("identifier", "field_identifier", "qualified_identifier", "destructor_name"):
                        nested_name = fn_decl.child_by_field_name("declarator")
                        if nested_name:
                            fn_decl = nested_name
                        else:
                            for c in fn_decl.children:
                                if c.type in ("identifier", "field_identifier", "qualified_identifier", "destructor_name"):
                                    fn_decl = c
                                    break
                            else:
                                break
                    name = self._node_text(fn_decl, source_bytes)

                if not name:
                    name = "<anonymous>"

                start_line, end_line = self._get_line_range(node, line_index)
                full_name = f"{parent_class}::{name}" if parent_class and "::" not in name else name

                units.append(
                    CodeUnit(
                        repository=repository,
                        commit_hash=commit_hash,
                        file_path=file_path,
                        unit_type=CodeUnitType.METHOD if parent_class else CodeUnitType.FUNCTION,
                        name=full_name,
                        start_line=start_line,
                        end_line=end_line,
                        source_code=self._node_text(node, source_bytes),
                        language="cpp",
                        parent_name=parent_class,
                    )
                )

            elif node.type in ("class_specifier", "struct_specifier"):
                name_node = node.child_by_field_name("name")
                class_name = self._node_text(name_node, source_bytes) if name_node else "<anonymous>"
                body_node = node.child_by_field_name("body")
                start_line, end_line = self._get_line_range(node, line_index)

                units.append(
                    CodeUnit(
                        repository=repository,
                        commit_hash=commit_hash,
                        file_path=file_path,
                        unit_type=CodeUnitType.CLASS,
                        name=class_name,
                        start_line=start_line,
                        end_line=end_line,
                        source_code=self._node_text(node, source_bytes),
                        language="cpp",
                        parent_name=parent_class,
                    )
                )

                if body_node:
                    for child in body_node.children:
                        visit(child, parent_class=class_name)

            else:
                for child in node.children:
                    visit(child, parent_class=parent_class)

        for child in root_node.children:
            visit(child)

        return units
