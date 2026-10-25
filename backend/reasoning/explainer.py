"""Reasoning and Explainability Engine for CodeGhost (Milestone 5).

Translates risk probabilities, AST complexity features, and retrieved historical
bug-fix pairs into comprehensive, human-interpretable risk explanations.
"""

import os
import re
from typing import Any, Dict, List, Optional

from backend.reasoning.schemas import (
    ComplexityFactor,
    HistoricalPrecedent,
    RiskExplanation,
)
from backend.risk.schemas import RiskCategory, RiskPrediction


class ReasoningEngine:
    """Core explanation engine synthesizing prediction metrics, AST features, and historical bug context."""

    def __init__(self, provider: str = "deterministic", api_key: Optional[str] = None):
        """Initialize reasoning engine.
        
        Args:
            provider: 'deterministic' (default, local AST/pattern synthesis),
                      'gemini' (Google Gemini API), or 'openai'.
            api_key: Optional API key for remote providers.
        """
        self.provider = provider.lower()
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def explain(
        self,
        prediction: RiskPrediction,
        code: str,
        file_path: Optional[str] = None,
        language: str = "python",
    ) -> RiskExplanation:
        """Generate a complete RiskExplanation for a function prediction.

        Args:
            prediction: Evaluated RiskPrediction from Milestone 4.
            code: Source code string of the function.
            file_path: Relative file path.
            language: Programming language.

        Returns:
            RiskExplanation model.
        """
        target_file = file_path or prediction.file_path

        # If remote provider requested and key available, attempt remote generation with fallback
        if self.provider in ("gemini", "google") and self.api_key:
            try:
                return self._explain_with_gemini(prediction, code, target_file, language)
            except Exception:
                # Graceful fallback to deterministic engine on network/auth issue
                pass

        return self._explain_deterministic(prediction, code, target_file, language)

    def _explain_deterministic(
        self,
        prediction: RiskPrediction,
        code: str,
        file_path: str,
        language: str,
    ) -> RiskExplanation:
        """Deterministic, grounded AST and rule-based explanation generator."""
        features = prediction.features_used or {}
        complexity_factors = self._extract_complexity_factors(code, features)
        historical_precedents = self._extract_historical_precedents(prediction, code)
        vulnerable_patterns = self._identify_vulnerable_patterns(code, language, historical_precedents)
        mitigation_strategies = self._formulate_mitigations(
            prediction.risk_category,
            complexity_factors,
            vulnerable_patterns,
            historical_precedents,
        )

        # Build summary
        cat_str = getattr(prediction.risk_category, "value", str(prediction.risk_category))
        if cat_str in (RiskCategory.HIGH.value, RiskCategory.CRITICAL.value, "HIGH", "CRITICAL"):
            summary = (
                f"Function '{prediction.function_name}' has been assessed as {cat_str} "
                f"risk (score {prediction.risk_score:.1f}/10.0, probability {prediction.risk_probability:.2%}). "
                f"It exhibits structural complexity coupled with strong alignment to "
                f"{len(historical_precedents)} past bug-fix event(s) in the repository."
            )
        elif cat_str in (RiskCategory.MEDIUM.value, "MEDIUM"):
            summary = (
                f"Function '{prediction.function_name}' exhibits moderate risk "
                f"(score {prediction.risk_score:.1f}/10.0). While not directly matching severe historical outages, "
                f"internal branch complexity and edge-case handling warrant review."
            )
        else:
            summary = (
                f"Function '{prediction.function_name}' is classified as LOW risk "
                f"(score {prediction.risk_score:.1f}/10.0). Code structure is concise and exhibits low similarity "
                f"to historical defect patterns."
            )

        # In-depth analysis narrative
        detailed_analysis = self._build_detailed_analysis(
            prediction,
            code,
            complexity_factors,
            historical_precedents,
            vulnerable_patterns,
        )

        return RiskExplanation(
            function_name=prediction.function_name,
            file_path=file_path,
            risk_score=prediction.risk_score,
            risk_category=prediction.risk_category,
            is_risky=prediction.is_risky,
            summary=summary,
            detailed_analysis=detailed_analysis,
            complexity_factors=complexity_factors,
            historical_precedents=historical_precedents,
            vulnerable_patterns=vulnerable_patterns,
            mitigation_strategies=mitigation_strategies,
            provider="deterministic",
            model_name=f"CodeGhost-DeterministicExplainer (v1.0; base={prediction.model_name})",
            metadata={
                "features_count": len(features),
                "historical_matches_count": len(prediction.top_retrieved_matches),
            },
        )

    def _extract_complexity_factors(
        self,
        code: str,
        features: Dict[str, float],
    ) -> List[ComplexityFactor]:
        """Extract and categorize AST structural complexity drivers."""
        factors: List[ComplexityFactor] = []
        loc = features.get("lines_of_code", len(code.splitlines()))
        cyclomatic = features.get("cyclomatic_complexity", 1.0)
        nesting = features.get("max_nesting_depth", 0.0)
        has_exceptions = features.get("has_exception_handling", 0.0)
        branches = features.get("num_branches", 0.0)

        if cyclomatic >= 8.0:
            factors.append(
                ComplexityFactor(
                    name="Cyclomatic Complexity",
                    value=int(cyclomatic),
                    risk_impact="HIGH",
                    description=(
                        f"High decision branch count ({int(cyclomatic)} paths). Multiple nested paths increase "
                        f"the likelihood of untested edge cases and state regressions."
                    ),
                )
            )
        elif cyclomatic >= 4.0:
            factors.append(
                ComplexityFactor(
                    name="Cyclomatic Complexity",
                    value=int(cyclomatic),
                    risk_impact="MEDIUM",
                    description=f"Moderate branch count ({int(cyclomatic)} paths) requiring targeted branch testing.",
                )
            )

        if nesting >= 3.0:
            factors.append(
                ComplexityFactor(
                    name="Nesting Depth",
                    value=int(nesting),
                    risk_impact="HIGH",
                    description=(
                        f"Deeply nested control structures ({int(nesting)} levels). High cognitive complexity "
                        f"correlates with boundary condition miscalculations."
                    ),
                )
            )

        if loc >= 40:
            factors.append(
                ComplexityFactor(
                    name="Function Length",
                    value=int(loc),
                    risk_impact="MEDIUM",
                    description=f"Long function body ({int(loc)} lines of code), suggesting potential violations of single responsibility.",
                )
            )

        if has_exceptions == 0.0 and (branches >= 2.0 or "open(" in code or "http" in code or ".get(" in code):
            factors.append(
                ComplexityFactor(
                    name="Missing Error Handling",
                    value="None detected",
                    risk_impact="HIGH" if branches >= 4.0 else "MEDIUM",
                    description="Function performs complex operations or I/O without try/except or defensive fallback guards.",
                )
            )

        return factors

    def _extract_historical_precedents(
        self,
        prediction: RiskPrediction,
        code: str,
    ) -> List[HistoricalPrecedent]:
        """Convert retrieved vector matches into contextual historical precedents."""
        precedents: List[HistoricalPrecedent] = []

        for match in prediction.top_retrieved_matches:
            # Handle both direct fields and dictionary payload
            commit_sha = getattr(match, "commit_sha", None) or (match.payload or {}).get("commit_sha", "unknown")
            commit_msg = getattr(match, "commit_message", None) or (match.payload or {}).get("commit_message", "Historical bug fix")
            file_path = getattr(match, "file_path", None) or (match.payload or {}).get("file_path", "unknown")
            func_name = getattr(match, "function_name", None) or (match.payload or {}).get("function_name", "unknown")
            buggy_code = getattr(match, "buggy_code", None) or (match.payload or {}).get("buggy_code", "")
            fixed_code = getattr(match, "fixed_code", None) or (match.payload or {}).get("fixed_code", "")
            diff = getattr(match, "diff", None) or (match.payload or {}).get("diff", "")
            sim = getattr(match, "similarity_score", None) or getattr(match, "similarity", 0.0)

            # Determine relevance reason
            reasons = []
            if sim >= 0.90:
                reasons.append(f"Very high structural/semantic similarity ({sim:.2%})")
            elif sim >= 0.75:
                reasons.append(f"Moderate semantic similarity ({sim:.2%})")

            if func_name == prediction.function_name:
                reasons.append("Identical function lineage historically modified for bug resolution")
            elif func_name.split(".")[-1] == prediction.function_name.split(".")[-1]:
                reasons.append("Matching method/action name in repository")

            # Check shared keywords or operations
            if ".partition(" in code and ".partition(" in buggy_code:
                reasons.append("Shared string partitioning operation on address/URI")
            elif ".split(" in code and ".split(" in buggy_code:
                reasons.append("Shared string splitting logic without delimiter validation")

            relevance = "; ".join(reasons) if reasons else f"Semantic vector alignment ({sim:.2%})"

            precedents.append(
                HistoricalPrecedent(
                    commit_sha=commit_sha,
                    similarity_score=round(sim, 4),
                    commit_message=commit_msg,
                    file_path=file_path,
                    function_name=func_name,
                    buggy_snippet=(buggy_code[:200] + "...") if len(buggy_code) > 200 else buggy_code,
                    fixed_snippet=(fixed_code[:200] + "...") if len(fixed_code) > 200 else fixed_code,
                    diff_summary=(diff[:300] + "...") if len(diff) > 300 else diff,
                    relevance_reason=relevance,
                )
            )

        return precedents

    def _identify_vulnerable_patterns(
        self,
        code: str,
        language: str,
        precedents: List[HistoricalPrecedent],
    ) -> List[str]:
        """Detect recurring software bug patterns and structural hazards."""
        patterns: List[str] = []

        # 1. String partitioning / splitting without bracket or format checks
        if re.search(r'\.(partition|split)\s*\(\s*["\']:', code):
            patterns.append(
                "Host/Port delimiter split: using single colon delimiter without bracketed IPv6 notation handling "
                "(e.g., '[::1]:5000' vs '127.0.0.1:5000')."
            )

        # 2. Direct dictionary or map lookup
        if re.search(r'\[\s*["\'][a-zA-Z0-9_-]+["\']\s*\]', code) and "try:" not in code:
            patterns.append(
                "Direct dictionary indexing without safe lookup (.get) or KeyError containment."
            )

        # 3. Environment or global state mutation
        if "os.environ" in code and ("=" in code or "setdefault" in code):
            patterns.append(
                "Process environment modification: modifying os.environ directly may cause thread-safety races or leak configuration."
            )

        # 4. Unvalidated type conversion
        if re.search(r'int\s*\(\s*[a-zA-Z0-9_.]+\s*\)', code) and "except" not in code:
            patterns.append(
                "Unchecked integer casting: int() without ValueError exception guard on external string input."
            )

        # 5. Nil / None attribute access
        if re.search(r'[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+\(', code) and "None" in code and "if " not in code:
            patterns.append(
                "Potential NoneType dereference before conditional validation."
            )

        # If precedents point to a known commit message pattern
        for prec in precedents:
            lower_msg = prec.commit_message.lower()
            if "ipv6" in lower_msg and "ipv6" not in [p.lower() for p in patterns]:
                patterns.append("Historical precedence: vulnerability to IPv6 address parsing regression.")
            if "reloader" in lower_msg or "socket" in lower_msg:
                patterns.append("Socket/network binding: race conditions or port reuse issues under reloader restarts.")

        if not patterns:
            patterns.append("General branch divergence: untested path combinations under atypical argument types.")

        return patterns

    def _formulate_mitigations(
        self,
        category: RiskCategory,
        complexity_factors: List[ComplexityFactor],
        vulnerable_patterns: List[str],
        precedents: List[HistoricalPrecedent],
    ) -> List[str]:
        """Generate concrete, actionable developer recommendations."""
        mitigations: List[str] = []

        for pat in vulnerable_patterns:
            if "IPv6" in pat or "colon" in pat:
                mitigations.append(
                    "Use urllib.parse.urlsplit or check for '[' and ']' bracket enclosures before partitioning by ':'."
                )
            elif "dictionary" in pat:
                mitigations.append("Replace direct subscripting with dict.get(key, default) or enclose in a KeyError block.")
            elif "integer casting" in pat:
                mitigations.append("Wrap int(...) conversions in a try/except ValueError block with a descriptive fallback.")

        for factor in complexity_factors:
            if factor.name == "Missing Error Handling":
                mitigations.append("Enclose dynamic calls and resource bindings in structured try/except blocks.")
            elif factor.name == "Nesting Depth":
                mitigations.append("Refactor deep conditionals using guard clauses (early returns) to flatten nesting hierarchy.")

        if precedents:
            top_sha = precedents[0].commit_sha[:8]
            mitigations.append(f"Review historical fix commit {top_sha} for canonical repository conventions on resolving this failure.")

        if not mitigations:
            mitigations.append("Add unit tests covering edge-case arguments and null/empty parameters.")

        return mitigations

    def _build_detailed_analysis(
        self,
        prediction: RiskPrediction,
        code: str,
        complexity_factors: List[ComplexityFactor],
        precedents: List[HistoricalPrecedent],
        vulnerable_patterns: List[str],
    ) -> str:
        """Construct a coherent multi-paragraph technical narrative."""
        sections = []

        # Paragraph 1: Quantitative baseline & prediction
        cat_str = getattr(prediction.risk_category, "value", str(prediction.risk_category))
        sections.append(
            f"CodeGhost evaluated function '{prediction.function_name}' using model '{prediction.model_name}'. "
            f"The evaluation yielded a risk probability of {prediction.risk_probability:.2%}, corresponding "
            f"to a calibrated risk score of {prediction.risk_score:.1f}/10.0 ({cat_str})."
        )

        # Paragraph 2: Complexity inspection
        if complexity_factors:
            factor_strs = [f"{f.name} ({f.value}): {f.description}" for f in complexity_factors]
            sections.append(
                f"Static AST inspection identified structural complexity drivers: " + " ".join(factor_strs)
            )
        else:
            sections.append(
                "Static AST inspection reveals clean code structure with standard modularity and linear control flow."
            )

        # Paragraph 3: Historical correlation
        if precedents:
            top = precedents[0]
            sections.append(
                f"Historical retrieval identified relevant precedents from the repository's git commit log. "
                f"Most notably, commit {top.commit_sha[:8]} ('{top.commit_message}') resolved a defect with "
                f"{top.similarity_score:.1%} structural similarity. {top.relevance_reason}."
            )
        else:
            sections.append(
                "No strong historical defect precedents were retrieved from repository history, indicating this code "
                "pattern has not historically suffered frequent regressions."
            )

        # Paragraph 4: Failure mechanics
        if vulnerable_patterns:
            sections.append(
                "Potential failure mechanics identified: " + " ".join(vulnerable_patterns)
            )

        return "\n\n".join(sections)

    def _explain_with_gemini(
        self,
        prediction: RiskPrediction,
        code: str,
        file_path: str,
        language: str,
    ) -> RiskExplanation:
        """Remote LLM reasoning synthesis using Google Gemini API."""
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            prompt = self._build_llm_prompt(prediction, code, file_path, language)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            # Parse response or fall back if empty
            if response.text:
                return self._parse_llm_response(response.text, prediction, file_path)
        except Exception:
            pass
        return self._explain_deterministic(prediction, code, file_path, language)

    def _build_llm_prompt(
        self,
        prediction: RiskPrediction,
        code: str,
        file_path: str,
        language: str,
    ) -> str:
        """Construct structured prompt for LLM explanation generation."""
        precedents_text = ""
        for idx, m in enumerate(prediction.top_retrieved_matches[:3], 1):
            p = m.payload or {}
            precedents_text += (
                f"\nHistorical Fix #{idx}:\n"
                f"- Commit: {p.get('commit_sha', 'unknown')[:8]}\n"
                f"- Message: {p.get('commit_message', '')}\n"
                f"- Similarity: {m.similarity:.2%}\n"
                f"- Buggy snippet: {p.get('buggy_code', '')[:150]}\n"
                f"- Fixed snippet: {p.get('fixed_code', '')[:150]}\n"
            )

        return f"""You are CodeGhost, an expert static analysis and software reliability AI.
Analyze the following function and explain why it is risky based on its historical bug similarity and AST features.

Function Name: {prediction.function_name}
File Path: {file_path}
Language: {language}
Assessed Risk Score: {prediction.risk_score}/10 ({prediction.risk_category.value})
Risk Probability: {prediction.risk_probability:.2%}

Source Code:
```
{code}
```

Historical Bug Precedents:
{precedents_text}

Provide your response in JSON format with the following fields:
{{
  "summary": "Brief 1-2 sentence executive summary",
  "detailed_analysis": "Comprehensive technical narrative of the risk",
  "vulnerable_patterns": ["List of specific code hazards or anti-patterns"],
  "mitigation_strategies": ["List of concrete code fixes or mitigations"]
}}
"""

    def _parse_llm_response(
        self,
        response_text: str,
        prediction: RiskPrediction,
        file_path: str,
    ) -> RiskExplanation:
        """Parse and validate JSON response from Gemini into RiskExplanation."""
        import json
        # Extract json block if surrounded by markdown fences
        clean_text = response_text.strip()
        if clean_text.startswith("```"):
            clean_text = re.sub(r"^```(?:json)?\n?", "", clean_text)
            clean_text = re.sub(r"\n?```$", "", clean_text)

        data = json.loads(clean_text)
        deterministic_fallback = self._explain_deterministic(prediction, "", file_path, "python")

        return RiskExplanation(
            function_name=prediction.function_name,
            file_path=file_path,
            risk_score=prediction.risk_score,
            risk_category=prediction.risk_category,
            is_risky=prediction.is_risky,
            summary=data.get("summary", deterministic_fallback.summary),
            detailed_analysis=data.get("detailed_analysis", deterministic_fallback.detailed_analysis),
            complexity_factors=deterministic_fallback.complexity_factors,
            historical_precedents=deterministic_fallback.historical_precedents,
            vulnerable_patterns=data.get("vulnerable_patterns", deterministic_fallback.vulnerable_patterns),
            mitigation_strategies=data.get("mitigation_strategies", deterministic_fallback.mitigation_strategies),
            provider="gemini",
            model_name="gemini-2.5-flash",
            metadata={"llm_generated": True},
        )
