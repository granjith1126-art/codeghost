"""LLM-based Risk Scorer using Qwen2.5-Coder (Milestone 4).

Constructs structured prompts with retrieved historical bug-fix context,
queries the model for risk evaluation, and strictly parses/validates JSON output.
Includes MockLLMScorer for offline, non-GPU, and fast unit test execution.
"""

from abc import ABC, abstractmethod
import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ValidationError

from backend.retrieval.schemas import RetrievedMatch
from backend.risk.baseline_model import map_probability_to_score
from backend.risk.schemas import RiskCategory, RiskExample, RiskPrediction, RiskThresholds

logger = logging.getLogger(__name__)

DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"


class LLMStructuredOutput(BaseModel):
    """Strict schema for LLM structured risk evaluation output."""
    risk_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted probability that the function has a bug")
    risk_score: float = Field(..., ge=1.0, le=10.0, description="Numerical risk score from 1.0 to 10.0")
    classification: str = Field(..., description="Categorical risk tier: LOW, MEDIUM, HIGH, or CRITICAL")
    reasoning_summary: Optional[str] = Field(None, description="Concise rationale explaining the score based on historical context")


class BaseLLMScorer(ABC):
    """Abstract base class for LLM risk scorers."""

    def __init__(self, thresholds: Optional[RiskThresholds] = None):
        self.thresholds = thresholds or RiskThresholds()

    def build_prompt(
        self,
        code: str,
        function_name: str,
        file_path: str,
        retrieved_matches: Optional[List[RetrievedMatch]] = None,
        language: str = "python",
    ) -> str:
        """Construct the structured prompt injecting target function and retrieved bug history."""
        retrieved_matches = retrieved_matches or []
        prompt_parts = [
            "You are CodeGhost Risk Evaluator. Your task is to analyze the target function and its historically similar bug-fix examples from the repository to predict if the current function is at high risk of bugs.",
            "",
            "CRITICAL: Similarity to a historical bug is evidence, but NOT proof of a bug. Evaluate whether the current code genuinely replicates the buggy pattern or contains structural vulnerabilities.",
            "",
            f"TARGET FUNCTION ({language}):",
            f"File: {file_path}",
            f"Function: {function_name}",
            "```" + language,
            code,
            "```",
            "",
            "HISTORICAL BUG-FIX EXAMPLES RETRIEVED FROM REPOSITORY:",
        ]

        if not retrieved_matches:
            prompt_parts.append("No historical bug-fix matches found for this function.")
        else:
            for idx, match in enumerate(retrieved_matches[:3], 1):
                prompt_parts.extend([
                    f"--- HISTORICAL EXAMPLE {idx} (Similarity: {match.similarity_score:.4f}, Commit: {match.commit_sha[:8]}) ---",
                    f"File: {match.file_path} | Function: {match.function_name}",
                    f"Commit Message: {match.commit_message}",
                    "BUGGY CODE BEFORE FIX:",
                    "```" + language,
                    match.buggy_code[:1000],  # Truncate to avoid context window explosion
                    "```",
                    "FIXED CODE AFTER FIX:",
                    "```" + language,
                    match.fixed_code[:1000],
                    "```",
                    f"DIFF:\n{match.diff[:500]}",
                    "",
                ])

        prompt_parts.extend([
            "OUTPUT INSTRUCTION:",
            "Respond ONLY with a valid JSON object matching this schema. Do not include markdown codeblocks or preamble.",
            "{",
            '  "risk_probability": <float between 0.0 and 1.0>,',
            '  "risk_score": <float between 1.0 and 10.0>,',
            '  "classification": "<LOW | MEDIUM | HIGH | CRITICAL>",',
            '  "reasoning_summary": "<one or two sentences explaining why the function is or is not risky>"',
            "}",
        ])

        return "\n".join(prompt_parts)

    def parse_and_validate_response(self, raw_response: str) -> LLMStructuredOutput:
        """
        Extract and validate JSON from model output.
        Guarantees that arbitrary natural language is never accepted as a numeric prediction.
        """
        cleaned = raw_response.strip()
        # Strip markdown fences if present
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Find first JSON object using regex if surrounded by other text
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Model output is not valid JSON: {raw_response[:200]}") from e

        # Validate against Pydantic schema
        try:
            parsed = LLMStructuredOutput.model_validate(data)
            return parsed
        except ValidationError as e:
            raise ValueError(f"Model output does not match expected risk schema: {e}") from e

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Generate raw text response from the model."""
        pass

    def predict(
        self,
        code: str,
        function_name: str,
        file_path: str,
        language: str = "python",
        retrieved_matches: Optional[List[RetrievedMatch]] = None,
        model_name: str = "Qwen2.5-Coder",
    ) -> RiskPrediction:
        """Generate structured risk prediction for a function."""
        retrieved_matches = retrieved_matches or []
        prompt = self.build_prompt(
            code=code,
            function_name=function_name,
            file_path=file_path,
            retrieved_matches=retrieved_matches,
            language=language,
        )

        raw_output = self.generate_response(prompt)
        structured = self.parse_and_validate_response(raw_output)

        # Categorize and clamp
        score = map_probability_to_score(structured.risk_probability)
        category = self.thresholds.categorize(score)
        is_risky = structured.risk_probability >= 0.5

        sims = [m.similarity_score for m in retrieved_matches]

        return RiskPrediction(
            function_name=function_name,
            file_path=file_path,
            risk_probability=round(structured.risk_probability, 4),
            risk_score=score,
            risk_category=category,
            is_risky=is_risky,
            model_name=model_name,
            features_used={"top1_similarity": sims[0] if sims else 0.0},
            top_retrieved_matches=retrieved_matches[:3],
            explanation_context={
                "similarity_scores": sims[:3],
                "reasoning_summary": structured.reasoning_summary,
                "raw_response": raw_output[:300],
            },
            reasoning_summary=structured.reasoning_summary,
        )


class MockLLMScorer(BaseLLMScorer):
    """
    Deterministic mock LLM scorer for unit tests and local CPU environments.
    Simulates Qwen2.5-Coder reasoning based on prompt contents and retrieval signals.
    """

    def __init__(self, thresholds: Optional[RiskThresholds] = None):
        super().__init__(thresholds=thresholds)

    def generate_response(self, prompt: str) -> str:
        """Simulate realistic model JSON generation based on prompt analysis."""
        has_high_sim = False
        # Check for high similarity in prompt
        sim_matches = re.findall(r"Similarity:\s*([0-9\.]+)", prompt)
        if sim_matches:
            top_sim = float(sim_matches[0])
            if top_sim >= 0.85:
                has_high_sim = True

        if has_high_sim:
            output = {
                "risk_probability": 0.82,
                "risk_score": 8.4,
                "classification": "HIGH",
                "reasoning_summary": "Function closely matches a historical bug fix pattern without error handling for edge cases.",
            }
        else:
            output = {
                "risk_probability": 0.18,
                "risk_score": 2.6,
                "classification": "LOW",
                "reasoning_summary": "No strong historical bug matches found; function structure exhibits standard control flow.",
            }
        return json.dumps(output)


class QwenLLMScorer(BaseLLMScorer):
    """
    Production Qwen2.5-Coder scorer using Hugging Face Transformers and PEFT.
    Supports 4-bit quantization and LoRA adapter weights.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_QWEN_MODEL_ID,
        adapter_path: Optional[str] = None,
        device: str = "auto",
        load_in_4bit: bool = True,
        thresholds: Optional[RiskThresholds] = None,
    ):
        super().__init__(thresholds=thresholds)
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        self._is_loaded = False

    def load(self) -> None:
        """Lazy loader for tokenizer and model weights."""
        if self._is_loaded:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as e:
            raise ImportError(
                "transformers, torch, and bitsandbytes are required for QwenLLMScorer. "
                "Install them or use MockLLMScorer for offline/CPU testing."
            ) from e

        logger.info("Loading Qwen tokenizer from '%s'...", self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

        quant_config = None
        if self.load_in_4bit and torch.cuda.is_available():
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        logger.info("Loading Qwen model from '%s'...", self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quant_config,
            device_map=self.device,
            trust_remote_code=True,
        )

        if self.adapter_path:
            from peft import PeftModel
            logger.info("Loading LoRA adapter from '%s'...", self.adapter_path)
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

        self._is_loaded = True

    def generate_response(self, prompt: str) -> str:
        """Run token generation through Qwen2.5-Coder."""
        if not self._is_loaded:
            self.load()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response_tokens = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(response_tokens, skip_special_tokens=True)
