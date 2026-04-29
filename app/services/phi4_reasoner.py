"""Phi-4 reasoning layer - structured reasoning over fused claim data via Ollama."""

import json
import re
from typing import Any, Dict, Optional

import httpx

from app.config import get_settings
from app.prompts.phi4_claim_reasoning_prompt import PHI4_SYSTEM_PROMPT


class Phi4Reasoner:
    """Handles structured reasoning using Phi-4 via Ollama."""

    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.ollama_base_url
        self.model = self.settings.ollama_model

    def reason(self, fusion_result: dict) -> dict:
        """
        Use Phi-4 to reason over the fused claim data.

        Phi-4 receives:
        - normalized_claim_input
        - missing_fields
        - conflicts
        - source_map

        Phi-4 should NOT:
        - Make final approval decisions
        - Output fake confidence scores
        - Guess missing information
        - Accuse of fraud definitively

        Args:
            fusion_result: Result from the fusion layer

        Returns:
            Dictionary containing:
            - model: The model used
            - claim_interpretation: Structured interpretation
            - error: Error message if something went wrong
        """
        try:
            # Build the prompt
            prompt = self._build_prompt(fusion_result)

            # Call Ollama
            response = self._call_ollama(prompt)

            # Parse the response
            interpretation = self._parse_response(response)

            return {
                "model": self.model,
                "claim_interpretation": interpretation,
                "raw_output": response
            }

        except Exception as e:
            return {
                "model": self.model,
                "error": str(e),
                "claim_interpretation": None
            }

    def _build_prompt(self, fusion_result: dict) -> str:
        """Build the prompt for Phi-4."""
        normalized = fusion_result.get("normalized_claim_input", {})
        missing_fields = fusion_result.get("missing_fields", [])
        conflicts = fusion_result.get("conflicts", [])
        source_map = fusion_result.get("source_map", {})

        prompt = PHI4_SYSTEM_PROMPT + "\n\n"

        prompt += "CLAIM DATA TO ANALYZE:\n"
        prompt += "```json\n"
        prompt += json.dumps({
            "normalized_claim_input": normalized,
            "missing_fields": missing_fields,
            "conflicts": conflicts,
            "source_map": source_map
        }, indent=2, default=str)
        prompt += "\n```\n\n"

        prompt += "Return your analysis as valid JSON only, no additional text."

        return prompt

    def _call_ollama(self, prompt: str, connect_timeout: float = 5.0, read_timeout: float = 60.0) -> str:
        """
        Call Ollama API to get Phi-4 response.

        Args:
            prompt: The full prompt to send
            connect_timeout: Connection timeout in seconds
            read_timeout: Read timeout in seconds

        Returns:
            The model's response text
        """
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent JSON
                "top_p": 0.9
            }
        }

        timeout = httpx.Timeout(connect=connect_timeout, read=read_timeout)

        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()

            result = response.json()
            return result.get("response", "")

    def _parse_response(self, response: str) -> Optional[dict]:
        """
        Parse the Phi-4 response into structured data.

        Attempts to extract JSON from the response.
        Falls back to safe error handling if parsing fails.
        """
        # Try to find JSON in the response
        json_match = re.search(
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
            response,
            re.DOTALL
        )

        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return self._validate_and_normalize(parsed)
            except json.JSONDecodeError:
                pass

        # Try to extract JSON from markdown code blocks
        code_block_match = re.search(
            r'```(?:json)?\s*([\s\S]*?)```',
            response
        )

        if code_block_match:
            try:
                parsed = json.loads(code_block_match.group(1))
                return self._validate_and_normalize(parsed)
            except json.JSONDecodeError:
                pass

        # Try to find JSON at the start
        try:
            # Find first { and last }
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                parsed = json.loads(json_str)
                return self._validate_and_normalize(parsed)
        except (json.JSONDecodeError, ValueError):
            pass

        # Return error indication
        return {
            "claim_type": "unknown",
            "accident_summary": "Failed to parse model response",
            "damage_summary": None,
            "damage_location": None,
            "possible_fault_indicators": [],
            "fraud_indicators": [],
            "requires_additional_information": ["Unable to parse AI reasoning response"],
            "source_conflicts_noted": [],
            "language_notes": f"Raw response: {response[:500]}"
        }

    def _validate_and_normalize(self, parsed: dict) -> dict:
        """
        Validate and normalize the parsed response.

        Ensures all expected fields are present and properly typed.
        """
        default_value = None

        return {
            "claim_type": parsed.get("claim_type", default_value),
            "accident_summary": parsed.get("accident_summary", default_value),
            "damage_summary": parsed.get("damage_summary", default_value),
            "damage_location": parsed.get("damage_location", default_value),
            "possible_fault_indicators": parsed.get("possible_fault_indicators", []),
            "fraud_indicators": parsed.get("fraud_indicators", []),
            "requires_additional_information": parsed.get(
                "requires_additional_information",
                []
            ),
            "source_conflicts_noted": parsed.get("source_conflicts_noted", []),
            "language_notes": parsed.get("language_notes", default_value)
        }
