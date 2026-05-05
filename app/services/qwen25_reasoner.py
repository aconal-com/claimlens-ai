"""qwen2.5:7b reasoning layer - structured reasoning over fused claim data via Ollama."""

import json
import re
import time
from typing import Any, Dict, Optional

import httpx

from app.config import get_settings


QWEN25_SYSTEM_PROMPT = """Task: Extract structured data from the European Motor Accident Statement text below.

Return a valid JSON object with these exact fields (null if not found):

{
  "accident_date": "DD-MM-YYYY or null",
  "accident_time": "HH:MM or null",
  "location": "string or null",
  "injuries": true/false/null",
  "property_damage_other": true/false/null",
  "witnesses": ["name, address, phone"] or null,
  "insured_name_a": "string or null",
  "insured_address_a": "string or null",
  "vehicle_a": "make model or null",
  "plate_a": "registration or null",
  "insurance_a": "company name or null",
  "policy_a": "policy number or null",
  "driver_name_a": "string or null",
  "license_a": "license number or null",
  "point_of_impact_a": "string or null",
  "damage_a": "description or null",
  "remarks_a": "string or null",
  "insured_name_b": "string or null",
  "insured_address_b": "string or null",
  "vehicle_b": "make model or null",
  "plate_b": "registration or null",
  "insurance_b": "company name or null",
  "policy_b": "policy number or null",
  "driver_name_b": "string or null",
  "license_b": "license number or null",
  "point_of_impact_b": "string or null",
  "damage_b": "description or null",
  "remarks_b": "string or null",
  "circumstances_a": [1,2,...] or null,
  "circumstances_b": [1,2,...] or null,
  "accident_summary": "string or null",
  "fault_assessment": "string or null",
  "requires_more_info": ["string"] or null
}

RULES:
- Return ONLY valid JSON. No markdown. No explanation. No thinking.
- Extract what you see. If not present, use null.
- For witnesses: extract name, address and phone if present.
- Date format: DD-MM-YYYY"""


class Qwen25Reasoner:
    """Handles structured reasoning using qwen2.5:7b via Ollama."""

    def __init__(self, model: str = "qwen2.5:7b"):
        self.settings = get_settings()
        self.base_url = self.settings.ollama_base_url
        self.model = model

    def reason(self, fusion_result: dict, user_text: str = None) -> dict:
        """
        Use qwen2.5:7b to reason over fused OCR + optional user text.

        Returns:
            Dictionary containing timing and claim interpretation.
        """
        start = time.time()
        try:
            # OCR text is in raw_sources.ocr_text (not normalized_claim_input.raw_text)
            ocr_text = fusion_result.get("raw_sources", {}).get("ocr_text", "")
            if user_text:
                ocr_text = ocr_text + "\n\nUSER ADDITIONAL INFO:\n" + user_text

            prompt = self._build_prompt(ocr_text)
            response = self._call_ollama(prompt)
            interpretation = self._parse_response(response)

            return {
                "model": self.model,
                "claim_interpretation": interpretation,
                "raw_output": response[:1000],
                "latency_ms": round((time.time() - start) * 1000)
            }

        except Exception as e:
            return {
                "model": self.model,
                "error": str(e),
                "claim_interpretation": None,
                "latency_ms": round((time.time() - start) * 1000)
            }

    def _build_prompt(self, ocr_text: str) -> str:
        prompt = QWEN25_SYSTEM_PROMPT + "\n\nOCR TEXT FROM ACCIDENT FORM:\n"
        prompt += "```\n" + (ocr_text[:8000] if ocr_text else "[No OCR text available]") + "\n```\n\n"
        prompt += "Return your analysis as valid JSON only."
        return prompt

    def _call_ollama(self, prompt: str, connect_timeout: float = 5.0, read_timeout: float = 120.0) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 1500
            }
        }
        timeout = httpx.Timeout(connect=connect_timeout, read=read_timeout, write=30.0, pool=30.0)
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")

    def _parse_response(self, response: str) -> Optional[dict]:
        if not response:
            return None

        # Try markdown code block
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try raw JSON braces
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(response[start:end])
            except json.JSONDecodeError:
                pass

        # Try whole text
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        return {"claim_type": "unknown", "accident_summary": response[:500]}
