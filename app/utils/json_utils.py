"""JSON utilities for ClaimLens AI."""

import json
import re
from typing import Any, Optional


def safe_json_parse(json_string: str) -> Optional[dict]:
    """
    Safely parse a JSON string, handling common issues.

    Args:
        json_string: String to parse as JSON

    Returns:
        Parsed dict or None if parsing fails
    """
    if not json_string or not json_string.strip():
        return None

    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        pass

    # Try to fix common issues
    fixed = json_string

    # Remove BOM
    if fixed.startswith('\ufeff'):
        fixed = fixed[1:]

    # Try again
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from text
    return extract_json_from_text(json_string)


def extract_json_from_text(text: str) -> Optional[dict]:
    """
    Extract JSON from text that may contain other content.

    Args:
        text: Text that may contain JSON

    Returns:
        Extracted JSON as dict or None
    """
    if not text:
        return None

    # Try to find JSON in markdown code blocks
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    matches = re.findall(code_block_pattern, text)

    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON objects
    json_pattern = r'\{[\s\S]*\}'
    matches = re.findall(json_pattern, text)

    # Try each match from largest to smallest
    matches.sort(key=len, reverse=True)

    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            pass

    return None


def pretty_print_json(data: Any) -> str:
    """
    Pretty print JSON data.

    Args:
        data: Data to print

    Returns:
        Formatted JSON string
    """
    return json.dumps(data, indent=2, default=str)


def compact_json(data: Any) -> str:
    """
    Create compact JSON string.

    Args:
        data: Data to serialize

    Returns:
        Compact JSON string
    """
    return json.dumps(data, separators=(',', ':'), default=str)
