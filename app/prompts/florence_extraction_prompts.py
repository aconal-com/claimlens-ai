"""
Florence-2 extraction prompts for claim form visual analysis.
"""

# Florence-2 task prompts for different extraction objectives

FLORENCE_TASKS = {
    "document_caption": {
        "task": "<DETAILED_CAPTION>",
        "description": "Get a detailed caption/description of the document"
    },
    "ocr": {
        "task": "<OCR>",
        "description": "Extract all visible text from the document"
    },
    "region_proposal": {
        "task": "<REGION_PROPOSAL>",
        "description": "Detect and propose regions/objects in the document"
    },
    "phrase_grounding": {
        "task": "<CAPTION_TO_PHRASE_GROUNDING>",
        "description": "Ground phrases to specific regions"
    }
}


def get_extraction_prompt_for_claim_form() -> str:
    """
    Get a prompt instructing Florence-2 to analyze a European motor accident claim form.

    Returns a structured instruction for the model.
    """
    return """Analyze this image as a European/Dutch motor accident claim form.

    Look for and extract:
    1. Visible damage to vehicles (location and type)
    2. Any accident diagrams or sketches
    3. Checked or marked boxes/fields
    4. Visible handwritten or printed notes
    5. Detected form regions (sections, fields, labels)

    Return concise, structured information only.
    If something is not visible or unclear, indicate that rather than guessing.

    Do NOT:
    - Make final insurance risk decisions
    - Determine fraud likelihood
    - Approve or reject claims
    - Infer information that is not visible

    Focus only on PERCEPTION and EXTRACTION.
    """


def get_damage_detection_prompt() -> str:
    """
    Get a prompt specifically for damage detection in claim forms.
    """
    return """Analyze this image for vehicle damage indicators.

    Look for:
    - Visible damage areas (dents, scratches, broken parts)
    - Damage location on the vehicle
    - Severity indicators (if observable)
    - Multiple vehicle involvement

    Return what you observe, not conclusions.
    """


def get_form_structure_prompt() -> str:
    """
    Get a prompt for understanding document/form structure.
    """
    return """Analyze the structure of this document.

    Identify:
    - Form sections and their labels
    - Data entry fields (text fields, checkboxes, etc.)
    - Marked or filled fields
    - Document type indicators

    Return a structured overview of the document layout.
    """
