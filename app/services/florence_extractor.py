"""Florence-2 extractor - visual/semantic document extraction."""

from pathlib import Path
from typing import Optional

from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from app.config import get_settings


class FlorenceExtractor:
    """Handles visual extraction using Florence-2 model."""

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize Florence-2 extractor.

        Args:
            model_name: Optional Florence-2 model name.
                       Defaults to settings.florence_model_name
        """
        self.settings = get_settings()
        self.model_name = model_name or self.settings.florence_model_name
        self._model = None
        self._processor = None
        self._available = None

    def is_available(self) -> bool:
        """Check if Florence-2 is available."""
        if self._available is not None:
            return self._available

        try:
            # Try to load the model
            self._load_model()
            self._available = True
            return True
        except Exception:
            self._available = False
            return False

    def _load_model(self):
        """Load Florence-2 model and processor."""
        if self._model is None:
            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32  # Use float32 for CPU
            )
            self._model.eval()

    def extract_claim_visuals(self, image_path: str) -> dict:
        """
        Extract visual/semantic information from a claim form image.

        Florence-2 is used for perception/extraction only - NOT for final decisions.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with:
            - source: Always "florence-2"
            - raw_output: Raw model output text
            - structured_output: Structured extraction
            - warnings: Any warnings
        """
        warnings = []

        if not self.is_available():
            return {
                "source": "florence-2",
                "raw_output": "",
                "structured_output": {
                    "visible_damage_area": None,
                    "accident_diagram_summary": None,
                    "checkboxes_or_marked_fields": [],
                    "visible_notes": [],
                    "detected_objects_or_regions": []
                },
                "warnings": ["Florence-2 model not available"]
            }

        try:
            # Load image
            image = Image.open(image_path).convert("RGB")

            # Load model if not already loaded
            if self._model is None:
                self._load_model()

            # For Florence-2, we use different tasks for document understanding
            # Task: <DOCUMENT_OCR> or <GENERAL_OCR> for text extraction
            # Task: <REGION_PROPOSAL> or <CAPTION_TO_PHRASE_GROUNDING> for structure

            # First, get a caption/description of the document
            caption = self._get_document_caption(image)

            # Get detected regions/objects
            regions = self._get_detected_regions(image)

            # Get OCR text if possible
            ocr_text = self._get_ocr_text(image)

            # Analyze for damage indicators
            damage_analysis = self._analyze_damage_indicators(image)

            # Combine results
            structured_output = {
                "visible_damage_area": damage_analysis.get("damage_area"),
                "accident_diagram_summary": self._extract_diagram_info(caption, regions),
                "checkboxes_or_marked_fields": regions.get("marked_fields", []),
                "visible_notes": ocr_text.get("handwritten_notes", []),
                "detected_objects_or_regions": regions.get("objects", [])
            }

            return {
                "source": "florence-2",
                "raw_output": caption + "\n" + str(regions),
                "structured_output": structured_output,
                "warnings": warnings
            }

        except Exception as e:
            warnings.append(f"Florence-2 extraction failed: {str(e)}")
            return {
                "source": "florence-2",
                "raw_output": "",
                "structured_output": {
                    "visible_damage_area": None,
                    "accident_diagram_summary": None,
                    "checkboxes_or_marked_fields": [],
                    "visible_notes": [],
                    "detected_objects_or_regions": []
                },
                "warnings": warnings
            }

    def _get_document_caption(self, image: Image.Image) -> str:
        """
        Get a caption/description of the document.

        Uses Florence-2's caption capability.
        """
        try:
            prompt = "<DETAILED_CAPTION>"
            inputs = self._processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )

            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                do_sample=False
            )

            generated_text = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=False
            )[0]

            # Parse the result
            parsed_answer = self._processor.post_process_generation(
                generated_text,
                task="<DETAILED_CAPTION>",
                image_size=(image.width, image.height)
            )

            return parsed_answer.get("<DETAILED_CAPTION>", "")
        except Exception:
            return ""

    def _get_detected_regions(self, image: Image.Image) -> dict:
        """
        Detect regions and objects in the document.

        Returns detected objects and marked fields.
        """
        result = {
            "objects": [],
            "marked_fields": []
        }

        try:
            # Use region proposal task
            prompt = "<REGION_PROPOSAL>"
            inputs = self._processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )

            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                do_sample=False
            )

            generated_text = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=False
            )[0]

            parsed = self._processor.post_process_generation(
                generated_text,
                task="<REGION_PROPOSAL>",
                image_size=(image.width, image.height)
            )

            # Extract region descriptions
            regions = parsed.get("<REGION_PROPOSAL>", [])
            if isinstance(regions, list):
                for region in regions:
                    if isinstance(region, dict) and "text" in region:
                        text = region["text"].lower()
                        result["objects"].append(region["text"])

                        # Check for checkboxes/markings
                        checkbox_indicators = ["checkbox", "marked", "checked", "x", "v"]
                        if any(ind in text for ind in checkbox_indicators):
                            result["marked_fields"].append(region["text"])

        except Exception:
            pass

        return result

    def _get_ocr_text(self, image: Image.Image) -> dict:
        """
        Get OCR text from the document.

        Focuses on extracting any visible text.
        """
        result = {
            "printed_text": [],
            "handwritten_notes": []
        }

        try:
            prompt = "<OCR>"
            inputs = self._processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )

            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False
            )

            generated_text = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=False
            )[0]

            parsed = self._processor.post_process_generation(
                generated_text,
                task="<OCR>",
                image_size=(image.width, image.height)
            )

            ocr_result = parsed.get("<OCR>", "")
            if ocr_result:
                result["printed_text"].append(ocr_result)

        except Exception:
            pass

        return result

    def _analyze_damage_indicators(self, image: Image.Image) -> dict:
        """
        Analyze the image for damage-related indicators.

        This is a simple heuristic-based analysis, not a ML model.
        """
        result = {
            "damage_area": None,
            "damage_indicators": []
        }

        # Note: This is a simplified analysis.
        # In production, you might use a specialized damage detection model.

        # Common damage-related keywords we might find in document annotations
        # The actual damage detection would require a specialized vision model

        return result

    def _extract_diagram_info(self, caption: str, regions: dict) -> Optional[str]:
        """
        Extract accident diagram information from caption and regions.

        Returns a summary if a diagram is detected.
        """
        caption_lower = caption.lower()

        # Check for diagram indicators
        diagram_indicators = ["diagram", "schematic", "illustration", "drawing"]
        has_diagram = any(ind in caption_lower for ind in diagram_indicators)

        if has_diagram or regions.get("objects"):
            # Summarize what was detected
            objects = regions.get("objects", [])
            if objects:
                return f"Document contains visual elements: {', '.join(objects[:5])}"

        return None
