"""Tesseract OCR service."""

import shutil
import subprocess
from typing import Optional

import pytesseract
from pytesseract import Output

from app.config import get_settings


class TesseractOCR:
    """Handles OCR extraction using Tesseract."""

    def __init__(self):
        self.settings = get_settings()
        self._available = None

    def is_available(self) -> bool:
        """Check if Tesseract is available and working."""
        if self._available is not None:
            return self._available

        try:
            # Check if tesseract command is available
            result = shutil.which("tesseract")
            if result is None:
                self._available = False
                return False

            # Try to get tesseract version
            version = pytesseract.get_tesseract_version()
            self._available = True
            return True
        except Exception:
            self._available = False
            return False

    def _get_lang_code(self, language: str) -> str:
        """
        Get the Tesseract language code for the given language.

        Args:
            language: Language code (e.g., 'nl', 'eng')

        Returns:
            Tesseract language code
        """
        if language == "nl":
            return self.settings.tesseract_dutch_lang
        else:
            return self.settings.tesseract_english_lang

    def extract_text(
        self,
        image_path: str,
        language: str = "nl"
    ) -> dict:
        """
        Extract text from an image using Tesseract OCR.

        Args:
            image_path: Path to the image file
            language: Language code (default: 'nl' for Dutch)

        Returns:
            Dictionary with:
            - source: Always "tesseract"
            - raw_text: The extracted text
            - language_used: The language code used
            - warnings: Any warnings or errors
        """
        warnings = []

        if not self.is_available():
            return {
                "source": "tesseract",
                "raw_text": "",
                "language_used": language,
                "warnings": ["Tesseract is not available"]
            }

        try:
            lang_code = self._get_lang_code(language)

            # Check if the language is available
            try:
                pytesseract.get_tesseract_lang_codes(lang_code)
            except Exception:
                # Fallback to English if Dutch is not available
                lang_code = self.settings.tesseract_english_lang
                warnings.append(
                    f"Language '{language}' not available, falling back to English"
                )

            # Extract text
            text = pytesseract.image_to_string(
                image_path,
                lang=lang_code,
                config="--psm 6"  # Assume uniform block of text
            )

            # Get detailed data for additional info
            data = pytesseract.image_to_data(
                image_path,
                lang=lang_code,
                output_type=Output.DICT
            )

            # Try to detect language (best effort)
            detected_lang = self._detect_language(text)

            return {
                "source": "tesseract",
                "raw_text": text.strip(),
                "language_used": detected_lang if detected_lang else lang_code,
                "warnings": warnings
            }

        except Exception as e:
            warnings.append(f"OCR extraction failed: {str(e)}")
            return {
                "source": "tesseract",
                "raw_text": "",
                "language_used": language,
                "warnings": warnings
            }

    def _detect_language(self, text: str) -> Optional[str]:
        """
        Simple language detection based on common words.

        This is a best-effort detection and not 100% accurate.

        Args:
            text: Text to analyze

        Returns:
            Detected language code or None
        """
        if not text:
            return None

        text_lower = text.lower()

        # Dutch indicators
        dutch_words = [
            "de", "het", "een", "van", "en", "in", "is", "op", "te", "voor",
            "met", "zijn", "dat", "niet", "aan", "er", "maar", "om", "ook",
            "als", "dan", "naar", "bij", "door", "uit", "nog", "wel",
            "schade", "voertuig", "verzekering", "claim", "datum", "locatie",
            "beschadigd", "bumper", "voor", "achter", "zijkant"
        ]

        # English indicators
        english_words = [
            "the", "a", "an", "is", "in", "on", "at", "to", "for", "with",
            "of", "and", "or", "but", "from", "by", "this", "that", "it",
            "damage", "vehicle", "insurance", "claim", "date", "location",
            "front", "rear", "side", "door", "hood", "trunk"
        ]

        text_words = set(text_lower.split())

        dutch_count = sum(1 for w in dutch_words if w in text_words)
        english_count = sum(1 for w in english_words if w in text_words)

        if dutch_count > english_count:
            return "nl"
        elif english_count > dutch_count:
            return "eng"

        return None

    def get_detailed_data(self, image_path: str, language: str = "nl") -> dict:
        """
        Get detailed OCR data including bounding boxes and confidence.

        Note: We don't fabricate confidence scores, but Tesseract provides
        metadata that we can include if available.

        Args:
            image_path: Path to the image file
            language: Language code

        Returns:
            Dictionary with detailed OCR data
        """
        try:
            lang_code = self._get_lang_code(language)
            data = pytesseract.image_to_data(
                image_path,
                lang=lang_code,
                output_type=Output.DICT
            )
            return data
        except Exception as e:
            return {"error": str(e)}
