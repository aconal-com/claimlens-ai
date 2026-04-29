"""Image preprocessor - handles image preprocessing for OCR and vision models."""

import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from app.config import get_settings


class ImagePreprocessor:
    """Handles image preprocessing for OCR and vision extraction."""

    def __init__(self):
        self.settings = get_settings()
        self.upload_dir = self.settings.upload_path

    def preprocess_for_ocr(
        self,
        image_path: str,
        claim_id: str,
        target_size: Optional[Tuple[int, int]] = None,
        enhance_contrast: bool = True
    ) -> str:
        """
        Preprocess an image for Tesseract OCR.

        Steps:
        1. Load image
        2. Convert to RGB if needed
        3. Optionally resize
        4. Optionally enhance contrast
        5. Save preprocessed image

        Args:
            image_path: Path to the source image
            claim_id: Claim identifier for naming
            target_size: Optional (width, height) to resize to
            enhance_contrast: Whether to apply contrast enhancement

        Returns:
            Path to the preprocessed image
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert to RGB (OpenCV loads as BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize if needed
        if target_size:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if enhance_contrast:
            # Convert to LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to the L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)

            # Merge channels and convert back
            lab = cv2.merge([l, a, b])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Convert to PIL Image for saving
        pil_img = Image.fromarray(img)

        # Save preprocessed image
        output_filename = f"{claim_id}_ocr_preprocessed.png"
        output_path = self.upload_dir / output_filename
        pil_img.save(str(output_path), "PNG")

        return str(output_path)

    def preprocess_for_vision(
        self,
        image_path: str,
        claim_id: str,
        max_size: int = 1024
    ) -> str:
        """
        Preprocess an image for Florence-2 vision model.

        For vision models, we want to preserve as much visual information as possible,
        so preprocessing is minimal.

        Args:
            image_path: Path to the source image
            claim_id: Claim identifier for naming
            max_size: Maximum dimension size (images larger will be resized)

        Returns:
            Path to the preprocessed image
        """
        # Load image
        img = Image.open(image_path)

        # Get current size
        width, height = img.size

        # Resize if too large (preserve aspect ratio)
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Save preprocessed image
        output_filename = f"{claim_id}_vision_preprocessed.png"
        output_path = self.upload_dir / output_filename
        img.save(str(output_path), "PNG")

        return str(output_path)

    def get_image_info(self, image_path: str) -> dict:
        """
        Get basic information about an image.

        Args:
            image_path: Path to the image

        Returns:
            Dictionary with image information
        """
        img = Image.open(image_path)
        width, height = img.size
        mode = img.mode

        return {
            "path": image_path,
            "width": width,
            "height": height,
            "mode": mode,
            "format": img.format
        }
