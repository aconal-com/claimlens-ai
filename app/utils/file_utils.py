"""File utilities for ClaimLens AI."""

import os
import shutil
from pathlib import Path
from typing import List, Optional


def ensure_directory(path: Path) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure
    """
    path.mkdir(parents=True, exist_ok=True)


def cleanup_temp_files(file_paths: List[str]) -> None:
    """
    Clean up temporary files.

    Args:
        file_paths: List of file paths to remove
    """
    for file_path in file_paths:
        try:
            path = Path(file_path)
            if path.exists():
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
        except Exception:
            pass  # Best effort cleanup


def get_file_extension(filename: str) -> str:
    """
    Get the file extension from a filename.

    Args:
        filename: The filename

    Returns:
        The extension (lowercase, without dot)
    """
    return Path(filename).suffix.lstrip('.').lower()


def get_file_size(file_path: str) -> int:
    """
    Get the size of a file in bytes.

    Args:
        file_path: Path to the file

    Returns:
        File size in bytes, or 0 if file doesn't exist
    """
    try:
        return Path(file_path).stat().st_size
    except Exception:
        return 0


def is_valid_image_file(filename: str) -> bool:
    """
    Check if a filename has a valid image extension.

    Args:
        filename: The filename to check

    Returns:
        True if valid image extension
    """
    valid_extensions = {'png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp', 'gif'}
    return get_file_extension(filename) in valid_extensions


def is_valid_pdf_file(filename: str) -> bool:
    """
    Check if a filename has a valid PDF extension.

    Args:
        filename: The filename to check

    Returns:
        True if valid PDF extension
    """
    return get_file_extension(filename) == 'pdf'


def generate_unique_filename(base_name: str, extension: str) -> str:
    """
    Generate a unique filename by appending a counter if needed.

    Args:
        base_name: Base name without extension
        extension: File extension

    Returns:
        Unique filename
    """
    counter = 1
    filename = f"{base_name}.{extension}"
    path = Path(filename)

    while path.exists():
        filename = f"{base_name}_{counter}.{extension}"
        path = Path(filename)
        counter += 1

    return filename
