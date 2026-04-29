"""API routes for ClaimLens AI."""

import uuid
from typing import Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import get_settings
from app.models.schemas import (
    AnalyzeTextOnlyRequest,
    AnalyzeTextOnlyResponse,
    ClaimAnalysisResponse,
    HealthResponse,
)
from app.services.document_loader import DocumentLoader
from app.services.image_preprocessor import ImagePreprocessor
from app.services.tesseract_ocr import TesseractOCR
from app.services.florence_extractor import FlorenceExtractor
from app.services.fusion import FusionEngine
from app.services.phi4_reasoner import Phi4Reasoner
from app.services.rules_engine import RulesEngine
from app.services.decision_engine import DecisionEngine
from app.services.audit_logger import AuditLogger

settings = get_settings()
router = APIRouter()


# Initialize services lazily
_tesseract: Optional[TesseractOCR] = None
_florence: Optional[FlorenceExtractor] = None
_phi4: Optional[Phi4Reasoner] = None


def get_tesseract() -> TesseractOCR:
    """Get Tesseract OCR service instance."""
    global _tesseract
    if _tesseract is None:
        _tesseract = TesseractOCR()
    return _tesseract


def get_florence() -> FlorenceExtractor:
    """Get Florence-2 extractor service instance."""
    global _florence
    if _florence is None:
        _florence = FlorenceExtractor()
    return _florence


def get_phi4() -> Phi4Reasoner:
    """Get Phi-4 reasoner service instance."""
    global _phi4
    if _phi4 is None:
        _phi4 = Phi4Reasoner()
    return _phi4


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check the health status of the API and its dependencies.
    """
    ollama_available = False
    florence_available = False
    tesseract_available = False

    # Check Ollama
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.ollama_base_url}/api/tags",
                timeout=5.0
            )
            ollama_available = response.status_code == 200
    except Exception:
        pass

    # Check Florence-2
    try:
        florence = get_florence()
        florence_available = florence.is_available()
    except Exception:
        pass

    # Check Tesseract
    try:
        tesseract = get_tesseract()
        tesseract_available = tesseract.is_available()
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        ollama_available=ollama_available,
        configured_model=settings.ollama_model,
        florence_available=florence_available,
        tesseract_available=tesseract_available,
    )


@router.post("/claims/analyze", response_model=ClaimAnalysisResponse)
async def analyze_claim(
    file: UploadFile = File(...),
    user_text: Optional[str] = Form(None),
    language: str = Form("nl"),
):
    """
    Analyze an uploaded claim form (PDF or image).

    Processes the document through the full pipeline:
    1. Validate file type
    2. Save uploaded file temporarily
    3. Convert PDF to images if needed
    4. Run Tesseract OCR
    5. Run Florence-2 visual extraction
    6. Run Fusion Layer
    7. Run Phi-4 reasoning
    8. Run Rules Engine
    9. Run Decision Engine
    10. Save audit log
    11. Return final JSON
    """
    claim_id = str(uuid.uuid4())
    warnings = []
    steps_completed = []

    # Validate file type
    allowed_types = ["application/pdf", "image/png", "image/jpeg", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: PDF, PNG, JPG, JPEG"
        )

    # Save uploaded file temporarily
    doc_loader = DocumentLoader()
    try:
        file_path = await doc_loader.save_upload(file, claim_id)
        steps_completed.append("file_upload")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    # Convert PDF to images if needed
    image_paths = []
    try:
        if file.content_type == "application/pdf":
            image_paths = doc_loader.convert_pdf_to_images(file_path)
            steps_completed.append("pdf_conversion")
        else:
            image_paths = [str(file_path)]
            steps_completed.append("image_loaded")
    except Exception as e:
        warnings.append(f"Document conversion failed: {str(e)}")

    # Process images
    preprocessor = ImagePreprocessor()
    ocr_output = None
    florence_output = None
    image_path = image_paths[0] if image_paths else None

    # OCR extraction
    if image_path:
        try:
            preprocessed_path = preprocessor.preprocess_for_ocr(image_path, claim_id)
            tesseract = get_tesseract()
            ocr_output = tesseract.extract_text(preprocessed_path, language)
            steps_completed.append("tesseract_ocr")
        except Exception as e:
            warnings.append(f"OCR extraction failed: {str(e)}")
            ocr_output = {
                "source": "tesseract",
                "raw_text": "",
                "language_used": language,
                "warnings": [str(e)]
            }

    # Florence-2 extraction
    if image_path:
        try:
            florence = get_florence()
            florence_output = florence.extract_claim_visuals(image_path)
            steps_completed.append("florence_extraction")
        except Exception as e:
            warnings.append(f"Florence-2 extraction failed: {str(e)}")
            florence_output = {
                "source": "florence-2",
                "raw_output": "",
                "structured_output": {
                    "visible_damage_area": None,
                    "accident_diagram_summary": None,
                    "checkboxes_or_marked_fields": [],
                    "visible_notes": [],
                    "detected_objects_or_regions": []
                },
                "warnings": [str(e)]
            }

    # Fusion layer
    fusion_engine = FusionEngine()
    fusion_result = fusion_engine.fuse(
        ocr_output=ocr_output,
        florence_output=florence_output,
        user_text=user_text
    )
    steps_completed.append("fusion")

    # Phi-4 reasoning
    phi4_reasoner = get_phi4()
    phi4_result = None
    try:
        phi4_result = phi4_reasoner.reason(fusion_result)
        steps_completed.append("phi4_reasoning")
    except Exception as e:
        warnings.append(f"Phi-4 reasoning failed: {str(e)}")
        phi4_result = {
            "model": settings.ollama_model,
            "error": str(e),
            "claim_interpretation": None
        }

    # Rules engine
    rules_engine = RulesEngine()
    rules_result = rules_engine.apply_rules(fusion_result, phi4_result)
    steps_completed.append("rules_engine")

    # Decision engine
    decision_engine = DecisionEngine()
    decision_result = decision_engine.decide(rules_result, phi4_result)
    steps_completed.append("decision_engine")

    # Build response
    response = ClaimAnalysisResponse(
        claim_id=claim_id,
        status="processed",
        input_summary={
            "file_name": file.filename,
            "file_type": file.content_type,
            "user_text_provided": user_text is not None and len(user_text) > 0,
            "language": language,
        },
        extraction={
            "ocr": ocr_output,
            "vision": florence_output,
        },
        fusion=fusion_result,
        ai_reasoning=phi4_result,
        rules=rules_result,
        decision=decision_result,
        audit={
            "pipeline_version": settings.pipeline_version,
            "steps_completed": steps_completed,
            "warnings": warnings,
        }
    )

    # Save audit log
    try:
        audit_logger = AuditLogger()
        await audit_logger.save_claim(claim_id, response.model_dump())
    except Exception as e:
        warnings.append(f"Failed to save audit log: {str(e)}")

    return response


@router.post("/claims/analyze-text-only", response_model=AnalyzeTextOnlyResponse)
async def analyze_text_only(request: AnalyzeTextOnlyRequest):
    """
    Analyze claim using only user-provided text (no file upload).

    Useful for:
    - Testing without document upload
    - Demo purposes
    - Fallback when document processing fails

    Skips OCR and Florence-2, but still runs:
    - Fusion
    - Phi-4 reasoning
    - Rules engine
    - Decision engine
    """
    claim_id = str(uuid.uuid4())
    warnings = []
    steps_completed = ["text_input_received"]

    # Build minimal extraction outputs
    ocr_output = {
        "source": "tesseract",
        "raw_text": "",
        "language_used": request.language or settings.default_language,
        "warnings": ["Text-only mode: OCR skipped"]
    }

    florence_output = {
        "source": "florence-2",
        "raw_output": "",
        "structured_output": {
            "visible_damage_area": None,
            "accident_diagram_summary": None,
            "checkboxes_or_marked_fields": [],
            "visible_notes": [],
            "detected_objects_or_regions": []
        },
        "warnings": ["Text-only mode: Florence-2 extraction skipped"]
    }

    # Fusion layer
    fusion_engine = FusionEngine()
    fusion_result = fusion_engine.fuse(
        ocr_output=ocr_output,
        florence_output=florence_output,
        user_text=request.user_text
    )
    steps_completed.append("fusion")

    # Phi-4 reasoning
    phi4_reasoner = get_phi4()
    phi4_result = None
    try:
        phi4_result = phi4_reasoner.reason(fusion_result)
        steps_completed.append("phi4_reasoning")
    except Exception as e:
        warnings.append(f"Phi-4 reasoning failed: {str(e)}")
        phi4_result = {
            "model": settings.ollama_model,
            "error": str(e),
            "claim_interpretation": None
        }

    # Rules engine
    rules_engine = RulesEngine()
    rules_result = rules_engine.apply_rules(fusion_result, phi4_result)
    steps_completed.append("rules_engine")

    # Decision engine
    decision_engine = DecisionEngine()
    decision_result = decision_engine.decide(rules_result, phi4_result)
    steps_completed.append("decision_engine")

    # Build response
    response = AnalyzeTextOnlyResponse(
        claim_id=claim_id,
        status="processed",
        input_summary={
            "user_text_provided": True,
            "language": request.language or settings.default_language,
            "mode": "text_only"
        },
        extraction={
            "ocr": ocr_output,
            "vision": florence_output,
        },
        fusion=fusion_result,
        ai_reasoning=phi4_result,
        rules=rules_result,
        decision=decision_result,
        audit={
            "pipeline_version": settings.pipeline_version,
            "steps_completed": steps_completed,
            "warnings": warnings,
        }
    )

    # Save audit log
    try:
        audit_logger = AuditLogger()
        await audit_logger.save_claim(claim_id, response.model_dump())
    except Exception as e:
        warnings.append(f"Failed to save audit log: {str(e)}")

    return response
