# ClaimLens AI — Local Multimodal Insurance Claims Intake System

A POC demonstrating how small/local AI models can fit into an insurance company-level architecture for claims intake and triage.

## What is ClaimLens AI?

ClaimLens AI is a modular, explainable, local-first AI pipeline that processes European/Dutch motor accident claim forms (PDF or image) containing printed text, handwritten notes, checkboxes, diagrams, and incomplete fields. It extracts, fuses, reasons over, and makes deterministic decisions about insurance claims — all locally.

**This is not a chatbot. This is not a simple FastAPI wrapper.**

## The Problem It Solves

Insurance claims intake is often:
- Slow (days to weeks for manual review)
- Expensive (large call centers for simple form processing)
- Opaque (no explanation for why a claim was flagged)
- Prone to over-reliance on black-box AI confidence scores

ClaimLens AI demonstrates a better architecture:
- Tesseract OCR for robust text extraction
- A reasoning model for structured JSON extraction (qwen2.5:7b via Ollama)
- Optional vision model for direct image analysis (qwen3-vl:4b via Ollama)
- Deterministic rules for business decisions
- Full auditability and explainability

## Why Small/Local Models Fit Insurance Use Cases

1. **Data Privacy**: Insurance claims contain sensitive personal data. Local processing keeps data on-premises.
2. **Explainability**: Small models with deterministic rules produce explainable decisions.
3. **Cost Efficiency**: No per-token API costs. Fixed infrastructure cost.
4. **Latency**: Local models eliminate network round-trips.
5. **Compliance**: Easier to audit what's running and how decisions are made.

## Architecture

```
[User Upload: PDF/Image + Optional Dutch Text]
                    ↓
        [Document Preprocessor]
                    ↓
        [Tesseract OCR Extractor]
                    ↓
        [qwen2.5:7b via Ollama]
            (structured JSON extraction from OCR text)
                    ↓
            [Fusion Layer]
                    ↓
            [Rules Engine]
                    ↓
          [Decision Engine]
                    ↓
        [Final JSON Response + Audit Log]
```

### Optional Vision Pipeline

For direct image analysis without OCR:

```
[Image] → [qwen3-vl:4b via Ollama]
              (vision-based extraction, outputs via 'thinking' field)
```

**Note:** qwen3-vl:4b requires detailed prompt engineering for reliable output. See `journey.md` for full model evaluation history.

### Layer Responsibilities

| Layer | Purpose | Model/Technology |
|-------|---------|-----------------|
| Tesseract OCR | Extract raw textual content from images/PDFs | Tesseract (local) |
| qwen2.5:7b | Structured reasoning and JSON extraction from OCR text | Ollama (local) |
| qwen3-vl:4b | Vision-based extraction (optional alternative to OCR) | Ollama (local) |
| Fusion | Merge outputs, detect conflicts | Deterministic logic |
| Rules Engine | Deterministic business rules | Python logic |
| Decision Engine | Final workflow decision | Python logic |
| Audit Logger | Persistent claim record | JSON files |

### Model Selection History

See `journey.md` for the full evaluation process. Summary:

| Model | Role | Status |
|-------|------|--------|
| Florence-2 | Vision extraction | Removed — misidentified forms, wrong fields |
| gemma4:e4b | Text reasoning | Removed — hallucinated fields, unreliable |
| phi4 | Text reasoning | Removed — replaced by qwen2.5:7b |
| **qwen2.5:7b** | Text reasoning | **Active** — clean JSON output, fast, reliable |
| **qwen3-vl:4b** | Vision extraction | Available — good with detailed prompts, checkbox detection needs work |

## Important: No Fake Confidence Scores

This POC intentionally does **not** fake confidence scores. Instead, it exposes:
- Source evidence (what was extracted from where)
- Missing fields (what couldn't be extracted)
- Conflicts (where sources disagree)
- Deterministic rule flags
- Manual review decisions

This is more honest and enterprise-appropriate than pretending the system knows how confident it is.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/aniketkarne-com/claimlens-ai.git
cd claimlens-ai
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-nld
```

**Verify:**
```bash
tesseract --version
```

### 5. Install and Run Ollama

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the reasoning model (required)
ollama pull qwen2.5:7b

# Pull the vision model (optional — for direct image extraction)
ollama pull qwen3-vl:4b

# Verify models
ollama list

# Start Ollama server
ollama serve
```

The server runs on `http://localhost:11434` by default. The default model is configured in `app/config.py` (`ollama_model`, defaults to `gemma4:e4b` — update this to `qwen2.5:7b` for the recommended pipeline).

### 6. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings if needed
```

### 7. Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Usage

### Health Check

```bash
curl http://localhost:8000/api/health
```

### Analyze Claim (with file upload)

```bash
curl -X POST http://localhost:8000/api/claims/analyze \
  -F "file=@path/to/claim_form.pdf" \
  -F "user_text=Ik werd van achteren aangereden bij een stoplicht. De achterbumper is beschadigd."
```

### Analyze Claim (text only - for testing/demo)

```bash
curl -X POST http://localhost:8000/api/claims/analyze-text-only \
  -H "Content-Type: application/json" \
  -d '{"user_text": "Ik werd van achteren aangereden bij een stoplicht. De achterbumper is beschadigd."}'
```

### Swagger UI

Open http://localhost:8000/docs and use the interactive interface.

## Sample Demo Cases

See `sample_data/sample_claim_texts.json` for test cases:

1. **simple_rear_end_collision** - Basic rear-end collision
2. **repeated_claim_signal** - Detects repeated claim keywords
3. **missing_information** - Missing critical fields
4. **conflicting_damage_location** - Conflict between text and visual

## Response Structure

```json
{
  "claim_id": "generated-uuid",
  "status": "processed",
  "input_summary": {
    "file_name": "claim.pdf",
    "file_type": "pdf",
    "user_text_provided": true,
    "language": "nl"
  },
  "extraction": {
    "ocr": { "raw_text": "...", "detected_language": "nl" },
    "vision": { "structured_output": {...} }
  },
  "fusion": {
    "normalized_claim_input": {...},
    "missing_fields": [],
    "conflicts": [],
    "source_map": {}
  },
  "ai_reasoning": {
    "model": "qwen2.5:7b",
    "claim_interpretation": {...}
  },
  "rules": {
    "applied_rules": [],
    "rule_flags": []
  },
  "decision": {
    "risk_level": "low|medium|high|unknown",
    "manual_review_required": true|false,
    "recommended_action": "auto_process|request_more_info|manual_review|reject_not_supported",
    "decision_reasons": []
  },
  "audit": {
    "pipeline_version": "0.1.0",
    "steps_completed": [],
    "warnings": []
  }
}
```

## Important Note on "auto_process"

`recommended_action: "auto_process"` does **NOT** mean payment approval. It means the claim can proceed to the next automated workflow step (e.g., damage estimation, adjuster assignment). Final payment decisions always require human review.

## Known Limitations

1. **Tesseract OCR accuracy**: Handwritten text recognition is limited. Use high-quality scans.
2. **qwen3-vl:4b checkbox detection**: Vision model hallucinates checkbox selections. Use OCR + qwen2.5:7b pipeline for reliable checkbox extraction.
3. **qwen3-vl:4b thinking field**: Model output goes to the `thinking` field instead of `response` — requires special parsing. See `app/services/qwen_vl_extractor.py`.
4. **Dutch language**: Optimized for Dutch. Other languages may work but are untested.
5. **Single-page processing**: POC processes first page of PDFs. Multi-page support is structural but not implemented.
6. **No database**: POC uses local JSON files. Production would add PostgreSQL/MongoDB.

## Future Improvements

- [ ] Improve checkbox detection (larger vision model or specialized detector)
- [ ] Multi-page PDF support
- [ ] Additional language support (English, German, French)
- [ ] Damage severity classification
- [ ] Integration with insurance company databases
- [ ] PostgreSQL for persistent claim storage
- [ ] Celery/Redis for async processing
- [ ] S3-compatible storage for uploaded files
- [ ] Real-time webhook notifications for decisions
- [ ] Dashboard for manual review queue

## Project Structure

```
claimlens-ai/
├── README.md
├── journey.md              # Model selection history and evaluation notes
├── requirements.txt
├── .env.example
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py        # API endpoints
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydantic models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_loader.py       # PDF/image loading
│   │   ├── image_preprocessor.py   # Image preprocessing
│   │   ├── tesseract_ocr.py         # Tesseract OCR
│   │   ├── qwen_vl_extractor.py     # qwen3-vl:4b vision extraction
│   │   ├── phi4_reasoner.py          # (legacy) gemma4:e4b reasoning
│   │   ├── fusion.py                # Fusion layer
│   │   ├── rules_engine.py          # Business rules
│   │   ├── decision_engine.py       # Decision engine
│   │   └── audit_logger.py          # Audit logging
│   └── utils/
│       ├── __init__.py
│       ├── json_utils.py
│       └── file_utils.py
├── sample_data/
│   ├── README.md
│   └── sample_claim_texts.json
├── tests/
│   ├── test_fusion.py
│   ├── test_rules_engine.py
│   └── test_decision_engine.py
├── logs/
│   └── claims/              # Audit logs
└── uploads/                 # Temporary file storage
```

## Ollama Commands Reference

```bash
# Check installed models
ollama list

# Pull models
ollama pull qwen2.5:7b          # Primary reasoning model (required)
ollama pull qwen3-vl:4b         # Vision model (optional)

# Run interactively
ollama run qwen2.5:7b

# Check running models
ollama ps

# Remove unused models
ollama rm phi4
ollama rm gemma4:e4b
```

## License

Private - All rights reserved.

## Authors

Built as a POC for demonstrating local AI in insurance claims processing.
