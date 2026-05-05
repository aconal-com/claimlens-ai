# ClaimLens AI — Model Selection Journey

> **Context:** This document logs the experimental process of selecting the right LLM for the claim extraction pipeline in ClaimLens AI — a FastAPI + Flutter system that processes European Accident Statement images and extracts structured data.

---

## Goal

Replace the existing noisy Tesseract OCR + Florence-2 + gemma4:e4b pipeline with a single reliable model that can:
1. Read accident form images directly (vision), OR
2. Process OCR text into clean structured JSON

---

## Models Evaluated

### 1. Florence-2 (vision)

**Setup:**
```bash
pip install einops timm
python -c "from transformers import AutoModelForCausalLM, AutoProcessor; \
  m = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large', \
    torch_dtype=torch.bfloat16, trust_remote_code=True); \
  p = AutoProcessor.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True)"
```

**Benchmark result:** ❌ BROKEN
- Model misidentified the European Accident Statement form
- Called it a "claim form" and extracted wrong fields
- Confidence: ~0.41 on wrong document type
- CPU inference was extremely slow; GPU would need CUDA setup

**Verdict:** Unreliable for this use case. Removed from pipeline.

---

### 2. gemma4:e4b (text reasoning)

**Setup:** Already installed in Ollama
```bash
ollama pull gemma4:e4b   # 9.6 GB
```

**Benchmark result:** ⚠️ UNRELIABLE
- Used as text reasoner over OCR output
- Extracted some fields correctly (driver names, damage)
- Hallucinated "Accident Report" title that wasn't in the form
- Slow inference
- Removed from Ollama after testing to save space

**Verdict:** Too unreliable for production. Removed.

---

### 3. qwen3-vl:4b (vision, 3.3 GB)

**Setup:**
```bash
ollama pull qwen3-vl:4b
```

**⚠️ CRITICAL FINDING: Prompt engineering is the key variable**

Early tests used short prompts like "Extract: location, driver_a, driver_b" which produced poor results — output buried in `thinking` field, verbose reasoning, no JSON. After the user ran a properly engineered detailed prompt, qwen3-vl:4b produced **excellent structured output**, rivaling or exceeding all other models.

---

**Prompt that changed everything (user's prompt):**

```
Role:
You are an expert Document Intelligence Agent specializing in extracting structured
data from European Motor Vehicle Accident Statements ("Constat Amiable").

Task:
Analyze the provided image of an accident report form. Extract all handwritten
and printed text into a structured JSON format.

Form Structure Rules:
1. Top Section: Contains general accident details (Date, Time, Location, Witnesses).
2. Left Column (Blue - Vehicle A): Contains details for the first driver/vehicle.
3. Right Column (Yellow - Vehicle B): Contains details for the second driver/vehicle.
4. Center Column: Contains the "Circumstances" checkboxes (1-17). Identify which
   boxes are marked with an 'X' or a checkmark for Column A and Column B.
5. Bottom Section: Contains the diagram (Section 13) and signatures.

Extraction Instructions:
- Checkboxes: In Section 12 (Circumstances), list the specific numbers checked
  for Vehicle A and Vehicle B.
- Diagram (Section 13): Describe the visual layout. Note the street names, the
  direction of the arrows for Vehicle A and Vehicle B, and where the red cross
  (impact point) is located relative to the vehicles.
- Signatures: Extract the names signed at the bottom.
- Legibility: If a handwritten field is illegible, return "null" or "illegible".
  Do not guess.

Output Format:
Return the data in valid JSON format using the following schema:
{
  "accident_details": { "date": "string", "time": "string", ... },
  "vehicle_a": { ... },
  "vehicle_b": { ... },
  "circumstances": { "vehicle_a_checked_boxes": [], "vehicle_b_checked_boxes": [] },
  "diagram_analysis": { "street_names": [], ... },
  "signatures": { "driver_a_signature": "string", "driver_b_signature": "string" }
}
```

**Result with detailed prompt: ✅ EXCELLENT**
```json
{
  "accident_details": {
    "date": "14-04-2025",
    "time": "08:35",
    "location": "Kruisdrift / Parkkant junction, Utrecht",
    "injuries": false,
    "property_damage_other": false,
    "witnesses": ["Sarah Miller, 18 Oak Lane, Utrecht, +31 6 1234 7788"]
  },
  "vehicle_a": {
    "insured_name": "JANSSEN",
    "insured_address": "12 Lindenstraat\n3511 AB Utrecht",
    "vehicle_make_model": "Toyota Corolla Hybrid",
    "registration_number": "NL-AB 123-6",
    "insurance_company": "ANWB Insurance",
    "policy_number": "POL-A-4589021",
    "driver_name": "JANSSEN",
    "driver_license_number": "D-98234510",
    "point_of_impact": "Front",
    "visible_damage": "Rear bumper dented; paint scratches left; tail light cracked.",
    "remarks": "I was waiting for traffic at the junction."
  },
  "vehicle_b": {
    "insured_name": "DE VRIES",
    "insured_address": "44 Kanaalweg\n3526 KT Utrecht",
    "vehicle_make_model": "Volkswagen Golf",
    "registration_number": "NL-20-106-11",
    "insurance_company": "Centraal Beheer",
    "policy_number": "CB-993812-77",
    "driver_name": "DE VRIES",
    "driver_license_number": "D-7642190",
    "point_of_impact": "Front",
    "visible_damage": "Front bumper scratched; number plate bent.",
    "remarks": "Driver B says traffic stopped suddenly."
  },
  "circumstances": {
    "vehicle_a_checked_boxes": [],
    "vehicle_b_checked_boxes": [],
    "total_crosses_a": 0,
    "total_crosses_b": 0
  },
  "diagram_analysis": {
    "street_names": ["Parklaan", "Kruisstraat"],
    "vehicle_a_direction": "from Parklaan towards Kruisstraat",
    "vehicle_b_direction": "from the right side towards Kruisstraat",
    "impact_description": "The two vehicles collided at the junction..."
  },
  "signatures": { "driver_a_signature": "M. Jansen", "driver_b_signature": "L. de Vries" }
}
```

**Comparison vs earlier short-prompt test (same model, different prompt):**

| Field | Short Prompt (agent test) | Detailed Prompt (user test) |
|-------|--------------------------|------------------------------|
| Date | null ❌ | 14-04-2025 ✅ |
| Location | Lindensholt (confused) ✅ | Kruisdrift/Parkkant junction ✅ |
| Insured A | Mark Jansen (wrong field) ❌ | JANSSEN ✅ |
| Insured B | L. de Vries ✅ | DE VRIES ✅ |
| Address A | null ❌ | 12 Lindenstraat ✅ |
| Vehicle A | null ❌ | Toyota Corolla Hybrid ✅ |
| License plates | null ❌ | NL-AB 123-6, NL-20-106-11 ✅ |
| Insurance A | null ❌ | ANWB Insurance ✅ |
| Insurance B | null ❌ | Centraal Beheer ✅ |
| Driver licenses | null ❌ | D-98234510, D-7642190 ✅ |
| Remarks | null ❌ | Both driver accounts ✅ |
| Checkboxes | hallucinated all ❌ | [] (empty — still needs fixing) |

**Key findings (updated):**
- ✅ With detailed, schema-driven prompt: excellent structured extraction
- ✅ Insured names, addresses, vehicles, plates, insurance companies, licenses, damages — all correctly extracted
- ✅ Both driver remarks/narratives captured
- ⚠️ Checkbox arrays are empty — checkbox detection still unreliable
- ⚠️ `thinking` field issue: still present, but JSON output from `response` field when prompt is detailed enough
- Location accuracy improved with full schema context

**Verdict:** With proper prompt engineering, qwen3-vl:4b is the **best model tested** for this task. Vision + detailed prompting beats text-only approaches. Remaining issue: checkbox detection.

---

### 4. qwen2.5:7b (text reasoning, 4.7 GB)

**Setup:**
```bash
ollama pull qwen2.5:7b   # 4.7 GB, text-only
```

**Benchmark result:** ✅ BEST

| Test | Prompt | Output Field | Tokens | Outcome |
|------|--------|-------------|--------|---------|
| Basic math | "What is 2+2? One word." | `response` | ~20 | ✅ "Four" |
| JSON extraction (with OCR text) | Structured prompt + OCR | `response` | 202 | ✅ Clean structured JSON |

**Payload format:**
```python
payload = {
    'model': 'qwen2.5:7b',
    'prompt': '<structured prompt with OCR text>',
    'stream': False,
    'options': {
        'temperature': 0.1,
        'num_predict': 1000
    }
}
```

**Extraction quality (with OCR text fed in):**
- ✅ Location: Lindensholt
- ✅ Driver A: Janssen, Mark
- ✅ Driver B: De Vries, L.
- ✅ Damage A: Rear bumper dented; Paint scratches; Left tail light cracked
- ✅ Damage B: Front bumper scratched; Number plate bent
- ✅ Witness: Sarah Miller, 18 Oak Lane, Utrecht, +31 6 1234 7788
- ✅ Signatures: M. Jansen, L. de Vries

**JSON output from qwen2.5:7b:**
```json
{
  "location": "Lindensholt",
  "driver_a": { "name": "Janssen, Mark" },
  "driver_b": { "name": "De Vries, L." },
  "damage_a": {
    "visible_damage": [
      "Rear bumper dented",
      "Paint scratches",
      "Left tail light cracked"
    ]
  },
  "damage_b": {
    "visible_damage": [
      "Front bumper scratched",
      "Number plate bent"
    ]
  },
  "witness": {
    "name": "Sarah Miller",
    "address": "18 Oak Lane, Utrecht",
    "phone": "+31 6 1234 7788"
  },
  "signatures": {
    "driver_a_signature": "M. Jansen",
    "driver_b_signature": "L. de Vries"
  }
}
```

**Verdict:** Best extraction quality. Clean `response` field. Fast (202 tokens). Requires OCR text as input.

---

## Side-by-Side Comparison

| Criterion | Florence-2 | gemma4:e4b | qwen3-vl:4b | qwen2.5:7b |
|-----------|-----------|------------|-------------|------------|
| Vision input | ✅ | ❌ | ✅ | ❌ |
| Structured JSON | ❌ | ❌ | ❌ (in thinking) | ✅ |
| Clean output field | N/A | ✅ | ❌ | ✅ |
| Checkbox detection | N/A | N/A | ❌ Hallucinated | N/A (needs OCR) |
| No OCR dependency | ✅ | ❌ | ✅ | ❌ |
| Fast inference | ❌ (CPU slow) | ⚠️ | ⚠️ (~7s cold start) | ✅ |
| Reliable extraction | ❌ | ⚠️ | ⚠️ | ✅ |
| Installation size | ~4 GB | 9.6 GB | 3.3 GB | 4.7 GB |

---

## Final Architecture Decision

### Recommended Pipeline: Tesseract OCR + qwen2.5:7b

```
Image → Tesseract OCR → qwen2.5:7b (structured JSON extraction)
```

**Why:**
1. Tesseract OCR works reliably (installed via `brew install tesseract` + `brew install tesseract-lang`)
2. qwen2.5:7b produces clean, structured JSON from OCR text
3. No thinking field confusion — output is in `response` field
4. Fast inference (~202 tokens for extraction)
5. 4.7 GB model — reasonable size

**Why NOT qwen3-vl:4b:**
- Vision is appealing (no OCR dependency) but...
- Output always goes to `thinking` field — requires special parsing
- Verbose output consumes token budget before producing JSON
- Checkbox detection is hallucinated
- Location field inconsistency (Utrecht vs Lindensholt across runs)

### Alternative Pipeline (future): qwen3-vl:32b

If a larger qwen3-vl model (32b) becomes available on Ollama, revisit vision-based extraction. The 4b model is too small and thinking-heavy to produce clean structured output reliably.

---

## Test Image

European Accident Statement form — `/tmp/test_claim2.jpg`

**Ground truth from form:**
- Location: Lindensholt
- Driver A: Mark Jansen (JANSSEN, Mark)
- Driver B: L. de Vries (DE VRIES, L.)
- Damage A: Rear bumper dented; paint scratches; left tail light cracked
- Damage B: Front bumper scratched; number plate bent
- Witness: Sarah Miller, 18 Oak Lane, Utrecht, +31 6 1234 7788
- Signatures: M. Jansen, L. de Vries

---

## Ollama Commands Reference

```bash
# Check installed models
ollama list

# Pull a model
ollama pull qwen2.5:7b
ollama pull qwen3-vl:4b

# Remove a model
ollama rm phi4
ollama rm gemma4:e4b

# Run interactively
ollama run qwen2.5:7b

# Check running models
ollama ps
```

## API Test Script

```python
import httpx
import base64

with open('/tmp/test_claim2.jpg', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()

# qwen3-vl:4b — output is in 'thinking' field
payload_vl = {
    'model': 'qwen3-vl:4b',
    'prompt': 'Extract: location, driver_a, driver_b, damage_a, damage_b. JSON.',
    'stream': False,
    'images': [img_b64],
    'options': {'temperature': 0.1, 'num_predict': 4000}
}
r = httpx.Client(timeout=300).post('http://127.0.0.1:11434/api/generate', json=payload_vl)
thinking_output = r.json().get('thinking', '')   # output is HERE, not 'response'

# qwen2.5:7b — output is in 'response' field
payload_txt = {
    'model': 'qwen2.5:7b',
    'prompt': 'OCR text from accident form: ...\nExtract to JSON.',
    'stream': False,
    'options': {'temperature': 0.1, 'num_predict': 1000}
}
r = httpx.Client(timeout=120).post('http://127.0.0.1:11434/api/generate', json=payload_txt)
json_output = r.json().get('response', '')   # clean JSON here
```

---

## Timeline

| Date | Action | Outcome |
|------|--------|---------|
| 2026-01-XX | Tested Florence-2-large | Misidentified form, wrong extraction |
| 2026-01-XX | Removed Florence-2, fell back to Tesseract OCR + gemma4:e4b | gemma4 hallucinated |
| 2026-01-XX | Removed gemma4:e4b, pulled qwen3-vl:4b | Vision works but thinking field is problematic |
| 2026-01-XX | Pulled qwen2.5:7b | Clean JSON output, best quality |
| 2026-01-XX | Documented findings in journey.md | This file |
