# Sample Data for ClaimLens AI

This directory contains sample claim data for testing the ClaimLens AI system.

## Files

- `sample_claim_texts.json` - Sample Dutch claim texts with test cases

## Usage

Use these sample texts with the `/api/claims/analyze-text-only` endpoint for testing:

```bash
curl -X POST http://localhost:8000/api/claims/analyze-text-only \
  -H "Content-Type: application/json" \
  -d '{"user_text": "Ik werd van achteren aangereden bij een stoplicht..."}'
```
