# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **`app/api/routes.py:483` — vision pipeline (qwen3-vl:4b) was always rejected as `reject_not_supported`.** The route built a `fusion_result` without a `raw_sources` key. The rules engine's `DOCUMENT_EXTRACTION_FAILED` check then fired unconditionally (all three sources `ocr_text` / `florence_structured` / `user_text` are empty by design for the vision pipeline — the model produces structured JSON, not text). The decision engine's `_make_high_risk_decision` maps that rule to `recommended_action: "reject_not_supported"`, so every valid vision-pipeline call ended in rejection even when the model successfully extracted every form field. The route now populates `raw_sources` with a JSON dump of the model's `structured_output` as a surrogate `ocr_text`, plus `florence_structured` (the `diagram_analysis` dict if present) and `user_text`. This matches the contract the rules engine already expects, without touching the rules engine itself.

- **`app/api/routes.py:445, 522` — vision-pipeline audit log was lying about latency.** `qwen_vl_extractor.extract()` returns `latency_seconds` (see `app/services/qwen_vl_extractor.py:125,138,152,165`). The route's exception-path dict and the `ai_reasoning` payload both read `latency_ms`, which was never set → the audit log recorded `latency_ms: 0` for every vision call regardless of wall-clock cost. Switched to `latency_seconds` in both spots.

### Tests

- `tests/test_rules_engine.py` — three regression tests pin the contract:
  - `test_vision_pipeline_with_structured_output_does_not_trigger_extraction_failed` — a vision-pipeline `fusion_result` with populated `raw_sources` must NOT trip `DOCUMENT_EXTRACTION_FAILED`.
  - `test_vision_pipeline_without_raw_sources_triggers_extraction_failed` — sanity check: the old (broken) shape still trips the rule, so the fix doesn't silently weaken it.
  - `test_text_only_pipeline_still_triggers_extraction_failed` — sanity check: the real extraction-failed case (all three sources empty) still trips the rule, so the fix doesn't mask real failures.
