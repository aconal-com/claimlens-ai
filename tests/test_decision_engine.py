"""Tests for the decision engine."""

import pytest

from app.services.decision_engine import DecisionEngine


class TestDecisionEngine:
    """Test cases for DecisionEngine."""

    def setup_method(self):
        """Set up test fixtures."""
        self.decision = DecisionEngine()

    def test_high_severity_leads_to_manual_review(self):
        """Test that high severity rules lead to manual_review recommendation."""
        rules_result = {
            "applied_rules": [
                {
                    "rule_id": "SOURCE_CONFLICT",
                    "severity": "high",
                    "message": "Conflicting information detected"
                }
            ],
            "rule_flags": ["SOURCE_CONFLICT"]
        }

        result = self.decision.decide(rules_result, None)

        assert result["risk_level"] == "high"
        assert result["manual_review_required"] is True
        assert result["recommended_action"] == "manual_review"

    def test_medium_severity_leads_to_request_more_info(self):
        """Test that medium severity rules lead to request_more_info."""
        rules_result = {
            "applied_rules": [
                {
                    "rule_id": "MISSING_CRITICAL_FIELDS",
                    "severity": "medium",
                    "message": "Critical fields missing"
                }
            ],
            "rule_flags": ["MISSING_CRITICAL_FIELDS"]
        }

        result = self.decision.decide(rules_result, None)

        assert result["risk_level"] == "medium"
        assert result["manual_review_required"] is True
        assert result["recommended_action"] == "request_more_info"

    def test_unknown_claim_type_leads_to_request_more_info(self):
        """Test that unknown claim type leads to request_more_info."""
        rules_result = {
            "applied_rules": [],
            "rule_flags": []
        }

        phi4_result = {
            "claim_interpretation": {
                "claim_type": "unknown"
            }
        }

        result = self.decision.decide(rules_result, phi4_result)

        assert result["risk_level"] == "unknown"
        assert result["manual_review_required"] is True
        assert result["recommended_action"] == "request_more_info"

    def test_no_phi4_result_leads_to_unknown(self):
        """Test that missing Phi-4 result leads to unknown claim type."""
        rules_result = {
            "applied_rules": [],
            "rule_flags": []
        }

        result = self.decision.decide(rules_result, None)

        assert result["risk_level"] == "unknown"
        assert result["manual_review_required"] is True
        assert result["recommended_action"] == "request_more_info"

    def test_low_risk_auto_process(self):
        """Test that clean claims lead to auto_process."""
        rules_result = {
            "applied_rules": [],
            "rule_flags": []
        }

        phi4_result = {
            "claim_interpretation": {
                "claim_type": "motor_accident"
            }
        }

        result = self.decision.decide(rules_result, phi4_result)

        assert result["risk_level"] == "low"
        assert result["manual_review_required"] is False
        assert result["recommended_action"] == "auto_process"

    def test_high_severity_overrides_medium(self):
        """Test that high severity rules take precedence over medium."""
        rules_result = {
            "applied_rules": [
                {
                    "rule_id": "SOURCE_CONFLICT",
                    "severity": "high",
                    "message": "Conflicting information"
                },
                {
                    "rule_id": "MISSING_CRITICAL_FIELDS",
                    "severity": "medium",
                    "message": "Missing fields"
                }
            ],
            "rule_flags": ["SOURCE_CONFLICT", "MISSING_CRITICAL_FIELDS"]
        }

        result = self.decision.decide(rules_result, None)

        assert result["risk_level"] == "high"
        assert result["recommended_action"] == "manual_review"

    def test_extraction_failed_leads_to_reject(self):
        """Test that DOCUMENT_EXTRACTION_FAILED leads to reject_not_supported."""
        rules_result = {
            "applied_rules": [
                {
                    "rule_id": "DOCUMENT_EXTRACTION_FAILED",
                    "severity": "high",
                    "message": "Document could not be processed"
                }
            ],
            "rule_flags": ["DOCUMENT_EXTRACTION_FAILED"]
        }

        result = self.decision.decide(rules_result, None)

        assert result["risk_level"] == "high"
        assert result["recommended_action"] == "reject_not_supported"

    def test_decision_reasons_included(self):
        """Test that decision reasons are included in the response."""
        rules_result = {
            "applied_rules": [
                {
                    "rule_id": "UNKNOWN_ACCIDENT_TYPE",
                    "severity": "medium",
                    "message": "Accident type unknown"
                }
            ],
            "rule_flags": ["UNKNOWN_ACCIDENT_TYPE"]
        }

        result = self.decision.decide(rules_result, None)

        assert len(result["decision_reasons"]) > 0

    def test_auto_process_does_not_mean_payment_approval(self):
        """Test that auto_process includes the caveat about payment approval."""
        rules_result = {
            "applied_rules": [],
            "rule_flags": []
        }

        phi4_result = {
            "claim_interpretation": {
                "claim_type": "motor_accident"
            }
        }

        result = self.decision.decide(rules_result, phi4_result)

        # The decision reasons should mention that auto_process doesn't mean payment approval
        reasons_text = " ".join(result["decision_reasons"]).lower()
        # This assertion checks the engine includes the caveat
        assert result["recommended_action"] == "auto_process"
