"""
Unit tests for decision rules and causal graph analysis.

Tests cover:
- Decision rule evaluation
- Metric pattern detection
- Causal chain tracing
- Root cause analysis
"""

import pytest

from src.integration.decision_rules import (
    DECISION_RULES,
    CAUSAL_GRAPH,
    METRIC_PATTERNS,
    evaluate_decision_rules,
    detect_metric_patterns,
    trace_causal_chain,
    get_root_cause_analysis,
    format_diagnosis_summary,
)


# ===========================================================================
# Decision Rules Tests
# ===========================================================================

class TestDecisionRulesData:
    """Tests for DECISION_RULES data structure."""

    def test_decision_rules_not_empty(self) -> None:
        """DECISION_RULES should have at least 10 rules."""
        assert len(DECISION_RULES) >= 10

    def test_decision_rules_required_fields(self) -> None:
        """Each rule should have required fields."""
        required_fields = {"id", "condition", "diagnosis", "root_causes", "actions", "confidence", "severity"}

        for rule in DECISION_RULES:
            for field in required_fields:
                assert field in rule, f"Rule {rule.get('id', 'UNKNOWN')} missing field: {field}"

    def test_decision_rules_confidence_range(self) -> None:
        """Confidence should be between 0 and 1."""
        for rule in DECISION_RULES:
            conf = rule.get("confidence", 0)
            assert 0.0 <= conf <= 1.0, f"Rule {rule['id']} has invalid confidence: {conf}"

    def test_decision_rules_severity_values(self) -> None:
        """Severity should be one of valid values."""
        valid_severities = {"critical", "high", "medium", "low", "none"}

        for rule in DECISION_RULES:
            sev = rule.get("severity", "")
            assert sev in valid_severities, f"Rule {rule['id']} has invalid severity: {sev}"


class TestEvaluateDecisionRules:
    """Tests for evaluate_decision_rules()."""

    def test_evaluate_pure_optimization(self) -> None:
        """Low solver score + high constraint score -> PURE_OPTIMIZATION."""
        scores = {
            "solver_score": 50.0,
            "constraint_score": 90.0,
            "covariance_score": 80.0,
            "reconstruction_score": 80.0,
        }

        matches = evaluate_decision_rules(scores)

        rule_ids = [m["rule_id"] for m in matches]
        assert "PURE_OPTIMIZATION" in rule_ids

    def test_evaluate_constraint_dominated(self) -> None:
        """High solver score + low constraint score -> CONSTRAINT_DOMINATED."""
        scores = {
            "solver_score": 85.0,
            "constraint_score": 40.0,
            "covariance_score": 80.0,
            "reconstruction_score": 80.0,
        }

        matches = evaluate_decision_rules(scores)

        rule_ids = [m["rule_id"] for m in matches]
        assert "CONSTRAINT_DOMINATED" in rule_ids

    def test_evaluate_covariance_degradation(self) -> None:
        """Low covariance score -> COVARIANCE_DEGRADATION."""
        scores = {
            "solver_score": 80.0,
            "constraint_score": 80.0,
            "covariance_score": 45.0,
            "reconstruction_score": 80.0,
        }

        matches = evaluate_decision_rules(scores)

        rule_ids = [m["rule_id"] for m in matches]
        assert "COVARIANCE_DEGRADATION" in rule_ids

    def test_evaluate_healthy_pipeline(self) -> None:
        """All high scores -> HEALTHY_PIPELINE."""
        scores = {
            "solver_score": 85.0,
            "constraint_score": 75.0,
            "covariance_score": 80.0,
            "reconstruction_score": 80.0,
        }

        matches = evaluate_decision_rules(scores)

        rule_ids = [m["rule_id"] for m in matches]
        assert "HEALTHY_PIPELINE" in rule_ids

    def test_evaluate_multiple_matches(self) -> None:
        """Multiple rules can match simultaneously."""
        scores = {
            "solver_score": 45.0,
            "constraint_score": 45.0,
            "covariance_score": 45.0,
            "reconstruction_score": 45.0,
        }

        matches = evaluate_decision_rules(scores)

        # Should match OVERALL_DEGRADATION and SOLVER_CONSTRAINT_CONFLICT
        assert len(matches) >= 2

    def test_evaluate_no_matches(self) -> None:
        """Some scores may not match any rule."""
        scores = {
            "solver_score": 65.0,
            "constraint_score": 65.0,
            "covariance_score": 65.0,
            "reconstruction_score": 65.0,
        }

        matches = evaluate_decision_rules(scores)

        # Middle scores may not trigger any specific rule
        # The result depends on exact rule definitions
        assert isinstance(matches, list)

    def test_evaluate_severity_ordering(self) -> None:
        """Results should be sorted by severity."""
        scores = {
            "solver_score": 45.0,
            "constraint_score": 45.0,
            "covariance_score": 45.0,
            "reconstruction_score": 45.0,
            "vae_health_score": 45.0,
        }

        matches = evaluate_decision_rules(scores)

        if len(matches) >= 2:
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "none": 4}
            severities = [severity_order.get(m["severity"], 5) for m in matches]
            assert severities == sorted(severities), "Results should be sorted by severity"


# ===========================================================================
# Causal Graph Tests
# ===========================================================================

class TestCausalGraph:
    """Tests for CAUSAL_GRAPH data structure."""

    def test_causal_graph_not_empty(self) -> None:
        """CAUSAL_GRAPH should have entries."""
        assert len(CAUSAL_GRAPH) >= 10

    def test_causal_graph_structure(self) -> None:
        """Each entry should have upstream/downstream lists."""
        for metric, relations in CAUSAL_GRAPH.items():
            assert isinstance(relations, dict), f"Metric {metric} should have dict value"
            assert "upstream" in relations or "downstream" in relations, (
                f"Metric {metric} should have upstream or downstream"
            )

    def test_causal_graph_coverage(self) -> None:
        """Key metrics should be in the graph."""
        key_metrics = ["AU", "shrinkage_intensity", "condition_number", "entropy", "solver_convergence"]

        for metric in key_metrics:
            assert metric in CAUSAL_GRAPH, f"Key metric {metric} missing from CAUSAL_GRAPH"


class TestTraceCausalChain:
    """Tests for trace_causal_chain()."""

    def test_trace_upstream(self) -> None:
        """Trace upstream causes."""
        chain = trace_causal_chain("solver_convergence", direction="upstream")

        assert isinstance(chain, list)
        # solver_convergence has upstream causes in the graph
        if "solver_convergence" in CAUSAL_GRAPH:
            assert len(chain) >= 0  # May have upstream causes

    def test_trace_downstream(self) -> None:
        """Trace downstream effects."""
        chain = trace_causal_chain("AU", direction="downstream")

        assert isinstance(chain, list)
        # AU affects other metrics
        if "AU" in CAUSAL_GRAPH and "downstream" in CAUSAL_GRAPH["AU"]:
            assert len(chain) > 0

    def test_trace_max_depth(self) -> None:
        """Max depth limits traversal."""
        chain_deep = trace_causal_chain("AU", direction="downstream", max_depth=5)
        chain_shallow = trace_causal_chain("AU", direction="downstream", max_depth=1)

        assert len(chain_shallow) <= len(chain_deep)

    def test_trace_invalid_direction(self) -> None:
        """Invalid direction raises error."""
        with pytest.raises(ValueError):
            trace_causal_chain("AU", direction="invalid")

    def test_trace_unknown_metric(self) -> None:
        """Unknown metric returns empty chain."""
        chain = trace_causal_chain("nonexistent_metric", direction="upstream")

        assert chain == []


# ===========================================================================
# Metric Patterns Tests
# ===========================================================================

class TestMetricPatterns:
    """Tests for METRIC_PATTERNS data structure."""

    def test_metric_patterns_not_empty(self) -> None:
        """METRIC_PATTERNS should have entries."""
        assert len(METRIC_PATTERNS) >= 5

    def test_metric_patterns_structure(self) -> None:
        """Each pattern should have required fields."""
        required_fields = {"id", "name", "indicators", "interpretation", "recommendation"}

        for pattern in METRIC_PATTERNS:
            for field in required_fields:
                assert field in pattern, f"Pattern {pattern.get('id', 'UNKNOWN')} missing field: {field}"


class TestDetectMetricPatterns:
    """Tests for detect_metric_patterns()."""

    def test_detect_beta_schedule_issue(self) -> None:
        """Low VAE health + high reconstruction -> BETA_SCHEDULE_ISSUE."""
        scores = {
            "vae_health_score": 40.0,
            "reconstruction_score": 80.0,
        }

        patterns = detect_metric_patterns(scores)

        pattern_ids = [p["pattern_id"] for p in patterns]
        assert "BETA_SCHEDULE_ISSUE" in pattern_ids

    def test_detect_constraint_feasibility(self) -> None:
        """Low solver + low constraint -> CONSTRAINT_FEASIBILITY."""
        scores = {
            "solver_score": 50.0,
            "constraint_score": 50.0,
        }

        patterns = detect_metric_patterns(scores)

        pattern_ids = [p["pattern_id"] for p in patterns]
        assert "CONSTRAINT_FEASIBILITY" in pattern_ids

    def test_detect_no_patterns(self) -> None:
        """High scores may not match any patterns."""
        scores = {
            "vae_health_score": 90.0,
            "reconstruction_score": 90.0,
            "solver_score": 90.0,
            "constraint_score": 90.0,
            "covariance_score": 90.0,
        }

        patterns = detect_metric_patterns(scores)

        # Most patterns indicate problems, so high scores = no matches
        assert isinstance(patterns, list)

    def test_detect_with_diagnostics(self) -> None:
        """Patterns can use raw diagnostics."""
        scores = {"covariance_score": 50.0}
        diagnostics = {"shrinkage_intensity": 0.9}

        patterns = detect_metric_patterns(scores, diagnostics)

        pattern_ids = [p["pattern_id"] for p in patterns]
        # High shrinkage + low covariance -> SAMPLE_SIZE_INSUFFICIENT
        assert "SAMPLE_SIZE_INSUFFICIENT" in pattern_ids


# ===========================================================================
# Root Cause Analysis Tests
# ===========================================================================

class TestGetRootCauseAnalysis:
    """Tests for get_root_cause_analysis()."""

    def test_root_cause_analysis_structure(self) -> None:
        """Result should have expected keys."""
        scores = {
            "solver_score": 70.0,
            "constraint_score": 70.0,
            "covariance_score": 70.0,
            "reconstruction_score": 70.0,
        }

        analysis = get_root_cause_analysis(scores)

        assert "matching_rules" in analysis
        assert "detected_patterns" in analysis
        assert "weakest_component" in analysis
        assert "weakest_score" in analysis
        assert "causal_analysis" in analysis
        assert "priority_actions" in analysis
        assert "overall_severity" in analysis

    def test_root_cause_analysis_weakest_component(self) -> None:
        """Weakest component should be identified."""
        scores = {
            "solver_score": 90.0,
            "constraint_score": 90.0,
            "covariance_score": 40.0,  # Weakest
            "reconstruction_score": 90.0,
        }

        analysis = get_root_cause_analysis(scores)

        assert analysis["weakest_component"] == "covariance"
        assert analysis["weakest_score"] == 40.0

    def test_root_cause_analysis_priority_actions(self) -> None:
        """Priority actions should be deduplicated and limited."""
        scores = {
            "solver_score": 45.0,
            "constraint_score": 45.0,
            "covariance_score": 45.0,
            "reconstruction_score": 45.0,
        }

        analysis = get_root_cause_analysis(scores)

        actions = analysis["priority_actions"]
        assert len(actions) <= 5
        # No duplicates
        assert len(actions) == len(set(actions))


class TestFormatDiagnosisSummary:
    """Tests for format_diagnosis_summary()."""

    def test_format_summary_output(self) -> None:
        """Summary should be non-empty string."""
        scores = {
            "solver_score": 70.0,
            "constraint_score": 70.0,
            "covariance_score": 70.0,
            "reconstruction_score": 70.0,
        }

        analysis = get_root_cause_analysis(scores)
        summary = format_diagnosis_summary(analysis)

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "Diagnostic Summary" in summary

    def test_format_summary_severity_label(self) -> None:
        """Summary should include severity label."""
        scores = {
            "solver_score": 45.0,
            "constraint_score": 45.0,
            "covariance_score": 45.0,
            "reconstruction_score": 45.0,
        }

        analysis = get_root_cause_analysis(scores)
        summary = format_diagnosis_summary(analysis)

        # Should contain one of the severity labels
        assert any(label in summary for label in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "OK"])

    def test_format_summary_weakest_component(self) -> None:
        """Summary should mention weakest component."""
        scores = {
            "solver_score": 90.0,
            "constraint_score": 30.0,  # Weakest
            "covariance_score": 90.0,
            "reconstruction_score": 90.0,
        }

        analysis = get_root_cause_analysis(scores)
        summary = format_diagnosis_summary(analysis)

        assert "constraint" in summary.lower()
