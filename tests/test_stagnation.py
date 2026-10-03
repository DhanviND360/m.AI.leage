"""Unit tests for the Stagnation and Loop Detection engine."""

import pytest
from mileage.agents.stagnation import (
    StagnationDetector,
    StagnationReason,
    StagnationReport,
)


class TestStagnationDetector:
    """Test StagnationDetector thresholds and halt conditions."""

    def test_repeated_error_threshold(self):
        detector = StagnationDetector(repeated_error_threshold=3)

        err_msg = "AssertionError: expected 42 but got 0 in test_calc.py"

        # 1st error
        rep1 = detector.record_error(err_msg)
        assert rep1 is None or not rep1.is_stagnant
        assert detector.consecutive_repeated_errors == 1

        # 2nd consecutive error
        rep2 = detector.record_error(err_msg)
        assert rep2 is None or not rep2.is_stagnant
        assert detector.consecutive_repeated_errors == 2

        # 3rd consecutive error -> must trigger stagnation
        rep3 = detector.record_error(err_msg)
        assert rep3 is not None
        assert rep3.is_stagnant is True
        assert rep3.reason == StagnationReason.REPEATED_ERRORS
        assert rep3.current_value == 3
        assert "Same error repeated 3 times" in rep3.message

    def test_different_error_resets_repeated_count(self):
        detector = StagnationDetector(repeated_error_threshold=3)

        detector.record_error("TypeError: cannot add str to int")
        detector.record_error("TypeError: cannot add str to int")
        assert detector.consecutive_repeated_errors == 2

        # Different error occurs
        detector.record_error("IndexError: list index out of range")
        assert detector.consecutive_repeated_errors == 1
        assert not detector.check().is_stagnant

    def test_unchanged_edits_threshold(self):
        detector = StagnationDetector(unchanged_edits_threshold=2)

        # 1st edit with zero changes
        rep1 = detector.record_edit("src/calc.py", is_changed=False)
        assert rep1 is None or not rep1.is_stagnant
        assert detector.consecutive_unchanged_edits == 1

        # 2nd edit with zero changes -> must trigger stagnation
        rep2 = detector.record_edit("src/calc.py", is_changed=False)
        assert rep2 is not None
        assert rep2.is_stagnant is True
        assert rep2.reason == StagnationReason.UNCHANGED_EDITS
        assert "0 changes" in rep2.message

    def test_successful_edit_resets_unchanged_count(self):
        detector = StagnationDetector(unchanged_edits_threshold=2)

        detector.record_edit("src/calc.py", is_changed=False)
        assert detector.consecutive_unchanged_edits == 1

        detector.record_edit("src/calc.py", is_changed=True, content="def foo(): pass")
        assert detector.consecutive_unchanged_edits == 0
        assert not detector.check().is_stagnant

    def test_max_retries_threshold(self):
        detector = StagnationDetector(max_iterations=3)

        # Iteration 1
        detector.record_iteration_start()
        detector.record_edit("src/calc.py", is_changed=True)
        rep1 = detector.record_iteration_end(tests_passed=False)
        assert rep1 is None or not rep1.is_stagnant

        # Iteration 2
        detector.record_iteration_start()
        detector.record_edit("src/calc.py", is_changed=True)
        rep2 = detector.record_iteration_end(tests_passed=False)
        assert rep2 is None or not rep2.is_stagnant

        # Iteration 3 (reaches max_iterations=3 without tests passing)
        detector.record_iteration_start()
        detector.record_edit("src/calc.py", is_changed=True)
        rep3 = detector.record_iteration_end(tests_passed=False)
        assert rep3 is not None
        assert rep3.is_stagnant is True
        assert rep3.reason == StagnationReason.MAX_RETRIES

    def test_stalled_execution_threshold(self):
        detector = StagnationDetector(stalled_threshold=2)

        # Iteration 1: no edits, tests fail
        detector.record_iteration_start()
        rep1 = detector.record_iteration_end(tests_passed=False)
        assert rep1 is None or not rep1.is_stagnant

        # Iteration 2: no edits, tests fail -> stalled threshold reached
        detector.record_iteration_start()
        rep2 = detector.record_iteration_end(tests_passed=False)
        assert rep2 is not None
        assert rep2.is_stagnant is True
        assert rep2.reason == StagnationReason.STALLED_EXECUTION

    def test_oscillation_detection(self):
        detector = StagnationDetector()

        # State A
        detector.record_edit("src/calc.py", is_changed=True, content="state_A")
        # State B
        detector.record_edit("src/calc.py", is_changed=True, content="state_B")
        # State A again! -> oscillation
        rep = detector.record_edit("src/calc.py", is_changed=True, content="state_A")
        assert rep is not None
        assert rep.is_stagnant is True
        assert rep.reason == StagnationReason.OSCILLATION

    def test_reset(self):
        detector = StagnationDetector(repeated_error_threshold=2)
        detector.record_error("Error 1")
        detector.record_error("Error 1")
        assert detector.check().is_stagnant

        detector.reset()
        assert not detector.check().is_stagnant
        assert detector.current_iteration == 0
        assert detector.consecutive_repeated_errors == 0
