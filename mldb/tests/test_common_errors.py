from __future__ import annotations

import unittest

from mldb.src.common.errors import (
    InvalidRequestError,
    LifecycleConflictError,
    MldbError,
    NotFoundError,
    UnsupportedOperationError,
    ValidationFailedError,
    ValidationIssue,
    ValidationReport,
)


class CommonErrorsTests(unittest.TestCase):
    def test_validation_report_preserves_issue_order_and_validity(self) -> None:
        first = ValidationIssue("first", "first issue", "a")
        second = ValidationIssue("second", "second issue", "b")

        report = ValidationReport((first, second))

        self.assertFalse(report.valid)
        self.assertEqual((first, second), report.issues)
        self.assertTrue(ValidationReport().valid)

    def test_validation_failed_error_retains_report(self) -> None:
        report = ValidationReport((ValidationIssue("x", "bad"),))

        error = ValidationFailedError(report)

        self.assertIs(report, error.report)
        self.assertIsInstance(error, MldbError)

    def test_frozen_error_types_share_expected_base(self) -> None:
        for error_type in (
            NotFoundError,
            InvalidRequestError,
            ValidationFailedError,
            UnsupportedOperationError,
            LifecycleConflictError,
        ):
            self.assertTrue(issubclass(error_type, MldbError))


if __name__ == "__main__":
    unittest.main()
