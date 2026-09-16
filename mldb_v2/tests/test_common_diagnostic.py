import pytest

from mldb_v2.src.common.diagnostic import _validate_diagnostic


def test_diagnostic_accepts_valid_code_and_message() -> None:
    value = {"code": "execution_failed2", "message": "backend failed"}
    assert _validate_diagnostic(value) is value


@pytest.mark.parametrize(
    "code", ["Execution_failed", "execution-failed", "2failed", "failed_", "failed__now"]
)
def test_diagnostic_rejects_invalid_code(code: str) -> None:
    with pytest.raises(ValueError):
        _validate_diagnostic({"code": code, "message": "x"})


@pytest.mark.parametrize("length", [4095, 4096])
def test_diagnostic_accepts_message_at_or_below_byte_limit(length: int) -> None:
    value = {"code": "ok", "message": "a" * length}
    assert _validate_diagnostic(value) is value


def test_diagnostic_rejects_message_above_byte_limit() -> None:
    with pytest.raises(ValueError):
        _validate_diagnostic({"code": "failed", "message": "a" * 4097})


def test_diagnostic_uses_utf8_byte_length_for_multibyte_message() -> None:
    assert _validate_diagnostic({"code": "ok", "message": "牌" * 1365}) is not None
    with pytest.raises(ValueError):
        _validate_diagnostic({"code": "failed", "message": "牌" * 1366})


def test_diagnostic_accepts_null_persisted_value() -> None:
    assert _validate_diagnostic(None) is None


def test_diagnostic_rejects_extra_metadata() -> None:
    with pytest.raises(ValueError):
        _validate_diagnostic({"code": "failed", "message": "x", "stack": "secret"})
