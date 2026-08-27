from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


TARGET_MODEL_SET_VERSION = "recognition-v2-2026-08-28"
OLD_DETECTOR_ARTIFACT = "nanodet-plus-m-320-composite-augmented.onnx"
OLD_DETECTOR_SHA256 = "4768daa5cb44e7bee37fbb69c36063800164d9e9e8c852e5b3c77bc88ce9ac76"
NEW_DETECTOR_ARTIFACT = "nanodet-plus-m-320-real-capture-ft10-l10.onnx"
SOURCE_RUN = ".local/recognition/nanodet_runs/E1_plus_m_320_real_capture_ft10_l10_seed42/model_best"
RUNTIME_SPEC = "nanodet-plus-m-320-v1"
INPUT_SHAPE = [1, 3, 320, 320]
OUTPUT_SHAPE = [1, 2125, 33]


def main() -> int:
    repository_root = Path(__file__).resolve().parents[3]
    source_path = repository_root / SOURCE_RUN / NEW_DETECTOR_ARTIFACT
    vendor_directory = repository_root / "vendor" / "recognition-models"
    destination_path = vendor_directory / NEW_DETECTOR_ARTIFACT
    old_destination_path = vendor_directory / OLD_DETECTOR_ARTIFACT
    model_set_path = (
        repository_root
        / "product"
        / "frontend"
        / "src"
        / "recognition"
        / "model-runtime"
        / "production-model-set.json"
    )
    provenance_path = vendor_directory / "provenance.json"

    for required in (source_path, model_set_path, provenance_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    model_set = read_json(model_set_path)
    provenance = read_json(provenance_path)
    validate_current_state(model_set, provenance, old_destination_path)

    source_sha256, source_bytes = file_identity(source_path)
    vendor_directory.mkdir(parents=True, exist_ok=True)

    temporary_path = vendor_directory / f".{NEW_DETECTOR_ARTIFACT}.tmp"
    if temporary_path.exists():
        temporary_path.unlink()
    shutil.copy2(source_path, temporary_path)
    copied_sha256, copied_bytes = file_identity(temporary_path)
    if (copied_sha256, copied_bytes) != (source_sha256, source_bytes):
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("Copied detector artifact does not match the selected source artifact.")
    os.replace(temporary_path, destination_path)

    detector = model_set["models"]["detector"]
    detector["url"] = f"{NEW_DETECTOR_ARTIFACT}?sha256={source_sha256}"
    detector["sha256"] = source_sha256
    model_set["modelSetVersion"] = TARGET_MODEL_SET_VERSION

    provenance_detector = provenance["models"]["detector"]
    provenance_detector.update(
        {
            "artifact": NEW_DETECTOR_ARTIFACT,
            "sourceRun": SOURCE_RUN,
            "sha256": source_sha256,
            "bytes": source_bytes,
            "runtimeSpec": RUNTIME_SPEC,
            "inputShape": INPUT_SHAPE,
            "outputShape": OUTPUT_SHAPE,
            "selectionEvidence": {
                "runtimeThreshold": 0.35,
                "realVal": {
                    "productionBaseline": {
                        "truePositives": 124,
                        "falsePositives": 3,
                        "falseNegatives": 4,
                        "f1": 0.9725,
                        "meldF1": 0.7273,
                    },
                    "selectedFineTune": {
                        "truePositives": 128,
                        "falsePositives": 3,
                        "falseNegatives": 0,
                        "f1": 0.9884,
                        "meldF1": 0.96,
                    },
                },
                "compositeVal": {
                    "productionBaseline": {
                        "truePositives": 1044,
                        "falsePositives": 28,
                        "falseNegatives": 8,
                        "f1": 0.9831,
                        "meldF1": 0.9968,
                    },
                    "selectedFineTune": {
                        "truePositives": 1045,
                        "falsePositives": 23,
                        "falseNegatives": 7,
                        "f1": 0.9858,
                        "meldF1": 0.9968,
                    },
                },
            },
        }
    )
    provenance["modelSetVersion"] = TARGET_MODEL_SET_VERSION

    write_json_atomic(model_set_path, model_set)
    write_json_atomic(provenance_path, provenance)

    if old_destination_path.is_file() and old_destination_path != destination_path:
        old_sha256, _old_bytes = file_identity(old_destination_path)
        if old_sha256 != OLD_DETECTOR_SHA256:
            raise RuntimeError(
                "Refusing to remove the previous detector artifact because its SHA-256 no longer "
                f"matches the known production baseline: {old_sha256}"
            )
        old_destination_path.unlink()

    print("promoted production detector")
    print(f"  model_set_version: {TARGET_MODEL_SET_VERSION}")
    print(f"  source: {source_path}")
    print(f"  artifact: {destination_path}")
    print(f"  sha256: {source_sha256}")
    print(f"  bytes: {source_bytes}")
    print(f"  manifest: {model_set_path}")
    print(f"  provenance: {provenance_path}")
    return 0


def validate_current_state(
    model_set: dict[str, Any],
    provenance: dict[str, Any],
    old_destination_path: Path,
) -> None:
    if model_set.get("schemaVersion") != 1 or provenance.get("schemaVersion") != 1:
        raise RuntimeError("Unexpected production model schema version.")

    detector = model_set.get("models", {}).get("detector")
    provenance_detector = provenance.get("models", {}).get("detector")
    if not isinstance(detector, dict) or not isinstance(provenance_detector, dict):
        raise RuntimeError("Production detector declaration is missing.")

    current_artifact = str(detector.get("url", "")).split("?", 1)[0]
    current_sha256 = detector.get("sha256")
    already_promoted = (
        model_set.get("modelSetVersion") == TARGET_MODEL_SET_VERSION
        and current_artifact == NEW_DETECTOR_ARTIFACT
    )
    if already_promoted:
        return

    if current_artifact != OLD_DETECTOR_ARTIFACT or current_sha256 != OLD_DETECTOR_SHA256:
        raise RuntimeError(
            "Refusing promotion from an unexpected detector state: "
            f"artifact={current_artifact!r}, sha256={current_sha256!r}"
        )
    if not old_destination_path.is_file():
        raise FileNotFoundError(old_destination_path)
    old_sha256, _old_bytes = file_identity(old_destination_path)
    if old_sha256 != OLD_DETECTOR_SHA256:
        raise RuntimeError(
            "Existing vendored production detector does not match the recorded baseline SHA-256."
        )


def file_identity(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    return digest.hexdigest(), byte_count


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
