from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mldb_v2.src.backend._clearml_sdk import _project_evaluation_artifacts


class _ObjectBytes:
    def __init__(self, objects: dict[str, bytes], *, fail_uri: str | None = None) -> None:
        self.objects = objects
        self.fail_uri = fail_uri

    def read_verified(self, ref: dict[str, object]) -> bytes:
        uri = str(ref["uri"])
        if uri == self.fail_uri:
            raise OSError("projection read failed")
        data = self.objects[uri]
        assert len(data) == ref["bytes"]
        assert hashlib.sha256(data).hexdigest() == ref["sha256"]
        return data


class _Logger:
    def __init__(self) -> None:
        self.images: list[dict[str, object]] = []
        self.tables: list[dict[str, object]] = []
        self.plots: list[dict[str, object]] = []
        self.media: list[dict[str, object]] = []

    def report_image(self, **kwargs: object) -> None:
        self.images.append(dict(kwargs))

    def report_table(self, **kwargs: object) -> None:
        self.tables.append(dict(kwargs))

    def report_plotly(self, **kwargs: object) -> None:
        self.plots.append(dict(kwargs))

    def report_media(self, **kwargs: object) -> None:
        self.media.append(dict(kwargs))


class _Artifact:
    def __init__(self, url: str) -> None:
        self.url = url


class _Task:
    def __init__(self) -> None:
        self.logger = _Logger()
        self.uploads: list[dict[str, object]] = []
        self.artifacts: dict[str, _Artifact] = {}

    def get_logger(self) -> _Logger:
        return self.logger

    def upload_artifact(self, **kwargs: object) -> bool:
        self.uploads.append(dict(kwargs))
        name = str(kwargs["name"])
        local_path = Path(str(kwargs["artifact_object"]))
        self.artifacts[name] = _Artifact(
            f"https://clearml-files.example.test/task/artifacts/{local_path.name}"
        )
        return True


def _ref(uri: str, data: bytes, artifact_format: str) -> dict[str, object]:
    return {
        "uri": uri,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "format": artifact_format,
        "schema": f"demo/{artifact_format}/v1",
    }


def _candidate(artifacts: dict[str, dict[str, object]]) -> dict[str, object]:
    return {"status": "completed", "result": {"metrics": {}, "artifacts": artifacts}}

def test_evaluation_artifacts_project_to_clearml_rich_surfaces(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CLEARML_WEB_HOST", "https://clearml.example.test")
    monkeypatch.setenv("CLEARML_FILES_HOST", "https://clearml-files.example.test")
    png = b"\x89PNG\r\n\x1a\nfixture"
    csv_data = b"case,accuracy\nfront,1.0\n"
    plot = json.dumps({"data": [{"type": "bar", "x": ["front"], "y": [1.0]}]}).encode()
    report = b'{"schema":"demo/report/v1"}\n'
    video = b"\x00\x00\x00\x18ftypmp42fixture"
    objects = {
        "s3://bucket/contact.png": png,
        "s3://bucket/summary.csv": csv_data,
        "s3://bucket/robustness.plotly.json": plot,
        "s3://bucket/report.json": report,
        "s3://bucket/overlay.mp4": video,
    }
    artifacts = {
        "contact_sheet": _ref("s3://bucket/contact.png", png, "png"),
        "summary_table": _ref("s3://bucket/summary.csv", csv_data, "csv"),
        "robustness_plot": _ref("s3://bucket/robustness.plotly.json", plot, "plotly-json"),
        "report_json": _ref("s3://bucket/report.json", report, "json"),
        "overlay_video": _ref("s3://bucket/overlay.mp4", video, "mp4"),
    }
    task = _Task()

    _project_evaluation_artifacts(
        task=task,
        candidate=_candidate(artifacts),
        object_bytes=_ObjectBytes(objects),
        projection_root=tmp_path,
    )

    assert [call["name"] for call in task.uploads] == [
        "evaluation/contact_sheet", "evaluation/summary_table",
        "evaluation/robustness_plot", "evaluation/report_json",
        "evaluation/overlay_video",
    ]
    assert [call["series"] for call in task.logger.images] == ["contact_sheet"]
    assert [call["series"] for call in task.logger.tables] == ["summary_table"]
    assert [call["series"] for call in task.logger.plots] == ["robustness_plot"]
    assert [call["series"] for call in task.logger.media] == ["overlay_video"]
    assert task.logger.plots[0]["figure"] == json.loads(plot)
    assert [call["wait_on_upload"] for call in task.uploads] == [False, False, False, False, True]
    assert task.logger.media[0]["title"] == "evaluation media"
    assert task.logger.media[0]["url"] == (
        "https://clearml.example.test/files/task/artifacts/overlay_video.mp4"
    )
    assert "local_path" not in task.logger.media[0]
    assert (tmp_path / "contact_sheet.png").read_bytes() == png
    assert (tmp_path / "summary_table.csv").read_bytes() == csv_data
    assert (tmp_path / "overlay_video.mp4").read_bytes() == video


def test_projection_failure_isolated_from_other_artifacts(tmp_path: Path) -> None:
    broken = b"broken"
    good = b"a,b\n1,2\n"
    broken_uri = "s3://bucket/broken.json"
    good_uri = "s3://bucket/good.csv"
    task = _Task()

    _project_evaluation_artifacts(
        task=task,
        candidate=_candidate({
            "broken": _ref(broken_uri, broken, "json"),
            "good": _ref(good_uri, good, "csv"),
        }),
        object_bytes=_ObjectBytes({broken_uri: broken, good_uri: good}, fail_uri=broken_uri),
        projection_root=tmp_path,
    )

    assert [call["name"] for call in task.uploads] == ["evaluation/good"]
    assert [call["series"] for call in task.logger.tables] == ["good"]
