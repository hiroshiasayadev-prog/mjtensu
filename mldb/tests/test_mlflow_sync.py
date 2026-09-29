from __future__ import annotations

from dataclasses import replace

import pytest

from tools.mldb.sync_mlflow import TrialProjection, sync_projections


class FakeMlflowClient:
    def __init__(self) -> None:
        self.experiments: dict[str, str] = {}
        self.runs: dict[str, list[dict]] = {}
        self.next_run = 1

    def get_or_create_experiment(self, name: str) -> tuple[str, bool]:
        if name in self.experiments:
            return self.experiments[name], False
        experiment_id = str(len(self.experiments) + 1)
        self.experiments[name] = experiment_id
        self.runs[experiment_id] = []
        return experiment_id, True

    def search_runs(self, experiment_id: str) -> list[dict]:
        return self.runs[experiment_id]

    def create_run(self, experiment_id: str, projection: TrialProjection) -> dict:
        run = {
            "info": {
                "run_id": f"r{self.next_run}",
                "status": "RUNNING",
                "run_name": projection.run_name,
            },
            "data": {
                "params": [],
                "metrics": [],
                "tags": [{"key": k, "value": v} for k, v in projection.tags.items()],
            },
        }
        self.next_run += 1
        self.runs[experiment_id].append(run)
        return run

    def log_batch(self, run_id: str, *, params, metrics, tags) -> None:
        run = self._run(run_id)
        for key, value in params.items():
            run["data"]["params"].append({"key": key, "value": value})
        for key, value in metrics.items():
            current = {item["key"]: item for item in run["data"]["metrics"]}
            current[key] = {"key": key, "value": value}
            run["data"]["metrics"] = list(current.values())
        current_tags = {item["key"]: item for item in run["data"]["tags"]}
        for key, value in tags.items():
            current_tags[key] = {"key": key, "value": value}
        run["data"]["tags"] = list(current_tags.values())

    def update_run(self, run_id: str, *, status: str, end_time_ms, run_name: str) -> None:
        run = self._run(run_id)
        run["info"]["status"] = status
        run["info"]["run_name"] = run_name
        if end_time_ms is not None:
            run["info"]["end_time"] = end_time_ms

    def _run(self, run_id: str) -> dict:
        for runs in self.runs.values():
            for run in runs:
                if run["info"]["run_id"] == run_id:
                    return run
        raise AssertionError(run_id)


def projection() -> TrialProjection:
    return TrialProjection(
        experiment_name="mldb/study-v1",
        sync_key="sr-20260909-001/trial-0001",
        run_name="sr-20260909-001/trial-0001",
        start_time_ms=1_000,
        end_time_ms=None,
        status="RUNNING",
        params={"architecture/stem_stride": "1", "train/seed": "42"},
        metrics={},
        tags={
            "mldb.sync_key": "sr-20260909-001/trial-0001",
            "mldb.study_run": "sr-20260909-001",
        },
    )


def test_sync_is_idempotent_and_updates_terminal_result() -> None:
    client = FakeMlflowClient()

    first = sync_projections(client, [projection()])
    assert first.experiments_created == 1
    assert first.runs_created == 1

    second = sync_projections(client, [projection()])
    assert second.runs_created == 0
    assert second.runs_updated == 0
    assert second.runs_unchanged == 1

    terminal = replace(
        projection(),
        status="FINISHED",
        end_time_ms=2_000,
        metrics={"eval/val/f1": 0.9},
        tags={**projection().tags, "mldb.study_status": "completed"},
    )
    third = sync_projections(client, [terminal])
    assert third.runs_updated == 1
    run = client.runs["1"][0]
    assert run["info"]["status"] == "FINISHED"
    assert run["info"]["end_time"] == 2_000
    assert run["data"]["metrics"] == [{"key": "eval/val/f1", "value": 0.9}]


def test_sync_rejects_conflicting_mlflow_param() -> None:
    client = FakeMlflowClient()
    sync_projections(client, [projection()])
    conflict = replace(
        projection(),
        params={"architecture/stem_stride": "2", "train/seed": "42"},
    )

    with pytest.raises(RuntimeError, match="MLflow param conflict"):
        sync_projections(client, [conflict])
