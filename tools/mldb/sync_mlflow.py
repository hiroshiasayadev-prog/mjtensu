from __future__ import annotations

import argparse
import base64
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mldb.src.api.controller import get_entity, list_entities
from mldb.src.catalog.architecture import Architecture
from mldb.src.common.ids import EntityKind
from mldb.src.evaluation.run import EvaluationRun, EvaluationRunStatus
from mldb.src.model.identity import Model, model_id_for_training_run
from mldb.src.repository._local_filesystem import LocalFilesystem
from mldb.src.repository.layout import RepositoryLayout
from mldb.src.study.run import StudyRun, StudyRunStatus
from mldb.src.training.run import TrainingRun, TrainingRunStatus


@dataclass(frozen=True, slots=True)
class TrialProjection:
    experiment_name: str
    sync_key: str
    run_name: str
    start_time_ms: int
    end_time_ms: int | None
    status: str
    params: Mapping[str, str]
    metrics: Mapping[str, float]
    tags: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class SyncStats:
    experiments_created: int = 0
    runs_created: int = 0
    runs_updated: int = 0
    runs_unchanged: int = 0


class MlflowApiError(RuntimeError):
    pass


class MlflowRestClient:
    def __init__(
        self,
        tracking_uri: str,
        *,
        username: str | None = None,
        password: str | None = None,
        verify_tls: bool = True,
    ) -> None:
        self.tracking_uri = tracking_uri.rstrip("/")
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0 Safari/537.36",
        }
        if username is not None:
            if password is None:
                raise ValueError("MLflow password is required when username is supplied")
            token = base64.b64encode(f"{username}:{password}".encode()).decode()
            self._headers["Authorization"] = f"Basic {token}"
        self._ssl_context = None if verify_tls else ssl._create_unverified_context()

    def get_or_create_experiment(self, name: str) -> tuple[str, bool]:
        try:
            payload = self._get(
                "/api/2.0/mlflow/experiments/get-by-name",
                {"experiment_name": name},
            )
            return str(payload["experiment"]["experiment_id"]), False
        except MlflowApiError as exc:
            if "RESOURCE_DOES_NOT_EXIST" not in str(exc):
                raise
        payload = self._post("/api/2.0/mlflow/experiments/create", {"name": name})
        return str(payload["experiment_id"]), True

    def search_runs(self, experiment_id: str) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            body: dict[str, Any] = {
                "experiment_ids": [experiment_id],
                "run_view_type": "ALL",
                "max_results": 50000,
            }
            if page_token:
                body["page_token"] = page_token
            payload = self._post("/api/2.0/mlflow/runs/search", body)
            runs.extend(payload.get("runs", []))
            page_token = payload.get("next_page_token")
            if not page_token:
                return runs

    def create_run(self, experiment_id: str, projection: TrialProjection) -> dict[str, Any]:
        body = {
            "experiment_id": experiment_id,
            "run_name": projection.run_name,
            "start_time": projection.start_time_ms,
            "tags": [{"key": k, "value": v} for k, v in projection.tags.items()],
        }
        return self._post("/api/2.0/mlflow/runs/create", body)["run"]

    def log_batch(
        self,
        run_id: str,
        *,
        params: Mapping[str, str],
        metrics: Mapping[str, float],
        tags: Mapping[str, str],
    ) -> None:
        now_ms = int(time.time() * 1000)
        body = {
            "run_id": run_id,
            "params": [{"key": k, "value": v} for k, v in params.items()],
            "metrics": [
                {"key": k, "value": v, "timestamp": now_ms, "step": 0}
                for k, v in metrics.items()
            ],
            "tags": [{"key": k, "value": v} for k, v in tags.items()],
        }
        if body["params"] or body["metrics"] or body["tags"]:
            self._post("/api/2.0/mlflow/runs/log-batch", body)

    def update_run(
        self,
        run_id: str,
        *,
        status: str,
        end_time_ms: int | None,
        run_name: str,
    ) -> None:
        body: dict[str, Any] = {
            "run_id": run_id,
            "status": status,
            "run_name": run_name,
        }
        if end_time_ms is not None:
            body["end_time"] = end_time_ms
        self._post("/api/2.0/mlflow/runs/update", body)

    def _get(self, path: str, query: Mapping[str, str]) -> dict[str, Any]:
        url = f"{self.tracking_uri}{path}?{urllib.parse.urlencode(query)}"
        return self._request(url, method="GET", body=None)

    def _post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        url = f"{self.tracking_uri}{path}"
        data = json.dumps(body, separators=(",", ":"), allow_nan=False).encode()
        return self._request(url, method="POST", body=data)

    def _request(self, url: str, *, method: str, body: bytes | None) -> dict[str, Any]:
        request = urllib.request.Request(url, data=body, headers=self._headers, method=method)
        try:
            with urllib.request.urlopen(request, context=self._ssl_context, timeout=30) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise MlflowApiError(f"MLflow {exc.code} {exc.reason}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise MlflowApiError(f"MLflow request failed: {exc.reason}") from exc
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MlflowApiError("MLflow returned a non-JSON response") from exc


def collect_projections(
    repo_root: Path,
    *,
    study_id: str | None = None,
    study_run_id: str | None = None,
) -> tuple[TrialProjection, ...]:
    layout = RepositoryLayout(repo_root.resolve())
    filesystem = LocalFilesystem()

    study_runs = _read_all(EntityKind.STUDY_RUN, layout, filesystem)
    training_runs = _read_all(EntityKind.TRAINING_RUN, layout, filesystem)
    evaluation_runs = _read_all(EntityKind.EVALUATION_RUN, layout, filesystem)

    selected_study_runs: dict[str, StudyRun] = {}
    for value in study_runs:
        assert isinstance(value, StudyRun)
        if study_id is not None and str(value.study) != study_id:
            continue
        if study_run_id is not None and str(value.id) != study_run_id:
            continue
        selected_study_runs[str(value.id)] = value

    training_by_trial: dict[tuple[str, str], list[TrainingRun]] = defaultdict(list)
    for value in training_runs:
        assert isinstance(value, TrainingRun)
        if value.study is None or str(value.study.run) not in selected_study_runs:
            continue
        training_by_trial[(str(value.study.run), value.study.trial)].append(value)

    evaluation_by_trial: dict[tuple[str, str], list[EvaluationRun]] = defaultdict(list)
    for value in evaluation_runs:
        assert isinstance(value, EvaluationRun)
        if value.study is None or str(value.study.run) not in selected_study_runs:
            continue
        evaluation_by_trial[(str(value.study.run), value.study.trial)].append(value)

    keys = sorted(set(training_by_trial) | set(evaluation_by_trial))
    projections: list[TrialProjection] = []
    study_cache: dict[str, Any] = {}
    architecture_cache: dict[str, Architecture] = {}
    model_cache: dict[str, Model] = {}
    training_cache: dict[str, TrainingRun] = {}

    for key in keys:
        sr_id, trial = key
        study_run = selected_study_runs[sr_id]
        study_key = str(study_run.study)
        study = study_cache.get(study_key)
        if study is None:
            study = get_entity(EntityKind.STUDY, study_key, layout, filesystem)
            study_cache[study_key] = study

        attempts = training_by_trial.get(key, [])
        evaluations = evaluation_by_trial.get(key, [])
        training = _select_training_attempt(attempts)
        model = _resolve_trial_model(
            training,
            evaluations,
            layout,
            filesystem,
            model_cache,
        )
        lineage_training = training
        if lineage_training is None and model is not None:
            tr_id = str(model.training_run)
            lineage_training = training_cache.get(tr_id)
            if lineage_training is None:
                value = get_entity(EntityKind.TRAINING_RUN, tr_id, layout, filesystem)
                assert isinstance(value, TrainingRun)
                training_cache[tr_id] = value
                lineage_training = value

        architecture = None
        if lineage_training is not None:
            architecture_id = str(lineage_training.architecture)
            architecture = architecture_cache.get(architecture_id)
            if architecture is None:
                value = get_entity(EntityKind.ARCHITECTURE, architecture_id, layout, filesystem)
                assert isinstance(value, Architecture)
                architecture_cache[architecture_id] = value
                architecture = value

        projections.append(
            _project_trial(
                study_run=study_run,
                study=study,
                trial=trial,
                training=lineage_training,
                training_attempts=attempts,
                model=model,
                architecture=architecture,
                evaluations=evaluations,
            )
        )

    return tuple(projections)


def _read_all(kind: EntityKind, layout: RepositoryLayout, filesystem: LocalFilesystem) -> tuple[Any, ...]:
    return tuple(
        get_entity(kind, summary.id, layout, filesystem)
        for summary in list_entities(kind, layout, filesystem)
    )


def _select_training_attempt(attempts: Iterable[TrainingRun]) -> TrainingRun | None:
    items = list(attempts)
    if not items:
        return None
    completed = [
        item
        for item in items
        if item.status is TrainingRunStatus.COMPLETED and item.result is not None
    ]
    pool = completed or items
    return max(pool, key=lambda item: (str(item.execution.started_at), str(item.id)))


def _select_evaluation_attempt(attempts: Iterable[EvaluationRun]) -> EvaluationRun:
    items = list(attempts)
    accepted = [
        item
        for item in items
        if item.status in {EvaluationRunStatus.COMPLETED, EvaluationRunStatus.COMPLETED_PARTIAL}
        and item.result is not None
    ]
    pool = accepted or items
    return max(pool, key=lambda item: (str(item.execution.started_at), str(item.id)))


def _resolve_trial_model(
    training: TrainingRun | None,
    evaluations: Iterable[EvaluationRun],
    layout: RepositoryLayout,
    filesystem: LocalFilesystem,
    cache: dict[str, Model],
) -> Model | None:
    model_ids = {str(item.model) for item in evaluations}
    if len(model_ids) > 1:
        raise RuntimeError(f"one Study trial references multiple Models: {sorted(model_ids)}")
    model_id = next(iter(model_ids), None)
    if model_id is None and training is not None:
        if training.status is TrainingRunStatus.COMPLETED and training.result is not None:
            model_id = str(model_id_for_training_run(training.id))
    if model_id is None:
        return None
    cached = cache.get(model_id)
    if cached is not None:
        return cached
    value = get_entity(EntityKind.MODEL, model_id, layout, filesystem)
    assert isinstance(value, Model)
    cache[model_id] = value
    return value


def _project_trial(
    *,
    study_run: StudyRun,
    study: Any,
    trial: str,
    training: TrainingRun | None,
    training_attempts: Iterable[TrainingRun],
    model: Model | None,
    architecture: Architecture | None,
    evaluations: Iterable[EvaluationRun],
) -> TrialProjection:
    evals_by_stage: dict[str, list[EvaluationRun]] = defaultdict(list)
    for item in evaluations:
        assert item.study is not None
        evals_by_stage[item.study.stage].append(item)
    selected_evals = {
        stage: _select_evaluation_attempt(items)
        for stage, items in sorted(evals_by_stage.items())
    }

    params: dict[str, str] = {}
    metrics: dict[str, float] = {}
    tags: dict[str, str] = {
        "mldb.sync_key": f"{study_run.id}/{trial}",
        "mldb.study": str(study_run.study),
        "mldb.study_run": str(study_run.id),
        "mldb.study_status": study_run.status.value,
        "mldb.trial": trial,
        "mldb.study_name": str(study.name),
    }

    if architecture is not None:
        params["architecture/id"] = str(architecture.id)
        params["architecture/family"] = architecture.family
        params["architecture/input"] = _param_value(dict(architecture.interface.input))
        if isinstance(architecture.parameters, Mapping):
            for name, value in architecture.parameters.items():
                params[f"architecture/{name}"] = _param_value(value)

    attempts = list(training_attempts)
    tags["mldb.training_attempts"] = str(len(attempts))
    if training is not None:
        tags["mldb.training_run"] = str(training.id)
        tags["mldb.training_status"] = training.status.value
        params["train/corpus"] = str(training.corpus)
        params["train/protocol"] = str(training.train_protocol)
        params["train/seed"] = str(training.execution.seed)
        for name, value in training.parameters.items():
            params[f"train/{name}"] = _param_value(value)

    if model is not None:
        tags["mldb.model"] = str(model.id)

    for stage, evaluation in selected_evals.items():
        prefix = f"eval/{stage}"
        tags[f"mldb.eval.{stage}.run"] = str(evaluation.id)
        tags[f"mldb.eval.{stage}.status"] = evaluation.status.value
        tags[f"mldb.eval.{stage}.attempts"] = str(len(evals_by_stage[stage]))
        params[f"{prefix}/corpus"] = str(evaluation.corpus)
        params[f"{prefix}/protocol"] = str(evaluation.evaluation_protocol)
        for name, value in evaluation.parameters.items():
            params[f"{prefix}/{name}"] = _param_value(value)
        if evaluation.result is not None:
            for name, value in evaluation.result.metrics.items():
                metrics[f"{prefix}/{name}"] = float(value)

    start_candidates = [study_run.execution.started_at]
    end_candidates: list[object] = []
    if training is not None:
        start_candidates.append(training.execution.started_at)
        if training.execution.finished_at is not None:
            end_candidates.append(training.execution.finished_at)
    for evaluation in selected_evals.values():
        start_candidates.append(evaluation.execution.started_at)
        if evaluation.execution.finished_at is not None:
            end_candidates.append(evaluation.execution.finished_at)
    if study_run.execution.finished_at is not None:
        end_candidates.append(study_run.execution.finished_at)

    status = _mlflow_status(study_run, training, selected_evals.values())
    end_time_ms = max((_timestamp_ms(v) for v in end_candidates), default=None)
    if status == "RUNNING":
        end_time_ms = None

    return TrialProjection(
        experiment_name=f"mldb/{study_run.study}",
        sync_key=f"{study_run.id}/{trial}",
        run_name=f"{study_run.id}/{trial}",
        start_time_ms=min(_timestamp_ms(v) for v in start_candidates),
        end_time_ms=end_time_ms,
        status=status,
        params=dict(sorted(params.items())),
        metrics=dict(sorted(metrics.items())),
        tags=dict(sorted(tags.items())),
    )


def _mlflow_status(
    study_run: StudyRun,
    training: TrainingRun | None,
    evaluations: Iterable[EvaluationRun],
) -> str:
    evals = list(evaluations)
    if training is not None and training.status is TrainingRunStatus.CANCELLED:
        return "KILLED"
    if any(item.status is EvaluationRunStatus.CANCELLED for item in evals):
        return "KILLED"
    if training is not None and training.status is TrainingRunStatus.FAILED:
        return "FAILED"
    if any(item.status is EvaluationRunStatus.FAILED for item in evals):
        return "FAILED"
    if study_run.status is StudyRunStatus.CANCELLED:
        return "KILLED"
    if study_run.status is StudyRunStatus.FAILED:
        return "FAILED"
    if study_run.status in {
        StudyRunStatus.COMPLETED,
        StudyRunStatus.COMPLETED_WITH_FAILURES,
    }:
        return "FINISHED"
    return "RUNNING"


def _param_value(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _timestamp_ms(value: object) -> int:
    if isinstance(value, datetime):
        return int(value.timestamp() * 1000)
    if isinstance(value, str):
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
    raise TypeError(f"unsupported timestamp value: {value!r}")


def sync_projections(
    client: MlflowRestClient,
    projections: Iterable[TrialProjection],
) -> SyncStats:
    stats = SyncStats()
    by_experiment: dict[str, list[TrialProjection]] = defaultdict(list)
    for projection in projections:
        by_experiment[projection.experiment_name].append(projection)

    for experiment_name, items in sorted(by_experiment.items()):
        experiment_id, created_experiment = client.get_or_create_experiment(experiment_name)
        if created_experiment:
            stats = _bump(stats, experiments_created=1)
        existing_runs = client.search_runs(experiment_id)
        by_sync_key = {
            _run_tags(run).get("mldb.sync_key"): run
            for run in existing_runs
            if _run_tags(run).get("mldb.sync_key")
        }

        for projection in sorted(items, key=lambda item: item.sync_key):
            existing = by_sync_key.get(projection.sync_key)
            created_run = existing is None
            if created_run:
                existing = client.create_run(experiment_id, projection)
                stats = _bump(stats, runs_created=1)
            assert existing is not None
            run_id = str(existing["info"]["run_id"])
            old_params = _run_params(existing)
            old_metrics = _run_metrics(existing)
            old_tags = _run_tags(existing)

            params = _missing_params(old_params, projection.params, projection.sync_key)
            metrics = {
                key: value
                for key, value in projection.metrics.items()
                if key not in old_metrics or float(old_metrics[key]) != value
            }
            tags = {
                key: value
                for key, value in projection.tags.items()
                if old_tags.get(key) != value
            }
            client.log_batch(run_id, params=params, metrics=metrics, tags=tags)

            old_status = str(existing["info"].get("status", "RUNNING"))
            old_name = str(existing["info"].get("run_name") or "")
            needs_status_update = old_status != projection.status or old_name != projection.run_name
            if needs_status_update:
                client.update_run(
                    run_id,
                    status=projection.status,
                    end_time_ms=projection.end_time_ms,
                    run_name=projection.run_name,
                )
            changed = created_run or bool(params or metrics or tags or needs_status_update)
            if not created_run and changed:
                stats = _bump(stats, runs_updated=1)
            elif not created_run:
                stats = _bump(stats, runs_unchanged=1)

    return stats


def _missing_params(
    existing: Mapping[str, str],
    desired: Mapping[str, str],
    sync_key: str,
) -> dict[str, str]:
    missing: dict[str, str] = {}
    for key, value in desired.items():
        if key not in existing:
            missing[key] = value
        elif existing[key] != value:
            raise RuntimeError(
                f"MLflow param conflict for {sync_key}: {key} is {existing[key]!r}, "
                f"MLDB requires {value!r}"
            )
    return missing


def _run_params(run: Mapping[str, Any]) -> dict[str, str]:
    return {str(item["key"]): str(item["value"]) for item in run.get("data", {}).get("params", [])}


def _run_metrics(run: Mapping[str, Any]) -> dict[str, float]:
    return {str(item["key"]): float(item["value"]) for item in run.get("data", {}).get("metrics", [])}


def _run_tags(run: Mapping[str, Any]) -> dict[str, str]:
    return {str(item["key"]): str(item["value"]) for item in run.get("data", {}).get("tags", [])}


def _bump(stats: SyncStats, **changes: int) -> SyncStats:
    return SyncStats(
        experiments_created=stats.experiments_created + changes.get("experiments_created", 0),
        runs_created=stats.runs_created + changes.get("runs_created", 0),
        runs_updated=stats.runs_updated + changes.get("runs_updated", 0),
        runs_unchanged=stats.runs_unchanged + changes.get("runs_unchanged", 0),
    )


def _projection_json(projection: TrialProjection) -> str:
    return json.dumps(
        {
            "experiment": projection.experiment_name,
            "sync_key": projection.sync_key,
            "run_name": projection.run_name,
            "status": projection.status,
            "params": projection.params,
            "metrics": projection.metrics,
            "tags": projection.tags,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mirror canonical MLDB Study trial params and metrics into MLflow Tracking."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--study-id")
    parser.add_argument("--study-run-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--insecure-tls",
        action="store_true",
        help="Disable TLS certificate verification for the MLflow request.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    projections = collect_projections(
        args.repo_root,
        study_id=args.study_id,
        study_run_id=args.study_run_id,
    )

    if args.dry_run:
        for projection in projections:
            print(_projection_json(projection))
        print(f"projections={len(projections)}", file=sys.stderr)
        return

    if not args.tracking_uri:
        raise SystemExit("--tracking-uri or MLFLOW_TRACKING_URI is required unless --dry-run is used")

    client = MlflowRestClient(
        args.tracking_uri,
        username=os.environ.get("MLFLOW_TRACKING_USERNAME"),
        password=os.environ.get("MLFLOW_TRACKING_PASSWORD"),
        verify_tls=not args.insecure_tls,
    )
    stats = sync_projections(client, projections)
    print(
        " ".join(
            (
                f"projections={len(projections)}",
                f"experiments_created={stats.experiments_created}",
                f"runs_created={stats.runs_created}",
                f"runs_updated={stats.runs_updated}",
                f"runs_unchanged={stats.runs_unchanged}",
            )
        )
    )


if __name__ == "__main__":
    main()
