from __future__ import annotations

import argparse
import hashlib
import pickle
import shlex
import subprocess
import sys
import tempfile
import threading
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mldb.src.orchestration import acquire, heartbeat, outcome
from mldb.src.orchestration._sqlite_queue import SQLiteQueue
from mldb.src.orchestration.candidates import CandidateStore
from mldb.src.orchestration.retry_policy import RetryDecision
from mldb.src.orchestration.worker_api import (
    AcquireRejection,
    AcquireWorkRequest,
    CandidateArtifactRef,
    CandidateUploadAcknowledgement,
    CandidateUploadRequest,
    EvaluationAssignment,
    EvaluationSuccessCandidate,
    EvaluationSucceeded,
    HeartbeatAccepted,
    HeartbeatRequest,
    ImmutableAssetDescriptor,
    NoWork,
    AttemptFailed,
    TrainingAssignment,
    TrainingSuccessCandidate,
    TrainingSucceeded,
)
from mldb.src.orchestration.worker_data import handle_candidate_upload
from mldb.src.repository._local_filesystem import LocalFilesystem
from mldb.src.repository.layout import RepositoryLayout



def _now_queue() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _now_runtime() -> datetime:
    return datetime.now(timezone.utc)


class _NoRetryPolicy:
    def after_preflight_failure(self, job, prior_attempts, *, at: str) -> RetryDecision:
        return RetryDecision(None)

    def after_unsatisfied_attempt(self, job, attempt_history, outcome_name, *, at: str) -> RetryDecision:
        return RetryDecision(None)


class _LocalAssetSource:
    def describe(self, source_path: Path, *, key: str, sha256: str, bytes: int | None):
        return ImmutableAssetDescriptor(
            key=key,
            retrieval_id=str(source_path.resolve()),
            sha256=sha256,
            bytes=bytes,
        )
    def retrieve(self, asset: ImmutableAssetDescriptor) -> bytes:
        content = Path(asset.retrieval_id).read_bytes()
        if hashlib.sha256(content).hexdigest() != asset.sha256:
            raise ValueError(f"Asset hash changed after assignment: {asset.key}")
        if asset.bytes is not None and len(content) != asset.bytes:
            raise ValueError(f"Asset byte count changed after assignment: {asset.key}")
        return content

    @staticmethod
    def path_for(asset: ImmutableAssetDescriptor) -> Path:
        return Path(asset.retrieval_id)


class _FileCandidateStore(CandidateStore):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, attempt_id: int, key: str) -> tuple[Path, Path]:
        directory = self.root / str(attempt_id)
        directory.mkdir(parents=True, exist_ok=True)
        safe = quote(key, safe="")
        return directory / f"{safe}.identity", directory / f"{safe}.bin"

    def put(self, attempt_id: int, key: str, content_identity: str, content: bytes):
        identity_path, content_path = self._paths(attempt_id, key)
        if identity_path.exists() or content_path.exists():
            if not identity_path.exists() or not content_path.exists():
                return CandidateUploadAcknowledgement.REJECTED
            if identity_path.read_text(encoding="utf-8") != content_identity:
                return CandidateUploadAcknowledgement.REJECTED
            if content_path.read_bytes() != content:
                return CandidateUploadAcknowledgement.REJECTED
            return CandidateUploadAcknowledgement.ALREADY_PRESENT
        content_path.write_bytes(content)
        identity_path.write_text(content_identity, encoding="utf-8")
        return CandidateUploadAcknowledgement.ACCEPTED
    def read(self, attempt_id: int, key: str, content_identity: str) -> bytes:
        identity_path, content_path = self._paths(attempt_id, key)
        if identity_path.read_text(encoding="utf-8") != content_identity:
            raise ValueError("Candidate identity mismatch")
        return content_path.read_bytes()


def _descriptors(assignment: TrainingAssignment | EvaluationAssignment):
    if isinstance(assignment, TrainingAssignment):
        return (
            assignment.corpus_artifact,
            assignment.architecture_implementation,
            assignment.train_protocol_implementation,
        )
    return (
        assignment.corpus_artifact,
        assignment.model_architecture_implementation,
        assignment.model_weights,
        assignment.evaluation_protocol_implementation,
    )


def _run(command: list[str]) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in command), flush=True)
    subprocess.run(command, check=True)


def _remote_shell(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _copy_runtime(repo_root: Path, host: str, remote_run: str) -> None:
    _run(["ssh", host, f"mkdir -p {shlex.quote(remote_run)}/package/mldb {shlex.quote(remote_run)}/assets {shlex.quote(remote_run)}/result"])
    _run(["scp", str(repo_root / "mldb" / "__init__.py"), f"{host}:{remote_run}/package/mldb/"])
    _run(["scp", "-r", str(repo_root / "mldb" / "src"), f"{host}:{remote_run}/package/mldb/"])
    _run(["scp", str(repo_root / "tools" / "mldb" / "remote_attempt.py"), f"{host}:{remote_run}/"])


def _ensure_remote_asset(
    source: Path,
    descriptor: ImmutableAssetDescriptor,
    *,
    host: str,
    remote_root: str,
    remote_run: str,
) -> None:
    cache_dir = f"{remote_root.rstrip('/')}/cache/assets/{descriptor.sha256[:2]}"
    cache_path = f"{cache_dir}/{descriptor.sha256}"
    run_name = descriptor.key.replace("/", "__")
    if descriptor.key in {
        "architecture_implementation",
        "train_protocol_implementation",
        "model_architecture_implementation",
        "evaluation_protocol_implementation",
    }:
        run_name += ".py"
    run_path = f"{remote_run}/assets/{run_name}"
    size_check = ""
    if descriptor.bytes is not None:
        size_check = f" && test $(wc -c < {shlex.quote(cache_path)}) -eq {descriptor.bytes}"
    check = (
        f"test -f {shlex.quote(cache_path)} "
        f"&& test $(sha256sum {shlex.quote(cache_path)} | cut -d' ' -f1) = {shlex.quote(descriptor.sha256)}"
        f"{size_check}"
    )
    hit = subprocess.run(["ssh", host, check], check=False).returncode == 0
    if hit:
        print(f"cache hit: {descriptor.key} {descriptor.sha256[:12]}", flush=True)
    else:
        print(f"cache miss: {descriptor.key} {descriptor.sha256[:12]}", flush=True)
        _run(["ssh", host, f"mkdir -p {shlex.quote(cache_dir)}"])
        temporary = f"{cache_path}.upload-{uuid.uuid4().hex}"
        _run(["scp", str(source), f"{host}:{temporary}"])
        verify = (
            f"test $(sha256sum {shlex.quote(temporary)} | cut -d' ' -f1) = {shlex.quote(descriptor.sha256)}"
        )
        if descriptor.bytes is not None:
            verify += f" && test $(wc -c < {shlex.quote(temporary)}) -eq {descriptor.bytes}"
        verify += f" && mv {shlex.quote(temporary)} {shlex.quote(cache_path)}"
        _run(["ssh", host, verify])
    _run(["ssh", host, f"ln -f {shlex.quote(cache_path)} {shlex.quote(run_path)}"])


def _heartbeat_loop(
    assignment: TrainingAssignment | EvaluationAssignment,
    interval_seconds: float,
    stop: threading.Event,
    failures: list[BaseException],
    layout: RepositoryLayout,
    filesystem: LocalFilesystem,
    queue: SQLiteQueue,
) -> None:
    request = HeartbeatRequest(assignment.attempt_id, assignment.lease_token)
    while not stop.wait(interval_seconds):
        try:
            response = heartbeat.handle_heartbeat(
                request,
                _now_queue(),
                layout,
                filesystem,
                queue,
            )
            if not isinstance(response, HeartbeatAccepted):
                failures.append(RuntimeError(f"heartbeat rejected: {response}"))
                return
            if response.cancel_requested:
                failures.append(RuntimeError("attempt cancellation requested"))
                return
        except BaseException as error:
            failures.append(error)
            return


def _execute_remote(
    assignment: TrainingAssignment | EvaluationAssignment,
    assets: _LocalAssetSource,
    repo_root: Path,
    host: str,
    remote_root: str,
    remote_python: str,
) -> dict[str, object]:
    remote_run = f"{remote_root.rstrip('/')}/attempt-{assignment.attempt_id}"
    with tempfile.TemporaryDirectory(prefix=f"mldb-attempt-{assignment.attempt_id}-") as temporary:
        local = Path(temporary)
        assignment_path = local / "assignment.pkl"
        with assignment_path.open("wb") as stream:
            pickle.dump(assignment, stream, protocol=pickle.HIGHEST_PROTOCOL)
        _copy_runtime(repo_root, host, remote_run)
        _run(["scp", str(assignment_path), f"{host}:{remote_run}/assignment.pkl"])
        for descriptor in _descriptors(assignment):
            _ensure_remote_asset(
                assets.path_for(descriptor),
                descriptor,
                host=host,
                remote_root=remote_root,
                remote_run=remote_run,
            )

        remote_command = [
            remote_python,
            f"{remote_run}/remote_attempt.py",
            "--package-root", f"{remote_run}/package",
            "--assignment", f"{remote_run}/assignment.pkl",
            "--assets-dir", f"{remote_run}/assets",
            "--result-dir", f"{remote_run}/result",
        ]
        shell = (
            f"PYTHONPATH={shlex.quote(remote_run + '/package')} "
            + _remote_shell(remote_command)
        )
        _run(["ssh", host, shell])

        local_result = local / "result"
        local_result.mkdir()
        _run(["scp", "-r", f"{host}:{remote_run}/result/.", str(local_result)])
        with (local_result / "result.pkl").open("rb") as stream:
            payload = pickle.load(stream)
        if payload["kind"] == "training":
            payload["weights_bytes"] = (local_result / str(payload["weights_file"])).read_bytes()
        else:
            payload["artifact_bytes"] = {
                key: (local_result / relative).read_bytes()
                for key, relative in payload["artifact_files"].items()
            }
        return payload


def _upload_candidate(
    assignment: TrainingAssignment | EvaluationAssignment,
    key: str,
    content: bytes,
    queue: SQLiteQueue,
    candidates: _FileCandidateStore,
) -> CandidateArtifactRef:
    digest = hashlib.sha256(content).hexdigest()
    acknowledgement = handle_candidate_upload(
        CandidateUploadRequest(
            attempt_id=assignment.attempt_id,
            lease_token=assignment.lease_token,
            key=key,
            content_identity=digest,
            content=content,
        ),
        _now_queue(),
        queue,
        candidates,
    )
    if acknowledgement not in {
        CandidateUploadAcknowledgement.ACCEPTED,
        CandidateUploadAcknowledgement.ALREADY_PRESENT,
    }:
        raise RuntimeError(f"candidate upload rejected for {key!r}")
    return CandidateArtifactRef(key=key, content_identity=digest, bytes=len(content))


def _report_success(
    assignment: TrainingAssignment | EvaluationAssignment,
    payload: dict[str, object],
    layout: RepositoryLayout,
    filesystem: LocalFilesystem,
    queue: SQLiteQueue,
    candidates: _FileCandidateStore,
    retry_policy: _NoRetryPolicy,
):
    if isinstance(assignment, TrainingAssignment):
        artifact = payload["artifact"]
        content = payload["weights_bytes"]
        ref = _upload_candidate(assignment, artifact.path, content, queue, candidates)
        reported = TrainingSucceeded(
            assignment.attempt_id,
            assignment.lease_token,
            TrainingSuccessCandidate(ref),
        )
    else:
        refs = {
            key: _upload_candidate(assignment, key, content, queue, candidates)
            for key, content in payload["artifact_bytes"].items()
        }
        reported = EvaluationSucceeded(
            assignment.attempt_id,
            assignment.lease_token,
            EvaluationSuccessCandidate(
                metrics=payload["metrics"],
                artifacts=refs,
                unavailable_outputs=payload["unavailable_outputs"],
            ),
        )
    return outcome.handle_attempt_outcome(
        reported,
        _now_runtime(),
        _now_queue(),
        layout,
        filesystem,
        queue,
        candidates,
        retry_policy,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull one MLDB job locally and execute only its domain work over SSH."
    )
    parser.add_argument("--host", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--remote-root", default=".cache/mjtensu-mldb-worker")
    parser.add_argument("--remote-python", default="/srv/bugrat/data-lv/mjtensu/nanodet/nanodet/.venv/bin/python")
    parser.add_argument("--worker-id")
    parser.add_argument("--heartbeat-seconds", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    worker_id = args.worker_id or f"ssh:{args.host}"

    # Prove transport works before allocating a child Run / Queue attempt.
    _run(["ssh", args.host, "true"])

    layout = RepositoryLayout(repo_root)
    filesystem = LocalFilesystem()
    queue = SQLiteQueue(repo_root)
    assets = _LocalAssetSource()
    candidates = _FileCandidateStore(repo_root / ".local" / "mldb" / "candidates")
    retry_policy = _NoRetryPolicy()

    response = acquire.handle_acquire_work(
        AcquireWorkRequest(
            worker_id=worker_id,
            acquire_token=uuid.uuid4().hex,
            accepts=frozenset({"training", "evaluation"}),
        ),
        date.today(),
        _now_runtime(),
        _now_queue(),
        uuid.uuid4().hex,
        layout,
        filesystem,
        queue,
        assets,
        retry_policy,
    )
    if isinstance(response, NoWork):
        print("no_work")
        return
    if isinstance(response, AcquireRejection):
        raise RuntimeError(f"acquire rejected: {response.type}: {response.message}")
    if not isinstance(response, (TrainingAssignment, EvaluationAssignment)):
        raise TypeError(f"unexpected acquire response: {type(response)!r}")

    assignment = response
    stop = threading.Event()
    heartbeat_failures: list[BaseException] = []
    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(
            assignment,
            float(args.heartbeat_seconds),
            stop,
            heartbeat_failures,
            layout,
            filesystem,
            queue,
        ),
        daemon=True,
        name=f"mldb-ssh-heartbeat-{assignment.attempt_id}",
    )
    heartbeat_thread.start()
    try:
        try:
            payload = _execute_remote(
                assignment,
                assets,
                repo_root,
                args.host,
                args.remote_root,
                args.remote_python,
            )
        except Exception as error:
            acknowledgement = outcome.handle_attempt_outcome(
                AttemptFailed(
                    assignment.attempt_id,
                    assignment.lease_token,
                    type(error).__name__,
                    str(error),
                ),
                _now_runtime(),
                _now_queue(),
                layout,
                filesystem,
                queue,
                candidates,
                retry_policy,
            )
            print(f"remote attempt failed: {acknowledgement}")
            raise
    finally:
        stop.set()
        heartbeat_thread.join(timeout=max(5.0, float(args.heartbeat_seconds) + 1.0))

    if heartbeat_failures:
        raise RuntimeError(f"heartbeat failed: {heartbeat_failures[0]}") from heartbeat_failures[0]

    acknowledgement = _report_success(
        assignment,
        payload,
        layout,
        filesystem,
        queue,
        candidates,
        retry_policy,
    )
    print(f"completed attempt={assignment.attempt_id} run={assignment.run_id} ack={acknowledgement}")


if __name__ == "__main__":
    main()
