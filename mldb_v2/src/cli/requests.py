"""Normalized MLDB v2 CLI command request shapes.

These are presentation/adapter requests, not parser objects and not replacements for Application
API domain response types.
"""

from typing import Literal, NotRequired, TypeAlias, TypedDict

from mldb_v2.src.cli.types import (
    CliCommandName,
    CliDefinitionKind,
    CliPresentation,
    CliResource,
    CommonSelectors,
)
from mldb_v2.src.common.ids import (
    EvaluationCoordinateId,
    NamespaceId,
    StudyId,
    StudyResultId,
    TrialId,
    TypedEntityId,
)
from mldb_v2.src.verification.definition_lifecycle import SealableDefinitionId


class PsRequest(TypedDict):
    command: Literal[CliCommandName.PS]
    all: bool
    selectors: NotRequired[CommonSelectors]
    presentation: NotRequired[CliPresentation]


class GetRequest(TypedDict):
    command: Literal[CliCommandName.GET]
    resource: CliResource
    typed_id: NotRequired[NamespaceId | TypedEntityId]
    selectors: NotRequired[CommonSelectors]
    presentation: NotRequired[CliPresentation]


class DescribeRequest(TypedDict):
    command: Literal[CliCommandName.DESCRIBE]
    resource: CliResource
    typed_id: NamespaceId | TypedEntityId
    presentation: NotRequired[CliPresentation]


class StatusRequest(TypedDict):
    command: Literal[CliCommandName.STATUS]
    study_result: NotRequired[StudyResultId]
    presentation: NotRequired[CliPresentation]


class ValidateRequest(TypedDict):
    command: Literal[CliCommandName.VALIDATE]
    kind: NotRequired[CliDefinitionKind]
    typed_id: NotRequired[SealableDefinitionId]
    namespace: NotRequired[NamespaceId]
    fail_fast: bool
    presentation: NotRequired[CliPresentation]


class VerifyRequest(TypedDict):
    command: Literal[CliCommandName.VERIFY]
    kind: NotRequired[CliDefinitionKind]
    typed_id: NotRequired[SealableDefinitionId]
    namespace: NotRequired[NamespaceId]
    fail_fast: bool
    presentation: NotRequired[CliPresentation]


class SealExactRequest(TypedDict):
    command: Literal[CliCommandName.SEAL]
    kind: CliDefinitionKind
    typed_id: SealableDefinitionId


class SealBulkRequest(TypedDict):
    command: Literal[CliCommandName.SEAL]
    kind: NotRequired[CliDefinitionKind]
    namespace: NamespaceId
    all: Literal[True]


SealRequest: TypeAlias = SealExactRequest | SealBulkRequest


class PlanRequest(TypedDict):
    command: Literal[CliCommandName.PLAN]
    study: StudyId


class RunRequest(TypedDict):
    command: Literal[CliCommandName.RUN]
    study: StudyId
    backend: NotRequired[str]


class ResumeRequest(TypedDict):
    command: Literal[CliCommandName.RESUME]
    study_result: StudyResultId


class RerunRequest(TypedDict):
    command: Literal[CliCommandName.RERUN]
    study_result: StudyResultId
    backend: NotRequired[str]


class CancelRequest(TypedDict):
    command: Literal[CliCommandName.CANCEL]
    study_result: StudyResultId


class AdvanceRequest(TypedDict):
    command: Literal[CliCommandName.ADVANCE]
    study_result: StudyResultId


class WatchRequest(TypedDict):
    command: Literal[CliCommandName.WATCH]
    study_result: NotRequired[StudyResultId]
    selectors: NotRequired[CommonSelectors]
    presentation: NotRequired[CliPresentation]


class LogsRequest(TypedDict):
    command: Literal[CliCommandName.LOGS]
    study_result: StudyResultId
    trial: NotRequired[TrialId]
    stage: NotRequired[EvaluationCoordinateId]
    failed_only: bool
    follow: bool
    presentation: NotRequired[CliPresentation]


class DoctorRequest(TypedDict):
    command: Literal[CliCommandName.DOCTOR]
    presentation: NotRequired[CliPresentation]


DiscoverInspectRequest: TypeAlias = PsRequest | GetRequest | DescribeRequest | StatusRequest
AuthoringRequest: TypeAlias = ValidateRequest | VerifyRequest | SealRequest | PlanRequest
ExecutionRequest: TypeAlias = RunRequest | ResumeRequest | RerunRequest | CancelRequest | AdvanceRequest
MonitorRequest: TypeAlias = WatchRequest | LogsRequest | DoctorRequest
CliCommandRequest: TypeAlias = (
    DiscoverInspectRequest | AuthoringRequest | ExecutionRequest | MonitorRequest
)
