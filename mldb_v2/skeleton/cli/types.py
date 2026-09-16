"""MLDB v2 CLI public spellings and presentation-only selector shapes."""

from enum import Enum
from typing import NewType, TypedDict

from mldb_v2.skeleton.common.ids import NamespaceId, StudyId

CliStatusSelector = NewType("CliStatusSelector", str)
CliSinceSelector = NewType("CliSinceSelector", str)


class CliCommandName(str, Enum):
    PS = "ps"
    GET = "get"
    DESCRIBE = "describe"
    STATUS = "status"
    VALIDATE = "validate"
    VERIFY = "verify"
    SEAL = "seal"
    PLAN = "plan"
    RUN = "run"
    RESUME = "resume"
    RERUN = "rerun"
    CANCEL = "cancel"
    ADVANCE = "advance"
    WATCH = "watch"
    LOGS = "logs"
    DOCTOR = "doctor"


class CliResource(str, Enum):
    """Exact initial public resource spellings; kind resolution is semantic, never a path scan."""

    NAMESPACES = "namespaces"
    DEFINITIONS = "definitions"
    TASKS = "tasks"
    CORPORA = "corpora"
    ARCHITECTURES = "architectures"
    TRAIN_PROTOCOLS = "train-protocols"
    EVALUATION_PROTOCOLS = "evaluation-protocols"
    STUDIES = "studies"
    PLANS = "plans"
    RUNS = "runs"
    TRAINING_RESULTS = "training-results"
    MODELS = "models"
    EVALUATION_RESULTS = "evaluation-results"


class CliDefinitionKind(str, Enum):
    TASK = "task"
    CORPUS = "corpus"
    ARCHITECTURE = "architecture"
    TRAIN_PROTOCOL = "train-protocol"
    EVALUATION_PROTOCOL = "evaluation-protocol"
    STUDY = "study"


class OutputFormat(str, Enum):
    TABLE = "table"
    WIDE = "wide"
    JSON = "json"
    YAML = "yaml"


class CommonSelectors(TypedDict, total=False):
    """Normalized shared narrowing vocabulary; omitted keys mean no narrowing."""

    namespace: NamespaceId
    status: CliStatusSelector
    study: StudyId
    since: CliSinceSelector
    limit: int


class CliPresentation(TypedDict, total=False):
    """Normalized output selection.

    ``--json`` normalizes to ``format=json``. Structured output is undecorated and preserves API
    fields: JSON lists are arrays, single values are objects, and ordering is deterministic.
    """

    format: OutputFormat
