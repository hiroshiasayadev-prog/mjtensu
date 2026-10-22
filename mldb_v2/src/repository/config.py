"""Runtime selection for the formal execution-record store."""

from __future__ import annotations

import os
from functools import lru_cache
from collections.abc import Mapping

from .execution_store import ExecutionRecordStore
from .postgresql import PostgresExecutionRecordStore


_POSTGRES_DSN_ENV = "MLDB_POSTGRES_DSN"


@lru_cache(maxsize=4)
def _postgres_store(dsn: str) -> PostgresExecutionRecordStore:
    return PostgresExecutionRecordStore(dsn)


def configured_execution_store(
    environment: Mapping[str, str] | None = None,
) -> ExecutionRecordStore | None:
    env = os.environ if environment is None else environment
    dsn = env.get(_POSTGRES_DSN_ENV)
    if dsn is None or dsn == "":
        return None
    if type(dsn) is not str or dsn.strip() != dsn:
        raise ValueError(f"{_POSTGRES_DSN_ENV} must be a non-empty trimmed DSN")
    return _postgres_store(dsn)
