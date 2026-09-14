"""Short repository-local Study Result mutation coordination."""

from __future__ import annotations

import os
from contextlib import AbstractContextManager
from pathlib import Path

from mldb_v2.src.common.ids import StudyResultId, _validate_typed_reference
from mldb_v2.src.repository._process_lock import _process_file_lock


class StudyResultMutationCoordinator:
    """Serialize short canonical mutations for one exact Study Result ID."""

    def __init__(self, repository_root: str | os.PathLike[str]) -> None:
        self._repository_root = Path(repository_root)
        self._lock_root = (
            self._repository_root / ".local" / "mldb_v2" / "study_result_locks"
        )

    def acquire(
        self,
        *,
        study_result_id: StudyResultId,
    ) -> AbstractContextManager[None]:
        validated = _validate_typed_reference(study_result_id)
        return _process_file_lock(lock_root=self._lock_root, key=validated)
