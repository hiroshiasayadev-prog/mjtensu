"""Short repository-local Study Result mutation coordination boundary."""

from contextlib import AbstractContextManager
from typing import Protocol

from mldb_v2.skeleton.common.ids import StudyResultId


class StudyResultMutationCoordinator(Protocol):
    """Serialize only short canonical mutations for one exact Study Result ID."""

    def acquire(
        self,
        *,
        study_result_id: StudyResultId,
    ) -> AbstractContextManager[None]: ...
