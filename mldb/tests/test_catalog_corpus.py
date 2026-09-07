from __future__ import annotations

import unittest

from mldb.src.catalog.corpus import (
    Corpus,
    CorpusArtifact,
    CorpusDataSpec,
    validate_corpus,
)
from mldb.src.common.ids import CorpusId, TaskId


class CatalogCorpusTests(unittest.TestCase):
    def test_valid_static_metadata_passes_without_runtime_resolution(self) -> None:
        corpus = self._corpus()

        report = validate_corpus(corpus)

        self.assertTrue(report.valid)

    def test_static_metadata_shape_errors_are_reported_in_order(self) -> None:
        corpus = self._corpus(
            schema="bad-schema",
            artifact=CorpusArtifact(format="other", sha256="xyz", bytes=True),
            data=CorpusDataSpec(schema="", table=""),
            splits={"": -1, "train": True},
        )

        report = validate_corpus(corpus)

        self.assertEqual(
            [
                "schema",
                "artifact.format",
                "artifact.sha256",
                "artifact.bytes",
                "data.schema",
                "data.table",
                "splits",
                "splits['']",
                "splits['train']",
            ],
            [issue.path for issue in report.issues],
        )

    def test_empty_split_mapping_is_valid_metadata(self) -> None:
        corpus = self._corpus(splits={})

        self.assertTrue(validate_corpus(corpus).valid)

    def _corpus(
        self,
        *,
        schema: str = "mjtensu.mldb/corpus/v1",
        artifact: CorpusArtifact | None = None,
        data: CorpusDataSpec | None = None,
        splits: dict[str, int] | None = None,
    ) -> Corpus:
        return Corpus(
            schema=schema,
            id=CorpusId("test-corpus-v1"),
            task=TaskId("test-task-v1"),
            artifact=artifact
            or CorpusArtifact(format="sqlite", sha256="a" * 64, bytes=123),
            data=data
            or CorpusDataSpec(
                schema="mjtensu.mldb/image-classification-corpus/v1",
                table="sample",
            ),
            representation={"kind": "image"},
            builder_parameters={"seed": 42},
            splits={"train": 10, "val": 2} if splits is None else splits,
            description="metadata only",
        )


if __name__ == "__main__":
    unittest.main()
