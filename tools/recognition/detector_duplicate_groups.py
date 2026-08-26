from __future__ import annotations

import hashlib
import sqlite3
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_DUPLICATE_OVERLAP_THRESHOLD = 0.80


@dataclass(frozen=True)
class DetectorCandidate:
    candidate_id: str
    capture_id: str
    region: str
    detection_index: int
    confidence: float
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float

    @property
    def rect(self) -> tuple[float, float, float, float]:
        return (self.bbox_x, self.bbox_y, self.bbox_width, self.bbox_height)


@dataclass(frozen=True)
class DuplicateMember:
    candidate: DetectorCandidate
    max_overlap_to_cluster: float


@dataclass(frozen=True)
class DuplicateCluster:
    cluster_id: str
    capture_id: str
    region: str
    winner: DetectorCandidate
    losers: tuple[DuplicateMember, ...]

    @property
    def members(self) -> tuple[DetectorCandidate, ...]:
        return (self.winner,) + tuple(member.candidate for member in self.losers)


@dataclass(frozen=True)
class DuplicatePlan:
    threshold: float
    winner_candidate_ids: frozenset[str]
    loser_candidate_ids: frozenset[str]
    clusters: tuple[DuplicateCluster, ...]

    @property
    def classifier_candidate_ids(self) -> frozenset[str]:
        return self.winner_candidate_ids


def rect_min_area_overlap(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    ax, ay, aw, ah = (float(value) for value in left)
    bx, by, bw, bh = (float(value) for value in right)
    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    minimum_area = min(area_a, area_b)
    if minimum_area <= 0.0:
        return 0.0
    intersection_width = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    intersection_height = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    return (intersection_width * intersection_height) / minimum_area


def build_duplicate_plan(
    candidates: Iterable[DetectorCandidate],
    *,
    threshold: float = DEFAULT_DUPLICATE_OVERLAP_THRESHOLD,
) -> DuplicatePlan:
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("duplicate overlap threshold must be in [0,1]")

    grouped: dict[tuple[str, str], list[DetectorCandidate]] = defaultdict(list)
    all_candidates: list[DetectorCandidate] = []
    seen_ids: set[str] = set()
    for candidate in candidates:
        if candidate.candidate_id in seen_ids:
            raise ValueError(f"duplicate candidate_id: {candidate.candidate_id}")
        seen_ids.add(candidate.candidate_id)
        all_candidates.append(candidate)
        grouped[(candidate.capture_id, candidate.region)].append(candidate)

    winners: set[str] = set()
    losers: set[str] = set()
    clusters: list[DuplicateCluster] = []

    for (capture_id, region), group in sorted(grouped.items()):
        group = sorted(group, key=lambda item: (item.detection_index, item.candidate_id))
        adjacency: list[set[int]] = [set() for _ in group]
        overlaps: dict[tuple[int, int], float] = {}

        for left_index, left in enumerate(group):
            for right_index in range(left_index + 1, len(group)):
                right = group[right_index]
                overlap = rect_min_area_overlap(left.rect, right.rect)
                if overlap < threshold:
                    continue
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)
                overlaps[(left_index, right_index)] = overlap

        visited: set[int] = set()
        for start in range(len(group)):
            if start in visited:
                continue
            stack = [start]
            visited.add(start)
            component_indices: list[int] = []
            while stack:
                current = stack.pop()
                component_indices.append(current)
                for neighbor in adjacency[current]:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    stack.append(neighbor)

            component_indices.sort()
            component = [group[index] for index in component_indices]
            winner = min(
                component,
                key=lambda item: (-item.confidence, item.detection_index, item.candidate_id),
            )
            winners.add(winner.candidate_id)

            if len(component) == 1:
                continue

            loser_members: list[DuplicateMember] = []
            for index in component_indices:
                candidate = group[index]
                if candidate.candidate_id == winner.candidate_id:
                    continue
                max_overlap = max(
                    (
                        overlaps[tuple(sorted((index, other_index)))]
                        for other_index in component_indices
                        if other_index != index
                        and tuple(sorted((index, other_index))) in overlaps
                    ),
                    default=0.0,
                )
                losers.add(candidate.candidate_id)
                loser_members.append(
                    DuplicateMember(
                        candidate=candidate,
                        max_overlap_to_cluster=max_overlap,
                    )
                )

            loser_members.sort(
                key=lambda item: (
                    -item.candidate.confidence,
                    item.candidate.detection_index,
                    item.candidate.candidate_id,
                )
            )
            member_ids = sorted(item.candidate_id for item in component)
            cluster_id = stable_cluster_id(
                capture_id=capture_id,
                region=region,
                threshold=threshold,
                member_candidate_ids=member_ids,
            )
            clusters.append(
                DuplicateCluster(
                    cluster_id=cluster_id,
                    capture_id=capture_id,
                    region=region,
                    winner=winner,
                    losers=tuple(loser_members),
                )
            )

    if winners & losers:
        overlap_ids = sorted(winners & losers)
        raise AssertionError(f"candidate cannot be both winner and loser: {overlap_ids[:5]}")
    if winners | losers != seen_ids:
        missing = sorted(seen_ids - (winners | losers))
        raise AssertionError(f"duplicate plan did not classify all candidates: {missing[:5]}")

    clusters.sort(
        key=lambda item: (
            item.capture_id,
            item.region,
            item.winner.detection_index,
            item.cluster_id,
        )
    )
    return DuplicatePlan(
        threshold=threshold,
        winner_candidate_ids=frozenset(winners),
        loser_candidate_ids=frozenset(losers),
        clusters=tuple(clusters),
    )


def stable_cluster_id(
    *,
    capture_id: str,
    region: str,
    threshold: float,
    member_candidate_ids: Sequence[str],
) -> str:
    payload = "\0".join(
        [capture_id, region, f"{threshold:.12g}", *sorted(member_candidate_ids)]
    )
    return "dup:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def load_detector_candidates(database: Path) -> tuple[DetectorCandidate, ...]:
    database = database.resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    with closing(
        sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=60)
    ) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT candidate_id, capture_id, region, detection_index, detection_confidence,
                   bbox_x, bbox_y, bbox_width, bbox_height
            FROM candidate
            ORDER BY capture_id, region, detection_index, candidate_id
            """
        ).fetchall()
    return tuple(
        DetectorCandidate(
            candidate_id=str(row["candidate_id"]),
            capture_id=str(row["capture_id"]),
            region=str(row["region"]),
            detection_index=int(row["detection_index"]),
            confidence=float(row["detection_confidence"]),
            bbox_x=float(row["bbox_x"]),
            bbox_y=float(row["bbox_y"]),
            bbox_width=float(row["bbox_width"]),
            bbox_height=float(row["bbox_height"]),
        )
        for row in rows
    )


def load_duplicate_threshold(
    database: Path,
    *,
    default: float = DEFAULT_DUPLICATE_OVERLAP_THRESHOLD,
) -> float:
    database = database.resolve()
    with closing(
        sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=30)
    ) as connection:
        row = connection.execute(
            "SELECT value FROM dataset_metadata WHERE key='duplicate_overlap_threshold'"
        ).fetchone()
    return float(default if row is None else row[0])


def load_duplicate_plan(
    database: Path,
    *,
    threshold: float | None = None,
) -> DuplicatePlan:
    resolved_threshold = (
        load_duplicate_threshold(database) if threshold is None else float(threshold)
    )
    return build_duplicate_plan(
        load_detector_candidates(database),
        threshold=resolved_threshold,
    )
