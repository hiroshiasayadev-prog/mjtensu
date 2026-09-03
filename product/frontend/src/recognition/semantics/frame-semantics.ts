import { groupMeldObservations } from './meld-grouping';
import type {
  ClassifiedRecognitionCandidate,
  FrameCommitEligibility,
  FrameObservationId,
  FrameRecognitionSnapshot,
  TileObservation,
} from './types';

const MINIMUM_VISIBLE_NON_DORA_TILES = 10;
const MINIMUM_COMPLETED_HAND_TILES = 2;

export function buildFrameRecognitionSnapshot(
  candidates: readonly ClassifiedRecognitionCandidate[],
): FrameRecognitionSnapshot {
  const observations = candidates.map(toObservation);
  const completedHand = recognizedTilesInRegion(observations, 'completed-hand');
  const doraIndicators = recognizedTilesInRegion(observations, 'dora-indicators');
  const meldObservations = observations.filter(
    (observation) =>
      observation.region === 'melds' && observation.classification.kind === 'tile',
  );
  const grouping = groupMeldObservations(meldObservations);
  const meldGroups = grouping.kind === 'stable' ? grouping.groups : [];
  const meldCommonAngleRadians =
    grouping.kind === 'stable' ? grouping.commonAngleRadians : null;
  const commitEligibility = determineEligibility(observations, grouping.kind);

  return {
    observations,
    meldGroups,
    meldCommonAngleRadians,
    draft: {
      completedHand,
      doraIndicators,
      meldGroups: meldGroups.map((group) => group.interpretation),
    },
    commitEligibility,
  };
}

function toObservation(
  candidate: ClassifiedRecognitionCandidate,
): TileObservation {
  return {
    id: candidate.id as FrameObservationId,
    region: candidate.region,
    bbox: { ...candidate.bbox },
    ...(candidate.obb === undefined ? {} : { obb: { ...candidate.obb } }),
    classification: candidate.classification,
  };
}

function recognizedTilesInRegion(
  observations: readonly TileObservation[],
  region: 'completed-hand' | 'dora-indicators',
) {
  return observations
    .filter(
      (observation) =>
        observation.region === region && observation.classification.kind === 'tile',
    )
    .sort(compareObservationPosition)
    .map((observation) => {
      if (observation.classification.kind !== 'tile') {
        throw new Error('Recognized ordering requires tile-classified observations');
      }
      return observation.classification.tile;
    });
}

function compareObservationPosition(
  left: TileObservation,
  right: TileObservation,
): number {
  const leftX = left.obb?.cx ?? left.bbox.x + left.bbox.width / 2;
  const rightX = right.obb?.cx ?? right.bbox.x + right.bbox.width / 2;
  if (leftX !== rightX) {
    return leftX - rightX;
  }
  const leftY = left.obb?.cy ?? left.bbox.y + left.bbox.height / 2;
  const rightY = right.obb?.cy ?? right.bbox.y + right.bbox.height / 2;
  if (leftY !== rightY) {
    return leftY - rightY;
  }
  return left.id.localeCompare(right.id);
}

function determineEligibility(
  observations: readonly TileObservation[],
  groupingKind: 'stable' | 'unstable',
): FrameCommitEligibility {
  if (groupingKind === 'unstable') {
    return {
      kind: 'ineligible',
      reason: 'unresolved-meld-geometry',
    };
  }

  let visibleNonDoraTiles = 0;
  let completedHandTiles = 0;
  for (const observation of observations) {
    if (observation.classification.kind !== 'tile') {
      continue;
    }
    if (observation.region === 'completed-hand') {
      completedHandTiles += 1;
      visibleNonDoraTiles += 1;
    } else if (observation.region === 'melds') {
      visibleNonDoraTiles += 1;
    }
  }

  if (
    visibleNonDoraTiles < MINIMUM_VISIBLE_NON_DORA_TILES ||
    completedHandTiles < MINIMUM_COMPLETED_HAND_TILES
  ) {
    return {
      kind: 'ineligible',
      reason: 'insufficient-visible-tiles',
    };
  }

  return { kind: 'eligible' };
}
