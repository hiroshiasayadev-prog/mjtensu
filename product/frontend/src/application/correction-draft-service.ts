import type {
  RecognizedMeldGroup,
  RecognizedStructure,
  TileIdentity,
  TileInstance,
  TileInstanceId,
  TileKind,
} from '@/domain';
import type { ScoringService, WinningStructureIssue } from '@/scoring';

export type CorrectionMeldGroupId = string & {
  readonly __brand: 'CorrectionMeldGroupId';
};

export interface CorrectionDraft {
  readonly completedHand: readonly TileInstance[];
  readonly meldGroups: readonly CorrectionMeldGroupDraft[];
  readonly doraIndicators: readonly TileInstance[];
}

export interface CorrectionMeldGroupDraft {
  readonly id: CorrectionMeldGroupId;
  readonly tiles: readonly TileInstance[];
  readonly kanOpenness: 'open' | 'concealed' | null;
}

export type CorrectionDestination =
  | {
      readonly kind: 'completed-hand';
    }
  | {
      readonly kind: 'dora-indicators';
    }
  | {
      readonly kind: 'meld';
      readonly groupId: CorrectionMeldGroupId;
    };

export type CorrectionCommand =
  | {
      readonly kind: 'add-tile';
      readonly destination: CorrectionDestination;
      readonly tile: TileIdentity;
      readonly index?: number;
    }
  | {
      readonly kind: 'replace-tile';
      readonly tileId: TileInstanceId;
      readonly tile: TileIdentity;
    }
  | {
      readonly kind: 'remove-tile';
      readonly tileId: TileInstanceId;
    }
  | {
      readonly kind: 'add-meld-group';
    }
  | {
      readonly kind: 'remove-meld-group';
      readonly groupId: CorrectionMeldGroupId;
    }
  | {
      readonly kind: 'move-tile';
      readonly tileId: TileInstanceId;
      readonly destination: CorrectionDestination;
      readonly index: number;
    }
  | {
      readonly kind: 'toggle-kan-openness';
      readonly groupId: CorrectionMeldGroupId;
    };

export interface CorrectionValidation {
  readonly issues: readonly CorrectionIssue[];
  readonly canCommit: boolean;
}

export type CorrectionIssueTarget =
  | {
      readonly kind: 'completed-hand';
    }
  | {
      readonly kind: 'meld';
      readonly groupId: CorrectionMeldGroupId;
    }
  | {
      readonly kind: 'winning-structure';
    };

export type CorrectionIssue =
  | {
      readonly kind: 'completed-hand-count';
      readonly target: { readonly kind: 'completed-hand' };
    }
  | {
      readonly kind: 'invalid-completed-hand-tile';
      readonly target: { readonly kind: 'completed-hand' };
    }
  | {
      readonly kind: 'invalid-meld';
      readonly target: {
        readonly kind: 'meld';
        readonly groupId: CorrectionMeldGroupId;
      };
    }
  | {
      readonly kind: 'not-winning-shape';
      readonly target: { readonly kind: 'winning-structure' };
    };

export type CorrectionCommit =
  | {
      readonly kind: 'invalid';
      readonly validation: CorrectionValidation;
    }
  | {
      readonly kind: 'valid';
      readonly structure: RecognizedStructure;
    };

export interface CorrectionEditorService {
  create(structure: RecognizedStructure): CorrectionDraft;
  update(draft: CorrectionDraft, command: CorrectionCommand): CorrectionDraft;
  validate(draft: CorrectionDraft): CorrectionValidation;
  commit(draft: CorrectionDraft): CorrectionCommit;
}

export interface CorrectionDraftIdGenerator {
  readonly nextTileInstanceId: () => TileInstanceId;
  readonly nextMeldGroupId: () => CorrectionMeldGroupId;
}

type TileLocation =
  | {
      readonly kind: 'completed-hand';
      readonly index: number;
    }
  | {
      readonly kind: 'dora-indicators';
      readonly index: number;
    }
  | {
      readonly kind: 'meld';
      readonly groupIndex: number;
      readonly tileIndex: number;
    };

type DraftRegion =
  | {
      readonly kind: 'completed-hand';
      readonly tiles: readonly TileInstance[];
    }
  | {
      readonly kind: 'dora-indicators';
      readonly tiles: readonly TileInstance[];
    }
  | {
      readonly kind: 'meld';
      readonly groupIndex: number;
      readonly tiles: readonly TileInstance[];
    };

const COMPLETED_HAND_SIZE = 14;
const MELD_LOGICAL_TILE_SIZE = 3;

export function createCorrectionEditorService(
  scoringService: ScoringService,
  idGenerator: CorrectionDraftIdGenerator = createDefaultCorrectionIdGenerator(),
): CorrectionEditorService {
  return {
    create(structure) {
      return {
        completedHand: [...structure.completedHand],
        meldGroups: structure.meldGroups.map((group) =>
          normalizeMeldGroupDraft({
            id: idGenerator.nextMeldGroupId(),
            tiles: [...group.tiles],
            kanOpenness: kanOpennessFromRecognizedGroup(group),
          }),
        ),
        doraIndicators: [...structure.doraIndicators],
      };
    },

    update(draft, command) {
      switch (command.kind) {
        case 'add-tile':
          return insertTile(
            draft,
            command.destination,
            {
              id: idGenerator.nextTileInstanceId(),
              tile: command.tile,
            },
            command.index,
          );

        case 'replace-tile':
          return replaceTile(draft, command.tileId, command.tile);

        case 'remove-tile':
          return removeTile(draft, command.tileId).draft;

        case 'add-meld-group':
          return {
            ...draft,
            meldGroups: [
              ...draft.meldGroups,
              {
                id: idGenerator.nextMeldGroupId(),
                tiles: [],
                kanOpenness: null,
              },
            ],
          };

        case 'remove-meld-group':
          return removeMeldGroup(draft, command.groupId);

        case 'move-tile':
          return moveTile(
            draft,
            command.tileId,
            command.destination,
            command.index,
          );

        case 'toggle-kan-openness':
          return toggleKanOpenness(draft, command.groupId);
      }
    },

    validate(draft) {
      return validateCorrectionDraft(draft, scoringService);
    },

    commit(draft) {
      const validation = validateCorrectionDraft(draft, scoringService);

      if (!validation.canCommit) {
        return { kind: 'invalid', validation };
      }

      return {
        kind: 'valid',
        structure: toRecognizedStructure(draft),
      };
    },
  };
}

export function deriveCorrectionMeldGroup(
  group: CorrectionMeldGroupDraft,
): RecognizedMeldGroup | null {
  const tiles = group.tiles;

  if (tiles.length === 3) {
    if (isSequence(tiles)) {
      return { kind: 'chi', tiles: [tiles[0], tiles[1], tiles[2]] };
    }

    if (areEqualKinds(tiles)) {
      return { kind: 'pon', tiles: [tiles[0], tiles[1], tiles[2]] };
    }

    return null;
  }

  if (
    tiles.length === 4 &&
    areEqualKinds(tiles) &&
    group.kanOpenness !== null
  ) {
    return {
      kind: group.kanOpenness === 'open' ? 'open-kan' : 'concealed-kan',
      tiles: [tiles[0], tiles[1], tiles[2], tiles[3]],
    };
  }

  return null;
}

function createDefaultCorrectionIdGenerator(): CorrectionDraftIdGenerator {
  let sequence = 0;

  return {
    nextTileInstanceId() {
      sequence += 1;
      return `correction-tile-${sequence}` as TileInstanceId;
    },
    nextMeldGroupId() {
      sequence += 1;
      return `correction-meld-${sequence}` as CorrectionMeldGroupId;
    },
  };
}

function kanOpennessFromRecognizedGroup(
  group: RecognizedMeldGroup,
): CorrectionMeldGroupDraft['kanOpenness'] {
  switch (group.kind) {
    case 'open-kan':
      return 'open';
    case 'concealed-kan':
      return 'concealed';
    case 'chi':
    case 'pon':
    case 'unresolved':
      return null;
  }
}

function normalizeMeldGroupDraft(
  group: CorrectionMeldGroupDraft,
): CorrectionMeldGroupDraft {
  if (!isKanCandidate(group.tiles)) {
    return { ...group, kanOpenness: null };
  }

  return {
    ...group,
    kanOpenness: group.kanOpenness ?? 'open',
  };
}

function insertTile(
  draft: CorrectionDraft,
  destination: CorrectionDestination,
  tile: TileInstance,
  index: number | undefined,
): CorrectionDraft {
  const region = getRegion(draft, destination);
  const insertionIndex = index ?? region.tiles.length;
  const nextTiles = insertAt(region.tiles, tile, insertionIndex);

  return replaceRegionTiles(draft, region, nextTiles);
}

function replaceTile(
  draft: CorrectionDraft,
  tileId: TileInstanceId,
  tile: TileIdentity,
): CorrectionDraft {
  const location = findTileLocation(draft, tileId);

  if (location === null) {
    throw new RangeError('correction tile was not found');
  }

  switch (location.kind) {
    case 'completed-hand':
      return {
        ...draft,
        completedHand: draft.completedHand.map((candidate, index) =>
          index === location.index ? { ...candidate, tile } : candidate,
        ),
      };

    case 'dora-indicators':
      return {
        ...draft,
        doraIndicators: draft.doraIndicators.map((candidate, index) =>
          index === location.index ? { ...candidate, tile } : candidate,
        ),
      };

    case 'meld':
      return {
        ...draft,
        meldGroups: draft.meldGroups.map((group, groupIndex) =>
          groupIndex === location.groupIndex
            ? normalizeMeldGroupDraft({
                ...group,
                tiles: group.tiles.map((candidate, tileIndex) =>
                  tileIndex === location.tileIndex
                    ? { ...candidate, tile }
                    : candidate,
                ),
              })
            : group,
        ),
      };
  }
}

function removeTile(
  draft: CorrectionDraft,
  tileId: TileInstanceId,
): { readonly draft: CorrectionDraft; readonly tile: TileInstance } {
  const location = findTileLocation(draft, tileId);

  if (location === null) {
    throw new RangeError('correction tile was not found');
  }

  switch (location.kind) {
    case 'completed-hand': {
      const tile = draft.completedHand[location.index];
      return {
        draft: {
          ...draft,
          completedHand: removeAt(draft.completedHand, location.index),
        },
        tile,
      };
    }

    case 'dora-indicators': {
      const tile = draft.doraIndicators[location.index];
      return {
        draft: {
          ...draft,
          doraIndicators: removeAt(draft.doraIndicators, location.index),
        },
        tile,
      };
    }

    case 'meld': {
      const group = draft.meldGroups[location.groupIndex];
      const tile = group.tiles[location.tileIndex];
      return {
        draft: {
          ...draft,
          meldGroups: draft.meldGroups.map((candidate, groupIndex) =>
            groupIndex === location.groupIndex
              ? normalizeMeldGroupDraft({
                  ...candidate,
                  tiles: removeAt(candidate.tiles, location.tileIndex),
                })
              : candidate,
          ),
        },
        tile,
      };
    }
  }
}

function removeMeldGroup(
  draft: CorrectionDraft,
  groupId: CorrectionMeldGroupId,
): CorrectionDraft {
  if (!draft.meldGroups.some((group) => group.id === groupId)) {
    throw new RangeError('correction meld group was not found');
  }

  return {
    ...draft,
    meldGroups: draft.meldGroups.filter((group) => group.id !== groupId),
  };
}

function moveTile(
  draft: CorrectionDraft,
  tileId: TileInstanceId,
  destination: CorrectionDestination,
  index: number,
): CorrectionDraft {
  const removal = removeTile(draft, tileId);

  return insertTile(removal.draft, destination, removal.tile, index);
}

function toggleKanOpenness(
  draft: CorrectionDraft,
  groupId: CorrectionMeldGroupId,
): CorrectionDraft {
  if (!draft.meldGroups.some((group) => group.id === groupId)) {
    throw new RangeError('correction meld group was not found');
  }

  return {
    ...draft,
    meldGroups: draft.meldGroups.map((group) => {
      if (group.id !== groupId) {
        return group;
      }

      if (!isKanCandidate(group.tiles)) {
        throw new RangeError(
          'kan openness can only be toggled for equal four-tile groups',
        );
      }

      return {
        ...group,
        kanOpenness: group.kanOpenness === 'concealed' ? 'open' : 'concealed',
      };
    }),
  };
}

function getRegion(
  draft: CorrectionDraft,
  destination: CorrectionDestination,
): DraftRegion {
  switch (destination.kind) {
    case 'completed-hand':
      return { kind: 'completed-hand', tiles: draft.completedHand };

    case 'dora-indicators':
      return { kind: 'dora-indicators', tiles: draft.doraIndicators };

    case 'meld': {
      const groupIndex = draft.meldGroups.findIndex(
        (group) => group.id === destination.groupId,
      );

      if (groupIndex === -1) {
        throw new RangeError('correction meld group was not found');
      }

      return {
        kind: 'meld',
        groupIndex,
        tiles: draft.meldGroups[groupIndex].tiles,
      };
    }
  }
}

function replaceRegionTiles(
  draft: CorrectionDraft,
  region: DraftRegion,
  tiles: readonly TileInstance[],
): CorrectionDraft {
  switch (region.kind) {
    case 'completed-hand':
      return { ...draft, completedHand: tiles };

    case 'dora-indicators':
      return { ...draft, doraIndicators: tiles };

    case 'meld':
      return {
        ...draft,
        meldGroups: draft.meldGroups.map((group, groupIndex) =>
          groupIndex === region.groupIndex
            ? normalizeMeldGroupDraft({ ...group, tiles })
            : group,
        ),
      };
  }
}

function insertAt<T>(
  items: readonly T[],
  item: T,
  index: number,
): readonly T[] {
  assertInsertionIndex(items, index);

  return [...items.slice(0, index), item, ...items.slice(index)];
}

function removeAt<T>(items: readonly T[], index: number): readonly T[] {
  return [...items.slice(0, index), ...items.slice(index + 1)];
}

function assertInsertionIndex(items: readonly unknown[], index: number): void {
  if (!Number.isInteger(index) || index < 0 || index > items.length) {
    throw new RangeError('correction insertion index is out of range');
  }
}

function findTileLocation(
  draft: CorrectionDraft,
  tileId: TileInstanceId,
): TileLocation | null {
  const completedHandIndex = draft.completedHand.findIndex(
    (tile) => tile.id === tileId,
  );

  if (completedHandIndex !== -1) {
    return { kind: 'completed-hand', index: completedHandIndex };
  }

  const doraIndicatorIndex = draft.doraIndicators.findIndex(
    (tile) => tile.id === tileId,
  );

  if (doraIndicatorIndex !== -1) {
    return { kind: 'dora-indicators', index: doraIndicatorIndex };
  }

  for (const [groupIndex, group] of draft.meldGroups.entries()) {
    const tileIndex = group.tiles.findIndex((tile) => tile.id === tileId);

    if (tileIndex !== -1) {
      return { kind: 'meld', groupIndex, tileIndex };
    }
  }

  return null;
}

function validateLocalStructure(
  draft: CorrectionDraft,
): readonly CorrectionIssue[] {
  const issues: CorrectionIssue[] = [];
  const expectedCompletedHandSize =
    COMPLETED_HAND_SIZE - draft.meldGroups.length * MELD_LOGICAL_TILE_SIZE;

  if (draft.completedHand.length !== expectedCompletedHandSize) {
    issues.push({
      kind: 'completed-hand-count',
      target: { kind: 'completed-hand' },
    });
  }

  for (const group of draft.meldGroups) {
    if (deriveCorrectionMeldGroup(group) === null) {
      issues.push({
        kind: 'invalid-meld',
        target: { kind: 'meld', groupId: group.id },
      });
    }
  }

  return issues;
}

function mapWinningStructureIssue(
  issue: WinningStructureIssue,
  draft: CorrectionDraft,
): CorrectionIssue {
  switch (issue.kind) {
    case 'completed-hand-count':
      return {
        kind: 'completed-hand-count',
        target: { kind: 'completed-hand' },
      };

    case 'completed-hand-tile':
      return {
        kind: 'invalid-completed-hand-tile',
        target: { kind: 'completed-hand' },
      };

    case 'meld-group': {
      const group = draft.meldGroups[issue.meldIndex];

      if (group === undefined) {
        return {
          kind: 'not-winning-shape',
          target: { kind: 'winning-structure' },
        };
      }

      return {
        kind: 'invalid-meld',
        target: { kind: 'meld', groupId: group.id },
      };
    }
  }
}

function validateCorrectionDraft(
  draft: CorrectionDraft,
  scoringService: ScoringService,
): CorrectionValidation {
  const localIssues = validateLocalStructure(draft);

  if (localIssues.length > 0) {
    return { issues: localIssues, canCommit: false };
  }

  const canonical = toRecognizedStructure(draft);
  const validation = scoringService.validateWinningStructure(canonical);

  switch (validation.kind) {
    case 'valid':
      return { issues: [], canCommit: true };

    case 'not-winning-shape':
      return {
        issues: [
          {
            kind: 'not-winning-shape',
            target: { kind: 'winning-structure' },
          },
        ],
        canCommit: false,
      };

    case 'invalid-structure':
      return {
        issues: validation.issues.map((issue) =>
          mapWinningStructureIssue(issue, draft),
        ),
        canCommit: false,
      };
  }
}

function toRecognizedStructure(draft: CorrectionDraft): RecognizedStructure {
  return {
    completedHand: draft.completedHand,
    doraIndicators: draft.doraIndicators,
    meldGroups: draft.meldGroups.map((group) => {
      const recognizedGroup = deriveCorrectionMeldGroup(group);

      if (recognizedGroup === null) {
        throw new RangeError('cannot materialize an invalid correction meld group');
      }

      return recognizedGroup;
    }),
  };
}

function isKanCandidate(tiles: readonly TileInstance[]): boolean {
  return tiles.length === 4 && areEqualKinds(tiles);
}

function areEqualKinds(tiles: readonly TileInstance[]): boolean {
  const first = tiles[0];

  return (
    first !== undefined &&
    tiles.every((tile) => tile.tile.kind === first.tile.kind)
  );
}

function isSequence(tiles: readonly TileInstance[]): boolean {
  const parsed = tiles.map(({ tile }) => parseSuitedKind(tile.kind));

  if (parsed.some((kind) => kind === null)) {
    return false;
  }

  const suitedKinds = [...(parsed as readonly SuitedKind[])].sort(
    (left, right) => left.number - right.number,
  );
  const [first, second, third] = suitedKinds;

  return (
    first.suit === second.suit &&
    second.suit === third.suit &&
    first.number + 1 === second.number &&
    second.number + 1 === third.number
  );
}

interface SuitedKind {
  readonly suit: 'm' | 'p' | 's';
  readonly number: number;
}

function parseSuitedKind(kind: TileKind): SuitedKind | null {
  const suit = kind.at(1);

  if (suit !== 'm' && suit !== 'p' && suit !== 's') {
    return null;
  }

  return {
    suit,
    number: Number(kind.at(0)),
  };
}
