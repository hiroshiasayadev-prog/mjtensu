import {
  createCorrectionEditorService,
  deriveCorrectionMeldGroup,
  type CorrectionDraftIdGenerator,
  type CorrectionMeldGroupDraft,
  type CorrectionMeldGroupId,
} from '@/application';
import type {
  RecognizedMeldGroup,
  RecognizedStructure,
  TileIdentity,
  TileInstance,
  TileInstanceId,
  TileKind,
} from '@/domain';
import type {
  ScoringDraft,
  ScoringInput,
  ScoringRuleProfile,
  ScoringService,
  WinningStructureValidation,
} from '@/scoring';
import { describe, expect, it } from 'vitest';

import { createFakeService } from './support';

function tileId(value: string): TileInstanceId {
  return value as TileInstanceId;
}

function groupId(value: string): CorrectionMeldGroupId {
  return value as CorrectionMeldGroupId;
}

function identity(kind: TileKind, red = false): TileIdentity {
  return { kind, red };
}

function tile(id: string, kind: TileKind, red = false): TileInstance {
  return { id: tileId(id), tile: identity(kind, red) };
}

function tiles(prefix: string, kinds: readonly TileKind[]): readonly TileInstance[] {
  return kinds.map((kind, index) => tile(`${prefix}-${index + 1}`, kind));
}

function structure(
  completedHand: readonly TileInstance[],
  overrides: Partial<RecognizedStructure> = {},
): RecognizedStructure {
  return {
    completedHand,
    meldGroups: [],
    doraIndicators: [],
    ...overrides,
  };
}

function draftGroup(
  groupIdValue: string,
  kinds: readonly TileKind[],
  kanOpenness: CorrectionMeldGroupDraft['kanOpenness'] = null,
): CorrectionMeldGroupDraft {
  return {
    id: groupId(groupIdValue),
    tiles: tiles(groupIdValue, kinds),
    kanOpenness,
  };
}

function idGenerator(): CorrectionDraftIdGenerator {
  let tileSequence = 0;
  let groupSequence = 0;

  return {
    nextTileInstanceId() {
      tileSequence += 1;
      return tileId(`new-tile-${tileSequence}`);
    },
    nextMeldGroupId() {
      groupSequence += 1;
      return groupId(`new-group-${groupSequence}`);
    },
  };
}

function fakeScoringService(
  validateWinningStructure: ScoringService['validateWinningStructure'] = () => ({
    kind: 'valid',
  }),
): ScoringService {
  return createFakeService<ScoringService>({
    validateWinningStructure,
    preview: (_draft: ScoringDraft, _ruleProfile: ScoringRuleProfile) => {
      throw new Error('correction validation must not inspect yaku readiness');
    },
    calculate: (_input: ScoringInput, _ruleProfile: ScoringRuleProfile) => {
      throw new Error('correction validation must not calculate score');
    },
  });
}

const closedWinningHand = tiles('closed', [
  '1m',
  '2m',
  '3m',
  '1p',
  '2p',
  '3p',
  '1s',
  '2s',
  '3s',
  '5z',
  '5z',
  '5z',
  '7z',
  '7z',
]);

describe('correction draft service', () => {
  it('creates a permissive draft from committed structure without placeholder tiles', () => {
    const service = createCorrectionEditorService(
      fakeScoringService(),
      idGenerator(),
    );
    const unresolvedTiles = tiles('unresolved', ['1m', '9m']);

    const draft = service.create(
      structure([tile('hand', '5m')], {
        meldGroups: [{ kind: 'unresolved', tiles: unresolvedTiles }],
        doraIndicators: [tile('dora', '1z')],
      }),
    );

    expect(draft.completedHand).toEqual([tile('hand', '5m')]);
    expect(draft.doraIndicators).toEqual([tile('dora', '1z')]);
    expect(draft.meldGroups).toEqual([
      {
        id: groupId('new-group-1'),
        tiles: unresolvedTiles,
        kanOpenness: null,
      },
    ]);
  });

  it('applies add, replace, remove, move, reorder, and meld-group commands locally', () => {
    const service = createCorrectionEditorService(
      fakeScoringService(),
      idGenerator(),
    );
    const draft = service.create(
      structure([tile('a', '1m'), tile('b', '2m'), tile('c', '3m')]),
    );

    const withTileAdded = service.update(draft, {
      kind: 'add-tile',
      destination: { kind: 'completed-hand' },
      tile: identity('4m'),
      index: 1,
    });
    const withReplacement = service.update(withTileAdded, {
      kind: 'replace-tile',
      tileId: tileId('b'),
      tile: identity('5m'),
    });
    const withAddedGroup = service.update(withReplacement, {
      kind: 'add-meld-group',
    });
    const group = withAddedGroup.meldGroups[0];
    const withMove = service.update(withAddedGroup, {
      kind: 'move-tile',
      tileId: tileId('a'),
      destination: { kind: 'meld', groupId: group.id },
      index: 0,
    });
    const withReorder = service.update(withMove, {
      kind: 'move-tile',
      tileId: tileId('c'),
      destination: { kind: 'completed-hand' },
      index: 0,
    });
    const withRemoval = service.update(withReorder, {
      kind: 'remove-tile',
      tileId: tileId('new-tile-1'),
    });
    const withoutGroup = service.update(withRemoval, {
      kind: 'remove-meld-group',
      groupId: group.id,
    });

    expect(withReplacement.completedHand.map(({ id }) => id)).toEqual([
      tileId('a'),
      tileId('new-tile-1'),
      tileId('b'),
      tileId('c'),
    ]);
    expect(withReplacement.completedHand[2]).toEqual(tile('b', '5m'));
    expect(withMove.meldGroups[0]?.tiles).toEqual([tile('a', '1m')]);
    expect(withReorder.completedHand.map(({ id }) => id)).toEqual([
      tileId('c'),
      tileId('new-tile-1'),
      tileId('b'),
    ]);
    expect(withRemoval.completedHand.map(({ id }) => id)).toEqual([
      tileId('c'),
      tileId('b'),
    ]);
    expect(withoutGroup.meldGroups).toEqual([]);
  });

  it('preserves IDs on identity replacement and creates or removes IDs only for add/delete', () => {
    const service = createCorrectionEditorService(
      fakeScoringService(),
      idGenerator(),
    );
    const draft = service.create(structure([tile('selected', '5m')]));

    const replaced = service.update(draft, {
      kind: 'replace-tile',
      tileId: tileId('selected'),
      tile: identity('6m'),
    });
    const added = service.update(replaced, {
      kind: 'add-tile',
      destination: { kind: 'completed-hand' },
      tile: identity('7m'),
    });
    const removed = service.update(added, {
      kind: 'remove-tile',
      tileId: tileId('selected'),
    });

    expect(replaced.completedHand).toEqual([tile('selected', '6m')]);
    expect(added.completedHand.at(-1)?.id).toBe(tileId('new-tile-1'));
    expect(removed.completedHand.some(({ id }) => id === tileId('selected'))).toBe(
      false,
    );
  });

  it('derives chi, pon, and kan from composition without assigning invalid meld metadata', () => {
    const cases: {
      readonly name: string;
      readonly group: CorrectionMeldGroupDraft;
      readonly kind: RecognizedMeldGroup['kind'] | null;
    }[] = [
      {
        name: 'same-suit sequence',
        group: draftGroup('chi', ['3m', '1m', '2m']),
        kind: 'chi',
      },
      {
        name: 'equal triple',
        group: draftGroup('pon', ['5m', '5m', '5m']),
        kind: 'pon',
      },
      {
        name: 'open equal four',
        group: draftGroup('open-kan', ['7p', '7p', '7p', '7p'], 'open'),
        kind: 'open-kan',
      },
      {
        name: 'concealed equal four',
        group: draftGroup(
          'concealed-kan',
          ['2s', '2s', '2s', '2s'],
          'concealed',
        ),
        kind: 'concealed-kan',
      },
      {
        name: 'malformed group',
        group: draftGroup('invalid', ['1z', '2z', '3z']),
        kind: null,
      },
    ];

    for (const { group, kind } of cases) {
      expect(deriveCorrectionMeldGroup(group)?.kind ?? null).toBe(kind);
    }
  });

  it('initializes and toggles kan openness only for equal four-tile groups', () => {
    const service = createCorrectionEditorService(
      fakeScoringService(),
      idGenerator(),
    );
    const draft = service.create(
      structure(tiles('hand', ['1m', '1m']), {
        meldGroups: [
          {
            kind: 'concealed-kan',
            tiles: [
              tile('kan-1', '9m'),
              tile('kan-2', '9m'),
              tile('kan-3', '9m'),
              tile('kan-4', '9m'),
            ],
          },
        ],
      }),
    );
    const group = draft.meldGroups[0];

    const toggled = service.update(draft, {
      kind: 'toggle-kan-openness',
      groupId: group.id,
    });
    const malformed = service.update(toggled, {
      kind: 'replace-tile',
      tileId: tileId('kan-4'),
      tile: identity('8m'),
    });

    expect(group.kanOpenness).toBe('concealed');
    expect(toggled.meldGroups[0]?.kanOpenness).toBe('open');
    expect(malformed.meldGroups[0]?.kanOpenness).toBeNull();
    expect(() =>
      service.update(malformed, {
        kind: 'toggle-kan-openness',
        groupId: group.id,
      }),
    ).toThrow(RangeError);
  });

  it('targets local validation issues to the completed hand or the malformed meld', () => {
    let shapeValidationCalls = 0;
    const service = createCorrectionEditorService(
      fakeScoringService(() => {
        shapeValidationCalls += 1;
        return { kind: 'valid' };
      }),
      idGenerator(),
    );
    const draft = service.create(
      structure(tiles('hand', Array<TileKind>(11).fill('1m')), {
        meldGroups: [
          {
            kind: 'unresolved',
            tiles: tiles('bad-meld', ['1m', '1m', '2m']),
          },
        ],
      }),
    );
    const badCountDraft = {
      ...draft,
      completedHand: draft.completedHand.slice(0, 10),
    };

    expect(service.validate(draft)).toEqual({
      canCommit: false,
      issues: [
        {
          kind: 'invalid-meld',
          target: { kind: 'meld', groupId: groupId('new-group-1') },
        },
      ],
    });
    expect(service.validate(badCountDraft).issues).toContainEqual({
      kind: 'completed-hand-count',
      target: { kind: 'completed-hand' },
    });
    expect(shapeValidationCalls).toBe(0);
  });

  it('maps scoring shape-validation issues to product correction targets', () => {
    const service = createCorrectionEditorService(
      fakeScoringService(
        (): WinningStructureValidation => ({
          kind: 'invalid-structure',
          issues: [{ kind: 'completed-hand-count' }, { kind: 'meld-group', meldIndex: 0 }],
        }),
      ),
      idGenerator(),
    );
    const draft = service.create(
      structure(tiles('hand', Array<TileKind>(11).fill('2m')), {
        meldGroups: [
          { kind: 'chi', tiles: [tile('chi-1', '1m'), tile('chi-2', '2m'), tile('chi-3', '3m')] },
        ],
      }),
    );

    expect(service.validate(draft)).toEqual({
      canCommit: false,
      issues: [
        {
          kind: 'completed-hand-count',
          target: { kind: 'completed-hand' },
        },
        {
          kind: 'invalid-meld',
          target: { kind: 'meld', groupId: groupId('new-group-1') },
        },
      ],
    });
  });

  it('targets unsupported complete shapes to the whole winning structure', () => {
    const service = createCorrectionEditorService(
      fakeScoringService(() => ({ kind: 'not-winning-shape' })),
      idGenerator(),
    );

    expect(service.validate(service.create(structure(closedWinningHand)))).toEqual({
      canCommit: false,
      issues: [
        {
          kind: 'not-winning-shape',
          target: { kind: 'winning-structure' },
        },
      ],
    });
  });

  it('commits only a validated canonical structure and ignores yaku or Conditions readiness', () => {
    const validatedStructures: RecognizedStructure[] = [];
    const service = createCorrectionEditorService(
      fakeScoringService((recognizedStructure) => {
        validatedStructures.push(recognizedStructure);
        return { kind: 'valid' };
      }),
      idGenerator(),
    );
    const draft = service.create(
      structure(tiles('hand', Array<TileKind>(11).fill('5p')), {
        meldGroups: [
          {
            kind: 'unresolved',
            tiles: [tile('meld-1', '6s'), tile('meld-2', '6s'), tile('meld-3', '6s')],
          },
        ],
      }),
    );

    const commit = service.commit(draft);

    expect(commit).toEqual({
      kind: 'valid',
      structure: {
        completedHand: draft.completedHand,
        doraIndicators: [],
        meldGroups: [
          {
            kind: 'pon',
            tiles: [tile('meld-1', '6s'), tile('meld-2', '6s'), tile('meld-3', '6s')],
          },
        ],
      },
    });
    expect(validatedStructures).toEqual([commit.kind === 'valid' ? commit.structure : null]);
  });

  it('returns an invalid commit when the scoring shape validator rejects the draft', () => {
    const service = createCorrectionEditorService(
      fakeScoringService(() => ({ kind: 'not-winning-shape' })),
      idGenerator(),
    );

    expect(service.commit(service.create(structure(closedWinningHand)))).toEqual({
      kind: 'invalid',
      validation: {
        canCommit: false,
        issues: [
          {
            kind: 'not-winning-shape',
            target: { kind: 'winning-structure' },
          },
        ],
      },
    });
  });
});
