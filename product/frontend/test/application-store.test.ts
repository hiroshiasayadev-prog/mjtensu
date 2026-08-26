import {
  INITIAL_SCORING_CONDITIONS,
  createApplicationStore,
  createScoringSessionFixture,
  selectActiveScoringSession,
  selectHasActiveScoringSession,
  type ApplicationSessionPort,
  type ScoringConditionAvailability,
  type ScoringConditionPolicy,
} from '@/application';
import type { RecognizedStructure, TileInstance, TileInstanceId } from '@/domain';
import {
  DEFAULT_RULE_PROFILE,
  type ScoringCalculation,
  type ScoringPreview,
} from '@/scoring';
import { describe, expect, it, vi } from 'vitest';

const preview: ScoringPreview = {
  kind: 'ready',
  yaku: [{ kind: 'regular', id: 'menzen-tsumo', han: 1 }],
};

const calculation: ScoringCalculation = {
  yaku: [{ kind: 'regular', id: 'menzen-tsumo', han: 1 }],
  dora: { dora: 0, akaDora: 0 },
  han: 1,
  fu: {
    kind: 'standard',
    base: 20,
    menzenRon: 0,
    tsumo: 2,
    melds: 0,
    pair: 0,
    wait: 0,
    rawTotal: 22,
    rounded: 30,
  },
  limit: null,
  winnerRole: 'dealer',
  winMethod: 'tsumo',
  payment: { kind: 'tsumo-dealer', eachOpponent: 500 },
  totalPoints: 1500,
};

const availability: ScoringConditionAvailability = {
  ippatsu: false,
  rinshan: true,
  chankan: false,
  haitei: true,
  houtei: false,
  tenhou: true,
  chiihou: false,
};

function tileId(value: string): TileInstanceId {
  return value as TileInstanceId;
}

function tile(id: string, kind: TileInstance['tile']['kind'] = '5m'): TileInstance {
  return {
    id: tileId(id),
    tile: { kind, red: false },
  };
}

function structure(
  completedHand: readonly TileInstance[] = [tile('left'), tile('right')],
): RecognizedStructure {
  return {
    completedHand,
    doraIndicators: [],
    meldGroups: [],
  };
}

function fakeSessionPort(
  overrides: Partial<ApplicationSessionPort> = {},
): ApplicationSessionPort {
  return {
    create: vi.fn((committedStructure, ruleProfile) => ({
      structure: committedStructure,
      winningTileId: committedStructure.completedHand.at(-1)?.id ?? tileId('missing'),
      conditions: INITIAL_SCORING_CONDITIONS,
      ruleProfile,
      latestResult: null,
    })),
    update: vi.fn((state, command) => ({
      ...state,
      winningTileId: command.kind === 'select-winning-tile'
        ? command.tileId
        : state.winningTileId,
      latestResult: null,
    })),
    preview: vi.fn(() => preview),
    calculate: vi.fn((state) => ({
      state: { ...state, latestResult: calculation },
      result: calculation,
    })),
    ...overrides,
  };
}

describe('application store', () => {
  it('starts empty and exposes route-consumable active-session selectors', () => {
    const store = createApplicationStore();

    expect(selectActiveScoringSession(store.getState())).toBeNull();
    expect(selectHasActiveScoringSession(store.getState())).toBe(false);
  });

  it('creates and replaces the active scoring session through the session port', () => {
    const firstStructure = structure([tile('first-left'), tile('first-right')]);
    const secondStructure = structure([tile('second-left'), tile('second-right')]);
    const port = fakeSessionPort();
    const store = createApplicationStore({}, { scoringSessionService: port });

    store.getState().createScoringSession(firstStructure, DEFAULT_RULE_PROFILE);
    const firstSession = store.getState().activeScoringSession;
    store.getState().createScoringSession(secondStructure, DEFAULT_RULE_PROFILE);

    expect(port.create).toHaveBeenNthCalledWith(
      1,
      firstStructure,
      DEFAULT_RULE_PROFILE,
    );
    expect(port.create).toHaveBeenNthCalledWith(
      2,
      secondStructure,
      DEFAULT_RULE_PROFILE,
    );
    expect(firstSession?.structure).toBe(firstStructure);
    expect(store.getState().activeScoringSession?.structure).toBe(secondStructure);
  });

  it('clears the prior session for a new recognition attempt without creating a replacement', () => {
    const existingSession = createScoringSessionFixture();
    const port = fakeSessionPort();
    const store = createApplicationStore(
      { activeScoringSession: existingSession },
      { scoringSessionService: port },
    );

    store.getState().beginNewRecognitionAttempt();

    expect(store.getState().activeScoringSession).toBeNull();
    expect(port.create).not.toHaveBeenCalled();
  });

  it('does not expose whole-session replacement on the production mutable surface', () => {
    const hydratedSession = createScoringSessionFixture({
      tileIdPrefix: 'hydrated',
      latestResult: calculation,
    });
    const store = createApplicationStore({ activeScoringSession: hydratedSession });

    expect(store.getState().activeScoringSession).toBe(hydratedSession);
    expect('installScoringSession' in store.getState()).toBe(false);
  });

  it('delegates session updates and preserves service-owned result invalidation semantics', () => {
    const staleSession = createScoringSessionFixture({ latestResult: calculation });
    const selectedTileId = staleSession.structure.completedHand[0].id;
    const port = fakeSessionPort();
    const store = createApplicationStore(
      { activeScoringSession: staleSession },
      { scoringSessionService: port },
    );

    store.getState().updateScoringSession({
      kind: 'select-winning-tile',
      tileId: selectedTileId,
    });

    expect(port.update).toHaveBeenCalledWith(staleSession, {
      kind: 'select-winning-tile',
      tileId: selectedTileId,
    });
    expect(store.getState().activeScoringSession).toEqual(
      expect.objectContaining({
        winningTileId: selectedTileId,
        latestResult: null,
      }),
    );
  });

  it('delegates preview, calculation, and shared condition availability without storing those services', () => {
    const session = createScoringSessionFixture();
    const port = fakeSessionPort();
    const conditionPolicy: ScoringConditionPolicy = {
      normalize: vi.fn((conditions) => conditions),
      availability: vi.fn(() => availability),
    };
    const store = createApplicationStore(
      { activeScoringSession: session },
      { conditionPolicy, scoringSessionService: port },
    );

    const receivedPreview = store.getState().previewScoringSession();
    const receivedAvailability = store.getState().getScoringConditionAvailability();
    const receivedCalculation = store.getState().calculateScoringSession();

    expect(receivedPreview).toBe(preview);
    expect(receivedAvailability).toBe(availability);
    expect(receivedCalculation.result).toBe(calculation);
    expect(port.preview).toHaveBeenCalledWith(session);
    expect(conditionPolicy.availability).toHaveBeenCalledWith(session.conditions);
    expect(port.calculate).toHaveBeenCalledWith(session);
    expect(store.getState().activeScoringSession?.latestResult).toBe(calculation);
  });

  it('keeps Zustand-owned data limited to the active scoring session or null', () => {
    const session = createScoringSessionFixture({ latestResult: calculation });
    const store = createApplicationStore({ activeScoringSession: session });
    const state = store.getState();
    const dataEntries = Object.entries(state).filter(
      ([, value]) => typeof value !== 'function',
    );

    expect(dataEntries).toEqual([['activeScoringSession', session]]);
    expect(Object.keys(session).sort()).toEqual([
      'conditions',
      'latestResult',
      'ruleProfile',
      'structure',
      'winningTileId',
    ]);
  });

  it('throws semantic no-session errors instead of fabricating partial session state', () => {
    const port = fakeSessionPort();
    const store = createApplicationStore({}, { scoringSessionService: port });

    expect(() =>
      store.getState().updateScoringSession({
        kind: 'select-winning-tile',
        tileId: tileId('missing'),
      }),
    ).toThrow('An active scoring session is required.');
    expect(() => store.getState().previewScoringSession()).toThrow(
      'An active scoring session is required.',
    );
    expect(() => store.getState().calculateScoringSession()).toThrow(
      'An active scoring session is required.',
    );
    expect(port.update).not.toHaveBeenCalled();
    expect(port.preview).not.toHaveBeenCalled();
    expect(port.calculate).not.toHaveBeenCalled();
    expect(store.getState().activeScoringSession).toBeNull();
  });
});
