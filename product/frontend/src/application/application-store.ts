import { createStore, type StoreApi } from 'zustand/vanilla';

import type { RecognizedStructure, TileInstance, TileInstanceId } from '@/domain';
import {
  DEFAULT_RULE_PROFILE,
  type ScoringCalculation,
  type ScoringPreview,
  type ScoringRuleProfile,
} from '@/scoring';

import type {
  ApplicationSessionPort,
  ApplicationStoreDependencies,
} from './application-store-dependencies';
import {
  scoringConditionPolicy,
  type ScoringConditionAvailability,
} from './scoring-condition-policy';
import {
  INITIAL_SCORING_CONDITIONS,
  type ScoringSessionCalculation,
  type ScoringSessionCommand,
  type ScoringSessionState,
} from './scoring-session-service';

export type ActiveScoringSessionState = ScoringSessionState;

export interface ApplicationStateSnapshot {
  readonly activeScoringSession: ActiveScoringSessionState | null;
}

/** Construction/test hydration only; never exposed as a mutable store action. */
export type ApplicationStoreHydrationState = Partial<ApplicationStateSnapshot>;

export interface ApplicationStoreState extends ApplicationStateSnapshot {
  beginNewRecognitionAttempt(): void;
  clearActiveScoringSession(): void;
  createScoringSession(
    structure: RecognizedStructure,
    ruleProfile: ScoringRuleProfile,
  ): ActiveScoringSessionState;
  updateScoringSession(
    command: ScoringSessionCommand,
  ): ActiveScoringSessionState;
  previewScoringSession(): ScoringPreview;
  calculateScoringSession(): ScoringSessionCalculation;
  getScoringConditionAvailability(): ScoringConditionAvailability;
}

export type ApplicationStore = StoreApi<ApplicationStoreState>;

export function createApplicationStore(
  hydrationState: ApplicationStoreHydrationState = {},
  dependencies: ApplicationStoreDependencies = {},
): ApplicationStore {
  const conditionPolicy = dependencies.conditionPolicy ?? scoringConditionPolicy;

  return createStore<ApplicationStoreState>()((set, get) => ({
    activeScoringSession: hydrationState.activeScoringSession ?? null,

    beginNewRecognitionAttempt: () => {
      set({ activeScoringSession: null });
    },

    clearActiveScoringSession: () => {
      set({ activeScoringSession: null });
    },

    createScoringSession: (structure, ruleProfile) => {
      const scoringSessionService = requireScoringSessionService(dependencies);
      const session = scoringSessionService.create(structure, ruleProfile);
      set({ activeScoringSession: session });
      return session;
    },

    updateScoringSession: (command) => {
      const scoringSessionService = requireScoringSessionService(dependencies);
      const current = requireActiveScoringSession(get());
      const session = scoringSessionService.update(current, command);
      set({ activeScoringSession: session });
      return session;
    },

    previewScoringSession: () => {
      const scoringSessionService = requireScoringSessionService(dependencies);
      return scoringSessionService.preview(requireActiveScoringSession(get()));
    },

    calculateScoringSession: () => {
      const scoringSessionService = requireScoringSessionService(dependencies);
      const calculation = scoringSessionService.calculate(
        requireActiveScoringSession(get()),
      );
      set({ activeScoringSession: calculation.state });
      return calculation;
    },

    getScoringConditionAvailability: () =>
      conditionPolicy.availability(requireActiveScoringSession(get()).conditions),
  }));
}

export function selectHasActiveScoringSession(
  state: ApplicationStoreState,
): boolean {
  return state.activeScoringSession !== null;
}

export function selectActiveScoringSession(
  state: ApplicationStoreState,
): ActiveScoringSessionState | null {
  return state.activeScoringSession;
}

function requireActiveScoringSession(
  state: ApplicationStateSnapshot,
): ActiveScoringSessionState {
  if (state.activeScoringSession === null) {
    throw new Error('An active scoring session is required.');
  }
  return state.activeScoringSession;
}

function requireScoringSessionService(
  dependencies: ApplicationStoreDependencies,
): ApplicationSessionPort {
  if (dependencies.scoringSessionService === undefined) {
    throw new Error('ScoringSessionService is not configured for this store.');
  }
  return dependencies.scoringSessionService;
}

export interface ScoringSessionFixtureOptions {
  readonly tileIdPrefix?: string;
  readonly latestResult?: ScoringCalculation | null;
}

export function createScoringSessionFixture({
  tileIdPrefix = 'fixture',
  latestResult = null,
}: ScoringSessionFixtureOptions = {}): ActiveScoringSessionState {
  const completedHand = [
    tile(`${tileIdPrefix}-left`, '1m'),
    tile(`${tileIdPrefix}-winning`, '2m'),
  ] as const;

  return {
    structure: {
      completedHand,
      doraIndicators: [],
      meldGroups: [],
    },
    winningTileId: completedHand[1].id,
    conditions: INITIAL_SCORING_CONDITIONS,
    ruleProfile: DEFAULT_RULE_PROFILE,
    latestResult,
  };
}

function tile(id: string, kind: TileInstance['tile']['kind']): TileInstance {
  return {
    id: id as TileInstanceId,
    tile: { kind, red: false },
  };
}
