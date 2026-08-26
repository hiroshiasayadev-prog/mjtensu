import { createStore, type StoreApi } from 'zustand/vanilla';

import type { TileInstance, TileInstanceId } from '@/domain';
import { DEFAULT_RULE_PROFILE, type ScoringCalculation } from '@/scoring';

import {
  INITIAL_SCORING_CONDITIONS,
  type ScoringSessionState,
} from './scoring-session-service';

export type ActiveScoringSessionState = ScoringSessionState;

export interface ApplicationStateSnapshot {
  readonly activeScoringSession: ActiveScoringSessionState | null;
}

export interface ApplicationStoreState extends ApplicationStateSnapshot {
  beginNewRecognitionAttempt(): void;
  clearActiveScoringSession(): void;
  installScoringSession(session: ActiveScoringSessionState): void;
}

export type ApplicationStore = StoreApi<ApplicationStoreState>;

export function createApplicationStore(
  initialState: Partial<ApplicationStateSnapshot> = {},
): ApplicationStore {
  return createStore<ApplicationStoreState>()((set) => ({
    activeScoringSession: initialState.activeScoringSession ?? null,
    beginNewRecognitionAttempt: () => {
      set({ activeScoringSession: null });
    },
    clearActiveScoringSession: () => {
      set({ activeScoringSession: null });
    },
    installScoringSession: (session) => {
      set({ activeScoringSession: session });
    },
  }));
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
