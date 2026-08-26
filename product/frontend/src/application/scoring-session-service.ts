import type { RecognizedMeldGroup, RecognizedStructure, TileInstanceId } from '@/domain';
import type {
  ScoringCalculation,
  ScoringConditions,
  ScoringConditionsDraft,
  ScoringInput,
  ScoringMeld,
  ScoringPreview,
  ScoringRuleProfile,
  ScoringService,
} from '@/scoring';

import {
  scoringConditionPolicy,
  type ScoringConditionPolicy,
} from './scoring-condition-policy';

export interface ScoringSessionState {
  readonly structure: RecognizedStructure;
  readonly winningTileId: TileInstanceId;
  readonly conditions: ScoringConditionsDraft;
  readonly ruleProfile: ScoringRuleProfile;
  readonly latestResult: ScoringCalculation | null;
}

export type ScoringSessionCommand =
  | {
      readonly kind: 'select-winning-tile';
      readonly tileId: TileInstanceId;
    }
  | {
      readonly kind: 'replace-structure';
      readonly structure: RecognizedStructure;
    }
  | {
      readonly kind: 'replace-conditions';
      readonly conditions: ScoringConditionsDraft;
    }
  | {
      readonly kind: 'replace-rule-profile';
      readonly ruleProfile: ScoringRuleProfile;
    };

export interface ScoringSessionCalculation {
  readonly state: ScoringSessionState;
  readonly result: ScoringCalculation;
}

export interface ScoringSessionService {
  create(
    structure: RecognizedStructure,
    ruleProfile: ScoringRuleProfile,
  ): ScoringSessionState;
  update(
    state: ScoringSessionState,
    command: ScoringSessionCommand,
  ): ScoringSessionState;
  preview(state: ScoringSessionState): ScoringPreview;
  calculate(state: ScoringSessionState): ScoringSessionCalculation;
}

export const INITIAL_SCORING_CONDITIONS: ScoringConditionsDraft = {
  winMethod: 'tsumo',
  roundWind: 'east',
  seatWind: 'east',
  riichi: 'none',
  ippatsu: false,
  rinshan: false,
  chankan: false,
  haitei: false,
  houtei: false,
  tenhou: false,
  chiihou: false,
};

export function createScoringSessionService(
  scoringService: ScoringService,
  conditionPolicy: ScoringConditionPolicy = scoringConditionPolicy,
): ScoringSessionService {
  return {
    create(structure, ruleProfile) {
      return {
        structure,
        winningTileId: rightmostCompletedTileId(structure),
        conditions: INITIAL_SCORING_CONDITIONS,
        ruleProfile,
        latestResult: null,
      };
    },

    update(state, command) {
      switch (command.kind) {
        case 'select-winning-tile':
          assertCompletedHandTile(state.structure, command.tileId);
          return { ...state, winningTileId: command.tileId, latestResult: null };

        case 'replace-structure':
          return {
            ...state,
            structure: command.structure,
            winningTileId: completedHandContains(
              command.structure,
              state.winningTileId,
            )
              ? state.winningTileId
              : rightmostCompletedTileId(command.structure),
            latestResult: null,
          };

        case 'replace-conditions':
          return {
            ...state,
            conditions: conditionPolicy.normalize(command.conditions),
            latestResult: null,
          };

        case 'replace-rule-profile':
          return {
            ...state,
            ruleProfile: command.ruleProfile,
            latestResult: null,
          };
      }
    },

    preview(state) {
      return scoringService.preview(
        {
          structure: state.structure,
          winningTileId: state.winningTileId,
          conditions: state.conditions,
        },
        state.ruleProfile,
      );
    },

    calculate(state) {
      const result = scoringService.calculate(
        toScoringInput(state),
        state.ruleProfile,
      );
      const nextState = { ...state, latestResult: result };

      return { state: nextState, result };
    },
  };
}

function completedHandContains(
  structure: RecognizedStructure,
  tileId: TileInstanceId,
): boolean {
  return structure.completedHand.some((tile) => tile.id === tileId);
}

function assertCompletedHandTile(
  structure: RecognizedStructure,
  tileId: TileInstanceId,
): void {
  if (!completedHandContains(structure, tileId)) {
    throw new RangeError('winningTileId must identify a completed-hand tile');
  }
}

function rightmostCompletedTileId(structure: RecognizedStructure): TileInstanceId {
  const tile = structure.completedHand.at(-1);
  if (tile === undefined) {
    throw new RangeError('scoring session requires a completed-hand tile');
  }
  return tile.id;
}

function toScoringInput(state: ScoringSessionState): ScoringInput {
  assertCompletedHandTile(state.structure, state.winningTileId);

  return {
    completedHand: state.structure.completedHand,
    melds: state.structure.meldGroups.map(toScoringMeld),
    doraIndicators: state.structure.doraIndicators.map(({ tile }) => tile),
    winningTileId: state.winningTileId,
    conditions: toStrictConditions(state.conditions),
  };
}

function toStrictConditions(conditions: ScoringConditionsDraft): ScoringConditions {
  const { winMethod, roundWind, seatWind } = conditions;

  if (winMethod === null || roundWind === null || seatWind === null) {
    throw new RangeError('scoring calculation requires complete conditions');
  }

  return { ...conditions, winMethod, roundWind, seatWind };
}

function toScoringMeld(group: RecognizedMeldGroup): ScoringMeld {
  switch (group.kind) {
    case 'chi':
    case 'pon':
      return {
        kind: group.kind,
        tiles: [group.tiles[0].tile, group.tiles[1].tile, group.tiles[2].tile],
      };

    case 'open-kan':
    case 'concealed-kan':
      return {
        kind: group.kind,
        tiles: [
          group.tiles[0].tile,
          group.tiles[1].tile,
          group.tiles[2].tile,
          group.tiles[3].tile,
        ],
      };

    case 'unresolved':
      throw new RangeError('scoring calculation requires resolved meld groups');
  }
}
