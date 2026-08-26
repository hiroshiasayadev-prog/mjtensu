import type { RecognizedMeldGroup, RecognizedStructure } from '@/domain';
import type {
  ScoringCalculation,
  ScoringConditions,
  ScoringDraft,
  ScoringError,
  ScoringInput,
  ScoringInputIssue,
  ScoringMeld,
  ScoringPreview,
  ScoringRequiredField,
  ScoringRuleProfile,
  ScoringService,
  WinningStructureValidation,
} from '@/scoring';

import type { AgariEngineV1, AgariScoreOutcomeV1 } from './agari-abi';
import {
  assertScoringConditionsContract,
  assertScoringRuleProfileContract,
  assertStrictScoringContract,
  createAgariScoreRequest,
  serializeWinningStructure,
  validateWinningStructureInput,
} from './agari-input-adapter';
import { normalizeAgariCalculation } from './agari-result-adapter';

export function createAgariScoringService(engine: AgariEngineV1): ScoringService {
  return {
    validateWinningStructure(structure) {
      return validateWinningStructure(engine, structure);
    },

    preview(draft, ruleProfile) {
      const prepared = prepareDraft(draft, ruleProfile);
      if (prepared.kind !== 'ready') {
        return prepared.preview;
      }

      const evaluation = evaluate(engine, prepared.input, ruleProfile);
      switch (evaluation.kind) {
        case 'scored':
          return { kind: 'ready', yaku: evaluation.calculation.yaku };
        case 'not-winning-shape':
          return { kind: 'invalid-winning-shape' };
        case 'no-yaku':
          return { kind: 'no-yaku' };
      }
    },

    calculate(input, ruleProfile) {
      try {
        assertStrictScoringContract(input, ruleProfile);
      } catch (cause) {
        throw scoringInputContractViolation(cause);
      }

      const evaluation = evaluate(engine, input, ruleProfile);
      if (evaluation.kind !== 'scored') {
        throw scoringInputContractViolation(
          new RangeError(
            `calculate() requires a scoring-ready input; engine returned ${evaluation.kind}`,
          ),
        );
      }
      return evaluation.calculation;
    },
  };
}

type Evaluation =
  | { readonly kind: 'scored'; readonly calculation: ScoringCalculation }
  | { readonly kind: 'not-winning-shape' }
  | { readonly kind: 'no-yaku' };

type PreparedDraft =
  | { readonly kind: 'preview'; readonly preview: Exclude<ScoringPreview, { kind: 'ready' }> }
  | { readonly kind: 'ready'; readonly input: ScoringInput };

function validateWinningStructure(
  engine: AgariEngineV1,
  structure: RecognizedStructure,
): WinningStructureValidation {
  const issues = validateWinningStructureInput(structure);
  if (issues.length > 0) {
    return { kind: 'invalid-structure', issues };
  }

  let hand: string;
  try {
    hand = serializeWinningStructure(structure);
  } catch (cause) {
    throw scoringAdapterFailure(cause);
  }

  try {
    const outcome = engine.validateWinningShape(hand);
    switch (outcome.status) {
      case 'winning':
        return { kind: 'valid' };
      case 'not-winning-shape':
        return { kind: 'not-winning-shape' };
      case 'invalid-request':
      case 'internal-error':
        throw scoringAdapterFailure(outcome.error);
    }

    throw scoringAdapterFailure(
      new RangeError('unknown Agari winning-shape outcome discriminant'),
    );
  } catch (error) {
    if (isScoringError(error)) {
      throw error;
    }
    throw scoringAdapterFailure(error);
  }
}

function prepareDraft(
  draft: ScoringDraft,
  ruleProfile: ScoringRuleProfile,
): PreparedDraft {
  try {
    assertScoringRuleProfileContract(ruleProfile);
  } catch (cause) {
    throw scoringInputContractViolation(cause);
  }

  const missing = missingRequiredFields(draft);
  if (missing.length > 0) {
    return { kind: 'preview', preview: { kind: 'incomplete', missing } };
  }

  const issues: ScoringInputIssue[] = [];
  if (!draft.structure.completedHand.some((tile) => tile.id === draft.winningTileId)) {
    issues.push({ kind: 'winning-tile-not-in-completed-hand' });
  }

  const structureIssues = validateWinningStructureInput(draft.structure);
  for (const issue of structureIssues) {
    switch (issue.kind) {
      case 'completed-hand-count':
      case 'completed-hand-tile':
        issues.push({ kind: 'invalid-structure' });
        break;
      case 'meld-group':
        if (draft.structure.meldGroups[issue.meldIndex]?.kind !== 'unresolved') {
          issues.push({ kind: 'invalid-meld', meldIndex: issue.meldIndex });
        }
        break;
    }
  }

  for (const [meldIndex, group] of draft.structure.meldGroups.entries()) {
    if (group.kind === 'unresolved') {
      issues.push({ kind: 'unresolved-meld', meldIndex });
    }
  }

  if (issues.length > 0) {
    return { kind: 'preview', preview: { kind: 'invalid-input', issues } };
  }

  const conditions = toStrictConditions(draft);
  if (conditions === null) {
    throw new RangeError('complete draft conditions unexpectedly remained nullable');
  }

  const input: ScoringInput = {
    completedHand: draft.structure.completedHand,
    melds: draft.structure.meldGroups.map(toScoringMeld),
    doraIndicators: draft.structure.doraIndicators.map(({ tile }) => tile),
    winningTileId: draft.winningTileId,
    conditions,
  };

  try {
    assertScoringConditionsContract(input.conditions, input.melds);
  } catch {
    return {
      kind: 'preview',
      preview: {
        kind: 'invalid-input',
        issues: [{ kind: 'contradictory-conditions' }],
      },
    };
  }

  try {
    assertStrictScoringContract(input, ruleProfile);
  } catch {
    return {
      kind: 'preview',
      preview: {
        kind: 'invalid-input',
        issues: [{ kind: 'invalid-structure' }],
      },
    };
  }

  return { kind: 'ready', input };
}

function evaluate(
  engine: AgariEngineV1,
  input: ScoringInput,
  ruleProfile: ScoringRuleProfile,
): Evaluation {
  let request;
  try {
    request = createAgariScoreRequest(input, ruleProfile);
  } catch (cause) {
    throw scoringInputContractViolation(cause);
  }

  let outcome: AgariScoreOutcomeV1;
  try {
    outcome = engine.scoreHand(request);
  } catch (cause) {
    throw scoringAdapterFailure(cause);
  }

  switch (outcome.status) {
    case 'not-winning-shape':
      return { kind: 'not-winning-shape' };
    case 'no-yaku':
      return { kind: 'no-yaku' };
    case 'invalid-request':
    case 'internal-error':
      throw scoringAdapterFailure(outcome.error);
    case 'scored':
      try {
        return {
          kind: 'scored',
          calculation: normalizeAgariCalculation(outcome.result, input),
        };
      } catch (cause) {
        throw scoringAdapterFailure(cause);
      }
  }

  throw scoringAdapterFailure(new RangeError('unknown Agari score outcome discriminant'));
}

function missingRequiredFields(draft: ScoringDraft): ScoringRequiredField[] {
  const missing: ScoringRequiredField[] = [];
  if (draft.conditions.winMethod === null) {
    missing.push('win-method');
  }
  if (draft.conditions.roundWind === null) {
    missing.push('round-wind');
  }
  if (draft.conditions.seatWind === null) {
    missing.push('seat-wind');
  }
  return missing;
}

function toStrictConditions(draft: ScoringDraft): ScoringConditions | null {
  const { winMethod, roundWind, seatWind } = draft.conditions;
  if (winMethod === null || roundWind === null || seatWind === null) {
    return null;
  }
  return { ...draft.conditions, winMethod, roundWind, seatWind };
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
      throw new RangeError('unresolved meld cannot become strict scoring input');
  }
}

function scoringInputContractViolation(cause: unknown): ScoringError {
  return { kind: 'input-contract-violation', cause };
}

function scoringAdapterFailure(cause: unknown): ScoringError {
  return { kind: 'adapter-failure', cause };
}

function isScoringError(error: unknown): error is ScoringError {
  if (typeof error !== 'object' || error === null || !('kind' in error)) {
    return false;
  }
  return (
    error.kind === 'input-contract-violation' || error.kind === 'adapter-failure'
  );
}
