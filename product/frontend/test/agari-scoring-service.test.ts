import type { RecognizedStructure, TileKind } from '@/domain';
import type { ScoringDraft, ScoringError } from '@/scoring';
import { describe, expect, it, vi } from 'vitest';

import type {
  AgariEngineV1,
  AgariScoreOutcomeV1,
  AgariWasmModuleV1,
} from '../src/scoring/agari/agari-abi';
import { createAgariScoringService } from '../src/scoring/agari/agari-scoring-service';
import { loadAgariScoringService } from '../src/scoring/agari/agari-wasm-loader';
import {
  BASE_CONDITIONS,
  closedInput,
  instance,
  standardScoredResult,
} from './agari-test-fixtures';

function recognizedStructure(): RecognizedStructure {
  const { input } = closedInput();
  return {
    completedHand: input.completedHand,
    meldGroups: [],
    doraIndicators: [instance('dora-1', '9m')],
  };
}

function closedStructure(kinds: readonly TileKind[]): RecognizedStructure {
  return {
    completedHand: kinds.map((kind, index) => instance(`shape-${index}`, kind)),
    meldGroups: [],
    doraIndicators: [],
  };
}

function draft(overrides: Partial<ScoringDraft> = {}): ScoringDraft {
  const structure = recognizedStructure();
  return {
    structure,
    winningTileId: structure.completedHand[13].id,
    conditions: BASE_CONDITIONS,
    ...overrides,
  };
}

function engineWithScore(outcome: AgariScoreOutcomeV1): AgariEngineV1 {
  return {
    scoreHand: vi.fn(() => outcome),
    validateWinningShape: vi.fn(() => ({ status: 'winning' } as const)),
  };
}

function caughtScoringError(operation: () => unknown): ScoringError {
  try {
    operation();
  } catch (error) {
    return error as ScoringError;
  }
  throw new Error('expected operation to throw');
}

describe('Agari ScoringService', () => {
  it('uses the dedicated shape API and maps winning/non-winning outcomes', () => {
    const validateWinningShape = vi
      .fn<AgariEngineV1['validateWinningShape']>()
      .mockReturnValueOnce({ status: 'winning' })
      .mockReturnValueOnce({ status: 'not-winning-shape' });
    const service = createAgariScoringService({
      scoreHand: vi.fn(() => ({ status: 'no-yaku' } as const)),
      validateWinningShape,
    });
    const structure = recognizedStructure();

    expect(service.validateWinningStructure(structure)).toEqual({ kind: 'valid' });
    expect(service.validateWinningStructure(structure)).toEqual({
      kind: 'not-winning-shape',
    });
    expect(validateWinningShape).toHaveBeenCalledTimes(2);
    expect(validateWinningShape.mock.calls[0]?.[0]).not.toContain('dora');
  });

  it('maps ordinary, chiitoitsu, kokushi, and non-winning shape API outcomes', () => {
    const validateWinningShape = vi
      .fn<AgariEngineV1['validateWinningShape']>()
      .mockReturnValueOnce({ status: 'winning' })
      .mockReturnValueOnce({ status: 'winning' })
      .mockReturnValueOnce({ status: 'winning' })
      .mockReturnValueOnce({ status: 'not-winning-shape' });
    const service = createAgariScoringService({
      scoreHand: vi.fn(() => ({ status: 'no-yaku' } as const)),
      validateWinningShape,
    });

    const ordinary = closedStructure([
      '1m', '2m', '3m', '4p', '5p', '6p', '7s', '8s', '9s',
      '1z', '1z', '1z', '2z', '2z',
    ]);
    const chiitoitsu = closedStructure([
      '1m', '1m', '2m', '2m', '3p', '3p', '4p', '4p',
      '5s', '5s', '6s', '6s', '7z', '7z',
    ]);
    const kokushi = closedStructure([
      '1m', '9m', '1p', '9p', '1s', '9s', '1z', '2z', '3z',
      '4z', '5z', '6z', '7z', '7z',
    ]);
    const nonWinning = closedStructure([
      '1m', '2m', '3m', '4m', '5p', '6p', '7p', '8p', '9s',
      '1z', '2z', '3z', '5z', '5z',
    ]);

    expect(service.validateWinningStructure(ordinary)).toEqual({ kind: 'valid' });
    expect(service.validateWinningStructure(chiitoitsu)).toEqual({ kind: 'valid' });
    expect(service.validateWinningStructure(kokushi)).toEqual({ kind: 'valid' });
    expect(service.validateWinningStructure(nonWinning)).toEqual({
      kind: 'not-winning-shape',
    });
    expect(validateWinningShape).toHaveBeenCalledTimes(4);
  });

  it('returns product-owned structural issues before invoking Agari', () => {
    const engine = engineWithScore({ status: 'no-yaku' });
    const service = createAgariScoringService(engine);
    const structure = recognizedStructure();
    const invalid = {
      ...structure,
      completedHand: structure.completedHand.slice(0, 13),
    };

    expect(service.validateWinningStructure(invalid)).toEqual({
      kind: 'invalid-structure',
      issues: [{ kind: 'completed-hand-count' }],
    });
    expect(engine.validateWinningShape).not.toHaveBeenCalled();
  });

  it('normalizes incomplete and contradictory drafts before scoring', () => {
    const engine = engineWithScore({ status: 'no-yaku' });
    const service = createAgariScoringService(engine);

    expect(
      service.preview(
        draft({ conditions: { ...BASE_CONDITIONS, winMethod: null } }),
        closedInput().ruleProfile,
      ),
    ).toEqual({ kind: 'incomplete', missing: ['win-method'] });

    expect(
      service.preview(
        draft({ conditions: { ...BASE_CONDITIONS, ippatsu: true } }),
        closedInput().ruleProfile,
      ),
    ).toEqual({
      kind: 'invalid-input',
      issues: [{ kind: 'contradictory-conditions' }],
    });
    expect(engine.scoreHand).not.toHaveBeenCalled();
  });

  it('maps stable not-winning-shape and no-yaku outcomes to preview states', () => {
    const notWinning = createAgariScoringService(
      engineWithScore({ status: 'not-winning-shape' }),
    );
    const noYaku = createAgariScoringService(engineWithScore({ status: 'no-yaku' }));
    const profile = closedInput().ruleProfile;

    expect(notWinning.preview(draft(), profile)).toEqual({
      kind: 'invalid-winning-shape',
    });
    expect(noYaku.preview(draft(), profile)).toEqual({ kind: 'no-yaku' });
  });

  it('shares the same strict request/evaluation semantics between preview and calculate', () => {
    const requests: unknown[] = [];
    const engine: AgariEngineV1 = {
      scoreHand(request) {
        requests.push(request);
        return { status: 'scored', result: standardScoredResult() };
      },
      validateWinningShape() {
        return { status: 'winning' };
      },
    };
    const service = createAgariScoringService(engine);
    const { input, ruleProfile } = closedInput();

    const preview = service.preview(draft(), ruleProfile);
    const calculation = service.calculate(input, ruleProfile);

    expect(preview).toEqual({
      kind: 'ready',
      yaku: [{ kind: 'regular', id: 'tanyao', han: 1 }],
    });
    expect(calculation.yaku).toEqual(preview.kind === 'ready' ? preview.yaku : []);
    expect(requests).toHaveLength(2);
    expect(requests[1]).toEqual(requests[0]);
  });

  it('maps invalid-request, internal-error, and invocation failures to adapter-failure', () => {
    for (const outcome of [
      {
        status: 'invalid-request',
        error: { code: 'invalid-hand', message: 'fixture' },
      },
      {
        status: 'internal-error',
        error: { code: 'internal', message: 'fixture' },
      },
    ] as const satisfies readonly AgariScoreOutcomeV1[]) {
      const service = createAgariScoringService(engineWithScore(outcome));
      expect(caughtScoringError(() => service.preview(draft(), closedInput().ruleProfile))).toMatchObject({
        kind: 'adapter-failure',
      });
    }

    const service = createAgariScoringService({
      scoreHand() {
        throw new Error('wasm invocation failed');
      },
      validateWinningShape() {
        return { status: 'winning' };
      },
    });
    expect(caughtScoringError(() => service.preview(draft(), closedInput().ruleProfile))).toMatchObject({
      kind: 'adapter-failure',
    });
  });

  it('loads and initializes WASM asynchronously, then exposes synchronous service operations', async () => {
    let initialized = false;
    const module: AgariWasmModuleV1 = {
      async default() {
        initialized = true;
      },
      score_hand_v1() {
        if (!initialized) {
          throw new Error('not initialized');
        }
        return { status: 'scored', result: standardScoredResult() };
      },
      validate_winning_shape_v1() {
        if (!initialized) {
          throw new Error('not initialized');
        }
        return { status: 'winning' };
      },
    };

    const service = await loadAgariScoringService(async () => module);
    expect(initialized).toBe(true);
    expect(service.validateWinningStructure(recognizedStructure())).toEqual({ kind: 'valid' });
    expect(service.preview(draft(), closedInput().ruleProfile).kind).toBe('ready');
  });
});
