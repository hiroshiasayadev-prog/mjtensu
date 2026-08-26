import type { ScoringService } from '@/scoring';

import type { AgariEngineV1, AgariWasmModuleV1 } from './agari-abi';
import { createAgariScoringService } from './agari-scoring-service';

export const PRODUCTION_AGARI_WASM_MODULE_PATH =
  '@agari-wasm/agari_wasm.js';

export type AgariWasmModuleLoader = () => Promise<AgariWasmModuleV1>;

export async function loadAgariScoringService(
  loadModule: AgariWasmModuleLoader,
): Promise<ScoringService> {
  let module: AgariWasmModuleV1;
  try {
    module = await loadModule();
    await module.default();
  } catch (cause) {
    throw {
      kind: 'adapter-failure',
      cause,
    } as const;
  }

  const engine: AgariEngineV1 = {
    scoreHand(request) {
      return module.score_hand_v1(request);
    },
    validateWinningShape(hand) {
      return module.validate_winning_shape_v1(hand);
    },
  };

  return createAgariScoringService(engine);
}

export async function loadProductionScoringService(): Promise<ScoringService> {
  return loadAgariScoringService(async () => {
    // Keep this import statically analyzable so Vite bundles the committed
    // canonical package (including its relative WASM asset) from repo-root
    // vendor/agari-wasm. The exported path above documents the production
    // module identity for tests/tooling; the literal here must match it.
    const loaded: unknown = await import('@agari-wasm/agari_wasm.js');
    return assertAgariWasmModule(loaded);
  });
}

function assertAgariWasmModule(value: unknown): AgariWasmModuleV1 {
  if (
    typeof value !== 'object' ||
    value === null ||
    !('default' in value) ||
    typeof value.default !== 'function' ||
    !('score_hand_v1' in value) ||
    typeof value.score_hand_v1 !== 'function' ||
    !('validate_winning_shape_v1' in value) ||
    typeof value.validate_winning_shape_v1 !== 'function'
  ) {
    throw new TypeError('loaded Agari WASM module does not expose the stable V1 ABI');
  }

  return value as AgariWasmModuleV1;
}
