import * as ortWasm from 'onnxruntime-web/wasm';
import * as ortWebgl from 'onnxruntime-web/webgl';
import type * as Ort from 'onnxruntime-web';

export type ProviderChoice = 'auto' | 'webgl' | 'wasm-simd' | 'wasm-threaded';
export type SelectedProvider = Exclude<ProviderChoice, 'auto'>;

interface OrtApi {
  env: typeof ortWasm.env;
  InferenceSession: typeof ortWasm.InferenceSession;
  Tensor: typeof ortWasm.Tensor;
}

export interface RuntimeHandle {
  session: Ort.InferenceSession;
  provider: SelectedProvider;
  wasmThreads: number;
  initializationFailures: string[];
  createFloat32Tensor(data: Float32Array, dims: readonly number[]): Ort.Tensor;
}

export function providerChoiceFromLocation(): ProviderChoice {
  const value = new URLSearchParams(window.location.search).get('provider');
  return value === 'webgl' || value === 'wasm-simd' || value === 'wasm-threaded'
    ? value
    : 'auto';
}

export async function initializeRuntime(
  choice: ProviderChoice,
  modelUrl: string,
): Promise<RuntimeHandle> {
  const candidates: SelectedProvider[] = choice === 'auto'
    ? ['webgl', 'wasm-simd', 'wasm-threaded']
    : [choice];
  const failures: string[] = [];
  for (const candidate of candidates) {
    try {
      const api = apiForProvider(candidate);
      api.env.logLevel = 'warning';
      const wasmThreads = configure(candidate);
      const session = await api.InferenceSession.create(modelUrl, {
        executionProviders: [candidate === 'webgl' ? 'webgl' : 'wasm'],
        graphOptimizationLevel: 'all',
        executionMode: 'sequential',
      });
      return {
        session,
        provider: candidate,
        wasmThreads,
        initializationFailures: failures,
        createFloat32Tensor: (data, dims) => new api.Tensor('float32', data, [...dims]),
      };
    } catch (error) {
      const message = `${candidate}: ${error instanceof Error ? error.message : String(error)}`;
      failures.push(message);
      console.warn('[runtime-init-failed]', message);
    }
  }
  throw new Error(`No execution provider initialized. ${failures.join(' | ')}`);
}

function apiForProvider(provider: SelectedProvider): OrtApi {
  return provider === 'webgl' ? ortWebgl : ortWasm;
}

function configure(provider: SelectedProvider): number {
  if (provider === 'webgl') return 0;
  ortWasm.env.wasm.proxy = false;
  ortWasm.env.wasm.simd = true;
  if (provider === 'wasm-simd') {
    ortWasm.env.wasm.numThreads = 1;
    return 1;
  }
  if (!window.crossOriginIsolated) {
    throw new Error('WASM threads require cross-origin isolation.');
  }
  const threads = Math.max(2, Math.min(4, navigator.hardwareConcurrency || 2));
  ortWasm.env.wasm.numThreads = threads;
  return threads;
}
