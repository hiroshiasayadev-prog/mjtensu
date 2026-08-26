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
  wasmProxy: boolean;
  initializationFailures: string[];
  createFloat32Tensor(data: Float32Array, dims: readonly number[]): Ort.Tensor;
}

export function providerChoiceFromLocation(): ProviderChoice {
  const value = new URLSearchParams(window.location.search).get('provider');
  switch (value) {
    case 'webgl':
    case 'wasm-simd':
    case 'wasm-threaded':
      return value;
    default:
      return 'auto';
  }
}

export async function initializeRuntime(
  choice: ProviderChoice,
  modelUrl: string,
): Promise<RuntimeHandle> {
  const candidates: SelectedProvider[] =
    choice === 'auto' ? ['webgl', 'wasm-simd', 'wasm-threaded'] : [choice];
  const failures: string[] = [];

  for (const candidate of candidates) {
    try {
      const api = apiForProvider(candidate);
      api.env.logLevel = 'warning';
      const { wasmThreads, wasmProxy } = configureEnvironment(candidate);
      const executionProvider = candidate === 'webgl' ? 'webgl' : 'wasm';
      const session = await api.InferenceSession.create(modelUrl, {
        executionProviders: [executionProvider],
        graphOptimizationLevel: 'all',
        executionMode: 'sequential',
      });
      return {
        session,
        provider: candidate,
        wasmThreads,
        wasmProxy,
        initializationFailures: failures,
        createFloat32Tensor: (data, dims) =>
          new api.Tensor('float32', data, [...dims]),
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

function configureEnvironment(
  provider: SelectedProvider,
): { wasmThreads: number; wasmProxy: boolean } {
  if (provider === 'webgl') {
    return { wasmThreads: 0, wasmProxy: false };
  }

  // Use the standard non-JSEP WASM bundle. Keep proxy disabled for the initial
  // provider benchmark so iOS does not need to import ORT through a module worker.
  ortWasm.env.wasm.proxy = false;
  ortWasm.env.wasm.simd = true;

  if (provider === 'wasm-simd') {
    ortWasm.env.wasm.numThreads = 1;
    return { wasmThreads: 1, wasmProxy: false };
  }

  if (!window.crossOriginIsolated) {
    throw new Error(
      'crossOriginIsolated is false. Serve COOP: same-origin and COEP: require-corp before testing WASM threads.',
    );
  }

  const threads = Math.max(2, Math.min(4, navigator.hardwareConcurrency || 2));
  ortWasm.env.wasm.numThreads = threads;
  return { wasmThreads: threads, wasmProxy: false };
}
