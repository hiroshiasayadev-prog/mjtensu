import * as ortWasm from 'onnxruntime-web/wasm';
import * as ortWebgl from 'onnxruntime-web/webgl';
import type * as Ort from 'onnxruntime-web';

import type {
  RecognitionInferenceSession,
  RecognitionInferenceSessionFactory,
} from './runtime';
import type { ExecutionProvider } from './types';

interface OrtApi {
  readonly env: typeof ortWasm.env;
  readonly InferenceSession: typeof ortWasm.InferenceSession;
  readonly Tensor: typeof ortWasm.Tensor;
}

export interface OnnxSessionFactoryEnvironment {
  readonly crossOriginIsolated: boolean;
  readonly hardwareConcurrency: number;
}

export interface OnnxSessionFactoryOptions {
  readonly environment?: OnnxSessionFactoryEnvironment;
}

export function createOnnxRecognitionSessionFactory(
  options: OnnxSessionFactoryOptions = {},
): RecognitionInferenceSessionFactory {
  const environment = options.environment ?? browserEnvironment();

  return {
    async create({ artifact, provider }) {
      const api = apiForProvider(provider);
      configureProvider(provider, environment);
      api.env.logLevel = 'warning';

      const session = await api.InferenceSession.create(artifact, {
        executionProviders: [provider === 'webgl' ? 'webgl' : 'wasm'],
        graphOptimizationLevel: 'all',
        executionMode: 'sequential',
      });
      return new OnnxRecognitionInferenceSession(session, api.Tensor);
    },
  };
}

class OnnxRecognitionInferenceSession implements RecognitionInferenceSession {
  constructor(
    private readonly session: Ort.InferenceSession,
    private readonly Tensor: typeof ortWasm.Tensor,
  ) {}

  get inputNames(): readonly string[] {
    return this.session.inputNames;
  }

  get outputNames(): readonly string[] {
    return this.session.outputNames;
  }

  createFloat32Tensor(data: Float32Array, dims: readonly number[]): unknown {
    return new this.Tensor('float32', data, [...dims]);
  }

  async run(
    feeds: Readonly<Record<string, unknown>>,
  ): Promise<Readonly<Record<string, unknown>>> {
    return this.session.run(feeds as Ort.InferenceSession.FeedsType);
  }

  async dispose(): Promise<void> {
    await this.session.release();
  }
}

function apiForProvider(provider: ExecutionProvider): OrtApi {
  return provider === 'webgl' ? ortWebgl : ortWasm;
}

function configureProvider(
  provider: ExecutionProvider,
  environment: OnnxSessionFactoryEnvironment,
): void {
  if (provider === 'webgl') {
    return;
  }

  ortWasm.env.wasm.proxy = false;
  ortWasm.env.wasm.simd = true;
  if (provider === 'wasm-simd') {
    ortWasm.env.wasm.numThreads = 1;
    return;
  }

  if (!environment.crossOriginIsolated) {
    throw new Error('WASM threaded execution requires cross-origin isolation.');
  }
  ortWasm.env.wasm.numThreads = Math.max(
    2,
    Math.min(4, environment.hardwareConcurrency || 2),
  );
}

function browserEnvironment(): OnnxSessionFactoryEnvironment {
  return {
    crossOriginIsolated: globalThis.crossOriginIsolated === true,
    hardwareConcurrency:
      typeof navigator === 'undefined' ? 1 : navigator.hardwareConcurrency || 1,
  };
}
