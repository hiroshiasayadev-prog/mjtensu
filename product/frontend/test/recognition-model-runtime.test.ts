import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { waitFor } from '@testing-library/dom';
import {
  createRecognitionModelAssets,
  type RecognitionModelArtifactFetcher,
  type RecognitionModelArtifactHasher,
  type RecognitionModelArtifactStore,
} from '@/recognition/model-runtime/assets';
import { validateRecognitionModelSetManifest } from '@/recognition/model-runtime/manifest';
import { getRecognitionClassifierNormalization } from '@/recognition/model-runtime/runtime-specs';
import {
  createRecognitionModelRuntime,
  type RecognitionInferenceSession,
  type RecognitionInferenceSessionFactory,
} from '@/recognition/model-runtime/runtime';
import type {
  ExecutionProvider,
  RecognitionModelRole,
  RecognitionModelSetManifest,
} from '@/recognition/model-runtime/types';
import { describe, expect, it } from 'vitest';

const modelBytes = {
  detector: new Uint8Array([1]),
  'tile-classifier': new Uint8Array([2]),
  'red-five-classifier': new Uint8Array([3]),
} as const satisfies Record<RecognitionModelRole, Uint8Array>;

const manifest = makeManifest();

describe('recognition model manifest validation', () => {
  it('accepts the three known role/runtime-spec contracts', () => {
    expect(validateRecognitionModelSetManifest(manifest)).toEqual(manifest);
  });

  it('binds classifier normalization to the selected production checkpoints', () => {
    expect(getRecognitionClassifierNormalization('c8-tile-35-v1')).toEqual({
      mean: [0.6815832403977466],
      std: [0.2725553681973969],
    });
    expect(getRecognitionClassifierNormalization('c8-red-five-v1')).toEqual({
      mean: [0.66025093606229934, 0.69172744263865471, 0.6489080530422624],
      std: [0.30491469480493394, 0.24924454491506576, 0.27107025824445752],
    });
  });

  it('normalizes unsupported schema, role/spec, provider, and integrity declarations', () => {
    expectThrown(
      () =>
        validateRecognitionModelSetManifest({
          ...manifest,
          schemaVersion: 2,
        }),
      { kind: 'model-incompatible', model: 'detector' },
    );
    expectThrown(
      () =>
        validateRecognitionModelSetManifest({
          ...manifest,
          models: {
            ...manifest.models,
            'tile-classifier': {
              ...manifest.models['tile-classifier'],
              runtimeSpec: 'nanodet-plus-m-320-v1',
            },
          },
        }),
      { kind: 'model-incompatible', model: 'tile-classifier' },
    );
    expectThrown(
      () =>
        validateRecognitionModelSetManifest({
          ...manifest,
          models: {
            ...manifest.models,
            'red-five-classifier': {
              ...manifest.models['red-five-classifier'],
              providerPreference: ['wasm-simd', 'wasm-simd'],
            },
          },
        }),
      { kind: 'model-incompatible', model: 'red-five-classifier' },
    );
    expectThrown(
      () =>
        validateRecognitionModelSetManifest({
          ...manifest,
          models: {
            ...manifest.models,
            detector: {
              ...manifest.models.detector,
              sha256: 'not-a-sha256',
            },
          },
        }),
      { kind: 'model-incompatible', model: 'detector' },
    );
  });
});

describe('recognition model asset acquisition', () => {
  it('deduplicates concurrent prefetch and runtime resolution without creating sessions during prefetch', async () => {
    const store = new MemoryArtifactStore();
    const fetcher = new GatedArtifactFetcher(modelBytes);
    const assets = createRecognitionModelAssets({
      store,
      fetcher,
      hasher: new FixtureHasher(),
    });
    const sessionFactory = new FakeSessionFactory();
    const runtime = createRecognitionModelRuntime({
      manifest,
      assets,
      sessions: sessionFactory,
    });

    const prefetchA = assets.prefetch(manifest);
    const prefetchB = assets.prefetch(manifest);
    const initialize = runtime.initialize();

    await waitFor(() => expect(fetcher.calls).toHaveLength(3));
    expect(sessionFactory.attempts).toHaveLength(0);

    fetcher.release();
    await Promise.all([prefetchA, prefetchB, initialize]);

    expect(fetcher.calls).toHaveLength(3);
    expect(sessionFactory.attempts).toHaveLength(3);
  });

  it('rejects a fetched integrity mismatch and retries a stale cached artifact from the source', async () => {
    const store = new MemoryArtifactStore();
    const fetcher = new MapArtifactFetcher(modelBytes);
    const assets = createRecognitionModelAssets({
      store,
      fetcher,
      hasher: new FixtureHasher(),
    });
    const detectorIdentity = `detector:${manifest.models.detector.sha256}`;
    await store.write(detectorIdentity, new Uint8Array([99]));

    await expect(assets.resolve(manifest, 'detector')).resolves.toMatchObject({
      role: 'detector',
      sha256: manifest.models.detector.sha256,
    });
    expect(store.removed).toContain(detectorIdentity);
    expect(fetcher.calls).toEqual(['/models/detector.onnx']);

    const badFetcher = new MapArtifactFetcher({
      ...modelBytes,
      detector: new Uint8Array([9]),
    });
    const badAssets = createRecognitionModelAssets({
      store: new MemoryArtifactStore(),
      fetcher: badFetcher,
      hasher: new FixtureHasher(),
    });
    await expect(badAssets.resolve(manifest, 'detector')).rejects.toEqual({
      kind: 'model-integrity-failure',
      model: 'detector',
    });
  });

  it('normalizes source acquisition failure as model-asset-unavailable', async () => {
    const assets = createRecognitionModelAssets({
      store: new MemoryArtifactStore(),
      fetcher: {
        async fetch() {
          throw new Error('offline');
        },
      },
      hasher: new FixtureHasher(),
    });

    await expect(assets.resolve(manifest, 'detector')).rejects.toEqual({
      kind: 'model-asset-unavailable',
      model: 'detector',
    });
  });

  it('does not permanently cache a failed background prefetch acquisition', async () => {
    const fetcher = new FailOnceArtifactFetcher(modelBytes, '/models/detector.onnx');
    const assets = createRecognitionModelAssets({
      store: new MemoryArtifactStore(),
      fetcher,
      hasher: new FixtureHasher(),
    });

    await expect(assets.prefetch(manifest)).rejects.toEqual({
      kind: 'model-asset-unavailable',
      model: 'detector',
    });
    await expect(assets.resolve(manifest, 'detector')).resolves.toMatchObject({
      role: 'detector',
    });
    expect(fetcher.calls.filter((url) => url === '/models/detector.onnx')).toHaveLength(2);
  });
});

describe('recognition model runtime initialization', () => {
  it('keeps ONNX Runtime implementation types out of the Recognition public entry point', () => {
    const publicEntry = readFileSync(
      resolve(process.cwd(), 'src/recognition/index.ts'),
      'utf8',
    );
    expect(publicEntry).not.toMatch(/onnxruntime|InferenceSession|Ort\./);
  });

  it('deduplicates concurrent initialize calls and preserves successful idempotence', async () => {
    const sessions = new GatedSessionFactory();
    const runtime = createRecognitionModelRuntime({
      manifest,
      assets: createStaticAssets(),
      sessions,
    });

    const first = runtime.initialize();
    const concurrent = runtime.initialize();
    expect(concurrent).toBe(first);

    await waitFor(() => expect(sessions.attempts).toHaveLength(1));
    sessions.release();
    await Promise.all([first, concurrent]);

    expect(sessions.attempts).toHaveLength(3);
    await runtime.initialize();
    expect(sessions.attempts).toHaveLength(3);
  });

  it('applies the configured provider fallback independently per role and exposes semantic diagnostics', async () => {
    const sessions = new FakeSessionFactory();
    sessions.fail('detector', 'wasm-simd');
    sessions.fail('tile-classifier', 'wasm-simd');
    sessions.fail('tile-classifier', 'wasm-threaded');
    const runtime = createRecognitionModelRuntime({
      manifest,
      assets: createStaticAssets(),
      sessions,
    });

    await runtime.initialize();

    expect(runtime.getInitializedModel('detector').provider).toBe('wasm-threaded');
    expect(runtime.getInitializedModel('tile-classifier').provider).toBe('webgl');
    expect(runtime.getInitializedModel('red-five-classifier').provider).toBe('wasm-simd');
    expect(runtime.getDiagnostics()).toEqual([
      {
        role: 'detector',
        runtimeSpec: 'nanodet-plus-m-320-v1',
        selectedProvider: 'wasm-threaded',
        failedProviders: ['wasm-simd'],
      },
      {
        role: 'tile-classifier',
        runtimeSpec: 'c8-tile-35-v1',
        selectedProvider: 'webgl',
        failedProviders: ['wasm-simd', 'wasm-threaded'],
      },
      {
        role: 'red-five-classifier',
        runtimeSpec: 'c8-red-five-v1',
        selectedProvider: 'wasm-simd',
        failedProviders: [],
      },
    ]);
  });

  it('normalizes provider exhaustion and permits a later healthy retry', async () => {
    const sessions = new FakeSessionFactory();
    sessions.fail('tile-classifier', 'wasm-simd');
    sessions.fail('tile-classifier', 'wasm-threaded');
    sessions.fail('tile-classifier', 'webgl');
    const runtime = createRecognitionModelRuntime({
      manifest,
      assets: createStaticAssets(),
      sessions,
    });

    await expect(runtime.initialize()).rejects.toEqual({
      kind: 'execution-provider-unavailable',
      model: 'tile-classifier',
    });

    const failedAttemptDetector = sessions.created[0];
    expect(failedAttemptDetector?.disposeCount).toBe(1);

    sessions.clearFailures();
    await expect(runtime.initialize()).resolves.toBeUndefined();
    expect(runtime.getInitializedModel('detector').provider).toBe('wasm-simd');
    expect(sessions.created).toHaveLength(4);
  });

  it('owns healthy sessions until runtime disposal and disposes them exactly once', async () => {
    const sessions = new FakeSessionFactory();
    const runtime = createRecognitionModelRuntime({
      manifest,
      assets: createStaticAssets(),
      sessions,
    });

    await runtime.initialize();
    const initialized = [
      runtime.getInitializedModel('detector').session as FakeSession,
      runtime.getInitializedModel('tile-classifier').session as FakeSession,
      runtime.getInitializedModel('red-five-classifier').session as FakeSession,
    ];

    await runtime.initialize();
    expect(initialized.map((session) => session.disposeCount)).toEqual([0, 0, 0]);

    await runtime.dispose();
    await runtime.dispose();
    expect(initialized.map((session) => session.disposeCount)).toEqual([1, 1, 1]);
    await expect(runtime.initialize()).rejects.toThrow(/disposed/);
  });
});

function makeManifest(): RecognitionModelSetManifest {
  const providerPreference = [
    'wasm-simd',
    'wasm-threaded',
    'webgl',
  ] as const satisfies readonly ExecutionProvider[];

  return {
    schemaVersion: 1,
    modelSetVersion: 'fixture-v1',
    models: {
      detector: {
        role: 'detector',
        url: '/models/detector.onnx',
        sha256: fixtureHash(modelBytes.detector),
        runtimeSpec: 'nanodet-plus-m-320-v1',
        providerPreference,
      },
      'tile-classifier': {
        role: 'tile-classifier',
        url: '/models/tile-classifier.onnx',
        sha256: fixtureHash(modelBytes['tile-classifier']),
        runtimeSpec: 'c8-tile-35-v1',
        providerPreference,
      },
      'red-five-classifier': {
        role: 'red-five-classifier',
        url: '/models/red-five-classifier.onnx',
        sha256: fixtureHash(modelBytes['red-five-classifier']),
        runtimeSpec: 'c8-red-five-v1',
        providerPreference,
      },
    },
  };
}

function fixtureHash(bytes: Uint8Array): string {
  const value = bytes[0] ?? 0;
  return value.toString(16).padStart(2, '0').repeat(32);
}

class FixtureHasher implements RecognitionModelArtifactHasher {
  async sha256(bytes: Uint8Array): Promise<string> {
    return fixtureHash(bytes);
  }
}

class MemoryArtifactStore implements RecognitionModelArtifactStore {
  readonly removed: string[] = [];
  private readonly values = new Map<string, Uint8Array>();

  async read(identity: string): Promise<Uint8Array | null> {
    const value = this.values.get(identity);
    return value === undefined ? null : new Uint8Array(value);
  }

  async write(identity: string, bytes: Uint8Array): Promise<void> {
    this.values.set(identity, new Uint8Array(bytes));
  }

  async remove(identity: string): Promise<void> {
    this.removed.push(identity);
    this.values.delete(identity);
  }
}

class MapArtifactFetcher implements RecognitionModelArtifactFetcher {
  readonly calls: string[] = [];

  constructor(
    private readonly values: Readonly<Record<RecognitionModelRole, Uint8Array>>,
  ) {}

  async fetch(url: string): Promise<Uint8Array> {
    this.calls.push(url);
    const role = roleForUrl(url);
    return new Uint8Array(this.values[role]);
  }
}

class FailOnceArtifactFetcher extends MapArtifactFetcher {
  private failed = false;

  constructor(
    values: Readonly<Record<RecognitionModelRole, Uint8Array>>,
    private readonly failUrl: string,
  ) {
    super(values);
  }

  override async fetch(url: string): Promise<Uint8Array> {
    if (url === this.failUrl && !this.failed) {
      this.failed = true;
      this.calls.push(url);
      throw new Error('temporary fetch failure');
    }
    return super.fetch(url);
  }
}

class GatedArtifactFetcher extends MapArtifactFetcher {
  private readonly gate: Promise<void>;
  private releaseGate!: () => void;

  constructor(values: Readonly<Record<RecognitionModelRole, Uint8Array>>) {
    super(values);
    this.gate = new Promise<void>((resolve) => {
      this.releaseGate = resolve;
    });
  }

  override async fetch(url: string): Promise<Uint8Array> {
    const pending = super.fetch(url);
    await this.gate;
    return pending;
  }

  release(): void {
    this.releaseGate();
  }
}

function roleForUrl(url: string): RecognitionModelRole {
  if (url.includes('red-five')) {
    return 'red-five-classifier';
  }
  if (url.includes('tile-classifier')) {
    return 'tile-classifier';
  }
  return 'detector';
}

function createStaticAssets() {
  return createRecognitionModelAssets({
    store: new MemoryArtifactStore(),
    fetcher: new MapArtifactFetcher(modelBytes),
    hasher: new FixtureHasher(),
  });
}

class FakeSession implements RecognitionInferenceSession {
  readonly inputNames = ['input'] as const;
  readonly outputNames = ['output'] as const;
  disposeCount = 0;

  createFloat32Tensor(data: Float32Array, dims: readonly number[]): unknown {
    return { data, dims };
  }

  async run(): Promise<Readonly<Record<string, unknown>>> {
    return {};
  }

  async dispose(): Promise<void> {
    this.disposeCount += 1;
  }
}

class FakeSessionFactory implements RecognitionInferenceSessionFactory {
  readonly attempts: Array<{
    readonly role: RecognitionModelRole;
    readonly provider: ExecutionProvider;
  }> = [];
  readonly created: FakeSession[] = [];
  private readonly failures = new Set<string>();

  fail(role: RecognitionModelRole, provider: ExecutionProvider): void {
    this.failures.add(`${role}:${provider}`);
  }

  clearFailures(): void {
    this.failures.clear();
  }

  async create(options: {
    readonly role: RecognitionModelRole;
    readonly provider: ExecutionProvider;
  }): Promise<RecognitionInferenceSession> {
    this.attempts.push({ role: options.role, provider: options.provider });
    if (this.failures.has(`${options.role}:${options.provider}`)) {
      throw new Error('provider unavailable');
    }
    const session = new FakeSession();
    this.created.push(session);
    return session;
  }
}

class GatedSessionFactory extends FakeSessionFactory {
  private readonly gate: Promise<void>;
  private releaseGate!: () => void;
  private first = true;

  constructor() {
    super();
    this.gate = new Promise<void>((resolve) => {
      this.releaseGate = resolve;
    });
  }

  override async create(options: {
    readonly role: RecognitionModelRole;
    readonly provider: ExecutionProvider;
  }): Promise<RecognitionInferenceSession> {
    if (this.first) {
      this.first = false;
      const pending = super.create(options);
      await this.gate;
      return pending;
    }
    return super.create(options);
  }

  release(): void {
    this.releaseGate();
  }
}

function expectThrown(
  action: () => unknown,
  expected: Readonly<Record<string, unknown>>,
): void {
  try {
    action();
  } catch (error) {
    expect(error).toEqual(expected);
    return;
  }
  throw new Error('Expected action to throw.');
}
