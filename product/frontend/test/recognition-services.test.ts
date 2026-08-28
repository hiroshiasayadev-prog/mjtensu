import type { TileKind } from '@/domain';
import type {
  RecognitionDebugCapture,
  RecognitionEvaluationTiming,
  RecognitionFrame,
  RecognitionFrameSource,
  RecognitionPipeline,
  RealtimeRecognitionListener,
  RealtimeRecognitionUpdate,
} from '@/recognition';
import { createProductionRecognitionPipeline } from '@/recognition/production-pipeline';
import {
  createRealtimeRecognizer,
  RECOGNITION_REQUEST_CADENCE_MS,
  type RecognitionCadenceScheduler,
} from '@/recognition/realtime-recognizer';
import { createRecognitionRuntimeComposition } from '@/recognition/production-runtime';
import type {
  InitializedRecognitionModel,
  RecognitionInferenceSession,
  RecognitionModelRuntime,
  RecognitionModelRuntimeInspection,
} from '@/recognition/model-runtime/runtime';
import type {
  RecognitionModelRole,
  RecognitionRuntimeError,
} from '@/recognition/model-runtime/types';
import type { RegionDetection, TensorOutput } from '@/recognition/detector/types';
import type { FrameRecognitionSnapshot } from '@/recognition/semantics/types';
import { describe, expect, it } from 'vitest';

describe('production recognition one-frame composition', () => {
  it('composes detector, classifiers, semantic observations, and normalized preview geometry', async () => {
    const detector = new FakeInferenceSession([
      tensorOutput(new Float32Array(1), [1, 1, 1]),
    ]);
    const base = new FakeInferenceSession([
      classifierOutput(
        concatenateFloat32(
          logitsFor('1m'),
          logitsFor('invalid'),
          logitsFor('5m'),
        ),
      ),
    ]);
    const redFive = new FakeInferenceSession([
      classifierOutput(new Float32Array([0, 1])),
    ]);
    const detections: RegionDetection[] = [
      detection('hand-1', 'completed_hand', 100, 700),
      detection('hand-invalid', 'completed_hand', 200, 700),
      detection('hand-red-five', 'completed_hand', 300, 700),
    ];
    const timings: RecognitionEvaluationTiming[] = [];
    const pipeline = createProductionRecognitionPipeline({
      modelRuntime: fakeModelInspection({ detector, base, redFive }),
      detectorPostprocessor: { process: () => detections },
      platform: fakePipelinePlatform(),
      onEvaluationTiming: (timing) => timings.push(timing),
    });

    const snapshot = await pipeline.evaluate(frame());

    expect(detector.runCalls).toHaveLength(1);
    expect(base.runCalls).toHaveLength(1);
    expect(redFive.runCalls).toHaveLength(1);
    expect(tensorDims(base.runCalls[0])).toEqual([3, 1, 64, 64]);
    expect(tensorDims(redFive.runCalls[0])).toEqual([1, 3, 64, 64]);
    expect(firstTensorValue(base.runCalls[0])).toBeCloseTo(
      (128 / 255 - 0.6815832403977466) / 0.2725553681973969,
    );
    expect(firstTensorValue(redFive.runCalls[0])).toBeCloseTo(
      (128 / 255 - 0.66025093606229934) / 0.30491469480493394,
    );
    expect(snapshot.observations).toEqual([
      expect.objectContaining({
        id: 'hand-1',
        region: 'completed-hand',
        bbox: { x: 0.1, y: 0.7, width: 0.04, height: 0.08 },
        classification: { kind: 'tile', tile: { kind: '1m', red: false } },
      }),
      expect.objectContaining({
        id: 'hand-invalid',
        classification: { kind: 'invalid' },
      }),
      expect.objectContaining({
        id: 'hand-red-five',
        classification: { kind: 'tile', tile: { kind: '5m', red: true } },
      }),
    ]);
    expect(snapshot.draft.completedHand).toEqual([
      { kind: '1m', red: false },
      { kind: '5m', red: true },
    ]);
    expect(snapshot.commitEligibility).toEqual({
      kind: 'ineligible',
      reason: 'insufficient-visible-tiles',
    });
    expect(timings).toHaveLength(1);
    expect(timings[0]).toMatchObject({
      candidateCount: 3,
      redFiveCandidateCount: 1,
    });
    expect(timings[0]?.cropExtractionMs).toBeGreaterThanOrEqual(0);
    expect(timings[0]?.baseClassifierPreprocessingMs).toBeGreaterThanOrEqual(0);
    expect(timings[0]?.baseClassifierInferenceMs).toBeGreaterThanOrEqual(0);
  });

  it('does not invoke either classifier when the detector yields no candidates', async () => {
    const detector = new FakeInferenceSession([
      tensorOutput(new Float32Array(1), [1, 1, 1]),
    ]);
    const base = new FakeInferenceSession([]);
    const redFive = new FakeInferenceSession([]);
    const pipeline = createProductionRecognitionPipeline({
      modelRuntime: fakeModelInspection({ detector, base, redFive }),
      classifierNormalizationOverride: testNormalization,
      detectorPostprocessor: { process: () => [] },
      platform: fakePipelinePlatform(),
    });

    const snapshot = await pipeline.evaluate(frame());

    expect(base.runCalls).toHaveLength(0);
    expect(redFive.runCalls).toHaveLength(0);
    expect(snapshot.observations).toEqual([]);
  });

  it('captures the exact claimed detector input and output through the debug seam', async () => {
    const detectorOutput = tensorOutput(new Float32Array([0.25]), [1, 1, 1]);
    const detector = new FakeInferenceSession([detectorOutput]);
    const base = new FakeInferenceSession([]);
    const redFive = new FakeInferenceSession([]);
    const captured: RecognitionDebugCapture[] = [];
    const debugCapture = { schemaVersion: 1 } as RecognitionDebugCapture;
    const inputFrame = frame(42);
    const pipeline = createProductionRecognitionPipeline({
      modelRuntime: fakeModelInspection({ detector, base, redFive }),
      classifierNormalizationOverride: testNormalization,
      detectorPostprocessor: { process: () => [] },
      platform: fakePipelinePlatform(),
      modelSetVersion: 'fixture-model-set',
      claimDebugCapture: () => true,
      debugCaptureBuilder: (input) => {
        expect(input.frame).toBe(inputFrame);
        expect(input.detectorInput).toHaveLength(3 * 320 * 320);
        expect(input.detectorOutput).toBe(detectorOutput);
        expect(input.modelSetVersion).toBe('fixture-model-set');
        return debugCapture;
      },
      onDebugCapture: (capture) => captured.push(capture),
    });

    await pipeline.evaluate(inputFrame);

    expect(captured).toEqual([debugCapture]);
  });

  it.each([
    ['detector', 'detector'],
    ['tile-classifier', 'tile-classifier'],
    ['red-five-classifier', 'red-five-classifier'],
  ] as const)(
    'normalizes %s inference failures without exposing session exceptions as the discriminant',
    async (failingStage, expectedModel) => {
      const failure = new Error(`${failingStage} low-level failure`);
      const detector = new FakeInferenceSession(
        failingStage === 'detector'
          ? [failure]
          : [tensorOutput(new Float32Array(1), [1, 1, 1])],
      );
      const base = new FakeInferenceSession(
        failingStage === 'tile-classifier'
          ? [failure]
          : [classifierOutput(logitsFor('5m'))],
      );
      const redFive = new FakeInferenceSession(
        failingStage === 'red-five-classifier'
          ? [failure]
          : [classifierOutput(new Float32Array([1, 0]))],
      );
      const pipeline = createProductionRecognitionPipeline({
        modelRuntime: fakeModelInspection({ detector, base, redFive }),
        classifierNormalizationOverride: testNormalization,
        detectorPostprocessor: {
          process: () => [detection('candidate', 'completed_hand', 100, 700)],
        },
        platform: fakePipelinePlatform(),
      });

      await expect(pipeline.evaluate(frame())).rejects.toMatchObject({
        kind: 'inference-failure',
        model: expectedModel,
        cause: failure,
      });
    },
  );

  it('maps incompatible model outputs to the model-incompatible runtime error', async () => {
    const pipeline = createProductionRecognitionPipeline({
      modelRuntime: fakeModelInspection({
        detector: new FakeInferenceSession([{}]),
        base: new FakeInferenceSession([classifierOutput(logitsFor('1m'))]),
        redFive: new FakeInferenceSession([classifierOutput(new Float32Array([1, 0]))]),
      }),
      classifierNormalizationOverride: testNormalization,
      detectorPostprocessor: { process: () => [] },
      platform: fakePipelinePlatform(),
    });

    await expect(pipeline.evaluate(frame())).rejects.toEqual({
      kind: 'model-incompatible',
      model: 'detector',
    });
  });
});

describe('recognition runtime lifecycle composition', () => {
  it('exposes actual selected model providers for target-device diagnostics', () => {
    const modelRuntime: RecognitionModelRuntime & RecognitionModelRuntimeInspection = {
      async initialize() {},
      async dispose() {},
      getInitializedModel() {
        throw new Error('not needed');
      },
      getDiagnostics() {
        return [
          {
            role: 'detector',
            runtimeSpec: 'nanodet-plus-m-320-v1',
            selectedProvider: 'webgl',
            failedProviders: ['wasm-simd', 'wasm-threaded'],
          },
          {
            role: 'tile-classifier',
            runtimeSpec: 'c8-tile-35-v1',
            selectedProvider: 'wasm-simd',
            failedProviders: [],
          },
          {
            role: 'red-five-classifier',
            runtimeSpec: 'c8-red-five-v1',
            selectedProvider: 'wasm-simd',
            failedProviders: [],
          },
        ];
      },
    };
    const runtime = createRecognitionRuntimeComposition({ modelRuntime });

    expect(runtime.getDiagnostics?.()).toEqual({
      models: modelRuntime.getDiagnostics(),
      recentEvaluations: [],
    });
  });

  it('keeps shared model sessions alive when a route-owned pipeline is disposed', async () => {
    let initialized = false;
    let modelDisposeCount = 0;
    const sessions = {
      detector: new FakeInferenceSession([]),
      base: new FakeInferenceSession([]),
      redFive: new FakeInferenceSession([]),
    };
    const modelRuntime: RecognitionModelRuntime & RecognitionModelRuntimeInspection = {
      async initialize() {
        initialized = true;
      },
      async dispose() {
        modelDisposeCount += 1;
      },
      getInitializedModel(role) {
        if (!initialized) {
          throw new Error('not initialized');
        }
        return initializedModel(role, sessionForRole(role, sessions));
      },
      getDiagnostics() {
        return [];
      },
    };
    const runtime = createRecognitionRuntimeComposition({
      modelRuntime,
      classifierNormalizationOverride: testNormalization,
    });

    expect(() => runtime.createPipeline()).toThrow('not initialized');
    await runtime.initialize();
    const first = runtime.createPipeline();
    await first.dispose();
    expect(modelDisposeCount).toBe(0);

    const second = runtime.createPipeline();
    await second.dispose();
    expect(modelDisposeCount).toBe(0);

    await runtime.dispose();
    expect(modelDisposeCount).toBe(1);
  });
});

describe('realtime recognition scheduling and stabilization', () => {
  it('requests immediately at a 100 ms cadence and never captures a queued frame while inference is active', async () => {
    const scheduler = new ManualScheduler();
    const first = deferred<FrameRecognitionSnapshot>();
    const pipeline = new FakePipeline(first.promise, Promise.resolve(eligibleSnapshot('1m')));
    const source = new CountingFrameSource();
    const listener = new RecordingListener();
    const recognizer = createRealtimeRecognizer(pipeline, { scheduler });

    const run = recognizer.start(source, listener);
    expect(scheduler.intervalMs).toBe(RECOGNITION_REQUEST_CADENCE_MS);
    expect(source.captureCount).toBe(1);
    expect(pipeline.evaluateCount).toBe(1);

    scheduler.tick();
    scheduler.tick();
    expect(source.captureCount).toBe(1);
    expect(pipeline.evaluateCount).toBe(1);

    first.resolve(eligibleSnapshot('1m'));
    await flushPromises();
    expect(listener.updates.map(({ kind }) => kind)).toEqual(['stabilizing']);

    scheduler.tick();
    await flushPromises();
    expect(source.captureCount).toBe(2);
    expect(pipeline.evaluateCount).toBe(2);

    run.stop();
    expect(pipeline.disposeCount).toBe(0);
    await recognizer.dispose();
    expect(pipeline.disposeCount).toBe(1);
  });

  it('keeps a single physical evaluation across run replacement and does not capture the new run until the stale evaluation finishes', async () => {
    const scheduler = new ManualScheduler();
    const pending = deferred<FrameRecognitionSnapshot>();
    const pipeline = new FakePipeline(
      pending.promise,
      Promise.resolve(eligibleSnapshot('3m')),
    );
    const firstSource = new CountingFrameSource();
    const secondSource = new CountingFrameSource();
    const listener = new RecordingListener();
    const recognizer = createRealtimeRecognizer(pipeline, { scheduler });

    recognizer.start(firstSource, listener);
    recognizer.start(secondSource, listener);
    expect(pipeline.evaluateCount).toBe(1);
    expect(firstSource.captureCount).toBe(1);
    expect(secondSource.captureCount).toBe(0);

    scheduler.tick();
    expect(secondSource.captureCount).toBe(0);

    pending.resolve(eligibleSnapshot('9m'));
    await flushPromises();
    expect(listener.updates).toEqual([]);

    scheduler.tick();
    await flushPromises();
    expect(secondSource.captureCount).toBe(1);
    expect(pipeline.evaluateCount).toBe(2);
    expect(listener.updates.map(({ kind }) => kind)).toEqual(['stabilizing']);

    await recognizer.dispose();
  });

  it('emits scanning for normal ineligible content without treating it as a runtime error', async () => {
    const scheduler = new ManualScheduler();
    const pipeline = new FakePipeline(Promise.resolve(ineligibleSnapshot()));
    const listener = new RecordingListener();
    const recognizer = createRealtimeRecognizer(pipeline, { scheduler });

    recognizer.start(new CountingFrameSource(), listener);
    await flushPromises();

    expect(listener.updates.map(({ kind }) => kind)).toEqual(['scanning']);
    expect(listener.errors).toEqual([]);
    await recognizer.dispose();
  });

  it('confirms exactly once after three equivalent completed evaluations and reset opens a fresh boundary', async () => {
    const scheduler = new ManualScheduler();
    const pipeline = new FakePipeline(Promise.resolve(eligibleSnapshot('5m')));
    const listener = new RecordingListener();
    const recognizer = createRealtimeRecognizer(pipeline, { scheduler });

    recognizer.start(new CountingFrameSource(), listener);
    await flushPromises();
    scheduler.tick();
    await flushPromises();
    scheduler.tick();
    await flushPromises();

    expect(listener.updates.map(({ kind }) => kind)).toEqual([
      'stabilizing',
      'stabilizing',
      'confirmed',
    ]);
    const firstConfirmation = listener.updates[2];
    expect(firstConfirmation?.kind).toBe('confirmed');
    if (firstConfirmation?.kind !== 'confirmed') {
      throw new Error('expected confirmation');
    }
    const firstIds = allTileIds(firstConfirmation.result);
    expect(new Set(firstIds).size).toBe(firstIds.length);

    scheduler.tick();
    await flushPromises();
    expect(pipeline.evaluateCount).toBe(3);

    recognizer.reset();
    scheduler.tick();
    await flushPromises();
    scheduler.tick();
    await flushPromises();
    scheduler.tick();
    await flushPromises();

    const confirmations = listener.updates.filter(
      (update): update is Extract<RealtimeRecognitionUpdate, { kind: 'confirmed' }> =>
        update.kind === 'confirmed',
    );
    expect(confirmations).toHaveLength(2);
    const secondIds = allTileIds(confirmations[1]?.result ?? confirmations[0].result);
    expect(secondIds.every((id) => !firstIds.includes(id))).toBe(true);

    await recognizer.dispose();
  });

  it('drops an in-flight result after stop and never disposes the pipeline merely because the route run stops', async () => {
    const scheduler = new ManualScheduler();
    const pending = deferred<FrameRecognitionSnapshot>();
    const pipeline = new FakePipeline(pending.promise);
    const source = new CountingFrameSource();
    const listener = new RecordingListener();
    const recognizer = createRealtimeRecognizer(pipeline, { scheduler });

    const run = recognizer.start(source, listener);
    run.stop();
    pending.resolve(eligibleSnapshot('1m'));
    await flushPromises();
    scheduler.tick();
    await flushPromises();

    expect(listener.updates).toEqual([]);
    expect(listener.errors).toEqual([]);
    expect(source.captureCount).toBe(1);
    expect(pipeline.disposeCount).toBe(0);

    await recognizer.dispose();
    expect(pipeline.disposeCount).toBe(1);
  });

  it('drops a pre-reset in-flight result and starts stabilization again only from later frames', async () => {
    const scheduler = new ManualScheduler();
    const pending = deferred<FrameRecognitionSnapshot>();
    const pipeline = new FakePipeline(
      pending.promise,
      Promise.resolve(eligibleSnapshot('2m')),
    );
    const listener = new RecordingListener();
    const recognizer = createRealtimeRecognizer(pipeline, { scheduler });

    recognizer.start(new CountingFrameSource(), listener);
    recognizer.reset();
    pending.resolve(eligibleSnapshot('9m'));
    await flushPromises();
    expect(listener.updates).toEqual([]);

    scheduler.tick();
    await flushPromises();
    expect(listener.updates.map(({ kind }) => kind)).toEqual(['stabilizing']);

    await recognizer.dispose();
  });

  it('reports normalized pipeline failures through onError and stops the failed run', async () => {
    const scheduler = new ManualScheduler();
    const error: RecognitionRuntimeError = {
      kind: 'inference-failure',
      model: 'tile-classifier',
      cause: new Error('runtime failed'),
    };
    const pipeline = new FakePipeline(Promise.reject(error));
    const source = new CountingFrameSource();
    const listener = new RecordingListener();
    const recognizer = createRealtimeRecognizer(pipeline, { scheduler });

    recognizer.start(source, listener);
    await flushPromises();

    expect(listener.errors).toEqual([error]);
    expect(scheduler.cancelCount).toBe(1);
    scheduler.tick();
    await flushPromises();
    expect(source.captureCount).toBe(1);

    await recognizer.dispose();
  });
});

const testNormalization = {
  base: { mean: [0], std: [1] },
  redFive: { mean: [0, 0, 0], std: [1, 1, 1] },
} as const;

function frame(capturedAtMs = 1): RecognitionFrame {
  return {
    source: {} as CanvasImageSource,
    sourceSize: { width: 1000, height: 1000 },
    regions: {
      'completed-hand': { x: 0, y: 0.7, width: 0.85, height: 0.2 },
      'dora-indicators': { x: 0, y: 0.05, width: 0.85, height: 0.2 },
      melds: { x: 0.75, y: 0, width: 0.25, height: 0.25 },
    },
    capturedAtMs,
  };
}

function fakePipelinePlatform() {
  return {
    buildComposite() {
      return {} as HTMLCanvasElement;
    },
    preprocessComposite() {
      return new Float32Array(3 * 320 * 320);
    },
    extractCrop() {
      return {
        width: 1,
        height: 1,
        channels: 4 as const,
        data: new Uint8ClampedArray([128, 128, 128, 255]),
      };
    },
  };
}

function detection(
  id: string,
  region: RegionDetection['region'],
  x: number,
  y: number,
): RegionDetection {
  return {
    id,
    detectionIndex: 0,
    classIndex: 0,
    confidence: 0.9,
    box: { x: 0, y: 0, width: 10, height: 10 },
    region,
    sourceBox: { x, y, width: 40, height: 80 },
  };
}

function tensorOutput(data: Float32Array, dims: readonly number[]): TensorOutput {
  return { data, dims, type: 'float32' };
}

function classifierOutput(data: Float32Array) {
  return { data };
}

function concatenateFloat32(...parts: readonly Float32Array[]): Float32Array {
  const output = new Float32Array(parts.reduce((sum, part) => sum + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    output.set(part, offset);
    offset += part.length;
  }
  return output;
}

function tensorDims(
  feeds: Readonly<Record<string, unknown>> | undefined,
): readonly number[] {
  const tensor = feeds?.input;
  if (
    typeof tensor !== 'object' ||
    tensor === null ||
    !('dims' in tensor) ||
    !Array.isArray(tensor.dims)
  ) {
    throw new Error('Expected fake classifier tensor dimensions');
  }
  return tensor.dims as readonly number[];
}

function firstTensorValue(feeds: Readonly<Record<string, unknown>> | undefined): number {
  const tensor = feeds?.input;
  if (
    typeof tensor !== 'object' ||
    tensor === null ||
    !('data' in tensor) ||
    !(tensor.data instanceof Float32Array)
  ) {
    throw new Error('Expected fake float32 classifier tensor input');
  }
  const first = tensor.data[0];
  if (first === undefined) {
    throw new Error('Expected non-empty classifier tensor input');
  }
  return first;
}

const baseLabels = [
  '1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m',
  '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p',
  '1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s',
  'east', 'south', 'west', 'north', 'white', 'green', 'red', 'invalid',
] as const;

function logitsFor(label: (typeof baseLabels)[number]): Float32Array {
  const logits = new Float32Array(baseLabels.length);
  logits.fill(-1);
  logits[baseLabels.indexOf(label)] = 10;
  return logits;
}

class FakeInferenceSession implements RecognitionInferenceSession {
  readonly inputNames = ['input'];
  readonly outputNames = ['output'];
  readonly runCalls: Readonly<Record<string, unknown>>[] = [];
  private outputIndex = 0;

  constructor(
    private readonly outputs: readonly (unknown | Error)[],
  ) {}

  createFloat32Tensor(data: Float32Array, dims: readonly number[]): unknown {
    return { data, dims };
  }

  async run(
    feeds: Readonly<Record<string, unknown>>,
  ): Promise<Readonly<Record<string, unknown>>> {
    this.runCalls.push(feeds);
    const output = this.outputs[Math.min(this.outputIndex, this.outputs.length - 1)];
    this.outputIndex += 1;
    if (output instanceof Error) {
      throw output;
    }
    if (output === undefined) {
      throw new Error('Fake session has no configured output');
    }
    return { output };
  }

  async dispose(): Promise<void> {}
}

function fakeModelInspection(sessions: {
  readonly detector: RecognitionInferenceSession;
  readonly base: RecognitionInferenceSession;
  readonly redFive: RecognitionInferenceSession;
}): RecognitionModelRuntimeInspection {
  return {
    getInitializedModel(role) {
      return initializedModel(role, sessionForRole(role, sessions));
    },
    getDiagnostics() {
      return [];
    },
  };
}

function sessionForRole(
  role: RecognitionModelRole,
  sessions: {
    readonly detector: RecognitionInferenceSession;
    readonly base: RecognitionInferenceSession;
    readonly redFive: RecognitionInferenceSession;
  },
): RecognitionInferenceSession {
  switch (role) {
    case 'detector':
      return sessions.detector;
    case 'tile-classifier':
      return sessions.base;
    case 'red-five-classifier':
      return sessions.redFive;
  }
}

function initializedModel(
  role: RecognitionModelRole,
  session: RecognitionInferenceSession,
): InitializedRecognitionModel {
  const runtimeSpec = {
    detector: 'nanodet-plus-m-320-v1',
    'tile-classifier': 'c8-tile-35-v1',
    'red-five-classifier': 'c8-red-five-v1',
  } as const;
  return {
    role,
    runtimeSpec: runtimeSpec[role],
    provider: 'wasm-simd',
    session,
  };
}

class ManualScheduler implements RecognitionCadenceScheduler {
  intervalMs: number | null = null;
  cancelCount = 0;
  private callback: (() => void) | null = null;
  private handle = {};

  scheduleRepeating(callback: () => void, intervalMs: number): unknown {
    this.callback = callback;
    this.intervalMs = intervalMs;
    return this.handle;
  }

  cancel(handle: unknown): void {
    expect(handle).toBe(this.handle);
    this.cancelCount += 1;
  }

  tick(): void {
    this.callback?.();
  }
}

class CountingFrameSource implements RecognitionFrameSource {
  captureCount = 0;

  captureLatest(): RecognitionFrame {
    this.captureCount += 1;
    return frame(this.captureCount);
  }
}

class RecordingListener implements RealtimeRecognitionListener {
  readonly updates: RealtimeRecognitionUpdate[] = [];
  readonly errors: RecognitionRuntimeError[] = [];

  onUpdate(update: RealtimeRecognitionUpdate): void {
    this.updates.push(update);
  }

  onError(error: RecognitionRuntimeError): void {
    this.errors.push(error);
  }
}

class FakePipeline implements RecognitionPipeline {
  evaluateCount = 0;
  disposeCount = 0;
  private outputIndex = 0;
  private readonly outputs: readonly Promise<FrameRecognitionSnapshot>[];

  constructor(...outputs: readonly Promise<FrameRecognitionSnapshot>[]) {
    this.outputs = outputs;
  }

  evaluate(): Promise<FrameRecognitionSnapshot> {
    this.evaluateCount += 1;
    const output = this.outputs[Math.min(this.outputIndex, this.outputs.length - 1)];
    this.outputIndex += 1;
    if (output === undefined) {
      return Promise.reject(new Error('Fake pipeline has no configured output'));
    }
    return output;
  }

  async dispose(): Promise<void> {
    this.disposeCount += 1;
  }
}

function eligibleSnapshot(firstKind: TileKind): FrameRecognitionSnapshot {
  return {
    observations: [],
    meldGroups: [],
    meldCommonAngleRadians: null,
    draft: {
      completedHand: [
        { kind: firstKind, red: false },
        { kind: '2m', red: false },
      ],
      doraIndicators: [{ kind: '3p', red: false }],
      meldGroups: [
        {
          kind: 'unresolved',
          tiles: [
            { kind: '1s', red: false },
            { kind: '4s', red: false },
            { kind: '7s', red: false },
          ],
        },
      ],
    },
    commitEligibility: { kind: 'eligible' },
  };
}

function ineligibleSnapshot(): FrameRecognitionSnapshot {
  return {
    observations: [],
    meldGroups: [],
    meldCommonAngleRadians: null,
    draft: {
      completedHand: [],
      doraIndicators: [],
      meldGroups: [],
    },
    commitEligibility: {
      kind: 'ineligible',
      reason: 'insufficient-visible-tiles',
    },
  };
}

function allTileIds(result: {
  readonly completedHand: readonly { readonly id: string }[];
  readonly doraIndicators: readonly { readonly id: string }[];
  readonly meldGroups: readonly {
    readonly tiles: readonly { readonly id: string }[];
  }[];
}): string[] {
  return [
    ...result.completedHand.map(({ id }) => id),
    ...result.doraIndicators.map(({ id }) => id),
    ...result.meldGroups.flatMap(({ tiles }) => tiles.map(({ id }) => id)),
  ];
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

async function flushPromises(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
}
