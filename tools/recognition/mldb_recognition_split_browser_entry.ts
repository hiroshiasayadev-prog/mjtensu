import {
  createProductionRecognitionPipeline,
  type RecognitionEvaluationTrace,
} from '@/recognition/production-pipeline';
import { createNanoDetPostprocessor } from '@/recognition/detector/detection-postprocessor';
import { createRecognitionModelRuntime, type RecognitionInferenceSessionFactory } from '@/recognition/model-runtime/runtime';
import type {
  RecognitionModelAssetResolver,
  ResolvedRecognitionModelArtifact,
} from '@/recognition/model-runtime/assets';
import type {
  RecognitionModelRole,
  RecognitionModelRuntimeSpec,
  RecognitionModelSetManifest,
} from '@/recognition/model-runtime/types';
import { RecognitionSemanticStabilizer, areFrameRecognitionDraftsEqual } from '@/recognition/semantics/stabilizer';
import type {
  FrameMeldInterpretation,
  FrameRecognitionDraft,
  FrameRecognitionSnapshot,
  NormalizedRect,
  RecognitionRegion,
} from '@/recognition/semantics/types';
import type { RecognitionFrame, RecognitionEvaluationTiming } from '@/recognition/contracts';

const ORT_WEB_DIST_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';
const CADENCE_MS = 100;

function createEvaluationSessionFactory(): RecognitionInferenceSessionFactory {
  return {
    async create({ artifact, provider }) {
      if (provider !== 'wasm-simd') {
        throw new Error(`unsupported evaluation provider: ${provider}`);
      }
      const ort = await import(/* @vite-ignore */ `${ORT_WEB_DIST_URL}ort.wasm.min.mjs`);
      ort.env.logLevel = 'warning';
      ort.env.wasm.proxy = false;
      ort.env.wasm.simd = true;
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.wasmPaths = ORT_WEB_DIST_URL;
      const session = await ort.InferenceSession.create(artifact, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
        executionMode: 'sequential',
      });
      return {
        get inputNames() { return session.inputNames; },
        get outputNames() { return session.outputNames; },
        createFloat32Tensor(data, dims) {
          return new ort.Tensor('float32', data, [...dims]);
        },
        run(feeds) {
          return session.run(feeds);
        },
        async dispose() {
          await session.release();
        },
      };
    },
  };
}

interface TileIdentityConfig {
  readonly kind: string;
  readonly red: boolean;
}

interface MeldConfig {
  readonly kind: string;
  readonly tiles: readonly TileIdentityConfig[];
}

interface GroundTruthConfig {
  readonly completed_hand: readonly TileIdentityConfig[];
  readonly dora_indicators: readonly TileIdentityConfig[];
  readonly melds: readonly MeldConfig[];
}

interface CaptureConfig {
  readonly logicalCapture: {
    readonly aspectRatio: '16:9' | '9:16';
    readonly rotation: 0 | 90 | -90;
  };
  readonly recognitionRegions: Readonly<Record<RecognitionRegion, NormalizedRect>>;
}

interface RuntimeModelConfig {
  readonly url: string;
  readonly sha256: string;
  readonly runtimeSpec: RecognitionModelRuntimeSpec;
  readonly inlineBase64?: string;
}

interface E2ETakeConfig {
  readonly id: string;
  readonly videoUrl: string;
  readonly capture: CaptureConfig;
  readonly sourceFps: number;
  readonly groundTruth?: GroundTruthConfig;
}

interface BrowserConfig {
  readonly mode: 'iphone-latency' | 'functional';
  readonly resultUrl?: string;
  readonly sampleIntervalMs?: number;
  readonly detectorScoreThreshold?: number;
  readonly baseClassifierBase64: string;
  readonly baseClassifierSha256: string;
  readonly baseClassifierRuntimeSpec: RecognitionModelRuntimeSpec;
  readonly baseNormalization: { readonly mean: readonly number[]; readonly std: readonly number[] };
  readonly detector: RuntimeModelConfig;
  readonly redFive: RuntimeModelConfig & {
    readonly normalization: { readonly mean: readonly number[]; readonly std: readonly number[] };
  };
  readonly takes: readonly E2ETakeConfig[];
}

declare global {
  var __MLDB_RECOGNITION_E2E_CONFIG__: BrowserConfig | undefined;
}

function bytesFromBase64(value: string): Uint8Array {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

class EvaluationAssetResolver implements RecognitionModelAssetResolver {
  private readonly cache = new Map<RecognitionModelRole, Uint8Array>();

  constructor(
    private readonly config: BrowserConfig,
    private readonly manifest: RecognitionModelSetManifest,
  ) {}

  async prefetch(_manifest: RecognitionModelSetManifest): Promise<void> {
    await Promise.all((['detector', 'tile-classifier', 'red-five-classifier'] as const).map((role) => this.resolve(this.manifest, role)));
  }

  async resolve(
    _manifest: RecognitionModelSetManifest,
    role: RecognitionModelRole,
  ): Promise<ResolvedRecognitionModelArtifact> {
    let bytes = this.cache.get(role);
    if (bytes === undefined) {
      if (role === 'tile-classifier') {
        bytes = bytesFromBase64(this.config.baseClassifierBase64);
      } else if (role === 'detector' && this.config.detector.inlineBase64 !== undefined) {
        bytes = bytesFromBase64(this.config.detector.inlineBase64);
      } else {
        const url = role === 'detector' ? this.config.detector.url : this.config.redFive.url;
        const response = await fetch(url, { cache: 'no-store' });
        if (!response.ok) {
          throw new Error(`${role} fetch failed: ${response.status}`);
        }
        bytes = new Uint8Array(await response.arrayBuffer());
      }
      this.cache.set(role, bytes);
    }
    const artifact = this.manifest.models[role];
    return {
      role,
      url: artifact.url,
      sha256: artifact.sha256,
      runtimeSpec: artifact.runtimeSpec,
      bytes,
    };
  }
}

function manifestFromConfig(config: BrowserConfig): RecognitionModelSetManifest {
  const providerPreference = ['wasm-simd'] as const;
  return {
    schemaVersion: 1,
    modelSetVersion: 'mldb-recognition-e2e-v1',
    models: {
      detector: {
        role: 'detector',
        url: config.detector.url,
        sha256: config.detector.sha256,
        runtimeSpec: config.detector.runtimeSpec,
        providerPreference,
      },
      'tile-classifier': {
        role: 'tile-classifier',
        url: 'inline://mldb-primary-model',
        sha256: config.baseClassifierSha256,
        runtimeSpec: config.baseClassifierRuntimeSpec,
        providerPreference,
      },
      'red-five-classifier': {
        role: 'red-five-classifier',
        url: config.redFive.url,
        sha256: config.redFive.sha256,
        runtimeSpec: config.redFive.runtimeSpec,
        providerPreference,
      },
    },
  };
}

function expectedDraft(gt: GroundTruthConfig): FrameRecognitionDraft {
  return {
    completedHand: gt.completed_hand.map((tile) => ({ kind: tile.kind, red: tile.red })) as FrameRecognitionDraft['completedHand'],
    doraIndicators: gt.dora_indicators.map((tile) => ({ kind: tile.kind, red: tile.red })) as FrameRecognitionDraft['doraIndicators'],
    meldGroups: gt.melds.map((meld) => ({
      kind: meld.kind,
      tiles: meld.tiles.map((tile) => ({ kind: tile.kind, red: tile.red })),
    })) as readonly FrameMeldInterpretation[],
  };
}

function tileArraysEqual(left: readonly { kind: string; red: boolean }[], right: readonly { kind: string; red: boolean }[]): boolean {
  return left.length === right.length && left.every((tile, index) => {
    const other = right[index];
    return other !== undefined && tile.kind === other.kind && tile.red === other.red;
  });
}

function meldArraysEqual(left: readonly FrameMeldInterpretation[], right: readonly FrameMeldInterpretation[]): boolean {
  return left.length === right.length && left.every((meld, index) => {
    const other = right[index];
    return other !== undefined && meld.kind === other.kind && tileArraysEqual(meld.tiles, other.tiles);
  });
}

function centerCrop(width: number, height: number, targetAspect: number) {
  const sourceAspect = width / height;
  if (sourceAspect > targetAspect) {
    const cropWidth = height * targetAspect;
    return { x: (width - cropWidth) / 2, y: 0, width: cropWidth, height };
  }
  const cropHeight = width / targetAspect;
  return { x: 0, y: (height - cropHeight) / 2, width, height: cropHeight };
}

function canonicalCaptureSize(
  videoWidth: number,
  videoHeight: number,
  aspectRatio: '16:9' | '9:16',
  rotation: 0 | 90 | -90,
): { width: number; height: number } {
  const sourceAspect = aspectRatio === '16:9' ? 16 / 9 : 9 / 16;
  const crop = centerCrop(videoWidth, videoHeight, sourceAspect);
  const sourceUnitWidth = aspectRatio === '16:9' ? 16 : 9;
  const sourceUnitHeight = aspectRatio === '16:9' ? 9 : 16;
  const outputUnitWidth = rotation === 0 ? sourceUnitWidth : sourceUnitHeight;
  const outputUnitHeight = rotation === 0 ? sourceUnitHeight : sourceUnitWidth;
  const units = Math.max(1, Math.floor(Math.min(
    crop.width / sourceUnitWidth,
    crop.height / sourceUnitHeight,
    1280 / Math.max(outputUnitWidth, outputUnitHeight),
  )));
  return { width: units * outputUnitWidth, height: units * outputUnitHeight };
}

function drawCanonicalFrame(
  context: CanvasRenderingContext2D,
  video: HTMLVideoElement,
  targetWidth: number,
  targetHeight: number,
  aspectRatio: '16:9' | '9:16',
  rotation: 0 | 90 | -90,
): void {
  const sourceAspect = aspectRatio === '16:9' ? 16 / 9 : 9 / 16;
  const crop = centerCrop(video.videoWidth, video.videoHeight, sourceAspect);
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.clearRect(0, 0, targetWidth, targetHeight);
  if (rotation === 0) {
    context.drawImage(video, crop.x, crop.y, crop.width, crop.height, 0, 0, targetWidth, targetHeight);
    return;
  }
  context.save();
  if (rotation === 90) {
    context.translate(targetWidth, 0);
    context.rotate(Math.PI / 2);
  } else {
    context.translate(0, targetHeight);
    context.rotate(-Math.PI / 2);
  }
  context.drawImage(video, crop.x, crop.y, crop.width, crop.height, 0, 0, targetHeight, targetWidth);
  context.restore();
}

function percentile(values: readonly number[], q: number): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const rank = Math.max(1, Math.ceil(q * sorted.length));
  return sorted[rank - 1] ?? null;
}

function mean(values: readonly number[]): number | null {
  return values.length === 0 ? null : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function waitForVideoMetadata(video: HTMLVideoElement): Promise<void> {
  if (video.readyState >= 1 && video.videoWidth > 0 && video.videoHeight > 0) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const ready = () => { cleanup(); resolve(); };
    const fail = () => { cleanup(); reject(new Error('video metadata load failed')); };
    const cleanup = () => {
      video.removeEventListener('loadedmetadata', ready);
      video.removeEventListener('error', fail);
    };
    video.addEventListener('loadedmetadata', ready, { once: true });
    video.addEventListener('error', fail, { once: true });
  });
}

async function runLatencyTake(
  take: E2ETakeConfig,
  pipeline: ReturnType<typeof createProductionRecognitionPipeline>,
  timingSink: RecognitionEvaluationTiming[],
) {
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';
  video.crossOrigin = 'anonymous';
  video.src = take.videoUrl;
  document.body.appendChild(video);
  await waitForVideoMetadata(video);

  const layout = take.capture.logicalCapture;
  const size = canonicalCaptureSize(video.videoWidth, video.videoHeight, layout.aspectRatio, layout.rotation);
  const canvas = document.createElement('canvas');
  canvas.width = size.width;
  canvas.height = size.height;
  const context = canvas.getContext('2d', { alpha: false, willReadFrequently: true });
  if (context === null) throw new Error('2D canvas unavailable');

  let timer: ReturnType<typeof setInterval> | null = null;
  let inFlight: Promise<void> | null = null;
  let ticks = 0;
  let skipped = 0;
  let evaluations = 0;
  const videoTimes: number[] = [];
  const timingStart = timingSink.length;

  const capture = (): RecognitionFrame | null => {
    if (video.readyState < 2 || video.videoWidth <= 0 || video.videoHeight <= 0) return null;
    drawCanonicalFrame(context, video, canvas.width, canvas.height, layout.aspectRatio, layout.rotation);
    return {
      source: canvas,
      sourceSize: { width: canvas.width, height: canvas.height },
      regions: take.capture.recognitionRegions,
      capturedAtMs: performance.now(),
    };
  };

  const evaluate = () => {
    ticks += 1;
    if (inFlight !== null) {
      skipped += 1;
      return;
    }
    const frame = capture();
    if (frame === null) return;
    const videoTime = video.currentTime;
    inFlight = pipeline.evaluate(frame).then(() => {
      evaluations += 1;
      videoTimes.push(videoTime);
    }).finally(() => {
      inFlight = null;
    });
  };

  try {
    const ended = new Promise<void>((resolve, reject) => {
      video.addEventListener('ended', () => resolve(), { once: true });
      video.addEventListener('error', () => reject(new Error('video playback failed')), { once: true });
    });
    timer = setInterval(evaluate, CADENCE_MS);
    await video.play();
    evaluate();
    await ended;
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
    if (inFlight !== null) await inFlight;
  } finally {
    if (timer !== null) clearInterval(timer);
    video.pause();
    video.removeAttribute('src');
    video.load();
    video.remove();
  }

  const timings = timingSink.slice(timingStart);
  const totals = timings.map((value) => value.totalMs);
  const durationSec = Number.isFinite(video.duration) ? video.duration : 30;
  const targetEvaluations = Math.max(1, Math.floor(durationSec * 1000 / CADENCE_MS));
  const gaps = videoTimes.slice(1).map((value, index) =>
    Math.max(0, (value - (videoTimes[index] ?? value)) * 1000),
  );
  return {
    id: take.id,
    source_video_duration_sec: durationSec,
    ticks,
    skipped_in_flight: skipped,
    evaluations,
    target_evaluations: targetEvaluations,
    cadence_fulfillment_rate: evaluations / targetEvaluations,
    effective_eval_hz: durationSec > 0 ? evaluations / durationSec : 0,
    eval_gap_ms_p50: percentile(gaps, 0.5),
    eval_gap_ms_p95: percentile(gaps, 0.95),
    eval_gap_ms_max: gaps.length === 0 ? null : Math.max(...gaps),
    timing: {
      count: timings.length,
      total_ms_mean: mean(totals),
      total_ms_p50: percentile(totals, 0.5),
      total_ms_p95: percentile(totals, 0.95),
      samples: timings,
    },
  };
}

async function primeVideoForSeeking(video: HTMLVideoElement): Promise<void> {
  if (video.readyState >= 2) return;
  const loaded = new Promise<void>((resolve, reject) => {
    const ready = () => { cleanup(); resolve(); };
    const fail = () => { cleanup(); reject(new Error('video decode prime failed')); };
    const cleanup = () => {
      video.removeEventListener('loadeddata', ready);
      video.removeEventListener('error', fail);
    };
    video.addEventListener('loadeddata', ready, { once: true });
    video.addEventListener('error', fail, { once: true });
  });
  await video.play();
  await loaded;
  video.pause();
}

function seekVideo(video: HTMLVideoElement, timeSec: number): Promise<void> {
  const target = Math.max(0, Math.min(timeSec, Math.max(0, video.duration - 0.001)));
  if (Math.abs(video.currentTime - target) < 0.0005 && video.readyState >= 2) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const done = () => { cleanup(); resolve(); };
    const fail = () => { cleanup(); reject(new Error('video seek failed')); };
    const cleanup = () => {
      video.removeEventListener('seeked', done);
      video.removeEventListener('error', fail);
    };
    video.addEventListener('seeked', done, { once: true });
    video.addEventListener('error', fail, { once: true });
    video.currentTime = target;
  });
}

function serializeStabilization(value: ReturnType<RecognitionSemanticStabilizer['getState']>) {
  if (value.kind === 'scanning') return { kind: 'scanning', consecutive: 0 };
  if (value.kind === 'stabilizing') {
    return { kind: 'stabilizing', consecutive: value.consecutive, candidate: value.candidate };
  }
  return { kind: 'confirmed', consecutive: 3, draft: value.draft };
}

async function runFunctionalTake(
  take: E2ETakeConfig,
  pipeline: ReturnType<typeof createProductionRecognitionPipeline>,
  getTrace: () => RecognitionEvaluationTrace | null,
  sampleIntervalMs: number,
) {
  if (take.groundTruth === undefined) throw new Error('functional take missing ground truth: ' + take.id);

  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';
  video.crossOrigin = 'anonymous';
  video.src = take.videoUrl;
  document.body.appendChild(video);
  await waitForVideoMetadata(video);
  await primeVideoForSeeking(video);

  const layout = take.capture.logicalCapture;
  const size = canonicalCaptureSize(video.videoWidth, video.videoHeight, layout.aspectRatio, layout.rotation);
  const canvas = document.createElement('canvas');
  canvas.width = size.width;
  canvas.height = size.height;
  const context = canvas.getContext('2d', { alpha: false, willReadFrequently: true });
  if (context === null) throw new Error('2D canvas unavailable');

  const expected = expectedDraft(take.groundTruth);
  const stabilizer = new RecognitionSemanticStabilizer();
  const rows: object[] = [];
  let gtStreak = 0;
  let exactFrames = 0;
  let handExactFrames = 0;
  let doraExactFrames = 0;
  let meldExactFrames = 0;
  let eligibleFrames = 0;
  let firstGtStreak3: object | null = null;
  let firstProductConfirm: object | null = null;

  const durationSec = Number.isFinite(video.duration) ? video.duration : 30;
  const stepSec = sampleIntervalMs / 1000;
  const count = Math.max(1, Math.floor(durationSec / stepSec));

  try {
    for (let evalIndex = 0; evalIndex < count; evalIndex += 1) {
      const videoTimeSec = Math.min(evalIndex * stepSec, Math.max(0, durationSec - 0.001));
      await seekVideo(video, videoTimeSec);
      drawCanonicalFrame(context, video, canvas.width, canvas.height, layout.aspectRatio, layout.rotation);
      const snapshot = await pipeline.evaluate({
        source: canvas,
        sourceSize: { width: canvas.width, height: canvas.height },
        regions: take.capture.recognitionRegions,
        capturedAtMs: videoTimeSec * 1000,
      });
      const trace = getTrace();
      if (trace === null) throw new Error('functional trace callback did not run');

      const handExact = tileArraysEqual(snapshot.draft.completedHand, expected.completedHand);
      const doraExact = tileArraysEqual(snapshot.draft.doraIndicators, expected.doraIndicators);
      const meldExact = meldArraysEqual(snapshot.draft.meldGroups, expected.meldGroups);
      const gtExact = areFrameRecognitionDraftsEqual(snapshot.draft, expected);
      const eligible = snapshot.commitEligibility.kind === 'eligible';
      if (handExact) handExactFrames += 1;
      if (doraExact) doraExactFrames += 1;
      if (meldExact) meldExactFrames += 1;
      if (gtExact) exactFrames += 1;
      if (eligible) eligibleFrames += 1;
      gtStreak = gtExact && eligible ? gtStreak + 1 : 0;

      const sourceFrame = Math.round(videoTimeSec * take.sourceFps);
      if (gtStreak >= 3 && firstGtStreak3 === null) {
        firstGtStreak3 = { eval_index: evalIndex, video_time_sec: videoTimeSec, source_frame: sourceFrame };
      }

      const stabilization = stabilizer.accept(snapshot);
      if (stabilization.kind === 'confirmed' && firstProductConfirm === null) {
        firstProductConfirm = {
          eval_index: evalIndex,
          video_time_sec: videoTimeSec,
          source_frame: sourceFrame,
          exact: areFrameRecognitionDraftsEqual(stabilization.draft, expected),
          draft: stabilization.draft,
        };
      }

      rows.push({
        take_id: take.id,
        eval_index: evalIndex,
        video_time_sec: videoTimeSec,
        source_frame: sourceFrame,
        source_fps: take.sourceFps,
        detections: trace.detections,
        snapshot,
        hand_exact: handExact,
        dora_exact: doraExact,
        meld_exact: meldExact,
        gt_exact: gtExact,
        gt_exact_consecutive: gtStreak,
        stabilization: serializeStabilization(stabilizer.getState()),
      });
    }
  } finally {
    video.pause();
    video.removeAttribute('src');
    video.load();
    video.remove();
  }

  return {
    id: take.id,
    source_video_duration_sec: durationSec,
    source_fps: take.sourceFps,
    evaluations: count,
    eligible_frames: eligibleFrames,
    exact_frames: exactFrames,
    completed_hand_exact_frames: handExactFrames,
    dora_exact_frames: doraExactFrames,
    meld_exact_frames: meldExactFrames,
    exact_frame_rate: exactFrames / count,
    completed_hand_exact_rate: handExactFrames / count,
    dora_exact_rate: doraExactFrames / count,
    meld_exact_rate: meldExactFrames / count,
    first_gt_streak3: firstGtStreak3,
    first_product_confirm: firstProductConfirm,
    rows,
  };
}

async function submitResult(config: BrowserConfig, payload: unknown): Promise<void> {
  if (config.resultUrl !== undefined) {
    const response = await fetch(config.resultUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error('result callback failed: ' + response.status);
    return;
  }
  const query = new URLSearchParams(location.search);
  const jobId = query.get('runner_job_id');
  const token = query.get('runner_token');
  if (!jobId || !token) throw new Error('browser runner job identity missing');
  const response = await fetch(
    '/v1/jobs/' + encodeURIComponent(jobId) + '/result?token=' + encodeURIComponent(token),
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    },
  );
  if (!response.ok) throw new Error('result callback failed: ' + response.status);
}

async function main(): Promise<void> {
  const config = globalThis.__MLDB_RECOGNITION_E2E_CONFIG__;
  if (config === undefined) throw new Error('MLDB E2E config missing');
  const manifest = manifestFromConfig(config);
  const assets = new EvaluationAssetResolver(config, manifest);
  const modelRuntime = createRecognitionModelRuntime({
    manifest,
    assets,
    sessions: createEvaluationSessionFactory(),
  });
  const timings: RecognitionEvaluationTiming[] = [];
  let latestTrace: RecognitionEvaluationTrace | null = null;
  let pipeline: ReturnType<typeof createProductionRecognitionPipeline> | null = null;
  try {
    await modelRuntime.initialize();
    const detectorPostprocessor = config.detectorScoreThreshold === undefined
      ? undefined
      : (() => {
          if (config.detector.runtimeSpec !== 'nanodet-plus-m-320-v1') {
            throw new Error('detectorScoreThreshold is only valid for NanoDet runtime');
          }
          if (
            !Number.isFinite(config.detectorScoreThreshold)
            || config.detectorScoreThreshold < 0
            || config.detectorScoreThreshold > 1
          ) {
            throw new Error('detectorScoreThreshold must be within [0, 1]');
          }
          return createNanoDetPostprocessor({
            confidenceThreshold: config.detectorScoreThreshold,
            nmsIouThreshold: 0.6,
            maximumDetections: 200,
            duplicateOverlapThreshold: 0.8,
          });
        })();
    pipeline = createProductionRecognitionPipeline({
      modelRuntime,
      classifierNormalizationOverride: {
        base: config.baseNormalization,
        redFive: config.redFive.normalization,
      },
      detectorPostprocessor,
      onEvaluationTiming: config.mode === 'iphone-latency' ? (timing) => timings.push(timing) : undefined,
      onEvaluationTrace: config.mode === 'functional' ? (trace) => { latestTrace = trace; } : undefined,
      modelSetVersion: manifest.modelSetVersion,
    });

    const takes = [];
    if (config.mode === 'iphone-latency') {
      for (const take of config.takes) takes.push(await runLatencyTake(take, pipeline, timings));
    } else {
      const interval = config.sampleIntervalMs ?? CADENCE_MS;
      if (!Number.isFinite(interval) || interval <= 0) throw new Error('sampleIntervalMs must be positive');
      for (const take of config.takes) {
        takes.push(await runFunctionalTake(take, pipeline, () => latestTrace, interval));
      }
    }

    await submitResult(config, {
      ok: true,
      mode: config.mode,
      takes,
      environment: {
        user_agent: navigator.userAgent,
        hardware_concurrency: navigator.hardwareConcurrency || 1,
        cross_origin_isolated: globalThis.crossOriginIsolated === true,
        secure_context: globalThis.isSecureContext === true,
        provider: 'wasm-simd',
        num_threads: 1,
      },
      model_diagnostics: modelRuntime.getDiagnostics(),
    });
  } finally {
    if (pipeline !== null) await pipeline.dispose();
    await modelRuntime.dispose();
  }
}

void main().catch(async (error) => {
  const config = globalThis.__MLDB_RECOGNITION_E2E_CONFIG__;
  if (config === undefined) return;
  try {
    await submitResult(config, {
      ok: false,
      mode: config.mode,
      error: error instanceof Error ? error.stack ?? error.message : String(error),
      environment: { user_agent: navigator.userAgent },
    });
  } catch {
  }
});
