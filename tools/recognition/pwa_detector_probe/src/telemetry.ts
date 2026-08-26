export interface TimingSample {
  preprocessMs: number;
  inferenceMs: number;
  decodeMs: number;
  endToEndMs: number;
  completedAt: number;
  detectionCount: number;
}

export interface TelemetrySnapshot {
  latest: TimingSample | null;
  rollingMedianMs: number;
  rollingP95Ms: number;
  effectiveHz: number;
  sampleCount: number;
}

export interface ProviderRunSummary {
  provider: string;
  completedAtIso: string;
  durationSeconds: number;
  sampleCount: number;
  expectedDetectorRequests: number;
  medianPreprocessMs: number;
  p95PreprocessMs: number;
  medianInferenceMs: number;
  p95InferenceMs: number;
  medianDecodeMs: number;
  p95DecodeMs: number;
  medianEndToEndMs: number;
  p95EndToEndMs: number;
  effectiveHz: number;
  firstTenSecondsMedianMs: number;
  lastTenSecondsMedianMs: number;
  slowdownPercent: number;
  firstTenSecondsInferenceMedianMs: number;
  lastTenSecondsInferenceMedianMs: number;
  inferenceSlowdownPercent: number;
  droppedFrames: number;
}

const MAX_SAMPLES = 1200;
const ROLLING_SAMPLE_COUNT = 120;
const HZ_WINDOW_MS = 5000;

export class TelemetrySeries {
  private readonly samples: TimingSample[] = [];

  add(sample: TimingSample): void {
    this.samples.push(sample);
    if (this.samples.length > MAX_SAMPLES) {
      this.samples.splice(0, this.samples.length - MAX_SAMPLES);
    }
  }

  clear(): void {
    this.samples.length = 0;
  }

  snapshot(now = performance.now()): TelemetrySnapshot {
    const rolling = this.samples.slice(-ROLLING_SAMPLE_COUNT);
    const latest = this.samples.at(-1) ?? null;
    return {
      latest,
      rollingMedianMs: percentile(
        rolling.map((sample) => sample.endToEndMs),
        0.5,
      ),
      rollingP95Ms: percentile(
        rolling.map((sample) => sample.endToEndMs),
        0.95,
      ),
      effectiveHz: calculateEffectiveHz(this.samples, now),
      sampleCount: this.samples.length,
    };
  }

  summarizeRun(
    provider: string,
    runStartedAt: number,
    runEndedAt: number,
    droppedFrames: number,
    detectorIntervalMs: number,
  ): ProviderRunSummary {
    const runSamples = this.samples.filter(
      (sample) => sample.completedAt >= runStartedAt && sample.completedAt <= runEndedAt,
    );
    const durationMs = Math.max(0, runEndedAt - runStartedAt);
    const firstWindowEnd = runStartedAt + Math.min(10_000, durationMs);
    const lastWindowStart = runEndedAt - Math.min(10_000, durationMs);
    const firstMedian = percentile(
      runSamples
        .filter((sample) => sample.completedAt <= firstWindowEnd)
        .map((sample) => sample.endToEndMs),
      0.5,
    );
    const lastMedian = percentile(
      runSamples
        .filter((sample) => sample.completedAt >= lastWindowStart)
        .map((sample) => sample.endToEndMs),
      0.5,
    );
    const slowdownPercent =
      firstMedian > 0 ? ((lastMedian - firstMedian) / firstMedian) * 100 : 0;
    const firstInferenceMedian = percentile(
      runSamples
        .filter((sample) => sample.completedAt <= firstWindowEnd)
        .map((sample) => sample.inferenceMs),
      0.5,
    );
    const lastInferenceMedian = percentile(
      runSamples
        .filter((sample) => sample.completedAt >= lastWindowStart)
        .map((sample) => sample.inferenceMs),
      0.5,
    );
    const inferenceSlowdownPercent =
      firstInferenceMedian > 0
        ? ((lastInferenceMedian - firstInferenceMedian) / firstInferenceMedian) * 100
        : 0;
    const expectedDetectorRequests =
      detectorIntervalMs > 0 ? Math.floor(durationMs / detectorIntervalMs) : 0;
    const inferredDroppedFrames = Math.max(
      0,
      expectedDetectorRequests - runSamples.length,
    );

    return {
      provider,
      completedAtIso: new Date().toISOString(),
      durationSeconds: durationMs / 1000,
      sampleCount: runSamples.length,
      expectedDetectorRequests,
      medianPreprocessMs: percentile(
        runSamples.map((sample) => sample.preprocessMs),
        0.5,
      ),
      p95PreprocessMs: percentile(
        runSamples.map((sample) => sample.preprocessMs),
        0.95,
      ),
      medianInferenceMs: percentile(
        runSamples.map((sample) => sample.inferenceMs),
        0.5,
      ),
      p95InferenceMs: percentile(
        runSamples.map((sample) => sample.inferenceMs),
        0.95,
      ),
      medianDecodeMs: percentile(
        runSamples.map((sample) => sample.decodeMs),
        0.5,
      ),
      p95DecodeMs: percentile(
        runSamples.map((sample) => sample.decodeMs),
        0.95,
      ),
      medianEndToEndMs: percentile(
        runSamples.map((sample) => sample.endToEndMs),
        0.5,
      ),
      p95EndToEndMs: percentile(
        runSamples.map((sample) => sample.endToEndMs),
        0.95,
      ),
      effectiveHz: durationMs > 0 ? (runSamples.length * 1000) / durationMs : 0,
      firstTenSecondsMedianMs: firstMedian,
      lastTenSecondsMedianMs: lastMedian,
      slowdownPercent,
      firstTenSecondsInferenceMedianMs: firstInferenceMedian,
      lastTenSecondsInferenceMedianMs: lastInferenceMedian,
      inferenceSlowdownPercent,
      droppedFrames: Math.max(droppedFrames, inferredDroppedFrames),
    };
  }
}

function calculateEffectiveHz(
  samples: TimingSample[],
  now: number,
  explicitWindowMs = HZ_WINDOW_MS,
): number {
  const windowStart = now - explicitWindowMs;
  const inWindow = samples.filter((sample) => sample.completedAt >= windowStart);
  if (inWindow.length < 2) {
    return 0;
  }
  const first = inWindow[0];
  const last = inWindow.at(-1);
  if (first === undefined || last === undefined || last.completedAt <= first.completedAt) {
    return 0;
  }
  return ((inWindow.length - 1) * 1000) / (last.completedAt - first.completedAt);
}

export function percentile(values: number[], quantile: number): number {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(quantile * sorted.length) - 1));
  return sorted[index] ?? 0;
}
