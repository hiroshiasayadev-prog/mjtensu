export interface TimingSample {
  preprocessMs: number;
  inferenceMs: number;
  decodeMs: number;
  endToEndMs: number;
  completedAt: number;
  detectionCount: number;
}

export class TelemetrySeries {
  private readonly samples: TimingSample[] = [];
  private dropped = 0;

  add(sample: TimingSample): void {
    this.samples.push(sample);
    if (this.samples.length > 600) this.samples.splice(0, this.samples.length - 600);
  }

  drop(): void { this.dropped += 1; }

  clear(): void {
    this.samples.length = 0;
    this.dropped = 0;
  }

  snapshot(now = performance.now()): Record<string, number> {
    const rolling = this.samples.slice(-120);
    const latest = rolling.at(-1);
    const recent = rolling.filter((sample) => sample.completedAt >= now - 5000);
    const hz = recent.length < 2
      ? 0
      : ((recent.length - 1) * 1000) / ((recent.at(-1)?.completedAt ?? now) - (recent[0]?.completedAt ?? now));
    return {
      preprocessMs: latest?.preprocessMs ?? 0,
      inferenceMs: latest?.inferenceMs ?? 0,
      decodeMs: latest?.decodeMs ?? 0,
      endToEndMs: latest?.endToEndMs ?? 0,
      medianEndToEndMs: percentile(rolling.map((sample) => sample.endToEndMs), 0.5),
      p95EndToEndMs: percentile(rolling.map((sample) => sample.endToEndMs), 0.95),
      effectiveHz: Number.isFinite(hz) ? hz : 0,
      detectionCount: latest?.detectionCount ?? 0,
      sampleCount: this.samples.length,
      droppedFrames: this.dropped,
    };
  }
}

function percentile(values: number[], quantile: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * quantile) - 1));
  return sorted[index] ?? 0;
}
