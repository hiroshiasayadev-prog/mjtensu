import type {
  RecognizedMeldGroup,
  RecognizedStructure,
  TileIdentity,
  TileInstance,
  TileInstanceId,
} from '@/domain';

import type {
  RecognitionFrameSource,
  RecognitionPipeline,
  RecognitionRun,
  RecognitionFrame,
  RealtimeRecognitionListener,
  RealtimeRecognizer,
} from './contracts';
import { isRecognitionRuntimeError } from './model-runtime/types';
import type { RecognitionRuntimeError } from './model-runtime/types';
import { RecognitionSemanticStabilizer } from './semantics/stabilizer';
import type {
  FrameMeldInterpretation,
  FrameRecognitionDraft,
  FrameRecognitionSnapshot,
} from './semantics/types';

export const RECOGNITION_REQUEST_CADENCE_MS = 100;

export interface RecognitionCadenceScheduler {
  scheduleRepeating(callback: () => void, intervalMs: number): unknown;
  cancel(handle: unknown): void;
}

export interface RealtimeRecognizerOptions {
  readonly scheduler?: RecognitionCadenceScheduler;
}

interface ActiveRun {
  readonly source: RecognitionFrameSource;
  readonly listener: RealtimeRecognitionListener;
  schedulerHandle: unknown;
  boundary: number;
  stopped: boolean;
  confirmedDelivered: boolean;
}

export function createRealtimeRecognizer(
  pipeline: RecognitionPipeline,
  options: RealtimeRecognizerOptions = {},
): RealtimeRecognizer {
  return new RealtimeRecognizerImpl(
    pipeline,
    options.scheduler ?? browserCadenceScheduler,
  );
}

class RealtimeRecognizerImpl implements RealtimeRecognizer {
  private readonly stabilizer = new RecognitionSemanticStabilizer();
  private activeRun: ActiveRun | null = null;
  private evaluationInFlight = false;
  private nextBoundary = 1;
  private nextTileId = 1;
  private disposed = false;
  private disposalInFlight: Promise<void> | null = null;

  constructor(
    private readonly pipeline: RecognitionPipeline,
    private readonly scheduler: RecognitionCadenceScheduler,
  ) {}

  start(
    source: RecognitionFrameSource,
    listener: RealtimeRecognitionListener,
  ): RecognitionRun {
    if (this.disposed) {
      throw new Error('Realtime recognizer has been disposed.');
    }

    this.stopActiveRun();
    this.stabilizer.reset();
    const run: ActiveRun = {
      source,
      listener,
      schedulerHandle: null,
      boundary: this.nextBoundary,
      stopped: false,
      confirmedDelivered: false,
    };
    this.nextBoundary += 1;
    run.schedulerHandle = this.scheduler.scheduleRepeating(
      () => {
        void this.requestEvaluation(run);
      },
      RECOGNITION_REQUEST_CADENCE_MS,
    );
    this.activeRun = run;

    // Recognition begins immediately; the 100 ms cadence governs subsequent
    // requests rather than adding a fixed startup delay.
    void this.requestEvaluation(run);

    return {
      stop: () => {
        this.stopRun(run);
      },
    };
  }

  reset(): void {
    if (this.disposed) {
      return;
    }

    this.stabilizer.reset();
    const run = this.activeRun;
    if (run !== null && !run.stopped) {
      run.boundary = this.nextBoundary;
      this.nextBoundary += 1;
      run.confirmedDelivered = false;
    }
  }

  dispose(): Promise<void> {
    if (this.disposalInFlight !== null) {
      return this.disposalInFlight;
    }
    this.disposed = true;
    this.stopActiveRun();
    this.disposalInFlight = this.pipeline.dispose();
    return this.disposalInFlight;
  }

  private async requestEvaluation(run: ActiveRun): Promise<void> {
    if (
      this.disposed ||
      run.stopped ||
      this.activeRun !== run ||
      this.evaluationInFlight ||
      run.confirmedDelivered
    ) {
      return;
    }

    let frame: RecognitionFrame | null;
    try {
      frame = run.source.captureLatest();
    } catch (error) {
      this.failRun(run, normalizeUnexpectedInferenceFailure(error));
      return;
    }
    if (frame === null) {
      return;
    }

    this.evaluationInFlight = true;
    const boundary = run.boundary;
    let snapshot: FrameRecognitionSnapshot;
    try {
      snapshot = await this.pipeline.evaluate(frame);
    } catch (error) {
      this.evaluationInFlight = false;
      if (!this.canAccept(run, boundary)) {
        return;
      }
      this.failRun(
        run,
        isRecognitionRuntimeError(error)
          ? error
          : normalizeUnexpectedInferenceFailure(error),
      );
      return;
    }
    this.evaluationInFlight = false;

    if (!this.canAccept(run, boundary)) {
      return;
    }

    const stabilization = this.stabilizer.accept(snapshot);
    switch (stabilization.kind) {
      case 'scanning':
        run.listener.onUpdate({ kind: 'scanning', snapshot });
        return;
      case 'stabilizing':
        run.listener.onUpdate({ kind: 'stabilizing', snapshot });
        return;
      case 'confirmed':
        if (run.confirmedDelivered) {
          return;
        }
        run.confirmedDelivered = true;
        run.listener.onUpdate({
          kind: 'confirmed',
          result: this.materializeStructure(stabilization.draft),
        });
        return;
    }
  }

  private canAccept(run: ActiveRun, boundary: number): boolean {
    return (
      !this.disposed &&
      !run.stopped &&
      this.activeRun === run &&
      run.boundary === boundary
    );
  }

  private failRun(run: ActiveRun, error: RecognitionRuntimeError): void {
    if (run.stopped || this.activeRun !== run) {
      return;
    }
    this.stopRun(run);
    run.listener.onError(error);
  }

  private stopActiveRun(): void {
    if (this.activeRun !== null) {
      this.stopRun(this.activeRun);
    }
  }

  private stopRun(run: ActiveRun): void {
    if (run.stopped) {
      return;
    }
    run.stopped = true;
    run.boundary = this.nextBoundary;
    this.nextBoundary += 1;
    this.scheduler.cancel(run.schedulerHandle);
    if (this.activeRun === run) {
      this.activeRun = null;
    }
  }

  private materializeStructure(
    draft: FrameRecognitionDraft,
  ): RecognizedStructure {
    return {
      completedHand: draft.completedHand.map((tile) => this.materializeTile(tile)),
      doraIndicators: draft.doraIndicators.map((tile) => this.materializeTile(tile)),
      meldGroups: draft.meldGroups.map((meld) => this.materializeMeld(meld)),
    };
  }

  private materializeMeld(meld: FrameMeldInterpretation): RecognizedMeldGroup {
    switch (meld.kind) {
      case 'chi':
        return { kind: 'chi', tiles: this.materializeThree(meld.tiles) };
      case 'pon':
        return { kind: 'pon', tiles: this.materializeThree(meld.tiles) };
      case 'open-kan':
        return { kind: 'open-kan', tiles: this.materializeFour(meld.tiles) };
      case 'concealed-kan':
        return {
          kind: 'concealed-kan',
          tiles: this.materializeFour(meld.tiles),
        };
      case 'unresolved':
        return {
          kind: 'unresolved',
          tiles: meld.tiles.map((tile) => this.materializeTile(tile)),
        };
    }
  }

  private materializeThree(
    tiles: readonly [TileIdentity, TileIdentity, TileIdentity],
  ): readonly [TileInstance, TileInstance, TileInstance] {
    return [
      this.materializeTile(tiles[0]),
      this.materializeTile(tiles[1]),
      this.materializeTile(tiles[2]),
    ];
  }

  private materializeFour(
    tiles: readonly [TileIdentity, TileIdentity, TileIdentity, TileIdentity],
  ): readonly [TileInstance, TileInstance, TileInstance, TileInstance] {
    return [
      this.materializeTile(tiles[0]),
      this.materializeTile(tiles[1]),
      this.materializeTile(tiles[2]),
      this.materializeTile(tiles[3]),
    ];
  }

  private materializeTile(tile: TileIdentity): TileInstance {
    const id = `recognition:${this.nextTileId}` as TileInstanceId;
    this.nextTileId += 1;
    return {
      id,
      tile: { kind: tile.kind, red: tile.red },
    };
  }
}

const browserCadenceScheduler: RecognitionCadenceScheduler = {
  scheduleRepeating(callback, intervalMs) {
    return globalThis.setInterval(callback, intervalMs);
  },
  cancel(handle) {
    globalThis.clearInterval(handle as ReturnType<typeof setInterval>);
  },
};

function normalizeUnexpectedInferenceFailure(
  cause: unknown,
): RecognitionRuntimeError {
  return {
    kind: 'inference-failure',
    model: 'detector',
    cause,
  };
}
