import type { RecognizedStructure } from '@/domain';

import type { RecognitionRuntimeError } from './model-runtime/types';
import type {
  FrameRecognitionSnapshot,
  NormalizedRect,
  RecognitionRegion,
} from './semantics/types';

export interface Size {
  readonly width: number;
  readonly height: number;
}

export interface RecognitionFrame {
  readonly source: CanvasImageSource;
  readonly sourceSize: Size;
  readonly regions: Readonly<Record<RecognitionRegion, NormalizedRect>>;
  readonly capturedAtMs: number;
}

export interface RecognitionPipeline {
  evaluate(frame: RecognitionFrame): Promise<FrameRecognitionSnapshot>;
  dispose(): Promise<void>;
}

export interface RecognitionRuntime {
  initialize(): Promise<void>;
  createPipeline(): RecognitionPipeline;
  dispose(): Promise<void>;
}

export interface RecognitionFrameSource {
  captureLatest(): RecognitionFrame | null;
}

export type RealtimeRecognitionUpdate =
  | {
      readonly kind: 'scanning';
      readonly snapshot: FrameRecognitionSnapshot;
    }
  | {
      readonly kind: 'stabilizing';
      readonly snapshot: FrameRecognitionSnapshot;
    }
  | {
      readonly kind: 'confirmed';
      readonly result: RecognizedStructure;
    };

export interface RealtimeRecognitionListener {
  onUpdate(update: RealtimeRecognitionUpdate): void;
  onError(error: RecognitionRuntimeError): void;
}

export interface RecognitionRun {
  stop(): void;
}

export interface RealtimeRecognizer {
  start(
    source: RecognitionFrameSource,
    listener: RealtimeRecognitionListener,
  ): RecognitionRun;
  reset(): void;
  dispose(): Promise<void>;
}
