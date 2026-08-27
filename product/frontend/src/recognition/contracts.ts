import type { RecognizedStructure } from '@/domain';

import type {
  ExecutionProvider,
  RecognitionModelRole,
  RecognitionModelRuntimeSpec,
  RecognitionRuntimeError,
} from './model-runtime/types';
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

export interface RecognitionEvaluationTiming {
  readonly totalMs: number;
  readonly candidateCount: number;
  readonly redFiveCandidateCount: number;
  readonly detectorPreprocessingMs: number;
  readonly detectorInferenceMs: number;
  readonly detectorPostprocessingMs: number;
  readonly cropExtractionMs: number;
  readonly baseClassifierPreprocessingMs: number;
  readonly baseClassifierInferenceMs: number;
  readonly redFiveClassifierPreprocessingMs: number;
  readonly redFiveClassifierInferenceMs: number;
}

export interface RecognitionRuntimeModelDiagnostic {
  readonly role: RecognitionModelRole;
  readonly runtimeSpec: RecognitionModelRuntimeSpec;
  readonly selectedProvider?: ExecutionProvider;
  readonly failedProviders: readonly ExecutionProvider[];
}

export interface RecognitionRuntimeDiagnostics {
  readonly models: readonly RecognitionRuntimeModelDiagnostic[];
  readonly recentEvaluations: readonly RecognitionEvaluationTiming[];
}

export interface RecognitionRuntime {
  initialize(): Promise<void>;
  createPipeline(): RecognitionPipeline;
  getDiagnostics?(): RecognitionRuntimeDiagnostics;
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
