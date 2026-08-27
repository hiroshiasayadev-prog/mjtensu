import type {
  RecognitionDebugCapture,
  RecognitionEvaluationTiming,
  RecognitionPipeline,
  RecognitionRuntime,
  RecognitionRuntimeDiagnostics,
} from './contracts';
import {
  createBrowserRecognitionModelAssets,
  type RecognitionModelAssetResolver,
} from './model-runtime/assets';
import { createOnnxRecognitionSessionFactory } from './model-runtime/onnx-session-factory';
import {
  createRecognitionModelRuntime,
  type RecognitionModelRuntime,
  type RecognitionModelRuntimeInspection,
} from './model-runtime/runtime';
import type { RecognitionModelSetManifest } from './model-runtime/types';
import {
  createProductionRecognitionPipeline,
  type ProductionClassifierNormalization,
} from './production-pipeline';

export interface ProductionRecognitionRuntimeOptions {
  readonly manifest: RecognitionModelSetManifest;
}

export interface RecognitionRuntimeCompositionOptions {
  readonly modelRuntime: RecognitionModelRuntime & RecognitionModelRuntimeInspection;
  /** Test-only/internal seam. Production resolves normalization from runtimeSpec. */
  readonly classifierNormalizationOverride?: ProductionClassifierNormalization;
  readonly modelSetVersion?: string;
}

export function createProductionRecognitionRuntime(
  options: ProductionRecognitionRuntimeOptions,
): RecognitionRuntime {
  return createProductionRecognitionRuntimeWithAssets(
    options.manifest,
    createBrowserRecognitionModelAssets(),
  );
}

export function createProductionRecognitionRuntimeWithAssets(
  manifest: RecognitionModelSetManifest,
  assets: RecognitionModelAssetResolver,
): RecognitionRuntime {
  const modelRuntime = createRecognitionModelRuntime({
    manifest,
    assets,
    sessions: createOnnxRecognitionSessionFactory(),
  });
  return createRecognitionRuntimeComposition({
    modelRuntime,
    modelSetVersion: manifest.modelSetVersion,
  });
}

export function createRecognitionRuntimeComposition(
  options: RecognitionRuntimeCompositionOptions,
): RecognitionRuntime {
  const pipelines = new Set<RecognitionPipeline>();
  const recentEvaluations: RecognitionEvaluationTiming[] = [];
  let disposed = false;
  let disposalInFlight: Promise<void> | null = null;
  let pendingDebugCapture: PendingDebugCapture | null = null;

  const recordEvaluationTiming = (timing: RecognitionEvaluationTiming) => {
    recentEvaluations.push(timing);
    if (recentEvaluations.length > 120) {
      recentEvaluations.shift();
    }
  };

  const claimDebugCapture = (): boolean => {
    if (pendingDebugCapture === null || pendingDebugCapture.claimed) {
      return false;
    }
    pendingDebugCapture.claimed = true;
    return true;
  };

  const completeDebugCapture = (capture: RecognitionDebugCapture): void => {
    const pending = pendingDebugCapture;
    if (pending === null || !pending.claimed) {
      return;
    }
    pendingDebugCapture = null;
    pending.resolve(capture);
  };

  const failDebugCapture = (error: unknown): void => {
    const pending = pendingDebugCapture;
    if (pending === null || !pending.claimed) {
      return;
    }
    pendingDebugCapture = null;
    pending.reject(error);
  };

  return {
    initialize() {
      if (disposed) {
        return Promise.reject(new Error('Recognition runtime has been disposed.'));
      }
      return options.modelRuntime.initialize();
    },

    createPipeline() {
      if (disposed) {
        throw new Error('Recognition runtime has been disposed.');
      }
      const pipeline = createProductionRecognitionPipeline({
        modelRuntime: options.modelRuntime,
        classifierNormalizationOverride: options.classifierNormalizationOverride,
        onEvaluationTiming: recordEvaluationTiming,
        modelSetVersion: options.modelSetVersion,
        claimDebugCapture,
        onDebugCapture: completeDebugCapture,
        onDebugCaptureFailure: failDebugCapture,
      });
      const tracked = trackPipeline(pipeline, pipelines);
      pipelines.add(tracked);
      return tracked;
    },

    getDiagnostics(): RecognitionRuntimeDiagnostics {
      return {
        models: options.modelRuntime.getDiagnostics().map((diagnostic) => ({
          ...diagnostic,
          failedProviders: [...diagnostic.failedProviders],
        })),
        recentEvaluations: [...recentEvaluations],
      };
    },

    requestDebugCapture() {
      if (disposed) {
        return Promise.reject(new Error('Recognition runtime has been disposed.'));
      }
      if (pendingDebugCapture !== null) {
        return pendingDebugCapture.promise;
      }

      let resolve!: (capture: RecognitionDebugCapture) => void;
      let reject!: (error: unknown) => void;
      const promise = new Promise<RecognitionDebugCapture>((resolvePromise, rejectPromise) => {
        resolve = resolvePromise;
        reject = rejectPromise;
      });
      pendingDebugCapture = { promise, resolve, reject, claimed: false };
      return promise;
    },

    dispose() {
      if (disposalInFlight !== null) {
        return disposalInFlight;
      }
      disposed = true;
      if (pendingDebugCapture !== null) {
        pendingDebugCapture.reject(new Error('Recognition runtime has been disposed.'));
        pendingDebugCapture = null;
      }
      disposalInFlight = (async () => {
        await Promise.allSettled([...pipelines].map((pipeline) => pipeline.dispose()));
        pipelines.clear();
        await options.modelRuntime.dispose();
      })();
      return disposalInFlight;
    },
  };
}

interface PendingDebugCapture {
  readonly promise: Promise<RecognitionDebugCapture>;
  readonly resolve: (capture: RecognitionDebugCapture) => void;
  readonly reject: (error: unknown) => void;
  claimed: boolean;
}

function trackPipeline(
  pipeline: RecognitionPipeline,
  pipelines: Set<RecognitionPipeline>,
): RecognitionPipeline {
  let disposed = false;
  let tracked: RecognitionPipeline;
  tracked = {
    evaluate(frame) {
      if (disposed) {
        return Promise.reject(new Error('Recognition pipeline has been disposed.'));
      }
      return pipeline.evaluate(frame);
    },
    async dispose() {
      if (disposed) {
        return;
      }
      disposed = true;
      pipelines.delete(tracked);
      await pipeline.dispose();
    },
  };
  return tracked;
}
