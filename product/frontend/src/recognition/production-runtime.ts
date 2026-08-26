import type { RecognitionPipeline, RecognitionRuntime } from './contracts';
import { createBrowserRecognitionModelAssets } from './model-runtime/assets';
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
}

export function createProductionRecognitionRuntime(
  options: ProductionRecognitionRuntimeOptions,
): RecognitionRuntime {
  const modelRuntime = createRecognitionModelRuntime({
    manifest: options.manifest,
    assets: createBrowserRecognitionModelAssets(),
    sessions: createOnnxRecognitionSessionFactory(),
  });
  return createRecognitionRuntimeComposition({ modelRuntime });
}

export function createRecognitionRuntimeComposition(
  options: RecognitionRuntimeCompositionOptions,
): RecognitionRuntime {
  const pipelines = new Set<RecognitionPipeline>();
  let disposed = false;
  let disposalInFlight: Promise<void> | null = null;

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
      });
      const tracked = trackPipeline(pipeline, pipelines);
      pipelines.add(tracked);
      return tracked;
    },

    dispose() {
      if (disposalInFlight !== null) {
        return disposalInFlight;
      }
      disposed = true;
      disposalInFlight = (async () => {
        await Promise.allSettled([...pipelines].map((pipeline) => pipeline.dispose()));
        pipelines.clear();
        await options.modelRuntime.dispose();
      })();
      return disposalInFlight;
    },
  };
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
