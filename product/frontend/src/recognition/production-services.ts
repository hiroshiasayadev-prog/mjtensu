import type {
  RecognitionFrameSource,
  RecognitionRun,
  RecognitionRuntime,
  RealtimeRecognitionListener,
  RealtimeRecognizer,
} from './contracts';
import { createBrowserRecognitionModelAssets } from './model-runtime/assets';
import { PRODUCTION_RECOGNITION_MODEL_SET } from './model-runtime/production-model-set';
import type { RecognitionModelAssets } from './model-runtime/types';
import { createProductionRecognitionRuntimeWithAssets } from './production-runtime';
import { createRealtimeRecognizer } from './realtime-recognizer';

export interface ProductionRecognitionServices {
  readonly assets: RecognitionModelAssets;
  readonly runtime: RecognitionRuntime;
  readonly recognizer: RealtimeRecognizer;
  prefetch(): Promise<void>;
  dispose(): Promise<void>;
}

export function createProductionRecognitionServices(): ProductionRecognitionServices {
  const manifest = PRODUCTION_RECOGNITION_MODEL_SET;
  const assets = createBrowserRecognitionModelAssets();
  const runtime = createProductionRecognitionRuntimeWithAssets(manifest, assets);
  const recognizer = createRuntimeBackedRealtimeRecognizer(runtime);
  let disposalInFlight: Promise<void> | null = null;

  return {
    assets,
    runtime,
    recognizer,
    prefetch() {
      return assets.prefetch(manifest);
    },
    dispose() {
      if (disposalInFlight !== null) {
        return disposalInFlight;
      }
      disposalInFlight = (async () => {
        await recognizer.dispose();
        await runtime.dispose();
      })();
      return disposalInFlight;
    },
  };
}

function createRuntimeBackedRealtimeRecognizer(
  runtime: RecognitionRuntime,
): RealtimeRecognizer {
  let delegate: RealtimeRecognizer | null = null;
  let disposed = false;

  const requireDelegate = (): RealtimeRecognizer => {
    if (disposed) {
      throw new Error('Realtime recognizer has been disposed.');
    }
    delegate ??= createRealtimeRecognizer(runtime.createPipeline());
    return delegate;
  };

  return {
    start(
      source: RecognitionFrameSource,
      listener: RealtimeRecognitionListener,
    ): RecognitionRun {
      return requireDelegate().start(source, listener);
    },
    reset() {
      if (disposed) {
        return;
      }
      delegate?.reset();
    },
    async dispose() {
      if (disposed) {
        return;
      }
      disposed = true;
      await delegate?.dispose();
    },
  };
}
