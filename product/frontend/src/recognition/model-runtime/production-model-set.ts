import productionModelSetSource from './production-model-set.json';
import { validateRecognitionModelSetManifest } from './manifest';

function publicModelUrl(url: string): string {
  return `${import.meta.env.BASE_URL}${url}`;
}

function withPublicUrl<T extends { readonly url: string }>(model: T) {
  return {
    ...model,
    url: publicModelUrl(model.url),
  };
}

export const PRODUCTION_RECOGNITION_MODEL_SET = validateRecognitionModelSetManifest({
  schemaVersion: productionModelSetSource.schemaVersion,
  modelSetVersion: productionModelSetSource.modelSetVersion,
  models: {
    detector: withPublicUrl(productionModelSetSource.models.detector),
    'tile-classifier': withPublicUrl(
      productionModelSetSource.models['tile-classifier'],
    ),
    'red-five-classifier': withPublicUrl(
      productionModelSetSource.models['red-five-classifier'],
    ),
  },
});
