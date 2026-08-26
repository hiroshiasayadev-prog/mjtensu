export const RECOGNITION_MODEL_ROLES = [
  'detector',
  'tile-classifier',
  'red-five-classifier',
] as const;

export type RecognitionModelRole = (typeof RECOGNITION_MODEL_ROLES)[number];

export const EXECUTION_PROVIDERS = [
  'wasm-simd',
  'wasm-threaded',
  'webgl',
] as const;

export type ExecutionProvider = (typeof EXECUTION_PROVIDERS)[number];

export const DEFAULT_EXECUTION_PROVIDER_PREFERENCE = [
  'wasm-simd',
  'wasm-threaded',
  'webgl',
] as const satisfies readonly ExecutionProvider[];

export const RECOGNITION_MODEL_RUNTIME_SPECS = [
  'nanodet-plus-m-320-v1',
  'c8-tile-35-v1',
  'c8-red-five-v1',
] as const;

export type RecognitionModelRuntimeSpec =
  (typeof RECOGNITION_MODEL_RUNTIME_SPECS)[number];

export interface RecognitionModelArtifactManifest {
  readonly role: RecognitionModelRole;
  readonly url: string;
  readonly sha256: string;
  readonly runtimeSpec: RecognitionModelRuntimeSpec;
  readonly providerPreference: readonly ExecutionProvider[];
}

export interface RecognitionModelSetManifest {
  readonly schemaVersion: 1;
  readonly modelSetVersion: string;
  readonly models: Readonly<
    Record<RecognitionModelRole, RecognitionModelArtifactManifest>
  >;
}

export type RecognitionRuntimeError =
  | {
      readonly kind: 'model-asset-unavailable';
      readonly model: RecognitionModelRole;
    }
  | {
      readonly kind: 'model-integrity-failure';
      readonly model: RecognitionModelRole;
    }
  | {
      readonly kind: 'model-incompatible';
      readonly model: RecognitionModelRole;
    }
  | {
      readonly kind: 'execution-provider-unavailable';
      readonly model: RecognitionModelRole;
    }
  | {
      readonly kind: 'model-initialization-failure';
      readonly model: RecognitionModelRole;
      readonly cause: unknown;
    }
  | {
      readonly kind: 'inference-failure';
      readonly model: RecognitionModelRole;
      readonly cause: unknown;
    };

export interface RecognitionModelAssets {
  prefetch(manifest: RecognitionModelSetManifest): Promise<void>;
}

export function isRecognitionRuntimeError(
  value: unknown,
): value is RecognitionRuntimeError {
  if (typeof value !== 'object' || value === null || !('kind' in value)) {
    return false;
  }

  switch ((value as { readonly kind?: unknown }).kind) {
    case 'model-asset-unavailable':
    case 'model-integrity-failure':
    case 'model-incompatible':
    case 'execution-provider-unavailable':
      return hasRecognitionModelRole(value);
    case 'model-initialization-failure':
    case 'inference-failure':
      return hasRecognitionModelRole(value) && 'cause' in value;
    default:
      return false;
  }
}

function hasRecognitionModelRole(
  value: object,
): value is { readonly model: RecognitionModelRole } {
  if (!('model' in value)) {
    return false;
  }
  return RECOGNITION_MODEL_ROLES.includes(
    value.model as RecognitionModelRole,
  );
}
