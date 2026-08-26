import { getRecognitionRuntimeSpecDescriptor, isRecognitionModelRuntimeSpec } from './runtime-specs';
import {
  EXECUTION_PROVIDERS,
  RECOGNITION_MODEL_ROLES,
  type ExecutionProvider,
  type RecognitionModelArtifactManifest,
  type RecognitionModelRole,
  type RecognitionModelSetManifest,
  type RecognitionRuntimeError,
} from './types';

const SHA256_PATTERN = /^[a-f0-9]{64}$/i;

export function validateRecognitionModelSetManifest(
  value: unknown,
): RecognitionModelSetManifest {
  if (!isRecord(value) || value.schemaVersion !== 1) {
    throw incompatible('detector');
  }
  if (
    typeof value.modelSetVersion !== 'string' ||
    value.modelSetVersion.trim().length === 0
  ) {
    throw incompatible('detector');
  }

  const modelManifestsValue = value.models;

  if (!isRecord(modelManifestsValue)) {
    throw incompatible('detector');
  }

  const modelManifests = modelManifestsValue;
  const modelKeys = Object.keys(modelManifests);
  if (
    modelKeys.length !== RECOGNITION_MODEL_ROLES.length ||
    modelKeys.some(
      (key) => !RECOGNITION_MODEL_ROLES.includes(key as RecognitionModelRole),
    )
  ) {
    throw incompatible('detector');
  }

  const models = Object.fromEntries(
    RECOGNITION_MODEL_ROLES.map((role) => [
      role,
      validateArtifactManifest(modelManifests[role], role),
    ]),
  ) as unknown as Record<RecognitionModelRole, RecognitionModelArtifactManifest>;

  return {
    schemaVersion: 1,
    modelSetVersion: value.modelSetVersion,
    models,
  };
}

function validateArtifactManifest(
  value: unknown,
  expectedRole: RecognitionModelRole,
): RecognitionModelArtifactManifest {
  if (!isRecord(value) || value.role !== expectedRole) {
    throw incompatible(expectedRole);
  }
  if (typeof value.url !== 'string' || value.url.trim().length === 0) {
    throw incompatible(expectedRole);
  }
  if (typeof value.sha256 !== 'string' || !SHA256_PATTERN.test(value.sha256)) {
    throw incompatible(expectedRole);
  }
  if (!isRecognitionModelRuntimeSpec(value.runtimeSpec)) {
    throw incompatible(expectedRole);
  }
  if (getRecognitionRuntimeSpecDescriptor(value.runtimeSpec).role !== expectedRole) {
    throw incompatible(expectedRole);
  }
  if (!isProviderPreference(value.providerPreference)) {
    throw incompatible(expectedRole);
  }

  return {
    role: expectedRole,
    url: value.url,
    sha256: value.sha256.toLowerCase(),
    runtimeSpec: value.runtimeSpec,
    providerPreference: [...value.providerPreference],
  };
}

function isProviderPreference(value: unknown): value is readonly ExecutionProvider[] {
  if (!Array.isArray(value) || value.length === 0) {
    return false;
  }

  const seen = new Set<ExecutionProvider>();
  for (const provider of value) {
    if (
      typeof provider !== 'string' ||
      !EXECUTION_PROVIDERS.includes(provider as ExecutionProvider) ||
      seen.has(provider as ExecutionProvider)
    ) {
      return false;
    }
    seen.add(provider as ExecutionProvider);
  }
  return true;
}

function incompatible(model: RecognitionModelRole): RecognitionRuntimeError {
  return { kind: 'model-incompatible', model };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
