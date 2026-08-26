import type { RecognitionModelAssetResolver, ResolvedRecognitionModelArtifact } from './assets';
import { validateRecognitionModelSetManifest } from './manifest';
import { getRecognitionRuntimeSpecDescriptor } from './runtime-specs';
import {
  RECOGNITION_MODEL_ROLES,
  isRecognitionRuntimeError,
  type ExecutionProvider,
  type RecognitionModelRole,
  type RecognitionModelRuntimeSpec,
  type RecognitionModelSetManifest,
  type RecognitionRuntimeError,
} from './types';

export interface RecognitionInferenceSession {
  readonly inputNames: readonly string[];
  readonly outputNames: readonly string[];
  createFloat32Tensor(data: Float32Array, dims: readonly number[]): unknown;
  run(
    feeds: Readonly<Record<string, unknown>>,
  ): Promise<Readonly<Record<string, unknown>>>;
  dispose(): Promise<void>;
}

export interface RecognitionInferenceSessionFactory {
  create(options: {
    readonly role: RecognitionModelRole;
    readonly runtimeSpec: RecognitionModelRuntimeSpec;
    readonly provider: ExecutionProvider;
    readonly artifact: Uint8Array;
  }): Promise<RecognitionInferenceSession>;
}

export interface RecognitionModelRuntimeDiagnostics {
  readonly role: RecognitionModelRole;
  readonly runtimeSpec: RecognitionModelRuntimeSpec;
  readonly selectedProvider?: ExecutionProvider;
  readonly failedProviders: readonly ExecutionProvider[];
}

export interface RecognitionModelRuntimeOptions {
  readonly manifest: RecognitionModelSetManifest;
  readonly assets: RecognitionModelAssetResolver;
  readonly sessions: RecognitionInferenceSessionFactory;
}

export interface RecognitionModelRuntime {
  initialize(): Promise<void>;
  dispose(): Promise<void>;
}

export interface InitializedRecognitionModel {
  readonly role: RecognitionModelRole;
  readonly runtimeSpec: RecognitionModelRuntimeSpec;
  readonly provider: ExecutionProvider;
  readonly session: RecognitionInferenceSession;
}

export function createRecognitionModelRuntime(
  options: RecognitionModelRuntimeOptions,
): RecognitionModelRuntime & RecognitionModelRuntimeInspection {
  return new RecognitionModelRuntimeImpl(options);
}

export interface RecognitionModelRuntimeInspection {
  getInitializedModel(role: RecognitionModelRole): InitializedRecognitionModel;
  getDiagnostics(): readonly RecognitionModelRuntimeDiagnostics[];
}

class RecognitionModelRuntimeImpl
  implements RecognitionModelRuntime, RecognitionModelRuntimeInspection
{
  private initializedModels: Map<RecognitionModelRole, InitializedRecognitionModel> | null = null;
  private initializationInFlight: Promise<void> | null = null;
  private disposalInFlight: Promise<void> | null = null;
  private disposalRequested = false;
  private disposed = false;
  private diagnostics = new Map<RecognitionModelRole, RecognitionModelRuntimeDiagnostics>();

  constructor(private readonly options: RecognitionModelRuntimeOptions) {}

  initialize(): Promise<void> {
    if (this.disposed || this.disposalRequested) {
      return Promise.reject(new Error('Recognition model runtime has been disposed.'));
    }
    if (this.initializedModels !== null) {
      return Promise.resolve();
    }
    if (this.initializationInFlight !== null) {
      return this.initializationInFlight;
    }

    const attempt = this.initializeAttempt();
    this.initializationInFlight = attempt;
    void attempt.then(
      () => {
        if (this.initializationInFlight === attempt) {
          this.initializationInFlight = null;
        }
      },
      () => {
        if (this.initializationInFlight === attempt) {
          this.initializationInFlight = null;
        }
      },
    );
    return attempt;
  }

  dispose(): Promise<void> {
    if (this.disposalInFlight !== null) {
      return this.disposalInFlight;
    }
    if (this.disposed) {
      return Promise.resolve();
    }

    this.disposalRequested = true;
    const disposal = this.disposeAttempt();
    this.disposalInFlight = disposal;
    return disposal;
  }

  getInitializedModel(role: RecognitionModelRole): InitializedRecognitionModel {
    const model = this.initializedModels?.get(role);
    if (model === undefined) {
      throw new Error('Recognition model runtime is not initialized.');
    }
    return model;
  }

  getDiagnostics(): readonly RecognitionModelRuntimeDiagnostics[] {
    return RECOGNITION_MODEL_ROLES.flatMap((role) => {
      const diagnostic = this.diagnostics.get(role);
      return diagnostic === undefined ? [] : [diagnostic];
    });
  }

  private async initializeAttempt(): Promise<void> {
    let validated: RecognitionModelSetManifest;
    try {
      validated = validateRecognitionModelSetManifest(this.options.manifest);
    } catch (error) {
      if (isRecognitionRuntimeError(error)) {
        throw error;
      }
      throw initializationFailure('detector', error);
    }

    const created = new Map<RecognitionModelRole, InitializedRecognitionModel>();
    this.diagnostics = new Map();
    let activeRole: RecognitionModelRole = 'detector';

    try {
      for (const role of RECOGNITION_MODEL_ROLES) {
        activeRole = role;
        const artifactManifest = validated.models[role];
        const descriptor = getRecognitionRuntimeSpecDescriptor(
          artifactManifest.runtimeSpec,
        );
        if (descriptor.role !== role) {
          throw incompatible(role);
        }

        const artifact = await this.options.assets.resolve(validated, role);
        const initialized = await this.createWithProviderFallback(
          artifact,
          artifactManifest.providerPreference,
        );
        created.set(role, initialized);
      }

      if (this.disposalRequested) {
        throw new Error('Recognition model runtime was disposed during initialization.');
      }

      this.initializedModels = created;
    } catch (error) {
      await disposeModels(created);
      this.initializedModels = null;
      if (isRecognitionRuntimeError(error)) {
        throw error;
      }
      throw initializationFailure(activeRole, error);
    }
  }

  private async createWithProviderFallback(
    artifact: ResolvedRecognitionModelArtifact,
    preference: readonly ExecutionProvider[],
  ): Promise<InitializedRecognitionModel> {
    const failedProviders: ExecutionProvider[] = [];

    for (const provider of preference) {
      try {
        const session = await this.options.sessions.create({
          role: artifact.role,
          runtimeSpec: artifact.runtimeSpec,
          provider,
          artifact: artifact.bytes,
        });
        this.diagnostics.set(artifact.role, {
          role: artifact.role,
          runtimeSpec: artifact.runtimeSpec,
          selectedProvider: provider,
          failedProviders: [...failedProviders],
        });
        return {
          role: artifact.role,
          runtimeSpec: artifact.runtimeSpec,
          provider,
          session,
        };
      } catch {
        failedProviders.push(provider);
      }
    }

    this.diagnostics.set(artifact.role, {
      role: artifact.role,
      runtimeSpec: artifact.runtimeSpec,
      failedProviders,
    });
    throw providerUnavailable(artifact.role);
  }

  private async disposeAttempt(): Promise<void> {
    const initialization = this.initializationInFlight;
    if (initialization !== null) {
      try {
        await initialization;
      } catch {
        // Failed initialization already disposes every session it created.
      }
    }

    const initialized = this.initializedModels;
    this.initializedModels = null;
    if (initialized !== null) {
      await disposeModels(initialized);
    }
    this.disposed = true;
  }
}

async function disposeModels(
  models: ReadonlyMap<RecognitionModelRole, InitializedRecognitionModel>,
): Promise<void> {
  await Promise.allSettled(
    [...models.values()].map((model) => model.session.dispose()),
  );
}

function providerUnavailable(
  model: RecognitionModelRole,
): RecognitionRuntimeError {
  return { kind: 'execution-provider-unavailable', model };
}

function incompatible(model: RecognitionModelRole): RecognitionRuntimeError {
  return { kind: 'model-incompatible', model };
}

function initializationFailure(
  model: RecognitionModelRole,
  cause: unknown,
): RecognitionRuntimeError {
  return { kind: 'model-initialization-failure', model, cause };
}
