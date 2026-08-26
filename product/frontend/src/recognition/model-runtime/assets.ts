import { validateRecognitionModelSetManifest } from './manifest';
import {
  RECOGNITION_MODEL_ROLES,
  type RecognitionModelArtifactManifest,
  type RecognitionModelAssets,
  type RecognitionModelRole,
  type RecognitionModelSetManifest,
  type RecognitionRuntimeError,
} from './types';

export interface ResolvedRecognitionModelArtifact {
  readonly role: RecognitionModelRole;
  readonly url: string;
  readonly sha256: string;
  readonly runtimeSpec: RecognitionModelArtifactManifest['runtimeSpec'];
  readonly bytes: Uint8Array;
}

export interface RecognitionModelAssetResolver extends RecognitionModelAssets {
  resolve(
    manifest: RecognitionModelSetManifest,
    role: RecognitionModelRole,
  ): Promise<ResolvedRecognitionModelArtifact>;
}

export interface RecognitionModelArtifactStore {
  read(identity: string): Promise<Uint8Array | null>;
  write(identity: string, bytes: Uint8Array): Promise<void>;
  remove(identity: string): Promise<void>;
}

export interface RecognitionModelArtifactFetcher {
  fetch(url: string): Promise<Uint8Array>;
}

export interface RecognitionModelArtifactHasher {
  sha256(bytes: Uint8Array): Promise<string>;
}

export interface RecognitionModelAssetsOptions {
  readonly store: RecognitionModelArtifactStore;
  readonly fetcher: RecognitionModelArtifactFetcher;
  readonly hasher: RecognitionModelArtifactHasher;
}

export function createRecognitionModelAssets(
  options: RecognitionModelAssetsOptions,
): RecognitionModelAssetResolver {
  return new RecognitionModelAssetsImpl(options);
}

export interface BrowserRecognitionModelAssetsOptions {
  readonly cacheName?: string;
  readonly cacheStorage?: CacheStorage;
  readonly fetchImplementation?: typeof fetch;
  readonly cryptoImplementation?: Crypto;
}

export function createBrowserRecognitionModelAssets(
  options: BrowserRecognitionModelAssetsOptions = {},
): RecognitionModelAssetResolver {
  const cacheStorage = options.cacheStorage ?? globalThis.caches;
  const fetchImplementation =
    options.fetchImplementation ?? globalThis.fetch.bind(globalThis);
  const cryptoImplementation = options.cryptoImplementation ?? globalThis.crypto;

  return createRecognitionModelAssets({
    store: new BrowserCacheArtifactStore(
      cacheStorage,
      options.cacheName ?? 'mjtensu-recognition-model-artifacts-v1',
    ),
    fetcher: new BrowserArtifactFetcher(fetchImplementation),
    hasher: new WebCryptoArtifactHasher(cryptoImplementation),
  });
}

class RecognitionModelAssetsImpl implements RecognitionModelAssetResolver {
  private readonly inFlight = new Map<
    string,
    Promise<ResolvedRecognitionModelArtifact>
  >();

  constructor(private readonly options: RecognitionModelAssetsOptions) {}

  async prefetch(manifest: RecognitionModelSetManifest): Promise<void> {
    const validated = validateRecognitionModelSetManifest(manifest);
    await Promise.all(
      RECOGNITION_MODEL_ROLES.map((role) =>
        this.resolveValidated(validated.models[role]),
      ),
    );
  }

  async resolve(
    manifest: RecognitionModelSetManifest,
    role: RecognitionModelRole,
  ): Promise<ResolvedRecognitionModelArtifact> {
    const validated = validateRecognitionModelSetManifest(manifest);
    return this.resolveValidated(validated.models[role]);
  }

  private resolveValidated(
    artifact: RecognitionModelArtifactManifest,
  ): Promise<ResolvedRecognitionModelArtifact> {
    const identity = artifactIdentity(artifact);
    const existing = this.inFlight.get(identity);
    if (existing !== undefined) {
      return existing;
    }

    const load = this.loadArtifact(artifact, identity);
    let pending: Promise<ResolvedRecognitionModelArtifact>;
    const clearInFlight = () => {
      if (this.inFlight.get(identity) === pending) {
        this.inFlight.delete(identity);
      }
    };
    pending = load.then(
      (resolved) => {
        clearInFlight();
        return resolved;
      },
      (error: unknown) => {
        clearInFlight();
        throw error;
      },
    );
    this.inFlight.set(identity, pending);
    return pending;
  }

  private async loadArtifact(
    artifact: RecognitionModelArtifactManifest,
    identity: string,
  ): Promise<ResolvedRecognitionModelArtifact> {
    const cached = await this.tryReadCached(identity);
    if (cached !== null) {
      if (await this.matchesIntegrity(cached, artifact.sha256)) {
        return resolvedArtifact(artifact, cached);
      }
      await this.tryRemoveCached(identity);
    }

    let fetched: Uint8Array;
    try {
      fetched = await this.options.fetcher.fetch(artifact.url);
    } catch {
      throw runtimeError('model-asset-unavailable', artifact.role);
    }

    if (!(await this.matchesIntegrity(fetched, artifact.sha256))) {
      throw runtimeError('model-integrity-failure', artifact.role);
    }

    try {
      await this.options.store.write(identity, fetched);
    } catch {
      // A valid fetched artifact is still usable for the current application
      // lifecycle even if persistent browser caching is unavailable.
    }
    return resolvedArtifact(artifact, fetched);
  }

  private async tryReadCached(identity: string): Promise<Uint8Array | null> {
    try {
      return await this.options.store.read(identity);
    } catch {
      return null;
    }
  }

  private async tryRemoveCached(identity: string): Promise<void> {
    try {
      await this.options.store.remove(identity);
    } catch {
      // A stale cache entry must not prevent a network resolution attempt.
    }
  }

  private async matchesIntegrity(
    bytes: Uint8Array,
    expectedSha256: string,
  ): Promise<boolean> {
    try {
      return (await this.options.hasher.sha256(bytes)).toLowerCase() === expectedSha256;
    } catch {
      return false;
    }
  }
}

class BrowserArtifactFetcher implements RecognitionModelArtifactFetcher {
  constructor(private readonly fetchImplementation: typeof fetch) {}

  async fetch(url: string): Promise<Uint8Array> {
    const response = await this.fetchImplementation(url, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`Model request failed with HTTP ${response.status}.`);
    }
    return new Uint8Array(await response.arrayBuffer());
  }
}

class BrowserCacheArtifactStore implements RecognitionModelArtifactStore {
  constructor(
    private readonly cacheStorage: CacheStorage,
    private readonly cacheName: string,
  ) {}

  async read(identity: string): Promise<Uint8Array | null> {
    const cache = await this.cacheStorage.open(this.cacheName);
    const response = await cache.match(cacheRequest(identity));
    if (response === undefined) {
      return null;
    }
    return new Uint8Array(await response.arrayBuffer());
  }

  async write(identity: string, bytes: Uint8Array): Promise<void> {
    const cache = await this.cacheStorage.open(this.cacheName);
    const body = new Uint8Array(bytes).buffer;
    await cache.put(cacheRequest(identity), new Response(body));
  }

  async remove(identity: string): Promise<void> {
    const cache = await this.cacheStorage.open(this.cacheName);
    await cache.delete(cacheRequest(identity));
  }
}

class WebCryptoArtifactHasher implements RecognitionModelArtifactHasher {
  constructor(private readonly cryptoImplementation: Crypto) {}

  async sha256(bytes: Uint8Array): Promise<string> {
    const digest = await this.cryptoImplementation.subtle.digest(
      'SHA-256',
      new Uint8Array(bytes),
    );
    return [...new Uint8Array(digest)]
      .map((value) => value.toString(16).padStart(2, '0'))
      .join('');
  }
}

function artifactIdentity(artifact: RecognitionModelArtifactManifest): string {
  return `${artifact.role}:${artifact.sha256}`;
}

function cacheRequest(identity: string): Request {
  return new Request(
    `https://mjtensu.invalid/__recognition_model_cache__/${encodeURIComponent(identity)}`,
  );
}

function resolvedArtifact(
  artifact: RecognitionModelArtifactManifest,
  bytes: Uint8Array,
): ResolvedRecognitionModelArtifact {
  return {
    role: artifact.role,
    url: artifact.url,
    sha256: artifact.sha256,
    runtimeSpec: artifact.runtimeSpec,
    bytes,
  };
}

function runtimeError(
  kind: 'model-asset-unavailable' | 'model-integrity-failure',
  model: RecognitionModelRole,
): RecognitionRuntimeError {
  return { kind, model };
}
