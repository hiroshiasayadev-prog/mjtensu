import { createHash } from 'node:crypto';

import { createBrowserRecognitionModelAssets } from '@/recognition/model-runtime/assets';
import type {
  RecognitionModelRole,
  RecognitionModelSetManifest,
} from '@/recognition/model-runtime/types';
import { describe, expect, it, vi } from 'vitest';

const bytesByRole = {
  detector: new Uint8Array([1, 2, 3]),
  'tile-classifier': new Uint8Array([4, 5, 6]),
  'red-five-classifier': new Uint8Array([7, 8, 9]),
} as const satisfies Record<RecognitionModelRole, Uint8Array>;

const manifest: RecognitionModelSetManifest = {
  schemaVersion: 1,
  modelSetVersion: 'browser-cache-fixture-v1',
  models: {
    detector: artifact('detector', 'nanodet-plus-m-320-v1'),
    'tile-classifier': artifact('tile-classifier', 'gray64-tile-35-v1'),
    'red-five-classifier': artifact('red-five-classifier', 'c8-red-five-v1'),
  },
};

describe('browser Recognition model cache', () => {
  it('reuses a complete integrity-checked model set offline across asset-resolver instances', async () => {
    const cacheStorage = new MemoryCacheStorage();
    const onlineFetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      const role = roleForUrl(url);
      return new Response(new Uint8Array(bytesByRole[role]).buffer);
    });
    const onlineAssets = createBrowserRecognitionModelAssets({
      cacheStorage: cacheStorage as unknown as CacheStorage,
      fetchImplementation: onlineFetch as typeof fetch,
      cryptoImplementation: fixtureCrypto(),
    });

    await onlineAssets.prefetch(manifest);
    expect(onlineFetch).toHaveBeenCalledTimes(3);

    const offlineFetch = vi.fn(async () => {
      throw new TypeError('offline');
    });
    const offlineAssets = createBrowserRecognitionModelAssets({
      cacheStorage: cacheStorage as unknown as CacheStorage,
      fetchImplementation: offlineFetch as unknown as typeof fetch,
      cryptoImplementation: fixtureCrypto(),
    });

    await expect(offlineAssets.prefetch(manifest)).resolves.toBeUndefined();
    expect(offlineFetch).not.toHaveBeenCalled();
  });
});

function artifact(
  role: RecognitionModelRole,
  runtimeSpec:
    | 'nanodet-plus-m-320-v1'
    | 'gray64-tile-35-v1'
    | 'c8-red-five-v1',
) {
  return {
    role,
    url: `/models/${role}.onnx?sha256=${sha256(bytesByRole[role])}`,
    sha256: sha256(bytesByRole[role]),
    runtimeSpec,
    providerPreference: ['wasm-simd', 'wasm-threaded', 'webgl'] as const,
  };
}

function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function roleForUrl(url: string): RecognitionModelRole {
  if (url.includes('red-five-classifier')) {
    return 'red-five-classifier';
  }
  if (url.includes('tile-classifier')) {
    return 'tile-classifier';
  }
  return 'detector';
}

class MemoryCacheStorage {
  private readonly caches = new Map<string, MemoryCache>();

  async open(name: string): Promise<Cache> {
    let cache = this.caches.get(name);
    if (cache === undefined) {
      cache = new MemoryCache();
      this.caches.set(name, cache);
    }
    return cache as unknown as Cache;
  }
}

class MemoryCache {
  private readonly entries = new Map<string, Response>();

  async match(request: RequestInfo | URL): Promise<Response | undefined> {
    return this.entries.get(requestKey(request))?.clone();
  }

  async put(request: RequestInfo | URL, response: Response): Promise<void> {
    this.entries.set(requestKey(request), response.clone());
  }

  async delete(request: RequestInfo | URL): Promise<boolean> {
    return this.entries.delete(requestKey(request));
  }
}

function requestKey(request: RequestInfo | URL): string {
  return request instanceof Request ? request.url : String(request);
}

function fixtureCrypto(): Crypto {
  return {
    subtle: {
      async digest(_algorithm: AlgorithmIdentifier, data: BufferSource) {
        const input = ArrayBuffer.isView(data)
          ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
          : new Uint8Array(data);
        const digest = createHash('sha256').update(input).digest();
        return digest.buffer.slice(
          digest.byteOffset,
          digest.byteOffset + digest.byteLength,
        ) as ArrayBuffer;
      },
    },
  } as unknown as Crypto;
}
