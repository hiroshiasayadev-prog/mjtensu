import { describe, expect, it } from 'vitest';

import { createDeferred, createFakeService } from './support';

interface ExampleService {
  readonly getStatus: () => Promise<'ready' | 'failed'>;
}

describe('shared fake-service support', () => {
  it('creates a typed fake service without concrete implementation imports', async () => {
    const service = createFakeService<ExampleService>({
      getStatus: async () => 'ready',
    });

    await expect(service.getStatus()).resolves.toBe('ready');
  });

  it('provides deterministic control of asynchronous service work', async () => {
    const deferred = createDeferred<number>();

    deferred.resolve(42);

    await expect(deferred.promise).resolves.toBe(42);
  });
});
