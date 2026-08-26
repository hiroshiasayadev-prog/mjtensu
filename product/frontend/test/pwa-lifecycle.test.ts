import {
  registerProductionPwaLifecycle,
  subscribeProductionPwaUpdates,
} from '@/pwa';
import { describe, expect, it, vi } from 'vitest';

class FakeServiceWorker extends EventTarget {
  state: ServiceWorkerState = 'installing';
  readonly postMessage = vi.fn();

  install(): void {
    this.state = 'installed';
    this.dispatchEvent(new Event('statechange'));
  }
}

class FakeServiceWorkerRegistration extends EventTarget {
  installing: ServiceWorker | null = null;
  waiting: ServiceWorker | null = null;
}

function makeContainer(
  registration: FakeServiceWorkerRegistration,
  controller: ServiceWorker | null,
): ServiceWorkerContainer & { register: ReturnType<typeof vi.fn> } {
  const register = vi.fn(async () => registration as unknown as ServiceWorkerRegistration);
  return {
    controller,
    register,
  } as unknown as ServiceWorkerContainer & { register: ReturnType<typeof vi.fn> };
}

describe('production PWA lifecycle', () => {
  it('reports an already waiting update without activating it or reloading the app', async () => {
    const waiting = new FakeServiceWorker();
    waiting.state = 'installed';
    const registration = new FakeServiceWorkerRegistration();
    registration.waiting = waiting as unknown as ServiceWorker;
    const container = makeContainer(
      registration,
      {} as ServiceWorker,
    );
    const optionListener = vi.fn();
    const subscriber = vi.fn();
    const unsubscribe = subscribeProductionPwaUpdates(subscriber);

    await registerProductionPwaLifecycle({
      serviceWorker: container,
      baseUrl: '/app',
      onUpdateAvailable: optionListener,
    });
    unsubscribe();

    expect(container.register).toHaveBeenCalledWith('/app/sw.js', {
      scope: '/app/',
      updateViaCache: 'none',
    });
    expect(optionListener).toHaveBeenCalledTimes(1);
    expect(subscriber).toHaveBeenCalledTimes(1);
    expect(waiting.postMessage).not.toHaveBeenCalled();
  });

  it('reports a newly installed update once while the existing client remains controlled', async () => {
    const installing = new FakeServiceWorker();
    const registration = new FakeServiceWorkerRegistration();
    registration.installing = installing as unknown as ServiceWorker;
    const container = makeContainer(registration, {} as ServiceWorker);
    const onUpdateAvailable = vi.fn();

    await registerProductionPwaLifecycle({
      serviceWorker: container,
      onUpdateAvailable,
    });
    registration.dispatchEvent(new Event('updatefound'));
    installing.install();
    installing.dispatchEvent(new Event('statechange'));

    expect(onUpdateAvailable).toHaveBeenCalledTimes(1);
    expect(installing.postMessage).not.toHaveBeenCalled();
  });

  it('distinguishes first-install offline readiness from an application update', async () => {
    const installing = new FakeServiceWorker();
    const registration = new FakeServiceWorkerRegistration();
    registration.installing = installing as unknown as ServiceWorker;
    const container = makeContainer(registration, null);
    const onOfflineReady = vi.fn();
    const onUpdateAvailable = vi.fn();

    await registerProductionPwaLifecycle({
      serviceWorker: container,
      onOfflineReady,
      onUpdateAvailable,
    });
    registration.dispatchEvent(new Event('updatefound'));
    installing.install();

    expect(onOfflineReady).toHaveBeenCalledTimes(1);
    expect(onUpdateAvailable).not.toHaveBeenCalled();
  });
});
