export type ProductionPwaUpdateListener = (
  registration: ServiceWorkerRegistration,
) => void;

export interface ProductionPwaLifecycleOptions {
  readonly serviceWorker?: ServiceWorkerContainer;
  readonly baseUrl?: string;
  readonly onOfflineReady?: (registration: ServiceWorkerRegistration) => void;
  readonly onUpdateAvailable?: ProductionPwaUpdateListener;
}

const updateListeners = new Set<ProductionPwaUpdateListener>();

export function subscribeProductionPwaUpdates(
  listener: ProductionPwaUpdateListener,
): () => void {
  updateListeners.add(listener);
  return () => {
    updateListeners.delete(listener);
  };
}

export async function registerProductionPwaLifecycle(
  options: ProductionPwaLifecycleOptions = {},
): Promise<ServiceWorkerRegistration | null> {
  const serviceWorker = resolveServiceWorkerContainer(options.serviceWorker);
  if (serviceWorker === null) {
    return null;
  }

  const baseUrl = normalizeBaseUrl(options.baseUrl ?? import.meta.env.BASE_URL);
  const registration = await serviceWorker.register(`${baseUrl}sw.js`, {
    scope: baseUrl,
    updateViaCache: 'none',
  });

  observeRegistration(serviceWorker, registration, options);
  return registration;
}

function observeRegistration(
  serviceWorker: ServiceWorkerContainer,
  registration: ServiceWorkerRegistration,
  options: ProductionPwaLifecycleOptions,
): void {
  const notifiedWorkers = new WeakSet<ServiceWorker>();

  const notifyUpdateAvailable = (worker: ServiceWorker): void => {
    if (notifiedWorkers.has(worker)) {
      return;
    }
    notifiedWorkers.add(worker);
    options.onUpdateAvailable?.(registration);
    for (const listener of updateListeners) {
      listener(registration);
    }
  };

  if (registration.waiting !== null && serviceWorker.controller !== null) {
    notifyUpdateAvailable(registration.waiting);
  }

  registration.addEventListener('updatefound', () => {
    const installing = registration.installing;
    if (installing === null) {
      return;
    }

    installing.addEventListener('statechange', () => {
      if (installing.state !== 'installed') {
        return;
      }

      if (serviceWorker.controller !== null) {
        notifyUpdateAvailable(installing);
        return;
      }

      options.onOfflineReady?.(registration);
    });
  });
}

function resolveServiceWorkerContainer(
  explicit: ServiceWorkerContainer | undefined,
): ServiceWorkerContainer | null {
  if (explicit !== undefined) {
    return explicit;
  }
  if (
    typeof navigator === 'undefined' ||
    !('serviceWorker' in navigator)
  ) {
    return null;
  }
  return navigator.serviceWorker;
}

function normalizeBaseUrl(value: string): string {
  const withLeadingSlash = value.startsWith('/') ? value : `/${value}`;
  return withLeadingSlash.endsWith('/')
    ? withLeadingSlash
    : `${withLeadingSlash}/`;
}
