import { expect, test, type BrowserContext, type Page, type Route } from '@playwright/test';

const RECOGNITION_HARNESS_PATH = '/test/e2e/recognition-production-artifacts.html';
const MODEL_CACHE_NAME = 'mjtensu-recognition-model-artifacts-v1';

test('shell/service-worker startup completes before deferred ONNX acquisition and then caches the model set', async ({
  page,
}) => {
  test.setTimeout(120_000);

  let releaseModels: (() => void) | undefined;
  const modelGate = new Promise<void>((resolve) => {
    releaseModels = resolve;
  });
  let modelRequests = 0;
  const holdModelRequest = async (route: Route) => {
    modelRequests += 1;
    await modelGate;
    await route.continue();
  };

  await page.route(
    (url) => url.pathname.endsWith('.onnx'),
    holdModelRequest,
  );
  await page.goto('/');

  await expect(page.getByRole('heading', { name: 'mjtensu' })).toBeVisible();
  await expect.poll(() => modelRequests).toBeGreaterThan(0);
  expect(await cachedModelCount(page)).toBe(0);
  await waitForActiveProductionWorker(page);

  releaseModels?.();
  await expect
    .poll(async () => cachedModelCount(page), { timeout: 90_000 })
    .toBe(3);
});

test('cached production shell and Recognition runtime remain available offline', async ({
  context,
  page,
}) => {
  test.setTimeout(180_000);

  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'mjtensu' })).toBeVisible();
  await waitForActiveProductionWorker(page);

  await page.goto(RECOGNITION_HARNESS_PATH);
  await expect
    .poll(async () =>
      page.evaluate(() => navigator.serviceWorker.controller?.scriptURL ?? null),
    )
    .toMatch(/\/sw\.js$/);
  await waitForRecognitionReady(page);
  await expect
    .poll(async () => cachedModelCount(page), { timeout: 90_000 })
    .toBe(3);

  await setOffline(context, true);
  await page.reload({ waitUntil: 'domcontentloaded' });
  await waitForRecognitionReady(page);
  await expect
    .poll(async () => cachedModelCount(page))
    .toBe(3);

  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'mjtensu' })).toBeVisible();
});

async function waitForActiveProductionWorker(page: Page): Promise<void> {
  await expect
    .poll(async () =>
      page.evaluate(async () => {
        const registration = await navigator.serviceWorker.getRegistration('/');
        return registration?.active?.state ?? null;
      }),
      { timeout: 60_000 },
    )
    .toBe('activated');
}

async function waitForRecognitionReady(page: Page): Promise<void> {
  await page.waitForFunction(
    () => window.__MJTENSU_RECOGNITION_ARTIFACTS__?.status !== 'running',
    undefined,
    { timeout: 110_000 },
  );
  const diagnostics = await page.evaluate(
    () => window.__MJTENSU_RECOGNITION_ARTIFACTS__,
  );
  expect(diagnostics.status, diagnostics.error).toBe('ready');
  expect(diagnostics.modelSetVersion).toBe('recognition-v1-2026-08-27');
}

async function cachedModelCount(page: Page): Promise<number> {
  return page.evaluate(async (cacheName) => {
    if (!(await caches.has(cacheName))) {
      return 0;
    }
    return (await (await caches.open(cacheName)).keys()).length;
  }, MODEL_CACHE_NAME);
}

async function setOffline(context: BrowserContext, offline: boolean): Promise<void> {
  await context.setOffline(offline);
}
