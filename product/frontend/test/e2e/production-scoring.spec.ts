import { expect, test, type Page } from '@playwright/test';

const HARNESS_PATH = '/test/e2e/production-scoring.html';

test('real Agari WASM scores through production Application/UI and survives a waiting update', async ({
  page,
}) => {
  test.setTimeout(120_000);

  await page.goto('/');
  await waitForActiveProductionWorker(page);

  await page.goto(HARNESS_PATH);
  await expect
    .poll(async () =>
      page.evaluate(() => navigator.serviceWorker.controller?.scriptURL ?? null),
    )
    .toMatch(/\/sw\.js$/);
  await expect
    .poll(async () =>
      page.evaluate(() => window.__MJTENSU_PRODUCTION_SCORING__),
    )
    .toMatchObject({ status: 'ready' });

  await expect(page).toHaveURL('/conditions');
  await expect(page.getByRole('heading', { name: '条件入力' })).toBeVisible();
  await expect(page.getByText('断么九 1翻')).toBeVisible();

  await page.getByRole('button', { name: '計算する' }).click();
  await expect(page).toHaveURL('/result');
  await expect(page.getByRole('heading', { name: '結果' })).toBeVisible();
  await expect(page.getByText('2,600点', { exact: true })).toBeVisible();

  await page.goBack();
  await expect(page).toHaveURL('/conditions');
  await expect(page.getByRole('heading', { name: '条件入力' })).toBeVisible();
  await page.goForward();
  await expect(page).toHaveURL('/result');
  await expect(page.getByText('2,600点', { exact: true })).toBeVisible();

  const buildIdentityBeforeUpdate = await readCachedBuildIdentity(page);
  expect(buildIdentityBeforeUpdate).toMatchObject({
    recognitionModelSetVersion: 'recognition-v1-2026-08-27',
    agariForkCommit: 'fb362b6db416e67984cdb36f704d8ebf6657662e',
    agariWasmSha256: '0e3297ed5f6807eac4d7369eb5846bc17e5ea4851470bf9d40c78ec6030e277c',
  });

  await page.evaluate(async () => {
    await navigator.serviceWorker.register('/update-probe-sw.js', {
      scope: '/',
      updateViaCache: 'none',
    });
  });
  await expect
    .poll(async () =>
      page.evaluate(async () => {
        const registration = await navigator.serviceWorker.getRegistration('/');
        return registration?.waiting?.scriptURL ?? null;
      }),
    )
    .toMatch(/\/update-probe-sw\.js$/);

  await expect(page.getByText('2,600点', { exact: true })).toBeVisible();
  await expect
    .poll(async () =>
      page.evaluate(() => navigator.serviceWorker.controller?.scriptURL ?? null),
    )
    .toMatch(/\/sw\.js$/);
  expect(await readCachedBuildIdentity(page)).toEqual(buildIdentityBeforeUpdate);
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

async function readCachedBuildIdentity(page: Page): Promise<{
  manifestUrl: string;
  recognitionModelSetVersion: string;
  agariForkCommit: string;
  agariWasmSha256: string;
}> {
  return page.evaluate(async () => {
    for (const cacheName of await caches.keys()) {
      const cache = await caches.open(cacheName);
      for (const request of await cache.keys()) {
        const url = new URL(request.url);
        if (!/^\/production-assets-[a-f0-9]+\.json$/.test(url.pathname)) {
          continue;
        }
        const response = await cache.match(request);
        if (response === undefined) {
          continue;
        }
        const manifest = await response.json() as {
          recognitionModelSet: { modelSetVersion: string };
          agariWasm: {
            provenance: { forkCommit: string; wasmSha256: string };
          };
        };
        return {
          manifestUrl: url.pathname,
          recognitionModelSetVersion: manifest.recognitionModelSet.modelSetVersion,
          agariForkCommit: manifest.agariWasm.provenance.forkCommit,
          agariWasmSha256: manifest.agariWasm.provenance.wasmSha256,
        };
      }
    }
    throw new Error('Build-pinned production asset manifest was not found in browser caches.');
  });
}
