import { expect, test } from '@playwright/test';

const HARNESS_PATH = '/test/e2e/recognition-production-artifacts.html';

test('production Recognition model set loads and executes bounded real-artifact fixtures', async ({
  page,
}) => {
  test.setTimeout(120_000);
  await page.goto(HARNESS_PATH);
  await page.waitForFunction(
    () => window.__MJTENSU_RECOGNITION_ARTIFACTS__?.status !== 'running',
    undefined,
    { timeout: 110_000 },
  );

  const diagnostics = await page.evaluate(
    () => window.__MJTENSU_RECOGNITION_ARTIFACTS__,
  );
  console.log(`R06_DIAGNOSTICS=${JSON.stringify(diagnostics)}`);

  expect(diagnostics.status, diagnostics.error).toBe('ready');
  expect(diagnostics.modelSetVersion).toBe('recognition-v1-2026-08-27');
  expect(diagnostics.providers).toEqual([
    {
      role: 'detector',
      runtimeSpec: 'nanodet-plus-m-320-v1',
      selectedProvider: 'wasm-simd',
      failedProviders: [],
    },
    {
      role: 'tile-classifier',
      runtimeSpec: 'c8-tile-35-v1',
      selectedProvider: 'wasm-simd',
      failedProviders: [],
    },
    {
      role: 'red-five-classifier',
      runtimeSpec: 'c8-red-five-v1',
      selectedProvider: 'wasm-simd',
      failedProviders: [],
    },
  ]);

  expect(diagnostics.baseFixture).toEqual({
    label: 'invalid',
    logits: [
      0.466319, -7.967026, -8.055287, -4.156061, -2.425662, -12.182508,
      -8.045768, -5.907656, -7.444675, 3.22815, -2.738037, -1.150066,
      -8.604076, -8.585097, -12.526736, -4.427233, -4.462465, -12.188858,
      -0.520438, -0.944048, -6.67593, -8.013445, -10.017096, -8.781754,
      -4.941472, -10.211926, -9.363744, -3.758189, 2.142481, -0.91901,
      -4.81181, 4.053436, -0.056283, -0.170189, 8.551346,
    ],
  });
  expect(diagnostics.redFiveFixture).toEqual({
    label: 'red',
    logits: [-8.893241, 9.968656],
  });

  expect(diagnostics.blankFrameSnapshot).toEqual({
    observations: [],
    meldGroups: [],
    draft: {
      completedHand: [],
      doraIndicators: [],
      meldGroups: [],
    },
    commitEligibility: {
      kind: 'ineligible',
      reason: 'insufficient-visible-tiles',
    },
  });
});
