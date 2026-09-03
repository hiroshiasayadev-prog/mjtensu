import { expect, test } from '@playwright/test';

const HARNESS_PATH = '/test/e2e/recognition-production-artifacts.html';

test('production Recognition model set loads and executes bounded real-artifact fixtures', async ({
  page,
}) => {
  test.setTimeout(120_000);
  await page.goto(HARNESS_PATH);
  await page.waitForFunction(
    () =>
      window.__MJTENSU_RECOGNITION_ARTIFACTS__ !== undefined &&
      window.__MJTENSU_RECOGNITION_ARTIFACTS__.status !== 'running',
    undefined,
    { timeout: 110_000 },
  );

  const diagnostics = await page.evaluate(
    () => window.__MJTENSU_RECOGNITION_ARTIFACTS__,
  );
  console.log(`R06_DIAGNOSTICS=${JSON.stringify(diagnostics)}`);

  expect(diagnostics.status, diagnostics.error).toBe('ready');
  expect(diagnostics.modelSetVersion).toBe('recognition-v8-2026-09-04');
  expect(diagnostics.providers).toEqual([
    {
      role: 'detector',
      runtimeSpec: 'rotated-fcos-nano-320-v1',
      selectedProvider: 'wasm-simd',
      failedProviders: [],
    },
    {
      role: 'tile-classifier',
      runtimeSpec: 'gray64-tile-35-v1',
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

  expect(diagnostics.baseFixture?.label).toEqual(expect.any(String));
  expect(diagnostics.baseFixture?.logits).toHaveLength(35);
  expect(diagnostics.baseFixture?.logits.every(Number.isFinite)).toBe(true);
  expect(diagnostics.redFiveFixture).toEqual({
    label: 'red',
    logits: [-8.893241, 9.968656],
  });

  expect(diagnostics.blankFrameSnapshot).toMatchObject({
    meldGroups: [],
    meldCommonAngleRadians: null,
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
  const blankFrameSnapshot = diagnostics.blankFrameSnapshot as {
    readonly observations: readonly {
      readonly classification: { readonly kind: string };
    }[];
  };
  expect(
    blankFrameSnapshot.observations.every(
      (observation) => observation.classification.kind === 'invalid',
    ),
  ).toBe(true);
});
