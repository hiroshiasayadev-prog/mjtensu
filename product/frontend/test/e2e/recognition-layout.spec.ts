import { expect, test, type Locator, type Page } from '@playwright/test';

const HARNESS_PATH = '/test/e2e/fake-flow.html';

test.describe('Recognition viewport controls', () => {
  test('keeps controls visible without absolute positioning on the portrait-locked landscape surface', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto(`${HARNESS_PATH}?scenario=hold-recognition`);
    await page.getByRole('button', { name: '判定する' }).click();

    const landscapeUi = page.getByTestId('recognition-landscape-ui-surface');
    const controlsLayer = page.getByTestId('recognition-global-controls-layer');
    const exit = page.getByTestId('recognition-global-exit');
    const debug = page.getByTestId('recognition-debug-capture');

    await expect(page).toHaveURL('/recognition');
    await expect(landscapeUi).toHaveCSS(
      'transform',
      /matrix\(0, 1, -1, 0/,
    );
    await expectControlsToUseNormalFlow(landscapeUi, controlsLayer, exit, debug);
    await expectControlInViewportAndHitTestable(page, exit);
    await expectControlInViewportAndHitTestable(page, debug);

    await page.setViewportSize({ width: 844, height: 390 });

    await expect(landscapeUi).toHaveCSS('transform', 'none');
    await expectControlsToUseNormalFlow(landscapeUi, controlsLayer, exit, debug);
    await expectControlInViewportAndHitTestable(page, exit);
    await expectControlInViewportAndHitTestable(page, debug);
  });
});

async function expectControlsToUseNormalFlow(
  landscapeUi: Locator,
  controlsLayer: Locator,
  exit: Locator,
  debug: Locator,
): Promise<void> {
  await expect(exit).toBeVisible();
  await expect(debug).toBeVisible();

  const structure = await controlsLayer.evaluate((layer) => ({
    parentTestId: layer.parentElement?.getAttribute('data-testid') ?? null,
    layerPosition: getComputedStyle(layer).position,
    exitPosition: getComputedStyle(
      layer.querySelector('[data-testid="recognition-global-exit"]')!,
    ).position,
    debugPosition: getComputedStyle(
      layer.querySelector('[data-testid="recognition-debug-capture"]')!,
    ).position,
  }));
  expect(structure).toEqual({
    parentTestId: 'recognition-landscape-ui-surface',
    layerPosition: 'static',
    exitPosition: 'static',
    debugPosition: 'static',
  });
  await expect(controlsLayer).toHaveCSS('display', 'flex');
  await expect(landscapeUi).toContainText('終了');
}

async function expectControlInViewportAndHitTestable(
  page: Page,
  locator: Locator,
): Promise<void> {
  await expect(locator).toBeInViewport();
  const box = await locator.boundingBox();
  expect(box).not.toBeNull();
  if (box === null) {
    return;
  }

  const viewport = page.viewportSize();
  expect(viewport).not.toBeNull();
  if (viewport === null) {
    return;
  }

  expect(box.x).toBeGreaterThanOrEqual(0);
  expect(box.y).toBeGreaterThanOrEqual(0);
  expect(box.x + box.width).toBeLessThanOrEqual(viewport.width);
  expect(box.y + box.height).toBeLessThanOrEqual(viewport.height);

  const hitTestable = await locator.evaluate((element) => {
    const rect = element.getBoundingClientRect();
    const x = rect.left + rect.width / 2;
    const y = rect.top + rect.height / 2;
    const hit = document.elementFromPoint(x, y);
    return hit === element || (hit instanceof Element && element.contains(hit));
  });
  expect(hitTestable).toBe(true);
}
