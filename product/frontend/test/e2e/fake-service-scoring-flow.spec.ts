import { expect, test, type Page } from '@playwright/test';

const HARNESS_PATH = '/test/e2e/fake-flow.html';

test.describe('fake-service scoring flow acceptance', () => {
  test('Top -> Recognition -> Conditions -> Result and history replacement', async ({ page }) => {
    await openHarness(page);

    await page.getByRole('button', { name: '判定する' }).click();
    await expect(page).toHaveURL('/conditions');
    await expect(page.getByRole('heading', { name: '条件入力' })).toBeVisible();
    await expect(page.getByText('現在の和了牌: recognition-1-hand-14')).toBeVisible();

    await page.goBack();
    await expect(page).toHaveURL('/');
    await expect(page.getByRole('heading', { name: 'mjtensu' })).toBeVisible();

    await page.getByRole('button', { name: '判定する' }).click();
    await expect(page).toHaveURL('/conditions');
    await page.getByRole('button', { name: '計算する' }).click();
    await expect(page).toHaveURL('/result');
    await expect(page.getByRole('heading', { name: '結果' })).toBeVisible();
    await expect(page.getByText('6,000点', { exact: true })).toBeVisible();
  });

  test('Recognition preparation exposes camera-first and runtime-first states', async ({ page }) => {
    await openHarness(page, 'camera-slow');
    await page.getByRole('button', { name: '判定する' }).click();
    await expect(page.getByRole('status')).toContainText('カメラを起動しています');
    await expect(page.getByLabel('カメラプレビュー')).toHaveCount(0);
    await expect(page).toHaveURL('/conditions');

    await openHarness(page, 'runtime-slow');
    await page.getByRole('button', { name: '判定する' }).click();
    await expect(page.getByLabel('カメラプレビュー')).toBeVisible();
    await expect(page.getByRole('status')).toContainText('認識モデルを準備しています');
    await expect(page).toHaveURL('/conditions');
  });

  test('camera failure retry is camera-owned and preserves healthy runtime', async ({ page }) => {
    await openHarness(page, 'camera-retry');
    await page.getByRole('button', { name: '判定する' }).click();

    await expect(page.getByRole('alert')).toContainText('カメラの使用が許可されていません');
    await expectDiagnostics(page, {
      cameraOpenCalls: 1,
      runtimeInitializeCalls: 1,
      recognizerStartCalls: 0,
    });

    await page.getByRole('button', { name: 'カメラを再試行' }).click();
    await expect(page).toHaveURL('/conditions');
    await expectDiagnostics(page, {
      cameraOpenCalls: 2,
      runtimeInitializeCalls: 1,
      recognizerStartCalls: 1,
    });
  });

  test('runtime failure retry is recognition-owned and preserves healthy camera', async ({ page }) => {
    await openHarness(page, 'runtime-retry');
    await page.getByRole('button', { name: '判定する' }).click();

    await expect(page.getByRole('alert')).toContainText('認識モデルを取得できませんでした');
    await expect(page.getByLabel('カメラプレビュー')).toBeVisible();
    await expectDiagnostics(page, {
      cameraOpenCalls: 1,
      runtimeInitializeCalls: 1,
      recognizerStartCalls: 0,
    });

    await page.getByRole('button', { name: '認識モデルを再試行' }).click();
    await expect(page).toHaveURL('/conditions');
    await expectDiagnostics(page, {
      cameraOpenCalls: 1,
      runtimeInitializeCalls: 2,
      recognizerStartCalls: 1,
    });
  });

  test('Conditions supports winning-tile selection, preview recovery, condition editing, and correction entry', async ({ page }) => {
    await reachConditions(page);

    const winningTiles = page.getByRole('group', { name: '和了牌選択' });
    await winningTiles.getByRole('button', { name: '9s 2' }).click();
    await expect(page.getByText('和了形として成立していません')).toBeVisible();
    await expect(page.getByRole('button', { name: '計算する' })).toBeDisabled();

    await winningTiles.getByRole('button', { name: '1m 1' }).click();
    await expect(page.getByText('現在の和了牌: recognition-1-hand-1')).toBeVisible();
    await expect(page.getByText('門前清自摸和 1翻')).toBeVisible();

    await page.getByRole('radio', { name: 'ロン' }).check();
    await expect(page.getByText('役なし')).toBeVisible();
    await expect(page.getByRole('button', { name: '計算する' })).toBeDisabled();

    await page.getByRole('radio', { name: 'リーチ', exact: true }).check();
    await expect(page.getByText('リーチ 1翻')).toBeVisible();
    await expect(page.getByRole('button', { name: '計算する' })).toBeEnabled();

    await page.getByRole('radio', { name: '西', exact: true }).first().check();
    await page.getByRole('radio', { name: '北', exact: true }).last().check();
    await expect(page.getByText(/入力の組み合わせを確認してください/)).toBeVisible();
    await expect(page.getByRole('button', { name: '計算する' })).toBeDisabled();

    await page.getByRole('radio', { name: '東', exact: true }).last().check();
    await expect(page.getByText('リーチ 1翻')).toBeVisible();
    await expect(page.getByRole('heading', { name: '牌姿修正' })).toBeVisible();
    await expect(page.getByRole('button', { name: '牌姿を反映' })).toBeEnabled();

    await page.getByRole('button', { name: '計算する' }).click();
    await expect(page).toHaveURL('/result');
  });

  test('Result -> Conditions -> Result recalculates changed conditions', async ({ page }) => {
    await reachResult(page);
    await expect(page.getByText('6,000点', { exact: true })).toBeVisible();

    await page.getByRole('button', { name: '条件を修正' }).click();
    await expect(page).toHaveURL('/conditions');
    await page.getByRole('radio', { name: '南', exact: true }).last().check();
    await page.getByRole('button', { name: '計算する' }).click();

    await expect(page).toHaveURL('/result');
    await expect(page.getByText('4,000点', { exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: '親子を修正' })).toHaveText('子');
  });

  test('pre-confirm recognition correction cancel preserves old Result', async ({ page }) => {
    await reachResult(page);
    await page.getByRole('button', { name: '認識結果を修正' }).click();
    await replaceFirstCorrectionTile(page, '2m');

    await page.getByRole('button', { name: 'キャンセル' }).click();
    await expect(page).toHaveURL('/result');
    await expect(page.getByText('6,000点', { exact: true })).toBeVisible();
    await expect(page.getByText('12,000点', { exact: true })).toHaveCount(0);
  });

  test('confirmed recognition correction immediately recalculates when still ready', async ({ page }) => {
    await reachResult(page);
    await page.getByRole('button', { name: '認識結果を修正' }).click();
    await replaceFirstCorrectionTile(page, '2m');

    await page.getByRole('button', { name: '修正を確定' }).click();
    await expect(page).toHaveURL('/result');
    await expect(page.getByText('12,000点', { exact: true })).toBeVisible();
    await expect(page.getByText('6,000点', { exact: true })).toHaveCount(0);
  });

  test('confirmed repair-needed correction falls back to Conditions and stale Result never returns', async ({ page }) => {
    await reachResult(page);
    await page.getByRole('button', { name: '認識結果を修正' }).click();
    await replaceFirstCorrectionTile(page, '9p');

    await page.getByRole('button', { name: '修正を確定' }).click();
    await expect(page).toHaveURL('/conditions');
    await expect(page.getByText('役なし')).toBeVisible();

    await page.goBack();
    await expect(page).toHaveURL('/result');
    await expect(page.getByText('計算結果がまだありません。')).toBeVisible();
    await expect(page.getByText('6,000点', { exact: true })).toHaveCount(0);
  });

  test('explicit new Recognition replaces the prior scoring session', async ({ page }) => {
    await reachResult(page);
    await page.getByRole('button', { name: 'もう一度判定' }).click();

    await expect(page).toHaveURL('/conditions');
    await expect(page.getByText('現在の和了牌: recognition-2-hand-14')).toBeVisible();

    await page.goBack();
    await expect(page).toHaveURL('/result');
    await expect(page.getByText('計算結果がまだありません。')).toBeVisible();
    await expect(page.getByText('6,000点', { exact: true })).toHaveCount(0);
  });

  test('Conditions and Result route guards redirect to Top without a session', async ({ page }) => {
    await openHarness(page, 'primary', '/conditions');
    await expect(page).toHaveURL('/');
    await expect(page.getByRole('heading', { name: 'mjtensu' })).toBeVisible();

    await openHarness(page, 'primary', '/result');
    await expect(page).toHaveURL('/');
    await expect(page.getByRole('heading', { name: 'mjtensu' })).toBeVisible();
  });

  test('Help round-trip preserves the active scoring session', async ({ page }) => {
    await reachResult(page);
    await page.getByRole('link', { name: 'mjtensu' }).click();
    await page.getByRole('button', { name: '使い方' }).click();
    await expect(page).toHaveURL('/help');
    await expect(page.getByRole('heading', { name: '使い方' })).toBeVisible();

    await page.getByRole('button', { name: 'トップへ戻る' }).click();
    await expect(page).toHaveURL('/');

    await page.goBack();
    await page.goBack();
    await page.goBack();
    await expect(page).toHaveURL('/result');
    await expect(page.getByText('6,000点', { exact: true })).toBeVisible();
  });
});

async function openHarness(
  page: Page,
  scenario = 'primary',
  route = '/',
): Promise<void> {
  await page.goto(
    `${HARNESS_PATH}?scenario=${encodeURIComponent(scenario)}&route=${encodeURIComponent(route)}`,
  );
  await expect(page.getByRole('heading', { name: route === '/' ? 'mjtensu' : /.+/ })).toBeVisible();
}

async function reachConditions(page: Page): Promise<void> {
  await openHarness(page);
  await page.getByRole('button', { name: '判定する' }).click();
  await expect(page).toHaveURL('/conditions');
  await expect(page.getByRole('heading', { name: '条件入力' })).toBeVisible();
}

async function reachResult(page: Page): Promise<void> {
  await reachConditions(page);
  await page.getByRole('button', { name: '計算する' }).click();
  await expect(page).toHaveURL('/result');
  await expect(page.getByRole('heading', { name: '結果' })).toBeVisible();
}

async function replaceFirstCorrectionTile(page: Page, tile: string): Promise<void> {
  await page.getByRole('button', { name: '手牌 1 1m' }).click();
  const selector = page.getByRole('dialog', { name: '牌を選択' });
  await expect(selector).toBeVisible();
  await selector.getByRole('button', { name: tile, exact: true }).click();
}

async function expectDiagnostics(
  page: Page,
  expected: Partial<Window['__MJTENSU_E2E__']>,
): Promise<void> {
  await expect
    .poll(async () => page.evaluate(() => window.__MJTENSU_E2E__))
    .toMatchObject(expected);
}
