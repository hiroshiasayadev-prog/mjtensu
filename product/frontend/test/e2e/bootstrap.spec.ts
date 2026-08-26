import { expect, test } from '@playwright/test';

test('production application bootstrap renders', async ({ page }) => {
  await page.goto('/');

  await expect(page.getByRole('heading', { name: 'mjtensu' })).toBeVisible();
});
