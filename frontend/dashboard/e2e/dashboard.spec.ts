import { test, expect } from '@playwright/test';

/**
 * Mock mode: the built-in demo fixtures, with the backend deliberately
 * unreachable. This proves the UI renders offline. It proves nothing
 * about the API contracts — see live-backend.spec.ts for those.
 */
test.describe('Dashboard (mock mode, no backend)', () => {
  test.beforeEach(async ({ page }) => {
    // Match by path, not by host. The app's base URL is
    // NEXT_PUBLIC_API_BASE_URL and only falls back to localhost:8000, so a
    // host-qualified pattern stopped matching the moment CI set that
    // variable to http://127.0.0.1:8000. The abort silently did nothing,
    // the app found the real backend CI had started, took the *live* path
    // instead of the mock one, and failed to load a run that only exists
    // as a fixture -- so "mock mode, no backend" was neither, and all
    // seven tests died in this hook waiting for a connection that was
    // never going to happen.
    await page.route('**/health', route => route.abort());
    await page.goto('/runs/demo-2026-04-25');
    await page.getByText('SSE connected').waitFor({ timeout: 10_000 });
    // Guard the guard: "SSE connected" alone does not distinguish the
    // fixture path from a live backend. The provenance label does, and it
    // is what would have caught the abort silently not matching.
    await expect(page.getByTestId('metrics-provenance')).toContainText(/Mock/i);
  });

  test('trajectory tree renders nodes', async ({ page }) => {
    const nodes = page.getByTestId('trajectory-node');
    await expect(nodes.first()).toBeVisible({ timeout: 8_000 });
    await expect(nodes).toHaveCount(7, { timeout: 15_000 });
  });

  test('decision log shows iteration chapters', async ({ page }) => {
    // The mock playback schedules iteration 4 at ~9.7s
    // (startMockSSE: (1500 + 3 * 1300) * 1.8). A 10s window raced it and
    // flaked; wait comfortably past the fixture's own schedule instead.
    const chapter = page.getByText('ITER 4 — more-specific-descriptions');
    await expect(chapter).toBeVisible({ timeout: 25_000 });
  });

  test('score chart renders data points', async ({ page }) => {
    // Switch to chart tab explicitly
    const chartTab = page.getByRole('button', { name: /chart/i });
    await chartTab.click();

    // ScoreChart circles have fill colors matching the whisper palette, not the dot-grid bg
    const dataCircles = page.locator('svg circle[fill="#6a9e78"], svg circle[fill="#8878a8"], svg circle[fill="#b06068"]');
    await expect(dataCircles.first()).toBeVisible({ timeout: 12_000 });
    expect(await dataCircles.count()).toBeGreaterThanOrEqual(1);
  });

  test('diff tab shows diff content', async ({ page }) => {
    // Select a tree node first
    const node = page.getByTestId('trajectory-node').first();
    if (await node.isVisible()) {
      await node.click();
    }

    const diffTab = page.getByRole('button', { name: /diff/i });
    await diffTab.click();

    // Offline fixture content must be labelled as such, never rendered
    // as if it came from a run.
    await expect(
      page.getByTestId('fixture-banner').first().or(page.getByTestId('diff-empty')),
    ).toBeVisible({ timeout: 10_000 });
  });

  test('right-click tree node opens fork modal', async ({ page }) => {
    const node = page.getByTestId('trajectory-node').first();
    await expect(node).toBeVisible({ timeout: 8_000 });

    await node.click({ button: 'right' });

    // Use the modal title which has the fork symbol prefix
    const modalTitle = page.locator('text=⑂ Create Fork');
    await expect(modalTitle).toBeVisible({ timeout: 5_000 });
  });

  test('SSE status indicator shows connected', async ({ page }) => {
    const indicator = page.locator('text=SSE connected');
    await expect(indicator).toBeVisible({ timeout: 8_000 });
  });

  test('META-HARNESS link navigates home', async ({ page }) => {
    const link = page.locator('a', { hasText: 'META-HARNESS' });
    await expect(link).toBeVisible();
    await expect(link).toHaveAttribute('href', '/');
  });
});
