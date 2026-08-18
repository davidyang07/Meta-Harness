import { defineConfig } from '@playwright/test';

/**
 * Two projects, two contracts:
 *
 * - `mock` renders the built-in demo fixtures with the backend
 *   deliberately unreachable. It proves the UI works offline.
 * - `live-backend` drives the real FastAPI backend (deterministic mock
 *   proposer, no LLM calls) and proves the product plumbing works:
 *   runs, checkpoints, forks, branch isolation, diffs, memory.
 *
 * Start the backend before running the live project:
 *
 *     uv run meta-harness serve --port 8000
 *
 * The live suite skips itself with a clear message if no backend answers.
 */
export default defineConfig({
  testDir: './e2e',
  // Live-backend specs create a real run and wait for LangGraph to
  // finish it, which is slower than a fixture render.
  timeout: 120_000,
  expect: { timeout: 10_000 },
  retries: 0,
  use: {
    baseURL: 'http://localhost:3000',
    headless: true,
  },
  webServer: {
    command: 'npm run dev',
    port: 3000,
    reuseExistingServer: true,
    timeout: 60_000,
  },
  projects: [
    {
      name: 'mock',
      testMatch: /dashboard\.spec\.ts/,
      use: { browserName: 'chromium' },
    },
    {
      name: 'live-backend',
      testMatch: /live-backend\.spec\.ts/,
      use: { browserName: 'chromium' },
    },
  ],
});
