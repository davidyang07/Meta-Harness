import { defineConfig } from '@playwright/test';

/**
 * Two projects, two contracts:
 *
 * - `mock` needs no backend. dashboard.spec.ts renders the built-in demo
 *   fixtures with the backend deliberately unreachable; fixture-backend
 *   .spec.ts scripts every API response and drives a fake EventSource
 *   event by event. Together they prove the UI and its state logic.
 * - `live-backend` drives the real FastAPI backend (deterministic mock
 *   proposer, no LLM calls) and proves the product plumbing works:
 *   runs, checkpoints, forks, branch isolation, diffs, memory.
 *
 * Start the backend before running the live project:
 *
 *     uv run meta-harness serve --port 8000
 *
 * The live suite skips itself with a clear message if no backend answers.
 *
 * DASHBOARD_PORT moves the dev server off 3000. Because an existing
 * server on the port is reused, a second checkout must pick its own
 * port or it will test the other checkout's code. Only the `mock`
 * project can move: the backend's CORS policy admits the dashboard on
 * port 3000 alone, so `live-backend` needs the default.
 */
const PORT = Number(process.env.DASHBOARD_PORT ?? 3000);

export default defineConfig({
  testDir: './e2e',
  // Live-backend specs create a real run and wait for LangGraph to
  // finish it, which is slower than a fixture render.
  timeout: 120_000,
  expect: { timeout: 10_000 },
  retries: 0,
  use: {
    baseURL: `http://localhost:${PORT}`,
    headless: true,
  },
  webServer: {
    command: `npm run dev -- --port ${PORT}`,
    port: PORT,
    reuseExistingServer: true,
    timeout: 60_000,
  },
  projects: [
    {
      name: 'mock',
      testMatch: /(dashboard|fixture-backend)\.spec\.ts/,
      use: { browserName: 'chromium' },
    },
    {
      name: 'live-backend',
      testMatch: /live-backend\.spec\.ts/,
      use: { browserName: 'chromium' },
    },
  ],
});
