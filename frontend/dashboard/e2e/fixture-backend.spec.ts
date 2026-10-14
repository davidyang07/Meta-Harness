import { test, expect, type Page, type Route } from '@playwright/test';

/**
 * The dashboard against a scripted backend.
 *
 * Every API response is a fixture served by `page.route`, and EventSource
 * is replaced by a fake the test drives event by event. That pins down
 * the state logic — statuses, threads, forks, the decision log, the
 * stream's lifecycle — deterministically, with no backend and no model.
 * It proves nothing about the real API; live-backend.spec.ts does that.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000';
const API_ORIGIN = new URL(API_BASE).origin;

const RUN = 'fixture-run';
const FORK = `${RUN}.fork.abcd1234`;

const runDetail = {
  run_id: RUN,
  thread_id: RUN,
  status: 'completed',
  current_iteration: 3,
  metrics_source: 'mock',
  manifest: { mock_bench: true, metrics_source: 'mock' },
  summary_rows: [
    {
      iteration: 0, candidate: 'baseline', parent_candidate_name: null,
      scores: { accuracy: 0.5, per_task: {} }, delta: 0, thread_id: RUN,
      accepted: true, reason: 'measured baseline (search root)',
    },
    {
      iteration: 1, candidate: 'cand-accepted', parent_candidate_name: 'baseline',
      scores: { accuracy: 0.7, per_task: {} }, delta: 0.2, thread_id: RUN,
      accepted: true, reason: 'accepted',
    },
    {
      iteration: 2, candidate: 'cand-rejected', parent_candidate_name: 'cand-accepted',
      scores: { accuracy: 0.6, per_task: {} }, delta: -0.1, thread_id: RUN,
      accepted: false, reason: 'regression',
    },
    {
      iteration: 3, candidate: 'cand-smoke', parent_candidate_name: 'cand-accepted',
      scores: { accuracy: 0, per_task: {} }, delta: null, thread_id: RUN,
      status: 'smoke_failed', accepted: false, reason: 'smoke_failed',
    },
    {
      iteration: 2, candidate: 'fork-cand', parent_candidate_name: 'cand-accepted',
      scores: { accuracy: 0.8, per_task: {} }, delta: 0.1, thread_id: FORK,
      accepted: true, reason: 'accepted',
    },
  ],
  frontier_val: { _best: { name: 'cand-accepted' }, _pareto_names: ['baseline', 'cand-accepted'] },
  branch_frontiers: {
    [RUN]: { _best: { name: 'cand-accepted' }, _pareto_names: ['baseline', 'cand-accepted'] },
    [FORK]: { _best: { name: 'fork-cand' }, _pareto_names: ['fork-cand'] },
  },
};

const trajectory = {
  trajectory: {
    run_id: RUN,
    threads: [
      { thread_id: RUN, parent_thread_id: null, parent_checkpoint_id: null, branch_id: null, name: null, status: 'completed', live: false },
      { thread_id: FORK, parent_thread_id: RUN, parent_checkpoint_id: 'ck-main-1', branch_id: 'abcd1234', name: 'left', status: 'completed', live: false },
    ],
    edges: [{ source: RUN, target: FORK, parent_checkpoint_id: 'ck-main-1' }],
  },
};

/** Replaces window.EventSource with one the test drives by hand. */
function installFakeEventSource() {
  type Fake = EventTarget & { url: string; readyState: number; close: () => void };
  const w = window as unknown as {
    __es: Fake[];
    __emit: (type: string, data: unknown, id?: string) => void;
    __transportError: (closed: boolean) => void;
    EventSource: unknown;
  };
  w.__es = [];
  class FakeEventSource extends EventTarget {
    static CONNECTING = 0;
    static OPEN = 1;
    static CLOSED = 2;
    url: string;
    readyState = 0;
    withCredentials = false;
    constructor(url: string) {
      super();
      this.url = url;
      w.__es.push(this as unknown as Fake);
      setTimeout(() => {
        if (this.readyState === 2) return;
        this.readyState = 1;
        this.dispatchEvent(new Event('open'));
      }, 0);
    }
    close() {
      this.readyState = 2;
    }
  }
  w.EventSource = FakeEventSource;
  w.__emit = (type, data, id) => {
    for (const es of w.__es) {
      if (es.readyState !== 1) continue;
      es.dispatchEvent(new MessageEvent(type, { data: JSON.stringify(data), lastEventId: id ?? '' }));
    }
  };
  w.__transportError = (closed) => {
    for (const es of w.__es) {
      if (es.readyState === 2) continue;
      es.readyState = closed ? 2 : 0;
      es.dispatchEvent(new Event('error'));
    }
  };
}

function json(route: Route, status: number, body: unknown) {
  return route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });
}

function apiPath(path: string) {
  return (url: URL) => url.origin === API_ORIGIN && url.pathname === path;
}

async function serveFixtureBackend(page: Page, run: string = RUN) {
  // Anything not scripted below is a 404, never a real backend.
  await page.route(url => url.origin === API_ORIGIN, route => json(route, 404, { detail: 'not scripted' }));
  await page.route(apiPath('/health'), route => json(route, 200, { status: 'ok', persistence: 'memory' }));
  await page.route(apiPath(`/runs/${run}`), route => json(route, 200, runDetail));
  await page.route(apiPath(`/runs/${run}/trajectory`), route => json(route, 200, trajectory));
  await page.route(apiPath(`/runs/${run}/checkpoints`), route => {
    const thread = new URL(route.request().url()).searchParams.get('thread_id');
    if (thread === FORK) {
      return json(route, 200, {
        checkpoints: [
          {
            checkpoint_id: 'ck-fork-2', thread_id: FORK, iteration: 2, next: ['propose'],
            values_summary: { best_candidate: 'fork-cand', iteration: 2 },
          },
        ],
      });
    }
    // The root thread's history is unavailable, as on a backend running
    // with in-memory persistence after a restart.
    return json(route, 404, { detail: 'run graph unavailable without an active in-process record' });
  });
}

async function emit(page: Page, type: string, data: Record<string, unknown>, id?: string) {
  await page.evaluate(
    ([t, d, i]) => (window as unknown as { __emit: (a: unknown, b: unknown, c: unknown) => void }).__emit(t, d, i),
    [type, data, id] as const,
  );
}

function node(page: Page, candidate: string) {
  return page.locator(`[data-testid="trajectory-node"][data-candidate="${candidate}"]`);
}

async function openStreams(page: Page): Promise<number> {
  return page.evaluate(
    () => (window as unknown as { __es: { readyState: number }[] }).__es.filter(es => es.readyState !== 2).length,
  );
}

test.describe('Dashboard against a scripted backend', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(installFakeEventSource);
    await serveFixtureBackend(page);
    await page.goto(`/runs/${RUN}`);
    await expect(page.getByText('SSE connected')).toBeVisible({ timeout: 30_000 });
    await expect(node(page, 'fork-cand')).toHaveCount(1, { timeout: 15_000 });
  });

  test('verdicts come from the backend: rejected and smoke-failed are distinct, bests are per thread', async ({ page }) => {
    await expect(node(page, 'baseline')).toHaveAttribute('data-status', 'seed');
    await expect(node(page, 'cand-accepted')).toHaveAttribute('data-status', 'best');
    await expect(node(page, 'cand-rejected')).toHaveAttribute('data-status', 'rejected');
    await expect(node(page, 'cand-smoke')).toHaveAttribute('data-status', 'smoke_failed');
    await expect(node(page, 'fork-cand')).toHaveAttribute('data-status', 'best');
    // No score, so no 0.00.
    await expect(node(page, 'cand-smoke').getByTestId('trajectory-node-score')).toHaveText('—');
    await expect(node(page, 'cand-smoke')).toContainText('Smoke failed');
  });

  test('a checkpoint 404 still loads branches and opens exactly one stream', async ({ page }) => {
    await expect(page.getByTestId('branch-count')).toContainText('1 branch');
    // React StrictMode mounts the page twice in dev; only one stream may survive.
    await page.waitForTimeout(500);
    expect(await openStreams(page)).toBe(1);
    // With no token configured the stream URL carries none.
    const url = await page.evaluate(() => (window as unknown as { __es: { url: string }[] }).__es[0].url);
    expect(url).toBe(`${API_BASE}/runs/${RUN}/stream`);
  });

  test('a backend error event is logged as a run event, not read as a disconnect', async ({ page }) => {
    await emit(page, 'error', { thread_id: RUN, node: 'outer_loop', message: 'proposer crashed' }, 'evt-error');
    await expect(page.getByTestId('run-events')).toContainText('outer_loop: proposer crashed');
    await expect(page.getByText('SSE connected')).toBeVisible();
    await expect(page.getByTestId('status-error')).toContainText('proposer crashed');
  });

  test('a transport error does read as a disconnect', async ({ page }) => {
    await page.evaluate(() => (window as unknown as { __transportError: (c: boolean) => void }).__transportError(false));
    await expect(page.getByText('Disconnected')).toBeVisible();
  });

  test('every per-task line of one eval-result survives, and iteration-complete finishes the real chapter', async ({ page }) => {
    await emit(page, 'candidate-created', {
      thread_id: RUN, candidate: 'cand-5', parent: 'cand-accepted', iteration: 4,
      hypothesis: 'retry flaky verify once',
    }, 'evt-c5');
    await emit(page, 'eval-result', {
      thread_id: RUN, candidate: 'cand-5', iteration: 4, accuracy: 0.6,
      per_task: {
        'task-001-alpha': { pass_rate: 1, trials: [true] },
        'task-002-beta': { pass_rate: 0.5, trials: [true, false] },
        'task-003-gamma': { pass_rate: 0, trials: [false] },
      },
    }, 'evt-e5');
    const chapter = page.getByTestId('iteration-chapter').filter({ hasText: 'cand-5' });
    await expect(chapter).toContainText('task alpha: 100% pass rate');
    await expect(chapter).toContainText('task beta: 50% pass rate');
    await expect(chapter).toContainText('task gamma: 0% pass rate');

    await emit(page, 'iteration-complete', { thread_id: RUN, iteration: 4, status: 'no_improvement' }, 'evt-i4');
    await expect(chapter).toContainText('Rejected');
    await expect(chapter).toContainText('iteration 4 complete: no improvement');
    await expect(page.getByText('_iter_4')).toHaveCount(0);
  });

  test('a rejection reported by frontier-updated is shown as rejected', async ({ page }) => {
    await emit(page, 'candidate-created', { thread_id: RUN, candidate: 'cand-6', parent: 'cand-accepted', iteration: 4 }, 'evt-c6');
    await emit(page, 'eval-result', { thread_id: RUN, candidate: 'cand-6', iteration: 4, accuracy: 0.65, per_task: {} }, 'evt-e6');
    await emit(page, 'frontier-updated', {
      thread_id: RUN, candidate: 'cand-6', iteration: 4, accepted: false, status: 'rejected',
      best_candidate: 'cand-accepted', frontier: ['baseline', 'cand-accepted'], delta: -0.05,
      scores: { accuracy: 0.65, per_task: {} },
    }, 'evt-f6');
    await expect(node(page, 'cand-6')).toHaveAttribute('data-status', 'rejected');
    await expect(node(page, 'cand-accepted')).toHaveAttribute('data-status', 'best');
  });

  test("a fork's frontier update leaves main's nodes alone", async ({ page }) => {
    await emit(page, 'candidate-created', { thread_id: FORK, candidate: 'fork-cand-2', parent: 'fork-cand', iteration: 3 }, 'evt-cf2');
    await emit(page, 'frontier-updated', {
      thread_id: FORK, candidate: 'fork-cand-2', iteration: 3, accepted: true, status: 'best',
      best_candidate: 'fork-cand-2', frontier: ['fork-cand-2'], delta: 0.05,
      scores: { accuracy: 0.85, per_task: {} },
    }, 'evt-ff2');
    await expect(node(page, 'fork-cand-2')).toHaveAttribute('data-status', 'best');
    await expect(node(page, 'fork-cand')).toHaveAttribute('data-status', 'accepted');
    // Main's best is not on the fork's frontier; it must not be demoted.
    await expect(node(page, 'cand-accepted')).toHaveAttribute('data-status', 'best');
  });

  test('forking a fork-branch node uses its own thread, and the fork is shown once', async ({ page }) => {
    const NEW_FORK = `${FORK}.fork.ffff0000`;
    let forkBody: Record<string, unknown> | null = null;
    await page.route(apiPath(`/runs/${RUN}/fork`), async route => {
      forkBody = route.request().postDataJSON();
      // The backend announces the fork on the stream before it answers.
      await emit(page, 'fork-created', {
        thread_id: NEW_FORK, parent_thread_id: FORK, parent_checkpoint_id: 'ck-fork-2',
        mods_summary: { keys: ['best_candidate', 'proposer_prior'], count: 2 },
      }, 'evt-fork');
      return json(route, 202, {
        thread_id: NEW_FORK, status: 'running', parent_checkpoint_id: 'ck-fork-2', branch_id: 'ffff0000',
      });
    });

    await node(page, 'fork-cand').dispatchEvent('contextmenu');
    await expect(page.getByRole('dialog', { name: 'Create fork' })).toBeVisible();
    // Resolved from the fork's own checkpoint history, not the root's.
    await expect(page.getByText('checkpoint ck-fork-2')).toBeVisible();
    await page.getByPlaceholder(/Explore few-shot/).fill('try a narrower plan');
    await page.getByRole('button', { name: 'Create fork', exact: true }).click();

    // The modal closes once the fork is accepted and the branches reloaded.
    await expect(page.getByRole('dialog', { name: 'Create fork' })).toHaveCount(0);
    await expect(page.getByText('Fork Created')).toHaveCount(1);
    // One card, and it is the dashboard's record: it knows the prior.
    await expect(page.getByText('try a narrower plan')).toBeVisible();
    expect(forkBody).toMatchObject({ parent_checkpoint_id: 'ck-fork-2', parent_thread_id: FORK });

    // The same fork replayed again (an SSE reconnect) is still one fork.
    await emit(page, 'fork-created', {
      thread_id: NEW_FORK, parent_thread_id: FORK, parent_checkpoint_id: 'ck-fork-2',
    }, 'evt-fork');
    await expect(page.getByText('Fork Created')).toHaveCount(1);
  });

  test("after a fork's node is auto-selected, the user can select another node", async ({ page }) => {
    const NEW_FORK = `${RUN}.fork.eeee0000`;
    const focus = page.getByTestId('candidate-focus');
    await emit(page, 'fork-created', { thread_id: NEW_FORK, parent_thread_id: RUN, parent_checkpoint_id: 'ck-main-1' }, 'evt-fork-b');
    await emit(page, 'candidate-created', { thread_id: NEW_FORK, candidate: 'branch-first', parent: 'cand-accepted', iteration: 2 }, 'evt-bf');
    await expect(focus).toContainText('branch-first');

    await node(page, 'cand-rejected').dispatchEvent('click');
    await expect(focus).toContainText('cand-rejected');
    // More events re-render the tree; the selection must not snap back.
    await emit(page, 'eval-result', { thread_id: NEW_FORK, candidate: 'branch-first', iteration: 2, accuracy: 0.7, per_task: {} }, 'evt-bf-e');
    await page.waitForTimeout(300);
    await expect(focus).toContainText('cand-rejected');
  });
});

test.describe('Dashboard lifecycle against a scripted backend', () => {
  test('leaving while the run is still loading leaks no stream', async ({ page }) => {
    await page.addInitScript(installFakeEventSource);
    await serveFixtureBackend(page);
    // Compile "/" now, so navigating to it below is quick in dev mode.
    await page.goto('/');
    let detailRequested = false;
    await page.route(apiPath(`/runs/${RUN}`), async route => {
      detailRequested = true;
      // Still loading when the user leaves. (The health probe is the same
      // kind of await, but its own 2s timeout caps how long it can hang.)
      await new Promise(resolve => setTimeout(resolve, 4000));
      await json(route, 200, runDetail).catch(() => undefined);
    });
    await page.goto(`/runs/${RUN}`);
    // Requested from an effect, so the page is hydrated and the link
    // below navigates client-side, in this same window.
    await expect.poll(() => detailRequested, { timeout: 30_000 }).toBe(true);
    await page.evaluate(() => {
      (window as unknown as { __marker: number }).__marker = 1;
    });
    await page.locator('a', { hasText: 'META-HARNESS' }).click();
    await page.waitForURL(url => url.pathname === '/');
    // Let the delayed response land and connect() run to completion.
    await page.waitForTimeout(6000);
    expect(await page.evaluate(() => (window as unknown as { __marker?: number }).__marker)).toBe(1);
    expect(await openStreams(page)).toBe(0);
  });

  test('a 401 from the backend explains how to fix the token', async ({ page }) => {
    await page.addInitScript(installFakeEventSource);
    await serveFixtureBackend(page);
    await page.route(apiPath(`/runs/${RUN}`), route => json(route, 401, { detail: 'missing or invalid API token' }));
    await page.goto(`/runs/${RUN}`);
    const error = page.getByTestId('status-error');
    await expect(error).toContainText('401', { timeout: 30_000 });
    await expect(error).toContainText('NEXT_PUBLIC_META_HARNESS_API_TOKEN');
    expect(await openStreams(page)).toBe(0);
  });
});
