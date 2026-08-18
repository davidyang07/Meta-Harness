import { test, expect, request as playwrightRequest, type APIRequestContext } from '@playwright/test';

/**
 * End-to-end against a REAL FastAPI backend.
 *
 * The mock-mode suite (dashboard.spec.ts) renders fixtures with the
 * backend deliberately unreachable, so it can pass while every API
 * contract is broken. This suite exercises the actual product plumbing:
 * create a run, read its checkpoints, fork one, and confirm the branch
 * shows up — all through the real REST/SSE surface.
 *
 * It stays free: the run uses the deterministic mock proposer and mock
 * benchmark, so no LLM is called. The backend, LangGraph, the
 * checkpointer and the fork machinery are all real.
 *
 * Requires a backend on API_BASE (default http://localhost:8000):
 *
 *     uv run meta-harness serve --port 8000
 */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000';

let api: APIRequestContext;
let runId: string;
let forkCheckpointId: string;

async function backendIsUp(ctx: APIRequestContext): Promise<boolean> {
  try {
    const res = await ctx.get(`${API_BASE}/health`);
    return res.ok();
  } catch {
    return false;
  }
}

/** Poll the run until it reaches a terminal status. */
async function waitForRun(ctx: APIRequestContext, id: string, timeoutMs = 90_000) {
  const deadline = Date.now() + timeoutMs;
  let last: Record<string, unknown> = {};
  while (Date.now() < deadline) {
    const res = await ctx.get(`${API_BASE}/runs/${id}`);
    if (res.ok()) {
      last = await res.json();
      if (last.status === 'completed' || last.status === 'failed') return last;
    }
    await new Promise(r => setTimeout(r, 400));
  }
  throw new Error(`run ${id} never finished; last status=${String(last.status)}`);
}

test.describe('Dashboard against a live backend', () => {
  test.beforeAll(async () => {
    api = await playwrightRequest.newContext();
    test.skip(
      !(await backendIsUp(api)),
      `no backend at ${API_BASE} — start it with "uv run meta-harness serve"`,
    );

    runId = `e2e-live-${Date.now()}`;
    const created = await api.post(`${API_BASE}/runs`, {
      data: {
        run_name: runId,
        proposer: 'mock',
        mock_bench: true,
        budget: 2,
        trials: 2,
        fresh: true,
      },
    });
    expect(created.status(), await created.text()).toBe(201);

    const finished = await waitForRun(api, runId);
    expect(finished.status).toBe('completed');
  });

  test.afterAll(async () => {
    if (runId) await api.delete(`${API_BASE}/runs/${runId}`).catch(() => {});
    await api.dispose();
  });

  // ── backend contracts ───────────────────────────────────────────────

  test('health reports which persistence backend is actually in use', async () => {
    const res = await api.get(`${API_BASE}/health`);
    expect(res.ok()).toBeTruthy();
    const body = await res.json();
    expect(body.status).toBe('ok');
    // Degradation must be visible, never silent.
    expect(body).toHaveProperty('persistence');
    if (body.persistence !== 'postgres') {
      expect(body.persistence_error).toBeTruthy();
    }
  });

  test('the run exposes a measured baseline as the root of the tree', async () => {
    const res = await api.get(`${API_BASE}/runs/${runId}`);
    const body = await res.json();
    const rows = body.summary_rows as Array<Record<string, unknown>>;

    expect(rows.length).toBeGreaterThanOrEqual(3); // baseline + 2 iterations
    expect(rows[0].candidate).toBe('baseline');
    expect(rows[0].iteration).toBe(0);
    expect(rows[0].parent_candidate_name).toBeNull();
    // Mock data is labelled as mock, never as a measurement.
    expect(body.metrics_source).toBe('mock');
    for (const row of rows) expect(row.thread_id).toBeTruthy();
  });

  test('checkpoint history exists and exposes a forkable point', async () => {
    const res = await api.get(`${API_BASE}/runs/${runId}/checkpoints`);
    expect(res.ok()).toBeTruthy();
    const { checkpoints } = await res.json();
    expect(checkpoints.length).toBeGreaterThan(3);

    const forkable = checkpoints.find(
      (c: { iteration: number; next: string[] }) =>
        c.iteration === 1 && c.next?.includes('propose'),
    );
    expect(forkable, 'expected a checkpoint with propose pending').toBeTruthy();
    forkCheckpointId = forkable.checkpoint_id;

    const single = await api.get(
      `${API_BASE}/runs/${runId}/checkpoints/${forkCheckpointId}`,
    );
    expect(single.ok()).toBeTruthy();
    const detail = await single.json();
    expect(detail.state).toHaveProperty('iteration');
  });

  test('forking a checkpoint creates a branch visible in the trajectory', async () => {
    const res = await api.post(`${API_BASE}/runs/${runId}/fork`, {
      data: {
        parent_checkpoint_id: forkCheckpointId,
        mods: { proposer_prior: 'e2e branch prior', budget_remaining: 1 },
        name: 'e2e-branch',
      },
    });
    expect(res.status()).toBe(202);
    const branch = await res.json();
    expect(branch.thread_id).toContain('.fork.');
    expect(branch.branch_id).toBeTruthy();

    // The branch is in the trajectory, attached to the checkpoint we forked.
    await expect
      .poll(
        async () => {
          const t = await api.get(`${API_BASE}/runs/${runId}/trajectory`);
          const { trajectory } = await t.json();
          return trajectory.threads.map((x: { thread_id: string }) => x.thread_id);
        },
        { timeout: 30_000 },
      )
      .toContain(branch.thread_id);

    const t = await api.get(`${API_BASE}/runs/${runId}/trajectory`);
    const { trajectory } = await t.json();
    const edge = trajectory.edges.find(
      (e: { target: string }) => e.target === branch.thread_id,
    );
    expect(edge.source).toBe(runId);
    expect(edge.parent_checkpoint_id).toBe(forkCheckpointId);

    // The branch's state modification actually reached the new thread.
    await expect
      .poll(
        async () => {
          const r = await api.get(`${API_BASE}/runs/${runId}`);
          const body = await r.json();
          return (body.summary_rows as Array<{ thread_id: string }>).filter(
            row => row.thread_id === branch.thread_id,
          ).length;
        },
        { timeout: 60_000 },
      )
      .toBeGreaterThan(0);
  });

  test('branch artifacts stay separate from the root branch', async () => {
    const res = await api.get(`${API_BASE}/runs/${runId}`);
    const body = await res.json();
    const rows = body.summary_rows as Array<{ candidate: string; thread_id: string }>;

    const threads = new Set(rows.map(r => r.thread_id));
    expect(threads.size).toBeGreaterThan(1);

    // No candidate name is claimed by two branches.
    const names = rows.map(r => r.candidate);
    expect(new Set(names).size).toBe(names.length);

    // Each branch keeps its own frontier.
    expect(Object.keys(body.branch_frontiers).length).toBe(threads.size);
  });

  test('candidate diff and test output come from recorded artifacts', async () => {
    const res = await api.get(`${API_BASE}/runs/${runId}`);
    const body = await res.json();
    const rows = body.summary_rows as Array<{ candidate: string }>;
    const candidate = rows[rows.length - 1].candidate;

    const diff = await api.get(
      `${API_BASE}/runs/${runId}/candidates/${encodeURIComponent(candidate)}/diff`,
    );
    expect(diff.ok()).toBeTruthy();
    const diffBody = await diff.json();
    expect(diffBody.candidate).toBe(candidate);
    expect(diffBody.diff.length).toBeGreaterThan(0);
    expect(diffBody.thread_id).toBeTruthy();

    const output = await api.get(
      `${API_BASE}/runs/${runId}/candidates/${encodeURIComponent(candidate)}/test-output`,
    );
    expect(output.ok()).toBeTruthy();
    const outputBody = await output.json();
    expect(outputBody.output).toContain('metrics_source');
    // A mock run must not advertise a fabricated cost.
    expect(outputBody.output).not.toContain('total_cost_usd: 0.0\n');
  });

  test('an unknown candidate 404s instead of inventing a diff', async () => {
    const res = await api.get(
      `${API_BASE}/runs/${runId}/candidates/does-not-exist/diff`,
    );
    expect(res.status()).toBe(404);
  });

  test('memory endpoint responds with a usable shape', async () => {
    const res = await api.get(`${API_BASE}/memory/coding-agent`);
    expect(res.ok()).toBeTruthy();
    const body = await res.json();
    expect(Array.isArray(body.entries)).toBeTruthy();
    expect(body).toHaveProperty('implemented');
  });

  // ── browser against the same live run ───────────────────────────────

  test('the dashboard renders the live run reconstructed from the backend', async ({
    page,
  }) => {
    await page.goto(`/runs/${runId}`);

    const nodes = page.getByTestId('trajectory-node');
    await expect(nodes.first()).toBeVisible({ timeout: 20_000 });
    // baseline + both branches' candidates, not one collapsed line.
    await expect
      .poll(async () => nodes.count(), { timeout: 20_000 })
      .toBeGreaterThanOrEqual(3);

    await expect(page.getByTestId('metrics-provenance')).toHaveText(
      /Mock data/i,
      { timeout: 10_000 },
    );
    await expect(page.getByTestId('branch-count')).toContainText('branches');
  });

  test('reloading rebuilds run state from the backend, not from memory', async ({
    page,
  }) => {
    await page.goto(`/runs/${runId}`);
    await expect(page.getByTestId('trajectory-node').first()).toBeVisible({
      timeout: 20_000,
    });
    const before = await page.getByTestId('trajectory-node').count();

    await page.reload();
    await expect(page.getByTestId('trajectory-node').first()).toBeVisible({
      timeout: 20_000,
    });
    await expect
      .poll(async () => page.getByTestId('trajectory-node').count(), {
        timeout: 20_000,
      })
      .toBe(before);
  });

  test('the fork modal opens on a live checkpoint', async ({ page }) => {
    await page.goto(`/runs/${runId}`);
    const node = page.getByTestId('trajectory-node').first();
    await expect(node).toBeVisible({ timeout: 20_000 });

    await node.click({ button: 'right' });
    await expect(page.locator('text=⑂ Create Fork')).toBeVisible({
      timeout: 10_000,
    });
  });

  test('an unknown run shows an error state rather than fixture data', async ({
    page,
  }) => {
    await page.goto('/runs/definitely-not-a-real-run');
    await expect(page.getByTestId('status-error')).toBeVisible({
      timeout: 15_000,
    });
  });
});
