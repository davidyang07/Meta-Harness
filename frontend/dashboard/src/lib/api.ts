import type {
  BranchNode,
  CandidateStatus,
  MemoryEntry,
  RunSummary,
  Scores,
  TreeNode,
} from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

// Must be read as a literal `process.env.NEXT_PUBLIC_…` so Next inlines it
// into the client bundle. It is a shared secret for a backend started
// with META_HARNESS_API_TOKEN; it is visible to anyone who can load the
// dashboard, so it gates the API, not the dashboard itself.
const API_TOKEN = (process.env.NEXT_PUBLIC_META_HARNESS_API_TOKEN ?? "").trim();

// ── Authentication ──

export const AUTH_ERROR_MESSAGE = API_TOKEN
  ? "backend rejected the API token (401): NEXT_PUBLIC_META_HARNESS_API_TOKEN " +
    "does not match the server's META_HARNESS_API_TOKEN"
  : "backend requires an API token (401): set NEXT_PUBLIC_META_HARNESS_API_TOKEN " +
    "to the server's META_HARNESS_API_TOKEN and restart the dashboard";

/** The backend answered 401: the token is missing or wrong. */
export class ApiUnauthorizedError extends Error {
  constructor() {
    super(AUTH_ERROR_MESSAGE);
    this.name = "ApiUnauthorizedError";
  }
}

const unauthorizedListeners = new Set<(message: string) => void>();

/**
 * Be told about every 401, including those from calls whose callers
 * deliberately degrade to an empty state (diffs, memory). Without this
 * a wrong token reads as "this run has no data".
 */
export function onUnauthorized(listener: (message: string) => void): () => void {
  unauthorizedListeners.add(listener);
  return () => {
    unauthorizedListeners.delete(listener);
  };
}

async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  if (API_TOKEN) headers.set("Authorization", `Bearer ${API_TOKEN}`);
  const res = await fetch(`${BASE_URL}${path}`, { ...init, headers });
  if (res.status === 401) {
    for (const listener of unauthorizedListeners) listener(AUTH_ERROR_MESSAGE);
    throw new ApiUnauthorizedError();
  }
  return res;
}

/**
 * URL of a run's SSE stream. EventSource cannot send headers, so the
 * token travels as a query parameter, which the backend accepts on its
 * SSE routes only.
 */
export function streamUrl(runId: string): string {
  const url = `${BASE_URL}/runs/${encodeURIComponent(runId)}/stream`;
  return API_TOKEN ? `${url}?token=${encodeURIComponent(API_TOKEN)}` : url;
}

// ── Availability check ──

export async function isBackendAvailable(): Promise<boolean> {
  try {
    // /health is the one route that never requires the token.
    const res = await fetch(`${BASE_URL}/health`, { signal: AbortSignal.timeout(2000) });
    return res.ok;
  } catch {
    return false;
  }
}

// ── Run list (home page) ──

export type RunListItem = {
  run_id: string;
  status: string;
  best_score: number | null;
  iteration: number;
};

export type CreateRunRequest = {
  run_name?: string;
  proposer?: "claude" | "mock";
  mock_bench?: boolean;
  budget?: number;
  fresh?: boolean;
  trials?: number;
  workers?: number;
};

export type CreateRunResponse = {
  run_id: string;
  thread_id: string;
  status: string;
  current_iteration: number;
};

export async function listRuns(): Promise<RunListItem[]> {
  const res = await apiFetch("/runs");
  if (!res.ok) return [];
  const data = await res.json();
  return Array.isArray(data) ? data : data.runs ?? [];
}

export async function createRun(payload: CreateRunRequest): Promise<CreateRunResponse> {
  const res = await apiFetch("/runs", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`failed to create run (${res.status})`);
  }
  return (await res.json()) as CreateRunResponse;
}

// ── Run detail ──

export async function fetchRunInfo(runId: string): Promise<unknown> {
  const res = await apiFetch(`/runs/${encodeURIComponent(runId)}`);
  if (!res.ok) throw new Error(`run ${runId} not found`);
  return res.json();
}

type RunDetail = {
  run_id?: string;
  runId?: string;
  thread_id?: string;
  threadId?: string;
  branches?: number;
  checkpoint_id?: string | null;
  checkpointId?: string | null;
  best_score?: number | null;
  bestScore?: number | null;
  status?: string;
  error?: string;
  current_iteration?: number;
  iteration?: number;
  summary_rows?: EvolutionRow[];
  frontier_val?: FrontierVal | null;
  branch_frontiers?: Record<string, FrontierVal>;
  metrics_source?: string;
  manifest?: {
    mock_proposer?: boolean;
    mock_bench?: boolean;
    metrics_source?: string;
    error?: string;
  };
};

type EvolutionRow = {
  candidate?: string;
  candidate_name?: string;
  parent_candidate_name?: string | null;
  iteration?: number;
  /** Absent on rows written before the backend recorded a verdict. */
  status?: string;
  accepted?: boolean;
  reason?: string;
  scores?: Scores;
  hypothesis?: string;
  axis?: "exploration" | "exploitation";
  delta?: number | null;
  is_fork_branch?: boolean;
  thread_id?: string;
  checkpoint_id?: string;
  label?: string;
  metrics_source?: string;
};

type FrontierVal = {
  _best?: {
    name?: string;
  };
  _pareto_names?: string[];
};

function asRunDetail(value: unknown): RunDetail {
  return value && typeof value === "object" ? (value as RunDetail) : {};
}

export async function getRunDetail(runId: string): Promise<RunDetail> {
  return asRunDetail(await fetchRunInfo(runId));
}

function metricsSourceOf(detail: RunDetail): RunSummary["metricsSource"] {
  const raw = detail.metrics_source ?? detail.manifest?.metrics_source;
  if (raw === "measured" || raw === "mock") return raw;
  // Fall back to the manifest flags for runs written before the field
  // existed, rather than guessing "measured".
  if (detail.manifest?.mock_bench || detail.manifest?.mock_proposer) return "mock";
  return "unknown";
}

export function toRunInfo(value: RunDetail): RunSummary {
  const detail = asRunDetail(value);
  const metricsSource = metricsSourceOf(detail);
  return {
    runId: detail.run_id ?? detail.runId ?? "",
    threadId: detail.thread_id ?? detail.threadId ?? detail.run_id ?? "",
    branches: detail.branches ?? 0,
    checkpointId: detail.checkpoint_id ?? detail.checkpointId ?? null,
    bestScore: detail.best_score ?? detail.bestScore ?? null,
    status: detail.status ?? "unknown",
    iteration: detail.current_iteration ?? detail.iteration ?? 0,
    isMock: metricsSource === "mock",
    metricsSource,
  };
}

/** Why a failed run failed, as the backend recorded it. */
export function runErrorOf(value: RunDetail): string | null {
  const detail = asRunDetail(value);
  return detail.error ?? detail.manifest?.error ?? null;
}

/** A thread id of the form "<run>.fork.<id>" identifies a forked branch. */
export function isForkThread(threadId: string | undefined): boolean {
  return Boolean(threadId && threadId.includes(".fork."));
}

const KNOWN_STATUSES: ReadonlySet<string> = new Set<CandidateStatus>([
  "seed",
  "accepted",
  "rejected",
  "best",
  "fork",
  "smoke_failed",
]);

/**
 * The backend's verdict on one evolution row, or null when the row
 * predates verdicts. Never guessed from the score: a candidate that
 * scored well can still have been rejected against a better parent.
 */
export function statusFromRow(row: EvolutionRow): CandidateStatus | null {
  if (row.status === "smoke_failed" || row.reason === "smoke_failed") return "smoke_failed";
  if (row.status && KNOWN_STATUSES.has(row.status)) return row.status as CandidateStatus;
  if (row.accepted === true) return "accepted";
  if (row.accepted === false) return "rejected";
  return null;
}

export function toTreeNodes(rows: EvolutionRow[]): TreeNode[] {
  return rows.map((r) => {
    const candidate = r.candidate ?? r.candidate_name ?? "";
    const threadId = r.thread_id;
    const verdict = statusFromRow(r);
    const parent = r.parent_candidate_name ?? null;
    // The root of the search is the measured baseline, not a proposal
    // that was accepted or rejected.
    const status: CandidateStatus =
      verdict === "smoke_failed" ? verdict : parent === null ? "seed" : verdict ?? "seed";
    return {
      candidate,
      parent_candidate_name: parent,
      iteration: r.iteration ?? 0,
      status,
      scores: r.scores ?? { accuracy: 0 },
      hypothesis: r.hypothesis,
      axis: r.axis,
      // A candidate that never loaded has no score to compare.
      delta: status === "smoke_failed" ? null : r.delta ?? null,
      // Branch identity comes from the thread id the backend stamped on
      // the row, not from a separate flag that can drift out of sync.
      isForkBranch: r.is_fork_branch ?? isForkThread(threadId),
      threadId,
      checkpointId: r.checkpoint_id,
    };
  });
}

export function toTreeNodesFromRunDetail(detail: RunDetail): TreeNode[] {
  const rows = detail.summary_rows ?? [];
  const nodes = toTreeNodes(rows);
  const rootThread = detail.thread_id ?? detail.run_id ?? "";

  // Every branch keeps its own frontier, so each has its own best. A
  // fork's frontier can name a candidate it inherited from main; that
  // says nothing about main, so a best only marks a node of its own thread.
  const frontiers = new Map<string, FrontierVal>(Object.entries(detail.branch_frontiers ?? {}));
  if (detail.frontier_val && !frontiers.has(rootThread)) {
    frontiers.set(rootThread, detail.frontier_val);
  }

  return nodes.map((node, idx) => {
    const frontier = frontiers.get(node.threadId ?? rootThread);
    if (!frontier) return node;
    const decidedAgainst = node.status === "rejected" || node.status === "smoke_failed";
    if (frontier._best?.name === node.candidate && !decidedAgainst) {
      return { ...node, status: "best" };
    }
    // Rows without a verdict: frontier membership is the best signal left.
    if (
      statusFromRow(rows[idx]) === null &&
      node.parent_candidate_name !== null &&
      (frontier._pareto_names ?? []).includes(node.candidate)
    ) {
      return { ...node, status: "accepted" };
    }
    return node;
  });
}

// ── Checkpoints ──

/**
 * A thread's checkpoint history. Omitting `threadId` (or passing the
 * run's own id) reads the root thread; a fork's id reads that fork.
 */
export async function fetchCheckpoints(runId: string, threadId?: string): Promise<unknown> {
  const query =
    threadId && threadId !== runId ? `?thread_id=${encodeURIComponent(threadId)}` : "";
  const res = await apiFetch(`/runs/${encodeURIComponent(runId)}/checkpoints${query}`);
  if (!res.ok) throw new Error(`checkpoints for ${threadId ?? runId} not found (${res.status})`);
  return res.json();
}

type CheckpointRow = {
  checkpoint_id?: string;
  thread_id?: string;
  iteration?: number;
  values_summary?: {
    best_candidate?: string;
    iteration?: number;
  };
};

export async function fetchCheckpointCandidateMap(runId: string): Promise<Map<string, string>> {
  const raw = await fetchCheckpoints(runId);
  const rows = (raw as { checkpoints?: unknown }).checkpoints;
  const map = new Map<string, string>();
  if (!Array.isArray(rows)) return map;
  for (const item of rows.toReversed()) {
    const row = item as CheckpointRow;
    const candidate = row.values_summary?.best_candidate;
    if (row.checkpoint_id && candidate) map.set(candidate, row.checkpoint_id);
  }
  return map;
}

export async function resolveCheckpointForNode(
  runId: string,
  node: { candidate: string; iteration: number; threadId?: string },
): Promise<string | null> {
  let raw: unknown;
  try {
    // A fork-branch node's checkpoints live on its own thread; the root
    // thread's history does not contain them.
    raw = await fetchCheckpoints(runId, node.threadId);
  } catch (error) {
    if (error instanceof ApiUnauthorizedError) throw error;
    // Some runs (especially dry-runs) may not expose checkpoint history yet.
    // Returning null lets callers surface a graceful "no checkpoint" message
    // instead of throwing an unhandled rejection from click handlers.
    return null;
  }
  const rows = (raw as { checkpoints?: unknown }).checkpoints;
  if (!Array.isArray(rows)) return null;

  const typed = rows as CheckpointRow[];

  // Prefer exact LangGraph thread lineage match.
  if (node.threadId) {
    const byThreadAndIter = typed
      .toReversed()
      .find((row) => row.thread_id === node.threadId && row.iteration === node.iteration);
    if (byThreadAndIter?.checkpoint_id) return byThreadAndIter.checkpoint_id;

    const byThread = typed.toReversed().find((row) => row.thread_id === node.threadId);
    if (byThread?.checkpoint_id) return byThread.checkpoint_id;
  }

  // Fallback for older payloads that only expose best candidate summaries.
  const byCandidate = typed.toReversed().find(
    (row) => row.values_summary?.best_candidate === node.candidate,
  );
  if (byCandidate?.checkpoint_id) return byCandidate.checkpoint_id;

  // Final fallback by iteration summary when candidate labels drift.
  const bySummaryIteration = typed
    .toReversed()
    .find((row) => row.values_summary?.iteration === node.iteration);
  if (bySummaryIteration?.checkpoint_id) return bySummaryIteration.checkpoint_id;

  return null;
}

// ── Trajectory (branch tree) ──

type TrajectoryThread = {
  thread_id?: string;
  parent_thread_id?: string | null;
  parent_checkpoint_id?: string | null;
  branch_id?: string | null;
  name?: string | null;
  status?: string;
  live?: boolean;
};

export async function fetchTrajectory(runId: string): Promise<BranchNode[]> {
  const res = await apiFetch(`/runs/${encodeURIComponent(runId)}/trajectory`);
  if (!res.ok) throw new Error(`trajectory for ${runId} not available (${res.status})`);
  const data = await res.json();
  const threads = data?.trajectory?.threads;
  if (!Array.isArray(threads)) return [];
  return threads.flatMap((raw: TrajectoryThread) =>
    raw?.thread_id
      ? [
          {
            threadId: raw.thread_id,
            parentThreadId: raw.parent_thread_id ?? null,
            parentCheckpointId: raw.parent_checkpoint_id ?? null,
            branchId: raw.branch_id ?? null,
            name: raw.name ?? null,
            status: raw.status ?? "unknown",
            live: Boolean(raw.live),
          },
        ]
      : [],
  );
}

// ── Forking ──

export async function postFork(
  runId: string,
  body: {
    parent_checkpoint_id: string;
    parent_thread_id?: string;
    mods?: Record<string, unknown>;
    name?: string;
  },
): Promise<unknown> {
  const res = await apiFetch(`/runs/${encodeURIComponent(runId)}/fork`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`fork failed for run ${runId} (${res.status})`);
  return res.json();
}

export async function forkRun(
  runId: string,
  body: {
    parent_checkpoint_id: string;
    parent_thread_id?: string;
    mods?: Record<string, unknown>;
    name?: string;
  },
): Promise<{ branch_id?: string; thread_id?: string }> {
  const result = await postFork(runId, body);
  return result as { branch_id?: string; thread_id?: string };
}

// ── Memory ──

export async function fetchMemory(namespace: string, limit = 50): Promise<unknown> {
  const res = await apiFetch(`/memory/${encodeURIComponent(namespace)}?limit=${limit}`);
  if (!res.ok) throw new Error(`memory namespace ${namespace} not found`);
  return res.json();
}

export async function listMemory(namespace: string, limit = 50): Promise<MemoryEntry[]> {
  const raw = await fetchMemory(namespace, limit);
  if (!raw || typeof raw !== "object") return [];
  const entries = (raw as { entries?: unknown }).entries;
  if (!Array.isArray(entries)) return [];
  return entries.flatMap((entry) => {
    if (!entry || typeof entry !== "object") return [];
    const value = entry as Partial<MemoryEntry>;
    return typeof value.key === "string" && typeof value.pattern === "string"
      ? [value as MemoryEntry]
      : [];
  });
}

export async function getDiff(runId: string, candidate: string): Promise<string | null> {
  const res = await apiFetch(
    `/runs/${encodeURIComponent(runId)}/candidates/${encodeURIComponent(candidate)}/diff`,
  );
  if (!res.ok) return null;
  const data = await res.json();
  return typeof data.diff === "string" && data.diff.length > 0 ? data.diff : null;
}

export async function getTestOutput(runId: string, candidate: string): Promise<string | null> {
  const res = await apiFetch(
    `/runs/${encodeURIComponent(runId)}/candidates/${encodeURIComponent(candidate)}/test-output`,
  );
  if (!res.ok) return null;
  const data = await res.json();
  return typeof data.output === "string" && data.output.length > 0 ? data.output : null;
}

export const API_BASE_URL = BASE_URL;
