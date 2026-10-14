// EventSource wrapper for the multiplexed SSE channel that the backend
// exposes at GET /runs/{run_id}/stream. The 11 event types are the
// closed allowlist from streaming.py; backend rejects unknowns with a
// 500 so dashboard typings can stay tight.

import { fetchRunInfo, fetchTrajectory, isForkThread, streamUrl } from "./api";

export type StreamingEventType =
  | "state-update"
  | "checkpoint-written"
  | "candidate-created"
  | "validate-result"
  | "eval-result"
  | "frontier-updated"
  | "iteration-complete"
  | "fork-created"
  | "branch-cancelled"
  | "memory-pattern-stored"
  | "error";

const STREAMING_EVENT_TYPES: readonly StreamingEventType[] = [
  "state-update",
  "checkpoint-written",
  "candidate-created",
  "validate-result",
  "eval-result",
  "frontier-updated",
  "iteration-complete",
  "fork-created",
  "branch-cancelled",
  "memory-pattern-stored",
  "error",
];

export type StreamingEvent<T = Record<string, unknown>> = {
  type: StreamingEventType;
  id: string;
  data: T;
};

export type StreamHandle = {
  close: () => void;
};

type StreamHandlers = Partial<Record<StreamingEventType, (e: StreamingEvent) => void>> & {
  onOpen?: () => void;
  /** The connection failed. `closed` means the browser will not retry. */
  onError?: (e: Event, closed: boolean) => void;
};

/** A dispatched SSE message carries string data; a transport error does not. */
function isMessageEvent(e: Event): e is MessageEvent<string> {
  return typeof (e as MessageEvent).data === "string";
}

export function subscribeToRun(runId: string, handlers: StreamHandlers): StreamHandle {
  const source = new EventSource(streamUrl(runId));
  if (handlers.onOpen) source.addEventListener("open", handlers.onOpen);

  const deliver = (type: StreamingEventType, evt: MessageEvent<string>) => {
    const handler = handlers[type];
    if (!handler) return;
    let data: unknown;
    try {
      data = JSON.parse(evt.data);
    } catch {
      return; // malformed payload — drop
    }
    if (!data || typeof data !== "object") return;
    handler({ type, id: evt.lastEventId, data: data as Record<string, unknown> });
  };

  // On an EventSource "error" is two different things: the transport
  // failing, and a backend event that happens to be named `error`.
  // Treating both as the first showed "Disconnected" on a healthy stream
  // the moment a run reported an error. Only a real message has data.
  source.addEventListener("error", (raw) => {
    if (isMessageEvent(raw)) {
      deliver("error", raw);
      return;
    }
    handlers.onError?.(raw, source.readyState === EventSource.CLOSED);
  });

  for (const type of STREAMING_EVENT_TYPES) {
    if (type === "error" || !handlers[type]) continue;
    source.addEventListener(type, (raw) => {
      if (isMessageEvent(raw)) deliver(type, raw);
    });
  }
  return { close: () => source.close() };
}

// ── Convenience wrappers used by the dashboard page ──

import type { Dispatch } from "react";
import { demoFixtureState } from "./state";
import { RUN_LEVEL } from "./types";
import type {
  CandidateStatus,
  DashboardAction,
  ForkEvent,
  IterationChapter,
  LogEntry,
  RunSummary,
  TreeNode,
} from "./types";

type RunEventData = {
  candidate?: string;
  checkpoint_id?: string | number;
  node?: string | number;
  valid?: boolean;
  error?: string;
  iteration?: string | number;
  status?: string | number;
  accepted?: boolean;
  thread_id?: string | number;
  key?: string | number;
  message?: string | number;
  run?: RunSummary;
  tree?: TreeNode[];
  log?: LogEntry;
  import_path?: string;
  parent?: string | null;
  accuracy?: number;
  scores?: { accuracy?: number; per_task?: TreeNode["scores"]["per_task"] };
  per_task?: TreeNode["scores"]["per_task"];
  frontier?: string[];
  best_candidate?: string | null;
  delta?: number | null;
  score_delta?: number | null;
  hypothesis?: string;
  axis?: string | null;
  parent_thread_id?: string;
  parent_checkpoint_id?: string;
  parent_candidate_name?: string | null;
  mods_summary?: { keys?: string[]; count?: number };
};

let fallbackSeq = 0;

/** The SSE id when the backend sent one, else a locally unique one. */
function eventId(e: StreamingEvent): string {
  if (e.id) return e.id;
  fallbackSeq += 1;
  return `${e.type}-${Date.now()}-${fallbackSeq}`;
}

function eventLogEntry(
  e: StreamingEvent,
  text: string,
  tag: LogEntry["tag"],
  opts: { id?: string; candidateName?: string } = {},
): LogEntry {
  const data = e.data as RunEventData;
  const details = JSON.stringify(e.data, null, 2);
  return {
    id: opts.id ?? eventId(e),
    timestamp: new Date().toISOString(),
    tag,
    text,
    candidateName:
      opts.candidateName ?? (typeof data.candidate === "string" ? data.candidate : RUN_LEVEL),
    threadId: valueLabel(data.thread_id) ?? undefined,
    iteration: typeof data.iteration === "number" ? data.iteration : undefined,
    details,
    expandedContent: details,
    expandable: true,
  };
}

function valueLabel(value: unknown): string | null {
  return typeof value === "string" || typeof value === "number" ? String(value) : null;
}

function axisOf(value: unknown): TreeNode["axis"] {
  return value === "exploration" || value === "exploitation" ? value : undefined;
}

function signed(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}

/** The verdict a frontier-updated event reports for its own candidate. */
function verdictOf(data: RunEventData): CandidateStatus | null {
  if (data.status === "smoke_failed") return "smoke_failed";
  if (data.accepted === true) return "accepted";
  if (data.accepted === false) return "rejected";
  if (data.status === "rejected") return "rejected";
  if (data.status === "best" || data.status === "accepted") return "accepted";
  return null;
}

/** Map iteration-complete's status onto a chapter verdict, or null if unknown. */
function chapterStatusFromEvent(status: string | null): IterationChapter["status"] | null {
  if (!status) return null;
  const normalized = status.toLowerCase();
  if (normalized === "smoke_failed") return "smoke_failed";
  if (["improved", "accepted", "best"].includes(normalized)) return "accepted";
  if (["no_improvement", "rejected", "regressed", "failed"].includes(normalized)) {
    return "rejected";
  }
  return null;
}

function forkEventFromEvent(data: RunEventData): ForkEvent {
  const threadId = valueLabel(data.thread_id) ?? undefined;
  const keys = data.mods_summary?.keys ?? [];
  return {
    timestamp: new Date().toISOString(),
    parentCandidate: valueLabel(data.parent_thread_id) ?? "run",
    checkpointId: valueLabel(data.parent_checkpoint_id) ?? "checkpoint",
    prior: keys.length > 0 ? `state changed: ${keys.join(", ")}` : "no state changes",
    branchId: threadId?.split(".fork.").at(-1) ?? "branch",
    rationale: "Fork created on the backend",
    threadId,
    origin: "stream",
  };
}

/** Pull the authoritative branch tree from the backend. */
export async function refreshBranches(
  runId: string,
  dispatch: Dispatch<DashboardAction>,
): Promise<void> {
  try {
    dispatch({ type: "SET_BRANCHES", payload: await fetchTrajectory(runId) });
  } catch {
    // A run with no trajectory endpoint yet simply has no branches to
    // show; that is an empty state, not an error banner. A 401 is
    // reported through api.onUnauthorized.
  }
}

export function startSSE(
  runId: string,
  dispatch: Dispatch<DashboardAction>,
): () => void {
  const handle = subscribeToRun(runId, {
    onOpen: () => dispatch({ type: "SET_SSE_CONNECTED", payload: true }),
    onError: (_e, closed) => {
      dispatch({ type: "SET_SSE_CONNECTED", payload: false });
      if (closed) {
        // The browser gives up only when the server refused the stream.
        // EventSource cannot see why, so ask the REST API: a 401 there
        // is reported through api.onUnauthorized.
        dispatch({ type: "SET_ERROR", payload: "event stream refused by the backend" });
        void fetchRunInfo(runId).catch(() => undefined);
      }
    },
    "state-update": (e) => {
      const data = e.data as RunEventData;
      if (data.run) dispatch({ type: "SET_RUN", payload: data.run });
    },
    "checkpoint-written": (e) => {
      const data = e.data as RunEventData;
      const checkpoint = valueLabel(data.checkpoint_id);
      const node = valueLabel(data.node) ?? "graph";
      if (checkpoint) dispatch({ type: "SET_CHECKPOINT", payload: checkpoint });
      dispatch({
        type: "ADD_LOG_ENTRY",
        payload: eventLogEntry(
          e,
          `checkpoint ${checkpoint ? checkpoint.slice(0, 8) : "unknown"} written after ${node}`,
          "checkpoint",
        ),
      });
    },
    "candidate-created": (e) => {
      const data = e.data as RunEventData;
      if (typeof data.candidate !== "string") return;
      const candidate = data.candidate;
      const threadId = valueLabel(data.thread_id) ?? undefined;
      const iteration = typeof data.iteration === "number" ? data.iteration : 0;
      const parent = data.parent_candidate_name ?? data.parent ?? null;
      const axis = axisOf(data.axis);
      // Lineage only. Status, score and delta are not this event's to set.
      dispatch({
        type: "ADD_TREE_NODE",
        payload: {
          candidate,
          parent_candidate_name: parent,
          iteration,
          hypothesis: data.hypothesis || undefined,
          axis,
          isForkBranch: isForkThread(threadId),
          threadId,
        },
      });
      dispatch({
        type: "ADD_ITERATION",
        payload: {
          iteration,
          candidateName: candidate,
          status: "running",
          phases: { propose: true, validate: false, benchmark: false, frontier: false },
          hypothesis: data.hypothesis || "no hypothesis recorded",
          isForkBranch: isForkThread(threadId),
          threadId,
        },
      });
      dispatch({
        type: "ADD_LOG_ENTRY",
        payload: eventLogEntry(
          e,
          `${candidate} proposed${parent ? ` from ${parent}` : ""}${axis ? ` (${axis})` : ""}`,
          "plan",
        ),
      });
    },
    "validate-result": (e) => {
      const data = e.data as RunEventData;
      const candidate = valueLabel(data.candidate) ?? "candidate";
      const valid = Boolean(data.valid);
      if (!valid && typeof data.candidate === "string") {
        // It never loaded, so it has no score: not 0%, not rejected.
        dispatch({
          type: "ADD_TREE_NODE",
          payload: {
            candidate: data.candidate,
            threadId: valueLabel(data.thread_id) ?? undefined,
            status: "smoke_failed",
            delta: null,
          },
        });
      }
      dispatch({
        type: "UPDATE_ITERATION",
        payload: {
          candidateName: candidate,
          phases: { propose: true, validate: true },
          status: valid ? undefined : "smoke_failed",
        },
      });
      dispatch({
        type: "ADD_LOG_ENTRY",
        payload: eventLogEntry(
          e,
          valid
            ? `${candidate} validated`
            : `${candidate} failed validation${data.error ? `: ${data.error}` : ""}`,
          valid ? "verify" : "fail",
        ),
      });
    },
    "eval-result": (e) => {
      const data = e.data as RunEventData;
      if (typeof data.candidate !== "string") return;
      const candidate = data.candidate;
      const threadId = valueLabel(data.thread_id) ?? undefined;
      const accuracy =
        typeof data.accuracy === "number" ? data.accuracy : data.scores?.accuracy;
      const perTask = data.per_task ?? data.scores?.per_task;
      // Scores only: whether this candidate is accepted is decided by
      // update_frontier and arrives with frontier-updated.
      dispatch({
        type: "ADD_TREE_NODE",
        payload: {
          candidate,
          parent_candidate_name: data.parent_candidate_name ?? data.parent ?? undefined,
          iteration: typeof data.iteration === "number" ? data.iteration : undefined,
          scores:
            typeof accuracy === "number" ? { accuracy, per_task: perTask } : undefined,
          isForkBranch: isForkThread(threadId),
          threadId,
        },
      });
      dispatch({
        type: "UPDATE_ITERATION",
        payload: {
          candidateName: candidate,
          phases: { propose: true, validate: true, benchmark: true },
        },
      });

      // One event, several log lines: each needs its own id, or every
      // line replaces the one before it.
      const baseId = eventId(e);
      const tasks = Object.entries(perTask ?? {}).sort(([a], [b]) => a.localeCompare(b));
      if (typeof accuracy === "number") {
        dispatch({
          type: "ADD_LOG_ENTRY",
          payload: eventLogEntry(
            e,
            `${candidate} benchmarked: accuracy ${accuracy.toFixed(2)} over ${tasks.length} task(s)`,
            "score",
            { id: `${baseId}:summary` },
          ),
        });
      }
      for (const [taskKey, taskValue] of tasks.slice(0, 6)) {
        const taskLabel = taskKey.replace(/^task-\d+-/, "");
        dispatch({
          type: "ADD_LOG_ENTRY",
          payload: eventLogEntry(
            e,
            `task ${taskLabel}: ${(taskValue.pass_rate * 100).toFixed(0)}% pass rate`,
            "score",
            { id: `${baseId}:task:${taskKey}` },
          ),
        });
      }
    },
    "iteration-complete": (e) => {
      const data = e.data as RunEventData;
      if (data.log) {
        dispatch({ type: "ADD_LOG_ENTRY", payload: data.log });
        return;
      }
      const iterationLabel = valueLabel(data.iteration) ?? "?";
      const statusLabel = valueLabel(data.status);
      const status = chapterStatusFromEvent(statusLabel);
      const threadId = valueLabel(data.thread_id) ?? undefined;
      if (typeof data.candidate === "string") {
        dispatch({
          type: "UPDATE_ITERATION",
          payload: {
            candidateName: data.candidate,
            status: status ?? undefined,
            phases: { propose: true, validate: true, benchmark: true, frontier: true },
          },
        });
      } else if (typeof data.iteration === "number") {
        // No candidate named: finish that iteration's chapter on that
        // thread. Inventing an "_iter_N" chapter left the real one running.
        dispatch({
          type: "COMPLETE_ITERATION",
          payload: { iteration: data.iteration, threadId, status },
        });
      }
      dispatch({
        type: "ADD_LOG_ENTRY",
        payload: eventLogEntry(
          e,
          `iteration ${iterationLabel} complete: ${(statusLabel ?? "no status").replace(/_/g, " ")}`,
          "score",
        ),
      });
    },
    "fork-created": (e) => {
      const data = e.data as RunEventData;
      dispatch({ type: "ADD_FORK_EVENT", payload: forkEventFromEvent(data) });
      dispatch({
        type: "ADD_LOG_ENTRY",
        payload: eventLogEntry(
          e,
          `branch ${valueLabel(data.thread_id) ?? "?"} forked from ` +
            `${valueLabel(data.parent_thread_id) ?? "run"}@` +
            `${(valueLabel(data.parent_checkpoint_id) ?? "").slice(0, 8)}`,
          "fork",
        ),
      });
      // Re-read the authoritative branch tree rather than guessing it
      // from the event payload.
      void refreshBranches(runId, dispatch);
    },
    "branch-cancelled": (e) => {
      const data = e.data as RunEventData;
      const thread = valueLabel(data.thread_id) ?? "branch";
      dispatch({
        type: "ADD_LOG_ENTRY",
        payload: eventLogEntry(e, `${thread} cancelled`, "fork"),
      });
      void refreshBranches(runId, dispatch);
    },
    "memory-pattern-stored": (e) => {
      const data = e.data as RunEventData;
      const key = valueLabel(data.key) ?? "pattern";
      const delta = typeof data.score_delta === "number" ? ` (${signed(data.score_delta)})` : "";
      // The backend keys a stored pattern by the candidate it came from.
      dispatch({
        type: "ADD_LOG_ENTRY",
        payload: eventLogEntry(e, `stored memory pattern from ${key}${delta}`, "memory", {
          candidateName: typeof data.key === "string" ? data.key : RUN_LEVEL,
        }),
      });
    },
    "frontier-updated": (e) => {
      const data = e.data as RunEventData;
      if (data.tree) dispatch({ type: "SET_TREE", payload: data.tree });
      const candidate = typeof data.candidate === "string" ? data.candidate : undefined;
      const threadId = valueLabel(data.thread_id) ?? undefined;
      const verdict = verdictOf(data);
      const delta = verdict === "smoke_failed" ? null : data.delta ?? null;

      if (candidate && verdict !== "smoke_failed" && typeof data.scores?.accuracy === "number") {
        dispatch({
          type: "ADD_TREE_NODE",
          payload: {
            candidate,
            threadId,
            scores: { accuracy: data.scores.accuracy, per_task: data.scores.per_task },
          },
        });
      }
      if (data.frontier || data.best_candidate || candidate) {
        dispatch({
          type: "APPLY_FRONTIER_UPDATE",
          payload: {
            threadId,
            candidate,
            accepted: typeof data.accepted === "boolean" ? data.accepted : undefined,
            status: valueLabel(data.status) ?? undefined,
            frontier: data.frontier ?? [],
            bestCandidate: data.best_candidate ?? null,
            delta,
          },
        });
      }
      if (candidate) {
        dispatch({
          type: "UPDATE_ITERATION",
          payload: {
            candidateName: candidate,
            status: verdict ?? undefined,
            phases: { propose: true, validate: true, benchmark: true, frontier: true },
          },
        });
        const accuracy = data.scores?.accuracy;
        const text =
          verdict === "smoke_failed"
            ? `${candidate} smoke failed: never loaded, so it has no score`
            : `${candidate} ${verdict ?? "evaluated"}` +
              (typeof accuracy === "number" ? `: accuracy ${accuracy.toFixed(2)}` : "") +
              (typeof delta === "number" ? ` (${signed(delta)})` : "");
        dispatch({
          type: "ADD_LOG_ENTRY",
          payload: eventLogEntry(e, text, verdict === "smoke_failed" ? "fail" : "score"),
        });
      }
    },
    "error": (e) => {
      const data = e.data as RunEventData;
      const message = valueLabel(data.message) ?? "unknown error";
      const where = valueLabel(data.node);
      dispatch({
        type: "ADD_LOG_ENTRY",
        payload: eventLogEntry(e, `${where ? `${where}: ` : ""}${message}`, "fail"),
      });
      dispatch({ type: "SET_ERROR", payload: message });
    },
  });
  return handle.close;
}

export function startMockSSE(
  dispatch: Dispatch<DashboardAction>,
): () => void {
  const MOCK_SPEED_MULTIPLIER = 1.8;
  const timers: ReturnType<typeof setTimeout>[] = [];
  const schedule = (delayMs: number, action: () => void) => {
    const timer = setTimeout(action, Math.round(delayMs * MOCK_SPEED_MULTIPLIER));
    timers.push(timer);
  };

  dispatch({ type: "SET_MODE", payload: "mock" });
  if (demoFixtureState.run) {
    dispatch({
      type: "SET_RUN",
      payload: { ...demoFixtureState.run, isMock: true },
    });
  }
  dispatch({ type: "SET_TREE", payload: [] });
  dispatch({ type: "SET_ITERATIONS", payload: [] });
  dispatch({ type: "SET_LOG_ENTRIES", payload: [] });
  dispatch({ type: "SET_SSE_CONNECTED", payload: true });

  const tree = demoFixtureState.tree ?? [];
  const iterations = demoFixtureState.iterations ?? [];
  const logs = demoFixtureState.logEntries ?? [];
  const forks = demoFixtureState.forkEvents ?? [];

  tree.forEach((node, idx) => {
    schedule(700 + idx * 950, () => {
      dispatch({ type: "ADD_TREE_NODE", payload: node });
    });
  });

  iterations.forEach((iteration, idx) => {
    schedule(1500 + idx * 1300, () => {
      dispatch({ type: "ADD_ITERATION", payload: iteration });
    });
  });

  logs.forEach((entry, idx) => {
    schedule(1200 + idx * 800, () => {
      dispatch({ type: "ADD_LOG_ENTRY", payload: entry });
    });
  });

  forks.forEach((fork, idx) => {
    schedule(2200 + idx * 1700, () => {
      dispatch({ type: "ADD_FORK_EVENT", payload: fork });
    });
  });

  if (demoFixtureState.selectedNode) {
    const selectedNodeDelay = Math.max(4500, 1000 + tree.length * 1200);
    schedule(selectedNodeDelay, () => {
      dispatch({ type: "SELECT_NODE", payload: demoFixtureState.selectedNode ?? null });
    });
  }

  return () => {
    for (const timer of timers) clearTimeout(timer);
    dispatch({ type: "SET_SSE_CONNECTED", payload: false });
  };
}
