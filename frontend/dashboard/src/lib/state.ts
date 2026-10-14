"use client";

import {
  createContext,
  useContext,
  useReducer,
  type Dispatch,
  type ReactNode,
} from "react";
import { createElement } from "react";
import type {
  CandidateStatus,
  DashboardAction,
  DashboardState,
  ForkEvent,
  FrontierUpdate,
  IterationChapter,
  LogEntry,
  TreeNode,
  TreeNodePatch,
} from "./types";

/**
 * Identity of a node in the search tree.
 *
 * A candidate name alone is not enough: two branches forked from the
 * same checkpoint reach the same iteration, and merging their nodes
 * makes one branch's result overwrite the other's on screen. The
 * backend qualifies candidate names per branch, and the UI keys on
 * (thread, candidate) so the two can never collapse into one node.
 */
export function nodeKey(node: { threadId?: string; candidate: string }): string {
  return `${node.threadId ?? ""}::${node.candidate}`;
}

/** Identity of a fork: its thread when known, else its branch id. */
export function forkKey(fork: Pick<ForkEvent, "threadId" | "branchId">): string {
  return fork.threadId ?? fork.branchId;
}

const initialTree: TreeNode[] = [
  {
    candidate: "baseline",
    parent_candidate_name: null,
    iteration: 0,
    status: "seed",
    scores: { accuracy: 0.62, per_task: {
      "fix-typo": { pass_rate: 0.8, trials: [true, true, true, true, false] },
      "add-function": { pass_rate: 0.6, trials: [true, true, true, false, false] },
      "refactor": { pass_rate: 0.4, trials: [true, true, false, false, false] },
      "handle-error": { pass_rate: 0.6, trials: [true, false, true, true, false] },
      "implement-spec": { pass_rate: 0.6, trials: [true, true, true, false, false] },
    }},
    hypothesis: "starting harness",
    axis: "exploration",
    delta: 0,
  },
  {
    candidate: "retry-on-schema-drift",
    parent_candidate_name: "baseline",
    iteration: 1,
    status: "accepted",
    scores: { accuracy: 0.7, per_task: {
      "fix-typo": { pass_rate: 1.0, trials: [true, true, true, true, true] },
      "add-function": { pass_rate: 0.6, trials: [true, true, true, false, false] },
      "refactor": { pass_rate: 0.4, trials: [true, true, false, false, false] },
      "handle-error": { pass_rate: 0.8, trials: [true, true, true, true, false] },
      "implement-spec": { pass_rate: 0.6, trials: [true, true, true, false, false] },
    }},
    hypothesis: "retry on schema_drift errors",
    axis: "exploitation",
    delta: 0.08,
  },
  {
    candidate: "tighter-tool-hashing",
    parent_candidate_name: "retry-on-schema-drift",
    iteration: 2,
    status: "rejected",
    scores: { accuracy: 0.66, per_task: {
      "fix-typo": { pass_rate: 1.0, trials: [true, true, true, true, true] },
      "add-function": { pass_rate: 0.6, trials: [true, true, true, false, false] },
      "refactor": { pass_rate: 0.2, trials: [true, false, false, false, false] },
      "handle-error": { pass_rate: 0.6, trials: [true, false, true, true, false] },
      "implement-spec": { pass_rate: 0.6, trials: [true, true, true, false, false] },
    }},
    hypothesis: "stricter tool-description hashing",
    axis: "exploration",
    delta: -0.04,
  },
  {
    candidate: "early-exit-on-auth",
    parent_candidate_name: "retry-on-schema-drift",
    iteration: 3,
    status: "accepted",
    scores: { accuracy: 0.74, per_task: {
      "fix-typo": { pass_rate: 1.0, trials: [true, true, true, true, true] },
      "add-function": { pass_rate: 0.6, trials: [true, true, true, false, false] },
      "refactor": { pass_rate: 0.6, trials: [true, true, true, false, false] },
      "handle-error": { pass_rate: 0.8, trials: [true, true, true, true, false] },
      "implement-spec": { pass_rate: 0.6, trials: [true, true, true, false, false] },
    }},
    hypothesis: "early-exit on auth failures",
    axis: "exploitation",
    delta: 0.04,
  },
  {
    candidate: "more-specific-descriptions",
    parent_candidate_name: "early-exit-on-auth",
    iteration: 4,
    status: "best",
    scores: { accuracy: 0.8, per_task: {
      "fix-typo": { pass_rate: 1.0, trials: [true, true, true, true, true] },
      "add-function": { pass_rate: 0.8, trials: [true, true, true, true, false] },
      "refactor": { pass_rate: 0.6, trials: [true, true, true, false, false] },
      "handle-error": { pass_rate: 0.8, trials: [true, true, true, true, false] },
      "implement-spec": { pass_rate: 0.8, trials: [true, true, true, true, false] },
    }},
    hypothesis: "more specific tool descriptions",
    axis: "exploitation",
    delta: 0.06,
  },
  {
    candidate: "rewrite-tool-descriptions-with-examples",
    parent_candidate_name: "retry-on-schema-drift",
    iteration: 2,
    status: "accepted",
    scores: { accuracy: 0.78, per_task: {
      "fix-typo": { pass_rate: 1.0, trials: [true, true, true, true, true] },
      "add-function": { pass_rate: 0.8, trials: [true, true, true, true, false] },
      "refactor": { pass_rate: 0.6, trials: [true, true, true, false, false] },
      "handle-error": { pass_rate: 0.8, trials: [true, true, false, true, true] },
      "implement-spec": { pass_rate: 0.6, trials: [true, true, true, false, false] },
    }},
    hypothesis: "rewrite tool descriptions w/ examples",
    axis: "exploration",
    delta: 0.16,
    isForkBranch: true,
    threadId: "demo.fork.abc12345",
  },
  {
    candidate: "few-shot-demos-on-descriptions",
    parent_candidate_name: "rewrite-tool-descriptions-with-examples",
    iteration: 3,
    status: "best",
    scores: { accuracy: 0.85, per_task: {
      "fix-typo": { pass_rate: 1.0, trials: [true, true, true, true, true] },
      "add-function": { pass_rate: 0.8, trials: [true, true, true, true, false] },
      "refactor": { pass_rate: 0.8, trials: [true, true, true, true, false] },
      "handle-error": { pass_rate: 0.8, trials: [true, true, true, true, false] },
      "implement-spec": { pass_rate: 0.8, trials: [true, true, true, true, false] },
    }},
    hypothesis: "add few-shot demos to descriptions",
    axis: "exploitation",
    delta: 0.07,
    isForkBranch: true,
    threadId: "demo.fork.abc12345",
  },
];

const initialIterations: IterationChapter[] = [
  { iteration: 1, candidateName: "retry-on-schema-drift", status: "accepted", phases: { propose: true, validate: true, benchmark: true, frontier: true }, hypothesis: "retry on schema_drift errors" },
  { iteration: 2, candidateName: "tighter-tool-hashing", status: "rejected", phases: { propose: true, validate: true, benchmark: true, frontier: false }, hypothesis: "stricter tool-description hashing" },
  { iteration: 3, candidateName: "early-exit-on-auth", status: "accepted", phases: { propose: true, validate: true, benchmark: true, frontier: true }, hypothesis: "early-exit on auth failures" },
  { iteration: 4, candidateName: "more-specific-descriptions", status: "best", phases: { propose: true, validate: true, benchmark: true, frontier: true }, hypothesis: "more specific tool descriptions" },
];

const initialLogEntries: LogEntry[] = [
  { id: "log-1", timestamp: "14:32:11", tag: "orient", text: "read 8 files in workspace", candidateName: "more-specific-descriptions" },
  { id: "log-2", timestamp: "14:32:14", tag: "plan", text: "submit_plan: rewrite tool descriptions to be more specific", candidateName: "more-specific-descriptions" },
  { id: "log-3", timestamp: "14:32:18", tag: "tool/read", text: "agents/early-exit-on-auth.py", candidateName: "more-specific-descriptions" },
  { id: "log-4", timestamp: "14:32:24", tag: "tool/patch", text: "applied patch to agents/more-specific-descriptions.py", candidateName: "more-specific-descriptions" },
  { id: "log-5", timestamp: "14:32:42", tag: "verify", text: "5 tests passed in 0.04s", candidateName: "more-specific-descriptions" },
  { id: "log-6", timestamp: "14:32:43", tag: "score", text: "accuracy=0.80 (+0.06) NEW BEST", candidateName: "more-specific-descriptions" },
];

const initialState: DashboardState = {
  mode: "live",
  run: null,
  tree: [],
  iterations: [],
  logEntries: [],
  forkEvents: [],
  filters: { activeFilter: "all", searchQuery: "" },
  contextTab: "chart",
  selectedNode: null,
  selectedLogLine: null,
  sseConnected: false,
  latestCheckpointId: null,
  lastError: null,
  branches: [],
};

export const demoFixtureState: Partial<DashboardState> = {
  run: {
    runId: "demo-2026-04-25",
    threadId: "demo-2026-04-25",
    branches: 2,
    checkpointId: "ckpt_4af9e21c",
    bestScore: 0.85,
    status: "running",
    iteration: 4,
  },
  tree: initialTree,
  iterations: initialIterations,
  logEntries: initialLogEntries,
  forkEvents: [
    {
      timestamp: "14:31:08",
      parentCandidate: "retry-on-schema-drift",
      checkpointId: "ckpt_2c81ef03",
      prior: "explore example-driven prompts instead of hash-based dedup",
      branchId: "fork.abc12345",
      rationale:
        "iteration 2's hash-strictness regressed; try the orthogonal direction",
      origin: "fixture",
    },
  ],
  selectedNode: "more-specific-descriptions",
};

/** Drop undefined fields so a partial update leaves them as they were. */
function definedFields<T extends object>(value: T): Partial<T> {
  return Object.fromEntries(
    Object.entries(value).filter(([, v]) => v !== undefined),
  ) as Partial<T>;
}

function mergeNode(existing: TreeNode | undefined, patch: TreeNodePatch): TreeNode {
  const fields = definedFields(patch);
  if (!existing) {
    return {
      parent_candidate_name: null,
      iteration: 0,
      status: "seed",
      scores: { accuracy: 0 },
      delta: null,
      ...fields,
      candidate: patch.candidate,
    };
  }
  return {
    ...existing,
    ...fields,
    // A later event without lineage must not erase what an earlier one said.
    parent_candidate_name: fields.parent_candidate_name ?? existing.parent_candidate_name,
    iteration: fields.iteration || existing.iteration,
    isForkBranch: Boolean(fields.isForkBranch || existing.isForkBranch),
  };
}

/**
 * The thread a node belongs to. Nodes recorded before the backend
 * stamped thread ids belong to the run's root thread.
 */
function threadOf(threadId: string | undefined, state: DashboardState): string {
  return threadId ?? state.run?.threadId ?? "";
}

/** What a frontier update says about the candidate it decided on. */
function decidedStatus(update: FrontierUpdate): CandidateStatus | null {
  if (update.status === "smoke_failed") return "smoke_failed";
  const isBest = Boolean(update.candidate) && update.bestCandidate === update.candidate;
  if (update.accepted === true) return isBest ? "best" : "accepted";
  if (update.accepted === false) return "rejected";
  if (update.status === "rejected") return "rejected";
  if (update.status === "best" || update.status === "accepted") {
    return isBest ? "best" : "accepted";
  }
  return null;
}

function applyFrontierUpdate(state: DashboardState, update: FrontierUpdate): TreeNode[] {
  const decided = decidedStatus(update);
  // Events from before the backend reported a verdict carry only the
  // frontier, so frontier membership is the best available signal.
  const legacy = update.accepted === undefined && update.status === undefined;
  return state.tree.map(node => {
    // A fork's frontier says nothing about main's nodes, and vice versa:
    // two branches can disagree about which candidate is best.
    if (
      update.threadId !== undefined &&
      threadOf(node.threadId, state) !== update.threadId
    ) {
      return node;
    }
    if (update.candidate && node.candidate === update.candidate) {
      return decided ? { ...node, status: decided, delta: update.delta } : node;
    }
    if (update.bestCandidate && node.candidate === update.bestCandidate) {
      const delta = legacy && !update.candidate ? update.delta : node.delta;
      return { ...node, status: "best", delta };
    }
    if (update.bestCandidate && node.status === "best") {
      // Superseded, not rejected: it was accepted when it was measured.
      return { ...node, status: node.parent_candidate_name ? "accepted" : "seed" };
    }
    if (legacy && update.frontier.includes(node.candidate)) {
      return { ...node, status: "accepted" };
    }
    return node;
  });
}

function addForkEvent(forks: ForkEvent[], incoming: ForkEvent): ForkEvent[] {
  const key = forkKey(incoming);
  const idx = forks.findIndex(f => forkKey(f) === key);
  if (idx === -1) return [...forks, incoming];
  // One fork, reported twice. The dashboard's own record knows the prior
  // and the node it was forked from; the stream's knows only ids.
  const existing = forks[idx];
  const merged =
    incoming.origin === "stream"
      ? { ...incoming, ...existing }
      : { ...existing, ...incoming };
  const next = [...forks];
  next[idx] = merged;
  return next;
}

function sameThread(
  a: string | undefined,
  b: string | undefined,
  state: DashboardState,
): boolean {
  return threadOf(a, state) === threadOf(b, state);
}

function reducer(state: DashboardState, action: DashboardAction): DashboardState {
  switch (action.type) {
    case "SET_MODE":
      return { ...state, mode: action.payload };
    case "SET_RUN":
      return { ...state, run: action.payload };
    case "SET_TREE":
      return { ...state, tree: action.payload };
    case "ADD_TREE_NODE": {
      const key = nodeKey(action.payload);
      const idx = state.tree.findIndex(n => nodeKey(n) === key);
      if (idx === -1) {
        return { ...state, tree: [...state.tree, mergeNode(undefined, action.payload)] };
      }
      const tree = [...state.tree];
      tree[idx] = mergeNode(tree[idx], action.payload);
      return { ...state, tree };
    }
    case "SET_CHECKPOINT_ID":
      return {
        ...state,
        tree: state.tree.map(node => (
          node.candidate === action.payload.candidate
            ? { ...node, checkpointId: action.payload.checkpointId }
            : node
        )),
      };
    case "SET_BRANCHES":
      return {
        ...state,
        branches: action.payload,
        run: state.run
          ? { ...state.run, branches: Math.max(action.payload.length - 1, 0) }
          : state.run,
      };
    case "APPLY_FRONTIER_UPDATE":
      return { ...state, tree: applyFrontierUpdate(state, action.payload) };
    case "SET_ITERATIONS":
      return { ...state, iterations: action.payload };
    case "ADD_LOG_ENTRY": {
      const existingIdx = state.logEntries.findIndex(entry => entry.id === action.payload.id);
      if (existingIdx === -1) {
        return { ...state, logEntries: [...state.logEntries, action.payload] };
      }
      const next = [...state.logEntries];
      next[existingIdx] = action.payload;
      return { ...state, logEntries: next };
    }
    case "SET_LOG_ENTRIES":
      return { ...state, logEntries: action.payload };
    case "ADD_FORK_EVENT":
      return { ...state, forkEvents: addForkEvent(state.forkEvents, action.payload) };
    case "SET_FILTER":
      return { ...state, filters: { ...state.filters, ...action.payload } };
    case "SET_CONTEXT_TAB":
      return { ...state, contextTab: action.payload };
    case "SELECT_NODE":
      return { ...state, selectedNode: action.payload };
    case "SELECT_LOG_LINE":
      return { ...state, selectedLogLine: action.payload };
    case "SET_SSE_CONNECTED":
      return { ...state, sseConnected: action.payload };
    case "ADD_ITERATION": {
      const without = state.iterations.filter(
        i => !(i.candidateName === action.payload.candidateName && i.iteration === action.payload.iteration),
      );
      return { ...state, iterations: [...without, action.payload] };
    }
    case "UPDATE_ITERATION": {
      const { candidateName, phases, ...rest } = action.payload;
      const fields = definedFields(rest);
      return {
        ...state,
        iterations: state.iterations.map(chapter =>
          chapter.candidateName === candidateName &&
          (fields.iteration === undefined || chapter.iteration === fields.iteration)
            ? {
                ...chapter,
                ...fields,
                phases: phases ? { ...chapter.phases, ...definedFields(phases) } : chapter.phases,
              }
            : chapter,
        ),
      };
    }
    case "COMPLETE_ITERATION": {
      const { iteration, threadId, status } = action.payload;
      return {
        ...state,
        iterations: state.iterations.map(chapter =>
          chapter.iteration === iteration && sameThread(chapter.threadId, threadId, state)
            ? {
                ...chapter,
                phases: { propose: true, validate: true, benchmark: true, frontier: true },
                // frontier-updated already gave the precise verdict, if it came.
                status: chapter.status === "running" && status ? status : chapter.status,
              }
            : chapter,
        ),
      };
    }
    case "SET_CHECKPOINT":
      return { ...state, latestCheckpointId: action.payload };
    case "SET_ERROR":
      return { ...state, lastError: action.payload };
    case "CANCEL_BRANCH": {
      const threadId = action.payload;
      return {
        ...state,
        tree: state.tree.map(n =>
          n.threadId === threadId ? { ...n, status: "rejected" as const } : n,
        ),
      };
    }
    case "RESET":
      return { ...initialState };
    default:
      return state;
  }
}

const StateContext = createContext<DashboardState | undefined>(undefined);
const DispatchContext = createContext<Dispatch<DashboardAction> | undefined>(undefined);

export function DashboardProvider({
  children,
  initial,
}: {
  children: ReactNode;
  initial?: Partial<DashboardState>;
}) {
  const [state, dispatch] = useReducer(reducer, { ...initialState, ...initial });
  return createElement(
    StateContext.Provider,
    { value: state },
    createElement(DispatchContext.Provider, { value: dispatch }, children),
  );
}

export function useDashboard(): DashboardState {
  const ctx = useContext(StateContext);
  if (!ctx) throw new Error("useDashboard must be used inside <DashboardProvider />");
  return ctx;
}

export function useDashboardDispatch(): Dispatch<DashboardAction> {
  const ctx = useContext(DispatchContext);
  if (!ctx) throw new Error("useDashboardDispatch must be used inside <DashboardProvider />");
  return ctx;
}

export { initialState };
export type { DashboardState, DashboardAction } from "./types";
