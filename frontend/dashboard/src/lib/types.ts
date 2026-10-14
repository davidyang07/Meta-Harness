// Dashboard type contracts. Mirrors backend INTERFACES.md §1, §6, §7
// shapes; the SSE event types match streaming.REGISTERED_EVENT_TYPES.

/**
 * Where a candidate stands in the search.
 *
 * `smoke_failed` is distinct from `rejected`: a rejected candidate was
 * measured and lost, a smoke-failed one never loaded, so it has no score
 * at all (its delta is null) and must never be drawn as 0%.
 */
export type CandidateStatus =
  | "seed"
  | "accepted"
  | "rejected"
  | "best"
  | "fork"
  | "smoke_failed";

export type LogTag =
  | "orient"
  | "plan"
  | "tool/read"
  | "tool/patch"
  | "act"
  | "verify"
  | "score"
  | "fail"
  | "fork"
  | "memory"
  | "checkpoint";

export type ContextTab = "chart" | "diff" | "test" | "memory";

export type LogFilter = "all" | "tools" | "verify" | "scores" | "forks";

export type Phase = "orient" | "plan" | "act" | "verify" | "submit";

export type OuterPhaseFlags = {
  propose: boolean;
  validate: boolean;
  benchmark: boolean;
  frontier: boolean;
};

export type Scores = {
  accuracy: number;
  per_task?: Record<string, { pass_rate: number; trials: boolean[] }>;
};

export type TreeNode = {
  candidate: string;
  parent_candidate_name: string | null;
  iteration: number;
  checkpointId?: string;
  status: CandidateStatus;
  scores: Scores;
  hypothesis?: string;
  axis?: "exploration" | "exploitation";
  delta: number | null;
  isForkBranch?: boolean;
  threadId?: string;
};

/**
 * A partial node. Fields left undefined are not touched when merged into
 * an existing node, so an event that knows only a candidate's lineage
 * (candidate-created) can never reset the score or status that a later
 * event (eval-result, frontier-updated) already recorded.
 */
export type TreeNodePatch = Partial<TreeNode> & { candidate: string };

/** `candidateName` of an entry that belongs to the run, not to one candidate. */
export const RUN_LEVEL = "outer-loop";

export type LogEntry = {
  id: string;
  timestamp: string;
  tag: LogTag;
  text: string;
  candidateName: string;
  details?: string;
  expandable?: boolean;
  expandedContent?: string;
  threadId?: string;
  /** Outer-loop iteration the entry belongs to, when the event named one. */
  iteration?: number;
};

export type ForkEvent = {
  timestamp: string;
  parentCandidate: string;
  checkpointId: string;
  prior: string;
  branchId: string;
  rationale: string;
  /**
   * The fork's LangGraph thread id. A fork the dashboard requested is
   * announced twice — once by the dashboard itself and once by the
   * backend's `fork-created` event — and this is what makes them one.
   */
  threadId?: string;
  /** Who reported the fork. Only the dashboard's own record knows the prior. */
  origin?: "dashboard" | "stream" | "fixture";
};

export type IterationChapter = {
  iteration: number;
  candidateName: string;
  status: CandidateStatus | "running";
  phases: OuterPhaseFlags;
  isForkBranch?: boolean;
  hypothesis?: string;
  threadId?: string;
};

export type RunSummary = {
  runId: string;
  threadId: string;
  branches: number;
  checkpointId: string | null;
  bestScore: number | null;
  status: string;
  iteration: number;
  isMock?: boolean;
  /**
   * Provenance of this run's numbers, straight from the backend
   * manifest. "mock" means synthetic demo data; "measured" means real
   * inner-loop trials. The UI must never present the two the same way.
   */
  metricsSource?: "measured" | "mock" | "unknown";
};

/** One branch of the run's search tree, as returned by GET /runs/{id}/trajectory. */
export type BranchNode = {
  threadId: string;
  parentThreadId: string | null;
  parentCheckpointId: string | null;
  branchId: string | null;
  name: string | null;
  status: string;
  /** True only while this backend process is actively driving the branch. */
  live: boolean;
};

export type MemoryEntry = {
  key: string;
  pattern: string;
  mechanism_axis?: string;
  score_delta?: number;
  evidence_run_ids?: string[];
  created_at?: string;
};

export type DashboardFilters = {
  activeFilter: LogFilter;
  searchQuery: string;
};

export type DashboardState = {
  mode: "live" | "mock";
  run: RunSummary | null;
  tree: TreeNode[];
  iterations: IterationChapter[];
  logEntries: LogEntry[];
  forkEvents: ForkEvent[];
  filters: DashboardFilters;
  contextTab: ContextTab;
  selectedNode: string | null;
  selectedLogLine: string | null;
  sseConnected: boolean;
  latestCheckpointId: string | null;
  lastError: string | null;
  branches: BranchNode[];
};

export type RunInfo = RunSummary;

export type EvolutionRow = {
  iteration: number;
  candidate: string;
  parent_candidate_name: string | null;
  scores: Scores;
  delta: number | null;
  outcome: string;
  thread_id?: string;
};

export type FrontierUpdate = {
  /** The thread whose frontier this is. Only that thread's nodes change. */
  threadId?: string;
  /** The candidate this update decided on. */
  candidate?: string;
  accepted?: boolean;
  /** The backend's verdict on `candidate`: "best", "rejected", "smoke_failed"… */
  status?: string;
  frontier: string[];
  bestCandidate: string | null;
  /** Delta of `candidate` against the thread's previous best. */
  delta: number | null;
};

export type DashboardAction =
  | { type: "SET_MODE"; payload: DashboardState["mode"] }
  | { type: "SET_RUN"; payload: RunSummary | null }
  | { type: "SET_TREE"; payload: TreeNode[] }
  | { type: "ADD_TREE_NODE"; payload: TreeNodePatch }
  | { type: "SET_CHECKPOINT_ID"; payload: { candidate: string; checkpointId: string } }
  | { type: "APPLY_FRONTIER_UPDATE"; payload: FrontierUpdate }
  | { type: "SET_ITERATIONS"; payload: IterationChapter[] }
  | { type: "ADD_ITERATION"; payload: IterationChapter }
  | {
      /** Merge into an existing chapter for this candidate; never creates one. */
      type: "UPDATE_ITERATION";
      payload: Omit<Partial<IterationChapter>, "phases"> & {
        candidateName: string;
        /** Phases to mark; the ones left out keep their current value. */
        phases?: Partial<OuterPhaseFlags>;
      };
    }
  | {
      /**
       * `iteration-complete` names an iteration and a thread, not a
       * candidate: finish whichever chapter that is, never invent one.
       */
      type: "COMPLETE_ITERATION";
      payload: {
        iteration: number;
        threadId?: string;
        status: IterationChapter["status"] | null;
      };
    }
  | { type: "ADD_LOG_ENTRY"; payload: LogEntry }
  | { type: "SET_LOG_ENTRIES"; payload: LogEntry[] }
  | { type: "ADD_FORK_EVENT"; payload: ForkEvent }
  | { type: "SET_FILTER"; payload: Partial<DashboardFilters> }
  | { type: "SET_CONTEXT_TAB"; payload: ContextTab }
  | { type: "SELECT_NODE"; payload: string | null }
  | { type: "SELECT_LOG_LINE"; payload: string | null }
  | { type: "SET_SSE_CONNECTED"; payload: boolean }
  | { type: "SET_CHECKPOINT"; payload: string }
  | { type: "SET_ERROR"; payload: string | null }
  | { type: "SET_BRANCHES"; payload: BranchNode[] }
  | { type: "CANCEL_BRANCH"; payload: string }
  | { type: "RESET" };
