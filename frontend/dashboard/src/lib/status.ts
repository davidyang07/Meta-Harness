// One mapping from a candidate's status to how it looks, used by the
// trajectory, the decision log, the inspector and the chart alike
// (frontend/DESIGN.md §2). Colour is never the only signal: every status
// also has a word and a dot shape.
import type { CandidateStatus, IterationChapter } from "./types";

export type StatusTone = "green" | "red" | "orange" | "yellow" | "blue" | "purple" | "neutral";

export type StatusMeta = {
  label: string;
  tone: StatusTone;
  /** filled: a solid dot; ring: an outlined dot; square: the baseline. */
  dot: "filled" | "ring" | "square";
  /** Lost or never ran: the row is dimmed. */
  dimmed: boolean;
};

export const TONE_HEX: Record<StatusTone, string> = {
  green: "#4cb782",
  red: "#eb5757",
  orange: "#f2994a",
  yellow: "#f2c94c",
  blue: "#4ea7fc",
  purple: "#bb87fc",
  neutral: "#8a8f98",
};

export const TONE_TEXT: Record<StatusTone, string> = {
  green: "text-status-green",
  red: "text-status-red",
  orange: "text-status-orange",
  yellow: "text-status-yellow",
  blue: "text-status-blue",
  purple: "text-status-purple",
  neutral: "text-ink-subtle",
};

export const TONE_TAG: Record<StatusTone, string> = {
  green: "text-status-green bg-status-green/12",
  red: "text-status-red bg-status-red/12",
  orange: "text-status-orange bg-status-orange/12",
  yellow: "text-status-yellow bg-status-yellow/12",
  blue: "text-status-blue bg-status-blue/12",
  purple: "text-status-purple bg-status-purple/12",
  neutral: "text-ink-subtle bg-surface-4",
};

const META: Record<CandidateStatus | "running", StatusMeta> = {
  best: { label: "Best", tone: "green", dot: "filled", dimmed: false },
  accepted: { label: "Accepted", tone: "green", dot: "ring", dimmed: false },
  rejected: { label: "Rejected", tone: "red", dot: "ring", dimmed: true },
  smoke_failed: { label: "Smoke failed", tone: "orange", dot: "ring", dimmed: true },
  seed: { label: "Baseline", tone: "neutral", dot: "square", dimmed: false },
  fork: { label: "Fork", tone: "purple", dot: "ring", dimmed: false },
  running: { label: "Running", tone: "blue", dot: "filled", dimmed: false },
};

export function statusMeta(status: CandidateStatus | IterationChapter["status"]): StatusMeta {
  return META[status] ?? META.seed;
}

/** A run's lifecycle status ("running", "completed", …) as a tone + word. */
export function runStatusMeta(status: string | undefined): { label: string; tone: StatusTone } {
  const s = (status ?? "unknown").toLowerCase();
  if (s === "running") return { label: "Running", tone: "blue" };
  if (s === "completed") return { label: "Completed", tone: "green" };
  if (s === "failed") return { label: "Failed", tone: "red" };
  if (s === "cancelled" || s === "cancelling") return { label: s === "cancelled" ? "Cancelled" : "Cancelling", tone: "orange" };
  if (s === "interrupted") return { label: "Interrupted", tone: "orange" };
  return { label: s.charAt(0).toUpperCase() + s.slice(1), tone: "neutral" };
}
