// Small display helpers shared by the panes (frontend/DESIGN.md §3, §7).

/** `HH:MM:SS` for an ISO timestamp; anything else is returned unchanged. */
export function clockTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleTimeString("en-GB", { hour12: false });
}

/** `1f1c81a8…824e`: long ids keep both ends, the full value goes in a tooltip. */
export function middleEllipsis(value: string, head = 8, tail = 4): string {
  if (value.length <= head + tail + 1) return value;
  return `${value.slice(0, head)}…${value.slice(-tail)}`;
}

/** `+0.33` / `-0.04` / `±0.00`, or `—` when unknown. */
export function signedDelta(delta: number | null | undefined): string {
  if (delta === null || delta === undefined) return "—";
  if (Math.abs(delta) < 0.005) return "±0.00";
  return `${delta > 0 ? "+" : ""}${delta.toFixed(2)}`;
}

/** Tone of a delta: gains green, losses red, no change neutral. */
export function deltaTone(delta: number | null | undefined): "green" | "red" | "neutral" {
  if (delta === null || delta === undefined || Math.abs(delta) < 0.005) return "neutral";
  return delta > 0 ? "green" : "red";
}

/** `task-008-paginate-off-by-one` → `paginate off by one`. */
export function taskLabel(taskKey: string): string {
  return taskKey.replace(/^task-\d+-/, "").replace(/^task-/, "").replace(/-/g, " ");
}
