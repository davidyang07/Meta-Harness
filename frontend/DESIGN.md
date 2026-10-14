---
version: 1
name: meta-harness-dashboard
description: >-
  A near-black, precise instrument for watching a self-improving agent
  search. Linear's visual language — a four-step charcoal surface ladder,
  hairline borders, light-gray type and one lavender accent — applied to a
  data-dense, three-pane dashboard laid out like LangGraph Studio: the
  thread's history on the left, what happened in the middle, an inspector
  on the right.
inspiration:
  visual: Linear (community DESIGN.md, getdesign.md/linear.app)
  layout: LangGraph / LangSmith Studio (thread history, fork from a checkpoint)

colors:
  canvas: "#010102"          # page background, behind the panes
  surface-1: "#0f1011"       # panes, cards
  surface-2: "#141516"       # hovered row, raised card, dialog
  surface-3: "#18191a"       # selected row, active tab, inputs
  surface-4: "#1e1f21"       # pressed, track of a bar
  hairline: "#23252a"        # every 1px divider and card border
  hairline-strong: "#34343a" # focused input, dialog border
  ink: "#f7f8f8"             # primary text, numbers that matter
  ink-muted: "#d0d6e0"       # body text in the log
  ink-subtle: "#8a8f98"      # labels, secondary text
  ink-tertiary: "#62666d"    # timestamps, disabled, footnotes
  accent: "#5e6ad2"          # Linear lavender: brand mark, primary button, focus, selection
  accent-hover: "#828fff"
  accent-ring: "rgba(94, 106, 210, 0.5)"
  # Status colours (Linear's product label palette). Only ever on dots,
  # small tags, deltas and chart marks — never as the fill of a large area.
  status-green: "#4cb782"    # accepted, best, measured, connected, passing
  status-red: "#eb5757"      # rejected, error, disconnected, failing
  status-orange: "#f2994a"   # smoke failed (never loaded, no score)
  status-yellow: "#f2c94c"   # mock / synthetic data, warnings
  status-blue: "#4ea7fc"     # running, informational
  status-purple: "#bb87fc"   # forks and branches

typography:
  family-sans: "Inter, -apple-system, system-ui, Segoe UI, sans-serif"
  family-mono: "JetBrains Mono, ui-monospace, SF Mono, Menlo, monospace"
  page-title:   { size: 20px, weight: 600, lineHeight: 1.3, letterSpacing: -0.3px }
  pane-title:   { size: 13px, weight: 500, lineHeight: 1.3, letterSpacing: 0 }
  body:         { size: 13px, weight: 400, lineHeight: 1.5, letterSpacing: 0 }
  body-sm:      { size: 12px, weight: 400, lineHeight: 1.45, letterSpacing: 0 }
  caption:      { size: 11px, weight: 500, lineHeight: 1.3, letterSpacing: 0.1px }
  stat:         { size: 20px, weight: 600, lineHeight: 1.1, letterSpacing: -0.3px, numeric: tabular }
  mono:         { size: 12px, weight: 400, lineHeight: 1.5, letterSpacing: 0 }

rounded:
  xs: 4px     # tags, status pills
  sm: 6px     # rows, segmented controls, buttons
  md: 8px     # cards, inputs
  lg: 12px    # dialogs

spacing:
  base: 4px
  scale: [4, 8, 12, 16, 20, 24, 32]
  row-height: 44px      # trajectory rows (two lines: name, then iteration · status · delta)
  pane-header: 44px
  top-bar: 48px
  status-bar: 28px
---

# Meta-Harness dashboard design

## 1. Atmosphere

The dashboard is an instrument, not a poster. Everything on screen is data
about a running search, so the chrome steps back: a near-black canvas,
charcoal panes separated by hairlines, light-gray type, and exactly one
chromatic accent. Colour appears only where it carries meaning — a status,
a delta, a chart mark — and is always paired with a word or a shape so it
is never the only signal.

Two rules from the original dashboard stay non-negotiable, because they
are about honesty rather than style:

- **Provenance is always visible.** Mock (synthetic) numbers carry a yellow
  "Mock data" tag in the top bar and the status bar. Measured numbers never
  look like mock ones.
- **Unknown is not zero.** A smoke-failed candidate shows "—", never 0.00,
  and is drawn in orange, not red.

**Scope.** This system covers the dashboard: the runs list at `/runs` and
every run page under it. The landing page at `/` deliberately keeps its
original animated title screen and palette (the `--color-void`, `--color-cyan`
… tokens at the end of `globals.css`); those tokens are for that page only.

## 2. Colour

Use the surface ladder for hierarchy instead of shadows: canvas → surface-1
(panes) → surface-2 (hover, dialogs) → surface-3 (selected, active,
inputs). Do not skip levels and do not add drop shadows on dark.

**Lavender (`accent`) is scarce.** It marks the brand glyph, the one
primary button on a screen, the focus ring and the selected row's left
edge. It is never a card fill, a section background or a chart line.

**Status colours** come from Linear's product label palette and follow one
mapping everywhere — tree dots, log tags, chart points, tags:

| Meaning | Colour | Shape / word that goes with it |
|---|---|---|
| Best candidate | green | filled dot + "Best" tag |
| Accepted | green | ring dot + "Accepted" |
| Rejected | red (dimmed row) | ring dot + "Rejected" |
| Smoke failed | orange | ring dot + "Smoke failed", score "—" |
| Baseline / seed | ink-subtle | square dot + "Baseline" |
| Fork branch | purple | its own lane in the history |
| Running | blue | pulsing dot + "Running" |
| Mock data | yellow | "Mock data" tag |

## 3. Typography

Inter for everything readable; JetBrains Mono only for things a user might
copy: candidate file paths, checkpoint and thread ids, timestamps, code,
diffs and test output. Numbers use tabular figures so columns line up.

- Sentence case everywhere ("Decision log", "Create fork"). No all-caps
  labels and no letter-spaced eyebrows.
- Body copy in the log is 13px `ink-muted`; labels are 12px `ink-subtle`;
  timestamps 11px mono `ink-tertiary`.
- Times are shown as `HH:MM:SS`, with the full ISO time in a tooltip.

## 4. Layout

```
┌──────────────────────────────────────────────────────────────────────────┐
│ ◆ Meta-Harness / Runs / demo-run   ● Completed  Mock data   [Replay run] │ 48
├───────────────────┬──────────────────────────────────┬───────────────────┤
│ Trajectory    7·1 │ Decision log ● live   All Tools … │ Overview Diff …   │ 44
├───────────────────┼──────────────────────────────────┼───────────────────┤
│ ●  baseline  0.67 │ Iteration 1  _mock_iter_1 Rejected│ Best  Δ  Acc  Br  │
│ │○ _iter_1   0.67 │  ✓ Propose → ✓ Validate → …       │ ───────────────── │
│ │○ _iter_2   0.67 │  12:01:28  plan   proposed from … │ Candidate …       │
│ ●╮ _iter_3   1.00 │  12:01:28  score  benchmarked …   │ Task      Pass    │
│ │●  fork_3   1.00 │  Accuracy 0.67  +0.00             │ fix typo  ██ 100% │
│ │○  _iter_4  1.00 │                                   │ Accuracy chart    │
├───────────────────┴──────────────────────────────────┴───────────────────┤
│ ● Connected   Mock data (synthetic)   1 branch   ckpt 1f1c81a8…   v0.1.0 │ 28
└──────────────────────────────────────────────────────────────────────────┘
   320px                       flexible                         480px
```

- Three panes on `surface-1`, separated by 1px `hairline` rules, edge to
  edge. No gaps, no rounded outer frame: the panes are the page.
- Pane headers are 44px with a 13px title on the left and controls on the
  right. Pane bodies scroll independently.
- Left pane 320px, right pane 480px, the decision log takes the rest. All
  three must stay readable at 1280px wide.

## 5. Patterns borrowed from LangGraph Studio

Studio shows a thread as a readable list of checkpoints with a **Fork**
action on each, not as a zoomable picture. The dashboard does the same.

- **Trajectory is a history list.** One 44px row per candidate, in
  iteration order, with a lane gutter on the left drawn like a git graph:
  the main thread is lane 0, each fork gets its own purple lane that
  branches off where it was forked. Each row shows the candidate name,
  "Iteration n" and the delta underneath, and the score right-aligned.
- **Fork from any row.** Hovering a row reveals a "Fork" button; right-click
  does the same. Both open the fork dialog pre-filled with that row's
  checkpoint.
- **Selecting a row drives the inspector**, as selecting a checkpoint does
  in Studio's state pane: the overview, diff and tests all follow the
  selection.
- **Run-level noise is folded.** Checkpoint writes collapse into one
  "N checkpoints written" line that expands on click; errors, forks and
  cancellations are always shown.

## 6. Components

**Top bar (48px, canvas).** Brand glyph (lavender) and "Meta-Harness" link
home, then a breadcrumb `Runs / <run id>`, the run status pill and, for
synthetic runs, the yellow "Mock data" tag. Secondary actions on the right.

**Status pill.** 22px tall, `rounded.xs`, `surface-3` background, a 6px
status dot and a sentence-case word.

**Tag.** 11px medium text in the status colour on a 12%-opacity tint of the
same colour, `rounded.xs`, 2px × 6px padding. Used for statuses and log
categories.

**Segmented control.** A `surface-3` track of small buttons; the active
one lifts to `surface-4` with `ink` text. Used for log filters and inspector
tabs.

**Buttons.** 30px tall, `rounded.sm`, 13px medium. Primary: lavender fill,
white text, one per screen. Secondary: `surface-3` fill, hairline border,
`ink` text. Ghost: no fill, `ink-subtle` text, `surface-3` on hover.

**Stat strip.** Four peer blocks in one row separated by hairlines: a 12px
label above a 20px tabular number — best score, latest Δ, accepted (of all
evaluated, with the breakdown in a tooltip) and fork branches. Deltas are
green/red with an explicit sign. Labels stay one or two words so they never
truncate.

**Per-task table.** Task name left, a thin pass-rate bar on a `surface-4`
track, the percentage right-aligned in tabular figures. Every task is
listed; nothing is truncated to fit.

**Chart.** Accuracy by iteration. The main thread is an `ink-muted` line,
forks are purple, accepted points green, rejected points hollow red, the
best point a larger green dot. The running best is a dashed green step.
A one-line legend sits directly under the plot (direct end-of-line labels
collide when two series finish at the same score); axes are 11px
`ink-tertiary`. Clicking a point selects that candidate.

**Decision log chapter.** A header row ("Iteration 3", the candidate name in
`ink`, its status tag), the phase steps as small check marks, the
hypothesis in `ink-subtle`, then the entries as compact rows: time, a
category tag, the message. The chapter ends with its outcome line:
accuracy, delta, verdict.

**Fork card.** `surface-2`, `rounded.md`, a 2px purple left edge, the title
"Fork created", then "From" (thread @ checkpoint, mono) and "Prior" (the
user's actual text).

**Dialog.** `surface-2`, `rounded.lg`, `hairline-strong` border over a 60%
black scrim. Title 15px semibold, labels 12px `ink-subtle`, inputs on
`surface-3` with a lavender focus ring.

## 7. Do and don't

**Do**
- Keep one primary (lavender) action per screen.
- Show every number with its unit or sign: `0.67`, `+0.33`, `100%`.
- Right-align numbers and use tabular figures.
- Pair every status colour with a word or a distinct dot shape.
- Truncate long ids with an ellipsis in the middle and put the full value
  in a tooltip.

**Don't**
- Don't use pure black (#000) for the canvas, or drop shadows on dark.
- Don't use lavender as a fill, a chart colour or a decoration.
- Don't introduce gradients, glows, scanlines or typing animations
  (the landing page's title screen is the one exception; see §1).
- Don't write labels in all caps or add letter-spacing to small text.
- Don't render a smoke-failed candidate as 0, or mock numbers without the
  "Mock data" tag.

## 8. Accessibility

- Text on `surface-1` meets WCAG AA: `ink` and `ink-muted` for anything a
  user must read; `ink-tertiary` only for timestamps and decoration.
- Every interactive row is keyboard reachable, shows the lavender focus
  ring, and opens the fork dialog with the context-menu key as well as a
  right-click.
- Status is never colour alone (see §2).

## 9. Known gaps

- Linear's custom typefaces are proprietary; Inter and JetBrains Mono are
  the open substitutes the source DESIGN.md recommends.
- There is no light theme. The source system has none, and the dashboard is
  used in dark rooms on projectors.
- The status palette is Linear's in-product label palette, which the
  marketing-site DESIGN.md mentions but does not tokenise.
