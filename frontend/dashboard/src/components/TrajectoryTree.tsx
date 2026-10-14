'use client';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { KeyboardEvent, MouseEvent } from 'react';
import { useParams } from 'next/navigation';
import { forkBranchCount, forkKey, useDashboard, useDashboardDispatch } from '@/lib/state';
import { ForkModal } from './ForkModal';
import { resolveCheckpointForNode } from '@/lib/api';
import { deltaTone, signedDelta } from '@/lib/format';
import { statusMeta, TONE_HEX, TONE_TEXT } from '@/lib/status';
import { Tag } from './ui/Badge';
import { ForkIcon } from './ui/Icons';
import type { ForkEvent, TreeNode } from '@/lib/types';

/** The first node of the branch a fork event created, if it has arrived. */
function nodeForFork(tree: TreeNode[], fork: ForkEvent): TreeNode | undefined {
  if (fork.threadId) return tree.find(node => node.threadId === fork.threadId);
  const rawBranchId = fork.branchId.trim();
  const normalizedBranchId = rawBranchId.replace(/^fork\./, '');
  return tree.find(node => {
    const threadId = node.threadId ?? '';
    return threadId.includes(`.fork.${normalizedBranchId}`) || threadId.includes(rawBranchId);
  });
}

// ── history layout ─────────────────────────────────────────────────────
//
// The trajectory is drawn the way LangGraph Studio shows a thread: a
// readable list, one row per candidate in iteration order, with a git-style
// lane gutter. The main thread is lane 0; every fork gets its own lane that
// branches off at the row it was forked from (frontend/DESIGN.md §5).

const ROW_H = 44;
const LANE_W = 14;
const GUTTER_PAD = 10;

type Row = { node: TreeNode; lane: number };
type Lane = { first: number; last: number; parentRow: number | null; parentLane: number; fork: boolean };

function laneKeyOf(node: TreeNode): string {
  return node.isForkBranch ? `fork:${node.threadId ?? node.candidate}` : 'main';
}

function layoutHistory(tree: TreeNode[]): { rows: Row[]; lanes: Lane[] } {
  const order = new Map(tree.map((node, i) => [node, i]));
  const laneIndex = new Map<string, number>([['main', 0]]);
  // Lanes are numbered by when their branch first appears in the history.
  for (const node of [...tree].sort((a, b) => a.iteration - b.iteration || order.get(a)! - order.get(b)!)) {
    const key = laneKeyOf(node);
    if (!laneIndex.has(key)) laneIndex.set(key, laneIndex.size);
  }
  const rows: Row[] = tree
    .map(node => ({ node, lane: laneIndex.get(laneKeyOf(node)) ?? 0 }))
    .sort(
      (a, b) =>
        a.node.iteration - b.node.iteration ||
        a.lane - b.lane ||
        order.get(a.node)! - order.get(b.node)!,
    );

  const rowOf = new Map(rows.map((row, i) => [row.node.candidate, i]));
  const lanes: Lane[] = Array.from({ length: laneIndex.size }, () => ({
    first: -1,
    last: -1,
    parentRow: null,
    parentLane: 0,
    fork: false,
  }));
  rows.forEach((row, i) => {
    const lane = lanes[row.lane];
    if (lane.first === -1) {
      lane.first = i;
      lane.fork = Boolean(row.node.isForkBranch);
      const parent = row.node.parent_candidate_name;
      const parentRow = parent !== null ? rowOf.get(parent) : undefined;
      if (lane.fork && parentRow !== undefined && parentRow < i) {
        lane.parentRow = parentRow;
        lane.parentLane = rows[parentRow].lane;
      }
    }
    lane.last = i;
  });
  return { rows, lanes };
}

function laneX(lane: number): number {
  return GUTTER_PAD + lane * LANE_W;
}

function Gutter({ index, row, lanes, width }: { index: number; row: Row; lanes: Lane[]; width: number }) {
  const mid = ROW_H / 2;
  const meta = statusMeta(row.node.status);
  const color = TONE_HEX[meta.tone];
  const x = laneX(row.lane);
  const segments: React.ReactNode[] = [];

  lanes.forEach((lane, laneNo) => {
    if (lane.first === -1) return;
    const start = lane.parentRow ?? lane.first;
    const stroke = lane.fork ? '#bb87fc' : '#34343a';
    const lx = laneX(laneNo);
    if (index > start && index <= lane.last) {
      segments.push(<line key={`t${laneNo}`} x1={lx} x2={lx} y1={0} y2={index === lane.first || index === lane.last ? mid : ROW_H} stroke={stroke} strokeWidth={1.5} strokeOpacity={lane.fork ? 0.7 : 1} />);
    }
    if (index >= lane.first && index < lane.last && index !== lane.parentRow) {
      segments.push(<line key={`b${laneNo}`} x1={lx} x2={lx} y1={mid} y2={ROW_H} stroke={stroke} strokeWidth={1.5} strokeOpacity={lane.fork ? 0.7 : 1} />);
    }
    if (index === lane.parentRow) {
      const px = laneX(lane.parentLane);
      segments.push(
        <path
          key={`f${laneNo}`}
          d={`M ${px} ${mid} C ${px} ${ROW_H - 4}, ${lx} ${mid + 6}, ${lx} ${ROW_H}`}
          fill="none"
          stroke="#bb87fc"
          strokeWidth={1.5}
          strokeOpacity={0.7}
        />,
      );
    }
  });

  return (
    <svg width={width} height={ROW_H} className="shrink-0" aria-hidden>
      {segments}
      {meta.dot === 'square' ? (
        <rect x={x - 3.5} y={mid - 3.5} width={7} height={7} rx={1.5} fill="#0f1011" stroke={color} strokeWidth={1.5} />
      ) : meta.dot === 'filled' ? (
        <circle cx={x} cy={mid} r={4.5} fill={color} stroke="#0f1011" strokeWidth={2} />
      ) : (
        <circle cx={x} cy={mid} r={3.75} fill="#0f1011" stroke={color} strokeWidth={1.5} />
      )}
    </svg>
  );
}

export function TrajectoryTree() {
  const params = useParams<{ run_id: string }>();
  const state = useDashboard();
  const { tree, selectedNode, forkEvents, mode, run } = state;
  const dispatch = useDashboardDispatch();
  const [forkTarget, setForkTarget] = useState<{ candidate: string; checkpointId: string; parentThreadId?: string } | null>(null);
  // The fork whose first node was already auto-selected. Without it the
  // effect re-selected that node on every selection change, so after any
  // fork the user could not select anything else.
  const handledForkRef = useRef<string | null>(null);

  useEffect(() => {
    const latestFork = forkEvents.at(-1);
    if (!latestFork) return;
    const key = forkKey(latestFork);
    if (handledForkRef.current === key) return;
    // The branch's first node may arrive after the fork event; the tree
    // changing re-runs this until it does, then never again for this fork.
    const branchNode = nodeForFork(tree, latestFork);
    if (!branchNode) return;
    handledForkRef.current = key;
    dispatch({ type: 'SELECT_NODE', payload: branchNode.candidate });
  }, [tree, forkEvents, dispatch]);

  const openForkFor = useCallback(
    async (node: TreeNode) => {
      let checkpointId = node.checkpointId ?? undefined;
      if (!checkpointId && mode === 'mock') {
        checkpointId = run?.checkpointId ?? undefined;
      }
      if (!checkpointId && mode === 'live') {
        // Resolved against the node's own thread: a fork-branch node's
        // checkpoints are not in the root thread's history.
        const resolvedCheckpointId = await resolveCheckpointForNode(params.run_id, {
          candidate: node.candidate,
          iteration: node.iteration,
          threadId: node.threadId,
        }).catch(() => null);
        checkpointId = resolvedCheckpointId ?? undefined;
        if (checkpointId) {
          dispatch({
            type: 'SET_CHECKPOINT_ID',
            payload: { candidate: node.candidate, checkpointId },
          });
        }
      }
      if (!checkpointId) {
        dispatch({
          type: 'ADD_LOG_ENTRY',
          payload: {
            id: `fork-pending-${Date.now()}`,
            timestamp: new Date().toISOString(),
            tag: 'fork',
            text: `${node.candidate} has no checkpoint yet; this candidate may not be persisted in checkpoints yet`,
            candidateName: node.candidate,
            threadId: node.threadId,
          },
        });
        return;
      }
      setForkTarget({
        candidate: node.candidate,
        checkpointId,
        parentThreadId: node.threadId,
      });
    },
    [dispatch, mode, params.run_id, run?.checkpointId],
  );

  const { rows, lanes } = useMemo(() => layoutHistory(tree), [tree]);
  const gutterWidth = GUTTER_PAD * 2 + Math.max(0, lanes.length - 1) * LANE_W;
  const forks = forkBranchCount(state);

  const onContextMenu = (event: MouseEvent, node: TreeNode) => {
    event.preventDefault();
    void openForkFor(node);
  };
  const onKeyDown = (event: KeyboardEvent, node: TreeNode) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      dispatch({ type: 'SELECT_NODE', payload: node.candidate });
    } else if (event.key === 'ContextMenu' || (event.shiftKey && event.key === 'F10')) {
      event.preventDefault();
      void openForkFor(node);
    }
  };

  return (
    <>
      <section className="flex-1 flex flex-col min-h-0 bg-surface-1">
        <div className="h-11 shrink-0 flex items-center justify-between px-4 border-b border-hairline">
          <h2 className="text-[13px] font-medium text-ink">Trajectory</h2>
          {tree.length > 0 && (
            <span className="text-[12px] text-ink-subtle tabular">
              {tree.length} {tree.length === 1 ? 'candidate' : 'candidates'} · {forks} {forks === 1 ? 'fork' : 'forks'}
            </span>
          )}
        </div>
        <div className="flex-1 overflow-y-auto py-1" role="list" aria-label="Candidate history">
          {tree.length === 0 && (
            <div className="px-4 py-6 text-[13px] text-ink-subtle">Waiting for the first candidate…</div>
          )}
          {rows.map((row, index) => {
            const { node } = row;
            const meta = statusMeta(node.status);
            const selected = selectedNode === node.candidate;
            const noScore = node.status === 'smoke_failed';
            return (
              <div
                key={`${node.threadId ?? 'main'}:${node.candidate}`}
                role="listitem"
                tabIndex={0}
                data-testid="trajectory-node"
                data-candidate={node.candidate}
                data-status={node.status}
                title={node.hypothesis}
                aria-current={selected ? "true" : undefined}
                onClick={() => dispatch({ type: 'SELECT_NODE', payload: node.candidate })}
                onContextMenu={event => onContextMenu(event, node)}
                onKeyDown={event => onKeyDown(event, node)}
                className={`group relative flex items-center pr-3 cursor-pointer outline-none transition-colors ${
                  selected
                    ? 'bg-surface-3 shadow-[inset_2px_0_0_var(--color-accent)]'
                    : 'hover:bg-surface-2 focus-visible:bg-surface-2'
                }`}
                style={{ height: ROW_H }}
              >
                <Gutter index={index} row={row} lanes={lanes} width={gutterWidth} />
                <div className={`min-w-0 flex-1 pl-1 ${meta.dimmed && !selected ? 'opacity-60' : ''}`}>
                  <div className="flex items-center gap-1.5 min-w-0">
                    <span className="truncate text-[13px] text-ink font-mono">{node.candidate}</span>
                    {node.status === 'best' && <Tag label="Best" tone="green" />}
                  </div>
                  <div className="flex items-center gap-1.5 text-[11px] text-ink-tertiary tabular">
                    <span>Iteration {node.iteration}</span>
                    {node.isForkBranch && <span className="text-status-purple">fork</span>}
                    {node.status !== 'best' && (
                      <span className={TONE_TEXT[meta.tone]}>{meta.label}</span>
                    )}
                    {!noScore && node.parent_candidate_name !== null && (
                      <span className={TONE_TEXT[deltaTone(node.delta)]}>{signedDelta(node.delta)}</span>
                    )}
                  </div>
                </div>
                <button
                  type="button"
                  onClick={event => {
                    event.stopPropagation();
                    void openForkFor(node);
                  }}
                  className="hidden group-hover:inline-flex group-focus-visible:inline-flex focus-visible:inline-flex items-center gap-1 h-6 px-2 mr-2 rounded bg-surface-4 border border-hairline-strong text-[12px] text-ink-muted hover:text-ink"
                  aria-label={`Fork from ${node.candidate}`}
                >
                  <ForkIcon /> Fork
                </button>
                <span
                  data-testid="trajectory-node-score"
                  className={`tabular text-[13px] font-medium ${noScore ? 'text-ink-tertiary' : 'text-ink'}`}
                >
                  {noScore ? '—' : node.scores.accuracy.toFixed(2)}
                </span>
              </div>
            );
          })}
        </div>
      </section>
      {forkTarget && (
        <ForkModal
          candidateName={forkTarget.candidate}
          checkpointId={forkTarget.checkpointId}
          parentThreadId={forkTarget.parentThreadId}
          onClose={() => setForkTarget(null)}
        />
      )}
    </>
  );
}
