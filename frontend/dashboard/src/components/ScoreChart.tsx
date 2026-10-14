'use client';
import { forkBranchCount, useDashboard, useDashboardDispatch } from '@/lib/state';
import { deltaTone, signedDelta, taskLabel } from '@/lib/format';
import { statusMeta, TONE_HEX, TONE_TEXT } from '@/lib/status';
import { Tag } from './ui/Badge';
import type { TreeNode } from '@/lib/types';

function Stat({ label, value, tone, title }: { label: string; value: string; tone?: keyof typeof TONE_TEXT; title?: string }) {
  return (
    <div className="px-3 py-2.5 min-w-0" title={title}>
      <div className="text-[12px] text-ink-subtle truncate">{label}</div>
      <div className={`mt-0.5 text-[20px] font-semibold tracking-[-0.3px] tabular ${tone ? TONE_TEXT[tone] : 'text-ink'}`}>
        {value}
      </div>
    </div>
  );
}

function scoreDomain(scores: number[]) {
  if (scores.length === 0) return { min: 0, max: 1 };
  const min = Math.max(0, Math.min(...scores) - 0.05);
  const max = Math.min(1, Math.max(...scores) + 0.05);
  return min === max ? { min: Math.max(0, min - 0.1), max: Math.min(1, max + 0.1) } : { min, max };
}

/** Accuracy by iteration (frontend/DESIGN.md §6, "Chart"). */
function AccuracyChart({ tree, selected }: { tree: TreeNode[]; selected: string | null }) {
  const dispatch = useDashboardDispatch();
  // A smoke-failed candidate never ran: it has no accuracy to plot, and
  // drawing it at 0 would read as a measured total failure.
  const scored = tree.filter(n => n.status !== 'smoke_failed');
  const main = scored.filter(n => !n.isForkBranch).sort((a, b) => a.iteration - b.iteration);
  const forks = scored.filter(n => n.isForkBranch).sort((a, b) => a.iteration - b.iteration);

  const width = 440;
  const height = 210;
  const pad = { top: 14, right: 16, bottom: 28, left: 36 };
  const w = width - pad.left - pad.right;
  const h = height - pad.top - pad.bottom;
  const maxIteration = Math.max(1, ...scored.map(n => n.iteration));
  const dom = scoreDomain(scored.map(n => n.scores.accuracy));
  const x = (v: number) => pad.left + (v / maxIteration) * w;
  const y = (v: number) => pad.top + h - ((v - dom.min) / Math.max(0.01, dom.max - dom.min)) * h;
  const yTicks = Array.from({ length: 4 }, (_, i) => dom.min + ((dom.max - dom.min) * i) / 3);
  const xTicks = Array.from({ length: maxIteration + 1 }, (_, i) => i);
  const path = (nodes: TreeNode[]) =>
    nodes.map((n, i) => `${i === 0 ? 'M' : 'L'} ${x(n.iteration)} ${y(n.scores.accuracy)}`).join(' ');

  const forkParent = scored.find(n => n.candidate === forks[0]?.parent_candidate_name);
  const forkLine = forkParent ? [forkParent, ...forks] : forks;

  // The running best: a dashed step through every accepted improvement.
  const winners = tree
    .filter(n => n.status === 'accepted' || n.status === 'best' || n.status === 'seed')
    .filter(n => n.status !== 'smoke_failed')
    .sort((a, b) => a.iteration - b.iteration);
  let best = -Infinity;
  const steps: string[] = [];
  for (const n of winners) {
    if (n.scores.accuracy < best) continue;
    steps.push(steps.length === 0
      ? `M ${x(n.iteration)} ${y(n.scores.accuracy)}`
      : `L ${x(n.iteration)} ${y(best)} L ${x(n.iteration)} ${y(n.scores.accuracy)}`);
    best = n.scores.accuracy;
  }
  if (steps.length) steps.push(`L ${x(maxIteration)} ${y(best)}`);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" aria-label="Accuracy by iteration">
      {yTicks.map(v => (
        <g key={v}>
          <line x1={pad.left} x2={width - pad.right} y1={y(v)} y2={y(v)} stroke="#23252a" />
          <text x={pad.left - 8} y={y(v) + 3.5} textAnchor="end" fill="#62666d" fontSize={11} className="tabular">
            {v.toFixed(2)}
          </text>
        </g>
      ))}
      {xTicks.map(v => (
        <text key={v} x={x(v)} y={height - 10} textAnchor="middle" fill="#62666d" fontSize={11}>
          {v}
        </text>
      ))}
      {steps.length > 0 && (
        <path d={steps.join(' ')} fill="none" stroke="#4cb782" strokeWidth={1.25} strokeDasharray="4 3" opacity={0.6} />
      )}
      {main.length > 1 && <path d={path(main)} fill="none" stroke="#d0d6e0" strokeWidth={1.5} opacity={0.75} />}
      {forkLine.length > 1 && <path d={path(forkLine)} fill="none" stroke="#bb87fc" strokeWidth={1.5} />}
      {scored.map(n => {
        const meta = statusMeta(n.status);
        const color = n.isForkBranch && n.status !== 'best' ? TONE_HEX.purple : TONE_HEX[meta.tone];
        const isSelected = selected === n.candidate;
        const r = n.status === 'best' ? 5 : 3.75;
        return (
          <g key={`${n.threadId ?? 'main'}:${n.candidate}`} onClick={() => dispatch({ type: 'SELECT_NODE', payload: n.candidate })} style={{ cursor: 'pointer' }}>
            <title>{`${n.candidate} · iteration ${n.iteration} · ${n.scores.accuracy.toFixed(2)} · ${meta.label}`}</title>
            {isSelected && <circle cx={x(n.iteration)} cy={y(n.scores.accuracy)} r={r + 4} fill="none" stroke="#5e6ad2" strokeWidth={1.5} />}
            <circle
              data-testid="chart-point"
              cx={x(n.iteration)}
              cy={y(n.scores.accuracy)}
              r={r}
              fill={meta.dot === 'ring' ? '#0f1011' : color}
              stroke={color}
              strokeWidth={1.5}
            />
          </g>
        );
      })}
    </svg>
  );
}

/** The inspector's overview: stats, the selected candidate, its tasks, the chart. */
export function ScoreChart() {
  const state = useDashboard();
  const { tree, selectedNode } = state;

  if (tree.length === 0) {
    return <div className="py-6 text-[13px] text-ink-subtle">Waiting for the first benchmark result…</div>;
  }

  const bestNode = tree.find(n => n.status === 'best');
  const scored = tree.filter(n => n.status !== 'smoke_failed');
  const latest = [...scored].sort((a, b) => b.iteration - a.iteration)[0];
  const accepted = tree.filter(n => n.status === 'accepted' || n.status === 'best').length;
  const rejected = tree.filter(n => n.status === 'rejected').length;
  const smokeFailed = tree.filter(n => n.status === 'smoke_failed').length;
  const branches = forkBranchCount(state);

  const focus = (selectedNode ? tree.find(n => n.candidate === selectedNode) : null) ?? bestNode ?? tree[0];
  const focusMeta = statusMeta(focus.status);
  const tasks = Object.entries(focus.scores.per_task ?? {})
    .map(([key, val]) => ({ key, label: taskLabel(key), rate: val.pass_rate, trials: val.trials }))
    .sort((a, b) => a.label.localeCompare(b.label));

  return (
    <div className="flex flex-col gap-5">
      <div className="grid grid-cols-4 divide-x divide-hairline rounded-md border border-hairline bg-surface-2">
        <Stat label="Best score" value={bestNode ? bestNode.scores.accuracy.toFixed(2) : '—'} />
        <Stat label="Latest Δ" value={signedDelta(latest?.delta)} tone={deltaTone(latest?.delta)} />
        <Stat
          label="Accepted"
          value={`${accepted} / ${accepted + rejected + smokeFailed}`}
          title={`${accepted} accepted, ${rejected} rejected${smokeFailed ? `, ${smokeFailed} smoke failed` : ''}`}
        />
        <Stat label="Branches" value={String(branches)} />
      </div>

      <section>
        <div className="text-[12px] text-ink-subtle">Candidate</div>
        <div data-testid="candidate-focus" className="mt-1 flex items-center gap-2 min-w-0">
          <span className="sr-only">Candidate focus:</span>
          <span className="font-mono text-[14px] font-medium text-ink truncate">{focus.candidate}</span>
          <Tag label={focusMeta.label} tone={focusMeta.tone} />
          {focus.isForkBranch && <Tag label="Fork" tone="purple" />}
        </div>
        <div className="mt-1 flex flex-wrap items-center gap-x-3 text-[12px] text-ink-subtle tabular">
          <span>Iteration {focus.iteration}</span>
          {focus.parent_candidate_name && (
            <span>
              from <span className="font-mono text-ink-muted">{focus.parent_candidate_name}</span>
            </span>
          )}
          {focus.status !== 'smoke_failed' && (
            <span>
              accuracy <span className="text-ink">{focus.scores.accuracy.toFixed(2)}</span>
              {focus.parent_candidate_name && (
                <span className={`ml-1 ${TONE_TEXT[deltaTone(focus.delta)]}`}>{signedDelta(focus.delta)}</span>
              )}
            </span>
          )}
        </div>
        {focus.hypothesis && <p className="mt-2 text-[13px] text-ink-muted leading-5">{focus.hypothesis}</p>}
      </section>

      <section>
        <div className="flex items-center justify-between text-[12px] text-ink-subtle">
          <span>Tasks</span>
          <span>Pass rate</span>
        </div>
        {tasks.length === 0 ? (
          <div className="mt-2 text-[13px] text-ink-subtle">No per-task results for this candidate.</div>
        ) : (
          <table className="mt-1 w-full text-[13px]">
            <tbody>
              {tasks.map(task => (
                <tr key={task.key} className="border-b border-hairline last:border-0">
                  <td className="py-1.5 pr-3 text-ink-muted first-letter:uppercase">{task.label}</td>
                  <td className="py-1.5 w-[45%]">
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 flex-1 rounded-full bg-surface-4 overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${Math.round(task.rate * 100)}%`,
                            background: task.rate >= 1 ? '#4cb782' : task.rate > 0 ? '#d0d6e0' : '#eb5757',
                          }}
                        />
                      </div>
                      <span className="w-10 text-right tabular text-ink">{Math.round(task.rate * 100)}%</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section>
        <div className="text-[12px] text-ink-subtle mb-1">Accuracy by iteration</div>
        <AccuracyChart tree={tree} selected={focus.candidate} />
        <div className="mt-1 flex items-center gap-4 text-[11px] text-ink-tertiary">
          <span className="flex items-center gap-1.5"><span className="w-3 h-px bg-[#d0d6e0]" />main thread</span>
          <span className="flex items-center gap-1.5"><span className="w-3 h-px bg-status-purple" />fork</span>
          <span className="flex items-center gap-1.5"><span className="w-3 border-t border-dashed border-status-green" />running best</span>
          <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full border border-status-red" />rejected</span>
        </div>
      </section>
    </div>
  );
}
