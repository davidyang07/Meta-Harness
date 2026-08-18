'use client';
import { useEffect, useMemo, useState } from 'react';
import { useParams } from 'next/navigation';
import { useDashboard, useDashboardDispatch } from '@/lib/state';
import { ScoreChart } from './ScoreChart';
import { DiffViewer } from './DiffViewer';
import { TestOutput } from './TestOutput';
import { MemoryPanel } from './MemoryPanel';
import { getDiff, getTestOutput } from '@/lib/api';

/** Real added/removed line counts, read off the diff itself. */
function diffStats(diff: string): { added: number; removed: number } {
  let added = 0;
  let removed = 0;
  for (const line of diff.split('\n')) {
    if (line.startsWith('+') && !line.startsWith('+++')) added += 1;
    else if (line.startsWith('-') && !line.startsWith('---')) removed += 1;
  }
  return { added, removed };
}

export function ContextPanel() {
  const params = useParams<{ run_id: string }>();
  const { contextTab, selectedNode, tree, mode, run } = useDashboard();
  const dispatch = useDashboardDispatch();
  const [diffResult, setDiffResult] = useState<{ candidate: string; value: string | null } | null>(null);
  const [testResult, setTestResult] = useState<{ candidate: string; value: string | null } | null>(null);

  const tabs = ['chart', 'diff', 'test', 'memory'] as const;
  const selected = selectedNode ?? tree.find(n => n.status === 'best')?.candidate ?? tree[0]?.candidate ?? null;
  const selectedTreeNode = tree.find(n => n.candidate === selected) ?? null;
  const diff = diffResult?.candidate === selected ? diffResult.value : null;
  const testOut = testResult?.candidate === selected ? testResult.value : null;
  const perTask = Object.entries(selectedTreeNode?.scores.per_task ?? {});

  // `mode === 'mock'` means NO BACKEND — the built-in offline demo. It is
  // the only context in which this panel may render illustrative text,
  // and it is labelled as such. A live run whose *metrics* are mock
  // (metricsSource === 'mock') still shows its own real artifacts; the
  // status bar reports the provenance.
  const offlineDemo = mode === 'mock';
  const hasFixtureTaskData = offlineDemo && perTask.length > 0;

  useEffect(() => {
    let cancelled = false;
    if (!selected || mode !== 'live') return;
    Promise.allSettled([
      getDiff(params.run_id, selected),
      getTestOutput(params.run_id, selected),
    ]).then(([d, t]) => {
      if (cancelled) return;
      setDiffResult({
        candidate: selected,
        value: d.status === 'fulfilled' ? d.value : null,
      });
      setTestResult({
        candidate: selected,
        value: t.status === 'fulfilled' ? t.value : null,
      });
    });
    return () => {
      cancelled = true;
    };
  }, [mode, params.run_id, selected]);

  const stats = useMemo(() => (diff ? diffStats(diff) : null), [diff]);

  // Derived rather than stored: a fetch is outstanding exactly while the
  // results we hold are for some other candidate. Setting a `loading`
  // flag inside the effect would trigger a cascading render.
  const loading =
    mode === 'live' && Boolean(selected) && diffResult?.candidate !== selected;

  const fixtureDiffPreview = hasFixtureTaskData
    ? perTask
      .slice(0, 4)
      .map(([taskName, taskStats]) => {
        const passPct = Math.round(taskStats.pass_rate * 100);
        return `@@ task:${taskName}
-${taskName}: unstable retries (${passPct - 10}% pass)
+${taskName}: stricter guard + typed fallback (${passPct}% pass)`;
      })
      .join('\n\n')
    : null;

  const fixtureTestOutput = hasFixtureTaskData
    ? [
      `illustrative suite for ${selected ?? 'candidate'}`,
      ...perTask.map(([taskName, taskStats]) => {
        const passCount = taskStats.trials.filter(Boolean).length;
        const total = taskStats.trials.length;
        const status = passCount === total ? 'PASS' : passCount === 0 ? 'FAIL' : 'FLAKY';
        return `${status}  ${taskName}  (${passCount}/${total}, ${Math.round(taskStats.pass_rate * 100)}%)`;
      }),
      '',
      `summary: ${perTask.reduce((acc, [, s]) => acc + s.trials.filter(Boolean).length, 0)}/${perTask.reduce((acc, [, s]) => acc + s.trials.length, 0)} checks passed`,
    ].join('\n')
    : null;

  const emptyState = (what: string) =>
    selected
      ? `No ${what} recorded for ${selected}.`
      : 'No candidate selected yet.';

  const fixtureBadge = (
    <div
      data-testid="fixture-banner"
      className="mb-3 inline-block rounded border border-amber/40 bg-amber/10 px-2 py-1 text-[9px] uppercase tracking-wide text-amber"
    >
      Offline demo — illustrative, not from a run
    </div>
  );

  return (
    <div className="flex-1 flex flex-col bg-panel rounded overflow-hidden min-h-0">
      <div className="h-11 flex items-center gap-1 px-6 bg-header border-b border-border shrink-0">
        {tabs.map(tab => (
          <button
            key={tab}
            onClick={() => dispatch({ type: 'SET_CONTEXT_TAB', payload: tab })}
            className={`px-3 py-1.5 text-[9px] font-semibold uppercase tracking-wide rounded transition-colors ${contextTab === tab
                ? 'text-cyan border-b-2 border-cyan'
                : 'text-text-mid hover:text-text-hi'
              }`}
          >
            {tab}
          </button>
        ))}
      </div>

      <div className="flex-1 flex flex-col overflow-y-auto px-6 py-5 min-h-0">
        {contextTab === 'chart' && <div className="flex-1"><ScoreChart /></div>}

        {contextTab === 'diff' && diff && stats && (
          <div>
            <div className="flex items-center gap-2 mb-4 text-xs">
              <span className="text-text-hi font-semibold">agents/{selected ?? 'candidate'}.py</span>
              {/* Counted from the diff, not a placeholder. */}
              <span className="text-green">+{stats.added}</span>
              <span className="text-red">-{stats.removed}</span>
              {run?.metricsSource === 'mock' && (
                <span className="text-amber">mock run</span>
              )}
            </div>
            <DiffViewer diff={diff} />
          </div>
        )}
        {contextTab === 'diff' && !diff && fixtureDiffPreview && (
          <div className="space-y-3">
            {fixtureBadge}
            <pre className="text-xs leading-5 text-text-hi whitespace-pre-wrap">{fixtureDiffPreview}</pre>
          </div>
        )}
        {contextTab === 'diff' && !diff && !fixtureDiffPreview && (
          <div className="text-text-mid text-xs" data-testid="diff-empty">
            {loading ? 'Loading diff…' : emptyState('diff')}
          </div>
        )}

        {contextTab === 'test' && testOut && <TestOutput output={testOut} />}
        {contextTab === 'test' && !testOut && fixtureTestOutput && (
          <div className="space-y-3">
            {fixtureBadge}
            <TestOutput output={fixtureTestOutput} />
          </div>
        )}
        {contextTab === 'test' && !testOut && !fixtureTestOutput && (
          <div className="text-text-mid text-xs" data-testid="test-empty">
            {loading ? 'Loading test output…' : emptyState('test output')}
          </div>
        )}

        {contextTab === 'memory' && <MemoryPanel />}
      </div>
    </div>
  );
}
