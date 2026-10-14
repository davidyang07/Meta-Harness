'use client';
import { forkBranchCount, useDashboard } from '@/lib/state';
import { middleEllipsis } from '@/lib/format';

/**
 * Provenance label. A viewer must be able to tell at a glance whether
 * the numbers on screen were measured or synthesized — the two are
 * never blended, and a demo run must never read as an experiment.
 */
function provenance(
  source: string | undefined,
  mode: string,
): { label: string; className: string } {
  if (source === 'mock') {
    return { label: 'Mock data (synthetic)', className: 'text-status-yellow' };
  }
  if (source === 'measured') {
    return { label: 'Measured data', className: 'text-status-green' };
  }
  if (mode === 'mock') {
    return { label: 'Mock mode (no backend)', className: 'text-status-yellow' };
  }
  return { label: 'Provenance unknown', className: 'text-ink-subtle' };
}

export function StatusBar() {
  const state = useDashboard();
  const { sseConnected, run, mode, latestCheckpointId, lastError, branches } = state;
  const ckpt = latestCheckpointId ?? run?.checkpointId;
  const { label, className } = provenance(run?.metricsSource, mode);
  const branchCount = forkBranchCount(state);
  const liveBranches = branches.filter(b => b.live).length;

  return (
    <footer className="h-7 shrink-0 flex items-center gap-5 px-4 bg-canvas border-t border-hairline text-[11px] text-ink-tertiary">
      <span className="flex items-center gap-1.5 text-ink-subtle">
        <span className={`w-1.5 h-1.5 rounded-full ${sseConnected ? 'bg-status-green' : 'bg-status-red'}`} />
        {sseConnected ? 'SSE connected' : 'Disconnected'}
      </span>
      <span className={className} data-testid="metrics-provenance">
        {label}
      </span>
      <span data-testid="branch-count" className="tabular">
        {branchCount} {branchCount === 1 ? 'branch' : 'branches'}
        {liveBranches > 0 ? ` (${liveBranches} live)` : ''}
      </span>
      <span className="font-mono" title={ckpt ?? undefined}>
        checkpoint {ckpt ? middleEllipsis(ckpt) : '—'}
      </span>
      {lastError && (
        <span className="text-status-red truncate" data-testid="status-error" title={lastError}>
          Error: {lastError}
        </span>
      )}
      <span className="ml-auto">v0.1.0</span>
    </footer>
  );
}
