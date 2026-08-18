'use client';
import { useDashboard } from '@/lib/state';

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
    return { label: 'Mock data (synthetic)', className: 'text-amber' };
  }
  if (source === 'measured') {
    return { label: 'Measured data', className: 'text-green' };
  }
  if (mode === 'mock') {
    return { label: 'Mock mode (no backend)', className: 'text-amber' };
  }
  return { label: 'Provenance unknown', className: 'text-text-mid' };
}

export function StatusBar() {
  const { sseConnected, run, mode, latestCheckpointId, lastError, branches } =
    useDashboard();
  const ckpt = latestCheckpointId ?? run?.checkpointId;
  const { label, className } = provenance(run?.metricsSource, mode);
  // The root thread is in the trajectory too, so branches = threads - 1.
  const branchCount = branches.length > 0 ? branches.length - 1 : run?.branches ?? 0;
  const liveBranches = branches.filter(b => b.live).length;

  return (
    <div className="h-7 flex items-center gap-6 px-6 bg-header border-t border-border text-[10px] tracking-wide text-text-lo uppercase">
      <span className="flex items-center gap-1.5">
        <span className={`w-1.5 h-1.5 rounded-full ${sseConnected ? 'bg-green' : 'bg-red'}`} />
        {sseConnected ? 'SSE connected' : 'Disconnected'}
      </span>
      <span className={className} data-testid="metrics-provenance">
        {label}
      </span>
      <span data-testid="branch-count">
        {branchCount} branches{liveBranches > 0 ? ` (${liveBranches} live)` : ''}
      </span>
      <span>ckpt: {ckpt ? `${ckpt.slice(0, 8)}…${ckpt.slice(-4)}` : '—'}</span>
      {lastError && <span className="text-red" data-testid="status-error">err: {lastError}</span>}
      <span className="ml-auto">v0.1.0</span>
    </div>
  );
}
