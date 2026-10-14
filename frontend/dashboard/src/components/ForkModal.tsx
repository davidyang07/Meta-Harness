'use client';
import { useState } from 'react';
import { useParams } from 'next/navigation';
import { useDashboardDispatch } from '@/lib/state';
import { forkRun } from '@/lib/api';
import { refreshBranches } from '@/lib/sse';
import { ForkIcon } from './ui/Icons';

type ForkModalProps = {
  candidateName: string;
  checkpointId: string;
  parentThreadId?: string;
  onClose: () => void;
};

export function ForkModal({ candidateName, checkpointId, parentThreadId, onClose }: ForkModalProps) {
  const params = useParams<{ run_id: string }>();
  const dispatch = useDashboardDispatch();
  const [prior, setPrior] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCreate = async () => {
    if (!prior.trim() || submitting) return;
    setSubmitting(true);

    const fallbackBranchId = `fork.${Math.random().toString(36).slice(2, 10)}`;
    const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });

    try {
      const result = await forkRun(params.run_id, {
        parent_checkpoint_id: checkpointId,
        parent_thread_id: parentThreadId,
        // Pin fork proposer lineage to the selected node so the new
        // candidate branches from the exact error/source point in the tree.
        mods: { proposer_prior: prior.trim(), best_candidate: candidateName },
        name: fallbackBranchId,
      });
      const branchId = result.branch_id ?? fallbackBranchId;
      dispatch({ type: 'SET_ERROR', payload: null });
      dispatch({
        type: 'ADD_LOG_ENTRY',
        payload: {
          id: `fork-started-${Date.now()}`,
          timestamp,
          tag: 'fork',
          text: `fork ${branchId} launched from ${parentThreadId ?? params.run_id}@${checkpointId.slice(0, 8)}…`,
          candidateName,
          threadId: result.thread_id,
        },
      });
      // Only a fork the backend actually accepted becomes a fork event.
      // Recording one on failure would draw a branch that does not exist.
      // The backend also announces it (fork-created); the thread id is
      // what lets the reducer recognise the two as one fork.
      dispatch({
        type: 'ADD_FORK_EVENT',
        payload: {
          timestamp,
          parentCandidate: candidateName,
          checkpointId,
          prior: prior.trim(),
          branchId,
          rationale: '',
          threadId: result.thread_id,
          origin: 'dashboard',
        },
      });
      // Pick up the new branch immediately rather than waiting for SSE.
      await refreshBranches(params.run_id, dispatch);
      setSubmitting(false);
      onClose();
      return;
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'fork request failed';
      setError(message);
      dispatch({ type: 'SET_ERROR', payload: message });
      dispatch({
        type: 'ADD_LOG_ENTRY',
        payload: {
          id: `fork-failed-${Date.now()}`,
          timestamp,
          tag: 'fail',
          text: `fork request failed for ${parentThreadId ?? params.run_id}@${checkpointId.slice(0, 8)}: ${message}`,
          candidateName,
        },
      });
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" onClick={onClose}>
      <div className="absolute inset-0 bg-black/60" />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="fork-dialog-title"
        className="relative w-[440px] rounded-xl bg-surface-2 border border-hairline-strong p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]"
        onClick={e => e.stopPropagation()}
        onKeyDown={e => {
          if (e.key === 'Escape') onClose();
        }}
      >
        <div className="flex items-center justify-between mb-4">
          <h2 id="fork-dialog-title" className="flex items-center gap-2 text-[15px] font-semibold text-ink">
            <ForkIcon className="text-status-purple" />
            Create fork
          </h2>
          <button
            onClick={onClose}
            aria-label="Close"
            className="w-7 h-7 rounded-md text-ink-subtle hover:text-ink hover:bg-surface-3"
          >
            ✕
          </button>
        </div>

        <div className="mb-4">
          <div className="text-[12px] text-ink-subtle">Fork from</div>
          <div className="mt-1.5 px-3 py-2 rounded-md bg-surface-3 border border-hairline text-[13px]">
            <span className="font-mono text-ink">{candidateName}</span>
            <span className="block mt-0.5 font-mono text-[11px] text-ink-tertiary break-all">checkpoint {checkpointId}</span>
          </div>
        </div>

        <label className="block mb-4">
          <span className="text-[12px] text-ink-subtle">New prior or hypothesis</span>
          <textarea
            autoFocus
            value={prior}
            onChange={e => setPrior(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) void handleCreate();
            }}
            placeholder="e.g. Explore few-shot examples instead of tool rewrites"
            className="mt-1.5 w-full h-24 px-3 py-2 rounded-md bg-surface-3 border border-hairline text-[13px] text-ink placeholder:text-ink-tertiary resize-none outline-none focus:border-hairline-strong focus:shadow-[0_0_0_2px_rgba(94,106,210,0.5)]"
          />
          <span className="mt-1 block text-[11px] text-ink-tertiary">
            The branch re-runs from this checkpoint with fresh model calls; the original thread is untouched.
          </span>
        </label>

        {error && (
          <div
            data-testid="fork-error"
            className="mb-4 px-3 py-2 rounded-md border border-status-red/40 bg-status-red/10 text-[12px] text-status-red"
          >
            {error}
          </div>
        )}

        <div className="flex justify-end gap-2">
          <button
            onClick={onClose}
            className="h-[30px] px-3 rounded-md bg-surface-3 border border-hairline text-[13px] font-medium text-ink hover:bg-surface-4"
          >
            Cancel
          </button>
          <button
            onClick={handleCreate}
            disabled={!prior.trim() || submitting}
            className="h-[30px] px-3 rounded-md bg-accent text-[13px] font-medium text-white hover:bg-accent-hover disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {submitting ? 'Creating…' : 'Create fork'}
          </button>
        </div>
      </div>
    </div>
  );
}
