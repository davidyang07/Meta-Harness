import type { ForkEvent as ForkEventType } from '@/lib/types';
import { clockTime, middleEllipsis } from '@/lib/format';

export function ForkEventCard({ fork }: { fork: ForkEventType }) {
  return (
    <div className="mb-2 rounded-md bg-surface-2 border border-hairline shadow-[inset_2px_0_0_var(--color-status-purple)] px-4 py-3">
      <div className="flex items-center gap-2">
        <span className="text-[13px] font-medium text-ink">Fork created</span>
        <span className="font-mono text-[11px] text-ink-tertiary" title={fork.timestamp}>
          {clockTime(fork.timestamp)}
        </span>
      </div>
      <dl className="mt-2 grid grid-cols-[56px_minmax(0,1fr)] gap-x-3 gap-y-1 text-[13px]">
        <dt className="text-ink-subtle">From</dt>
        <dd className="min-w-0 text-ink-muted">
          <span className="font-mono text-[12px]">{fork.parentCandidate}</span>
          <span className="text-ink-tertiary"> @ </span>
          <span className="font-mono text-[12px] text-ink-subtle" title={fork.checkpointId}>
            {middleEllipsis(fork.checkpointId)}
          </span>
        </dd>
        <dt className="text-ink-subtle">Prior</dt>
        <dd className="text-ink">{fork.prior}</dd>
        {fork.rationale && (
          <>
            <dt className="text-ink-subtle">Why</dt>
            <dd className="text-ink-muted">{fork.rationale}</dd>
          </>
        )}
      </dl>
    </div>
  );
}
