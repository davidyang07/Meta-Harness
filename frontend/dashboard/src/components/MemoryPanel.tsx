'use client';
import { useEffect, useState } from 'react';
import { listMemory } from '@/lib/api';
import { useDashboard } from '@/lib/state';
import { clockTime } from '@/lib/format';
import { Tag } from './ui/Badge';
import type { MemoryEntry } from '@/lib/types';

function PatternCard({ text, meta }: { text: string; meta: string }) {
  return (
    <div className="rounded-md bg-surface-2 border border-hairline px-3 py-2.5">
      <div className="text-[13px] text-ink-muted leading-5">{text}</div>
      <div className="mt-1 font-mono text-[11px] text-ink-tertiary">{meta}</div>
    </div>
  );
}

export function MemoryPanel() {
  const { logEntries, mode } = useDashboard();
  const [storedPatterns, setStoredPatterns] = useState<MemoryEntry[]>([]);

  const memoryEntries = logEntries.filter(e => e.tag === 'memory');
  const totalPatterns = storedPatterns.length + memoryEntries.length;

  useEffect(() => {
    if (mode !== 'live') return;
    let cancelled = false;
    listMemory('coding-agent', 20)
      .then(entries => {
        if (!cancelled) setStoredPatterns(entries);
      })
      .catch(() => {
        if (!cancelled) setStoredPatterns([]);
      });
    return () => {
      cancelled = true;
    };
  }, [mode]);

  return (
    <div className="flex flex-col gap-5">
      <div>
        <div className="text-[13px] font-medium text-ink">Cross-run memory</div>
        <p className="mt-0.5 text-[12px] text-ink-subtle">
          Patterns from measured, accepted candidates that seed the next run&apos;s proposer.
          {' '}{totalPatterns} {totalPatterns === 1 ? 'pattern' : 'patterns'}.
        </p>
      </div>

      {storedPatterns.length > 0 && (
        <section>
          <h3 className="mb-2 text-[12px] text-ink-subtle">From previous runs</h3>
          <div className="flex flex-col gap-2">
            {storedPatterns.map(p => (
              <PatternCard
                key={p.key}
                text={p.pattern}
                meta={[
                  p.evidence_run_ids?.join(', ') ?? p.mechanism_axis ?? 'memory',
                  p.created_at ? new Date(p.created_at).toLocaleString() : null,
                ].filter(Boolean).join(' · ')}
              />
            ))}
          </div>
        </section>
      )}

      {memoryEntries.length > 0 && (
        <section>
          <h3 className="mb-2 flex items-center gap-2 text-[12px] text-ink-subtle">
            This run {mode === 'mock' && <Tag label="Offline demo" tone="yellow" />}
          </h3>
          <div className="flex flex-col gap-2">
            {memoryEntries.map(e => (
              <PatternCard key={e.id} text={e.text} meta={`${clockTime(e.timestamp)} · ${e.candidateName}`} />
            ))}
          </div>
        </section>
      )}

      {totalPatterns === 0 && (
        <div className="text-[13px] text-ink-subtle">
          No patterns yet. A mock run never writes memory; a measured run stores one when a candidate is accepted.
        </div>
      )}
    </div>
  );
}
