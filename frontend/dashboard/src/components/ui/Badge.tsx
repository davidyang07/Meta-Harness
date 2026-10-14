import { TONE_TAG, TONE_HEX, type StatusTone } from '@/lib/status';

/** A small status or category tag (frontend/DESIGN.md §6, "Tag"). */
export function Tag({ label, tone, className = '' }: { label: string; tone: StatusTone; className?: string }) {
  return (
    <span
      className={`inline-flex items-center h-5 px-1.5 rounded text-[11px] font-medium leading-none whitespace-nowrap ${TONE_TAG[tone]} ${className}`}
    >
      {label}
    </span>
  );
}

/** A status pill: dot + word on surface-3 (frontend/DESIGN.md §6). */
export function StatusPill({
  label,
  tone,
  pulse = false,
  testId,
}: {
  label: string;
  tone: StatusTone;
  pulse?: boolean;
  testId?: string;
}) {
  return (
    <span
      data-testid={testId}
      className="inline-flex items-center gap-1.5 h-[22px] px-2 rounded bg-surface-3 border border-hairline text-[12px] text-ink-muted whitespace-nowrap"
    >
      <span
        className={`w-1.5 h-1.5 rounded-full ${pulse ? 'animate-pulse-dot' : ''}`}
        style={{ background: TONE_HEX[tone] }}
      />
      {label}
    </span>
  );
}
