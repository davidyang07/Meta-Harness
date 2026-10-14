'use client';
import Link from 'next/link';
import { useDashboard } from '@/lib/state';
import { useAuthStatus } from '@/lib/auth-client';
import { runStatusMeta } from '@/lib/status';
import { StatusPill, Tag } from './ui/Badge';

/** The lavender brand glyph: the only decorative use of the accent. */
export function BrandMark() {
  return (
    <span aria-hidden className="inline-flex items-center justify-center w-5 h-5 rounded-[5px] bg-accent">
      <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
        <path d="M2 2.5h3v3H2zM7 6.5h3v3H7z" fill="white" />
        <path d="M3.5 5.5v2.5h3.5" stroke="white" strokeWidth="1.2" />
      </svg>
    </span>
  );
}

export function TopBar({ onReplay }: { onReplay?: () => void }) {
  const { run, mode } = useDashboard();
  const { enabled: authEnabled, profile } = useAuthStatus();
  const status = runStatusMeta(run?.status);
  const synthetic = run?.metricsSource === 'mock' || mode === 'mock';

  return (
    <header className="h-12 shrink-0 flex items-center justify-between gap-4 px-4 bg-canvas border-b border-hairline">
      <nav className="flex items-center gap-2 min-w-0 text-[13px]">
        <Link href="/" className="flex items-center gap-2 font-semibold text-ink hover:text-white">
          <BrandMark />
          Meta-Harness
        </Link>
        <span className="text-ink-tertiary">/</span>
        <Link href="/" className="text-ink-subtle hover:text-ink">Runs</Link>
        {run && (
          <>
            <span className="text-ink-tertiary">/</span>
            <span className="font-mono text-[12px] text-ink truncate" title={run.runId}>{run.runId}</span>
          </>
        )}
        {run && (
          <span className="ml-2 flex items-center gap-2">
            <StatusPill label={status.label} tone={status.tone} pulse={status.tone === 'blue'} />
            <span className="text-[12px] text-ink-subtle tabular">Iteration {run.iteration}</span>
            {synthetic && <Tag label="Mock data" tone="yellow" />}
          </span>
        )}
      </nav>
      <div className="flex items-center gap-3 shrink-0">
        {onReplay && (
          <button
            onClick={onReplay}
            className="h-[30px] px-3 rounded-md bg-surface-3 border border-hairline text-[13px] font-medium text-ink hover:bg-surface-4 transition-colors"
          >
            Replay run
          </button>
        )}
        {profile && (
          <span className="text-[12px] text-ink-subtle truncate max-w-[220px]">
            {profile.name ?? profile.email ?? 'Signed in'}
          </span>
        )}
        {/* Without Auth0 there are no /auth routes; a link would 404. */}
        {authEnabled && (
          <a
            href={profile ? '/auth/logout' : '/auth/login'}
            className="text-[13px] text-ink-subtle hover:text-ink transition-colors"
          >
            {profile ? 'Log out' : 'Log in'}
          </a>
        )}
      </div>
    </header>
  );
}
