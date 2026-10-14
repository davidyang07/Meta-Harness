'use client';
import Link from 'next/link';
import { useDashboard } from '@/lib/state';
import { useAuthStatus } from '@/lib/auth-client';

export function TopBar({ onReplay }: { onReplay?: () => void }) {
  const { run, mode } = useDashboard();
  const { enabled: authEnabled, profile } = useAuthStatus();

  return (
    <div className="h-12 flex items-center justify-between px-6 bg-header border-b border-border">
      <Link href="/" className="text-cyan text-sm font-semibold tracking-[3px] uppercase hover:text-text-hi transition-colors">META-HARNESS</Link>
      <div className="flex items-center gap-4">
        {onReplay && (
          <button
            onClick={onReplay}
            className="text-[10px] text-amber uppercase tracking-wide border border-border px-2 py-1 rounded hover:text-text-hi hover:border-amber/40 transition-colors"
          >
            Replay run
          </button>
        )}
        {run && (
          <span className="text-[10px] text-text-mid uppercase tracking-wide">
            {run.isMock ? 'mock run' : mode} · {run.status} · iter {run.iteration}
          </span>
        )}
        {profile && (
          <span className="text-[10px] text-text-mid truncate max-w-[220px]">
            {profile.name ?? profile.email ?? 'authenticated user'}
          </span>
        )}
        {/* Without Auth0 there are no /auth routes; a link would 404. */}
        {authEnabled && (
          <a
            href={profile ? '/auth/logout' : '/auth/login'}
            className="text-[10px] text-cyan uppercase tracking-wide hover:text-text-hi transition-colors"
          >
            {profile ? 'Logout' : 'Login'}
          </a>
        )}
      </div>
    </div>
  );
}
