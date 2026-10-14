'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  API_BASE_URL,
  ApiUnauthorizedError,
  createRun,
  listRuns,
  isBackendAvailable,
  type RunListItem,
} from '@/lib/api';
import { useAuthStatus } from '@/lib/auth-client';
import { runStatusMeta } from '@/lib/status';
import { BrandMark } from '@/components/TopBar';
import { StatusPill } from '@/components/ui/Badge';
import { Segmented } from '@/components/ui/FilterBar';

const DEMO_RUN = 'demo-2026-04-25';

type PresetSuite = {
  id: string;
  label: string;
  description: string;
  payload: { budget: number; trials: number; workers: number; fresh: boolean };
};

const PRESET_SUITES: PresetSuite[] = [
  {
    id: 'quick-smoke',
    label: 'Quick smoke',
    description: '1 iteration, 1 trial per task. Confirms the wiring.',
    payload: { budget: 1, trials: 1, workers: 1, fresh: true },
  },
  {
    id: 'balanced',
    label: 'Balanced',
    description: '3 iterations, 2 trials per task. A realistic local signal.',
    payload: { budget: 3, trials: 2, workers: 2, fresh: true },
  },
  {
    id: 'stress',
    label: 'Stress',
    description: '5 iterations, 5 trials per task. Heavier event volume.',
    payload: { budget: 5, trials: 5, workers: 3, fresh: true },
  },
];

type Proposer = 'mock' | 'claude';
type Bench = 'mock' | 'measured';

function relativeTime(iso: string | null | undefined): string {
  if (!iso) return '—';
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return '—';
  const minutes = Math.round((Date.now() - then) / 60000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  return new Date(iso).toLocaleDateString();
}

export default function RunsPage() {
  const router = useRouter();
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [live, setLive] = useState<boolean | null>(null);
  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState<string | null>(null);
  const [presetId, setPresetId] = useState<string>(PRESET_SUITES[1].id);
  const [proposer, setProposer] = useState<Proposer>('mock');
  const [bench, setBench] = useState<Bench>('mock');
  const [authError, setAuthError] = useState<string | null>(null);
  const { enabled: authEnabled, profile } = useAuthStatus();

  useEffect(() => {
    let cancelled = false;
    isBackendAvailable().then(ok => {
      if (cancelled) return;
      setLive(ok);
      if (!ok) return;
      listRuns()
        .then(list => {
          if (!cancelled) setRuns(list);
        })
        .catch(error => {
          if (cancelled) return;
          setRuns([]);
          if (error instanceof ApiUnauthorizedError) setAuthError(error.message);
        });
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const preset = PRESET_SUITES.find(item => item.id === presetId) ?? PRESET_SUITES[0];

  const handleLaunch = async () => {
    if (launching) return;
    setLaunchError(null);
    setLaunching(true);
    try {
      const run = await createRun({
        run_name: `preset-${preset.id}-${Date.now()}`,
        ...preset.payload,
        proposer,
        mock_bench: bench === 'mock',
      });
      router.push(`/runs/${run.run_id}`);
    } catch (error) {
      setLaunchError(error instanceof Error ? error.message : 'failed to start the run');
      setLaunching(false);
    }
  };

  return (
    <div className="h-full flex flex-col bg-canvas overflow-hidden">
      <header className="h-12 shrink-0 flex items-center justify-between px-4 border-b border-hairline">
        <nav className="flex items-center gap-2 text-[13px]">
          <Link href="/" className="flex items-center gap-2 font-semibold text-ink">
            <BrandMark />
            Meta-Harness
          </Link>
          <span className="text-ink-tertiary">/</span>
          <span className="text-ink-subtle">Runs</span>
        </nav>
        <div className="flex items-center gap-3 text-[13px]">
          {live !== null && (
            <StatusPill label={live ? 'Backend connected' : 'Backend offline'} tone={live ? 'green' : 'red'} />
          )}
          {/* Without Auth0 there are no /auth routes; the links would 404. */}
          {authEnabled && (
            <a href={profile ? '/auth/logout' : '/auth/login'} className="text-ink-subtle hover:text-ink">
              {profile ? 'Log out' : 'Log in'}
            </a>
          )}
        </div>
      </header>

      <main className="flex-1 overflow-y-auto">
        <div className="max-w-[1040px] mx-auto px-8 py-10">
          <div className="mb-8">
            <h1 className="text-[20px] font-semibold tracking-[-0.3px] text-ink">Runs</h1>
            <p className="mt-1 text-[13px] text-ink-subtle max-w-[640px]">
              Each run evolves a coding-agent harness: a proposer rewrites the agent, every candidate is
              benchmarked on frozen tasks, and every step is checkpointed so you can fork from it.
            </p>
          </div>

          {authError && (
            <div data-testid="auth-error" className="mb-6 rounded-md border border-status-red/40 bg-status-red/10 px-3 py-2 text-[13px] text-status-red">
              {authError}
            </div>
          )}

          <section className="mb-8 rounded-lg border border-hairline bg-surface-1">
            <div className="px-5 pt-4 pb-3 border-b border-hairline">
              <h2 className="text-[13px] font-medium text-ink">Start a run</h2>
            </div>
            <div className="px-5 py-4 grid gap-5">
              <div className="flex flex-wrap items-center gap-x-8 gap-y-3">
                <div className="flex items-center gap-3">
                  <span className="text-[12px] text-ink-subtle w-[70px]">Proposer</span>
                  <Segmented<Proposer>
                    ariaLabel="Proposer"
                    options={['mock', 'claude']}
                    labels={{ mock: 'Mock', claude: 'Claude CLI' }}
                    active={proposer}
                    onSelect={setProposer}
                  />
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-[12px] text-ink-subtle w-[70px]">Benchmark</span>
                  <Segmented<Bench>
                    ariaLabel="Benchmark"
                    options={['mock', 'measured']}
                    labels={{ mock: 'Mock scores', measured: 'Measured' }}
                    active={bench}
                    onSelect={setBench}
                  />
                </div>
              </div>

              <div role="radiogroup" aria-label="Run size" className="grid grid-cols-1 md:grid-cols-3 gap-2">
                {PRESET_SUITES.map(item => {
                  const selected = item.id === presetId;
                  return (
                    <button
                      key={item.id}
                      role="radio"
                      aria-checked={selected}
                      onClick={() => setPresetId(item.id)}
                      className={`text-left rounded-md border px-3 py-2.5 transition-colors ${
                        selected
                          ? 'border-hairline-strong bg-surface-3 shadow-[inset_2px_0_0_var(--color-accent)]'
                          : 'border-hairline bg-surface-2 hover:bg-surface-3'
                      }`}
                    >
                      <div className="text-[13px] font-medium text-ink">{item.label}</div>
                      <div className="mt-0.5 text-[12px] text-ink-subtle">{item.description}</div>
                    </button>
                  );
                })}
              </div>

              <div className="flex flex-wrap items-center justify-between gap-3">
                <p className="text-[12px] text-ink-subtle">
                  {bench === 'mock'
                    ? 'Mock scores are synthetic and free; the run is labelled “Mock data” everywhere.'
                    : 'Measured runs call the inner-loop model for every trial and cost real money.'}
                  {proposer === 'claude' && ' The Claude CLI proposer needs the claude command and credentials.'}
                </p>
                <button
                  onClick={() => void handleLaunch()}
                  disabled={launching || live === false}
                  className="h-[30px] px-3 rounded-md bg-accent text-[13px] font-medium text-white hover:bg-accent-hover disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  {launching ? 'Starting…' : 'Start run'}
                </button>
              </div>
              {launchError && <div className="text-[13px] text-status-red">{launchError}</div>}
            </div>
          </section>

          <section className="rounded-lg border border-hairline bg-surface-1 overflow-hidden">
            <div className="px-5 pt-4 pb-3 border-b border-hairline flex items-center justify-between">
              <h2 className="text-[13px] font-medium text-ink">Recent runs</h2>
              {runs.length > 0 && <span className="text-[12px] text-ink-subtle tabular">{runs.length}</span>}
            </div>

            {live === false && (
              <div className="px-5 py-6 text-[13px] text-ink-subtle">
                No backend answered at <span className="font-mono text-ink-muted">{API_BASE_URL}</span>. Start it with{' '}
                <span className="font-mono text-ink-muted">uv run meta-harness serve</span>, or{' '}
                <Link href={`/runs/${DEMO_RUN}`} className="text-ink underline decoration-hairline-strong underline-offset-2 hover:decoration-accent">
                  open the offline demo
                </Link>
                .
              </div>
            )}
            {live && runs.length === 0 && !authError && (
              <div className="px-5 py-6 text-[13px] text-ink-subtle">No runs yet. Start one above.</div>
            )}
            {live === null && <div className="px-5 py-6 text-[13px] text-ink-subtle">Connecting…</div>}

            {runs.length > 0 && (
              <table className="w-full text-[13px]">
                <thead>
                  <tr className="text-[12px] text-ink-subtle text-left border-b border-hairline">
                    <th className="font-normal px-5 py-2">Run</th>
                    <th className="font-normal px-3 py-2">Status</th>
                    <th className="font-normal px-3 py-2 text-right">Iteration</th>
                    <th className="font-normal px-3 py-2 text-right">Best score</th>
                    <th className="font-normal px-5 py-2 text-right">Started</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map(run => {
                    const status = runStatusMeta(run.status);
                    return (
                      <tr
                        key={run.run_id}
                        onClick={() => router.push(`/runs/${run.run_id}`)}
                        className="border-b border-hairline last:border-0 hover:bg-surface-2 cursor-pointer"
                      >
                        <td className="px-5 py-2.5">
                          <Link href={`/runs/${run.run_id}`} className="font-mono text-[12px] text-ink hover:underline">
                            {run.run_id}
                          </Link>
                        </td>
                        <td className="px-3 py-2.5">
                          <StatusPill label={status.label} tone={status.tone} pulse={status.tone === 'blue'} />
                        </td>
                        <td className="px-3 py-2.5 text-right tabular text-ink-muted">{run.iteration}</td>
                        <td className="px-3 py-2.5 text-right tabular text-ink">
                          {run.best_score === null || run.best_score === undefined ? '—' : run.best_score.toFixed(2)}
                        </td>
                        <td className="px-5 py-2.5 text-right text-ink-subtle">{relativeTime(run.started_at)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}
