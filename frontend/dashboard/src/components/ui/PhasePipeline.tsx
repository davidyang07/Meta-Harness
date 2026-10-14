type Props = { phases: { propose: boolean; validate: boolean; benchmark: boolean; frontier: boolean } };

const STEPS = [
  ['propose', 'Propose'],
  ['validate', 'Validate'],
  ['benchmark', 'Benchmark'],
  ['frontier', 'Frontier'],
] as const;

/** The outer loop's four steps for one iteration, done or pending. */
export function PhasePipeline({ phases }: Props) {
  return (
    <div className="flex items-center gap-1.5 text-[12px]">
      {STEPS.map(([key, label], i) => (
        <span key={key} className="flex items-center gap-1.5">
          {i > 0 && <span className="text-ink-tertiary">›</span>}
          <span className={phases[key] ? 'text-ink-subtle' : 'text-ink-tertiary'}>
            <span className={phases[key] ? 'text-status-green' : ''}>{phases[key] ? '✓' : '○'}</span> {label}
          </span>
        </span>
      ))}
    </div>
  );
}
