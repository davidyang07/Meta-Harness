'use client';

type SegmentedProps<T extends string> = {
  options: readonly T[];
  active: T;
  onSelect: (value: T) => void;
  labels?: Partial<Record<T, string>>;
  ariaLabel: string;
};

/** A segmented control: log filters, inspector tabs (frontend/DESIGN.md §6). */
export function Segmented<T extends string>({ options, active, onSelect, labels, ariaLabel }: SegmentedProps<T>) {
  return (
    <div role="tablist" aria-label={ariaLabel} className="flex items-center gap-0.5 p-0.5 rounded-md bg-surface-3 border border-hairline">
      {options.map(option => {
        const selected = active === option;
        return (
          <button
            key={option}
            role="tab"
            aria-selected={selected}
            onClick={() => onSelect(option)}
            className={`h-6 px-2.5 rounded text-[12px] font-medium transition-colors ${
              selected
                ? 'bg-surface-4 text-ink shadow-[inset_0_0_0_1px_var(--color-hairline-strong)]'
                : 'text-ink-subtle hover:text-ink'
            }`}
          >
            {labels?.[option] ?? option.charAt(0).toUpperCase() + option.slice(1)}
          </button>
        );
      })}
    </div>
  );
}
