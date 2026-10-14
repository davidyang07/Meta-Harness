'use client';

export function TestOutput({ output }: { output: string }) {
  const lines = output.split('\n');

  return (
    <pre className="overflow-auto rounded-md bg-surface-2 border border-hairline p-3 font-mono text-[12px] leading-[1.6]">
      {lines.map((line, i) => {
        let color = 'text-ink-subtle';
        if (/\bPASS(ED)?\b/.test(line) || / passed\b/.test(line)) color = 'text-status-green';
        if (/\bFAIL(ED)?\b|Error/.test(line) || / failed\b/.test(line)) color = 'text-status-red';
        if (/\bFLAKY\b/.test(line)) color = 'text-status-yellow';
        return <div key={i} className={color}>{line || ' '}</div>;
      })}
    </pre>
  );
}
