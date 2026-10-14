'use client';
import { DiffEditor, type BeforeMount } from '@monaco-editor/react';

/** Monaco in the dashboard's palette (frontend/DESIGN.md §2). */
const defineTheme: BeforeMount = monaco => {
  monaco.editor.defineTheme('meta-harness', {
    base: 'vs-dark',
    inherit: true,
    rules: [
      { token: 'comment', foreground: '62666d' },
      { token: 'string', foreground: 'c9b8f5' },
      { token: 'keyword', foreground: '9aa4f0' },
      { token: 'number', foreground: 'f2c94c' },
    ],
    colors: {
      'editor.background': '#141516',
      'editor.foreground': '#d0d6e0',
      'editorLineNumber.foreground': '#62666d',
      'editorLineNumber.activeForeground': '#8a8f98',
      'editor.lineHighlightBackground': '#18191a',
      'editorGutter.background': '#141516',
      'diffEditor.insertedTextBackground': '#4cb78226',
      'diffEditor.removedTextBackground': '#eb575726',
      'diffEditor.insertedLineBackground': '#4cb78214',
      'diffEditor.removedLineBackground': '#eb575714',
      'diffEditor.diagonalFill': '#23252a',
      'scrollbarSlider.background': '#34343a80',
    },
  });
};

export function DiffViewer({ diff }: { diff: string }) {
  // Parse the unified diff to extract original and modified content
  const lines = diff.split('\n');
  const original: string[] = [];
  const modified: string[] = [];

  for (const line of lines) {
    if (line.startsWith('---') || line.startsWith('+++') || line.startsWith('@@')) continue;
    if (line.startsWith('-')) {
      original.push(line.slice(1));
    } else if (line.startsWith('+')) {
      modified.push(line.slice(1));
    } else {
      original.push(line.startsWith(' ') ? line.slice(1) : line);
      modified.push(line.startsWith(' ') ? line.slice(1) : line);
    }
  }

  // Inline mode shows removed and added lines one after the other, and
  // long lines wrap, so size for both sides with some room to spare.
  const lineCount = Math.max(original.length + modified.length, 8);
  const height = Math.min(Math.max(lineCount * 22 + 24, 200), 720);

  return (
    <div className="rounded-md border border-hairline overflow-hidden">
      <DiffEditor
        height={`${height}px`}
        language="python"
        original={original.join('\n')}
        modified={modified.join('\n')}
        theme="meta-harness"
        beforeMount={defineTheme}
        options={{
          readOnly: true,
          renderSideBySide: false,
          minimap: { enabled: false },
          scrollBeyondLastLine: false,
          fontSize: 12,
          fontFamily: "var(--font-jetbrains), 'JetBrains Mono', monospace",
          lineNumbers: 'on',
          renderOverviewRuler: false,
          overviewRulerBorder: false,
          wordWrap: 'on',
          scrollbar: { verticalScrollbarSize: 6, horizontalScrollbarSize: 6 },
        }}
      />
    </div>
  );
}
