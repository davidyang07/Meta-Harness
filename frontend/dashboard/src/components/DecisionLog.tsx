'use client';
import { useEffect, useRef, useState } from 'react';
import { useDashboard, useDashboardDispatch } from '@/lib/state';
import { Segmented } from './ui/FilterBar';
import { Tag } from './ui/Badge';
import { PhasePipeline } from './ui/PhasePipeline';
import { ForkEventCard } from './ForkEvent';
import { clockTime, deltaTone, signedDelta } from '@/lib/format';
import { statusMeta, TONE_TEXT, type StatusTone } from '@/lib/status';
import { RUN_LEVEL } from '@/lib/types';
import type { IterationChapter, LogEntry, LogFilter, LogTag, TreeNode } from '@/lib/types';

const TAGS: Record<LogTag, { label: string; tone: StatusTone }> = {
  orient: { label: 'Orient', tone: 'neutral' },
  plan: { label: 'Plan', tone: 'blue' },
  'tool/read': { label: 'Read', tone: 'neutral' },
  'tool/patch': { label: 'Patch', tone: 'purple' },
  act: { label: 'Act', tone: 'blue' },
  verify: { label: 'Verify', tone: 'yellow' },
  score: { label: 'Score', tone: 'green' },
  fail: { label: 'Error', tone: 'red' },
  fork: { label: 'Fork', tone: 'purple' },
  memory: { label: 'Memory', tone: 'yellow' },
  checkpoint: { label: 'Checkpoint', tone: 'neutral' },
};

const FILTERS: readonly LogFilter[] = ['all', 'tools', 'verify', 'scores', 'forks'];

/**
 * Does a log entry belong to this chapter? Entries name their candidate;
 * run-level ones (iteration-complete) name an iteration and a thread
 * instead, and belong to the chapter of that iteration on that thread.
 */
function belongsTo(entry: LogEntry, chapter: IterationChapter): boolean {
  if (entry.candidateName === chapter.candidateName) return true;
  return (
    entry.candidateName === RUN_LEVEL &&
    entry.iteration !== undefined &&
    entry.iteration === chapter.iteration &&
    entry.threadId !== undefined &&
    entry.threadId === chapter.threadId
  );
}

function nodeFor(tree: TreeNode[], chapter: IterationChapter): TreeNode | undefined {
  return (
    tree.find(n => n.candidate === chapter.candidateName && n.threadId === chapter.threadId) ??
    tree.find(n => n.candidate === chapter.candidateName)
  );
}

/** The chapter's verdict, read off the node's recorded status and scores. */
function Outcome({ chapter, node }: { chapter: IterationChapter; node: TreeNode }) {
  const smokeFailed = chapter.status === 'smoke_failed' || node.status === 'smoke_failed';
  const meta = statusMeta(smokeFailed ? 'smoke_failed' : chapter.status);
  return (
    <div
      data-testid="chapter-outcome"
      className="mt-2 flex items-center gap-3 h-9 px-3 rounded-md bg-surface-2 border border-hairline text-[13px]"
    >
      <span className="text-ink-subtle">Accuracy</span>
      {smokeFailed ? (
        <span className="text-ink-tertiary">— never loaded, no score</span>
      ) : (
        <>
          <span className="tabular font-medium text-ink">{node.scores.accuracy.toFixed(2)}</span>
          {node.delta !== null && (
            <span className={`tabular ${TONE_TEXT[deltaTone(node.delta)]}`}>{signedDelta(node.delta)}</span>
          )}
        </>
      )}
      <span className="ml-auto">
        <Tag label={meta.label} tone={meta.tone} />
      </span>
    </div>
  );
}

function EntryRow({
  entry,
  selected,
  expanded,
  onClick,
}: {
  entry: LogEntry;
  selected: boolean;
  expanded: boolean;
  onClick: () => void;
}) {
  const tag = TAGS[entry.tag] ?? TAGS.checkpoint;
  return (
    <div className="animate-fade-in" data-testid="log-entry">
      <div
        onClick={onClick}
        className={`grid grid-cols-[64px_84px_minmax(0,1fr)_12px] items-start gap-2 px-2 py-1 -mx-2 rounded cursor-pointer transition-colors ${
          selected ? 'bg-surface-3' : 'hover:bg-surface-2'
        }`}
      >
        <span className="font-mono text-[11px] leading-5 text-ink-tertiary tabular" title={entry.timestamp}>
          {clockTime(entry.timestamp)}
        </span>
        <span className="leading-5">
          <Tag label={tag.label} tone={tag.tone} />
        </span>
        <span className="text-[13px] leading-5 text-ink-muted break-words">{entry.text}</span>
        {entry.expandable ? (
          <span className="text-ink-tertiary text-[11px] leading-5">{expanded ? '▾' : '▸'}</span>
        ) : (
          <span />
        )}
      </div>
      {entry.expandable && expanded && entry.expandedContent && (
        <pre className="mt-1 mb-2 ml-[156px] p-3 rounded-md bg-surface-2 border border-hairline font-mono text-[12px] text-ink-subtle overflow-x-auto">
          {entry.expandedContent}
        </pre>
      )}
    </div>
  );
}

export function DecisionLog() {
  const { iterations, logEntries, forkEvents, filters, selectedLogLine, run, tree, sseConnected } = useDashboard();
  const dispatch = useDashboardDispatch();
  const scrollRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [expandedLines, setExpandedLines] = useState<Set<string>>(new Set());
  const [showCheckpoints, setShowCheckpoints] = useState(false);

  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logEntries.length, autoScroll]);

  const handleScroll = () => {
    if (!scrollRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    setAutoScroll(scrollHeight - scrollTop - clientHeight < 50);
  };

  const filteredEntries = logEntries.filter(e => {
    if (filters.activeFilter === 'all') return true;
    if (filters.activeFilter === 'tools') return e.tag.startsWith('tool/');
    if (filters.activeFilter === 'verify') return e.tag === 'verify';
    if (filters.activeFilter === 'scores') return e.tag === 'score';
    if (filters.activeFilter === 'forks') return e.tag === 'fork' || e.tag === 'memory';
    return true;
  }).filter(e => !filters.searchQuery || e.text.toLowerCase().includes(filters.searchQuery.toLowerCase()));

  // Whatever no chapter claims is about the run itself — errors,
  // checkpoints, forks, cancellations — and must still be shown. A run
  // that fails before its first candidate has nothing else to show.
  const runLevelEntries = filteredEntries.filter(
    e => !iterations.some(chapter => belongsTo(e, chapter)),
  );
  // Checkpoint writes are many and routine: folded into one line, the way
  // Studio collapses a thread's history until it is asked for.
  const checkpointEntries = runLevelEntries.filter(e => e.tag === 'checkpoint');
  const notableRunEntries = runLevelEntries.filter(e => e.tag !== 'checkpoint');
  const runReportedError = logEntries.some(
    e => e.tag === 'fail' && !iterations.some(chapter => belongsTo(e, chapter)),
  );

  const toggleExpand = (id: string) => {
    setExpandedLines(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const renderEntry = (entry: LogEntry) => (
    <EntryRow
      key={entry.id}
      entry={entry}
      selected={selectedLogLine === entry.id}
      expanded={expandedLines.has(entry.id)}
      onClick={() => {
        dispatch({ type: 'SELECT_LOG_LINE', payload: entry.id });
        if (entry.expandable) toggleExpand(entry.id);
      }}
    />
  );

  return (
    <section className="flex-1 flex flex-col min-h-0 bg-surface-1">
      <div className="h-11 shrink-0 flex items-center justify-between gap-3 px-4 border-b border-hairline">
        <div className="flex items-center gap-2">
          <h2 className="text-[13px] font-medium text-ink">Decision log</h2>
          {sseConnected && (
            <span className="flex items-center gap-1 text-[12px] text-ink-subtle">
              <span className="w-1.5 h-1.5 rounded-full bg-status-green animate-pulse-dot" />
              Live
            </span>
          )}
        </div>
        <Segmented
          ariaLabel="Filter the decision log"
          options={FILTERS}
          active={filters.activeFilter}
          onSelect={f => dispatch({ type: 'SET_FILTER', payload: { activeFilter: f } })}
        />
      </div>

      <div ref={scrollRef} onScroll={handleScroll} className="relative flex-1 overflow-y-auto px-5 py-4">
        {iterations.length === 0 && forkEvents.length === 0 && (
          <div className="mb-4 rounded-md border border-hairline bg-surface-2 px-4 py-3">
            <div className="text-[13px] font-medium text-ink">Waiting for the first candidate</div>
            <div className="mt-1 text-[13px] text-ink-subtle">
              {runReportedError
                ? 'No candidate completed: the run reported an error (see run events below).'
                : run?.status === 'running'
                  ? 'The proposer and the first benchmark pass are running. Entries appear here when iteration 1 begins.'
                  : 'No decision events have been streamed for this run yet.'}
            </div>
          </div>
        )}

        {forkEvents.length > 0 && (
          <div className="mb-5">
            <h3 className="mb-2 text-[12px] font-medium text-ink-subtle">Forks</h3>
            {forkEvents.map((fork, i) => (
              <ForkEventCard key={fork.threadId ?? `${fork.branchId}-${i}`} fork={fork} />
            ))}
          </div>
        )}

        {iterations.map((chapter, idx) => {
          const chapterEntries = filteredEntries.filter(e => belongsTo(e, chapter));
          const chapterKey = `${chapter.candidateName}-${chapter.iteration}-${chapter.threadId ?? 'thread'}-${idx}`;
          const node = nodeFor(tree, chapter);
          const meta = statusMeta(chapter.status);
          const memoryStored = logEntries.some(
            e => e.tag === 'memory' && e.candidateName === chapter.candidateName,
          );
          return (
            <article
              key={chapterKey}
              className="mb-6 animate-fade-in"
              data-testid="iteration-chapter"
            >
              <header className="mb-2">
                <div className="flex items-center gap-2 min-w-0">
                  <span className="text-[12px] text-ink-subtle tabular whitespace-nowrap">
                    Iteration {chapter.iteration}
                  </span>
                  <span className="font-mono text-[13px] font-medium text-ink truncate">{chapter.candidateName}</span>
                  {chapter.isForkBranch && <Tag label="Fork" tone="purple" />}
                  {chapter.status !== 'running' ? (
                    <Tag label={meta.label} tone={meta.tone} />
                  ) : (
                    <Tag label="Running" tone="blue" />
                  )}
                </div>
                <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1">
                  <PhasePipeline phases={chapter.phases} />
                  {memoryStored && (
                    <span className="text-[12px] text-status-yellow">Stored as a cross-run memory pattern</span>
                  )}
                </div>
                {chapter.hypothesis && (
                  <p className="mt-1.5 text-[13px] text-ink-subtle leading-5">{chapter.hypothesis}</p>
                )}
              </header>

              <div className="flex flex-col border-l border-hairline pl-3">
                {chapterEntries.map(renderEntry)}
              </div>

              {chapter.status !== 'running' && node && <Outcome chapter={chapter} node={node} />}
            </article>
          );
        })}

        {runLevelEntries.length > 0 && (
          <section className="mb-4 animate-fade-in" data-testid="run-events">
            <header className="mb-2">
              <h3 className="text-[13px] font-medium text-ink">Run events</h3>
              <p className="mt-0.5 text-[12px] text-ink-subtle">
                Errors, forks and cancellations for the run as a whole.
              </p>
            </header>
            <div className="flex flex-col border-l border-hairline pl-3">
              {notableRunEntries.map(renderEntry)}
              {checkpointEntries.length > 0 && (
                <>
                  <button
                    type="button"
                    onClick={() => setShowCheckpoints(v => !v)}
                    className="self-start my-1 text-[12px] text-ink-subtle hover:text-ink"
                    aria-expanded={showCheckpoints}
                  >
                    {showCheckpoints ? '▾' : '▸'} {checkpointEntries.length}{' '}
                    {checkpointEntries.length === 1 ? 'checkpoint' : 'checkpoints'} written
                  </button>
                  {showCheckpoints && checkpointEntries.map(renderEntry)}
                </>
              )}
            </div>
          </section>
        )}

        {!autoScroll && (
          <button
            onClick={() => {
              setAutoScroll(true);
              scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
            }}
            className="sticky bottom-2 left-1/2 -translate-x-1/2 h-[30px] px-3 rounded-md bg-surface-3 border border-hairline-strong text-[13px] text-ink-muted hover:text-ink"
          >
            ↓ Jump to latest
          </button>
        )}
      </div>
    </section>
  );
}
