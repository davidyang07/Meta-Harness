'use client';
import { useRef, useEffect, useState } from 'react';
import { useDashboard, useDashboardDispatch } from '@/lib/state';
import { FilterBar } from './ui/FilterBar';
import { Badge } from './ui/Badge';
import { PhasePipeline } from './ui/PhasePipeline';
import { ForkEventCard } from './ForkEvent';
import { RUN_LEVEL } from '@/lib/types';
import type { IterationChapter, LogEntry, LogFilter, LogTag, TreeNode } from '@/lib/types';

const TAG_COLORS: Record<LogTag, string> = {
  orient: 'text-[#606888]',
  plan: 'text-blue',
  'tool/read': 'text-green',
  'tool/patch': 'text-green',
  act: 'text-cyan',
  verify: 'text-amber',
  score: 'text-green',
  fail: 'text-red',
  fork: 'text-purple',
  memory: 'text-amber',
  checkpoint: 'text-text-mid',
};

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
  if (chapter.status === 'smoke_failed' || node.status === 'smoke_failed') {
    return (
      <div
        data-testid="chapter-outcome"
        className="mt-2 flex items-center justify-between px-3 py-2 rounded border-l-2 border-amber bg-amber/10"
      >
        <span className="text-[8px] font-semibold uppercase tracking-wide text-amber">
          Smoke failed — never loaded, no score
        </span>
        <span className="text-text-hi text-[11px] font-semibold">—</span>
      </div>
    );
  }
  const won = chapter.status === 'accepted' || chapter.status === 'best';
  return (
    <div
      data-testid="chapter-outcome"
      className={`mt-2 flex items-center justify-between px-3 py-2 rounded border-l-2 ${
        won ? 'border-green bg-green-bg' : 'border-red bg-red-bg'
      }`}
    >
      <span className={`text-[8px] font-semibold uppercase tracking-wide ${won ? 'text-green' : 'text-red'}`}>
        Accuracy — {chapter.status}
      </span>
      <div className="flex items-center gap-1.5">
        <span className="text-text-hi text-[11px] font-semibold">
          {node.scores.accuracy.toFixed(2)}
        </span>
        {node.delta !== null && (
          <span className={`text-[9px] ${node.delta >= 0 ? 'text-cyan' : 'text-red'}`}>
            {node.delta >= 0 ? '+' : ''}
            {node.delta.toFixed(2)}
          </span>
        )}
      </div>
    </div>
  );
}

export function DecisionLog() {
  const { iterations, logEntries, forkEvents, filters, selectedLogLine, run, tree } = useDashboard();
  const dispatch = useDashboardDispatch();
  const scrollRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [expandedLines, setExpandedLines] = useState<Set<string>>(new Set());

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

  const filtersForBar: LogFilter[] = ['all', 'tools', 'verify', 'scores', 'forks'];

  const renderEntry = (entry: LogEntry) => (
    <div key={entry.id} className="animate-fade-in-slow" data-testid="log-entry">
      <div
        className={`flex items-start gap-2 cursor-pointer group ${
          selectedLogLine === entry.id ? 'border-l-2 border-cyan pl-1.5 -ml-1.5' : ''
        }`}
        onClick={() => {
          dispatch({ type: 'SELECT_LOG_LINE', payload: entry.id });
          if (entry.expandable) toggleExpand(entry.id);
        }}
      >
        <span className="text-text-ghost text-[8px] w-12 shrink-0 pt-0.5">{entry.timestamp}</span>
        <span className={`px-1.5 py-0.5 bg-hover rounded text-[7px] font-semibold uppercase tracking-wide shrink-0 ${TAG_COLORS[entry.tag]}`}>
          {entry.tag}
        </span>
        <span className="text-text-hi text-[10px] leading-[1.6]">{entry.text}</span>
        {entry.expandable && (
          <span className="text-text-ghost text-[9px] ml-auto shrink-0">
            {expandedLines.has(entry.id) ? '▾' : '▸'}
          </span>
        )}
      </div>
      {entry.expandable && expandedLines.has(entry.id) && entry.expandedContent && (
        <pre className="mt-1 ml-[60px] p-2 bg-header rounded text-[9px] text-text-mid overflow-x-auto">{entry.expandedContent}</pre>
      )}
    </div>
  );

  return (
    <div className="flex-1 flex flex-col bg-panel rounded overflow-hidden min-h-0">
      <div className="h-11 flex items-center justify-between px-6 bg-header border-b border-border shrink-0">
        <div className="flex items-center gap-3">
          <span className="text-[10px] font-semibold text-text-hi uppercase tracking-wide">Decision Log</span>
          <span className="flex items-center gap-1 text-[8px] text-green font-semibold uppercase">
            <span className="w-1 h-1 rounded-full bg-green" />
            Live
          </span>
        </div>
        <FilterBar
          filters={filtersForBar}
          active={filters.activeFilter}
          onSelect={f => dispatch({ type: 'SET_FILTER', payload: { activeFilter: f } })}
        />
      </div>

      <div ref={scrollRef} onScroll={handleScroll} className="flex-1 overflow-y-auto px-6 py-5">
        {iterations.length === 0 && forkEvents.length === 0 && (
          <div className="rounded border border-border bg-header/70 px-4 py-4 mb-4">
            <div className="text-[10px] uppercase tracking-wide text-text-mid">Awaiting first candidate</div>
            <div className="text-[11px] text-text-hi mt-1">
              {runReportedError
                ? 'No candidate completed: the run reported an error (see run events below).'
                : run?.status === 'running'
                  ? 'Real benchmark run is warming up (proposer + first benchmark pass). Logs will stream in once iteration 1 begins.'
                  : 'No streamed decision events yet for this run.'}
            </div>
          </div>
        )}
        {forkEvents.length > 0 && (
          <div className="mb-3 flex items-center gap-2">
            <span className="text-purple text-[10px] font-semibold uppercase tracking-wide">Fork timeline</span>
            <span className="text-text-ghost text-[10px]">branching decisions and rationale</span>
          </div>
        )}
        {forkEvents.map((fork, i) => (
          <ForkEventCard key={fork.threadId ?? `${fork.branchId}-${i}`} fork={fork} />
        ))}

        {iterations.map((chapter, idx) => {
          const chapterEntries = filteredEntries.filter(e => belongsTo(e, chapter));
          const chapterKey = `${chapter.candidateName}-${chapter.iteration}-${chapter.threadId ?? 'thread'}-${idx}`;
          const node = nodeFor(tree, chapter);
          const memoryStored = logEntries.some(
            e => e.tag === 'memory' && e.candidateName === chapter.candidateName,
          );
          return (
            <div key={chapterKey} className="mb-4 animate-fade-in-slow" data-testid="iteration-chapter">
              <div className={`border-l-2 ${chapter.isForkBranch ? 'border-purple' : 'border-border-active'} pl-3 py-2 mb-2`}>
                <div className="flex items-center gap-2 mb-1">
                  <span className={`text-[10px] font-semibold ${chapter.isForkBranch ? 'text-purple' : 'text-text-hi'}`}>
                    ITER {chapter.iteration}{chapter.isForkBranch ? "'" : ''} — {chapter.candidateName}
                  </span>
                  {chapter.status === 'accepted' && <Badge label="Accepted" color="green" />}
                  {chapter.status === 'best' && <Badge label="Best" color="cyan" />}
                  {chapter.status === 'rejected' && <Badge label="Rejected" color="red" />}
                  {chapter.status === 'smoke_failed' && <Badge label="Smoke failed" color="amber" />}
                </div>
                <PhasePipeline phases={chapter.phases} />
                <p className="text-text-mid text-[10px] italic mt-1 leading-relaxed">{chapter.hypothesis}</p>
                {memoryStored && (
                  <span className="inline-flex items-center gap-1 mt-1 text-[8px] text-amber uppercase tracking-wide">
                    <span className="w-1 h-1 rounded-full bg-amber" />
                    stored as a cross-run memory pattern
                  </span>
                )}
              </div>

              <div className="flex flex-col gap-[8px]">
                {chapterEntries.map(renderEntry)}
              </div>

              {chapter.status !== 'running' && node && <Outcome chapter={chapter} node={node} />}
            </div>
          );
        })}

        {runLevelEntries.length > 0 && (
          <div className="mb-4 animate-fade-in-slow" data-testid="run-events">
            <div className="border-l-2 border-border-active pl-3 py-2 mb-2">
              <span className="text-[10px] font-semibold text-text-hi uppercase tracking-wide">Run events</span>
              <p className="text-text-mid text-[10px] mt-1">
                Errors, checkpoints, forks and cancellations for the run as a whole.
              </p>
            </div>
            <div className="flex flex-col gap-[8px]">{runLevelEntries.map(renderEntry)}</div>
          </div>
        )}

        {!autoScroll && (
          <button
            onClick={() => {
              setAutoScroll(true);
              scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
            }}
            className="fixed bottom-12 left-1/2 -translate-x-1/2 px-4 py-2 bg-header border border-border rounded text-xs text-text-mid hover:text-text-hi"
          >
            {'↓'} Jump to latest
          </button>
        )}
      </div>
    </div>
  );
}
