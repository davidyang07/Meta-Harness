'use client';

import { useEffect, useCallback, useRef } from 'react';
import { useParams } from 'next/navigation';
import { DashboardProvider, useDashboardDispatch } from '@/lib/state';
import { startSSE, startMockSSE, refreshBranches } from '@/lib/sse';
import {
  ApiUnauthorizedError,
  fetchCheckpointCandidateMap,
  getRunDetail,
  isBackendAvailable,
  onUnauthorized,
  runErrorOf,
  toRunInfo,
  toTreeNodesFromRunDetail,
} from '@/lib/api';
import { TopBar } from '@/components/TopBar';
import { TrajectoryTree } from '@/components/TrajectoryTree';
import { DecisionLog } from '@/components/DecisionLog';
import { ContextPanel } from '@/components/ContextPanel';
import { StatusBar } from '@/components/StatusBar';

function DashboardShell() {
  const params = useParams<{ run_id: string }>();
  const runId = params.run_id;
  const dispatch = useDashboardDispatch();
  const latestDetailRef = useRef<Awaited<ReturnType<typeof getRunDetail>> | null>(null);
  const replayTimersRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  const clearReplayTimers = useCallback(() => {
    for (const timer of replayTimersRef.current) clearTimeout(timer);
    replayTimersRef.current = [];
  }, []);

  const replayRunProgression = useCallback(() => {
    const detail = latestDetailRef.current;
    if (!detail) return;
    clearReplayTimers();

    const schedule = (delayMs: number, action: () => void) => {
      const timer = setTimeout(action, delayMs);
      replayTimersRef.current.push(timer);
    };

    const runInfo = toRunInfo(detail);
    const nodes = toTreeNodesFromRunDetail(detail).sort((a, b) => a.iteration - b.iteration);

    dispatch({ type: 'RESET' });
    dispatch({ type: 'SET_MODE', payload: 'live' });
    dispatch({ type: 'SET_RUN', payload: { ...runInfo, iteration: 0 } });
    dispatch({ type: 'SET_SSE_CONNECTED', payload: true });

    nodes.forEach((node, idx) => {
      schedule(900 + idx * 2200, () => {
        dispatch({ type: 'ADD_TREE_NODE', payload: node });
        if (node.checkpointId) {
          dispatch({
            type: 'SET_CHECKPOINT_ID',
            payload: { candidate: node.candidate, checkpointId: node.checkpointId },
          });
        }
        dispatch({
          type: 'ADD_ITERATION',
          payload: {
            iteration: node.iteration,
            candidateName: node.candidate,
            status: node.status === 'best' ? 'accepted' : node.status,
            phases: { propose: true, validate: true, benchmark: true, frontier: true },
            hypothesis: node.hypothesis ?? `candidate ${node.candidate}`,
            isForkBranch: node.isForkBranch,
            threadId: node.threadId,
          },
        });
        dispatch({
          type: 'ADD_LOG_ENTRY',
          payload: {
            id: `replay-${node.threadId ?? ''}-${node.candidate}-${node.iteration}`,
            timestamp: new Date().toISOString(),
            tag: node.status === 'smoke_failed' ? 'fail' : 'score',
            text:
              node.status === 'smoke_failed'
                ? `iter ${node.iteration} ${node.candidate} smoke failed (no score)`
                : `iter ${node.iteration} ${node.candidate} accuracy ${(node.scores.accuracy * 100).toFixed(0)}%`,
            candidateName: node.candidate,
            threadId: node.threadId,
          },
        });
      });
    });
  }, [clearReplayTimers, dispatch]);

  const connect = useCallback(
    async (signal: AbortSignal): Promise<(() => void) | undefined> => {
      const live = await isBackendAvailable();
      // Every await is a point where the page may have unmounted (or, in
      // StrictMode, been mounted twice). Nothing past one may act for a
      // view that no longer exists.
      if (signal.aborted) return undefined;
      dispatch({ type: 'SET_MODE', payload: live ? 'live' : 'mock' });

      if (live) {
        try {
          const detail = await getRunDetail(runId);
          if (signal.aborted) return undefined;
          latestDetailRef.current = detail;
          const runInfo = toRunInfo(detail);
          // `mode` is transport ("is a backend answering"), NOT provenance.
          // Conflating them made a live mock-bench run fall back to the
          // offline fixtures and render fabricated diff/test text as if it
          // were the run's own. Provenance lives on run.metricsSource and
          // is surfaced by the status bar.
          dispatch({ type: 'SET_RUN', payload: runInfo });
          const runError = runErrorOf(detail);
          if (runInfo.status === 'failed' && runError) {
            dispatch({ type: 'SET_ERROR', payload: `run failed: ${runError}` });
          }
          for (const node of toTreeNodesFromRunDetail(detail)) {
            dispatch({ type: 'ADD_TREE_NODE', payload: node });
          }
        } catch (error) {
          if (signal.aborted) return undefined;
          dispatch({ type: 'SET_SSE_CONNECTED', payload: false });
          dispatch({
            type: 'SET_ERROR',
            payload: error instanceof Error ? error.message : `run ${runId} unavailable`,
          });
          return undefined;
        }

        // Checkpoint history is an enrichment, not a precondition: a
        // backend running with in-memory persistence 404s here, and that
        // must not cost the run its branches and its live stream.
        try {
          const checkpoints = await fetchCheckpointCandidateMap(runId);
          if (signal.aborted) return undefined;
          for (const [candidate, checkpointId] of checkpoints) {
            dispatch({ type: 'SET_CHECKPOINT_ID', payload: { candidate, checkpointId } });
          }
        } catch (error) {
          if (signal.aborted) return undefined;
          if (error instanceof ApiUnauthorizedError) {
            dispatch({ type: 'SET_ERROR', payload: error.message });
            return undefined;
          }
        }

        // Reconstructed from backend artifacts, so a page refresh
        // rebuilds the branch tree rather than losing it.
        await refreshBranches(runId, dispatch);
        if (signal.aborted) return undefined;
        return startSSE(runId, dispatch);
      }

      if (runId === 'demo-2026-04-25') {
        return startMockSSE(dispatch);
      }

      // No backend and not the built-in demo: show an explicit empty
      // state rather than fixture data dressed up as this run.
      dispatch({ type: 'SET_SSE_CONNECTED', payload: false });
      dispatch({
        type: 'SET_ERROR',
        payload: `backend unavailable — cannot load run ${runId}`,
      });
      return undefined;
    },
    [runId, dispatch],
  );

  // A 401 anywhere — including calls that degrade to an empty state —
  // is a configuration problem the user has to see.
  useEffect(
    () => onUnauthorized(message => dispatch({ type: 'SET_ERROR', payload: message })),
    [dispatch],
  );

  useEffect(() => {
    const controller = new AbortController();
    let cleanup: (() => void) | undefined;
    void connect(controller.signal).then(fn => {
      // Cleaned up while connect() was still awaiting (StrictMode's
      // double mount, or leaving during the health probe): close what it
      // opened instead of leaking a second EventSource that duplicates
      // every event.
      if (controller.signal.aborted) {
        fn?.();
        return;
      }
      cleanup = fn;
    });
    return () => {
      controller.abort();
      cleanup?.();
      cleanup = undefined;
      clearReplayTimers();
    };
  }, [clearReplayTimers, connect]);

  return (
    <div className="h-full flex flex-col bg-panel overflow-hidden">
      <TopBar onReplay={replayRunProgression} />
      <div className="flex-1 flex gap-4 p-4 min-h-0 w-full">
        <div className="w-[220px] shrink-0 flex flex-col min-h-0 min-w-0">
          <TrajectoryTree />
        </div>
        <div className="flex-[4] flex flex-col min-h-0 min-w-0">
          <DecisionLog />
        </div>
        <div className="flex-[3] flex flex-col min-h-0 min-w-0">
          <ContextPanel />
        </div>
      </div>
      <StatusBar />
    </div>
  );
}

export default function RunPage() {
  return (
    <DashboardProvider>
      <DashboardShell />
    </DashboardProvider>
  );
}
