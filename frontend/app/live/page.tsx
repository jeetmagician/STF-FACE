"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, getRecentCaptures } from "@/lib/api";
import type { CaptureLogEntry } from "@/lib/types";

const POLL_INTERVAL_MS = 3000;

function confidenceTone(key: string): string {
  if (key === "very_high" || key === "high") return "text-signal-high";
  if (key === "moderate") return "text-signal-mid";
  return "text-signal-low";
}

function timeAgo(epochSeconds: number, now: number): string {
  const seconds = Math.max(0, Math.round(now / 1000 - epochSeconds));
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  return `${hours}h ago`;
}

export default function LivePage() {
  const [captures, setCaptures] = useState<CaptureLogEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const latestSeenId = useRef<string | null>(null);
  const [flashId, setFlashId] = useState<string | null>(null);

  const poll = useCallback(() => {
    getRecentCaptures(15)
      .then((entries) => {
        setCaptures(entries);
        setError(null);
        const newest = entries[0]?.id ?? null;
        if (newest && newest !== latestSeenId.current) {
          if (latestSeenId.current !== null) {
            setFlashId(newest);
            window.setTimeout(() => setFlashId(null), 1600);
          }
          latestSeenId.current = newest;
        }
      })
      .catch((cause: unknown) => {
        setError(
          cause instanceof ApiError
            ? cause.message
            : "Could not reach the analysis service.",
        );
      });
  }, []);

  useEffect(() => {
    poll();
    const interval = window.setInterval(poll, POLL_INTERVAL_MS);
    return () => window.clearInterval(interval);
  }, [poll]);

  useEffect(() => {
    const tick = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(tick);
  }, []);

  return (
    <div className="mx-auto max-w-4xl px-6 py-12">
      <p className="label-mono">Module 03 · Device feed</p>
      <h1 className="mt-1 text-2xl font-semibold tracking-tight text-ink-100">
        Live capture
      </h1>
      <p className="prose-note mt-2 max-w-2xl">
        Results from an unattended capture device (e.g. a button-triggered
        camera) posting to <code className="text-ink-300">/api/database-search/device-capture</code>.
        This page polls every few seconds - nothing here is pushed to you.
      </p>

      <div className="mt-4 flex items-center gap-2">
        <span className="chip !text-scan-400 !border-scan-500/25">
          <span className="status-dot animate-blink" />
          Polling every {POLL_INTERVAL_MS / 1000}s
        </span>
      </div>

      {error && (
        <div className="mt-6 rounded-lg border border-signal-low/30 bg-signal-low/[0.07] px-4 py-3">
          <p className="text-sm text-signal-low">{error}</p>
        </div>
      )}

      {captures === null && !error && (
        <p className="mt-8 text-sm text-ink-400">Loading…</p>
      )}

      {captures !== null && captures.length === 0 && !error && (
        <div className="mt-8 panel p-6 text-center">
          <p className="text-sm text-ink-300">No captures yet.</p>
          <p className="prose-note mt-2">
            Press the button on the capture device. Its photo, search result,
            and a thumbnail will appear here - the original photo itself is
            never kept, only this record.
          </p>
        </div>
      )}

      {captures !== null && captures.length > 0 && (
        <ul className="mt-8 space-y-3">
          {captures.map((entry) => (
            <li
              key={entry.id}
              className={`panel flex items-center gap-4 p-4 transition-shadow ${
                flashId === entry.id ? "ring-2 ring-scan-500/60" : ""
              }`}
            >
              <div className="reticle relative h-16 w-16 shrink-0 overflow-hidden rounded-md">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={entry.query_preview}
                  alt=""
                  className="h-full w-full object-cover"
                />
              </div>

              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="chip">{entry.device_id ?? "unknown device"}</span>
                  <span className="font-mono text-[10px] text-ink-500">
                    {timeAgo(entry.captured_at, now)}
                  </span>
                </div>
                {entry.top_match ? (
                  <p className="mt-1.5 truncate text-sm text-ink-200">
                    Closest match: {entry.top_match.filename}
                  </p>
                ) : (
                  <p className="mt-1.5 text-sm text-ink-500">No match found</p>
                )}
              </div>

              {entry.top_match && (
                <div className="shrink-0 text-right">
                  <p
                    className={`font-mono text-lg ${confidenceTone(entry.top_match.confidence_key)}`}
                  >
                    {entry.top_match.similarity_score.toFixed(1)}
                  </p>
                  <p className="font-mono text-[10px] uppercase tracking-wide text-ink-500">
                    {entry.top_match.confidence_level}
                  </p>
                </div>
              )}
            </li>
          ))}
        </ul>
      )}

      <p className="mt-8 text-[11px] leading-relaxed text-ink-500">
        Only a rendered thumbnail and the search result are kept per capture,
        never the original uploaded photo - so there is no "see full
        comparison" here the way there is from a live search. Restarting the
        server clears this list.
      </p>
    </div>
  );
}
