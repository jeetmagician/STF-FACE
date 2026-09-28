"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import Dropzone from "@/components/Dropzone";
import ResultPanel from "@/components/ResultPanel";
import {
  ApiError,
  compareDatabaseMatch,
  detectFaces,
  getDatabaseSearchStatus,
  searchDatabase,
} from "@/lib/api";
import type {
  AnalyzeResponse,
  DatabaseMatchInfo,
  DatabaseSearchResponse,
  DatabaseSearchStatus,
  LocalPhoto,
} from "@/lib/types";

type Stage = "upload" | "searching" | "result";

function makeId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function confidenceTone(key: string): string {
  if (key === "very_high" || key === "high") return "text-signal-high";
  if (key === "moderate") return "text-signal-mid";
  return "text-signal-low";
}

export default function SearchPage() {
  const [status, setStatus] = useState<DatabaseSearchStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [photo, setPhoto] = useState<LocalPhoto | null>(null);
  const [stage, setStage] = useState<Stage>("upload");
  const [result, setResult] = useState<DatabaseSearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [consent, setConsent] = useState(false);

  const [activeMatch, setActiveMatch] = useState<DatabaseMatchInfo | null>(null);
  const [matchDetail, setMatchDetail] = useState<AnalyzeResponse | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  const objectUrls = useRef<Set<string>>(new Set());

  useEffect(() => {
    getDatabaseSearchStatus()
      .then(setStatus)
      .catch(() => setStatusError("Could not reach the analysis service."));
  }, []);

  useEffect(() => {
    const urls = objectUrls.current;
    return () => {
      urls.forEach((url) => URL.revokeObjectURL(url));
      urls.clear();
    };
  }, []);

  const addPhoto = useCallback((files: File[]) => {
    const file = files[0];
    if (!file) return;
    const previewUrl = URL.createObjectURL(file);
    objectUrls.current.add(previewUrl);
    setPhoto({
      id: makeId(),
      file,
      previewUrl,
      detecting: true,
      selectedFaceIndex: 0,
    });
    setError(null);
  }, []);

  const runDetection = useCallback((current: LocalPhoto) => {
    detectFaces(current.file)
      .then((detection) => {
        setPhoto((prev) =>
          prev && prev.id === current.id
            ? { ...prev, detection, detecting: false, selectedFaceIndex: 0 }
            : prev,
        );
      })
      .catch((cause: unknown) => {
        const message =
          cause instanceof ApiError ? cause.message : "Could not check this image.";
        setPhoto((prev) =>
          prev && prev.id === current.id
            ? { ...prev, detecting: false, detectError: message }
            : prev,
        );
      });
  }, []);

  useEffect(() => {
    if (photo && photo.detecting && !photo.detection && !photo.detectError) {
      runDetection(photo);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [photo?.id]);

  const removePhoto = useCallback(() => {
    if (photo) {
      URL.revokeObjectURL(photo.previewUrl);
      objectUrls.current.delete(photo.previewUrl);
    }
    setPhoto(null);
  }, [photo]);

  const selectFace = useCallback((_photoId: string, faceIndex: number) => {
    setPhoto((prev) => (prev ? { ...prev, selectedFaceIndex: faceIndex } : prev));
  }, []);

  const hasPhoto = photo !== null;
  const busy = photo?.detecting ?? false;
  const undecided = (photo?.detection?.face_count ?? 0) === 0;
  const canSearch =
    hasPhoto && !busy && !undecided && consent && stage !== "searching" && !!status?.enabled;

  const handleSearch = useCallback(async () => {
    if (!photo) return;
    setStage("searching");
    setError(null);
    try {
      const response = await searchDatabase({
        file: photo.file,
        faceIndex: photo.selectedFaceIndex,
      });
      setResult(response);
      setStage("result");
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (cause: unknown) {
      setStage("upload");
      setError(
        cause instanceof ApiError
          ? cause.message
          : "Something went wrong while searching the database.",
      );
    }
  }, [photo]);

  const reset = useCallback(() => {
    if (photo) {
      URL.revokeObjectURL(photo.previewUrl);
      objectUrls.current.delete(photo.previewUrl);
    }
    setPhoto(null);
    setResult(null);
    setError(null);
    setConsent(false);
    setStage("upload");
    setActiveMatch(null);
    setMatchDetail(null);
    setDetailError(null);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [photo]);

  const openDetail = useCallback(
    async (match: DatabaseMatchInfo) => {
      if (!photo) return;
      setActiveMatch(match);
      setMatchDetail(null);
      setDetailError(null);
      setDetailLoading(true);
      try {
        const detail = await compareDatabaseMatch({
          file: photo.file,
          path: match.path,
          faceIndex: photo.selectedFaceIndex,
        });
        setMatchDetail(detail);
      } catch (cause: unknown) {
        setDetailError(
          cause instanceof ApiError
            ? cause.message
            : "Could not load the comparison details for this match.",
        );
      } finally {
        setDetailLoading(false);
      }
    },
    [photo],
  );

  const closeDetail = useCallback(() => {
    setActiveMatch(null);
    setMatchDetail(null);
    setDetailError(null);
  }, []);

  if (stage === "result" && result && activeMatch) {
    return (
      <div className="mx-auto max-w-4xl px-6 py-12">
        <div className="no-print mb-6 flex flex-wrap items-center gap-3">
          <button type="button" onClick={closeDetail} className="btn-ghost">
            ← Back to results
          </button>
          {matchDetail && (
            <button
              type="button"
              onClick={() => window.print()}
              className="btn-primary"
            >
              Export as PDF
            </button>
          )}
        </div>

        <div className="no-print mb-6 flex items-center gap-4 panel p-4">
          <div className="reticle relative h-14 w-14 shrink-0 overflow-hidden rounded-md">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={activeMatch.thumbnail}
              alt=""
              className="h-full w-full object-cover"
            />
          </div>
          <div className="min-w-0">
            <p className="truncate text-sm text-ink-200">{activeMatch.filename}</p>
            <p className="truncate text-[11px] text-ink-500">{activeMatch.path}</p>
          </div>
        </div>

        {detailLoading && (
          <p className="text-sm text-ink-400">Loading the full comparison…</p>
        )}

        {detailError && (
          <div className="rounded-lg border border-signal-low/30 bg-signal-low/[0.07] px-4 py-3">
            <p className="text-sm text-signal-low">{detailError}</p>
          </div>
        )}

        {matchDetail && (
          <ResultPanel
            result={matchDetail}
            onReset={closeDetail}
            resetLabel="Back to results"
            oldLabel="Your query photo"
            newLabel="Matched database photo"
          />
        )}
      </div>
    );
  }

  if (stage === "result" && result) {
    return (
      <div className="mx-auto max-w-4xl px-6 py-12">
        <p className="label-mono">Database search · Results</p>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight text-ink-100">
          Search results
        </h1>
        <p className="prose-note mt-2 max-w-2xl">
          Ranked against {result.indexed_photo_count} indexed photograph
          {result.indexed_photo_count === 1 ? "" : "s"} in the local database folder.
          Click a match to see the full comparison behind its score.
        </p>

        {result.warnings.length > 0 && (
          <div className="mt-6 space-y-2 rounded-lg border border-brass-600/25 bg-brass-600/[0.06] px-4 py-3">
            {result.warnings.map((warning, index) => (
              <p key={index} className="text-xs leading-relaxed text-brass-300">
                {warning}
              </p>
            ))}
          </div>
        )}

        {result.matches.length === 0 ? (
          <p className="mt-8 text-sm text-ink-400">No searchable photographs were found.</p>
        ) : (
          <ul className="mt-8 space-y-3">
            {result.matches.map((match) => (
              <li key={match.path}>
                <button
                  type="button"
                  onClick={() => openDetail(match)}
                  className="panel flex w-full items-center gap-4 p-4 text-left transition-colors hover:border-scan-500/40"
                >
                  <span className="chip !px-1.5 tabular-nums">
                    {String(match.rank).padStart(2, "0")}
                  </span>
                  <div className="reticle relative h-16 w-16 shrink-0 overflow-hidden rounded-md">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={match.thumbnail}
                      alt=""
                      className="h-full w-full object-cover"
                    />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm text-ink-200">{match.filename}</p>
                    <p className="truncate font-mono text-[11px] text-ink-500">
                      {match.path}
                    </p>
                  </div>
                  <div className="shrink-0 text-right">
                    <p className={`font-mono text-lg ${confidenceTone(match.confidence_key)}`}>
                      {match.similarity_score.toFixed(1)}
                    </p>
                    <p className="font-mono text-[10px] uppercase tracking-wide text-ink-500">
                      {match.confidence_level}
                    </p>
                  </div>
                </button>
              </li>
            ))}
          </ul>
        )}

        <div className="mt-8">
          <button type="button" onClick={reset} className="btn-ghost">
            New search
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-4xl px-6 py-12">
      <p className="label-mono">Module 02 · 1:N lookup</p>
      <h1 className="mt-1 text-2xl font-semibold tracking-tight text-ink-100">
        Search profile from database
      </h1>
      <p className="prose-note mt-2 max-w-2xl">
        Upload one photograph to search against a folder of photographs on this
        server, and see the closest-ranked matches. This searches only that
        local folder - never anything external or shared.
      </p>

      {statusError && (
        <div className="mt-6 rounded-lg border border-signal-low/30 bg-signal-low/[0.07] px-4 py-3">
          <p className="text-sm text-signal-low">{statusError}</p>
        </div>
      )}

      {status && !status.enabled && (
        <div className="mt-6 rounded-lg border border-brass-600/25 bg-brass-600/[0.06] px-4 py-3">
          <p className="text-xs leading-relaxed text-brass-300">
            <strong className="text-brass-200">Database search is disabled.</strong>{" "}
            Set <code className="text-brass-200">ENABLE_DATABASE_SEARCH=true</code> and{" "}
            <code className="text-brass-200">DATABASE_SEARCH_DIR</code> in the backend
            configuration to turn it on. This is off by default: see{" "}
            <Link href="/privacy" className="underline hover:text-brass-200">
              Privacy
            </Link>{" "}
            for why.
          </p>
        </div>
      )}

      {status?.enabled && !status.directory_exists && (
        <div className="mt-6 rounded-lg border border-signal-low/30 bg-signal-low/[0.07] px-4 py-3">
          <p className="text-sm text-signal-low">
            The configured database folder ({status.directory ?? "unset"}) does not
            exist on this server.
          </p>
        </div>
      )}

      {status?.enabled && status.directory_exists && (
        <div className="mt-6 rounded-lg border border-ink-800 bg-ink-900/50 px-4 py-3">
          <p className="text-xs leading-relaxed text-ink-400">
            Searching <span className="text-ink-200">{status.directory}</span>.{" "}
            {status.indexed_photo_count === null
              ? "The index will be built on first search."
              : `${status.indexed_photo_count} photograph(s) indexed.`}
          </p>
        </div>
      )}

      <div className="mt-8 max-w-md">
        <Dropzone
          label="Photo to search for"
          hint="The face to look up in the database"
          photos={photo ? [photo] : []}
          maxPhotos={1}
          disabled={stage === "searching"}
          onAdd={addPhoto}
          onRemove={removePhoto}
          onSelectFace={selectFace}
        />
      </div>

      {error && (
        <div className="mt-6 rounded-lg border border-signal-low/30 bg-signal-low/[0.07] px-4 py-3">
          <p className="text-sm text-signal-low">{error}</p>
        </div>
      )}

      {hasPhoto && (
        <label className="mt-7 flex cursor-pointer items-start gap-3 rounded-lg border border-ink-800 bg-ink-900/50 px-4 py-3.5">
          <input
            type="checkbox"
            checked={consent}
            onChange={(event) => setConsent(event.target.checked)}
            className="mt-0.5 h-4 w-4 shrink-0 accent-brass-500"
          />
          <span className="text-xs leading-relaxed text-ink-400">
            I have the right to process this photograph, and I understand that a
            match is a similarity measurement against my own local database, not
            proof of identity.
          </span>
        </label>
      )}

      <div className="mt-6 flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={handleSearch}
          disabled={!canSearch}
          className="btn-primary"
        >
          {stage === "searching" ? (
            <>
              <svg viewBox="0 0 16 16" className="h-4 w-4 animate-spin" aria-hidden="true">
                <circle
                  cx="8"
                  cy="8"
                  r="6"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeDasharray="28"
                  strokeDashoffset="10"
                  strokeLinecap="round"
                />
              </svg>
              Searching…
            </>
          ) : (
            "Search database"
          )}
        </button>

        {hasPhoto && (
          <button
            type="button"
            onClick={reset}
            className="btn-ghost"
            disabled={stage === "searching"}
          >
            Clear
          </button>
        )}

        {busy && <span className="text-xs text-ink-500">Checking image for a face…</span>}
        {!busy && hasPhoto && undecided && (
          <span className="text-xs text-signal-low">This image has no detectable face.</span>
        )}
      </div>
    </div>
  );
}
