"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import Dropzone from "@/components/Dropzone";
import ResultPanel from "@/components/ResultPanel";
import { ApiError, analyze, detectFaces, getHealth } from "@/lib/api";
import type { AnalyzeResponse, HealthResponse, LocalPhoto } from "@/lib/types";

const MAX_PER_SIDE = 5;

type Stage = "upload" | "analyzing" | "result";

function makeId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

export default function ComparePage() {
  const [oldPhotos, setOldPhotos] = useState<LocalPhoto[]>([]);
  const [newPhotos, setNewPhotos] = useState<LocalPhoto[]>([]);
  const [stage, setStage] = useState<Stage>("upload");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [errorReasons, setErrorReasons] = useState<string[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [consent, setConsent] = useState(false);

  const objectUrls = useRef<Set<string>>(new Set());

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  // Blob URLs are a memory leak if they outlive their photo.
  useEffect(() => {
    const urls = objectUrls.current;
    return () => {
      urls.forEach((url) => URL.revokeObjectURL(url));
      urls.clear();
    };
  }, []);

  const runDetection = useCallback(
    (photo: LocalPhoto, side: "old" | "new") => {
      const setter = side === "old" ? setOldPhotos : setNewPhotos;
      detectFaces(photo.file)
        .then((detection) => {
          setter((current) =>
            current.map((item) =>
              item.id === photo.id
                ? { ...item, detection, detecting: false, selectedFaceIndex: 0 }
                : item,
            ),
          );
        })
        .catch((cause: unknown) => {
          const message =
            cause instanceof ApiError ? cause.message : "Could not check this image.";
          setter((current) =>
            current.map((item) =>
              item.id === photo.id
                ? { ...item, detecting: false, detectError: message }
                : item,
            ),
          );
        });
    },
    [],
  );

  const addPhotos = useCallback(
    (files: File[], side: "old" | "new") => {
      const created = files.map<LocalPhoto>((file) => {
        const previewUrl = URL.createObjectURL(file);
        objectUrls.current.add(previewUrl);
        return {
          id: makeId(),
          file,
          previewUrl,
          detecting: true,
          selectedFaceIndex: 0,
        };
      });

      const setter = side === "old" ? setOldPhotos : setNewPhotos;
      setter((current) => [...current, ...created]);
      created.forEach((photo) => runDetection(photo, side));
      setError(null);
      setErrorReasons([]);
    },
    [runDetection],
  );

  const removePhoto = useCallback((id: string, side: "old" | "new") => {
    const setter = side === "old" ? setOldPhotos : setNewPhotos;
    setter((current) => {
      const target = current.find((photo) => photo.id === id);
      if (target) {
        URL.revokeObjectURL(target.previewUrl);
        objectUrls.current.delete(target.previewUrl);
      }
      return current.filter((photo) => photo.id !== id);
    });
  }, []);

  const selectFace = useCallback(
    (photoId: string, faceIndex: number, side: "old" | "new") => {
      const setter = side === "old" ? setOldPhotos : setNewPhotos;
      setter((current) =>
        current.map((photo) =>
          photo.id === photoId ? { ...photo, selectedFaceIndex: faceIndex } : photo,
        ),
      );
    },
    [],
  );

  const busy = [...oldPhotos, ...newPhotos].some((photo) => photo.detecting);
  const hasBoth = oldPhotos.length > 0 && newPhotos.length > 0;
  const undecided = [...oldPhotos, ...newPhotos].some(
    (photo) => (photo.detection?.face_count ?? 0) === 0,
  );
  const canAnalyze = hasBoth && !busy && !undecided && consent && stage !== "analyzing";

  const handleAnalyze = useCallback(async () => {
    setStage("analyzing");
    setError(null);
    setErrorReasons([]);
    try {
      const response = await analyze({
        oldPhotos: oldPhotos.map((photo) => photo.file),
        newPhotos: newPhotos.map((photo) => photo.file),
        oldFaceIndices: oldPhotos.map((photo) => photo.selectedFaceIndex),
        newFaceIndices: newPhotos.map((photo) => photo.selectedFaceIndex),
      });
      setResult(response);
      setStage("result");
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (cause: unknown) {
      setStage("upload");
      if (cause instanceof ApiError) {
        setError(cause.message);
        setErrorReasons(cause.payload.quality?.reasons ?? []);
      } else {
        setError("Something went wrong while analysing the photographs.");
      }
    }
  }, [oldPhotos, newPhotos]);

  const reset = useCallback(() => {
    [...oldPhotos, ...newPhotos].forEach((photo) => {
      URL.revokeObjectURL(photo.previewUrl);
      objectUrls.current.delete(photo.previewUrl);
    });
    setOldPhotos([]);
    setNewPhotos([]);
    setResult(null);
    setError(null);
    setErrorReasons([]);
    setConsent(false);
    setStage("upload");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [oldPhotos, newPhotos]);

  if (stage === "result" && result) {
    return (
      <div className="mx-auto max-w-4xl px-6 py-12">
        <ResultPanel result={result} onReset={reset} />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-4xl px-6 py-12">
      <p className="label-mono">Module 01 · 1:1 verification</p>
      <h1 className="mt-1 text-2xl font-semibold tracking-tight text-ink-100">
        Compare photographs
      </h1>
      <p className="prose-note mt-2 max-w-2xl">
        Add at least one photograph on each side. Several per side gives a more
        dependable result, because pooling suppresses artefacts particular to
        any single image.
      </p>

      {health && !health.calibration_validated && (
        <div className="mt-6 rounded-lg border border-brass-600/25 bg-brass-600/[0.06] px-4 py-3">
          <p className="text-xs leading-relaxed text-brass-300">
            <strong className="text-brass-200">This instance is uncalibrated.</strong>{" "}
            It is running on placeholder parameters, so percentages describe an
            assumption rather than a measurement.{" "}
            <Link href="/methodology" className="underline hover:text-brass-200">
              What this means
            </Link>
          </p>
        </div>
      )}

      <div className="mt-8 grid gap-5 lg:grid-cols-2">
        <Dropzone
          label="Older photographs"
          hint="The earlier appearance"
          photos={oldPhotos}
          maxPhotos={MAX_PER_SIDE}
          disabled={stage === "analyzing"}
          onAdd={(files) => addPhotos(files, "old")}
          onRemove={(id) => removePhoto(id, "old")}
          onSelectFace={(photoId, index) => selectFace(photoId, index, "old")}
        />
        <Dropzone
          label="Newer photographs"
          hint="The later appearance"
          photos={newPhotos}
          maxPhotos={MAX_PER_SIDE}
          disabled={stage === "analyzing"}
          onAdd={(files) => addPhotos(files, "new")}
          onRemove={(id) => removePhoto(id, "new")}
          onSelectFace={(photoId, index) => selectFace(photoId, index, "new")}
        />
      </div>

      {error && (
        <div className="mt-6 rounded-lg border border-signal-low/30 bg-signal-low/[0.07] px-4 py-3">
          <p className="text-sm text-signal-low">{error}</p>
          {errorReasons.length > 0 && (
            <ul className="mt-2 space-y-1">
              {errorReasons.map((reason, index) => (
                <li key={index} className="text-xs leading-relaxed text-ink-400">
                  · {reason}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {hasBoth && (
        <label className="mt-7 flex cursor-pointer items-start gap-3 rounded-lg border border-ink-800 bg-ink-900/50 px-4 py-3.5">
          <input
            type="checkbox"
            checked={consent}
            onChange={(event) => setConsent(event.target.checked)}
            className="mt-0.5 h-4 w-4 shrink-0 accent-brass-500"
          />
          <span className="text-xs leading-relaxed text-ink-400">
            I have the right to process these photographs, and I understand that
            the result is a similarity measurement that does not establish
            anyone&rsquo;s identity.
          </span>
        </label>
      )}

      <div className="mt-6 flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={handleAnalyze}
          disabled={!canAnalyze}
          className="btn-primary"
        >
          {stage === "analyzing" ? (
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
              Analysing…
            </>
          ) : (
            "Analyse similarity"
          )}
        </button>

        {(oldPhotos.length > 0 || newPhotos.length > 0) && (
          <button
            type="button"
            onClick={reset}
            className="btn-ghost"
            disabled={stage === "analyzing"}
          >
            Clear
          </button>
        )}

        {busy && (
          <span className="text-xs text-ink-500">Checking images for faces…</span>
        )}
        {!busy && hasBoth && undecided && (
          <span className="text-xs text-signal-low">
            One or more images have no detectable face.
          </span>
        )}
      </div>

      <p className="mt-8 text-[11px] leading-relaxed text-ink-500">
        Photographs are sent to the analysis service, held in memory for the
        duration of the comparison, and discarded. They are never written to
        disk and never leave your own infrastructure.{" "}
        <Link href="/privacy" className="underline hover:text-ink-300">
          Full details
        </Link>
      </p>
    </div>
  );
}
