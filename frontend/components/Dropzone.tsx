"use client";

import { useCallback, useRef, useState } from "react";
import type { LocalPhoto } from "@/lib/types";

const ACCEPTED = "image/jpeg,image/png,image/webp,image/bmp,image/tiff";

interface DropzoneProps {
  label: string;
  hint: string;
  photos: LocalPhoto[];
  maxPhotos: number;
  disabled?: boolean;
  onAdd: (files: File[]) => void;
  onRemove: (id: string) => void;
  onSelectFace: (photoId: string, faceIndex: number) => void;
}

function QualityPip({ score }: { score: number }) {
  const tone =
    score >= 70 ? "bg-signal-high" : score >= 45 ? "bg-signal-mid" : "bg-signal-low";
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={`h-1.5 w-1.5 rounded-full ${tone}`} />
      <span className="font-mono text-[11px] text-ink-400">
        {Math.round(score)}
      </span>
    </span>
  );
}

function PhotoCard({
  photo,
  onRemove,
  onSelectFace,
}: {
  photo: LocalPhoto;
  onRemove: (id: string) => void;
  onSelectFace: (photoId: string, faceIndex: number) => void;
}) {
  const detection = photo.detection;
  const faceCount = detection?.face_count ?? 0;
  const activeFace = detection?.faces[photo.selectedFaceIndex];

  return (
    <li className="panel-inset overflow-hidden">
      <div className="flex gap-3 p-3">
        <div className="reticle relative h-20 w-20 shrink-0 overflow-hidden rounded-md bg-ink-900">
          {/* Local blob preview: never uploaded anywhere to be displayed. */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={photo.previewUrl}
            alt=""
            className="h-full w-full object-cover"
          />
          {photo.detecting && (
            <div className="absolute inset-0 overflow-hidden bg-ink-950/60">
              <div className="h-full w-full animate-sweep bg-gradient-to-r from-transparent via-scan-500/25 to-transparent" />
            </div>
          )}
        </div>

        <div className="min-w-0 flex-1">
          <p className="truncate text-xs text-ink-300">{photo.file.name}</p>

          <div className="mt-1.5 text-[11px]">
            {photo.detecting && <span className="text-ink-500">Checking…</span>}

            {photo.detectError && (
              <span className="text-signal-low">{photo.detectError}</span>
            )}

            {detection && !photo.detecting && (
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                <span className="text-signal-high">
                  {faceCount === 1 ? "Face detected" : `${faceCount} faces`}
                </span>
                {activeFace && <QualityPip score={activeFace.quality.composite_score} />}
                {activeFace && !activeFace.quality.usable && (
                  <span className="text-signal-low">Below quality threshold</span>
                )}
              </div>
            )}
          </div>

          {activeFace?.quality.warnings.length ? (
            <p className="mt-1.5 line-clamp-2 text-[11px] leading-snug text-ink-500">
              {activeFace.quality.warnings[0]}
            </p>
          ) : null}
        </div>

        <button
          type="button"
          onClick={() => onRemove(photo.id)}
          className="h-6 w-6 shrink-0 rounded text-ink-500 transition-colors hover:bg-ink-800 hover:text-ink-200"
          aria-label={`Remove ${photo.file.name}`}
        >
          <svg viewBox="0 0 16 16" className="mx-auto h-3.5 w-3.5">
            <path
              d="M4 4l8 8M12 4l-8 8"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
            />
          </svg>
        </button>
      </div>

      {/* Several faces: the user must choose. Never picked silently. */}
      {faceCount > 1 && detection && (
        <div className="border-t border-ink-800 bg-ink-950/40 p-3">
          <p className="text-[11px] text-brass-400">
            Several faces found — choose the one to compare.
          </p>
          <div className="mt-2 flex flex-wrap gap-2">
            {detection.faces.map((face) => {
              const selected = face.index === photo.selectedFaceIndex;
              return (
                <button
                  key={face.index}
                  type="button"
                  onClick={() => onSelectFace(photo.id, face.index)}
                  aria-pressed={selected}
                  className={`relative overflow-hidden rounded-md border-2 transition-colors ${
                    selected
                      ? "border-brass-500"
                      : "border-transparent hover:border-ink-600"
                  }`}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={face.thumbnail}
                    alt={`Face ${face.index + 1}`}
                    className="h-14 w-14 object-cover"
                  />
                  <span className="absolute bottom-0 left-0 right-0 bg-ink-950/80 py-0.5 text-center font-mono text-[9px] text-ink-300">
                    {Math.round(face.quality.composite_score)}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </li>
  );
}

export default function Dropzone({
  label,
  hint,
  photos,
  maxPhotos,
  disabled,
  onAdd,
  onRemove,
  onSelectFace,
}: DropzoneProps) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const atCapacity = photos.length >= maxPhotos;

  const handleFiles = useCallback(
    (fileList: FileList | null) => {
      if (!fileList) return;
      const remaining = maxPhotos - photos.length;
      if (remaining <= 0) return;
      onAdd(Array.from(fileList).slice(0, remaining));
    },
    [maxPhotos, photos.length, onAdd],
  );

  return (
    <div className="panel flex flex-col p-5">
      <div className="flex items-baseline justify-between">
        <h3 className="text-sm font-medium text-ink-100">{label}</h3>
        <span className="font-mono text-[11px] text-ink-500">
          {photos.length}/{maxPhotos}
        </span>
      </div>
      <p className="prose-note mt-1 text-xs">{hint}</p>

      {!atCapacity && (
        <div
          role="button"
          tabIndex={disabled ? -1 : 0}
          aria-disabled={disabled}
          onClick={() => !disabled && inputRef.current?.click()}
          onKeyDown={(event) => {
            if (disabled) return;
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              inputRef.current?.click();
            }
          }}
          onDragOver={(event) => {
            event.preventDefault();
            if (!disabled) setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            if (!disabled) handleFiles(event.dataTransfer.files);
          }}
          className={`mt-4 cursor-pointer rounded-lg border border-dashed px-4 py-8 text-center transition-colors ${
            disabled
              ? "cursor-not-allowed border-ink-800 opacity-50"
              : dragging
                ? "border-brass-500 bg-brass-500/[0.06]"
                : "border-ink-700 hover:border-ink-600 hover:bg-ink-800/30"
          }`}
        >
          <svg
            viewBox="0 0 24 24"
            className="mx-auto h-6 w-6 text-ink-600"
            aria-hidden="true"
          >
            <path
              d="M12 16V4m0 0L8 8m4-4l4 4M4 17v2a2 2 0 002 2h12a2 2 0 002-2v-2"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <p className="mt-3 text-sm text-ink-300">
            Drop {photos.length > 0 ? "another photo" : "a photo"} here
          </p>
          <p className="mt-1 text-xs text-ink-500">
            or click to browse · JPEG, PNG, WebP
          </p>
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED}
        multiple={maxPhotos > 1}
        className="hidden"
        disabled={disabled}
        onChange={(event) => {
          handleFiles(event.target.files);
          event.target.value = "";
        }}
      />

      {photos.length > 0 && (
        <ul className="mt-4 space-y-2">
          {photos.map((photo) => (
            <PhotoCard
              key={photo.id}
              photo={photo}
              onRemove={onRemove}
              onSelectFace={onSelectFace}
            />
          ))}
        </ul>
      )}
    </div>
  );
}
