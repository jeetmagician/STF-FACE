"use client";

import type { AnalyzeResponse } from "@/lib/types";
import QualityPanel from "./QualityPanel";
import RegionTable from "./RegionTable";
import ScoreDial from "./ScoreDial";

const SEVERITY_TONE: Record<string, string> = {
  high: "border-signal-low/30 bg-signal-low/[0.07] text-signal-low",
  moderate: "border-signal-mid/25 bg-signal-mid/[0.06] text-signal-mid",
  inherent: "border-ink-700 bg-ink-800/30 text-ink-400",
};

function Figure({
  src,
  caption,
  alt,
}: {
  src: string;
  caption: string;
  alt: string;
}) {
  return (
    <figure>
      <div className="overflow-hidden rounded-lg border border-ink-800 bg-ink-950">
        {/* Base64 data URI returned inline by the API — not a hosted image. */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={src} alt={alt} className="w-full object-contain" />
      </div>
      <figcaption className="mt-2 text-[11px] text-ink-500">{caption}</figcaption>
    </figure>
  );
}

export default function ResultPanel({
  result,
  onReset,
  resetLabel = "Compare another pair",
  oldLabel = "Older photograph",
  newLabel = "Newer photograph",
}: {
  result: AnalyzeResponse;
  onReset: () => void;
  resetLabel?: string;
  oldLabel?: string;
  newLabel?: string;
}) {
  const { scoring, calibration, visualisations: viz } = result;

  return (
    <div className="space-y-6">
      {/* Headline */}
      <section className="panel animate-fade-up p-8">
        <div className="grid gap-8 lg:grid-cols-[auto,1fr] lg:items-center">
          <ScoreDial
            score={result.similarity_score}
            label={result.confidence_level}
            bandKey={result.confidence_key}
            validated={calibration.is_validated}
          />

          <div>
            <p className="text-base leading-relaxed text-ink-200">
              {result.statement}
            </p>
            <p className="prose-note mt-3">{result.guidance}</p>

            <div className="mt-5 rounded-lg border border-ink-700 bg-ink-950/50 px-4 py-3">
              <p className="label-mono">Evidence strength</p>
              <p className="mt-1.5 text-sm leading-relaxed text-ink-300">
                {scoring.evidence_statement}
              </p>
            </div>

            <p className="mt-4 text-xs font-medium text-brass-400">
              {result.disclaimer}
            </p>
          </div>
        </div>
      </section>

      {/* Warnings */}
      {result.warnings.length > 0 && (
        <section
          className="animate-fade-up rounded-xl border border-brass-600/25 bg-brass-600/[0.06] p-6"
          style={{ animationDelay: "60ms" }}
        >
          <h3 className="text-sm font-medium text-brass-300">
            Factors affecting this result
          </h3>
          <ul className="mt-3 space-y-2">
            {result.warnings.map((warning, index) => (
              <li
                key={index}
                className="flex gap-2.5 text-xs leading-relaxed text-ink-300"
              >
                <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-brass-500" />
                {warning}
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* Side by side */}
      {viz && (
        <section
          className="panel animate-fade-up p-6"
          style={{ animationDelay: "90ms" }}
        >
          <h3 className="text-sm font-medium text-ink-100">Visual comparison</h3>

          <div className="mt-5 grid gap-5 sm:grid-cols-2">
            <div>
              <p className="label-mono mb-2">{oldLabel}</p>
              <Figure
                src={viz.old_annotated}
                alt={`${oldLabel} with the detected face outlined`}
                caption="Detected face and landmarks"
              />
            </div>
            <div>
              <p className="label-mono mb-2">{newLabel}</p>
              <Figure
                src={viz.new_annotated}
                alt={`${newLabel} with the detected face outlined`}
                caption="Detected face and landmarks"
              />
            </div>
          </div>

          <div className="mt-7 border-t border-ink-800 pt-6">
            <p className="label-mono mb-3">
              Aligned to a common frame
            </p>
            <div className="grid gap-5 sm:grid-cols-3">
              <Figure
                src={viz.old_aligned}
                alt="Older face aligned to the canonical frame"
                caption="Older, aligned"
              />
              <Figure
                src={viz.new_aligned}
                alt="Newer face aligned to the canonical frame"
                caption="Newer, aligned"
              />
              <Figure
                src={viz.change_heatmap}
                alt="Heatmap of regional appearance differences"
                caption="Regional difference"
              />
            </div>
            <p className="mt-3 text-[11px] leading-relaxed text-ink-500">
              {viz.heatmap_caption}
            </p>
          </div>
        </section>
      )}

      {/* Narrative */}
      <section
        className="panel animate-fade-up p-6"
        style={{ animationDelay: "120ms" }}
      >
        <h3 className="text-sm font-medium text-ink-100">Assessment</h3>
        <p className="mt-3 text-sm leading-relaxed text-ink-300">
          {result.analysis}
        </p>
      </section>

      <div className="animate-fade-up" style={{ animationDelay: "150ms" }}>
        <RegionTable analysis={result.region_analysis} />
      </div>

      {/* Numbers */}
      <section
        className="panel animate-fade-up p-6"
        style={{ animationDelay: "180ms" }}
      >
        <h3 className="text-sm font-medium text-ink-100">
          Score derivation
        </h3>

        <div className="mt-5 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {[
            {
              label: "Raw cosine",
              value: scoring.raw_embedding_similarity.toFixed(4),
              note: "Between pooled embeddings",
            },
            {
              label: "Likelihood ratio",
              value: `${scoring.likelihood_ratio.toFixed(1)}:1`,
              note: `ln LR = ${scoring.log_likelihood_ratio.toFixed(2)}`,
            },
            {
              label: "Prior assumed",
              value: scoring.prior_used.toFixed(2),
              note: "Configurable",
            },
            {
              label: "Calibrated score",
              value: `${scoring.calibrated_similarity_score.toFixed(1)}%`,
              note: "LR combined with prior",
            },
          ].map((item) => (
            <div key={item.label} className="panel-inset p-4">
              <p className="label-mono">{item.label}</p>
              <p className="mt-1.5 font-mono text-lg text-ink-100">
                {item.value}
              </p>
              <p className="mt-1 text-[10px] text-ink-500">{item.note}</p>
            </div>
          ))}
        </div>

        <p className="mt-4 text-[11px] leading-relaxed text-ink-500">
          {scoring.prior_note}
        </p>

        <div className="mt-5 grid gap-4 sm:grid-cols-2">
          <QualityPanel
            title={`${oldLabel} quality`}
            quality={result.image_quality_old}
          />
          <QualityPanel
            title={`${newLabel} quality`}
            quality={result.image_quality_new}
          />
        </div>

        <div className="mt-5 grid gap-x-8 gap-y-2 border-t border-ink-800 pt-5 text-[11px] sm:grid-cols-2">
          <div className="flex justify-between gap-4">
            <span className="text-ink-500">Model</span>
            <span className="text-right font-mono text-ink-300">
              {result.model.recognizer}
            </span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-ink-500">Detector</span>
            <span className="text-right font-mono text-ink-300">
              {result.model.detector}
            </span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-ink-500">Embedding</span>
            <span className="text-right font-mono text-ink-300">
              {result.model.embedding_dim}-d
            </span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-ink-500">Pose difference</span>
            <span className="text-right font-mono text-ink-300">
              {result.pose_difference.toFixed(0)}°
            </span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-ink-500">Photos pooled</span>
            <span className="text-right font-mono text-ink-300">
              {result.templates.old_photo_count} → {result.templates.new_photo_count}
            </span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-ink-500">Calibration</span>
            <span
              className={`text-right font-mono ${
                calibration.is_validated ? "text-signal-high" : "text-signal-low"
              }`}
            >
              {calibration.is_validated ? calibration.profile : "unvalidated"}
            </span>
          </div>
        </div>

        {!result.model.commercial_use_permitted && (
          <p className="mt-4 rounded border border-ink-700 bg-ink-950/50 px-3 py-2 text-[11px] leading-relaxed text-ink-400">
            Model licence: {result.model.license}
          </p>
        )}

        {result.templates.old_photo_count + result.templates.new_photo_count > 2 && (
          <div className="mt-5 border-t border-ink-800 pt-5">
            <p className="label-mono">Photo-to-photo similarity</p>
            <p className="mt-2 text-[11px] text-ink-400">
              Range {result.templates.pairwise.min.toFixed(3)} to{" "}
              {result.templates.pairwise.max.toFixed(3)}, mean{" "}
              {result.templates.pairwise.mean.toFixed(3)} (spread{" "}
              {result.templates.pairwise.spread.toFixed(3)}). A wide spread means
              the photographs disagree with one another.
            </p>
          </div>
        )}
      </section>

      {/* Uncertainty */}
      <section
        className="panel animate-fade-up p-6"
        style={{ animationDelay: "210ms" }}
      >
        <h3 className="text-sm font-medium text-ink-100">
          Sources of uncertainty
        </h3>
        <div className="mt-4 space-y-3">
          {result.uncertainty_sources.map((source) => (
            <div
              key={source.factor}
              className={`rounded-lg border px-4 py-3 ${
                SEVERITY_TONE[source.severity] ?? SEVERITY_TONE.inherent
              }`}
            >
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-xs font-medium">{source.factor}</span>
                <span className="font-mono text-[9px] uppercase tracking-wider opacity-70">
                  {source.severity}
                </span>
              </div>
              <p className="mt-1.5 text-[11px] leading-relaxed text-ink-400">
                {source.detail}
              </p>
            </div>
          ))}
        </div>

        <p className="mt-5 text-[11px] leading-relaxed text-ink-500">
          {result.method_statement}
        </p>
      </section>

      <div
        className="no-print flex animate-fade-up justify-center pt-2"
        style={{ animationDelay: "240ms" }}
      >
        <button type="button" onClick={onReset} className="btn-ghost">
          {resetLabel}
        </button>
      </div>
    </div>
  );
}
