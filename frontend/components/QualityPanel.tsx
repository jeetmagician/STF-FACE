"use client";

import type { QualityInfo } from "@/lib/types";

function Metric({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail?: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-2">
      <span className="text-xs text-ink-400">{label}</span>
      <span className="text-right">
        <span className="font-mono text-xs text-ink-200">{value}</span>
        {detail && (
          <span className="ml-2 font-mono text-[10px] text-ink-500">{detail}</span>
        )}
      </span>
    </div>
  );
}

export default function QualityPanel({
  title,
  quality,
}: {
  title: string;
  quality: QualityInfo;
}) {
  const tone =
    quality.composite_score >= 70
      ? "text-signal-high"
      : quality.composite_score >= 45
        ? "text-signal-mid"
        : "text-signal-low";

  return (
    <div className="panel-inset p-4">
      <div className="flex items-baseline justify-between">
        <h4 className="text-xs font-medium text-ink-200">{title}</h4>
        <span className={`font-mono text-sm ${tone}`}>
          {quality.composite_score.toFixed(0)}
          <span className="text-[10px] text-ink-500">/100</span>
        </span>
      </div>

      <div className="mt-2 divide-y divide-ink-800/60">
        <Metric
          label="Face width"
          value={`${quality.face_width_px.toFixed(0)} px`}
        />
        <Metric
          label="Sharpness"
          value={quality.blur_score.toFixed(0)}
          detail="Laplacian var."
        />
        <Metric
          label="Exposure"
          value={quality.exposure_score.toFixed(2)}
          detail="0–1"
        />
        <Metric
          label="Head pose"
          value={`${quality.pose.yaw_deg.toFixed(0)}° / ${quality.pose.pitch_deg.toFixed(0)}°`}
          detail="yaw / pitch"
        />
        <Metric
          label="Detection"
          value={quality.detection_score.toFixed(3)}
          detail="confidence"
        />
      </div>

      {(quality.occlusion.suspected_sunglasses || quality.occlusion.suspected_mask) && (
        <p className="mt-3 rounded border border-brass-600/25 bg-brass-600/[0.07] px-2.5 py-2 text-[11px] leading-relaxed text-brass-300">
          Possible occlusion:{" "}
          {[
            quality.occlusion.suspected_sunglasses && "eye region",
            quality.occlusion.suspected_mask && "lower face",
          ]
            .filter(Boolean)
            .join(" and ")}
          . This is a heuristic and can misfire on shadow, dark-framed glasses
          or facial hair.
        </p>
      )}
    </div>
  );
}
