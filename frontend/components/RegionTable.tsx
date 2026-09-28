"use client";

import type { RegionAnalysisInfo } from "@/lib/types";

function StabilityBadge({ weight }: { weight: number }) {
  const label = weight >= 0.8 ? "Stable" : weight >= 0.5 ? "Variable" : "Alterable";
  const tone =
    weight >= 0.8
      ? "border-signal-high/30 text-signal-high"
      : weight >= 0.5
        ? "border-ink-600 text-ink-400"
        : "border-signal-low/30 text-signal-low";
  return (
    <span
      className={`rounded border px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider ${tone}`}
      title={`Heuristic persistence weight ${weight.toFixed(2)} — how durable this region generally is across time, weight change and surgery`}
    >
      {label}
    </span>
  );
}

export default function RegionTable({
  analysis,
}: {
  analysis: RegionAnalysisInfo;
}) {
  const ranked = [...analysis.regions].sort(
    (a, b) => b.change_magnitude - a.change_magnitude,
  );

  return (
    <div className="panel p-6">
      <h3 className="text-sm font-medium text-ink-100">
        Which regions differ
      </h3>
      <p className="prose-note mt-2 text-xs">{analysis.interpretation}</p>

      <div className="mt-5 space-y-2">
        {ranked.map((region) => {
          const percent = Math.round(region.change_magnitude * 100);
          const tone =
            region.change_magnitude >= 0.55
              ? "bg-signal-low"
              : region.change_magnitude >= 0.35
                ? "bg-signal-mid"
                : "bg-ink-600";
          return (
            <div key={region.key} className="flex items-center gap-3">
              <div className="flex w-44 shrink-0 items-center gap-2">
                <span className="truncate text-xs text-ink-300">
                  {region.label}
                </span>
                <StabilityBadge weight={region.stability_weight} />
              </div>
              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-ink-800">
                <div
                  className={`h-full rounded-full ${tone} transition-all duration-700`}
                  style={{ width: `${Math.max(2, percent)}%` }}
                />
              </div>
              <span className="w-10 shrink-0 text-right font-mono text-[11px] text-ink-400">
                {percent}%
              </span>
            </div>
          );
        })}
      </div>

      <p className="mt-4 text-[11px] leading-relaxed text-ink-500">
        Bars show measured appearance difference between the two aligned faces,
        largest first. The badge is a heuristic ranking of how durable each
        region generally is — the nose is marked alterable because rhinoplasty
        and ageing both change it substantially.
      </p>

      {analysis.geometric_measures.length > 0 && (
        <details className="group mt-5">
          <summary className="cursor-pointer list-none text-xs text-brass-400 transition-colors hover:text-brass-300">
            <span className="inline-flex items-center gap-1.5">
              <svg
                viewBox="0 0 16 16"
                className="h-3 w-3 transition-transform group-open:rotate-90"
                aria-hidden="true"
              >
                <path
                  d="M6 4l4 4-4 4"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.6"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              Geometric measurements ({analysis.geometric_measures.length})
            </span>
          </summary>

          <div
            className={`mt-3 rounded border px-3 py-2 text-[11px] leading-relaxed ${
              analysis.geometry_reliable
                ? "border-ink-800 bg-ink-950/40 text-ink-400"
                : "border-signal-low/25 bg-signal-low/[0.06] text-signal-low"
            }`}
          >
            {analysis.geometry_caveat}
          </div>

          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-left text-[11px]">
              <thead>
                <tr className="border-b border-ink-800 text-ink-500">
                  <th className="pb-1.5 pr-3 font-normal">Measurement</th>
                  <th className="pb-1.5 pr-3 text-right font-normal">Older</th>
                  <th className="pb-1.5 pr-3 text-right font-normal">Newer</th>
                  <th className="pb-1.5 text-right font-normal">Change</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ink-800/60">
                {analysis.geometric_measures.map((measure) => {
                  const pct = measure.relative_delta * 100;
                  const notable = Math.abs(pct) >= 12;
                  return (
                    <tr key={measure.key}>
                      <td className="py-1.5 pr-3 text-ink-300">{measure.label}</td>
                      <td className="py-1.5 pr-3 text-right font-mono text-ink-400">
                        {measure.old_value.toFixed(3)}
                      </td>
                      <td className="py-1.5 pr-3 text-right font-mono text-ink-400">
                        {measure.new_value.toFixed(3)}
                      </td>
                      <td
                        className={`py-1.5 text-right font-mono ${
                          notable ? "text-signal-mid" : "text-ink-500"
                        }`}
                      >
                        {pct >= 0 ? "+" : ""}
                        {pct.toFixed(1)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <p className="mt-2 text-[11px] text-ink-500">
            Distances are divided by inter-ocular distance, which makes them
            independent of image size but not of head pose.
            {analysis.landmark_density <= 5 &&
              " The active backend provides 5 facial keypoints, so contour-based measurements (face width, jaw width) are unavailable."}
          </p>
        </details>
      )}
    </div>
  );
}
