"use client";

import { useEffect, useState } from "react";

const TONES: Record<string, { stroke: string; text: string }> = {
  very_high: { stroke: "#5fa88a", text: "text-signal-high" },
  high: { stroke: "#8fbd8a", text: "text-signal-high" },
  moderate: { stroke: "#d9ad5c", text: "text-signal-mid" },
  low: { stroke: "#c97b6b", text: "text-signal-low" },
};

interface ScoreDialProps {
  score: number;
  label: string;
  bandKey: string;
  validated: boolean;
}

/**
 * An arc, not a gauge with a green zone.
 *
 * A dial with a "pass" region would imply a threshold above which the answer
 * is yes. There is no such threshold here, so the arc is a continuous sweep
 * with band boundaries marked as faint ticks rather than coloured zones.
 */
export default function ScoreDial({
  score,
  label,
  bandKey,
  validated,
}: ScoreDialProps) {
  const [shown, setShown] = useState(0);
  const tone = TONES[bandKey] ?? TONES.moderate;

  useEffect(() => {
    const reduce =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) {
      setShown(score);
      return;
    }
    let frame = 0;
    const start = performance.now();
    const duration = 900;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      setShown(score * eased);
      if (t < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [score]);

  const radius = 78;
  const circumference = Math.PI * radius; // semicircle
  const progress = (shown / 100) * circumference;

  // Minor graduations every 10 units, like an analogue measurement
  // instrument - purely a scale reference, coloured identically regardless
  // of value so they never read as a pass/fail zone.
  const minorTicks = Array.from({ length: 11 }, (_, i) => i * 10);

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 200 124" className="w-full max-w-[280px]" role="img"
           aria-label={`Similarity score ${score.toFixed(1)} out of 100, ${label}`}>
        {/* Outer bezel: instrument casing, not data. */}
        <path
          d={`M ${100 - radius - 13} 100 A ${radius + 13} ${radius + 13} 0 0 1 ${100 + radius + 13} 100`}
          fill="none"
          stroke="#1d1d25"
          strokeWidth="1"
        />

        {minorTicks.map((tick) => {
          const angle = Math.PI * (1 - tick / 100);
          const inner = radius + 10;
          const outer = radius + (tick % 50 === 0 ? 15 : 13);
          const x1 = 100 + Math.cos(angle) * inner;
          const y1 = 100 - Math.sin(angle) * inner;
          const x2 = 100 + Math.cos(angle) * outer;
          const y2 = 100 - Math.sin(angle) * outer;
          return (
            <line
              key={tick}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke="#3b3b49"
              strokeWidth={tick % 50 === 0 ? 1.4 : 1}
            />
          );
        })}

        <path
          d={`M ${100 - radius} 100 A ${radius} ${radius} 0 0 1 ${100 + radius} 100`}
          fill="none"
          stroke="#2a2a35"
          strokeWidth="9"
          strokeLinecap="round"
        />
        <path
          d={`M ${100 - radius} 100 A ${radius} ${radius} 0 0 1 ${100 + radius} 100`}
          fill="none"
          stroke={tone.stroke}
          strokeWidth="9"
          strokeLinecap="round"
          strokeDasharray={`${progress} ${circumference}`}
          style={{ transition: "stroke 0.4s ease" }}
        />

        {/* Needle marking the live value, instrument-style. */}
        {(() => {
          const angle = Math.PI * (1 - shown / 100);
          const x = 100 + Math.cos(angle) * (radius - 15);
          const y = 100 - Math.sin(angle) * (radius - 15);
          return <circle cx={x} cy={y} r="2.4" fill={tone.stroke} />;
        })()}

        {/* Band boundaries as faint ticks: informative, not prescriptive. */}
        {[45, 70, 85].map((boundary) => {
          const angle = Math.PI * (1 - boundary / 100);
          const x1 = 100 + Math.cos(angle) * (radius - 7);
          const y1 = 100 - Math.sin(angle) * (radius - 7);
          const x2 = 100 + Math.cos(angle) * (radius + 7);
          const y2 = 100 - Math.sin(angle) * (radius + 7);
          return (
            <line
              key={boundary}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke="#5a5a6b"
              strokeWidth="1"
            />
          );
        })}

        <text
          x="100"
          y="86"
          textAnchor="middle"
          className="fill-ink-100 font-mono"
          style={{ fontSize: "34px", fontWeight: 600, letterSpacing: "-0.02em" }}
        >
          {shown.toFixed(1)}
        </text>
        <text
          x="100"
          y="103"
          textAnchor="middle"
          className="fill-ink-500 font-mono"
          style={{ fontSize: "10px", letterSpacing: "0.12em" }}
        >
          SIMILARITY INDEX · 0–100
        </text>
      </svg>

      <p className={`mt-1 font-mono text-lg font-medium uppercase tracking-wide ${tone.text}`}>
        {label}
      </p>

      {!validated && (
        <p className="chip mt-3 max-w-[260px] text-center normal-case tracking-normal text-brass-400">
          Uncalibrated — placeholder parameters, not a measurement
        </p>
      )}
    </div>
  );
}
