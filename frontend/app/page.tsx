import Link from "next/link";

const PIPELINE = [
  { step: "Quality", detail: "Blur, exposure, resolution, pose, occlusion" },
  { step: "Detect", detail: "Locate faces; ask if several are present" },
  { step: "Align", detail: "Warp to a canonical frame from 5 keypoints" },
  { step: "Embed", detail: "Face descriptor from a recognition model" },
  { step: "Compare", detail: "Cosine similarity between pooled templates" },
  { step: "Calibrate", detail: "Likelihood ratio, then a stated prior" },
];

export default function LandingPage() {
  return (
    <div className="mx-auto max-w-6xl px-6">
      <section className="py-20 sm:py-28">
        <p className="label-mono animate-fade-up">
          Facial similarity assessment
        </p>

        <h1
          className="mt-5 max-w-3xl animate-fade-up text-4xl font-semibold leading-[1.12] tracking-tight text-ink-100 sm:text-5xl"
          style={{ animationDelay: "60ms" }}
        >
          How similar are two faces,
          <span className="text-ink-400"> and how much should that convince you?</span>
        </h1>

        <p
          className="mt-6 max-w-2xl animate-fade-up text-lg leading-relaxed text-ink-400"
          style={{ animationDelay: "120ms" }}
        >
          Upload photographs taken at different times — across ageing, weight
          change, facial hair, different lighting, or cosmetic surgery — and
          Facet reports how similar the faces are according to a
          face-recognition model, together with an honest account of how much
          that measurement is worth.
        </p>

        <div
          className="mt-10 flex animate-fade-up flex-wrap items-center gap-3"
          style={{ animationDelay: "180ms" }}
        >
          <Link href="/compare" className="btn-primary">
            Compare photos
            <svg viewBox="0 0 16 16" className="h-4 w-4" aria-hidden="true">
              <path
                d="M3 8h10M9 4l4 4-4 4"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </Link>
          <Link href="/methodology" className="btn-ghost">
            How it works
          </Link>
        </div>
      </section>

      {/* The limits come before the features, deliberately. */}
      <section className="panel animate-fade-up p-7" style={{ animationDelay: "240ms" }}>
        <h2 className="flex items-center gap-2 font-mono text-xs font-semibold uppercase tracking-[0.1em] text-ink-200">
          <span className="h-3 w-0.5 shrink-0 bg-scan-500" aria-hidden="true" />
          What this tool does not do
        </h2>
        <div className="mt-5 grid gap-6 sm:grid-cols-3">
          <div>
            <p className="text-sm font-medium text-brass-400">
              It does not identify anyone
            </p>
            <p className="prose-note mt-2">
              Facet compares two sets of photographs you already have. An
              optional, off-by-default search against a single local folder
              you name is the only exception, and it cannot reach any
              external or shared data — see{" "}
              <Link href="/privacy" className="underline hover:text-ink-200">
                Privacy
              </Link>
              .
            </p>
          </div>
          <div>
            <p className="text-sm font-medium text-brass-400">
              It does not prove identity
            </p>
            <p className="prose-note mt-2">
              Even a very high score is consistent with a sibling, a look-alike
              or a twin. No score, at any threshold, establishes that two
              photographs show the same person.
            </p>
          </div>
          <div>
            <p className="text-sm font-medium text-brass-400">
              It does not infer traits
            </p>
            <p className="prose-note mt-2">
              No estimation of race, ethnicity, religion, health, character or
              criminality. Those inferences are not supported by face data and
              are not implemented.
            </p>
          </div>
        </div>
      </section>

      <section className="mt-16">
        <h2 className="flex items-center gap-2 font-mono text-xs font-semibold uppercase tracking-[0.1em] text-ink-200">
          <span className="h-3 w-0.5 shrink-0 bg-scan-500" aria-hidden="true" />
          The pipeline
        </h2>
        <p className="prose-note mt-2 max-w-2xl">
          Comparison runs on face embeddings, not pixel differences. Every stage
          can reject the input rather than pass a degraded signal downstream.
        </p>

        <ol className="mt-7 grid gap-px overflow-hidden rounded-xl border border-ink-800 bg-ink-800 sm:grid-cols-2 lg:grid-cols-3">
          {PIPELINE.map((item, index) => (
            <li key={item.step} className="bg-ink-900/70 p-5">
              <div className="flex items-baseline gap-3">
                <span className="font-mono text-xs text-brass-500">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <span className="text-sm font-medium text-ink-100">
                  {item.step}
                </span>
              </div>
              <p className="mt-2 text-xs leading-relaxed text-ink-400">
                {item.detail}
              </p>
            </li>
          ))}
        </ol>
      </section>

      <section className="mt-16 grid gap-6 lg:grid-cols-2">
        <div className="panel p-7">
          <h2 className="flex items-center gap-2 font-mono text-xs font-semibold uppercase tracking-[0.1em] text-ink-200">
            <span className="h-3 w-0.5 shrink-0 bg-scan-500" aria-hidden="true" />
            Built for faces that have changed
          </h2>
          <p className="prose-note mt-3">
            Rhinoplasty can substantially alter nasal geometry while leaving the
            rest of the face intact. Facet reports the global embedding
            similarity as its score, and separately describes which regions
            differ — so a changed nose reads as a changed nose, not as a
            different person.
          </p>
          <p className="prose-note mt-3">
            That region breakdown is descriptive. It is never folded into the
            score, because a single face embedding cannot be decomposed into
            per-region contributions, and pretending otherwise would invent an
            explanation the model cannot give.
          </p>
        </div>

        <div className="panel p-7">
          <h2 className="flex items-center gap-2 font-mono text-xs font-semibold uppercase tracking-[0.1em] text-ink-200">
            <span className="h-3 w-0.5 shrink-0 bg-scan-500" aria-hidden="true" />
            Your photographs stay yours
          </h2>
          <p className="prose-note mt-3">
            Images are processed in memory and never written to disk. Nothing is
            stored after the comparison returns, no third-party service receives
            your photographs, and no uploaded image is ever reachable by URL.
          </p>
          <Link
            href="/privacy"
            className="mt-4 inline-flex items-center gap-1.5 text-sm text-brass-400 transition-colors hover:text-brass-300"
          >
            Read the full privacy account
            <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" aria-hidden="true">
              <path
                d="M3 8h10M9 4l4 4-4 4"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </Link>
        </div>
      </section>

      <section className="mt-16 rounded-xl border border-brass-600/25 bg-brass-600/[0.06] p-7">
        <h2 className="flex items-center gap-2 font-mono text-xs font-semibold uppercase tracking-[0.1em] text-brass-300">
          <span className="h-3 w-0.5 shrink-0 bg-brass-500" aria-hidden="true" />
          Before you rely on a number
        </h2>
        <p className="mt-3 max-w-3xl text-sm leading-relaxed text-ink-300">
          A freshly installed Facet is <strong className="text-ink-100">not
          calibrated</strong>. It ships with placeholder parameters that are
          documented starting points, not measurements, and every result says so
          until you fit calibration on photographs representative of your own
          use case. Percentages from an uncalibrated system describe an
          assumption, not evidence.
        </p>
        <Link
          href="/methodology"
          className="mt-4 inline-flex items-center gap-1.5 text-sm text-brass-400 transition-colors hover:text-brass-300"
        >
          Why calibration cannot be shipped in the box
          <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" aria-hidden="true">
            <path
              d="M3 8h10M9 4l4 4-4 4"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </Link>
      </section>
    </div>
  );
}
