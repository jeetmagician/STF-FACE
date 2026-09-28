export const metadata = {
  title: "Methodology — Facet",
};

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="border-t border-ink-800 py-7 first:border-t-0 first:pt-0">
      <h2 className="text-sm font-medium text-ink-100">{title}</h2>
      <div className="mt-3 space-y-3 text-sm leading-relaxed text-ink-400">
        {children}
      </div>
    </section>
  );
}

function Formula({ children }: { children: React.ReactNode }) {
  return (
    <div className="my-4 rounded-lg border border-ink-800 bg-ink-950/50 px-4 py-3">
      <code className="font-mono text-xs text-brass-300">{children}</code>
    </div>
  );
}

export default function MethodologyPage() {
  return (
    <div className="mx-auto max-w-3xl px-6 py-12">
      <h1 className="text-2xl font-semibold tracking-tight text-ink-100">
        How the score is produced
      </h1>
      <p className="mt-3 text-sm leading-relaxed text-ink-400">
        And, more importantly, what it is worth.
      </p>

      <div className="mt-10">
        <Section title="Embeddings, not pixels">
          <p>
            Two photographs of the same person can differ in every pixel while
            depicting an obviously identical face. Comparison therefore runs on
            face embeddings: each aligned face is passed through a recognition
            network trained so that images of one person land near each other in
            a high-dimensional space, and images of different people land apart.
            Similarity is the cosine of the angle between two such vectors.
          </p>
          <p>
            Before embedding, each face is warped to a canonical frame using a
            similarity transform fitted to five facial keypoints. This removes
            scale, in-plane rotation and translation, so the network sees faces
            presented consistently rather than at whatever size and angle the
            camera happened to catch them.
          </p>
        </Section>

        <Section title="Why a cosine is not a percentage">
          <p>
            A cosine similarity of 0.42 is not &ldquo;42% likely to be the same
            person&rdquo;. It is a distance in a learned space, and its meaning
            depends entirely on how that space distributes scores for genuine
            and impostor pairs — which varies with the model, the population,
            the capture conditions, and the difficulty of the pairs.
          </p>
          <p>
            The same 0.42 might be a comfortable match for well-lit frontal
            photographs taken a year apart, and thoroughly ambiguous for
            photographs separated by fifteen years. Multiplying it by one
            hundred produces a number that looks like a probability and is not
            one.
          </p>
        </Section>

        <Section title="The likelihood ratio">
          <p>
            Instead, the system estimates how the score distributes under each
            hypothesis and reports the ratio of those two densities at the
            observed score:
          </p>
          <Formula>
            LR = p(score | same person) ÷ p(score | different people)
          </Formula>
          <p>
            This is the evidence the comparison carries, and it depends on no
            assumption about how likely a match was beforehand. A likelihood
            ratio of 20:1 means the observed score is twenty times more probable
            if the photographs show one person than if they show two.
          </p>
          <p>
            The percentage shown in the interface is then derived by combining
            that evidence with an explicit prior:
          </p>
          <Formula>
            logit(posterior) = ln(LR) + logit(prior)
          </Formula>
          <p>
            The prior is unavoidable — a probability that two photographs show
            the same person depends on how the pair was selected — so it is
            exposed as a configurable value and printed alongside every result
            rather than hidden inside a constant. Pairs drawn from a passport
            renewal queue have a very different base rate from pairs drawn from
            two unrelated social media accounts, and the same score should not
            yield the same percentage in both settings.
          </p>
        </Section>

        <Section title="Why this instance may say it is uncalibrated">
          <p>
            Those two densities have to be estimated from labelled pairs. A
            freshly installed system has none, so it runs on documented
            placeholder parameters and says so on every result.
          </p>
          <p>
            Those placeholders are not a soft default that is &ldquo;probably
            about right&rdquo;. They are a starting point chosen to be plausible
            and to put the decision point near each model&rsquo;s published
            operating threshold, and they will be wrong for your data in ways
            that are not predictable from the outside. Calibrating means running{" "}
            <code className="rounded bg-ink-800 px-1 py-0.5 font-mono text-[11px] text-ink-300">
              scripts/fit_calibration.py
            </code>{" "}
            over your own labelled pairs.
          </p>
          <p>
            This matters most for exactly the cases this tool is aimed at.
            Public benchmarks are dominated by easy pairs; photographs separated
            by many years, or by facial surgery, sit in a shifted part of the
            score distribution. A system calibrated on benchmark-like data and
            then pointed at hard cases will systematically overstate the
            evidence against genuine pairs.
          </p>
        </Section>

        <Section title="A ceiling on how certain the result can sound">
          <p>
            The reported likelihood ratio is capped. Estimating a ratio of a
            million to one would require characterising the far tails of both
            distributions, and tails are precisely where a few thousand
            validation pairs provide no information — anything reported out
            there is extrapolation dressed as measurement.
          </p>
          <p>
            The cap means the strongest available result is roughly 55:1, which
            surfaces as a similarity score near 98 rather than 99.99. That is a
            deliberate refusal to sound more certain than the data permits.
          </p>
        </Section>

        <Section title="Faces that have changed">
          <p>
            Rhinoplasty can substantially alter nasal geometry while leaving the
            rest of the face intact. Recognition networks are largely robust to
            this, because they are trained across enormous variation in
            expression, pose and appearance and do not depend on any single
            feature. The global embedding similarity is therefore reported as
            the score.
          </p>
          <p>
            Alongside it, the system describes which facial regions differ,
            using gradient-orientation descriptors over canonical regions of the
            aligned face and anthropometric ratios normalised by inter-ocular
            distance. That analysis is <strong className="text-ink-200">descriptive
            only</strong> and is never folded into the score.
          </p>
          <p>
            The reason is technical. A face embedding cannot be decomposed into
            per-region contributions — the vector has no region-indexed
            structure — so &ldquo;weighting the periocular region more
            heavily&rdquo; inside a single embedding is not a thing that can be
            done. A system claiming to do it is inventing an interpretation the
            model does not support. What can honestly be said is: here is the
            similarity, and separately, here is what changed.
          </p>
        </Section>

        <Section title="Refusing rather than guessing">
          <p>
            Every image passes a quality gate measuring blur, exposure,
            resolution, head pose, detection confidence and probable occlusion.
            An image that fails produces a refusal, not a score.
          </p>
          <p>
            This is the single most important behaviour in the system. A
            degraded photograph still yields a number, and that number still
            looks authoritative, but it carries far less information than its
            presentation implies. Declining to answer is more useful than
            answering badly.
          </p>
        </Section>

        <Section title="What a high score still does not settle">
          <p>
            Close relatives score highly. Identical twins score very highly. No
            face-recognition model resolves this, and no threshold exists above
            which it stops being true.
          </p>
          <p>
            The result is a measurement of facial similarity under one model and
            one calibration. It is not proof of identity, it carries no
            evidential weight, and it should not be the basis of a consequential
            decision about a person.
          </p>
        </Section>
      </div>
    </div>
  );
}
