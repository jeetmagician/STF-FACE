export const metadata = {
  title: "Privacy — Facet",
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

export default function PrivacyPage() {
  return (
    <div className="mx-auto max-w-3xl px-6 py-12">
      <h1 className="text-2xl font-semibold tracking-tight text-ink-100">
        What happens to your photographs
      </h1>
      <p className="mt-3 text-sm leading-relaxed text-ink-400">
        This page describes what the software actually does. If you are running
        your own instance, you can verify every claim below against the source.
      </p>

      <div className="mt-10">
        <Section title="What is processed">
          <p>
            The image bytes you upload are decoded into a pixel array in memory.
            From that array the system derives a face bounding box, five facial
            keypoints, an aligned 112×112 crop, quality measurements, and a
            numerical face embedding. The embedding is a vector of numbers; it
            is not reversible into a recognisable photograph, though it is still
            biometric data and should be treated as personal information.
          </p>
          <p>
            EXIF metadata — including any GPS coordinates, camera serial number,
            and timestamps — is discarded during decoding. Orientation is the
            one EXIF field that is read, and only so the image can be turned the
            right way up before it is thrown away with the rest.
          </p>
        </Section>

        <Section title="Whether images are stored">
          <p>
            No. Uploaded photographs are never written to disk. There is no
            upload directory, no temporary file, and therefore no cleanup job
            that could fail and leave images behind. Processing happens entirely
            on in-memory arrays that are released when the request completes.
          </p>
          <p>
            One narrow exception exists for usability: if an image contains
            several faces, the decoded arrays are held in process memory so you
            can choose which face to compare without re-uploading. That hold is
            capped at five minutes by default, the entry is erased the moment
            analysis finishes, the pixel buffers are overwritten on release, and
            it can be disabled outright by setting{" "}
            <code className="rounded bg-ink-800 px-1 py-0.5 font-mono text-[11px] text-ink-300">
              RETAIN_FOR_FACE_SELECTION=false
            </code>
            .
          </p>
        </Section>

        <Section title="How long anything is retained">
          <p>
            For the duration of the request, typically under a second. The
            face-selection hold described above is the only state that outlives
            a single request, and it expires on a timer that runs whether or not
            you return.
          </p>
          <p>
            Nothing is retained after the process exits. Restarting the service
            clears everything, because there is nothing persistent to clear.
          </p>
        </Section>

        <Section title="Whether third parties receive your images">
          <p>
            No. All analysis runs on the machine you deploy to. The
            face-recognition models run locally, downloaded once at install
            time. There is no external API call in the analysis path, no
            telemetry, and no analytics.
          </p>
          <p>
            The visual output returned to your browser is embedded directly in
            the response as base64 data. No uploaded image is ever given a URL,
            so there is no address at which one could be fetched — by you, by
            us, or by anyone who guessed correctly.
          </p>
        </Section>

        <Section title="What is logged">
          <p>
            Ordinary server logs: request paths, status codes, timings, and
            error traces. Image content is never logged. Filenames are not
            recorded. If you deploy behind a proxy, check that its own access
            logs do not capture more than you intend.
          </p>
        </Section>

        <Section title="Photographs of other people">
          <p>
            Biometric processing of a person&rsquo;s face is regulated in many
            jurisdictions, and in several it requires that person&rsquo;s
            explicit consent regardless of how the photograph was obtained.
            Being able to see an image is not the same as being permitted to run
            biometric analysis on it.
          </p>
          <p>
            This software gives you no legal basis you did not already have.
            Establishing one is your responsibility, and it is a real one.
          </p>
        </Section>

        <Section title="What this tool cannot be used for">
          <p>
            Facet compares photographs you supply. It holds no database of
            faces, performs no one-to-many search, and cannot be asked who
            someone is. That is a design decision: one-to-many identification is
            the capability that turns face comparison into surveillance, and it
            is deliberately absent.
          </p>
          <p>
            No sensitive characteristic is inferred — not race, ethnicity,
            religion, health, sexuality, character, or criminality. Such
            inferences are not supported by face data, and the code to attempt
            them does not exist here.
          </p>
        </Section>
      </div>
    </div>
  );
}
