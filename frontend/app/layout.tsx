import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Facet — Facial similarity assessment",
  description:
    "Compare two sets of photographs and receive a calibrated facial " +
    "similarity assessment. Reports similarity under a model; does not " +
    "determine identity.",
  robots: { index: false, follow: false },
};

function Header() {
  return (
    <header className="no-print sticky top-0 z-20 border-b border-ink-800 bg-ink-950/85 backdrop-blur-md">
      {/* A thin top rule in the instrument accent, not the brand colour -
          reads as chrome, not as a callout. */}
      <div className="h-px bg-gradient-to-r from-transparent via-scan-500/40 to-transparent" />
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <Link href="/" className="group flex items-center gap-3">
          <svg
            viewBox="0 0 32 32"
            className="h-7 w-7 text-brass-500"
            aria-hidden="true"
          >
            <circle
              cx="16"
              cy="16"
              r="13"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              opacity="0.35"
            />
            <circle
              cx="16"
              cy="16"
              r="7.5"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
            />
            <path
              d="M16 3.2v5.4M16 23.4v5.4M3.2 16h5.4M23.4 16h5.4"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
          </svg>
          <span className="flex items-baseline gap-2">
            <span className="text-base font-medium tracking-tight text-ink-100">
              Facet
            </span>
            <span className="label-mono hidden sm:inline">
              Facial similarity console
            </span>
          </span>
        </Link>

        <div className="flex items-center gap-7">
          <nav className="flex items-center gap-6 font-mono text-[11px] uppercase tracking-[0.1em] text-ink-400">
            <Link href="/methodology" className="transition-colors hover:text-ink-100">
              Methodology
            </Link>
            <Link href="/privacy" className="transition-colors hover:text-ink-100">
              Privacy
            </Link>
            <Link href="/compare" className="transition-colors hover:text-ink-100">
              Compare
            </Link>
            <Link href="/search" className="transition-colors hover:text-ink-100">
              Database search
            </Link>
          </nav>
          <span className="chip !text-scan-400 !border-scan-500/25">
            <span className="status-dot animate-blink" />
            Local
          </span>
        </div>
      </div>
    </header>
  );
}

function Footer() {
  return (
    <footer className="no-print mt-24 border-t border-ink-800/80">
      <div className="mx-auto max-w-6xl px-6 py-10">
        <p className="label-mono mb-2">Notice</p>
        <p className="max-w-3xl text-xs leading-relaxed text-ink-500">
          Facet reports facial similarity as measured by a face-recognition
          model under a stated calibration. It does not determine identity, and
          its output carries no evidential weight. Do not use it to make
          consequential decisions about a person, and do not use it on
          photographs you have no right to process.
        </p>
      </div>
    </footer>
  );
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <Header />
        <main>{children}</main>
        <Footer />
      </body>
    </html>
  );
}
