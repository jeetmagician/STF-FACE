import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#0a0a0c",
          900: "#101014",
          850: "#16161c",
          800: "#1d1d25",
          700: "#2a2a35",
          600: "#3b3b49",
          500: "#5a5a6b",
          400: "#8b8b9c",
          300: "#b4b4c2",
          200: "#d6d6de",
          100: "#eeeef2",
        },
        brass: {
          600: "#a67c2e",
          500: "#c39440",
          400: "#d9ad5c",
          300: "#e8c688",
        },
        signal: {
          high: "#5fa88a",
          mid: "#d9ad5c",
          low: "#c97b6b",
        },
        // A cool, instrument-panel accent distinct from the warm brand brass -
        // reserved for technical chrome (reticles, active/live indicators,
        // focus ticks), never for score or band colour, which stay governed
        // by `signal` so the UI cannot imply a pass/fail threshold.
        scan: {
          500: "#3fb8cf",
          400: "#63c9dc",
          300: "#93dbe8",
        },
      },
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      backgroundImage: {
        grid: "linear-gradient(to right, rgba(139,139,156,0.06) 1px, transparent 1px), linear-gradient(to bottom, rgba(139,139,156,0.06) 1px, transparent 1px)",
      },
      backgroundSize: {
        grid: "28px 28px",
      },
      keyframes: {
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        sweep: {
          "0%": { transform: "translateX(-100%)" },
          "100%": { transform: "translateX(100%)" },
        },
        scanline: {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100%)" },
        },
        blink: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.35" },
        },
      },
      animation: {
        "fade-up": "fade-up 0.45s cubic-bezier(0.16, 1, 0.3, 1) both",
        sweep: "sweep 1.6s ease-in-out infinite",
        scanline: "scanline 2.4s linear infinite",
        blink: "blink 2s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;
