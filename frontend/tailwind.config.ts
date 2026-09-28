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
      },
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
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
      },
      animation: {
        "fade-up": "fade-up 0.45s cubic-bezier(0.16, 1, 0.3, 1) both",
        sweep: "sweep 1.6s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;
