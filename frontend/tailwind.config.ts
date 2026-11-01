import type { Config } from 'tailwindcss'

const config: Config = {
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}', './lib/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: { 950: '#050505', 900: '#0b0b0c', 850: '#0f0f11', 800: '#131317', 700: '#191920', 600: '#22222a' },
        fog: {
          50: '#f5f5f7',
          100: '#ededf0',
          200: '#d4d4dc',
          300: '#b4b4c0',
          400: '#8f8f9c',
          500: '#66666f',
          600: '#42424b',
          700: '#2a2a31',
        },
        line: { soft: '#1a1b23', DEFAULT: '#232530', strong: '#30333f' },
        ice: '#bcd6ec',
        space: { 200: '#cfe0f2', 400: '#a9c7ea', 600: '#7fa3d0' },
        silver: '#d8d8e0',
        rose: '#c297a0',
        leaf: '#9ec9ae',
        amber: '#cfc3a2',
      },
      fontFamily: {
        sans: ['var(--font-sans)', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['var(--font-mono)', 'ui-monospace', 'monospace'],
        lora: ['var(--font-lora)', 'Georgia', 'serif'],
      },
      boxShadow: {
        glow: '0 0 100px 10px rgba(190, 205, 235, 0.08)',
        card: '0 24px 70px rgba(0, 0, 0, 0.5)',
        inner: 'inset 0 1px 0 rgba(255,255,255,0.05)',
      },
      animation: {
        'twinkle': 'twinkle 4s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}

export default config
