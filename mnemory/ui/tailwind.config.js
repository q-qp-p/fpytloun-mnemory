/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./mnemory/ui/static/**/*.{html,js}"],
  theme: {
    extend: {
      colors: {
        brand: {
          bg: '#0B1220',
          surface: '#121A2B',
          elevated: '#1A2438',
          accent: '#22D3EE',
          'accent-hover': '#67E8F9',
          'accent-active': '#0891B2',
          border: '#1E293B',
        },
        mem: {
          fact: '#3B82F6',
          preference: '#8B5CF6',
          episodic: '#F59E0B',
          procedural: '#10B981',
          context: '#64748B',
        },
        imp: {
          low: '#64748B',
          normal: '#22D3EE',
          high: '#F59E0B',
          critical: '#EF4444',
        },
      },
      textColor: {
        primary: '#E6EDF3',
        secondary: '#94A3B8',
        muted: '#64748B',
        disabled: '#475569',
      },
      boxShadow: {
        glow: '0 0 24px rgba(34, 211, 238, 0.35)',
        'glow-sm': '0 0 12px rgba(34, 211, 238, 0.2)',
      },
      backgroundImage: {
        'gradient-accent': 'linear-gradient(135deg, #22D3EE 0%, #2563EB 100%)',
      },
    },
  },
  plugins: [],
}
