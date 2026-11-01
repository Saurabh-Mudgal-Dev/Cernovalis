'use client'

import { cn } from '@/lib/cn'

export function VerdictBadge({ verdict, large = false }: { verdict: 'REAL' | 'FAKE'; large?: boolean }) {
  const fake = verdict === 'FAKE'
  return (
    <span
      className={cn(
        'inline-flex items-center justify-center rounded-xl border font-medium tracking-[0.18em]',
        large ? 'px-6 py-3 text-3xl' : 'px-2.5 py-1 text-[10px]',
        fake
          ? 'border-rose/30 bg-rose/[0.07] text-rose'
          : 'border-white/25 bg-white/[0.07] text-fog-50'
      )}
    >
      {verdict}
    </span>
  )
}

export function RiskChip({ level }: { level: 'LOW' | 'MEDIUM' | 'HIGH' }) {
  const tone =
    level === 'LOW' ? 'text-leaf' : level === 'MEDIUM' ? 'text-amber' : 'text-rose'
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border border-line-strong bg-white/[0.03] px-3 py-1 text-xs font-medium',
        tone
      )}
    >
      <span className={cn('h-1.5 w-1.5 rounded-full bg-current', tone)} />
      {level} RISK
    </span>
  )
}

export function ConfidenceGauge({ value, label = 'Confidence' }: { value: number; label?: string }) {
  const pct = Math.round(value * 100)
  const r = 46
  const c = 2 * Math.PI * r
  const dash = (value * c).toFixed(2)
  return (
    <div className="relative flex h-[132px] w-[132px] items-center justify-center">
      <svg viewBox="0 0 120 120" className="h-full w-full -rotate-90">
        <circle cx="60" cy="60" r={r} fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="7" />
        <circle
          cx="60"
          cy="60"
          r={r}
          fill="none"
          stroke="url(#gaugeGrad)"
          strokeWidth="7"
          strokeLinecap="round"
          strokeDasharray={`${dash} ${c}`}
        />
        <defs>
          <linearGradient id="gaugeGrad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#d8d8e0" />
            <stop offset="100%" stopColor="#a9c7ea" />
          </linearGradient>
        </defs>
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-xl font-medium text-fog-50">{pct}%</span>
        <span className="text-[10px] uppercase tracking-widest text-fog-500">{label}</span>
      </div>
    </div>
  )
}
