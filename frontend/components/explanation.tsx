'use client'

import { cn } from '@/lib/cn'
import type { FeatureContribution, ModelContribution } from '@/lib/schemas'
import { VerdictBadge } from './verdict'

const impactTone = {
  low: 'text-fog-400 border-line-strong',
  medium: 'text-ice/90 border-ice/30',
  high: 'text-rose border-rose/35',
} as const

export function FeatureImpactList({ features }: { features: FeatureContribution[] }) {
  if (features.length === 0) return <EmptyNote text="No feature data returned for this article." />
  return (
    <ul className="space-y-3">
      {features.map((f, i) => (
        <li key={`${f.feature}-${i}`} className="flex items-center gap-4">
          <span className="w-56 shrink-0 truncate font-mono text-xs text-fog-200" title={f.feature}>
            {f.feature}
          </span>
          <div className="h-1 flex-1 overflow-hidden rounded-full bg-white/[0.06]">
            <div
              className={cn(
                'h-full rounded-full',
                f.impact === 'high' ? 'bg-rose/70' : f.impact === 'medium' ? 'bg-ice/60' : 'bg-fog-500/70'
              )}
              style={{ width: `${Math.min(100, Math.max(6, Math.abs(f.value) * 100))}%` }}
            />
          </div>
          <span className="w-20 shrink-0 text-right font-mono text-xs text-fog-500">{f.value.toFixed(3)}</span>
          <span
            className={cn(
              'w-16 shrink-0 rounded-full border px-2 py-0.5 text-center text-[10px] font-medium uppercase tracking-wider',
              impactTone[f.impact]
            )}
          >
            {f.impact}
          </span>
        </li>
      ))}
    </ul>
  )
}

export function LinguisticFlags({ flags }: { flags: string[] }) {
  if (flags.length === 0) return <EmptyNote text="No unusual linguistic patterns were detected in this article." />
  return (
    <ul className="space-y-2.5">
      {flags.map((flag, i) => (
        <li key={i} className="flex items-start gap-3 text-sm text-fog-300">
          <span className="mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full bg-rose/60" />
          {flag}
        </li>
      ))}
    </ul>
  )
}

export function ModelContributionsTable({
  contributions,
}: {
  contributions: Record<string, ModelContribution>
}) {
  const entries = Object.entries(contributions)
  if (entries.length === 0) return <EmptyNote text="No per-model contributions were returned." />
  return (
    <div className="overflow-hidden rounded-xl border border-line">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-line bg-white/[0.02] text-left text-[10px] uppercase tracking-widest text-fog-600">
            <th className="px-4 py-2.5 font-medium">Model</th>
            <th className="px-4 py-2.5 font-medium">Prediction</th>
            <th className="px-4 py-2.5 font-medium">Confidence</th>
            <th className="px-4 py-2.5 text-right font-medium">Weight</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([name, c]) => (
            <tr key={name} className="border-b border-line/60 last:border-0 hover:bg-white/[0.02]">
              <td className="px-4 py-2.5 text-fog-200">{formatModelName(name)}</td>
              <td className="px-4 py-2.5">
                <VerdictBadge verdict={c.prediction} />
              </td>
              <td className="px-4 py-2.5">
                <div className="flex items-center gap-2.5">
                  <div className="h-1 w-24 overflow-hidden rounded-full bg-white/[0.06]">
                    <div className="h-full rounded-full bg-ice/60" style={{ width: `${c.confidence * 100}%` }} />
                  </div>
                  <span className="font-mono text-xs text-fog-400">{(c.confidence * 100).toFixed(1)}%</span>
                </div>
              </td>
              <td className="px-4 py-2.5 text-right font-mono text-xs text-fog-600">{c.confidence.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function formatModelName(name: string): string {
  return name
    .replace(/[_-]/g, ' ')
    .replace(/(meta.*classifier)/, 'Stacked meta-classifier')
    .split(' ')
    .map((w) => (w.length <= 3 && w === w.toLowerCase() ? w : w[0].toUpperCase() + w.slice(1)))
    .join(' ')
}

function EmptyNote({ text }: { text: string }) {
  return <p className="rounded-xl border border-line bg-white/[0.02] px-4 py-3 text-sm text-fog-500">{text}</p>
}
