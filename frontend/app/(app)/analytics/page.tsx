'use client'

import { useMemo, useState } from 'react'
import Link from 'next/link'
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  AreaChart,
  Area,
} from 'recharts'
import { PageHeader, GlassPanel, Tabs } from '@/components/ui/misc'
import { Button } from '@/components/ui/button'
import EmptyState from '@/components/EmptyState'
import { useHistory } from '@/lib/history'
import { exportHistoryCsv, exportHistoryJson } from '@/lib/export'
import { Download, ArrowRight } from 'lucide-react'
import { cn } from '@/lib/cn'

const tooltipStyle = {
  backgroundColor: 'rgba(10, 10, 12, 0.92)',
  border: '1px solid rgba(255,255,255,0.1)',
  borderRadius: 10,
  fontSize: 12,
  color: '#ededf0',
}

function dayLabel(iso: string) {
  try {
    return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
  } catch {
    return iso
  }
}

const SAMPLE_DISTRIBUTION = [
  { day: 'Oct 24', real: 64, fake: 38 },
  { day: 'Oct 25', real: 88, fake: 46 },
  { day: 'Oct 26', real: 71, fake: 55 },
  { day: 'Oct 27', real: 96, fake: 40 },
  { day: 'Oct 28', real: 104, fake: 58 },
  { day: 'Oct 29', real: 92, fake: 44 },
  { day: 'Oct 30', real: 118, fake: 62 },
]

const SAMPLE_CONFIDENCE = [
  { band: '50–60%', real: 18, fake: 26 },
  { band: '60–70%', real: 44, fake: 61 },
  { band: '70–80%', real: 96, fake: 130 },
  { band: '80–90%', real: 210, fake: 240 },
  { band: '90–95%', real: 380, fake: 340 },
  { band: '95–100%', real: 640, fake: 610 },
]

export default function AnalyticsPage() {
  const entries = useHistory()
  const [chartTab, setChartTab] = useState<string>('distribution')
  const [useSample, setUseSample] = useState(false)

  const distribution = useMemo(() => {
    if (entries.length === 0 && useSample) return SAMPLE_DISTRIBUTION
    const byDay = new Map<string, { day: string; real: number; fake: number }>()
    const now = Date.parse(new Date().toISOString().slice(0, 10))
    for (let i = 6; i >= 0; i--) {
      const key = new Date(now - i * 86400000)
      const label = key.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
      byDay.set(key.toISOString().slice(0, 10), { day: label, real: 0, fake: 0 })
    }
    for (const e of entries) {
      const key = new Date(e.timestamp).toISOString().slice(0, 10)
      const cell = byDay.get(key)
      if (cell) cell[e.verdict === 'REAL' ? 'real' : 'fake']++
    }
    return [...byDay.values()]
  }, [entries, useSample])

  const confidence = useMemo(() => {
    if (entries.length === 0 && useSample) return SAMPLE_CONFIDENCE
    const bands = ['50–60%', '60–70%', '70–80%', '80–90%', '90–95%', '95–100%']
    const idx = (c: number) => (c < 0.6 ? 0 : c < 0.7 ? 1 : c < 0.8 ? 2 : c < 0.9 ? 3 : c < 0.95 ? 4 : 5)
    const acc = bands.map((band) => ({ band, real: 0, fake: 0 }))
    for (const e of entries) acc[idx(e.confidence)][e.verdict === 'REAL' ? 'real' : 'fake']++
    return acc
  }, [entries, useSample])

  const errorAnalysis = useMemo(() => {
    const total = entries.length
    const fake = entries.filter((e) => e.verdict === 'FAKE')
    const real = total - fake.length
    const highRisk = entries.filter((e) => e.risk_level === 'HIGH').length
    const lowConf = entries.filter((e) => e.confidence < 0.7).length
    const avgFake = fake.length ? fake.reduce((s, e) => s + e.confidence, 0) / fake.length : 0
    return { total, real, fake: fake.length, highRisk, lowConf, avgFake }
  }, [entries])

  const hasData = entries.length > 0

  return (
    <div>
      <PageHeader title="Analytics" description="Trends, distributions and export for your local analyses.">
        <Button
          variant="glass"
          size="sm"
          disabled={!hasData}
          onClick={() => exportHistoryCsv(entries)}
        >
          <Download className="h-3.5 w-3.5" /> Export CSV
        </Button>
        <Button
          variant="ghost"
          size="sm"
          disabled={!hasData}
          onClick={() => exportHistoryJson(entries)}
        >
          <Download className="h-3.5 w-3.5" /> JSON
        </Button>
      </PageHeader>

      {!hasData && !useSample && (
        <div className="mb-6">
          <EmptyState
            title="Analytics is waiting for data"
            description="Charts populate as you analyze articles. Inspect the UI with illustrative sample data in the meantime."
            action={
              <Button variant="primary" onClick={() => setUseSample(true)}>
                Show sample data
              </Button>
            }
          />
        </div>
      )}

      <div className={cn(!hasData && !useSample && 'hidden')}>
        <div className="mb-4 flex flex-wrap items-center gap-3">
          <Tabs
            value={chartTab}
            onChange={setChartTab}
            items={[
              { id: 'distribution', label: 'Real vs Fake over time' },
              { id: 'confidence', label: 'Confidence distribution' },
              { id: 'errors', label: 'Error analysis' },
            ]}
          />
          {!hasData && (
            <span className="rounded-full border border-amber/30 bg-amber/[0.06] px-3 py-1 text-[10px] uppercase tracking-wider text-amber">
              Sample data
            </span>
          )}
        </div>

        {chartTab === 'distribution' && (
          <GlassPanel className="p-6">
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                {hasData ? (
                  <BarChart data={distribution} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
                    <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
                    <XAxis dataKey="day" tick={{ fill: '#66666f', fontSize: 10 }} tickLine={false} axisLine={false} />
                    <YAxis tick={{ fill: '#66666f', fontSize: 10 }} tickLine={false} axisLine={false} allowDecimals={false} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Legend wrapperStyle={{ fontSize: 11, color: '#8f8f9c' }} />
                    <Bar dataKey="real" stackId="a" fill="#e3e6ee" radius={[0, 0, 0, 0]} maxBarSize={38} />
                    <Bar dataKey="fake" stackId="a" fill="#4a4a54" radius={[3, 3, 0, 0]} maxBarSize={38} />
                  </BarChart>
                ) : (
                  <AreaChart data={distribution} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
                    <defs>
                      <linearGradient id="aReal" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#e6eaf2" stopOpacity={0.35} />
                        <stop offset="100%" stopColor="#e6eaf2" stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="aFake" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#4a4a54" stopOpacity={0.4} />
                        <stop offset="100%" stopColor="#4a4a54" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
                    <XAxis dataKey="day" tick={{ fill: '#66666f', fontSize: 10 }} tickLine={false} axisLine={false} />
                    <YAxis tick={{ fill: '#66666f', fontSize: 10 }} tickLine={false} axisLine={false} allowDecimals={false} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Legend wrapperStyle={{ fontSize: 11, color: '#8f8f9c' }} />
                    <Area type="monotone" dataKey="real" stroke="#d4d4dc" fill="url(#aReal)" strokeWidth={1.6} />
                    <Area type="monotone" dataKey="fake" stroke="#6a6a76" fill="url(#aFake)" strokeWidth={1.6} />
                  </AreaChart>
                )}
              </ResponsiveContainer>
            </div>
          </GlassPanel>
        )}

        {chartTab === 'confidence' && (
          <GlassPanel className="p-6">
            <p className="mb-5 text-xs text-fog-500">
              Model probability distribution of your past analyses, grouped by verdict.
            </p>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={confidence} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
                  <XAxis dataKey="band" tick={{ fill: '#66666f', fontSize: 10 }} tickLine={false} axisLine={false} />
                  <YAxis tick={{ fill: '#66666f', fontSize: 10 }} tickLine={false} axisLine={false} allowDecimals={false} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Legend wrapperStyle={{ fontSize: 11, color: '#8f8f9c' }} />
                  <Bar dataKey="real" fill="#e3e6ee" maxBarSize={42} radius={[3, 3, 0, 0]} />
                  <Bar dataKey="fake" fill="#4a4a54" maxBarSize={42} radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </GlassPanel>
        )}

        {chartTab === 'errors' && (
          <div className="grid gap-4 md:grid-cols-3">
            {[
              { label: 'Analyses flagged HIGH risk', value: errorAnalysis.highRisk, note: 'Treat these with extra caution.' },
              { label: 'Low-confidence verdicts (<70%)', value: errorAnalysis.lowConf, note: 'Closer to the model boarder — reread manually.' },
              {
                label: 'Mean confidence for FAKE verdicts',
                value: errorAnalysis.fake ? `${(errorAnalysis.avgFake * 100).toFixed(1)}%` : '—',
                note: 'Across your recorded history.',
              },
            ].map((c) => (
              <GlassPanel key={c.label} className="p-5">
                <p className="text-[11px] uppercase tracking-widest text-fog-500">{c.label}</p>
                <p className="mt-3 text-3xl font-medium text-fog-50">{c.value}</p>
                <p className="mt-2 text-xs text-fog-600">{c.note}</p>
              </GlassPanel>
            ))}
          </div>
        )}
      </div>

    </div>
  )
}
