'use client'

import Link from 'next/link'
import { useMemo } from 'react'
import {
  FileSearch,
  Fingerprint,
  ShieldCheck,
  Target,
  Zap,
  Trophy,
  ArrowRight,
} from 'lucide-react'
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts'
import { PageHeader } from '@/components/ui/misc'
import { Button } from '@/components/ui/button'
import EmptyState from '@/components/EmptyState'
import { useHistory } from '@/lib/history'
import { MODELS } from '@/lib/metrics'

function dayKey(iso: string) {
  try {
    return new Date(iso).toISOString().slice(0, 10)
  } catch {
    return ''
  }
}

function labelFor(key: string) {
  const d = new Date(key)
  if (Number.isNaN(d.getTime())) return key
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

const tooltipStyle = {
  backgroundColor: 'rgba(10, 10, 12, 0.92)',
  border: '1px solid rgba(255,255,255,0.1)',
  borderRadius: 10,
  fontSize: 12,
  color: '#ededf0',
}

export default function DashboardPage() {
  const entries = useHistory()

  const stats = useMemo(() => {
    const total = entries.length
    const fake = entries.filter((e) => e.verdict === 'FAKE').length
    const real = total - fake
    const avgConf = total ? entries.reduce((s, e) => s + e.confidence, 0) / total : null
    const fastest = entries.length
      ? entries.reduce((a, b) => (b.inferenceMs < a.inferenceMs ? b : a))
      : null
    const best = MODELS.reduce((a, b) => (b.accuracy > a.accuracy ? b : a))
    return { total, fake, real, avgConf, fastest, best }
  }, [entries])

  const trend = useMemo(() => {
    const byDay = new Map<string, { day: string; real: number; fake: number }>()
    const now = Date.parse(new Date().toISOString().slice(0, 10))
    for (let i = 13; i >= 0; i--) {
      const key = new Date(now - i * 86400000).toISOString().slice(0, 10)
      byDay.set(key, { day: labelFor(key), real: 0, fake: 0 })
    }
    for (const e of entries) {
      const cell = byDay.get(dayKey(e.timestamp))
      if (cell) cell[e.verdict === 'REAL' ? 'real' : 'fake']++
    }
    return [...byDay.values()]
  }, [entries])

  const best = MODELS.reduce((a, b) => (b.accuracy > a.accuracy ? b : a))

  const cards = [
    { icon: FileSearch, label: 'Articles Analyzed', value: stats.total.toLocaleString(), sub: 'All time' },
    { icon: Fingerprint, label: 'Detected Fake', value: stats.fake.toLocaleString(), sub: 'Flagged as FAKE', tone: 'text-rose' },
    { icon: ShieldCheck, label: 'Detected Real', value: stats.real.toLocaleString(), sub: 'Assessed as REAL', tone: 'text-leaf' },
    {
      icon: Target,
      label: 'Avg. Confidence',
      value: stats.avgConf === null ? '—' : `${(stats.avgConf * 100).toFixed(1)}%`,
      sub: 'Mean of verdicts',
    },
    {
      icon: Zap,
      label: 'Fastest Model',
      value: stats.fastest ? stats.fastest.model : '—',
      sub: stats.fastest ? `${stats.fastest.inferenceMs} ms round-trip` : 'No runs yet',
    },
    { icon: Trophy, label: 'Best Accuracy', value: '99.6%', sub: best.name },
  ]

  if (entries.length === 0) {
    return (
      <div>
        <PageHeader title="Dashboard" description="Overview of your fake news detection system." />
        <EmptyState
          title="No analyses yet"
          description="Your dashboard populates automatically after your first analysis."
          action={
            <Link href="/detect">
              <Button variant="primary">
                Start Analysis <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
          }
        />
        <div className="mt-4 grid grid-cols-2 gap-4 md:grid-cols-3">
          {cards.map((c) => (
            <div key={c.label} className="glass rounded-2xl p-5 opacity-50">
              <p className="text-[11px] uppercase tracking-widest text-fog-500">{c.label}</p>
              <p className="mt-2 text-2xl font-medium text-fog-300">—</p>
            </div>
          ))}
        </div>
      </div>
    )
  }

  const donut = [
    { name: 'Real', value: stats.real },
    { name: 'Fake', value: stats.fake },
  ]

  return (
    <div>
      <PageHeader title="Dashboard" description="Overview of your fake news detection system.">
        <Link href="/history">
          <Button variant="ghost" size="sm">
            View history <ArrowRight className="h-3.5 w-3.5" />
          </Button>
        </Link>
      </PageHeader>

      <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
        {cards.map(({ icon: Icon, label, value, sub, tone }) => (
          <div key={label} className="glass rounded-2xl p-5 shadow-inner">
            <div className="flex items-center justify-between">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl border border-line bg-white/[0.03] text-fog-300">
                <Icon className="h-4 w-4" />
              </div>
            </div>
            <p className="mt-4 text-2xl font-medium text-fog-50">{value}</p>
            <p className={`mt-0.5 text-xs ${tone ?? 'text-fog-400'}`}>{label}</p>
            <p className="mt-2 truncate text-[11px] text-fog-600">{sub}</p>
          </div>
        ))}
      </div>

      <div className="mt-6 grid gap-6 xl:grid-cols-[3fr_2fr]">
        <div className="glass rounded-2xl p-6">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-sm font-medium text-fog-200">Analysis Trend</h2>
            <div className="flex items-center gap-4 text-[10px] uppercase tracking-widest text-fog-600">
              <span className="flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-fog-200" /> Real
              </span>
              <span className="flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-fog-600" /> Fake
              </span>
            </div>
          </div>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trend} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
                <defs>
                  <linearGradient id="gradReal" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#e6eaf2" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="#e6eaf2" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="gradFake" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#4a4a54" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="#4a4a54" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="day" tick={{ fill: '#66666f', fontSize: 10 }} tickLine={false} axisLine={false} interval={2} />
                <YAxis tick={{ fill: '#66666f', fontSize: 10 }} tickLine={false} axisLine={false} allowDecimals={false} />
                <Tooltip contentStyle={tooltipStyle} />
                <Area type="monotone" dataKey="real" stroke="#d4d4dc" strokeWidth={1.6} fill="url(#gradReal)" />
                <Area type="monotone" dataKey="fake" stroke="#6a6a76" strokeWidth={1.6} fill="url(#gradFake)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="glass rounded-2xl p-6">
          <h2 className="mb-4 text-sm font-medium text-fog-200">News Distribution</h2>
          <div className="flex items-center gap-6">
            <div className="relative h-44 w-44 shrink-0">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={donut}
                    dataKey="value"
                    innerRadius={58}
                    outerRadius={78}
                    paddingAngle={3}
                    strokeWidth={0}
                    startAngle={90}
                    endAngle={-270}
                  >
                    <Cell fill="#e3e6ee" />
                    <Cell fill="#3f3f49" />
                  </Pie>
                  <Tooltip contentStyle={tooltipStyle} />
                </PieChart>
              </ResponsiveContainer>
              <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-lg font-medium text-fog-50">{stats.total.toLocaleString()}</span>
                <span className="text-[10px] uppercase tracking-widest text-fog-600">Articles</span>
              </div>
            </div>
            <div className="space-y-3 text-sm">
              <div className="flex items-center gap-2.5">
                <span className="h-2 w-2 rounded-full bg-fog-200" />
                <span className="text-fog-300">
                  {stats.real} <span className="text-fog-600">real</span>
                </span>
              </div>
              <div className="flex items-center gap-2.5">
                <span className="h-2 w-2 rounded-full bg-fog-600" />
                <span className="text-fog-300">
                  {stats.fake} <span className="text-fog-600">fake</span>
                </span>
              </div>
              {stats.total > 0 && (
                <p className="pt-2 text-xs text-fog-600">
                  {(stats.real / stats.total * 100).toFixed(0)}% of your articles were assessed as real.
                </p>
              )}
            </div>
          </div>
        </div>
      </div>

      </div>
    )
}
