'use client'

import { useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { Search as SearchIcon, Download, Trash2, X, ChevronDown } from 'lucide-react'
import { PageHeader, GlassPanel, Tabs } from '@/components/ui/misc'
import { Button } from '@/components/ui/button'
import { VerdictBadge, RiskChip, ConfidenceGauge } from '@/components/verdict'
import { FeatureImpactList, LinguisticFlags, ModelContributionsTable } from '@/components/explanation'
import EmptyState from '@/components/EmptyState'
import { useHistory, clearHistory, type HistoryEntry } from '@/lib/history'
import { exportHistoryCsv } from '@/lib/export'
import { cn } from '@/lib/cn'

function formatWhen(iso: string) {
  try {
    return new Date(iso).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
  } catch {
    return iso
  }
}

export default function HistoryPage() {
  const entries = useHistory()
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState('all')
  const [selected, setSelected] = useState<HistoryEntry | null>(null)
  const [showTab, setShowTab] = useState<string>('summary')
  const [confirmingClear, setConfirmingClear] = useState(false)
  const [visible, setVisible] = useState(10)

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    return entries.filter((e) => {
      if (filter !== 'all' && e.verdict !== filter) return false
      if (q && !e.original_input.toLowerCase().includes(q)) return false
      return true
    })
  }, [entries, query, filter])

  const openDetail = (entry: HistoryEntry) => {
    setSelected(entry)
    setShowTab('summary')
  }

  return (
    <div>
      <PageHeader title="History" description="Your past news analysis results.">
        <Button variant="glass" size="sm" disabled={entries.length === 0} onClick={() => exportHistoryCsv(entries)}>
          <Download className="h-3.5 w-3.5" /> Export CSV
        </Button>
        <Button
          variant="ghost"
          size="sm"
          disabled={entries.length === 0}
          onClick={() => {
            if (confirmingClear) {
              clearHistory()
              setConfirmingClear(false)
            } else {
              setConfirmingClear(true)
              setTimeout(() => setConfirmingClear(false), 4000)
            }
          }}
          className={cn(confirmingClear && 'text-rose')}
        >
          <Trash2 className="h-3.5 w-3.5" /> {confirmingClear ? 'Click again to confirm' : 'Clear'}
        </Button>
      </PageHeader>

      <div className="mb-6 flex flex-wrap items-center gap-3">
        <div className="relative min-w-[240px] flex-1">
          <SearchIcon className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-fog-600" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search articles…"
            className="h-10 w-full rounded-full border border-line bg-white/[0.03] pl-10 pr-4 text-sm text-fog-100 placeholder:text-fog-600 outline-none transition-colors focus:border-fog-500/60"
          />
        </div>
        <Tabs
          value={filter}
          onChange={setFilter}
          items={[
            { id: 'all', label: 'All Results' },
            { id: 'REAL', label: 'Real' },
            { id: 'FAKE', label: 'Fake' },
          ]}
        />
      </div>

      {entries.length === 0 ? (
        <EmptyState title="No history yet" description="Analyzed articles are stored locally on this device and appear here." />
      ) : filtered.length === 0 ? (
        <EmptyState title="No matches" description="Try a different search term or filter." />
      ) : (
        <>
          <GlassPanel className="overflow-hidden p-0">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[640px] text-sm">
                <thead>
                  <tr className="border-b border-line bg-white/[0.02] text-left text-[10px] uppercase tracking-widest text-fog-600">
                    <th className="px-6 py-3.5 font-medium">Article (Preview)</th>
                    <th className="px-4 py-3.5 font-medium">Result</th>
                    <th className="px-4 py-3.5 font-medium">Confidence</th>
                    <th className="px-4 py-3.5 font-medium">Risk</th>
                    <th className="px-4 py-3.5 text-right font-medium">Date</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.slice(0, visible).map((e) => (
                    <tr
                      key={e.id}
                      onClick={() => openDetail(e)}
                      className="cursor-pointer border-b border-line/50 last:border-0 hover:bg-white/[0.025]"
                    >
                      <td className="max-w-sm px-6 py-3.5">
                        <p className="truncate text-fog-200">{e.original_input.replace(/\s+/g, ' ').slice(0, 120)}</p>
                      </td>
                      <td className="px-4 py-3.5">
                        <VerdictBadge verdict={e.verdict} />
                      </td>
                      <td className="px-4 py-3.5 font-mono text-fog-300">{(e.confidence * 100).toFixed(1)}%</td>
                      <td className="px-4 py-3.5">
                        <RiskChip level={e.risk_level} />
                      </td>
                      <td className="px-4 py-3.5 text-right text-xs text-fog-500">{formatWhen(e.timestamp)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </GlassPanel>
          {visible < filtered.length && (
            <div className="mt-4 text-center">
              <Button variant="ghost" onClick={() => setVisible((v) => v + 10)}>
                Show older analysis <ChevronDown className="h-3.5 w-3.5" />
              </Button>
            </div>
          )}
        </>
      )}

      {selected && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <div className="absolute inset-0 bg-black/75 backdrop-blur-sm" onClick={() => setSelected(null)} />
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 0.3, ease: 'easeOut' }}
            className="glass-strong relative z-10 max-h-[85vh] w-full max-w-2xl overflow-y-auto rounded-2xl p-6 shadow-card md:p-8"
          >
            <button
              type="button"
              onClick={() => setSelected(null)}
              className="absolute right-5 top-5 text-fog-500 hover:text-fog-100"
              aria-label="Close detail"
            >
              <X className="h-4 w-4" />
            </button>
            <div className="flex flex-wrap items-center gap-4">
              <VerdictBadge verdict={selected.verdict} large />
              <div className="space-y-1 text-sm text-fog-400">
                <p className="font-mono text-fog-300">{(selected.confidence * 100).toFixed(1)}% confidence</p>
                <p>
                  {selected.model} · {selected.inferenceMs} ms · {formatWhen(selected.timestamp)}
                </p>
              </div>
              <div className="ml-auto">
                <ConfidenceGauge value={selected.confidence} label="" />
              </div>
            </div>
            <div className="mt-5 flex items-center gap-3 border-t border-line pt-4 text-sm">
              <RiskChip level={selected.risk_level} />
              <span className="text-fog-500">
                Source Credibility: <span className="text-fog-300">{selected.source_credibility.label}</span>
              </span>
            </div>
            <p className="mt-5 rounded-xl border border-line bg-white/[0.02] px-4 py-3 text-sm leading-relaxed text-fog-300">
              {selected.original_input.replace(/\s+/g, ' ').slice(0, 400)}
            </p>
            <div className="mt-6">
              <Tabs
                value={showTab}
                onChange={setShowTab}
                items={[
                  { id: 'summary', label: 'Explanation' },
                  { id: 'features', label: 'Top Features' },
                  { id: 'models', label: 'Model Contributions' },
                ]}
              />
              <div className="mt-5 text-sm">
                {showTab === 'summary' && <LinguisticFlags flags={selected.explanation.linguistic_flags} />}
                {showTab === 'features' && <FeatureImpactList features={selected.explanation.top_features} />}
                {showTab === 'models' && (
                  <ModelContributionsTable contributions={selected.explanation.model_contributions} />
                )}
              </div>
            </div>
          </motion.div>
        </div>
      )}

    </div>
  )
}
