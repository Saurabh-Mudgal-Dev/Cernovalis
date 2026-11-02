'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { ExternalLink } from 'lucide-react'
import { PageHeader, GlassPanel } from '@/components/ui/misc'
import { Button } from '@/components/ui/button'
import { Input, Select } from '@/components/ui/field'
import { getApiBase, setApiBase, fetchHealth } from '@/lib/api'
import { clearHistory, getHistory } from '@/lib/history'
import { DEFAULT_MODEL_KEY, MODEL_OPTIONS } from '@/lib/metrics'

export default function SettingsPage() {
  const [base, setBase] = useState('')
  const [savedBase, setSavedBase] = useState('')
  const [defaultModel, setDefaultModel] = useState<string>(MODEL_OPTIONS[0])
  const [healthText, setHealthText] = useState<string | null>(null)
  const [historyCount, setHistoryCount] = useState(0)

  useEffect(() => {
    setSavedBase(getApiBase())
    setBase(getApiBase())
    setDefaultModel(window.localStorage.getItem(DEFAULT_MODEL_KEY) ?? MODEL_OPTIONS[0])
    setHistoryCount(getHistory().length)
  }, [])

  const saveBase = () => {
    setApiBase(base.trim() || null)
    setSavedBase(getApiBase())
  }

  const saveModel = (value: string) => {
    setDefaultModel(value)
    window.localStorage.setItem(DEFAULT_MODEL_KEY, value)
  }

  const checkHealth = async () => {
    setHealthText('Checking…')
    const h = await fetchHealth()
    setHealthText(h.status === 'ok' ? 'Online — the API answered /health with "ok".' : 'Offline — cannot reach the API.')
  }

  return (
    <div className="max-w-3xl">
      <PageHeader title="Settings" description="Preferences, API connection and data management." />

      <div className="space-y-5">
        <GlassPanel className="p-6">
          <h2 className="text-sm font-medium text-fog-200">API Connection</h2>
          <p className="mt-1.5 text-xs leading-relaxed text-fog-500">
            Base URL of the FastAPI backend. Leave empty to use the built-in Next.js proxy to
            <span className="font-mono text-fog-300"> http://localhost:8000</span> — no CORS changes needed.
          </p>
          <div className="mt-4 flex gap-3">
            <Input
              value={base}
              onChange={(e) => setBase(e.target.value)}
              placeholder="(default) proxied to http://localhost:8000"
              className="h-11 font-mono text-xs"
            />
            <Button variant="primary" onClick={saveBase} className="shrink-0">
              Save
            </Button>
          </div>
          <p className="mt-2.5 font-mono text-[11px] text-fog-600">
            active: {savedBase || 'same-origin (proxy /api → localhost:8000)'}
          </p>
          <div className="mt-4 border-t border-line pt-4">
            <Button variant="glass" size="sm" onClick={checkHealth}>
              Test connection
            </Button>
            {healthText && <p className="mt-2.5 text-xs text-fog-400">{healthText}</p>}
          </div>
        </GlassPanel>

        <GlassPanel className="p-6">
          <h2 className="text-sm font-medium text-fog-200">Default Model</h2>
          <p className="mt-1.5 text-xs text-fog-500">Preselected on the Detect News page.</p>
          <Select
            ariaLabel="Default model"
            value={defaultModel}
            onChange={saveModel}
            options={MODEL_OPTIONS as unknown as string[]}
            className="mt-4"
          />
          <p className="mt-2.5 text-[11px] text-fog-600">
            Auto uses the backend stacked ensemble; a fixed choice is honored when that artifact is loaded, otherwise the ensemble predicts.
          </p>
        </GlassPanel>

        <GlassPanel className="p-6">
          <h2 className="text-sm font-medium text-fog-200">Local Data</h2>
          <p className="mt-1.5 text-xs text-fog-500">
            {historyCount} analysis record{historyCount === 1 ? '' : 's'} stored in this browser.
          </p>
          <Button
            variant="ghost"
            size="sm"
            disabled={historyCount === 0}
            onClick={() => {
              clearHistory()
              setHistoryCount(0)
            }}
            className="mt-3 text-rose"
          >
            Clear history
          </Button>
        </GlassPanel>

        <GlassPanel className="p-6">
          <h2 className="text-sm font-medium text-fog-200">About</h2>
          <p className="mt-2 text-xs leading-relaxed text-fog-500">
            Cernovalis classifies news articles with an ensemble of classical TF·IDF models and a
            GloVe-initialized BiLSTM, combined through soft voting and a stacked meta-classifier.
            Benchmark results, preprocessing details and limitations are documented in
            <span className="font-mono text-fog-300"> docs/model-card.md</span>.
          </p>
          <div className="mt-4 flex flex-wrap gap-3">
            <Link href="/model-comparison">
              <Button variant="glass" size="sm">
                <ExternalLink className="h-3.5 w-3.5" /> View model comparison
              </Button>
            </Link>
          </div>
          <div className="mt-4 flex items-center justify-between border-t border-line pt-4 text-xs text-fog-600">
            <span>Theme</span>
            <span className="rounded-full border border-line-strong bg-white/[0.03] px-3 py-1 text-fog-300">Dark · fixed</span>
          </div>
        </GlassPanel>
      </div>
    </div>
  )
}
