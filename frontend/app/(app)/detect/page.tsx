'use client'

import { useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { FileText, Sparkles, ArrowRight } from 'lucide-react'
import DataSphere from '@/components/DataSphere'
import BatchUpload from '@/components/BatchUpload'
import { VerdictBadge, RiskChip, ConfidenceGauge } from '@/components/verdict'
import { FeatureImpactList, LinguisticFlags, ModelContributionsTable } from '@/components/explanation'
import { Button } from '@/components/ui/button'
import { Textarea, Select } from '@/components/ui/field'
import { GlassPanel, Tabs } from '@/components/ui/misc'
import { analyzeArticle } from '@/lib/api'
import { addHistory } from '@/lib/history'
import { DEFAULT_MODEL_KEY, MODEL_OPTIONS } from '@/lib/metrics'
import type { AnalyzeResult } from '@/lib/schemas'
import { cn } from '@/lib/cn'

const STEP_LABELS = ['Preprocessing', 'Model Inference', 'Generating Explanation']

function formatDate(iso: string) {
  try {
    return new Date(iso).toLocaleString()
  } catch {
    return iso
  }
}

export default function DetectPage() {
  const [mode, setMode] = useState<string>('text')
  const [text, setText] = useState('')
  const [model, setModel] = useState<string>(MODEL_OPTIONS[0] as string)
  const [analyzing, setAnalyzing] = useState(false)
  const [result, setResult] = useState<AnalyzeResult | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const [clickedModel, setClickedModel] = useState<string>(MODEL_OPTIONS[0] as string)
  const [error, setError] = useState<string | null>(null)
  const [explanationTab, setExplanationTab] = useState<string>('summary')

  useEffect(() => {
    const saved = window.localStorage.getItem(DEFAULT_MODEL_KEY)
    if (saved) setModel(saved)
  }, [])

  const stepIndex = useMemo(() => {
    if (!analyzing) return -1
    const bucket = elapsed * 2.2
    return Math.min(STEP_LABELS.length - 1, Math.floor(bucket / 600))
  }, [analyzing, elapsed])

  useEffect(() => {
    if (!analyzing) return
    const id = setInterval(() => setElapsed((e) => e + 50), 50)
    return () => clearInterval(id)
  }, [analyzing])

  const run = async () => {
    if (!text.trim() || analyzing) return
    setAnalyzing(true)
    setResult(null)
    setError(null)
    setElapsed(0)
    setClickedModel(model)
    const t0 = performance.now()
    try {
      const res = await analyzeArticle({ text: text.trim() })
      const ms = Math.round(performance.now() - t0)
      setResult(res)
      setElapsed(ms)
      addHistory({ ...res, model: model === MODEL_OPTIONS[0] ? 'auto (ensemble)' : model, inferenceMs: ms })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Something went wrong.')
    } finally {
      setAnalyzing(false)
    }
  }

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-medium tracking-tight text-fog-50 md:text-3xl">Detect News</h1>
        <p className="mt-1.5 text-sm text-fog-500">Analyze a news article to check its credibility.</p>
      </div>

      <div className="grid gap-8 xl:grid-cols-[minmax(0,1fr)_320px]">
        <div className="space-y-6">
          <GlassPanel className="p-6 shadow-card md:p-8">
            <div className="mb-6 flex items-start gap-3">
              <span className="flex h-10 w-10 items-center justify-center rounded-xl border border-line bg-white/[0.03] text-fog-300">
                <FileText className="h-4 w-4" />
              </span>
            </div>
            <Tabs
              className="mb-5"
              value={mode}
              onChange={setMode}
              items={[
                { id: 'text', label: 'Article Text' },
                { id: 'batch', label: 'Batch CSV' },
                { id: 'url', label: 'URL', hint: 'Coming soon' },
              ]}
            />

            {mode === 'text' && (
              <div className="space-y-5">
                <div className="space-y-2">
                  <Textarea
                    rows={9}
                    maxLength={100000}
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="Paste the news article here…"
                    className="min-h-[220px]"
                  />
                  <div className="text-right font-mono text-[11px] text-fog-600">
                    {text.length.toLocaleString()} / 100,000
                  </div>
                </div>
                <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
                  <label className="text-xs font-medium uppercase tracking-wider text-fog-500" htmlFor="model">
                    Select Model
                  </label>
                  <Select
                    id="model"
                    ariaLabel="Select model"
                    value={model}
                    onChange={setModel}
                    options={MODEL_OPTIONS as unknown as string[]}
                    className="flex-1"
                    triggerClassName="h-12"
                  />
                  <Button
                    variant="primary"
                    size="lg"
                    loading={analyzing}
                    disabled={!text.trim()}
                    onClick={run}
                    className="sm:w-auto w-full"
                  >
                    {analyzing ? 'Analyzing…' : 'Analyze'}
                    {!analyzing && <ArrowRight className="h-4 w-4" />}
                  </Button>
                </div>
                <p className="text-xs text-fog-600">
                  Auto routes the article through the stacked ensemble of all loaded models.
                </p>
              </div>
            )}

            {mode === 'url' && (
              <div className="rounded-xl border border-line bg-white/[0.02] px-4 py-8 text-center text-sm text-fog-500">
                URL analysis is not available yet.
              </div>
            )}

            {mode === 'batch' && <BatchUpload />}
          </GlassPanel>

          {error && (
            <p className="glass rounded-xl border-rose/25 px-4 py-3 text-sm text-rose">{error}</p>
          )}

          {result && (
            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.45, ease: 'easeOut' }}
              className="space-y-4"
            >
              <GlassPanel className="glass-strong p-6 shadow-card md:p-8">
                <div className="flex flex-col gap-8 md:flex-row md:items-center md:justify-between">
                  <div className="space-y-3">
                    <VerdictBadge verdict={result.verdict} large />
                    <p className="max-w-xs text-sm leading-relaxed text-fog-400">
                      {result.verdict === 'REAL'
                        ? 'This article is likely to be legitimate and credible.'
                        : 'This article shows patterns commonly found in fabricated or sensationalized content.'}
                    </p>
                  </div>
                  <div className="flex items-center gap-8">
                    <div className="space-y-3 text-sm">
                      <div>
                        <p className="text-[10px] uppercase tracking-widest text-fog-600">Confidence</p>
                        <p className="font-mono text-fog-100">{(result.confidence * 100).toFixed(1)}%</p>
                      </div>
                      <div>
                        <p className="text-[10px] uppercase tracking-widest text-fog-600">Model</p>
                        <p className="text-fog-100">
                          {clickedModel === MODEL_OPTIONS[0] ? 'Stacked Ensemble' : clickedModel}
                        </p>
                      </div>
                      <div>
                        <p className="text-[10px] uppercase tracking-widest text-fog-600">Inference Time</p>
                        <p className="font-mono text-fog-100">{elapsed} ms</p>
                      </div>
                    </div>
                    <ConfidenceGauge value={result.confidence} />
                  </div>
                </div>

                <div className="mt-6 flex flex-wrap items-center gap-3 border-t border-line pt-5 text-sm">
                  <span className="text-fog-500">Risk Level:</span>
                  <RiskChip level={result.risk_level} />
                  <span className="ml-4 text-fog-500">Source Credibility:</span>
                  {result.source_credibility.label === 'UNKNOWN' ? (
                    <span className="text-fog-400">Not assessed yet</span>
                  ) : (
                    <span className="text-fog-200">
                      {result.source_credibility.label}
                      {result.source_credibility.domain && ` · ${result.source_credibility.domain}`}
                    </span>
                  )}
                </div>

                <div className="mt-7">
                  <Tabs
                    value={explanationTab}
                    onChange={setExplanationTab}
                    items={[
                      { id: 'summary', label: 'Explanation' },
                      { id: 'features', label: 'Top Features' },
                      { id: 'models', label: 'Model Contributions' },
                    ]}
                  />
                  <div className="mt-5 text-sm">
                    {explanationTab === 'summary' && (
                      <div className="space-y-5">
                        <p className="leading-relaxed text-fog-300">
                          {result.verdict === 'REAL'
                            ? 'The content shows characteristics of trustworthy news: measured tone, consistent information, and low signs of sensationalism or misinformation.'
                            : 'The content matches patterns frequently observed in fabricated articles: sensational phrasing, unusual emphasis, or structures that diverge from standard reporting.'}
                        </p>
                        {result.explanation.linguistic_flags.length > 0 && (
                          <div className="space-y-3">
                            <p className="text-[10px] uppercase tracking-widest text-fog-600">Linguistic flags</p>
                            <LinguisticFlags flags={result.explanation.linguistic_flags} />
                          </div>
                        )}
                      </div>
                    )}
                    {explanationTab === 'features' && (
                      <FeatureImpactList features={result.explanation.top_features} />
                    )}
                    {explanationTab === 'models' && (
                      <ModelContributionsTable contributions={result.explanation.model_contributions} />
                    )}
                  </div>
                </div>
              </GlassPanel>
            </motion.div>
          )}
        </div>

        <div className="hidden xl:flex xl:flex-col xl:items-center xl:gap-8">
          <DataSphere active={analyzing} className="h-[300px] w-[300px]" />
          <div className="w-full space-y-2">
            <p className="mb-4 text-[10px] uppercase tracking-widest text-fog-600">Analyzing Article</p>
            {STEP_LABELS.map((step, i) => (
              <div
                key={step}
                className={cn(
                  'flex items-center gap-3 rounded-xl px-3 py-2 text-sm transition-colors duration-300',
                  stepIndex === i && analyzing ? 'text-fog-100' : 'text-fog-600'
                )}
              >
                <span
                  className={cn(
                    'h-1.5 w-1.5 rounded-full transition-all duration-300',
                    stepIndex === i && analyzing ? 'bg-ice shadow-[0_0_8px_rgba(188,214,236,0.7)]' : 'bg-fog-700'
                  )}
                />
                {step}
              </div>
            ))}
            {!analyzing && !result && (
              <div className="mt-6 flex items-start gap-2 rounded-xl border border-line bg-white/[0.02] px-3 py-2.5">
                <Sparkles className="mt-0.5 h-3.5 w-3.5 shrink-0 text-fog-600" />
                <p className="text-xs leading-relaxed text-fog-600">
                  Results are also saved to local history for your dashboard and analytics.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
