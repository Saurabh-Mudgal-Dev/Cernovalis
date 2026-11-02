'use client'

import { useState } from 'react'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts'
import { PageHeader, GlassPanel, Tabs } from '@/components/ui/misc'
import { MODELS, FEATURE_IMPORTANCE, BILSTM_TRAINING, rocCurve, type ModelMetric } from '@/lib/metrics'
import { cn } from '@/lib/cn'

const tooltipStyle = {
  backgroundColor: 'rgba(10, 10, 12, 0.92)',
  border: '1px solid rgba(255,255,255,0.1)',
  borderRadius: 10,
  fontSize: 12,
  color: '#ededf0',
}

const pct = (v: number) => `${(v * 100).toFixed(1)}%`

function Confusion({ model }: { model: ModelMetric }) {
  const rowOne = [model.confusion.fakeFake, model.confusion.fakeReal]
  const rowTwo = [model.confusion.realFake, model.confusion.realReal]
  const max = Math.max(model.confusion.fakeFake, model.confusion.realReal)
  const cellClass = (v: number, correct: boolean) =>
    cn(
      'flex h-32 flex-col items-center justify-center rounded-2xl border md:h-40',
      correct ? 'border-white/15' : 'border-rose/25'
    )
  const cellStyle = (v: number, correct: boolean) => ({
    background: `rgba(${correct ? '230,234,242' : '194,151,160'}, ${(0.05 + (v / max) * 0.32).toFixed(3)})`,
  })

  return (
    <div className="text-xs">
      <div className="grid grid-cols-[22px_1fr_1fr] gap-2">
        <div />
        <div className="pb-1.5 text-center text-[11px] uppercase tracking-widest text-fog-500">Pred. Fake</div>
        <div className="pb-1.5 text-center text-[11px] uppercase tracking-widest text-fog-500">Pred. Real</div>
        <div className="flex items-center justify-center text-[11px] uppercase tracking-widest text-fog-500 [writing-mode:vertical-rl]">
          Fake
        </div>
        {rowOne.map((v, i) => (
          <div key={i} className={cellClass(v, i === 0)} style={cellStyle(v, i === 0)}>
            <span className="text-2xl font-medium text-fog-50 md:text-3xl">{v.toLocaleString()}</span>
          </div>
        ))}
        <div className="flex items-center justify-center text-[11px] uppercase tracking-widest text-fog-500 [writing-mode:vertical-rl]">
          Real
        </div>
        {rowTwo.map((v, i) => (
          <div key={i} className={cellClass(v, i === 1)} style={cellStyle(v, i === 1)}>
            <span className="text-2xl font-medium text-fog-50 md:text-3xl">{v.toLocaleString()}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function ModelComparisonPage() {
  const [chartTab, setChartTab] = useState<string>('roc')
  const best = MODELS.reduce((a, b) => (b.accuracy > a.accuracy ? b : a))

  const rocData = ['Logistic Regression', 'XGBoost', 'BiLSTM'].map((name) => ({
    name,
    model: MODELS.find((m) => m.name === name)!,
  }))
  const curves = rocData.map(({ name, model }, i) => ({
    name,
    auc: model.accuracy - 0.004,
    data: rocCurve(model.accuracy - 0.004),
    stroke: ['#d4d4dc', '#bcd6ec', '#7fa3d0'][i],
  }))
  const insetCurves = curves.map(({ name, auc, stroke }) => ({
    name,
    auc,
    stroke,
    data: rocCurve(auc, 400, 0.1),
  }))
  const diagonal = Array.from({ length: 101 }, (_, i) => ({ fpr: i / 100, tpr: i / 100 }))

  const M = Math.ceil(curves[0].data.length)
  const mergedRoc = (() => {
    const rows: { fpr: number; lr: number; xgb: number; bilstm: number; random: number }[] = []
    for (let i = 0; i < M; i++) {
      rows.push({
        fpr: curves[0].data[i].fpr,
        lr: curves[0].data[i].tpr,
        xgb: curves[1].data[i].tpr,
        bilstm: curves[2].data[i].tpr,
        random: diagonal[Math.min(i * (diagonal.length - 1), M - 1)]?.tpr ?? 0,
      })
    }
    return rows
  })()

  const mergedInset = (() => {
    const rows: { fpr: number; lr: number; xgb: number; bilstm: number }[] = []
    for (let i = 0; i < insetCurves[0].data.length; i++) {
      rows.push({
        fpr: insetCurves[0].data[i].fpr,
        lr: insetCurves[0].data[i].tpr,
        xgb: insetCurves[1].data[i].tpr,
        bilstm: insetCurves[2].data[i].tpr,
      })
    }
    return rows
  })()

  const percent = (v: unknown) => `${Math.round(Number(v) * 100)}%`

  return (
    <div>
      <PageHeader title="Model Comparison" description="Benchmark on the ISOT test split · docs/model-card.md" />

      <GlassPanel className="overflow-x-auto p-0">
        <table className="w-full min-w-[720px] text-sm">
          <thead>
            <tr className="border-b border-line bg-white/[0.02] text-left text-[10px] uppercase tracking-widest text-fog-600">
              <th className="px-6 py-3.5 font-medium">Model</th>
              <th className="px-4 py-3.5 font-medium">Accuracy</th>
              <th className="px-4 py-3.5 font-medium">Macro-F1</th>
              <th className="px-4 py-3.5 font-medium">Fake F1</th>
              <th className="px-4 py-3.5 font-medium">Real F1</th>
              <th className="px-6 py-3.5 text-right font-medium">Inference</th>
            </tr>
          </thead>
          <tbody>
            {MODELS.map((m) => (
              <tr key={m.name} className="border-b border-line/50 last:border-0 hover:bg-white/[0.02]">
                <td className="px-6 py-3.5">
                  <span className="text-fog-100">{m.name}</span>
                  {m.name === best.name && (
                    <span className="ml-2 rounded-full border border-ice/30 bg-ice/[0.07] px-2 py-0.5 text-[9px] uppercase tracking-wider text-ice">
                      Best
                    </span>
                  )}
                </td>
                <td className="px-4 py-3.5 font-mono text-fog-200">{pct(m.accuracy)}</td>
                <td className="px-4 py-3.5 font-mono text-fog-300">{pct(m.macroF1)}</td>
                <td className="px-4 py-3.5 font-mono text-fog-300">{pct(m.fakeF1)}</td>
                <td className="px-4 py-3.5 font-mono text-fog-300">{pct(m.realF1)}</td>
                <td className="px-6 py-3.5 text-right font-mono text-fog-400">{m.inferenceMs} ms</td>
              </tr>
            ))}
          </tbody>
        </table>
      </GlassPanel>
      <p className="mt-2 text-[11px] text-fog-600">
        Inference times are indicative single-article round-trip estimates for comparison only.
      </p>

      <GlassPanel className="mt-8 p-6">
        <Tabs
          className="mb-6"
          value={chartTab}
          onChange={setChartTab}
          items={[
            { id: 'roc', label: 'ROC Curve' },
            { id: 'confusion', label: 'Confusion Matrix' },
            { id: 'importance', label: 'Feature Importance' },
            { id: 'training', label: 'Training History' },
          ]}
        />

        {chartTab === 'roc' && (
          <div>
            <div className="grid gap-6 lg:grid-cols-2">
              <div>
                <div className="mb-3 flex items-center justify-between">
                  <p className="text-xs font-medium uppercase tracking-widest text-fog-400">Full curve</p>
                  <p className="text-[11px] text-fog-600">Ideal sits in the top-left corner ←</p>
                </div>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={mergedRoc} margin={{ top: 4, right: 16, left: -12, bottom: 0 }}>
                      <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                      <XAxis
                        dataKey="fpr"
                        domain={[0, 1]}
                        tickFormatter={percent}
                        tick={{ fill: '#66666f', fontSize: 10 }}
                        tickLine={false}
                        axisLine={false}
                      />
                      <YAxis
                        domain={[0, 1]}
                        tickFormatter={percent}
                        tick={{ fill: '#66666f', fontSize: 10 }}
                        tickLine={false}
                        axisLine={false}
                      />
                      <Tooltip
                        contentStyle={tooltipStyle}
                        labelFormatter={(v) => `Wrongly flagged (FPR): ${percent(v)}`}
                        formatter={(value) => `${percent(value)} detected`}
                      />
                      <Legend wrapperStyle={{ fontSize: 10, color: '#8f8f9c' }} />
                      <Line name="LR · AUC 0.988" dataKey="lr" stroke="#d4d4dc" strokeWidth={1.5} dot={false} />
                      <Line name="XGBoost · AUC 0.985" dataKey="xgb" stroke="#bcd6ec" strokeWidth={1.5} dot={false} />
                      <Line name="BiLSTM · AUC 0.990" dataKey="bilstm" stroke="#7fa3d0" strokeWidth={1.5} dot={false} />
                      <Line name="Random (AUC 0.50)" dataKey="random" stroke="rgba(255,255,255,0.18)" strokeWidth={1} strokeDasharray="4 4" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div>
                <div className="mb-3 flex items-center justify-between">
                  <p className="text-xs font-medium uppercase tracking-widest text-fog-400">Operating region detail</p>
                  <p className="text-[11px] text-fog-600">FPR 0–10% · where these models really live</p>
                </div>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={mergedInset} margin={{ top: 4, right: 16, left: -12, bottom: 0 }}>
                      <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                      <XAxis
                        dataKey="fpr"
                        domain={[0, 0.1]}
                        tickFormatter={percent}
                        tick={{ fill: '#66666f', fontSize: 10 }}
                        tickLine={false}
                        axisLine={false}
                      />
                      <YAxis
                        domain={[0.6, 1]}
                        tickFormatter={percent}
                        tick={{ fill: '#66666f', fontSize: 10 }}
                        tickLine={false}
                        axisLine={false}
                      />
                      <Tooltip
                        contentStyle={tooltipStyle}
                        labelFormatter={(v) => `Wrongly flagged (FPR): ${percent(v)}`}
                        formatter={(value) => `${percent(value)} detected`}
                      />
                      <Legend wrapperStyle={{ fontSize: 10, color: '#8f8f9c' }} />
                      <Line name="LR · AUC 0.988" dataKey="lr" stroke="#d4d4dc" strokeWidth={1.5} dot={false} />
                      <Line name="XGBoost · AUC 0.985" dataKey="xgb" stroke="#bcd6ec" strokeWidth={1.5} dot={false} />
                      <Line name="BiLSTM · AUC 0.990" dataKey="bilstm" stroke="#7fa3d0" strokeWidth={1.5} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
            <p className="mt-3 text-[11px] text-fog-600">
              AUC = probability the model ranks a random fake article above a random real one. Curves are
              illustrative; no ROC data is stored in the repository.
            </p>
          </div>
        )}

        {chartTab === 'confusion' && (
          <div className="grid gap-8 md:grid-cols-2">
            <div>
              <p className="mb-3 text-xs font-medium uppercase tracking-widest text-fog-400">Stacked meta-classifier</p>
              <Confusion model={best} />
            </div>
            <div>
              <p className="mb-3 text-xs font-medium uppercase tracking-widest text-fog-400">BiLSTM</p>
              <Confusion model={MODELS.find((m) => m.name === 'BiLSTM')!} />
            </div>
          </div>
        )}

        {chartTab === 'importance' && (
          <ul className="space-y-3.5">
            {FEATURE_IMPORTANCE.map((f) => (
              <li key={f.feature} className="flex items-center gap-4">
                <span className="w-52 shrink-0 truncate font-mono text-xs text-fog-200">{f.feature}</span>
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/[0.06]">
                  <div className="h-full rounded-full bg-ice/60" style={{ width: `${f.value * 100}%` }} />
                </div>
                <span className="w-14 shrink-0 text-right font-mono text-xs text-fog-500">
                  {f.value.toFixed(2)}
                </span>
              </li>
            ))}
            <li className="pt-1 text-[11px] text-fog-600">
              Relative XGBoost gain importance, normalized for illustration.
            </li>
          </ul>
        )}

        {chartTab === 'training' && (
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={BILSTM_TRAINING} margin={{ top: 4, right: 16, left: -14, bottom: 0 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  dataKey="epoch"
                  tick={{ fill: '#66666f', fontSize: 10 }}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis
                  tick={{ fill: '#66666f', fontSize: 10 }}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip contentStyle={tooltipStyle} labelFormatter={(v) => `Epoch ${v}`} />
                <Legend wrapperStyle={{ fontSize: 11, color: '#8f8f9c' }} />
                <Line type="monotone" dataKey="loss" name="train loss" stroke="#d4d4dc" strokeWidth={1.5} dot={false} />
                <Line type="monotone" dataKey="valLoss" name="val loss" stroke="#bcd6ec" strokeWidth={1.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </GlassPanel>

    </div>
  )
}
