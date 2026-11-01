'use client'

import { useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { FileUp, FileCheck2 } from 'lucide-react'
import { batchAnalyze } from '@/lib/api'
import { type BatchResult } from '@/lib/schemas'
import { Button } from './ui/button'
import { VerdictBadge } from './verdict'
import { cn } from '@/lib/cn'

export default function BatchUpload() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState<BatchResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const pick = (f: File | null) => {
    setError(null)
    setResult(null)
    if (!f) return
    if (!f.name.toLowerCase().endsWith('.csv')) {
      setError('Only CSV files are supported.')
      return
    }
    setFile(f)
  }

  const submit = async () => {
    if (!file) return
    setUploading(true)
    setProgress(0)
    setError(null)
    try {
      const res = await batchAnalyze(file, setProgress)
      setResult(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Something went wrong.')
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="space-y-5">
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault()
          pick(e.dataTransfer.files?.[0] ?? null)
        }}
        onClick={() => inputRef.current?.click()}
        className={cn(
          'flex cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-line-strong bg-white/[0.02] px-6 py-10 text-center transition-colors hover:border-fog-500/50 hover:bg-white/[0.03]'
        )}
      >
        <FileUp className="mb-3 h-6 w-6 text-fog-500" />
        <p className="text-sm text-fog-200">
          {file ? file.name : 'Drop a CSV file here, or click to browse'}
        </p>
        <p className="mt-1.5 text-xs text-fog-600">
          Requires a `text` column · up to 1,000 rows · UTF-8
        </p>
        <input
          ref={inputRef}
          type="file"
          accept=".csv"
          className="hidden"
          onChange={(e) => pick(e.target.files?.[0] ?? null)}
        />
      </div>

      {file && !uploading && !result && (
        <Button variant="primary" onClick={submit} className="w-full">
          Analyze Batch
        </Button>
      )}

      {uploading && (
        <div className="space-y-2">
          <div className="h-1 overflow-hidden rounded-full bg-white/[0.06]">
            <div className="h-full rounded-full bg-ice/70 transition-all" style={{ width: `${progress}%` }} />
          </div>
          <p className="text-center text-xs text-fog-500">Uploading… {progress}%</p>
        </div>
      )}

      {error && (
        <p className="rounded-xl border border-rose/25 bg-rose/[0.05] px-4 py-3 text-sm text-rose">{error}</p>
      )}

      {result && (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
          <div className="flex items-center gap-4 text-sm text-fog-300">
            <FileCheck2 className="h-4 w-4 text-leaf" />
            <span>
              Processed {result.total} rows — <span className="text-fog-100">{result.succeeded} succeeded</span>
              {result.failed > 0 && <span className="text-rose"> · {result.failed} failed</span>}
            </span>
          </div>
          <div className="max-h-80 overflow-y-auto rounded-xl border border-line">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-ink-900/95 backdrop-blur">
                <tr className="border-b border-line text-left text-[10px] uppercase tracking-widest text-fog-600">
                  <th className="px-4 py-2.5 font-medium">Row</th>
                  <th className="px-4 py-2.5 font-medium">Prediction</th>
                  <th className="px-4 py-2.5 font-medium">Confidence</th>
                  <th className="px-4 py-2.5 font-medium">Model</th>
                  <th className="px-4 py-2.5 font-medium">Error</th>
                </tr>
              </thead>
              <tbody>
                {result.results.map((item) => (
                  <tr key={item.row} className="border-b border-line/60 last:border-0 hover:bg-white/[0.02]">
                    <td className="px-4 py-2.5 font-mono text-xs text-fog-500">{item.row}</td>
                    {item.result ? (
                      <>
                        <td className="px-4 py-2.5">
                          <VerdictBadge verdict={item.result.verdict} />
                        </td>
                        <td className="px-4 py-2.5 font-mono text-xs text-fog-300">
                          {(item.result.confidence * 100).toFixed(1)}%
                        </td>
                        <td className="px-4 py-2.5">
                          <span className="text-fog-600">ensemble</span>
                        </td>
                        <td className="px-4 py-2.5 text-fog-700">—</td>
                      </>
                    ) : (
                      <td colSpan={4} className="px-4 py-2.5 text-xs text-rose/80">{item.error}</td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>
      )}
    </div>
  )
}
