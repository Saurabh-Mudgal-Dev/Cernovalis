import type { HistoryEntry } from './history'

export function downloadFile(name: string, content: string, mime = 'text/plain') {
  const blob = new Blob([content], { type: mime })
  const link = document.createElement('a')
  link.href = URL.createObjectURL(blob)
  link.download = name
  link.click()
  URL.revokeObjectURL(link.href)
}

function csvCell(value: unknown): string {
  const s = String(value ?? '')
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}

export function historyToCsv(entries: HistoryEntry[]): string {
  const header = ['id', 'timestamp', 'verdict', 'confidence', 'risk_level', 'model', 'inference_ms', 'original_input']
  const rows = entries.map((e) =>
    [e.id, e.timestamp, e.verdict, e.confidence, e.risk_level, e.model, e.inferenceMs, e.original_input]
      .map(csvCell)
      .join(',')
  )
  return [header.join(','), ...rows].join('\n')
}

export function exportHistoryCsv(entries: HistoryEntry[]) {
  if (entries.length === 0) return
  downloadFile('cernovalis-history.csv', historyToCsv(entries), 'text/csv')
}

export function exportHistoryJson(entries: HistoryEntry[]) {
  if (entries.length === 0) return
  downloadFile('cernovalis-history.json', JSON.stringify(entries, null, 2), 'application/json')
}
