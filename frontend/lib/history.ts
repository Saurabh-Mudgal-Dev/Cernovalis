'use client'

import { useEffect, useState } from 'react'
import type { AnalyzeResult } from './schemas'

export interface HistoryEntry extends AnalyzeResult {
  model: string
  inferenceMs: number
}

const KEY = 'cernovalis-history'
const MAX_ENTRIES = 500

export function getHistory(): HistoryEntry[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = window.localStorage.getItem(KEY)
    const parsed: unknown = raw ? JSON.parse(raw) : []
    return Array.isArray(parsed) ? (parsed as HistoryEntry[]) : []
  } catch {
    return []
  }
}

export function addHistory(entry: HistoryEntry) {
  const list = getHistory()
  list.unshift(entry)
  if (list.length > MAX_ENTRIES) list.length = MAX_ENTRIES
  window.localStorage.setItem(KEY, JSON.stringify(list))
  window.dispatchEvent(new Event('history-changed'))
}

export function clearHistory() {
  window.localStorage.removeItem(KEY)
  window.dispatchEvent(new Event('history-changed'))
}

export function useHistory(): HistoryEntry[] {
  const [entries, setEntries] = useState<HistoryEntry[]>([])

  useEffect(() => {
    const update = () => setEntries(getHistory())
    update()
    window.addEventListener('history-changed', update)
    window.addEventListener('storage', update)
    return () => {
      window.removeEventListener('history-changed', update)
      window.removeEventListener('storage', update)
    }
  }, [])

  return entries
}
