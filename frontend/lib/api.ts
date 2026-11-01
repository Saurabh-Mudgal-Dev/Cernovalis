import axios from 'axios'
import {
  analyzeResponseSchema,
  batchResponseSchema,
  healthSchema,
  type AnalyzeResult,
  type BatchResult,
  type Health,
} from './schemas'

const BASE_KEY = 'cernovalis-api-base'

export function getApiBase(): string {
  if (typeof window === 'undefined') return ''
  const saved = window.localStorage.getItem(BASE_KEY)
  if (saved !== null) return saved.replace(/\/+$/, '')
  return (process.env.NEXT_PUBLIC_API_URL ?? '').replace(/\/+$/, '')
}

export function setApiBase(value: string | null) {
  if (typeof window === 'undefined') return
  if (value === null || value === '') window.localStorage.removeItem(BASE_KEY)
  else window.localStorage.setItem(BASE_KEY, value.replace(/\/+$/, ''))
  window.dispatchEvent(new Event('api-base-changed'))
}

function url(path: string): string {
  const base = getApiBase()
  if (!base) return path.startsWith('/api') || path === '/health' ? path : `/api${path}`
  return path === '/health' ? `${base}/health` : `${base}${path}`
}

function normalizeError(e: unknown): Error {
  if (axios.isAxiosError(e)) {
    const status = e.response?.status
    const detail = (e.response?.data as { detail?: unknown } | undefined)?.detail
    if (status === 503) return new Error('Models not loaded. Please try again shortly.')
    if (status === 501) return new Error('URL analysis is not available yet.')
    if (status === 422) return new Error(typeof detail === 'string' ? detail : 'Please enter text to analyze.')
    if (typeof detail === 'string') return new Error(detail)
    if (status) return new Error(`Request failed with status ${status}.`)
    return new Error('Cannot reach the analysis API. Is the backend running?')
  }
  return e instanceof Error ? e : new Error('Something went wrong.')
}

export async function analyzeArticle(input: { text?: string; url?: string }): Promise<AnalyzeResult> {
  try {
    const res = await axios.post(url('/api/v1/analyze'), input, { timeout: 30000 })
    const parsed = analyzeResponseSchema.safeParse(res.data)
    if (!parsed.success) throw new Error('The API returned an unexpected response format.')
    return parsed.data
  } catch (e) {
    throw normalizeError(e)
  }
}

export async function batchAnalyze(
  file: File,
  onProgress?: (percent: number) => void
): Promise<BatchResult> {
  try {
    const form = new FormData()
    form.append('file', file)
    const res = await axios.post(url('/api/v1/batch'), form, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 300000,
      onUploadProgress: (e) => {
        if (e.total) onProgress?.(Math.round((e.loaded / e.total) * 100))
      },
    })
    const parsed = batchResponseSchema.safeParse(res.data)
    if (!parsed.success) throw new Error('The API returned an unexpected batch response format.')
    return parsed.data
  } catch (e) {
    throw normalizeError(e)
  }
}

export async function fetchHealth(): Promise<Health> {
  try {
    const res = await axios.get(url('/health'), { timeout: 6000 })
    const parsed = healthSchema.safeParse(res.data)
    return parsed.success ? parsed.data : { status: 'unknown' }
  } catch {
    return { status: 'unreachable' }
  }
}
