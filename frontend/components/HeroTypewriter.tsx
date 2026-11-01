'use client'

import { useEffect, useState } from 'react'

export function useRotatingHeading(words: string[], typeMs = 70, deleteMs = 34, holdMs = 2400) {
  const [text, setText] = useState('')
  const [holding, setHolding] = useState(false)

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setText(words[1] ?? words[0])
      return
    }
    let index = 0
    let chars = 0
    let deleting = false
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>

    const step = () => {
      if (cancelled) return
      const word = words[index % words.length]
      setHolding(false)
      if (!deleting) {
        chars++
        setText(word.slice(0, chars))
        if (chars >= word.length) {
          deleting = true
          setHolding(true)
          timer = setTimeout(step, holdMs)
          return
        }
        timer = setTimeout(step, typeMs)
      } else {
        chars--
        setText(word.slice(0, chars))
        if (chars <= 0) {
          deleting = false
          index++
          timer = setTimeout(step, 350)
          return
        }
        timer = setTimeout(step, deleteMs)
      }
    }

    timer = setTimeout(step, 500)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [words])

  return { text, holding }
}

export function Caret() {
  return (
    <span
      aria-hidden
      className="ml-[3px] inline-block h-[0.85em] w-[3px] translate-y-[0.09em] rounded-full bg-space-400 shadow-[0_0_8px_rgba(169,199,234,0.6)]"
      style={{ animation: 'caret 1.05s steps(1) infinite' }}
    />
  )
}
