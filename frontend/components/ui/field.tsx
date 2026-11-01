'use client'

import { forwardRef, useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Check, ChevronDown } from 'lucide-react'
import { cn } from '@/lib/cn'

const fieldClass =
  'w-full rounded-xl bg-white/[0.03] border border-line px-3.5 text-sm text-fog-100 placeholder:text-fog-600 outline-none transition-colors focus:border-fog-500/60 focus:bg-white/[0.05] disabled:opacity-50'

export const Input = forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  function Input({ className, ...props }, ref) {
    return <input ref={ref} className={cn(fieldClass, 'h-10', className)} {...props} />
  }
)

export const Textarea = forwardRef<HTMLTextAreaElement, React.TextareaHTMLAttributes<HTMLTextAreaElement>>(
  function Textarea({ className, ...props }, ref) {
    return <textarea ref={ref} className={cn(fieldClass, 'resize-none leading-relaxed', className)} {...props} />
  }
)

interface SelectProps {
  id?: string
  ariaLabel?: string
  value: string
  onChange: (value: string) => void
  options: string[]
  className?: string
  triggerClassName?: string
}

export function Select({ id, ariaLabel, value, onChange, options, className, triggerClassName }: SelectProps) {
  const [open, setOpen] = useState(false)
  const wrapRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  return (
    <div ref={wrapRef} className={cn('relative', className)}>
      <button
        id={id}
        type="button"
        role="combobox"
        aria-label={ariaLabel}
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className={cn(fieldClass, 'flex h-10 cursor-pointer items-center justify-between pr-3 text-left', triggerClassName)}
      >
        <span className="truncate">{value}</span>
        <ChevronDown
          className={cn('h-4 w-4 shrink-0 text-fog-500 transition-transform duration-200', open && 'rotate-180')}
        />
      </button>
      <AnimatePresence>
        {open && (
          <motion.ul
            role="listbox"
            initial={{ opacity: 0, y: -6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -6, scale: 0.98 }}
            transition={{ duration: 0.16, ease: 'easeOut' }}
            className="glass-strong absolute left-0 right-0 z-50 mt-2 max-h-64 overflow-y-auto rounded-xl p-1.5 shadow-card"
            style={{ background: 'rgba(9, 10, 14, 0.93)' }}
          >
            {options.map((opt) => (
              <li key={opt} role="option" aria-selected={opt === value}>
                <button
                  type="button"
                  onClick={() => {
                    onChange(opt)
                    setOpen(false)
                  }}
                  className={cn(
                    'flex w-full items-center justify-between rounded-lg px-3 py-2 text-left text-sm transition-colors',
                    opt === value ? 'bg-white/[0.07] text-space-200' : 'text-fog-300 hover:bg-white/[0.04] hover:text-fog-100'
                  )}
                >
                  {opt}
                  {opt === value && <Check className="h-3.5 w-3.5 text-space-400" />}
                </button>
              </li>
            ))}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  )
}
