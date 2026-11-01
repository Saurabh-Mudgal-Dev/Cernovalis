'use client'

import { forwardRef } from 'react'
import { Loader2 } from 'lucide-react'
import { cn } from '@/lib/cn'

const variants = {
  primary:
    'bg-[#e9ebf0] text-ink-950 hover:bg-white active:bg-[#d7d9df] font-medium shadow-[0_10px_40px_rgba(178,204,240,0.18)]',
  glass: 'glass text-fog-200 hover:bg-white/[0.06] hover:text-fog-100',
  ghost: 'text-fog-400 hover:text-fog-100 hover:bg-white/[0.04]',
  outline: 'border border-line-strong text-fog-200 hover:bg-white/[0.04]',
}

const sizes = {
  sm: 'h-8 px-3 text-xs gap-1.5',
  md: 'h-10 px-5 text-sm gap-2',
  lg: 'h-12 px-7 text-sm gap-2',
  icon: 'h-9 w-9',
}

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: keyof typeof variants
  size?: keyof typeof sizes
  loading?: boolean
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { className, variant = 'glass', size = 'md', loading, children, disabled, ...props },
  ref
) {
  return (
    <button
      ref={ref}
      className={cn(
        'inline-flex select-none items-center justify-center rounded-full transition-all duration-200 outline-none',
        'focus-visible:ring-2 focus-visible:ring-fog-500/50',
        'disabled:cursor-not-allowed disabled:opacity-45 disabled:hover:bg-transparent',
        variants[variant],
        sizes[size],
        className
      )}
      disabled={disabled || loading}
      {...props}
    >
      {loading && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
      {children}
    </button>
  )
})
