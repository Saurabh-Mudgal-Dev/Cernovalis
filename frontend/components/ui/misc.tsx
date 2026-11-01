'use client'

import { cn } from '@/lib/cn'

export function Badge({
  className,
  tone = 'default',
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & { tone?: 'default' | 'muted' }) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full px-2.5 py-0.5 text-[10px] font-medium uppercase tracking-wider',
        tone === 'default' ? 'border border-line-strong bg-white/[0.04] text-fog-300' : 'text-fog-500',
        className
      )}
      {...props}
    />
  )
}

export function Tabs({
  items,
  value,
  onChange,
  className,
}: {
  items: { id: string; label: string; disabled?: boolean; hint?: string }[]
  value: string
  onChange: (id: string) => void
  className?: string
}) {
  return (
    <div className={cn('glass inline-flex max-w-full items-center overflow-x-auto rounded-full p-1', className)}>
      {items.map((item) => (
        <button
          key={item.id}
          type="button"
          disabled={item.disabled}
          onClick={() => onChange(item.id)}
          className={cn(
            'inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full px-4 py-1.5 text-xs font-medium transition-all duration-200 outline-none focus-visible:ring-2 focus-visible:ring-fog-500/50',
            value === item.id
              ? 'bg-white/[0.08] text-fog-100 shadow-[inset_0_1px_0_rgba(255,255,255,0.07)]'
              : 'text-fog-500 hover:text-fog-200',
            item.disabled && 'cursor-not-allowed opacity-45 hover:text-fog-500'
          )}
        >
          {item.label}
          {item.hint && (
            <span className="rounded-full border border-line-strong px-1.5 py-[1px] text-[9px] uppercase tracking-wide text-fog-600">
              {item.hint}
            </span>
          )}
        </button>
      ))}
    </div>
  )
}

export function GlassPanel({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('glass rounded-2xl', className)} {...props} />
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn('animate-pulse rounded-xl bg-white/[0.05]', className)} />
}

export function PageHeader({
  title,
  description,
  children,
}: {
  title: string
  description: string
  children?: React.ReactNode
}) {
  return (
    <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-2xl font-medium tracking-tight text-fog-50 md:text-3xl">{title}</h1>
        <p className="mt-1.5 text-sm text-fog-500">{description}</p>
      </div>
      {children && <div className="flex items-center gap-2">{children}</div>}
    </div>
  )
}
