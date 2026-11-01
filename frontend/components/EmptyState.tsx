'use client'

import { Sparkles } from 'lucide-react'

export default function EmptyState({
  title,
  description,
  action,
}: {
  title: string
  description: string
  action?: React.ReactNode
}) {
  return (
    <div className="glass flex flex-col items-center justify-center rounded-2xl px-8 py-16 text-center">
      <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-2xl border border-line bg-white/[0.03] text-fog-500">
        <Sparkles className="h-5 w-5" />
      </div>
      <h3 className="text-base font-medium text-fog-200">{title}</h3>
      <p className="mt-2 max-w-sm text-sm text-fog-500">{description}</p>
      {action && <div className="mt-6">{action}</div>}
    </div>
  )
}
