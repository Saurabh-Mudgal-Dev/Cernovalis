'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useEffect, useState } from 'react'
import {
  LayoutDashboard,
  Search,
  Layers,
  BarChart3,
  History as HistoryIcon,
  Settings,
  Menu,
  X,
} from 'lucide-react'
import { fetchHealth } from '@/lib/api'
import { cn } from '@/lib/cn'

const NAV = [
  { href: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { href: '/detect', label: 'Detect News', icon: Search },
  { href: '/model-comparison', label: 'Model Comparison', icon: Layers },
  { href: '/analytics', label: 'Analytics', icon: BarChart3 },
  { href: '/history', label: 'History', icon: HistoryIcon },
  { href: '/settings', label: 'Settings', icon: Settings },
]

function ApiStatus() {
  const [online, setOnline] = useState<boolean | null>(null)

  useEffect(() => {
    let alive = true
    const check = async () => {
      const h = await fetchHealth()
      if (alive) setOnline(h.status === 'ok')
    }
    check()
    const id = setInterval(check, 45000)
    return () => {
      alive = false
      clearInterval(id)
    }
  }, [])

  return (
    <div className="glass flex items-center gap-2.5 rounded-full px-4 py-2 text-xs text-fog-400">
      <span
        className={cn(
          'h-2 w-2 rounded-full',
          online === null ? 'bg-fog-600 animate-pulse' : online ? 'bg-leaf' : 'bg-rose'
        )}
      />
      API Status: {online === null ? 'Checking…' : online ? 'Online' : 'Offline'}
    </div>
  )
}

function Logo() {
  return (
    <Link href="/" className="flex items-center gap-2.5">
      <span className="h-6 w-6 rounded-full bg-[radial-gradient(circle_at_35%_30%,rgba(255,255,255,0.9),rgba(140,150,180,0.35)_45%,rgba(10,10,14,0.9))]" />
      <span className="text-sm font-medium tracking-wide text-fog-100">Cernovalis</span>
    </Link>
  )
}

export default function Sidebar() {
  const pathname = usePathname()
  const [open, setOpen] = useState(false)

  const content = (
    <div className="flex h-full flex-col justify-between px-4 py-6">
      <div>
        <div className="px-3 pb-8">
          <Logo />
        </div>
        <nav className="space-y-1">
          {NAV.map(({ href, label, icon: Icon }) => {
            const active = pathname === href
            return (
              <Link
                key={href}
                href={href}
                onClick={() => setOpen(false)}
                className={cn(
                  'flex items-center gap-3 rounded-xl px-3.5 py-2.5 text-sm transition-colors',
                  active
                    ? 'glass-strong text-fog-100'
                    : 'text-fog-500 hover:bg-white/[0.03] hover:text-fog-200'
                )}
              >
                <Icon className={cn('h-4 w-4', active && 'text-fog-200')} />
                {label}
              </Link>
            )
          })}
        </nav>
      </div>
      <div className="px-1">
        <ApiStatus />
      </div>
    </div>
  )

  return (
    <>
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-64 border-r border-line/60 bg-ink-950/40 backdrop-blur-md lg:block">
        {content}
      </aside>

      <header className="glass fixed inset-x-0 top-0 z-30 flex items-center justify-between border-0 border-b border-line/60 px-4 py-3 lg:hidden">
        <Logo />
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="glass flex h-9 w-9 items-center justify-center rounded-full text-fog-300"
          aria-label="Open navigation"
        >
          <Menu className="h-4 w-4" />
        </button>
      </header>

      {open && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={() => setOpen(false)} />
          <aside className="glass-strong absolute inset-y-0 left-0 w-72 border-r border-line/60 bg-ink-950/95">
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="absolute right-4 top-6 text-fog-500 hover:text-fog-200"
              aria-label="Close navigation"
            >
              <X className="h-4 w-4" />
            </button>
            {content}
          </aside>
        </div>
      )}
    </>
  )
}
