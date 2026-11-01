'use client'

import Link from 'next/link'
import { motion } from 'framer-motion'
import { ArrowRight, BarChart3, Layers, BadgeCheck, ScanSearch } from 'lucide-react'
import DataSphere from '@/components/DataSphere'
import { Button } from '@/components/ui/button'
import { useRotatingHeading, Caret } from '@/components/HeroTypewriter'

const ROTATING_WORDS = ['Cernovalis', 'Quid est verum, quid est fictum?']

const NAV_LINKS = [
  { label: 'Features', href: '#features' },
  { label: 'Models', href: '/model-comparison' },
  { label: 'Analytics', href: '/analytics' },
]

const FEATURES = [
  { icon: Layers, title: 'Multi-Model AI', sub: 'Ensemble of ML & DL models' },
  { icon: BadgeCheck, title: 'High Accuracy', sub: 'Validated test performance' },
  { icon: BarChart3, title: 'Rich Analytics', sub: 'In-depth insights' },
]

const easeEase = [0.22, 1, 0.36, 1] as const

export default function LandingPage() {
  const { text: word } = useRotatingHeading(ROTATING_WORDS)

  return (
    <div className="relative min-h-screen overflow-hidden">
      <header className="absolute inset-x-0 top-0 z-20">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-6 lg:px-10">
          <Link href="/" className="flex items-center gap-2.5">
            <span className="h-6 w-6 rounded-full bg-[radial-gradient(circle_at_35%_30%,rgba(255,255,255,0.9),rgba(140,150,180,0.35)_45%,rgba(10,10,14,0.9))]" />
            <span className="text-sm font-medium tracking-wide text-fog-100">Cernovalis</span>
          </Link>
          <nav className="hidden items-center gap-8 text-sm text-fog-400 md:flex">
            {NAV_LINKS.map((l) => (
              <Link key={l.label} href={l.href} className="transition-colors hover:text-fog-100">
                {l.label}
              </Link>
            ))}
          </nav>
          <Link href="/detect">
            <Button variant="primary" size="sm">
              Get Started <ArrowRight className="h-3.5 w-3.5" />
            </Button>
          </Link>
        </div>
      </header>

      <main className="relative z-10 mx-auto grid max-w-7xl items-center gap-10 px-6 pb-16 pt-28 lg:grid-cols-2 lg:gap-4 lg:px-10 lg:pt-16 min-h-screen">
        <div className="order-2 lg:order-1">
          <p className="mb-5 text-[11px] font-medium uppercase tracking-[0.32em] text-fog-500">
            AI-Powered News Verification
          </p>

          <h1 className="font-lora text-5xl font-medium leading-[1.08] tracking-tight text-fog-50 md:text-6xl xl:text-7xl">
            {word}
            <Caret />
          </h1>

          <p className="mt-6 max-w-md text-base leading-relaxed text-fog-400">
            Analyze news articles with advanced AI models to detect misinformation and assess
            content credibility.
          </p>

          <div className="mt-9 flex flex-wrap items-center gap-3.5">
            <Link href="/detect">
              <Button variant="primary" size="lg">
                Start Analysis <ScanSearch className="h-4 w-4" />
              </Button>
            </Link>
            <Link href="/analytics">
              <Button variant="glass" size="lg">
                <BarChart3 className="h-4 w-4" /> View Analytics
              </Button>
            </Link>
          </div>

          <div
            id="features"
            className="mt-16 grid gap-6 border-t border-line/70 pt-8 sm:grid-cols-3"
          >
            {FEATURES.map(({ icon: Icon, title, sub }) => (
              <div key={title} className="flex items-start gap-3.5">
                <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-line bg-white/[0.03] text-fog-300">
                  <Icon className="h-4 w-4" />
                </span>
                <div>
                  <p className="text-sm font-medium text-fog-100">{title}</p>
                  <p className="mt-0.5 text-xs text-fog-500">{sub}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <motion.div
          initial={{ opacity: 0, scale: 0.94 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1.4, ease: easeEase }}
          className="relative order-1 lg:order-2 flex items-center justify-center"
        >
          <DataSphere className="h-[340px] w-[340px] sm:h-[440px] sm:w-[440px] xl:h-[600px] xl:w-[600px]" />
          <div
            aria-hidden
            className="pointer-events-none absolute inset-x-10 bottom-8 h-24 rounded-full bg-[radial-gradient(ellipse_at_center,rgba(152,182,224,0.13),transparent_70%)] blur-md"
          />
        </motion.div>
      </main>
    </div>
  )
}
