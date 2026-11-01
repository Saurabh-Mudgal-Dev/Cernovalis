import type { Viewport } from 'next'
import { Inter, JetBrains_Mono, Lora } from 'next/font/google'
import StarfieldBackground from '@/components/StarfieldBackground'
import { cn } from '@/lib/cn'
import './globals.css'

const sans = Inter({ subsets: ['latin'], variable: '--font-sans', display: 'swap' })
const mono = JetBrains_Mono({ subsets: ['latin'], variable: '--font-mono', display: 'swap' })
const lora = Lora({ subsets: ['latin'], variable: '--font-lora', display: 'swap' })

export const metadata = {
  title: 'Cernovalis — Fake News Detection',
  description:
    'Analyze news articles with advanced AI models to detect misinformation and assess content credibility.',
}

export const viewport: Viewport = {
  themeColor: '#050505',
  width: 'device-width',
  initialScale: 1,
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={cn(sans.variable, mono.variable, lora.variable, 'dark')} suppressHydrationWarning>
      <body className="min-h-screen font-sans antialiased">
        <StarfieldBackground />
        <div className="relative z-10">{children}</div>
      </body>
    </html>
  )
}
