'use client'

import { useEffect, useState } from 'react'
import { cn } from '@/lib/cn'

interface Star {
  left: number
  top: number
  size: number
  delay: number
  duration: number
  opacity: number
  blue: boolean
}

export default function StarfieldBackground() {
  const [stars, setStars] = useState<Star[]>([])

  useEffect(() => {
    const seeded = (n: number) => {
      let x = 1234567 + n
      return () => {
        x = (x * 1103515245 + 12345) % 2147483648
        return x / 2147483648
      }
    }
    const rand = seeded(Date.now() >> 12)
    setStars(
      Array.from({ length: 140 }, () => ({
        left: rand() * 100,
        top: rand() * 100,
        size: rand() < 0.85 ? 1 : 2,
        delay: rand() * 6,
        duration: 3 + rand() * 6,
        opacity: 0.2 + rand() * 0.6,
        blue: rand() < 0.45,
      }))
    )
  }, [])

  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
      {stars.map((s, i) => (
        <div
          key={i}
          className={cn('absolute rounded-full', s.blue ? 'bg-space-400' : 'bg-white')}
          style={{
            left: `${s.left}%`,
            top: `${s.top}%`,
            width: s.size,
            height: s.size,
            opacity: s.opacity,
            animation: `twinkle ${s.duration}s ease-in-out ${s.delay}s infinite`,
          }}
        />
      ))}
    </div>
  )
}
