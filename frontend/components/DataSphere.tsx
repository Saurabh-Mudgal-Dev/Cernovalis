'use client'

import { useEffect, useRef } from 'react'
import { cn } from '@/lib/cn'

interface Props {
  active?: boolean
  className?: string
}

type Vec3 = [number, number, number]

export default function DataSphere({ active = false, className }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const activeRef = useRef(active)
  activeRef.current = active

  useEffect(() => {
    const container = containerRef.current
    const canvas = canvasRef.current
    if (!container || !canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let w = 0
    let h = 0
    let dpr = 1
    const resize = () => {
      const rect = container.getBoundingClientRect()
      dpr = Math.min(window.devicePixelRatio || 1, 2)
      w = rect.width
      h = rect.height
      canvas.width = Math.max(1, w * dpr)
      canvas.height = Math.max(1, h * dpr)
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(container)

    const count = 620
    const pts: { v: Vec3; ice: boolean }[] = []
    const golden = Math.PI * (3 - Math.sqrt(5))
    for (let i = 0; i < count; i++) {
      const y = 1 - (i / (count - 1)) * 2
      const r0 = Math.sqrt(Math.max(0, 1 - y * y))
      const th = golden * i
      pts.push({ v: [Math.cos(th) * r0, y, Math.sin(th) * r0], ice: Math.random() < 0.22 })
    }

    const links: [number, number][] = []
    for (let i = 0; i < count; i++) {
      for (let j = i + 1; j < Math.min(i + 48, count); j++) {
        const a = pts[i].v
        const b = pts[j].v
        const dx = a[0] - b[0]
        const dy = a[1] - b[1]
        const dz = a[2] - b[2]
        if (dx * dx + dy * dy + dz * dz < 0.085) links.push([i, j])
      }
    }

    const rings = [
      { r: 1.18, tilt: 0.42, yaw: 0.4, speed: 0.05 },
      { r: 1.36, tilt: -0.58, yaw: 1.9, speed: -0.035 },
      { r: 1.55, tilt: 1.05, yaw: 3.5, speed: 0.022 },
    ]

    const hum = Array.from({ length: 46 }, () => {
      const a = Math.random() * Math.PI * 2
      const r = 1.15 + Math.random() * 0.75
      return {
        v: [Math.cos(a) * r, (Math.random() - 0.5) * r * 1.1, Math.sin(a) * r] as Vec3,
        phase: Math.random() * Math.PI * 2,
      }
    })

    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    let t = 0
    let raf = 0

    const frame = () => {
      const isActive = activeRef.current
      const speed = isActive ? 3.4 : 1
      t += reduceMotion ? 0 : 0.016 * speed

      const breathe = 1 + 0.02 * Math.sin(t * 0.6) + (isActive ? 0.025 * Math.sin(t * 3.2) : 0)
      const R = Math.min(w, h) * 0.32 * breathe
      const cx = w / 2
      const cy = h / 2

      const rotY = t * 0.13
      const cosY = Math.cos(rotY)
      const sinY = Math.sin(rotY)
      const tilt = 0.46
      const cosT = Math.cos(tilt)
      const sinT = Math.sin(tilt)

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, w, h)

      const halo = ctx.createRadialGradient(cx, cy, R * 0.3, cx, cy, R * 1.14)
      const haloA = isActive ? 0.17 : 0.11
      halo.addColorStop(0, `rgba(162, 192, 233, ${haloA})`)
      halo.addColorStop(0.72, `rgba(148, 180, 226, ${haloA * 0.4})`)
      halo.addColorStop(1, 'rgba(148, 180, 226, 0)')
      ctx.fillStyle = halo
      ctx.beginPath()
      ctx.arc(cx, cy, R * 1.14, 0, Math.PI * 2)
      ctx.fill()

      const proj = pts.map(({ v }) => {
        const x1 = v[0] * cosY + v[2] * sinY
        const z1 = -v[0] * sinY + v[2] * cosY
        const y2 = v[1] * cosT - z1 * sinT
        const z2 = v[1] * sinT + z1 * cosT
        const s = 3 / (3 - z2 * 0.85)
        return { x: cx + x1 * R * s, y: cy + y2 * R * s, z: z2, s }
      })

      ctx.lineWidth = 0.6
      for (const [i, j] of links) {
        const a = proj[i]
        const b = proj[j]
        const depth = 0.55 - 0.45 * ((a.z + b.z) / 2)
        const alpha = 0.12 * Math.max(0.08, depth)
        ctx.strokeStyle = `rgba(190, 208, 238, ${alpha})`
        ctx.beginPath()
        ctx.moveTo(a.x, a.y)
        ctx.lineTo(b.x, b.y)
        ctx.stroke()
      }

      for (let i = 0; i < proj.length; i++) {
        const p = proj[i]
        const depth = Math.max(0.1, 0.55 - 0.45 * p.z)
        const alpha = Math.min(0.95, 0.32 + 0.62 * depth)
        const size = (0.7 + 0.9 * (p.s - 0.75)) * 1.15
        ctx.fillStyle = pts[i].ice ? `rgba(188, 214, 236, ${alpha})` : `rgba(230, 234, 242, ${alpha})`
        ctx.beginPath()
        ctx.arc(p.x, p.y, size, 0, Math.PI * 2)
        ctx.fill()
      }

      for (const ring of rings) {
        const yaw = ring.yaw + t * ring.speed * (isActive ? 4 : 1)
        ctx.beginPath()
        for (let i = 0; i <= 110; i++) {
          const a = (i / 110) * Math.PI * 2
          const rv: Vec3 = [Math.cos(a) * ring.r, Math.sin(a) * ring.r, 0]
          const x1 = rv[0] * Math.cos(yaw) + rv[2] * Math.sin(yaw)
          const z1 = -rv[0] * Math.sin(yaw) + rv[2] * Math.cos(yaw)
          const y2 = rv[1] * Math.cos(ring.tilt) - z1 * Math.sin(ring.tilt)
          const z2 = rv[1] * Math.sin(ring.tilt) + z1 * Math.cos(ring.tilt)
          const s = 3 / (3 - z2 * 0.85)
          const px = cx + x1 * R * s
          const py = cy + y2 * R * s
          if (i === 0) ctx.moveTo(px, py)
          else ctx.lineTo(px, py)
        }
        ctx.strokeStyle = `rgba(200, 216, 242, ${isActive ? 0.16 : 0.09})`
        ctx.lineWidth = 0.7
        ctx.stroke()
      }

      for (let i = 0; i < hum.length; i++) {
        const p = hum[i]
        const tw = 0.25 + 0.55 * Math.abs(Math.sin(t * 0.7 + p.phase))
        const x = cx + p.v[0] * R * 0.9 + Math.sin(t * 0.2 + p.phase) * 6
        const y = cy + p.v[1] * R * 0.9 + Math.cos(t * 0.16 + p.phase) * 6
        ctx.fillStyle = `rgba(215, 224, 240, ${tw * 0.5})`
        ctx.beginPath()
        ctx.arc(x, y, 0.9 + (i % 3) * 0.3, 0, Math.PI * 2)
        ctx.fill()
      }


      if (!reduceMotion) raf = requestAnimationFrame(frame)
    }

    raf = requestAnimationFrame(frame)
    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
    }
  }, [])

  return (
    <div ref={containerRef} className={cn('relative', className)} aria-hidden>
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
    </div>
  )
}
