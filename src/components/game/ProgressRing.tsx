import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

/** 环形进度（SVG）。颜色通过 CSS 变量传入，默认 accent */
export function ProgressRing({
  value,
  size = 72,
  stroke = 8,
  color = 'var(--accent)',
  track = 'var(--surface-3)',
  className,
  children,
  label,
}: {
  value: number
  size?: number
  stroke?: number
  color?: string
  track?: string
  className?: string
  children?: ReactNode
  label?: string
}) {
  const r = (size - stroke) / 2
  const c = 2 * Math.PI * r
  const v = Math.max(0, Math.min(1, value))
  return (
    <div className={cn('relative inline-grid shrink-0 place-items-center', className)} style={{ width: size, height: size }} role="img" aria-label={label}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={track} strokeWidth={stroke} />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - v)}
          style={{ transition: 'stroke-dashoffset 0.6s ease' }}
        />
      </svg>
      <div className="absolute inset-0 grid place-items-center">{children}</div>
    </div>
  )
}

/** 线性进度条 */
export function Bar({ value, color = 'var(--accent)', className, height = 8 }: { value: number; color?: string; className?: string; height?: number }) {
  return (
    <div className={cn('cq-bar', className)} style={{ height }}>
      <i style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%`, ['--bar' as string]: color }} />
    </div>
  )
}
