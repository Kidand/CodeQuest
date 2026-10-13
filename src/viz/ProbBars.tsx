import { motion } from 'motion/react'
import { cn, fmt } from '@/lib/utils'
import { toneColor, type VizTone } from './tones'

export interface ProbItem {
  label: string
  p: number
  tone?: VizTone
  /** 变灰（例如被 top-k / top-p 过滤） */
  dim?: boolean
  /** 右侧附加说明 */
  note?: string
}

/**
 * 概率分布条形图（采样、softmax 温度、路由权重、优势值…）。
 * orientation='h'（默认）：每行一个 token，适合移动端；'v'：竖条。
 */
export function ProbBars({
  items,
  max,
  orientation = 'h',
  format = (p) => fmt(p, 3),
  height = 180,
  className,
}: {
  items: ProbItem[]
  max?: number
  orientation?: 'h' | 'v'
  format?: (p: number) => string
  height?: number
  className?: string
}) {
  const m = max ?? Math.max(1e-9, ...items.map((i) => Math.abs(i.p)))
  if (orientation === 'v') {
    return (
      <div className={cn('mx-auto flex w-max items-end gap-2', className)} style={{ height }}>
        {items.map((it, i) => (
          <div key={i} className="flex w-11 flex-col items-center justify-end" style={{ height, opacity: it.dim ? 0.35 : 1 }}>
            <span className="mb-1 font-mono text-[10.5px] text-muted tabular-nums">{format(it.p)}</span>
            <motion.div
              className="w-full rounded-t-md"
              initial={false}
              animate={{ height: Math.max(2, (Math.abs(it.p) / m) * (height - 40)) }}
              transition={{ type: 'spring', stiffness: 300, damping: 30 }}
              style={{ background: toneColor(it.tone ?? 'accent') }}
            />
            <span className="mt-1 max-w-full truncate font-mono text-[11px] text-fg">{it.label}</span>
          </div>
        ))}
      </div>
    )
  }
  return (
    <div className={cn('flex w-full min-w-[260px] flex-col gap-1.5', className)}>
      {items.map((it, i) => (
        <div key={i} className="flex items-center gap-2 text-[13px]" style={{ opacity: it.dim ? 0.38 : 1 }}>
          <span className="w-16 shrink-0 truncate text-right font-mono text-fg sm:w-24">{it.label}</span>
          <div className="relative h-5 min-w-0 flex-1 overflow-hidden rounded-md bg-surface-2">
            <motion.div
              className="absolute inset-y-0 left-0 rounded-md"
              initial={false}
              animate={{ width: `${(Math.abs(it.p) / m) * 100}%` }}
              transition={{ type: 'spring', stiffness: 300, damping: 30 }}
              style={{ background: toneColor(it.tone ?? 'accent') }}
            />
          </div>
          <span className="w-14 shrink-0 text-right font-mono text-xs text-muted tabular-nums">{format(it.p)}</span>
          {it.note != null && <span className="w-16 shrink-0 truncate text-xs text-subtle">{it.note}</span>}
        </div>
      ))}
    </div>
  )
}
