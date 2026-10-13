import { motion } from 'motion/react'
import { cn } from '@/lib/utils'
import { fitCell, useElementWidth } from './useWidth'
import { toneColor, toneSoft, type VizTone } from './tones'

export interface ArrayPointer {
  index: number
  label: string
  tone?: VizTone
  /** 指针显示在上方还是下方，默认 below */
  side?: 'above' | 'below'
}
export interface ArrayRange {
  from: number
  to: number // 含
  tone?: VizTone
  label?: string
}

/**
 * 数组可视化。
 *  - mode='cells'：方格（默认），适合二分/双指针/滑窗/DP 一维表
 *  - mode='bars'：柱状，适合排序
 * ids 提供稳定 key 时，元素交换会有位移动画（排序可视化必备）。
 */
export function ArrayView({
  values,
  ids,
  mode = 'cells',
  highlights,
  dim,
  pointers = [],
  ranges = [],
  showIndex = true,
  cellSize = 44,
  barHeight = 160,
  onCellClick,
  format = (v) => String(v),
  fit = true,
  className,
}: {
  values: (number | string)[]
  ids?: (string | number)[]
  mode?: 'cells' | 'bars'
  highlights?: Record<number, VizTone>
  /** 变暗的下标（如已排除的搜索区间） */
  dim?: (i: number) => boolean
  pointers?: ArrayPointer[]
  ranges?: ArrayRange[]
  showIndex?: boolean
  cellSize?: number
  barHeight?: number
  onCellClick?: (i: number) => void
  format?: (v: number | string) => string
  /** 按容器宽度自动缩小格子（cellSize 视为上限），默认开启；移动端不必横向滚动 */
  fit?: boolean
  className?: string
}) {
  const n = values.length
  const gap = n > 9 ? 4 : 6
  const [wrapRef, avail] = useElementWidth()
  cellSize = fit ? fitCell(avail, n, gap, cellSize, 18) : cellSize
  const valueFont = cellSize < 24 ? 10 : cellSize < 30 ? 11 : cellSize < 38 ? 12.5 : 14
  const nums = values.map((v) => (typeof v === 'number' ? v : 0))
  const maxV = Math.max(1, ...nums.map(Math.abs))
  const width = n * cellSize + (n - 1) * gap
  const above = pointers.filter((p) => p.side === 'above')
  const below = pointers.filter((p) => p.side !== 'above')

  const pointerRow = (ps: ArrayPointer[], side: 'above' | 'below') => {
    if (!ps.length) return null
    // 同一下标多个指针：合并标签
    const byIdx = new Map<number, ArrayPointer[]>()
    ps.forEach((p) => byIdx.set(p.index, [...(byIdx.get(p.index) ?? []), p]))
    return (
      <div className="relative" style={{ width, height: 34 }}>
        {[...byIdx.entries()].map(([idx, group]) => (
          <motion.div
            key={group.map((g) => g.label).join('|')}
            className="absolute flex flex-col items-center"
            initial={false}
            animate={{ left: Math.max(-cellSize / 2, Math.min(width - cellSize / 2, idx * (cellSize + gap))) }}
            transition={{ type: 'spring', stiffness: 380, damping: 32 }}
            style={{ width: cellSize, top: 0, flexDirection: side === 'above' ? 'column-reverse' : 'column' }}
          >
            <span className="text-[13px] leading-none" style={{ color: toneColor(group[0].tone ?? 'accent') }}>
              {side === 'above' ? '▼' : '▲'}
            </span>
            <span className="mt-0.5 whitespace-nowrap font-mono text-[11px] font-semibold leading-tight" style={{ color: toneColor(group[0].tone ?? 'accent') }}>
              {group.map((g) => g.label).join(',')}
            </span>
          </motion.div>
        ))}
      </div>
    )
  }

  return (
    <div ref={wrapRef} className="w-full min-w-0">
    <div className={cn('mx-auto w-max select-none', className)}>
      {pointerRow(above, 'above')}
      {/* 区间标注 */}
      {ranges.length > 0 && (
        <div className="relative" style={{ width, height: 18 * ranges.length }}>
          {ranges.map((r, k) => {
            const left = r.from * (cellSize + gap)
            const w = (r.to - r.from) * (cellSize + gap) + cellSize
            return (
              <motion.div
                key={k}
                className="absolute flex items-center justify-center rounded-md border text-[10.5px] font-medium"
                initial={false}
                animate={{ left, width: Math.max(0, w) }}
                transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                style={{
                  top: k * 18,
                  height: 14,
                  borderColor: toneColor(r.tone ?? 'accent'),
                  background: toneSoft(r.tone ?? 'accent', 18),
                  color: toneColor(r.tone ?? 'accent'),
                }}
              >
                {r.label}
              </motion.div>
            )
          })}
        </div>
      )}
      <div className="relative flex" style={{ gap, height: mode === 'bars' ? barHeight : cellSize, alignItems: mode === 'bars' ? 'flex-end' : 'stretch' }}>
        {values.map((v, i) => {
          const tone = highlights?.[i]
          const isDim = dim?.(i) ?? false
          const key = ids ? ids[i] : i
          const common = {
            layout: ids ? (true as const) : undefined,
            transition: { type: 'spring' as const, stiffness: 420, damping: 34 },
            onClick: onCellClick ? () => onCellClick(i) : undefined,
          }
          if (mode === 'bars') {
            const h = Math.max(6, (Math.abs(nums[i]) / maxV) * (barHeight - 22))
            return (
              <motion.div key={key} {...common} className={cn('flex flex-col items-center justify-end', onCellClick && 'cursor-pointer')} style={{ width: cellSize, opacity: isDim ? 0.35 : 1 }}>
                <span className="mb-1 font-mono text-[11px] text-muted tabular-nums">{format(v)}</span>
                <motion.div
                  className="w-full rounded-t-md border"
                  initial={false}
                  animate={{ height: h }}
                  style={{
                    background: tone ? toneColor(tone) : 'var(--surface-3)',
                    borderColor: tone ? toneColor(tone) : 'var(--border-strong)',
                  }}
                />
              </motion.div>
            )
          }
          return (
            <motion.div
              key={key}
              {...common}
              className={cn(
                'flex items-center justify-center rounded-lg border-2 font-mono font-semibold tabular-nums transition-colors duration-200',
                onCellClick && 'cursor-pointer hover:brightness-95',
              )}
              style={{
                width: cellSize,
                fontSize: valueFont,
                opacity: isDim ? 0.32 : 1,
                background: tone ? toneSoft(tone, 26) : 'var(--surface)',
                borderColor: tone ? toneColor(tone) : 'var(--border)',
                color: 'var(--fg)',
              }}
            >
              {format(v)}
            </motion.div>
          )
        })}
      </div>
      {showIndex && (
        <div className="mt-1 flex" style={{ gap }}>
          {values.map((_, i) => (
            <span key={i} className="text-center font-mono text-[10.5px] text-subtle" style={{ width: cellSize }}>
              {i}
            </span>
          ))}
        </div>
      )}
      {pointerRow(below, 'below')}
    </div>
    </div>
  )
}
