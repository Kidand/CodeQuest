/**
 * algo-search 章节内复用的小组件（仅本章使用）。
 *  - C：行内代码胶囊（caption 里用）
 *  - ThresholdBars：带阈值线的柱状图（按容器实际宽度布局，文字不随缩放变小）
 */
import { motion } from 'motion/react'
import { toneColor, useElementWidth, type VizTone } from '@/viz'

export const C = ({ children }: { children: React.ReactNode }) => (
  <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[12.5px]">{children}</code>
)

export interface BarPointer {
  index: number
  label: string
  tone?: VizTone
}

/**
 * 柱状图 + 水平阈值线。用于「二分答案」的 check(x) 曲线、显存预算等。
 * 每根柱子的颜色由 tones[i] 决定；dim(i) 为 true 时变暗（已排除区间）。
 */
export function ThresholdBars({
  values,
  labels,
  threshold,
  thresholdLabel,
  tones,
  pointers = [],
  dim,
  plotHeight = 170,
  format = (v) => String(v),
  xTitle,
}: {
  values: number[]
  labels: string[]
  threshold?: number
  thresholdLabel?: string
  tones?: (VizTone | undefined)[]
  pointers?: BarPointer[]
  dim?: (i: number) => boolean
  plotHeight?: number
  format?: (v: number) => string
  xTitle?: string
}) {
  const [ref, measured] = useElementWidth()
  const width = Math.max(300, measured || 640)
  const n = values.length
  const padL = 6
  const padR = thresholdLabel ? 64 : 6
  const padT = 20
  const slot = (width - padL - padR) / n
  const bw = Math.max(8, Math.min(46, slot * 0.7))
  const maxV = Math.max(1e-9, ...values, (threshold ?? 0) * 1.12)
  const y = (v: number) => padT + plotHeight * (1 - v / maxV)
  const baseY = padT + plotHeight
  const labelY = baseY + 15
  const ptrY = labelY + 8
  const total = ptrY + (pointers.length ? 30 : 4) + (xTitle ? 14 : 0)
  const cx = (i: number) => padL + slot * i + slot / 2

  // 同一下标多个指针合并
  const byIdx = new Map<number, BarPointer[]>()
  pointers.forEach((p) => byIdx.set(p.index, [...(byIdx.get(p.index) ?? []), p]))

  return (
    <div ref={ref} className="w-full min-w-0">
      <svg width={width} height={total} viewBox={`0 0 ${width} ${total}`} className="block select-none">
        <line x1={padL} x2={width - padR} y1={baseY} y2={baseY} stroke="var(--border-strong)" strokeWidth={1} />
        {values.map((v, i) => {
          const tone = tones?.[i]
          const isDim = dim?.(i) ?? false
          const top = y(v)
          return (
            <g key={i} opacity={isDim ? 0.3 : 1} style={{ transition: 'opacity 200ms' }}>
              <motion.rect
                x={cx(i) - bw / 2}
                width={bw}
                rx={4}
                initial={false}
                animate={{ y: top, height: Math.max(1, baseY - top) }}
                transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                fill={tone ? toneColor(tone) : 'var(--surface-3)'}
                fillOpacity={tone ? 0.8 : 1}
                stroke={tone ? toneColor(tone) : 'var(--border-strong)'}
              />
              <text x={cx(i)} y={top - 5} textAnchor="middle" fontSize={11} fill="var(--muted)" className="font-mono tabular-nums">
                {format(v)}
              </text>
              <text x={cx(i)} y={labelY} textAnchor="middle" fontSize={11} fill="var(--fg)" className="font-mono">
                {labels[i]}
              </text>
            </g>
          )
        })}
        {threshold != null && (
          <g>
            <line
              x1={padL}
              x2={width - padR + 4}
              y1={y(threshold)}
              y2={y(threshold)}
              stroke="var(--danger)"
              strokeWidth={1.6}
              strokeDasharray="6 4"
            />
            {thresholdLabel && (
              <text x={width - padR + 8} y={y(threshold) + 4} fontSize={11.5} fontWeight={600} fill="var(--danger)" className="font-mono">
                {thresholdLabel}
              </text>
            )}
          </g>
        )}
        {[...byIdx.entries()].map(([idx, group]) => (
          <motion.g key={group.map((g) => g.label).join('|')} initial={false} animate={{ x: cx(idx) }} transition={{ type: 'spring', stiffness: 380, damping: 32 }}>
            <text x={0} y={ptrY + 10} textAnchor="middle" fontSize={12} fill={toneColor(group[0].tone ?? 'accent')}>
              ▲
            </text>
            <text x={0} y={ptrY + 23} textAnchor="middle" fontSize={11} fontWeight={600} fill={toneColor(group[0].tone ?? 'accent')} className="font-mono">
              {group.map((g) => g.label).join(',')}
            </text>
          </motion.g>
        ))}
        {xTitle && (
          <text x={width / 2} y={total - 2} textAnchor="middle" fontSize={11} fill="var(--subtle)">
            {xTitle}
          </text>
        )}
      </svg>
    </div>
  )
}
