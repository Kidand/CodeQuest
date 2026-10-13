/**
 * algo-backtrack 章节可视化共用组件：
 *  - C / Sub / Stat：行内代码、分区小标题、数字胶囊
 *  - RecTree：按容器宽度布局的递归树（叶子等距，父节点居中；文字字号固定，不随缩放变小）
 */
import type { ReactNode } from 'react'
import { motion } from 'motion/react'
import { toneColor, toneSoft, useElementWidth, type VizTone } from '@/viz'

export const C = ({ children }: { children: ReactNode }) => (
  <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[12.5px]">{children}</code>
)

export const Sub = ({ children }: { children: ReactNode }) => <div className="mb-1.5 text-xs font-medium text-muted">{children}</div>

export function Stat({ label, value, tone }: { label: ReactNode; value: ReactNode; tone?: VizTone }) {
  return (
    <span className="inline-flex items-baseline gap-1.5 rounded-lg border border-border bg-surface px-2 py-1 text-xs">
      <span className="text-muted">{label}</span>
      <b className="font-mono tabular-nums" style={{ color: tone ? toneColor(tone) : 'var(--fg)' }}>
        {value}
      </b>
    </span>
  )
}

export interface RNode {
  id: string | number
  label: string
  children?: RNode[]
  tone?: VizTone
  dim?: boolean
  /** 被剪掉的分支：虚线框 */
  dashed?: boolean
  edgeLabel?: string
  edgeTone?: VizTone
  /** 节点下方的小字（如 ✂、✓） */
  sub?: string
  subTone?: VizTone
  /** 隐藏（尚未访问）：保留布局位置但不绘制 */
  hidden?: boolean
}

interface P {
  n: RNode
  u: number
  d: number
  parent?: P
}

export function RecTree({ root, levelGap = 56, maxBox = 58, minWidth = 300 }: { root: RNode; levelGap?: number; maxBox?: number; minWidth?: number }) {
  const [ref, measured] = useElementWidth()
  const W = Math.max(minWidth, measured || 640)
  const out: P[] = []
  let leaf = 0
  const walk = (n: RNode, d: number, parent?: P): number => {
    const p: P = { n, u: 0, d, parent }
    out.push(p)
    const kids = n.children ?? []
    if (!kids.length) {
      p.u = leaf++
      return p.u
    }
    const us = kids.map((k) => walk(k, d + 1, p))
    p.u = (us[0] + us[us.length - 1]) / 2
    return p.u
  }
  walk(root, 0)
  const L = Math.max(1, leaf)
  const slot = W / L
  const bw = Math.min(maxBox, slot * 0.86)
  const bh = 24
  const font = bw < 34 ? 10 : 11
  const showEdge = slot >= 50
  const maxD = Math.max(...out.map((p) => p.d))
  const hasSub = out.some((p) => p.n.sub && !p.n.hidden)
  const H = maxD * levelGap + bh + (hasSub ? 16 : 4) + 4
  const x = (p: P) => slot * (p.u + 0.5)
  const y = (p: P) => 2 + p.d * levelGap

  return (
    <div ref={ref} className="w-full min-w-0">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="block select-none">
        {out.map((p) =>
          p.parent && !p.n.hidden ? (
            <g key={`e-${p.n.id}`} opacity={p.n.dim ? 0.4 : 1}>
              <line
                x1={x(p.parent)}
                y1={y(p.parent) + bh}
                x2={x(p)}
                y2={y(p)}
                stroke={p.n.edgeTone ? toneColor(p.n.edgeTone) : 'var(--border-strong)'}
                strokeWidth={p.n.edgeTone ? 2.2 : 1.3}
                strokeDasharray={p.n.dashed ? '4 3' : undefined}
              />
              {showEdge && p.n.edgeLabel && (
                <text
                  x={(x(p.parent) + x(p)) / 2}
                  y={(y(p.parent) + bh + y(p)) / 2 + 3}
                  textAnchor="middle"
                  fontSize={10}
                  fill="var(--muted)"
                  paintOrder="stroke"
                  stroke="var(--surface)"
                  strokeWidth={3}
                  className="font-mono"
                >
                  {p.n.edgeLabel}
                </text>
              )}
            </g>
          ) : null,
        )}
        {out.map((p) => {
          if (p.n.hidden) return null
          const t = p.n.tone
          return (
            <motion.g key={`n-${p.n.id}`} initial={{ opacity: 0 }} animate={{ opacity: p.n.dim ? 0.4 : 1 }} transition={{ duration: 0.25 }}>
              <rect
                x={x(p) - bw / 2}
                y={y(p)}
                width={bw}
                height={bh}
                rx={6}
                fill={t ? toneSoft(t, 28) : 'var(--surface)'}
                stroke={t ? toneColor(t) : 'var(--border-strong)'}
                strokeWidth={t ? 2 : 1.3}
                strokeDasharray={p.n.dashed ? '4 3' : undefined}
              />
              <text x={x(p)} y={y(p) + bh / 2 + 4} textAnchor="middle" fontSize={font} fontWeight={600} fill="var(--fg)" className="font-mono">
                {p.n.label}
              </text>
              {p.n.sub && (
                <text x={x(p)} y={y(p) + bh + 12} textAnchor="middle" fontSize={10} fontWeight={600} fill={p.n.subTone ? toneColor(p.n.subTone) : 'var(--muted)'}>
                  {p.n.sub}
                </text>
              )}
            </motion.g>
          )
        })}
      </svg>
    </div>
  )
}
