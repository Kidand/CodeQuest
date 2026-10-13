/**
 * algo-ds 章节可视化共用组件：
 *  - C / Sub / Stat：行内代码、分区小标题、数字胶囊
 *  - ListLane：单链表「车道」图（按容器宽度布局，节点位置 / 指针改向都有动画）
 *      节点画成 [值 | next] 两格；next 为 None 时在 next 格画斜线（CLRS 记法）
 *      相邻向右的边画直线；向左的边画上方弧线；跨越的向右边画下方弧线
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

export type LaneId = string | number

export interface LaneItem {
  id: LaneId
  label: string
  tone?: VizTone
  dim?: boolean
  /** 哨兵 / 哑节点：虚线框 */
  dashed?: boolean
}

export interface LanePtr {
  /** 指向的节点 id；'L' / 'R' 表示左右两侧的 None 幽灵位 */
  at: LaneId
  label: string
  tone?: VizTone
}

const SPRING = { type: 'spring' as const, stiffness: 260, damping: 30 }
const MARKER_TONES: (VizTone | 'base')[] = ['base', 'accent', 'success', 'warning', 'danger', 'info']

export function ListLane({
  items,
  next,
  pointers = [],
  ghosts = [true, true],
  edgeTones = {},
  maxBox = 54,
}: {
  /** 显示顺序（从左到右） */
  items: LaneItem[]
  /** next 指针：id → id；null 表示 None；undefined 表示不画 next 格内容 */
  next: Record<string, LaneId | null | undefined>
  pointers?: LanePtr[]
  /** 是否显示左 / 右的 None 幽灵位（指针指向 None 时用） */
  ghosts?: [boolean, boolean]
  /** 边的颜色：按起点 id */
  edgeTones?: Record<string, VizTone>
  maxBox?: number
}) {
  const [ref, measured] = useElementWidth()
  const W = Math.max(280, measured || 640)
  const slots: (LaneId | 'L' | 'R')[] = [...(ghosts[0] ? ['L' as const] : []), ...items.map((it) => it.id), ...(ghosts[1] ? ['R' as const] : [])]
  const n = slots.length
  const slotW = W / n
  const bw = Math.min(maxBox, slotW * 0.74)
  const pw = Math.max(9, Math.round(bw * 0.28))
  const bh = 30
  const small = bw < 36
  const slotOf = new Map<LaneId, number>()
  slots.forEach((s, i) => slotOf.set(s, i))
  const cx = (s: number) => slotW * (s + 0.5)

  // 边与弧高
  type Edge = { from: LaneId; d: string; tone?: VizTone; kind: 'up' | 'down' | 'flat' }
  const edgesRaw: { from: LaneId; sf: number; st: number }[] = []
  let maxUp = 0
  let maxDown = 0
  for (const it of items) {
    const t = next[String(it.id)]
    if (t == null) continue
    const sf = slotOf.get(it.id)!
    const st = slotOf.get(t)
    if (st == null) continue
    edgesRaw.push({ from: it.id, sf, st })
    const span = Math.min(4, Math.abs(st - sf))
    if (st < sf) maxUp = Math.max(maxUp, 14 + 7 * span)
    else if (st > sf + 1) maxDown = Math.max(maxDown, 14 + 7 * span)
  }
  const top = 6 + maxUp
  const cy = top + bh / 2
  const bottom = top + bh
  const byPtr = new Map<number, LanePtr[]>()
  pointers.forEach((p) => {
    const s = slotOf.get(p.at)
    if (s == null) return
    byPtr.set(s, [...(byPtr.get(s) ?? []), p])
  })
  const maxStack = Math.max(0, ...[...byPtr.values()].map((g) => g.length))
  const ptrTop = bottom + maxDown + 6
  const H = ptrTop + (maxStack ? 14 + 13 * maxStack : 0) + 4

  const edges: Edge[] = edgesRaw.map(({ from, sf, st }) => {
    const xr = cx(sf) + bw / 2 - pw / 2
    const span = Math.min(4, Math.abs(st - sf))
    let d: string
    let kind: Edge['kind']
    if (st === sf + 1) {
      const x2 = cx(st) - bw / 2 - 2
      d = `M${xr},${cy} Q${(xr + x2) / 2},${cy} ${x2},${cy}`
      kind = 'flat'
    } else if (st < sf) {
      const x2 = cx(st) + bw * 0.12
      const h = 14 + 7 * span
      d = `M${xr},${top} Q${(xr + x2) / 2},${top - h * 1.6} ${x2},${top - 2}`
      kind = 'up'
    } else {
      const x2 = cx(st) - bw * 0.12
      const h = 14 + 7 * span
      d = `M${xr},${bottom} Q${(xr + x2) / 2},${bottom + h * 1.6} ${x2},${bottom + 2}`
      kind = 'down'
    }
    return { from, d, kind, tone: edgeTones[String(from)] }
  })

  const markerId = (t?: VizTone) => `ds-lane-arr-${t ?? 'base'}`

  return (
    <div ref={ref} className="w-full min-w-0">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="block select-none overflow-visible">
        <defs>
          {MARKER_TONES.map((t) => (
            <marker key={String(t)} id={`ds-lane-arr-${t}`} viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" fill={t === 'base' ? 'var(--muted)' : toneColor(t as VizTone)} />
            </marker>
          ))}
        </defs>
        {/* 幽灵位 None */}
        {slots.map((s, i) =>
          s === 'L' || s === 'R' ? (
            <text key={s} x={cx(i)} y={cy + 4} textAnchor="middle" fontSize={11} fill="var(--subtle)" className="font-mono">
              None
            </text>
          ) : null,
        )}
        {/* 边 */}
        {edges.map((e) => (
          <motion.path
            key={String(e.from)}
            initial={false}
            animate={{ d: e.d }}
            transition={{ duration: 0.35 }}
            fill="none"
            stroke={e.tone ? toneColor(e.tone) : 'var(--muted)'}
            strokeWidth={e.tone ? 2 : 1.5}
            markerEnd={`url(#${markerId(e.tone)})`}
          />
        ))}
        {/* 节点 */}
        {items.map((it) => {
          const s = slotOf.get(it.id)!
          const x0 = -bw / 2
          const isNull = next[String(it.id)] === null
          const tone = it.tone
          return (
            <motion.g key={String(it.id)} initial={false} animate={{ x: cx(s), opacity: it.dim ? 0.35 : 1 }} transition={SPRING}>
              <rect
                x={x0}
                y={top}
                width={bw}
                height={bh}
                rx={6}
                fill={tone ? toneSoft(tone, 26) : 'var(--surface)'}
                stroke={tone ? toneColor(tone) : 'var(--border-strong)'}
                strokeWidth={tone ? 2 : 1.4}
                strokeDasharray={it.dashed ? '4 3' : undefined}
              />
              <line x1={x0 + bw - pw} x2={x0 + bw - pw} y1={top} y2={bottom} stroke={tone ? toneColor(tone) : 'var(--border-strong)'} strokeWidth={1} />
              {isNull && <line x1={x0 + bw - pw + 2} y1={bottom - 3} x2={x0 + bw - 2} y2={top + 3} stroke="var(--muted)" strokeWidth={1.4} />}
              <text
                x={x0 + (bw - pw) / 2}
                y={cy + 4.5}
                textAnchor="middle"
                fontSize={small ? 11 : 13}
                fontWeight={600}
                fill="var(--fg)"
                className="font-mono"
              >
                {it.label}
              </text>
            </motion.g>
          )
        })}
        {/* 指针 */}
        {[...byPtr.entries()].map(([s, group]) => (
          <motion.g key={group.map((g) => g.label).join('|')} initial={false} animate={{ x: cx(s) }} transition={SPRING}>
            <text x={0} y={ptrTop + 9} textAnchor="middle" fontSize={11} fill={toneColor(group[0].tone ?? 'accent')}>
              ▲
            </text>
            {group.map((g, k) => (
              <text key={g.label} x={0} y={ptrTop + 23 + 13 * k} textAnchor="middle" fontSize={11} fontWeight={600} fill={toneColor(g.tone ?? 'accent')} className="font-mono">
                {g.label}
              </text>
            ))}
          </motion.g>
        ))}
      </svg>
    </div>
  )
}
