/**
 * llm-infer 章节内复用的小组件（仅本章使用，不跨章节 import）。
 *  - C：行内代码胶囊
 *  - Stat：数值卡片
 *  - PlotFrame：按容器真实宽度布局的 SVG 坐标系（文字不随缩放变小），支持对数轴
 *  - PQBars：同一组 token 上并排对比两个分布（target p / draft q）
 *  - softmax / sampleIdx：可视化里用的小数学函数
 */
import type { ReactNode } from 'react'
import { motion } from 'motion/react'
import { toneColor, useElementWidth, type VizTone } from '@/viz'

export const C = ({ children }: { children: ReactNode }) => (
  <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[12.5px]">{children}</code>
)

export function Stat({ label, value, tone, sub }: { label: ReactNode; value: ReactNode; tone?: VizTone; sub?: ReactNode }) {
  return (
    <div className="min-w-0 rounded-xl border border-border bg-surface px-3 py-2">
      <div className="truncate text-[11px] text-muted">{label}</div>
      <div className="font-mono text-[15px] font-semibold tabular-nums" style={{ color: tone ? toneColor(tone) : 'var(--fg)' }}>
        {value}
      </div>
      {sub != null && <div className="truncate text-[11px] text-subtle">{sub}</div>}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* 坐标系                                                              */
/* ------------------------------------------------------------------ */

export interface Axis {
  min: number
  max: number
  log?: boolean
  ticks: number[]
  fmt?: (v: number) => string
  label?: string
}

export interface Scales {
  sx: (v: number) => number
  sy: (v: number) => number
  W: number
  H: number
  l: number
  r: number
  t: number
  b: number
}

export function scaler(ax: Axis, a: number, b: number) {
  const f = (v: number) => (ax.log ? Math.log10(Math.max(v, 1e-12)) : v)
  const lo = f(ax.min)
  const hi = f(ax.max)
  return (v: number) => a + ((f(v) - lo) / (hi - lo)) * (b - a)
}

/**
 * 坐标系外壳：测量容器宽度，SVG 的 viewBox 与像素一一对应（字号即真实像素）。
 * children 以 render-prop 形式拿到比例尺，自行绘制折线、点、标注。
 */
export function PlotFrame({
  x,
  y,
  height = 250,
  pad = { l: 48, r: 14, t: 18, b: 38 },
  minWidth = 300,
  children,
}: {
  x: Axis
  y: Axis
  height?: number
  pad?: { l: number; r: number; t: number; b: number }
  minWidth?: number
  children: (s: Scales) => ReactNode
}) {
  const [ref, measured] = useElementWidth()
  const W = Math.max(minWidth, measured || 560)
  const H = height
  const s: Scales = {
    sx: scaler(x, pad.l, W - pad.r),
    sy: scaler(y, H - pad.b, pad.t),
    W,
    H,
    ...pad,
  }
  const fx = x.fmt ?? ((v: number) => String(v))
  const fy = y.fmt ?? ((v: number) => String(v))
  return (
    <div ref={ref} className="w-full min-w-0">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="block overflow-visible">
        <g fontSize={11} fill="var(--muted)" className="font-mono">
          {y.ticks.map((t) => (
            <g key={`y${t}`}>
              <line x1={pad.l} x2={W - pad.r} y1={s.sy(t)} y2={s.sy(t)} stroke="var(--border)" strokeWidth={0.8} />
              <text x={pad.l - 6} y={s.sy(t) + 3.5} textAnchor="end">
                {fy(t)}
              </text>
            </g>
          ))}
          {x.ticks.map((t) => (
            <g key={`x${t}`}>
              <line x1={s.sx(t)} x2={s.sx(t)} y1={pad.t} y2={H - pad.b} stroke="var(--border)" strokeWidth={0.8} />
              <text x={s.sx(t)} y={H - pad.b + 15} textAnchor="middle">
                {fx(t)}
              </text>
            </g>
          ))}
        </g>
        <line x1={pad.l} x2={W - pad.r} y1={H - pad.b} y2={H - pad.b} stroke="var(--border-strong)" />
        <line x1={pad.l} x2={pad.l} y1={pad.t} y2={H - pad.b} stroke="var(--border-strong)" />
        {x.label && (
          <text x={(pad.l + W - pad.r) / 2} y={H - 5} textAnchor="middle" fontSize={11.5} fill="var(--fg)">
            {x.label}
          </text>
        )}
        {y.label && (
          <text x={pad.l} y={11} fontSize={11.5} fill="var(--fg)">
            {y.label}
          </text>
        )}
        {children(s)}
      </svg>
    </div>
  )
}

export function pathOf(pts: [number, number][], s: Scales, y?: Axis) {
  return pts
    .filter(([, v]) => Number.isFinite(v))
    .map(([u, v], i) => {
      const vv = y ? Math.max(y.min, Math.min(y.max, v)) : v
      return `${i ? 'L' : 'M'}${s.sx(u).toFixed(1)},${s.sy(vv).toFixed(1)}`
    })
    .join(' ')
}

/** 在 [min,max]（可对数）上均匀取 n+1 个点 */
export function sampleAxis(ax: Axis, n = 120) {
  const out: number[] = []
  for (let i = 0; i <= n; i++) {
    const t = i / n
    out.push(ax.log ? 10 ** (Math.log10(ax.min) + t * (Math.log10(ax.max) - Math.log10(ax.min))) : ax.min + t * (ax.max - ax.min))
  }
  return out
}

/* ------------------------------------------------------------------ */
/* 分布对比                                                            */
/* ------------------------------------------------------------------ */

/**
 * 同一组 token 上并排画两个分布：上条 p（target），下条 q（draft）。
 * highlight 的行加粗描边；extra 为每行右侧的附加文字。
 */
export function PQBars({
  labels,
  p,
  q,
  highlight,
  extra,
  pLabel = 'p',
  qLabel = 'q',
  pTone = 'accent',
  qTone = 'info',
  max,
  onRowClick,
}: {
  labels: string[]
  p: number[]
  q?: number[]
  highlight?: number
  extra?: (i: number) => ReactNode
  pLabel?: string
  qLabel?: string
  pTone?: VizTone
  qTone?: VizTone
  max?: number
  onRowClick?: (i: number) => void
}) {
  const m = max ?? Math.max(1e-9, ...p, ...(q ?? []))
  return (
    <div className="flex w-full min-w-0 flex-col gap-1.5">
      {labels.map((lab, i) => (
        <div
          key={i}
          onClick={onRowClick ? () => onRowClick(i) : undefined}
          className={`grid items-center gap-x-2 rounded-lg px-1 py-0.5 text-[12px] ${onRowClick ? 'cursor-pointer hover:bg-surface-2' : ''}`}
          style={{
            gridTemplateColumns: extra ? '3.6rem minmax(0,1fr) auto' : '3.6rem minmax(0,1fr)',
            boxShadow: highlight === i ? `inset 0 0 0 1.5px ${toneColor('gold')}` : undefined,
          }}
        >
          <span className="truncate text-right font-mono text-fg">{lab}</span>
          <div className="flex min-w-0 flex-col gap-0.5">
            <Bar v={p[i]} m={m} tone={pTone} text={`${pLabel} ${p[i].toFixed(3)}`} />
            {q && <Bar v={q[i]} m={m} tone={qTone} text={`${qLabel} ${q[i].toFixed(3)}`} />}
          </div>
          {extra && <span className="whitespace-nowrap font-mono text-[11px] text-muted">{extra(i)}</span>}
        </div>
      ))}
    </div>
  )
}

function Bar({ v, m, tone, text }: { v: number; m: number; tone: VizTone; text: string }) {
  return (
    <div className="relative h-[15px] min-w-0 overflow-hidden rounded bg-surface-2">
      <motion.div
        className="absolute inset-y-0 left-0 rounded"
        initial={false}
        animate={{ width: `${(Math.max(0, v) / m) * 100}%` }}
        transition={{ type: 'spring', stiffness: 300, damping: 30 }}
        style={{ background: toneColor(tone), opacity: 0.85 }}
      />
      <span className="absolute inset-y-0 left-1.5 flex items-center font-mono text-[10.5px] text-fg">{text}</span>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* 数学小工具                                                          */
/* ------------------------------------------------------------------ */

export function softmax(xs: number[], t = 1) {
  const m = Math.max(...xs)
  const e = xs.map((x) => Math.exp((x - m) / t))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

/** 逆 CDF：第一个 cumsum > u·sum 的下标（与实战题约定一致） */
export function sampleIdx(w: number[], u: number) {
  const total = w.reduce((a, b) => a + Math.max(0, b), 0)
  let acc = 0
  for (let i = 0; i < w.length; i++) {
    acc += Math.max(0, w[i])
    if (acc > u * total) return i
  }
  return w.length - 1
}

export const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`
