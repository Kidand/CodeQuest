/**
 * 本章自绘 SVG 小工具：坐标映射、坐标轴、折线、数值卡片。
 * 颜色一律走 CSS 变量 / toneColor，自动适配暗色主题。
 * viewBox 宽 420、字号 13：在 390px 手机上缩放后仍 ≥ 10px。
 */
import type { ReactNode } from 'react'
import { toneColor, type VizTone } from '@/viz'

export interface Axis {
  min: number
  max: number
  ticks: number[]
  label?: string
  fmt?: (v: number) => string
}

export interface PlotBox {
  W: number
  H: number
  pad: { l: number; r: number; t: number; b: number }
}

export const BOX: PlotBox = { W: 420, H: 250, pad: { l: 42, r: 20, t: 24, b: 38 } }

export function scaler(ax: Axis, a: number, b: number) {
  return (v: number) => a + ((v - ax.min) / (ax.max - ax.min)) * (b - a)
}

export function scales(x: Axis, y: Axis, box: PlotBox = BOX) {
  return { sx: scaler(x, box.pad.l, box.W - box.pad.r), sy: scaler(y, box.H - box.pad.b, box.pad.t) }
}

/** 坐标轴 + 网格线 */
export function Axes({ x, y, box = BOX }: { x: Axis; y: Axis; box?: PlotBox }) {
  const { sx, sy } = scales(x, y, box)
  const fx = x.fmt ?? ((v: number) => String(v))
  const fy = y.fmt ?? ((v: number) => String(v))
  return (
    <g fontSize={13} fill="var(--muted)">
      {y.ticks.map((t) => (
        <g key={`y${t}`}>
          <line x1={box.pad.l} x2={box.W - box.pad.r} y1={sy(t)} y2={sy(t)} stroke="var(--border)" strokeWidth={t === 0 ? 1.4 : 0.7} />
          <text x={box.pad.l - 5} y={sy(t) + 4} textAnchor="end">
            {fy(t)}
          </text>
        </g>
      ))}
      {x.ticks.map((t) => (
        <g key={`x${t}`}>
          <line x1={sx(t)} x2={sx(t)} y1={box.pad.t} y2={box.H - box.pad.b} stroke="var(--border)" strokeWidth={t === 0 ? 1.4 : 0.7} />
          <text x={sx(t)} y={box.H - box.pad.b + 16} textAnchor="middle">
            {fx(t)}
          </text>
        </g>
      ))}
      {x.label && (
        <text x={(box.pad.l + box.W - box.pad.r) / 2} y={box.H - 3} textAnchor="middle" fill="var(--fg)">
          {x.label}
        </text>
      )}
      {y.label && (
        <text x={4} y={14} fill="var(--fg)">
          {y.label}
        </text>
      )}
    </g>
  )
}

/** 函数采样 */
export function sample(fn: (x: number) => number, x: Axis, n = 160): [number, number][] {
  const pts: [number, number][] = []
  for (let i = 0; i <= n; i++) {
    const v = x.min + (i / n) * (x.max - x.min)
    pts.push([v, fn(v)])
  }
  return pts
}

export function Line({
  pts,
  x,
  y,
  tone = 'accent',
  dashed,
  width = 2.4,
  box = BOX,
  opacity = 1,
}: {
  pts: [number, number][]
  x: Axis
  y: Axis
  tone?: VizTone
  dashed?: boolean
  width?: number
  box?: PlotBox
  opacity?: number
}) {
  const { sx, sy } = scales(x, y, box)
  const d = pts
    .filter(([, v]) => Number.isFinite(v))
    .map(([u, v], i) => `${i ? 'L' : 'M'}${sx(u).toFixed(1)},${sy(Math.max(y.min, Math.min(y.max, v))).toFixed(1)}`)
    .join(' ')
  return (
    <path
      d={d}
      fill="none"
      stroke={toneColor(tone)}
      strokeWidth={width}
      strokeDasharray={dashed ? '6 4' : undefined}
      strokeLinejoin="round"
      opacity={opacity}
    />
  )
}

/** 坐标点 */
export function Dot({ x, y, px, py, tone = 'accent', r = 5, box = BOX }: { x: Axis; y: Axis; px: number; py: number; tone?: VizTone; r?: number; box?: PlotBox }) {
  const { sx, sy } = scales(x, y, box)
  const cy = Math.max(y.min, Math.min(y.max, py))
  return <circle cx={sx(px)} cy={sy(cy)} r={r} fill={toneColor(tone)} stroke="var(--surface)" strokeWidth={2} />
}

export function Svg({ children, box = BOX, className }: { children: ReactNode; box?: PlotBox; className?: string }) {
  return (
    <svg viewBox={`0 0 ${box.W} ${box.H}`} width="100%" className={className} style={{ maxWidth: box.W * 1.35, display: 'block', margin: '0 auto' }}>
      {children}
    </svg>
  )
}

/** 数值卡片 */
export function Stat({ label, value, tone, sub }: { label: ReactNode; value: ReactNode; tone?: VizTone; sub?: ReactNode }) {
  return (
    <div className="min-w-0 rounded-xl border border-border bg-surface px-3 py-2">
      <div className="text-[11.5px] text-muted">{label}</div>
      <div className="font-mono text-[15px] font-semibold tabular-nums" style={{ color: tone ? toneColor(tone) : 'var(--fg)' }}>
        {value}
      </div>
      {sub != null && <div className="text-[11px] text-subtle">{sub}</div>}
    </div>
  )
}

export const f3 = (v: number) => (Number.isFinite(v) ? (Math.abs(v) < 5e-4 ? '0' : v.toFixed(3)) : '—')
export const f2 = (v: number) => (Number.isFinite(v) ? (Math.abs(v) < 5e-3 ? '0' : v.toFixed(2)) : '—')

/** 数值稳定的 log σ(x) 与 σ(x) */
export const logSigmoid = (x: number) => (x >= 0 ? -Math.log1p(Math.exp(-x)) : x - Math.log1p(Math.exp(x)))
export const sigmoid = (x: number) => (x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)))

/** 行内代码 */
export const C = ({ children }: { children: ReactNode }) => (
  <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[12.5px]">{children}</code>
)
