/**
 * 本章自绘 SVG 折线图的小工具：坐标映射 + 坐标轴 + 折线。
 * 颜色全部走 CSS 变量 / toneColor，自动适配暗色。
 */
import type { ReactNode } from 'react'
import { toneColor, type VizTone } from '@/viz'

export interface Axis {
  min: number
  max: number
  /** 对数刻度（底 10） */
  log?: boolean
  ticks: number[]
  label?: string
  fmt?: (v: number) => string
}

export interface PlotBox {
  W: number
  H: number
  pad: { l: number; r: number; t: number; b: number }
}

export const BOX: PlotBox = { W: 440, H: 262, pad: { l: 44, r: 14, t: 26, b: 36 } }

export function scaler(ax: Axis, a: number, b: number) {
  const f = (v: number) => (ax.log ? Math.log10(v) : v)
  const lo = f(ax.min)
  const hi = f(ax.max)
  return (v: number) => a + ((f(v) - lo) / (hi - lo)) * (b - a)
}

export function useScales(x: Axis, y: Axis, box: PlotBox = BOX) {
  const sx = scaler(x, box.pad.l, box.W - box.pad.r)
  const sy = scaler(y, box.H - box.pad.b, box.pad.t)
  return { sx, sy }
}

/** 坐标轴 + 网格线 */
export function Axes({ x, y, box = BOX }: { x: Axis; y: Axis; box?: PlotBox }) {
  const { sx, sy } = useScales(x, y, box)
  const fx = x.fmt ?? ((v: number) => String(v))
  const fy = y.fmt ?? ((v: number) => String(v))
  return (
    <g fontSize={12} fill="var(--muted)">
      {y.ticks.map((t) => (
        <g key={`y${t}`}>
          <line x1={box.pad.l} x2={box.W - box.pad.r} y1={sy(t)} y2={sy(t)} stroke="var(--border)" strokeWidth={t === 0 ? 1.2 : 0.6} />
          <text x={box.pad.l - 5} y={sy(t) + 3.5} textAnchor="end">
            {fy(t)}
          </text>
        </g>
      ))}
      {x.ticks.map((t) => (
        <g key={`x${t}`}>
          <line x1={sx(t)} x2={sx(t)} y1={box.pad.t} y2={box.H - box.pad.b} stroke="var(--border)" strokeWidth={0.6} />
          <text x={sx(t)} y={box.H - box.pad.b + 14} textAnchor="middle">
            {fx(t)}
          </text>
        </g>
      ))}
      {x.label && (
        <text x={(box.pad.l + box.W - box.pad.r) / 2} y={box.H - 4} textAnchor="middle" fill="var(--fg)">
          {x.label}
        </text>
      )}
      {y.label && (
        <text x={6} y={13} fill="var(--fg)" fontSize={12}>
          {y.label}
        </text>
      )}
    </g>
  )
}

/** 由函数采样生成折线点 */
export function sample(fn: (x: number) => number, x: Axis, n = 160): [number, number][] {
  const pts: [number, number][] = []
  for (let i = 0; i <= n; i++) {
    const t = i / n
    const v = x.log ? 10 ** (Math.log10(x.min) + t * (Math.log10(x.max) - Math.log10(x.min))) : x.min + t * (x.max - x.min)
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
  width = 2.2,
  box = BOX,
  clampY = true,
}: {
  pts: [number, number][]
  x: Axis
  y: Axis
  tone?: VizTone
  dashed?: boolean
  width?: number
  box?: PlotBox
  clampY?: boolean
}) {
  const { sx, sy } = useScales(x, y, box)
  const d = pts
    .filter(([, v]) => Number.isFinite(v))
    .map(([u, v], i) => `${i ? 'L' : 'M'}${sx(u).toFixed(1)},${sy(clampY ? Math.max(y.min, Math.min(y.max, v)) : v).toFixed(1)}`)
    .join(' ')
  return <path d={d} fill="none" stroke={toneColor(tone)} strokeWidth={width} strokeDasharray={dashed ? '5 4' : undefined} strokeLinejoin="round" />
}

/** SVG 外壳：viewBox + 100% 宽 */
export function Svg({ children, box = BOX, className }: { children: ReactNode; box?: PlotBox; className?: string }) {
  return (
    <svg viewBox={`0 0 ${box.W} ${box.H}`} width="100%" className={className} style={{ maxWidth: box.W * 1.4, display: 'block', margin: '0 auto' }}>
      {children}
    </svg>
  )
}

/** 小标签：数值卡片 */
export function Stat({ label, value, tone }: { label: ReactNode; value: ReactNode; tone?: VizTone }) {
  return (
    <div className="rounded-xl border border-border bg-surface px-3 py-2">
      <div className="text-[11px] text-muted">{label}</div>
      <div className="font-mono text-[15px] font-semibold tabular-nums" style={{ color: tone ? toneColor(tone) : 'var(--fg)' }}>
        {value}
      </div>
    </div>
  )
}

export const f3 = (v: number) => (Number.isFinite(v) ? (Math.abs(v) < 5e-4 ? '0' : v.toFixed(3)) : '—')
