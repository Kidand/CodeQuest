/**
 * 本章自绘 SVG 折线图的小工具：坐标映射 + 坐标轴 + 折线 + 数值卡片。
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

export const BOX: PlotBox = { W: 440, H: 250, pad: { l: 44, r: 14, t: 24, b: 36 } }

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

export function Line({
  pts,
  x,
  y,
  tone = 'accent',
  dashed,
  width = 2.2,
  box = BOX,
}: {
  pts: [number, number][]
  x: Axis
  y: Axis
  tone?: VizTone
  dashed?: boolean
  width?: number
  box?: PlotBox
}) {
  const { sx, sy } = useScales(x, y, box)
  const d = pts
    .filter(([, v]) => Number.isFinite(v))
    .map(([u, v], i) => `${i ? 'L' : 'M'}${sx(u).toFixed(1)},${sy(Math.max(y.min, Math.min(y.max, v))).toFixed(1)}`)
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

/** 数值卡片 */
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

/** 竖向柱状图（负载等离散量），带一条参考线（如理想均值） */
export function LoadBars({
  values,
  labels,
  tones,
  refValue: refLine,
  refLabel,
  max,
  height = 150,
  fmt = (v: number) => String(Math.round(v)),
}: {
  values: number[]
  labels: string[]
  tones?: VizTone[]
  refValue?: number
  refLabel?: string
  max?: number
  height?: number
  fmt?: (v: number) => string
}) {
  const m = max ?? Math.max(1e-9, ...values, refLine ?? 0)
  const n = values.length
  const W = Math.max(220, n * 34)
  const H = height
  const top = refLabel ? 30 : 16
  const bot = 22
  const bw = Math.min(26, (W - 8) / n - 6)
  const sy = (v: number) => H - bot - (v / m) * (H - top - bot)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: W * 1.3, display: 'block', margin: '0 auto' }}>
      {values.map((v, i) => {
        const cx = 4 + ((i + 0.5) * (W - 8)) / n
        const y = sy(v)
        return (
          <g key={i}>
            <rect x={cx - bw / 2} y={y} width={bw} height={Math.max(0, H - bot - y)} rx={4} fill={toneColor(tones?.[i] ?? 'accent')} style={{ transition: 'y 300ms, height 300ms' }} />
            <text x={cx} y={y - 4} textAnchor="middle" fontSize={11} fill="var(--muted)" className="font-mono">
              {fmt(v)}
            </text>
            <text x={cx} y={H - 6} textAnchor="middle" fontSize={11} fill="var(--fg)" className="font-mono">
              {labels[i]}
            </text>
          </g>
        )
      })}
      {refLine != null && (
        <g>
          <line x1={2} x2={W - 2} y1={sy(refLine)} y2={sy(refLine)} stroke={toneColor('danger')} strokeWidth={1.4} strokeDasharray="5 4" />
          {refLabel && (
            <text x={W - 4} y={11} textAnchor="end" fontSize={11} fill={toneColor('danger')}>
              {refLabel}
            </text>
          )}
        </g>
      )}
    </svg>
  )
}
