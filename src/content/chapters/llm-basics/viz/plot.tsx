/**
 * 本章自绘 SVG 图表的小工具：按容器实际宽度布局（文字始终是真实像素大小，手机上不会被缩到 10px 以下），
 * 坐标映射 + 坐标轴 + 折线 + 数值卡片。颜色全部走 CSS 变量 / toneColor，自动适配暗色。
 */
import type { ReactNode } from 'react'
import { toneColor, useElementWidth, type VizTone } from '@/viz'

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

export function scaler(ax: Axis, a: number, b: number) {
  const f = (v: number) => (ax.log ? Math.log10(Math.max(v, 1e-300)) : v)
  const lo = f(ax.min)
  const hi = f(ax.max)
  return (v: number) => a + ((f(v) - lo) / (hi - lo)) * (b - a)
}

export function scales(x: Axis, y: Axis, box: PlotBox) {
  return { sx: scaler(x, box.pad.l, box.W - box.pad.r), sy: scaler(y, box.H - box.pad.b, box.pad.t) }
}

/**
 * 响应式 SVG 外壳：测量容器宽度，viewBox 宽 = 实际像素宽（上限 maxW）。
 *   <Plot height={240}>{(box) => <><Axes box={box} .../><Line box={box} .../></>}</Plot>
 */
export function Plot({
  height = 240,
  maxW = 640,
  pad,
  children,
  className,
}: {
  height?: number
  maxW?: number
  pad?: Partial<PlotBox['pad']>
  children: (box: PlotBox) => ReactNode
  className?: string
}) {
  const [ref, w] = useElementWidth()
  const W = Math.max(260, Math.min(maxW, w || maxW))
  const box: PlotBox = { W, H: height, pad: { l: 46, r: 14, t: 22, b: 38, ...pad } }
  return (
    <div ref={ref} className={className ?? 'w-full min-w-0'}>
      <svg viewBox={`0 0 ${box.W} ${box.H}`} width={box.W} height={box.H} className="mx-auto block overflow-visible">
        {children(box)}
      </svg>
    </div>
  )
}

/** 坐标轴 + 网格线 */
export function Axes({ x, y, box }: { x: Axis; y: Axis; box: PlotBox }) {
  const { sx, sy } = scales(x, y, box)
  const fx = x.fmt ?? ((v: number) => String(v))
  const fy = y.fmt ?? ((v: number) => String(v))
  return (
    <g fontSize={11} fill="var(--muted)">
      {y.ticks.map((t) => (
        <g key={`y${t}`}>
          <line x1={box.pad.l} x2={box.W - box.pad.r} y1={sy(t)} y2={sy(t)} stroke="var(--border)" strokeWidth={t === 0 && !y.log ? 1.3 : 0.7} />
          <text x={box.pad.l - 5} y={sy(t) + 3.5} textAnchor="end">
            {fy(t)}
          </text>
        </g>
      ))}
      {x.ticks.map((t) => (
        <g key={`x${t}`}>
          <line x1={sx(t)} x2={sx(t)} y1={box.pad.t} y2={box.H - box.pad.b} stroke="var(--border)" strokeWidth={t === 0 && !x.log ? 1.3 : 0.7} />
          <text x={sx(t)} y={box.H - box.pad.b + 14} textAnchor="middle">
            {fx(t)}
          </text>
        </g>
      ))}
      {x.label && (
        <text x={(box.pad.l + box.W - box.pad.r) / 2} y={box.H - 5} textAnchor="middle" fill="var(--fg)" fontSize={11.5}>
          {x.label}
        </text>
      )}
      {y.label && (
        <text x={4} y={12} fill="var(--fg)" fontSize={11.5}>
          {y.label}
        </text>
      )}
    </g>
  )
}

/** 由函数采样生成折线点 */
export function sample(fn: (x: number) => number, x: Axis, n = 200): [number, number][] {
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
  box,
  tone = 'accent',
  dashed,
  width = 2.2,
  opacity = 1,
}: {
  pts: [number, number][]
  x: Axis
  y: Axis
  box: PlotBox
  tone?: VizTone
  dashed?: boolean
  width?: number
  opacity?: number
}) {
  const { sx, sy } = scales(x, y, box)
  const lo = y.log ? y.min : y.min - (y.max - y.min) * 0.05
  const hi = y.log ? y.max : y.max + (y.max - y.min) * 0.05
  let d = ''
  let pen = false
  for (const [u, v] of pts) {
    if (!Number.isFinite(v) || (y.log && v <= 0)) {
      pen = false
      continue
    }
    const yy = sy(Math.max(lo, Math.min(hi, v)))
    d += `${pen ? 'L' : 'M'}${sx(u).toFixed(1)},${yy.toFixed(1)} `
    pen = true
  }
  return (
    <path
      d={d}
      fill="none"
      stroke={toneColor(tone)}
      strokeWidth={width}
      strokeOpacity={opacity}
      strokeDasharray={dashed ? '5 4' : undefined}
      strokeLinejoin="round"
      strokeLinecap="round"
    />
  )
}

/** 数据点 */
export function Dot({ x, y, box, at, tone = 'accent', r = 4.5 }: { x: Axis; y: Axis; box: PlotBox; at: [number, number]; tone?: VizTone; r?: number }) {
  const { sx, sy } = scales(x, y, box)
  if (!Number.isFinite(at[1])) return null
  const yy = Math.max(box.pad.t, Math.min(box.H - box.pad.b, sy(at[1])))
  return <circle cx={sx(at[0])} cy={yy} r={r} fill={toneColor(tone)} stroke="var(--surface)" strokeWidth={1.5} />
}

/** 水平色带（如目标区间） */
export function HBand({ x, y, box, from, to, tone = 'success', label }: { x: Axis; y: Axis; box: PlotBox; from: number; to: number; tone?: VizTone; label?: string }) {
  const { sy } = scales(x, y, box)
  const y0 = sy(Math.min(y.max, to))
  const y1 = sy(Math.max(y.min, from))
  return (
    <g>
      <rect x={box.pad.l} y={y0} width={box.W - box.pad.l - box.pad.r} height={Math.max(0, y1 - y0)} fill={toneColor(tone)} fillOpacity={0.12} />
      {label && (
        <text x={box.W - box.pad.r - 4} y={y0 + 12} textAnchor="end" fontSize={10.5} fill={toneColor(tone)}>
          {label}
        </text>
      )}
    </g>
  )
}

/** 竖直参考线 */
export function VLine({ x, y, box, at, tone = 'muted', label, dashed = true }: { x: Axis; y: Axis; box: PlotBox; at: number; tone?: VizTone; label?: string; dashed?: boolean }) {
  const { sx } = scales(x, y, box)
  const xx = sx(at)
  return (
    <g>
      <line x1={xx} x2={xx} y1={box.pad.t} y2={box.H - box.pad.b} stroke={toneColor(tone)} strokeWidth={1.3} strokeDasharray={dashed ? '4 3' : undefined} />
      {label && (
        <text x={xx + 4} y={box.pad.t + 10} fontSize={10.5} fill={toneColor(tone)}>
          {label}
        </text>
      )}
    </g>
  )
}

/** 小标签：数值卡片 */
export function Stat({ label, value, tone, sub }: { label: ReactNode; value: ReactNode; tone?: VizTone; sub?: ReactNode }) {
  return (
    <div className="min-w-0 rounded-xl border border-border bg-surface px-3 py-2">
      <div className="truncate text-[11px] text-muted">{label}</div>
      <div className="truncate font-mono text-[15px] font-semibold tabular-nums" style={{ color: tone ? toneColor(tone) : 'var(--fg)' }}>
        {value}
      </div>
      {sub != null && <div className="truncate text-[11px] text-subtle">{sub}</div>}
    </div>
  )
}

/** 行内代码 */
export const C = ({ children }: { children: ReactNode }) => (
  <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[12.5px]">{children}</code>
)

/**
 * 发散条形图：以 0 为中心，负值向左、正值向右（梯度、更新量、logit 等）。
 */
export function DivBars({
  items,
  max,
  format = (v) => v.toFixed(3),
}: {
  items: { label: string; v: number; tone?: VizTone; dim?: boolean; note?: string }[]
  max?: number
  format?: (v: number) => string
}) {
  const m = max ?? Math.max(1e-9, ...items.map((i) => Math.abs(i.v)))
  return (
    <div className="flex w-full min-w-0 flex-col gap-1.5">
      {items.map((it, i) => {
        const w = Math.min(1, Math.abs(it.v) / m) * 50
        const neg = it.v < 0
        return (
          <div key={i} className="flex items-center gap-2 text-[13px]" style={{ opacity: it.dim ? 0.38 : 1 }}>
            <span className="w-14 shrink-0 truncate text-right font-mono text-fg">{it.label}</span>
            <div className="relative h-5 min-w-0 flex-1 rounded-md bg-surface-2">
              <div className="absolute inset-y-0 left-1/2 w-px bg-border-strong" />
              <div
                className="absolute inset-y-0.5 rounded-sm transition-all duration-300"
                style={{
                  left: neg ? `${50 - w}%` : '50%',
                  width: `${w}%`,
                  background: toneColor(it.tone ?? (neg ? 'info' : 'danger')),
                }}
              />
            </div>
            <span className="w-16 shrink-0 text-right font-mono text-xs text-muted tabular-nums">{format(it.v)}</span>
            {it.note != null && <span className="w-12 shrink-0 truncate text-xs text-subtle">{it.note}</span>}
          </div>
        )
      })}
    </div>
  )
}

/** 数值格式：有限值保留 digits 位，非有限值显示 inf / nan */
export function fnum(v: number, digits = 3) {
  if (Number.isNaN(v)) return 'NaN'
  if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf'
  if (v === 0) return '0'
  const a = Math.abs(v)
  if (a >= 1e5 || a < 1e-3) return v.toExponential(2)
  return String(Number(v.toFixed(digits)))
}
