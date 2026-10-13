/**
 * 本章通用 DP 表：在格子上叠加「依赖箭头」，直观展示 dp[i][j] 由哪些格子转移而来。
 *
 *  - 按容器宽度自动计算格子尺寸（不靠 viewBox 缩放，保证手机上文字 ≥ 10px）
 *  - fill(r,c) 给格子上底色；ring(r,c) 给格子描边（当前格 / 回溯路径）
 *  - arrows：依赖箭头。同一行且相隔 ≥ 2 的箭头画成上方弧线（1D DP 常见），其它画直线
 *  - null 值表示「尚未计算」，显示为浅灰底 + ·
 */
import { useId } from 'react'
import { fitCell, toneColor, toneSoft, useElementWidth, type VizTone } from '@/viz'
import { cn } from '@/lib/utils'

export type Cell = number | string | null

export interface DpArrow {
  from: [number, number]
  to: [number, number]
  tone?: VizTone
  label?: string
  dashed?: boolean
  /** 强制画成弧线（默认：同行且相隔 ≥ 2 时自动弧线） */
  arc?: boolean
}

const MARKER_TONES: VizTone[] = ['accent', 'success', 'warning', 'danger', 'info', 'muted', 'gold', 1, 2, 3, 4, 5, 6, 7, 8]

export function DpGrid({
  data,
  rowLabels,
  colLabels,
  corner,
  cellSize = 46,
  minCell = 24,
  gap = 4,
  fill,
  ring,
  dim,
  arrows = [],
  format = (v) => String(v),
  onCellClick,
  className,
}: {
  data: Cell[][]
  rowLabels?: string[]
  colLabels?: string[]
  corner?: string
  cellSize?: number
  minCell?: number
  gap?: number
  fill?: (r: number, c: number) => VizTone | undefined | null | false
  ring?: (r: number, c: number) => VizTone | undefined | null | false
  dim?: (r: number, c: number) => boolean
  arrows?: DpArrow[]
  format?: (v: number | string) => string
  onCellClick?: (r: number, c: number) => void
  className?: string
}) {
  const uid = useId().replace(/[^a-zA-Z0-9]/g, '')
  const [wrapRef, avail] = useElementWidth()
  const rows = data.length
  const cols = data[0]?.length ?? 0
  const maxLabel = Math.max(0, ...(rowLabels ?? []).map((s) => labelWidth(s)))
  const labelW = rowLabels ? maxLabel + 10 : 0
  const cell = fitCell(avail ? avail - labelW - 4 : 0, cols, gap, cellSize, minCell)
  const hasArcRow0 = arrows.some((a) => isArc(a) && (a.from[0] === 0 || a.to[0] === 0))
  const padTop = hasArcRow0 ? Math.round(Math.max(cell * 1.1, 40)) : 4
  // 顶部有弧线时，把列标签放到表格下方，避免与弧线重叠
  const labelsBelow = hasArcRow0 && !!colLabels
  const headerH = colLabels && !labelsBelow ? 18 : 0
  const x0 = labelW
  const y0 = padTop + headerH
  const W = x0 + cols * cell + (cols - 1) * gap + 4
  const gridBottom = y0 + rows * cell + (rows - 1) * gap
  const H = gridBottom + (labelsBelow ? 20 : 4)
  const cx = (c: number) => x0 + c * (cell + gap) + cell / 2
  const cy = (r: number) => y0 + r * (cell + gap) + cell / 2
  const font = cell < 28 ? 10 : cell < 36 ? 11 : cell < 44 ? 12 : 13

  return (
    <div ref={wrapRef} className={cn('w-full min-w-0', className)}>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="mx-auto block overflow-visible select-none">
        <defs>
          {MARKER_TONES.map((t) => (
            <marker
              key={String(t)}
              id={`dpa-${uid}-${t}`}
              viewBox="0 0 10 10"
              refX="8.5"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M0,0 L10,5 L0,10 z" fill={toneColor(t)} />
            </marker>
          ))}
        </defs>
        {colLabels?.map((l, c) => (
          <text key={`cl${c}`} x={cx(c)} y={labelsBelow ? gridBottom + 14 : padTop + 12} textAnchor="middle" fontSize={11} fill="var(--muted)" className="font-mono">
            {l}
          </text>
        ))}
        {rowLabels?.map((l, r) => (
          <text key={`rl${r}`} x={labelW - 6} y={cy(r)} textAnchor="end" dominantBaseline="central" fontSize={11} fill="var(--muted)" className="font-mono">
            {l}
          </text>
        ))}
        {corner && rowLabels && colLabels && !labelsBelow && (
          <text x={labelW - 6} y={padTop + 12} textAnchor="end" fontSize={10} fill="var(--subtle)" className="font-mono">
            {corner}
          </text>
        )}
        {data.map((row, r) =>
          row.map((v, c) => {
            const f = fill?.(r, c)
            const rg = ring?.(r, c)
            const isDim = dim?.(r, c) ?? false
            const pending = v == null
            const x = x0 + c * (cell + gap)
            const y = y0 + r * (cell + gap)
            return (
              <g
                key={`${r}-${c}`}
                opacity={isDim ? 0.3 : 1}
                onClick={onCellClick ? () => onCellClick(r, c) : undefined}
                style={{ cursor: onCellClick ? 'pointer' : undefined, transition: 'opacity 200ms' }}
              >
                <rect
                  x={x + 0.5}
                  y={y + 0.5}
                  width={cell - 1}
                  height={cell - 1}
                  rx={6}
                  fill={f ? toneSoft(f, 28) : pending ? 'var(--surface-2)' : 'var(--surface)'}
                  stroke={rg ? toneColor(rg) : 'var(--border-strong)'}
                  strokeWidth={rg ? 2.4 : 1}
                  style={{ transition: 'fill 200ms, stroke 200ms' }}
                />
                <text
                  x={x + cell / 2}
                  y={y + cell / 2}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fontSize={font}
                  fontWeight={rg ? 700 : 500}
                  fill={pending ? 'var(--subtle)' : 'var(--fg)'}
                  className="font-mono tabular-nums"
                >
                  {pending ? '·' : format(v)}
                </text>
              </g>
            )
          }),
        )}
        {arrows.map((a, k) => {
          const tone = a.tone ?? 'accent'
          const [r1, c1] = a.from
          const [r2, c2] = a.to
          const x1 = cx(c1)
          const y1 = cy(r1)
          const x2 = cx(c2)
          const y2 = cy(r2)
          let d: string
          let lx: number
          let ly: number
          let dotX: number | null = null
          let dotY = 0
          if (isArc(a)) {
            // 弧线：从两个格子的上沿出发，向上拱起
            const sy = y1 - cell / 2
            const ey = y2 - cell / 2 - 1
            // 弧高按跨度线性分配：跨度越大拱得越高，避免多条弧挤在一起
            const maxH = r1 === 0 || r2 === 0 ? padTop - 4 : cell * 0.8
            const span = Math.abs(c2 - c1)
            const h = Math.min(maxH, 8 + ((maxH - 8) * (span - 1)) / Math.max(1, cols - 2))
            const mx = (x1 + x2) / 2
            const my = Math.min(sy, ey) - h * 2
            d = `M${x1},${sy} Q${mx},${my} ${x2},${ey}`
            lx = mx
            ly = (sy + 2 * my + ey) / 4 - 4
          } else {
            const dx = x2 - x1
            const dy = y2 - y1
            const len = Math.hypot(dx, dy) || 1
            const ux = dx / len
            const uy = dy / len
            // 起点：依赖格内部靠近目标的一侧；终点：目标格边界
            const startOff = cell * 0.28
            const sx = x1 + ux * startOff
            const sy = y1 + uy * startOff
            const tEdge = Math.min(Math.abs(ux) > 1e-6 ? cell / 2 / Math.abs(ux) : Infinity, Math.abs(uy) > 1e-6 ? cell / 2 / Math.abs(uy) : Infinity)
            const ex = x2 - ux * (tEdge + 1.5)
            const ey = y2 - uy * (tEdge + 1.5)
            d = `M${sx},${sy} L${ex},${ey}`
            dotX = sx
            dotY = sy
            lx = (sx + ex) / 2
            ly = (sy + ey) / 2 - 4
          }
          return (
            <g key={k} style={{ pointerEvents: 'none' }}>
              <path
                d={d}
                fill="none"
                stroke={toneColor(tone)}
                strokeWidth={2}
                strokeDasharray={a.dashed ? '4 3' : undefined}
                markerEnd={`url(#dpa-${uid}-${tone})`}
                strokeLinecap="round"
              />
              {dotX != null && <circle cx={dotX} cy={dotY} r={2.4} fill={toneColor(tone)} />}
              {a.label && (
                <text
                  x={lx}
                  y={ly}
                  textAnchor="middle"
                  fontSize={10.5}
                  fontWeight={600}
                  fill={toneColor(tone)}
                  className="font-mono"
                  paintOrder="stroke"
                  stroke="var(--surface)"
                  strokeWidth={3}
                >
                  {a.label}
                </text>
              )}
            </g>
          )
        })}
      </svg>
    </div>
  )
}

function isArc(a: DpArrow) {
  if (a.arc != null) return a.arc
  return a.from[0] === a.to[0] && Math.abs(a.from[1] - a.to[1]) >= 2
}

/** 估算等宽标签宽度（px），中文按 2 个字符宽 */
function labelWidth(s: string) {
  let w = 0
  for (const ch of s) w += /[一-鿿]/.test(ch) ? 12 : 7
  return w
}
