import { cn, fmt } from '@/lib/utils'
import { fitCell, useElementWidth } from './useWidth'
import { divergingColor, heatColor, textOnHeat, toneColor, toneSoft, type VizTone } from './tones'

/**
 * 矩阵 / 热力图（注意力分数、DP 表、权重矩阵、掩码…）。
 *  - scale='seq'：顺序色（domain 默认 [min,max]）；'div'：发散色（以 0 为中心）；'none'：不着色
 *  - highlight(r,c) 返回 tone 时给单元格描边；masked(r,c) 为 true 时显示斜线遮罩
 */
export function MatrixView({
  data,
  rowLabels,
  colLabels,
  scale = 'seq',
  domain,
  format = (v) => fmt(v, 2),
  showValues = true,
  highlight,
  masked,
  cellSize = 40,
  onCellClick,
  corner,
  fit = true,
  className,
}: {
  data: (number | null)[][]
  rowLabels?: React.ReactNode[]
  colLabels?: React.ReactNode[]
  scale?: 'seq' | 'div' | 'none'
  domain?: [number, number]
  format?: (v: number) => string
  showValues?: boolean
  highlight?: (r: number, c: number) => VizTone | undefined | null | false
  masked?: (r: number, c: number) => boolean
  cellSize?: number
  onCellClick?: (r: number, c: number) => void
  corner?: React.ReactNode
  /** 按容器宽度自动缩小格子（cellSize 视为上限），默认开启 */
  fit?: boolean
  className?: string
}) {
  const [wrapRef, avail] = useElementWidth()
  const flat = data.flat().filter((v): v is number => v != null && Number.isFinite(v))
  const lo = domain?.[0] ?? Math.min(0, ...flat)
  const hi = domain?.[1] ?? Math.max(...flat, 1e-9)
  const maxAbs = Math.max(Math.abs(lo), Math.abs(hi), 1e-9)
  const cols = data[0]?.length ?? 0
  if (fit) cellSize = fitCell(avail - (rowLabels ? 44 : 0), cols, 3, cellSize, 18)
  const fontSize = cellSize < 26 ? 9 : cellSize < 34 ? 10 : cellSize < 44 ? 11 : 12

  return (
    <div ref={wrapRef} className="w-full min-w-0">
    <div className={cn('mx-auto w-max', className)}>
      <div
        className="grid"
        style={{
          gridTemplateColumns: `${rowLabels ? 'auto ' : ''}repeat(${cols}, ${cellSize}px)`,
          gap: 3,
        }}
      >
        {colLabels && (
          <>
            {rowLabels && <div className="flex items-end justify-end pr-1 text-[10px] text-subtle">{corner}</div>}
            {colLabels.map((l, c) => (
              <div key={c} className="truncate pb-1 text-center font-mono text-[11px] text-muted" style={{ width: cellSize }}>
                {l}
              </div>
            ))}
          </>
        )}
        {data.map((row, r) => (
          <Row key={r}>
            {rowLabels && <div className="flex items-center justify-end whitespace-nowrap pr-2 font-mono text-[11px] text-muted">{rowLabels[r]}</div>}
            {row.map((v, c) => {
              const isMasked = masked?.(r, c) ?? false
              const hl = highlight?.(r, c)
              let bg = 'var(--surface)'
              let fg = 'var(--fg)'
              if (v != null && !isMasked && scale !== 'none') {
                if (scale === 'div') {
                  const t = v / maxAbs
                  bg = divergingColor(t)
                  fg = textOnHeat(t)
                } else {
                  const t = hi === lo ? 0 : (v - lo) / (hi - lo)
                  bg = heatColor(t)
                  fg = textOnHeat(t)
                }
              }
              if (hl && scale === 'none') bg = toneSoft(hl, 24)
              return (
                <div
                  key={c}
                  onClick={onCellClick ? () => onCellClick(r, c) : undefined}
                  className={cn(
                    'relative flex items-center justify-center overflow-hidden rounded-md font-mono tabular-nums transition-[background,box-shadow] duration-200',
                    onCellClick && 'cursor-pointer hover:brightness-95',
                  )}
                  style={{
                    width: cellSize,
                    height: cellSize,
                    // 只用 background 简写（与 backgroundImage 混用会在重渲染时触发 React 警告）
                    background: isMasked
                      ? 'repeating-linear-gradient(135deg, transparent 0 5px, color-mix(in oklab, var(--subtle) 30%, transparent) 5px 6px), var(--surface-2)'
                      : bg,
                    color: isMasked ? 'var(--subtle)' : fg,
                    fontSize,
                    boxShadow: hl ? `inset 0 0 0 2px ${toneColor(hl)}` : 'inset 0 0 0 1px var(--border)',
                  }}
                >
                  {showValues && !isMasked && v != null ? format(v) : isMasked ? '' : v == null ? '·' : ''}
                </div>
              )
            })}
          </Row>
        ))}
      </div>
    </div>
    </div>
  )
}

const Row = ({ children }: { children: React.ReactNode }) => <>{children}</>
