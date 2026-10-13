/**
 * 网格棋盘：按容器宽度自动缩放格子（手机上不需要横向滚动）。
 * 每个格子由 cell(r, c) 返回 { text, fill, ring, dashed, dim }。
 */
import type { ReactNode } from 'react'
import { fitCell, toneColor, toneSoft, useElementWidth, type VizTone } from '@/viz'

export interface GridCell {
  text?: ReactNode
  /** 填充色 tone；'water' 表示水 / 空位 */
  fill?: VizTone | 'water'
  /** 实色边框（当前处理的格子等） */
  ring?: VizTone
  dashed?: boolean
  dim?: boolean
}

export function GridBoard({ rows, cols, cell, max = 44 }: { rows: number; cols: number; cell: (r: number, c: number) => GridCell; max?: number }) {
  const [ref, avail] = useElementWidth()
  const gap = 3
  const size = fitCell(avail, cols, gap, max, 20)
  const font = size < 26 ? 10 : size < 34 ? 11.5 : 13
  return (
    <div ref={ref} className="w-full min-w-0">
      <div className="mx-auto grid w-max" style={{ gridTemplateColumns: `repeat(${cols}, ${size}px)`, gap }}>
        {Array.from({ length: rows * cols }, (_, k) => {
          const r = Math.floor(k / cols)
          const c = k % cols
          const x = cell(r, c)
          const water = x.fill === 'water'
          const tone = !water ? (x.fill as VizTone | undefined) : undefined
          return (
            <div
              key={k}
              className="flex items-center justify-center rounded-md font-mono font-semibold transition-colors duration-200"
              style={{
                width: size,
                height: size,
                fontSize: font,
                background: water ? 'var(--surface-2)' : tone ? toneSoft(tone, 38) : 'var(--surface)',
                border: `${x.ring ? 2.5 : 1.2}px ${x.dashed ? 'dashed' : 'solid'} ${x.ring ? toneColor(x.ring) : tone ? toneColor(tone) : water ? 'var(--border)' : 'var(--border-strong)'}`,
                opacity: x.dim ? 0.45 : 1,
                color: water ? 'var(--subtle)' : 'var(--fg)',
              }}
            >
              {x.text}
            </div>
          )
        })}
      </div>
    </div>
  )
}
