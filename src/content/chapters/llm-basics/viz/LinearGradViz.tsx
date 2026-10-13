/**
 * Linear 层 Y = X W + b 的反向：用“形状对齐”记公式。
 * 点击梯度矩阵的任意格子，高亮参与计算的那一行 / 列：
 *   dW[i, j] = Σ_b X[b, i] · dY[b, j]   →  dW = Xᵀ dY   (in×B)(B×out)
 *   dX[b, i] = Σ_j dY[b, j] · W[i, j]   →  dX = dY Wᵀ   (B×out)(out×in)
 *   db[j]    = Σ_b dY[b, j]
 */
import { useMemo, useState } from 'react'
import { Legend, MatrixView, VizFrame } from '@/viz'
import { Segmented } from '@/components/ui'
import { C } from './plot'

const X = [
  [1, 0, 2],
  [-1, 1, 0],
  [0, 2, 1],
  [1, 1, -1],
]
const W = [
  [0.5, -1, 0, 1],
  [1, 0.5, -0.5, 0],
  [0, 1, 1, -1],
]
const DY = [
  [0.1, -0.2, 0, 0.3],
  [0, 0.1, 0.2, -0.1],
  [0.2, 0, -0.1, 0.1],
  [-0.1, 0.3, 0, 0.2],
]
const B = 4
const IN = 3
const OUT = 4
type Mode = 'dW' | 'dX' | 'db'
const r2 = (v: number) => Math.round(v * 100) / 100

export function LinearGradViz() {
  const [mode, setMode] = useState<Mode>('dW')
  const [sel, setSel] = useState<[number, number]>([0, 1])
  const dW = useMemo(() => Array.from({ length: IN }, (_, i) => Array.from({ length: OUT }, (_, j) => r2(X.reduce((s, row, b) => s + row[i] * DY[b][j], 0)))), [])
  const dX = useMemo(() => Array.from({ length: B }, (_, b) => Array.from({ length: IN }, (_, i) => r2(DY[b].reduce((s, g, j) => s + g * W[i][j], 0)))), [])
  const db = useMemo(() => [Array.from({ length: OUT }, (_, j) => r2(DY.reduce((s, row) => s + row[j], 0)))], [])
  const pick = (m: Mode) => {
    setMode(m)
    setSel(m === 'dW' ? [0, 1] : m === 'dX' ? [2, 1] : [0, 3])
  }
  const [r, c] = sel

  let terms = ''
  if (mode === 'dW') terms = X.map((row, b) => `${row[r]}·${DY[b][c]}`).join(' + ') + ` = ${dW[r][c]}`
  else if (mode === 'dX') terms = DY[r].map((g, j) => `${g}·${W[c][j]}`).join(' + ') + ` = ${dX[r][c]}`
  else terms = DY.map((row) => `${row[c]}`).join(' + ') + ` = ${db[0][c]}`

  const res = mode === 'dW' ? dW : mode === 'dX' ? dX : db
  const resLabel = mode === 'dW' ? `dW = Xᵀ·dY  (${IN}×${OUT})` : mode === 'dX' ? `dX = dY·Wᵀ  (${B}×${IN})` : `db = dY.sum(0)  (${OUT},)`

  return (
    <VizFrame
      title="Linear 反向：梯度的形状必须和参数一样，公式由形状唯一确定"
      caption={
        <>
          {mode === 'dW' && <>dW[{r}, {c}] = Σ_b X[b, {r}]·dY[b, {c}]：X 的第 {r} 列与 dY 的第 {c} 列做内积（对 batch 求和）。</>}
          {mode === 'dX' && <>dX[{r}, {c}] = Σ_j dY[{r}, j]·W[{c}, j]：dY 的第 {r} 行与 W 的第 {c} 行做内积。</>}
          {mode === 'db' && <>db[{c}] = Σ_b dY[b, {c}]：前向里 b 被广播到每个样本，反向就把各样本的梯度加起来。</>}
          {' '}<C>{terms}</C>。点击右侧结果矩阵的格子换一个位置。
        </>
      }
      legend={<Legend items={[{ label: '参与计算的元素', tone: 'accent', variant: 'soft' }, { label: '当前梯度元素', tone: 'gold', variant: 'soft' }]} />}
      controls={
        <Segmented
          size="sm"
          value={mode}
          onChange={pick}
          options={[
            { value: 'dW', label: 'dW = Xᵀ dY' },
            { value: 'dX', label: 'dX = dY Wᵀ' },
            { value: 'db', label: 'db = Σ dY' },
          ]}
        />
      }
    >
      <div className="grid items-start gap-4 @2xl:grid-cols-3">
        <div className="min-w-0">
          <div className="mb-1 font-mono text-[11.5px] text-muted">{mode === 'dX' ? `W  (${IN}×${OUT})` : `X  (${B}×${IN})`}</div>
          {mode === 'dX' ? (
            <MatrixView data={W} scale="none" cellSize={40} format={(v) => String(v)} highlight={(i) => (i === c ? 'accent' : null)} />
          ) : (
            <MatrixView data={X} scale="none" cellSize={40} format={(v) => String(v)} highlight={(_, i) => (mode === 'dW' && i === r ? 'accent' : null)} />
          )}
        </div>
        <div className="min-w-0">
          <div className="mb-1 font-mono text-[11.5px] text-muted">dY  ({B}×{OUT})（上游梯度）</div>
          <MatrixView
            data={DY}
            scale="none"
            cellSize={40}
            format={(v) => String(v)}
            highlight={(b, j) => ((mode === 'dX' && b === r) || (mode !== 'dX' && j === c) ? 'accent' : null)}
          />
        </div>
        <div className="min-w-0">
          <div className="mb-1 font-mono text-[11.5px] text-muted">{resLabel}</div>
          <MatrixView data={res} scale="none" cellSize={40} format={(v) => String(v)} highlight={(i, j) => (i === r && j === c ? 'gold' : null)} onCellClick={(i, j) => setSel([i, j])} />
        </div>
      </div>
    </VizFrame>
  )
}
