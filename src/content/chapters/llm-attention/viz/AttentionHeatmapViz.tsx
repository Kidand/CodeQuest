/**
 * SDPA 第一图：点一个 query token，看它在所有 key 上的注意力分布（软寻址）。
 * Q、K 为人工设计的 4 维向量（维度含义：动物 / 物体 / 动作 / 功能词），权重由 softmax(QKᵀ/√d) 实算。
 */
import { useMemo, useState } from 'react'
import { Legend, MatrixView, ProbBars, VizFrame } from '@/viz'
import { Switch } from '@/components/ui'
import { softmax, dot } from './math'

const TOKENS = ['小猫', '坐在', '垫子', '上', '因为', '它', '很', '软']
// key：「我是什么」
const K = [
  [3, 0, 0, 0],
  [0, 0, 3, 0],
  [0, 3, 0, 0],
  [0, 1, 0, 1.5],
  [0, 0, 0, 2],
  [0.5, 1.5, 0, 0.5],
  [0, 0, 0, 1.5],
  [0, 1, 0.5, 0],
]
// query：「我在找什么」
const Q = [
  [1, 0, 0.5, 0],
  [2, 1, 0, 0],
  [0, 0.5, 1.5, 0.5],
  [0, 2, 0, 0],
  [0, 0, 2, 0.5],
  [0.5, 2.5, 0, 0],
  [0, 0.5, 0, 1],
  [0.3, 2, 0, 0],
]
const D = 4

export function AttentionHeatmapViz() {
  const [row, setRow] = useState(5)
  const [causal, setCausal] = useState(false)
  const scores = useMemo(() => Q.map((q) => K.map((k) => dot(q, k) / Math.sqrt(D))), [])
  const weights = useMemo(
    () => scores.map((r, i) => softmax(r.map((s, j) => (causal && j > i ? -Infinity : s)))),
    [scores, causal],
  )
  const w = weights[row]
  const top = w.indexOf(Math.max(...w))

  return (
    <VizFrame
      title="注意力 = query 对所有 key 的软寻址"
      caption={
        <>
          query「<b>{TOKENS[row]}</b>」的打分 <span className="font-mono">qᵀk/√d</span> 经 softmax 后，权重最大的是「<b>{TOKENS[top]}</b>」（
          {(w[top] * 100).toFixed(1)}%）。输出 = 各 token 的 value 按这一行权重加权求和。点击热力图任一行换一个 query。
        </>
      }
      legend={<Legend items={[{ label: '注意力权重（每行和为 1）', tone: 'accent', variant: 'soft' }, { label: '选中的 query 行', tone: 'warning', variant: 'soft' }]} />}
      controls={<Switch checked={causal} onChange={setCausal} label="因果掩码（只能看自己和左边）" />}
    >
      <div className="grid items-start gap-5 @3xl:grid-cols-[1.3fr_1fr]">
        <div className="min-w-0">
          <div className="mb-2 text-xs font-medium text-muted">softmax(QKᵀ/√d)：行 = query，列 = key</div>
          <MatrixView
            data={weights.map((r) => r.map((v) => (Number.isNaN(v) ? null : v)))}
            rowLabels={TOKENS}
            colLabels={TOKENS}
            domain={[0, 1]}
            cellSize={40}
            format={(v) => (v < 0.005 ? '0' : v.toFixed(2).replace(/^0/, ''))}
            masked={(r, c) => causal && c > r}
            highlight={(r) => (r === row ? 'warning' : null)}
            onCellClick={(r) => setRow(r)}
          />
        </div>
        <div className="min-w-0">
          <div className="mb-2 text-xs font-medium text-muted">「{TOKENS[row]}」这一行的分布</div>
          <ProbBars items={TOKENS.map((t, j) => ({ label: t, p: w[j], tone: j === top ? 'success' : 'accent', dim: causal && j > row }))} max={1} />
        </div>
      </div>
    </VizFrame>
  )
}
