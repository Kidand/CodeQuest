/**
 * 难度偏差：0/1 奖励下，一道题对梯度的总权重 Σ|Â_i| 随组内准确率 p 的变化。
 * GRPO 除以组内 std 后，过易 / 过难（p 接近 0 或 1）的题相对权重被放大。
 */
import { useState } from 'react'
import { Legend, VizFrame, toneColor } from '@/viz'
import { Slider } from '@/components/ui'
import { Axes, Line, Svg, Stat, useScales, type Axis } from './plot'

function mass(k: number, G: number) {
  const p = k / G
  const dr = (2 * k * (G - k)) / G // Σ|r_i − mean|
  const std = Math.sqrt((G * p * (1 - p)) / (G - 1)) // ddof=1
  return { dr, grpo: dr / std, std }
}

export function StdBiasViz() {
  const [G, setG] = useState(16)
  const [k, setK] = useState(1)
  const kk = Math.min(k, G - 1)
  const half = Math.floor(G / 2)
  const refDr = mass(half, G).dr
  const refGr = mass(half, G).grpo
  const ptsDr: [number, number][] = []
  const ptsGr: [number, number][] = []
  for (let j = 1; j < G; j++) {
    const m = mass(j, G)
    ptsDr.push([j / G, m.dr / refDr])
    ptsGr.push([j / G, m.grpo / refGr])
  }
  const cur = mass(kk, G)
  const x: Axis = { min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], label: '组内准确率 p = k / G', fmt: (v) => v.toFixed(2) }
  const y: Axis = { min: 0, max: 1.1, ticks: [0, 0.25, 0.5, 0.75, 1], label: '题目总权重（p=0.5 处归一为 1）', fmt: (v) => v.toFixed(2) }
  const { sx, sy } = useScales(x, y)
  const boost = cur.grpo / refGr / (cur.dr / refDr)

  return (
    <VizFrame
      title="std 归一化：越“一边倒”的题，相对权重被放得越大"
      caption={
        <>
          G = {G}，k = {kk}（p = {(kk / G).toFixed(3)}）：组内 std = <b className="font-mono">{cur.std.toFixed(3)}</b>。
          相对 p = 0.5 的题，GRPO 给它的权重是 Dr.GRPO 的 <b className="font-mono">{boost.toFixed(2)}×</b>——
          除以 std 等于对“几乎全对 / 全错”的题额外加权，这就是 Dr.GRPO 说的<b>题目级难度偏差</b>。
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Dr.GRPO：Σ|r − mean|', tone: 'info', variant: 'line' },
            { label: 'GRPO：Σ|r − mean| / std', tone: 'danger', variant: 'line' },
          ]}
        />
      }
      controls={
        <>
          <Slider className="w-44" label="组大小 G" value={G} min={4} max={64} step={2} onChange={setG} />
          <Slider className="w-44" label="答对数 k" value={kk} min={1} max={G - 1} step={1} onChange={setK} />
        </>
      }
    >
      <div className="grid items-center gap-4 @3xl:grid-cols-[minmax(0,3fr)_minmax(0,1fr)]">
        <Svg>
          <Axes x={x} y={y} />
          <Line pts={ptsDr} x={x} y={y} tone="info" />
          <Line pts={ptsGr} x={x} y={y} tone="danger" />
          <circle cx={sx(kk / G)} cy={sy(cur.dr / refDr)} r={5} fill={toneColor('info')} />
          <circle cx={sx(kk / G)} cy={sy(Math.min(1.1, cur.grpo / refGr))} r={5} fill={toneColor('danger')} />
        </Svg>
        <div className="grid grid-cols-3 gap-2 @3xl:grid-cols-1">
          <Stat label="组内 std" value={cur.std.toFixed(3)} />
          <Stat label="1/std 放大倍数" value={(1 / cur.std).toFixed(2)} tone="danger" />
          <Stat label="相对 p=0.5 的额外加权" value={`${boost.toFixed(2)}×`} tone="warning" />
        </div>
      </div>
    </VizFrame>
  )
}
