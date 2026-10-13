/**
 * 高熵“分叉 token”：一段推理里只有少数位置真正在做选择。
 * 每个位置给出 top-4 候选概率（其余质量平均摊到 60 个尾部 token），计算熵并高亮 top-ρ。
 */
import { useMemo, useState } from 'react'
import { Legend, TokenRow, VizFrame } from '@/viz'
import { Slider } from '@/components/ui'

const SEQ: { tok: string; top: number[] }[] = [
  { tok: 'To', top: [0.9, 0.05, 0.02, 0.01] },
  { tok: ' solve', top: [0.97, 0.01, 0.01, 0.005] },
  { tok: ',', top: [0.95, 0.02, 0.01, 0.01] },
  { tok: ' let', top: [0.55, 0.25, 0.1, 0.05] },
  { tok: ' x', top: [0.93, 0.03, 0.02, 0.01] },
  { tok: ' =', top: [0.98, 0.01, 0.005, 0.003] },
  { tok: ' 3', top: [0.99, 0.004, 0.003, 0.001] },
  { tok: '.', top: [0.8, 0.12, 0.04, 0.02] },
  { tok: ' Wait', top: [0.32, 0.28, 0.2, 0.1] },
  { tok: ',', top: [0.96, 0.02, 0.01, 0.005] },
  { tok: ' check', top: [0.4, 0.3, 0.15, 0.1] },
  { tok: ' again', top: [0.88, 0.06, 0.03, 0.01] },
  { tok: ':', top: [0.92, 0.04, 0.02, 0.01] },
  { tok: ' Suppose', top: [0.3, 0.25, 0.25, 0.1] },
  { tok: ' x', top: [0.9, 0.05, 0.03, 0.01] },
  { tok: ' >', top: [0.5, 0.3, 0.1, 0.05] },
  { tok: ' 0', top: [0.97, 0.01, 0.01, 0.005] },
  { tok: ' thus', top: [0.6, 0.2, 0.1, 0.05] },
]
const TAIL = 60

function entropy(top: number[]) {
  const rest = Math.max(0, 1 - top.reduce((a, b) => a + b, 0))
  const probs = [...top, ...Array(TAIL).fill(rest / TAIL)]
  return -probs.reduce((s, p) => s + (p > 0 ? p * Math.log(p) : 0), 0)
}

export function ForkingViz() {
  const [rho, setRho] = useState(0.2)
  const H = useMemo(() => SEQ.map((s) => entropy(s.top)), [])
  const k = Math.ceil(rho * SEQ.length)
  const order = H.map((h, i) => [h, i] as const).sort((a, b) => b[0] - a[0])
  const sel = new Set(order.slice(0, k).map(([, i]) => i))
  const thr = order[k - 1]?.[0] ?? 0

  return (
    <VizFrame
      title="少数高熵 token 决定推理走向"
      caption={
        <>
          18 个 token 中熵最高的 {k} 个（ρ = {rho.toFixed(2)}，阈值 H ≥ {thr.toFixed(2)}）：
          <b>{[...sel].sort((a, b) => a - b).map((i) => SEQ[i].tok.trim()).join('、')}</b>。
          它们是 “Wait / check / Suppose” 这类改变推理方向的位置；其余 token 几乎是确定的。只在这些 token 上算策略梯度，就是 “80/20” 论文的做法。
        </>
      }
      legend={
        <Legend
          items={[
            { label: '被选中的高熵 token', tone: 'warning', variant: 'soft' },
            { label: '低熵 token（下方数字为熵）', tone: 'muted', variant: 'soft' },
          ]}
        />
      }
      controls={<Slider className="min-w-[220px] flex-1" label="保留比例 ρ" value={rho} min={0.05} max={1} step={0.05} onChange={setRho} format={(v) => v.toFixed(2)} />}
    >
      <TokenRow tokens={SEQ.map((s, i) => ({ text: s.tok, sub: H[i].toFixed(2), tone: sel.has(i) ? 'warning' : undefined, dim: !sel.has(i) }))} />
    </VizFrame>
  )
}
