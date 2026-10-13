/**
 * KTO：前景理论（Kahneman-Tversky）价值函数 → 用 logistic 近似的 KTO 损失。
 * 左：KT 价值函数（α = 0.88、λ = 2.25，参考点 z0）；右：KTO 对 desirable / undesirable 样本的 loss 与梯度大小。
 */
import { useState } from 'react'
import { Legend, VizFrame, toneColor } from '@/viz'
import { Slider } from '@/components/ui'
import { Axes, Line, Stat, Svg, f2, sample, sigmoid, type Axis } from './plot'

export function KtoViz() {
  const [beta, setBeta] = useState(0.1)
  const [ld, setLd] = useState(1)
  const [lu, setLu] = useState(1)
  const [pd, setPd] = useState(0.5)

  const xz: Axis = { min: -10, max: 10, ticks: [-10, -5, 0, 5, 10], label: '结果 z − z₀' }
  const yv: Axis = { min: -18, max: 8, ticks: [-15, -10, -5, 0, 5], label: 'v' }
  const kt = (z: number) => (z >= 0 ? z ** 0.88 : -2.25 * (-z) ** 0.88)

  const xu: Axis = { min: -40, max: 40, ticks: [-40, -20, 0, 20, 40], label: 'r_θ − z₀（log π/π_ref − KL 参考点）' }
  const ymax = Math.max(ld, lu) * 1.1
  const yl: Axis = { min: 0, max: ymax, ticks: [0, Math.round(ymax * 50) / 100, Math.round(ymax / 1.1 * 100) / 100], fmt: (v) => v.toFixed(2), label: 'loss' }
  const lossD = (u: number) => ld * (1 - sigmoid(beta * u))
  const lossU = (u: number) => lu * (1 - sigmoid(-beta * u))
  const gD = (u: number) => (ld * beta * sigmoid(beta * u) * (1 - sigmoid(beta * u))) / (Math.max(ld, lu) * beta * 0.25) * ymax * 0.5

  const ratio = (ld * pd) / (lu * (1 - pd))
  const ok = ratio >= 1 && ratio <= 1.5

  return (
    <VizFrame
      title="KTO：用前景理论处理单条好/坏标注"
      caption={
        <>
          左图是 Kahneman-Tversky 价值函数：收益区凹（风险厌恶），损失区凸且更陡（损失厌恶，λ = 2.25）。KTO 用 logistic 函数替代幂函数（论文称幂函数数值不稳定），
          参考点 z₀ 是策略与参考模型的 KL。右图：好样本的隐式奖励高于 z₀ 越多、坏样本低于 z₀ 越多，loss 越接近 0 且梯度饱和——离参考点很远的样本几乎不再贡献梯度。
          当前 λ_D·n_D / (λ_U·n_U) = <b className="font-mono" style={{ color: toneColor(ok ? 'success' : 'danger') }}>{f2(ratio)}</b>
          {ok ? '，在论文建议的 [1, 1.5] 区间内。' : '，不在论文建议的 [1, 1.5] 区间内，应调整 λ_D 或 λ_U。'}
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'desirable（好）样本 loss', tone: 'success', variant: 'line' },
            { label: 'undesirable（坏）样本 loss', tone: 'danger', variant: 'line' },
            { label: '好样本梯度大小（示意）', tone: 'muted', variant: 'line' },
          ]}
        />
      }
      controls={
        <>
          <Slider className="min-w-[140px] flex-1" label="β" value={beta} min={0.02} max={0.5} step={0.01} onChange={setBeta} format={(v) => v.toFixed(2)} />
          <Slider className="min-w-[140px] flex-1" label="λ_D" value={ld} min={0.5} max={15} step={0.05} onChange={setLd} format={(v) => v.toFixed(2)} />
          <Slider className="min-w-[140px] flex-1" label="λ_U" value={lu} min={0.5} max={3} step={0.05} onChange={setLu} format={(v) => v.toFixed(2)} />
          <Slider className="min-w-[140px] flex-1" label="好样本占比" value={pd} min={0.05} max={0.95} step={0.05} onChange={setPd} format={(v) => v.toFixed(2)} />
        </>
      }
    >
      <div className="grid gap-4 @xl:grid-cols-2">
        <div>
          <div className="mb-1 text-xs font-medium text-muted">① Kahneman-Tversky 价值函数（α = 0.88，λ = 2.25）</div>
          <Svg>
            <Axes x={xz} y={yv} />
            <Line pts={sample(kt, xz, 200)} x={xz} y={yv} tone="accent" />
          </Svg>
        </div>
        <div>
          <div className="mb-1 text-xs font-medium text-muted">② KTO 的单样本 loss = λ_y − v(x, y)</div>
          <Svg>
            <Axes x={xu} y={yl} />
            <Line pts={sample(gD, xu)} x={xu} y={yl} tone="muted" dashed width={1.8} />
            <Line pts={sample(lossD, xu)} x={xu} y={yl} tone="success" />
            <Line pts={sample(lossU, xu)} x={xu} y={yl} tone="danger" />
          </Svg>
        </div>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2">
        <Stat label="λ_D·n_D / (λ_U·n_U)" value={f2(ratio)} tone={ok ? 'success' : 'danger'} sub="建议 ∈ [1, 1.5]" />
        <Stat label="u = 0 处的 loss（好 / 坏）" value={`${f2(lossD(0))} / ${f2(lossU(0))}`} sub="= λ_y / 2" />
      </div>
    </VizFrame>
  )
}
