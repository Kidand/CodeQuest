/**
 * DPO 的 loss 与梯度权重随 “log-ratio margin” Δ 和 β 的变化。
 *   Δ = log(π_θ/π_ref)(y_w) − log(π_θ/π_ref)(y_l)
 *   loss = −log σ(βΔ)，  ∂loss/∂Δ = −β·σ(−βΔ)
 */
import { useState } from 'react'
import { Legend, VizFrame } from '@/viz'
import { Slider } from '@/components/ui'
import { Axes, Dot, Line, Stat, Svg, f3, logSigmoid, sample, sigmoid, type Axis } from './plot'

export function DpoLossViz() {
  const [beta, setBeta] = useState(0.1)
  const [delta, setDelta] = useState(-6)
  const x: Axis = { min: -30, max: 30, ticks: [-30, -20, -10, 0, 10, 20, 30], label: 'Δ = log-ratio margin（nats）' }
  const yl: Axis = { min: 0, max: 3.5, ticks: [0, 0.693, 1.5, 2.5, 3.5], fmt: (v) => (v === 0.693 ? 'ln2' : String(v)), label: 'loss' }
  const yw: Axis = { min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], label: '权重' }

  const loss = (d: number, b: number) => -logSigmoid(b * d)
  const w = (d: number, b: number) => sigmoid(-b * d)
  const cur = { loss: loss(delta, beta), w: w(delta, beta) }

  return (
    <VizFrame
      title="β 与 margin 如何决定 loss 和梯度"
      caption={
        <>
          Δ = <b className="font-mono">{delta.toFixed(1)}</b>、β = <b className="font-mono">{beta.toFixed(2)}</b>：隐式奖励差 βΔ = <span className="font-mono">{f3(beta * delta)}</span>，
          模型认为 y_w 更好的概率 σ(βΔ) = <span className="font-mono">{f3(1 - cur.w)}</span>。
          {delta < 0
            ? ' 这对偏好被排反了（σ(βΔ) < 0.5），权重 σ(−βΔ) > 0.5——排得越反，权重越接近 1，DPO 会重点纠正它。'
            : cur.w < 0.1
              ? ' 已经排得很对，权重很小：几乎不再推它（但永远不为 0）。'
              : ' 排序正确但还不够自信，仍有可观的梯度。'}
        </>
      }
      legend={
        <Legend
          items={[
            { label: `当前 β 的 loss / 权重 σ(−βΔ)`, tone: 'accent', variant: 'line' },
            { label: '梯度系数 β·σ(−βΔ)', tone: 'warning', variant: 'line' },
            { label: 'β = 0.5 对照', tone: 'muted', variant: 'line' },
          ]}
        />
      }
      controls={
        <>
          <Slider className="min-w-[180px] flex-1" label="β" value={beta} min={0.02} max={1} step={0.01} onChange={setBeta} format={(v) => v.toFixed(2)} />
          <Slider className="min-w-[180px] flex-1" label="margin Δ" value={delta} min={-30} max={30} step={0.5} onChange={setDelta} format={(v) => v.toFixed(1)} />
        </>
      }
    >
      <div className="grid gap-4 @xl:grid-cols-2">
        <div>
          <div className="mb-1 text-xs font-medium text-muted">① loss = −log σ(βΔ)：Δ = 0 时恒为 ln2 ≈ 0.693</div>
          <Svg>
            <Axes x={x} y={yl} />
            <Line pts={sample((d) => loss(d, 0.5), x)} x={x} y={yl} tone="muted" dashed width={1.8} />
            <Line pts={sample((d) => loss(d, beta), x)} x={x} y={yl} tone="accent" />
            <Dot x={x} y={yl} px={delta} py={cur.loss} tone="accent" />
          </Svg>
        </div>
        <div>
          <div className="mb-1 text-xs font-medium text-muted">② 每对样本的梯度权重 σ(−βΔ)</div>
          <Svg>
            <Axes x={x} y={yw} />
            <Line pts={sample((d) => w(d, 0.5), x)} x={x} y={yw} tone="muted" dashed width={1.8} />
            <Line pts={sample((d) => beta * w(d, beta), x)} x={x} y={yw} tone="warning" width={2} />
            <Line pts={sample((d) => w(d, beta), x)} x={x} y={yw} tone="accent" />
            <Dot x={x} y={yw} px={delta} py={cur.w} tone="accent" />
          </Svg>
        </div>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2 @xl:grid-cols-4">
        <Stat label="loss" value={f3(cur.loss)} tone="accent" />
        <Stat label="权重 σ(−βΔ)" value={f3(cur.w)} />
        <Stat label="∂loss/∂Δ = −β·σ(−βΔ)" value={f3(-beta * cur.w)} tone="warning" />
        <Stat label="隐式奖励差 βΔ" value={f3(beta * delta)} />
      </div>
    </VizFrame>
  )
}
