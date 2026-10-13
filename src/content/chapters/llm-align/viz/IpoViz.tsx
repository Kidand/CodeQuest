/**
 * 确定性偏好（同一对样本永远 y_w ≻ y_l）下：
 *   DPO 的 −log σ(βh) 没有最小值点，h 被一直推大（π(y_l) → 0，KL 约束失效）；
 *   IPO 的 (h − 1/(2τ))² 在 h = 1/(2τ) 处停下。
 * h = log[π(y_w)π_ref(y_l) / (π(y_l)π_ref(y_w))]，即 DPO 里的 log-ratio margin Δ。
 */
import { useMemo, useState } from 'react'
import { Legend, VizFrame } from '@/viz'
import { Slider } from '@/components/ui'
import { Axes, Dot, Line, Stat, Svg, f2, logSigmoid, sample, sigmoid, type Axis } from './plot'

const STEPS = 3000

function simulate(beta: number) {
  // 在 h 空间做梯度下降（等价于只有一个自由度的“策略”）
  const dpo: number[] = [0]
  const ipo: number[] = [0]
  let hd = 0
  let hi = 0
  const target = 1 / (2 * beta)
  const lrD = 1 / (beta * beta) / 40 // 让两种方法在 h=0 处的初始步长同量级
  const lrI = 0.02
  for (let k = 0; k < STEPS; k++) {
    hd += lrD * beta * sigmoid(-beta * hd)
    hi -= lrI * 2 * (hi - target)
    dpo.push(hd)
    ipo.push(hi)
  }
  return { dpo, ipo, target }
}

export function IpoViz() {
  const [beta, setBeta] = useState(0.1)
  const [step, setStep] = useState(1500)
  const sim = useMemo(() => simulate(beta), [beta])
  const hD = sim.dpo[step]
  const hI = sim.ipo[step]
  const hmax = Math.max(sim.dpo[STEPS], sim.target) * 1.1
  const xs: Axis = { min: 0, max: STEPS, ticks: [0, 1000, 2000, 3000], fmt: (v) => (v === 0 ? '0' : `${v / 1000}k`), label: '训练步数' }
  const yh: Axis = { min: 0, max: hmax, ticks: [0, Math.round(sim.target * 10) / 10, Math.round(hmax / 1.1)], fmt: (v) => v.toFixed(1), label: 'margin h' }
  const xh: Axis = { min: -1 / beta, max: 4 / beta, ticks: [0, sim.target, 2 / beta, 4 / beta].map((v) => Math.round(v * 10) / 10), fmt: (v) => v.toFixed(0), label: 'h' }
  const yl: Axis = { min: 0, max: 2, ticks: [0, 0.693, 1, 2], fmt: (v) => (v === 0.693 ? 'ln2' : String(v)), label: 'loss' }
  const ipoScale = (h: number) => (h - sim.target) ** 2 * beta * beta // 纵向缩放到同一量级，便于对比形状

  return (
    <VizFrame
      title="确定性偏好：DPO 停不下，IPO 有终点"
      caption={
        <>
          第 {step} 步：DPO 的 margin h = <b className="font-mono">{f2(hD)}</b>（模型认为 y_w 更好的概率 σ(βh) = {f2(sigmoid(beta * hD))}，仍在增长）；
          IPO 的 h = <b className="font-mono">{f2(hI)}</b>，收敛到目标 1/(2τ) = <b className="font-mono">{f2(sim.target)}</b>。
          DPO 的 loss 没有最小值点，只要数据说“y_w 永远更好”，它就会持续压低 π(y_l)，β 只能减慢、不能阻止这一过程。
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'DPO：−log σ(βh)', tone: 'danger', variant: 'line' },
            { label: 'IPO：(h − 1/(2τ))²（纵轴已缩放）', tone: 'success', variant: 'line' },
          ]}
        />
      }
      controls={
        <>
          <Slider className="min-w-[160px] flex-1" label="β（= IPO 的 τ）" value={beta} min={0.05} max={0.5} step={0.01} onChange={setBeta} format={(v) => v.toFixed(2)} />
          <Slider className="min-w-[160px] flex-1" label="训练步数" value={step} min={0} max={STEPS} step={10} onChange={setStep} />
        </>
      }
    >
      <div className="grid gap-4 @xl:grid-cols-2">
        <div>
          <div className="mb-1 text-xs font-medium text-muted">① loss 关于 h 的形状</div>
          <Svg>
            <Axes x={xh} y={yl} />
            <Line pts={sample((h) => -logSigmoid(beta * h), xh)} x={xh} y={yl} tone="danger" />
            <Line pts={sample(ipoScale, xh)} x={xh} y={yl} tone="success" />
            <Dot x={xh} y={yl} px={Math.min(hD, xh.max)} py={-logSigmoid(beta * hD)} tone="danger" />
            <Dot x={xh} y={yl} px={hI} py={ipoScale(hI)} tone="success" />
          </Svg>
        </div>
        <div>
          <div className="mb-1 text-xs font-medium text-muted">② 同一对样本反复训练时 h 的轨迹</div>
          <Svg>
            <Axes x={xs} y={yh} />
            <Line pts={sim.dpo.filter((_, k) => k % 15 === 0).map((v, k) => [k * 15, v])} x={xs} y={yh} tone="danger" />
            <Line pts={sim.ipo.filter((_, k) => k % 15 === 0).map((v, k) => [k * 15, v])} x={xs} y={yh} tone="success" />
            <Line pts={[[0, sim.target], [STEPS, sim.target]]} x={xs} y={yh} tone="muted" dashed width={1.5} />
            <Dot x={xs} y={yh} px={step} py={hD} tone="danger" />
            <Dot x={xs} y={yh} px={step} py={hI} tone="success" />
          </Svg>
        </div>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2">
        <Stat label="DPO 的 h（无上界）" value={f2(hD)} tone="danger" />
        <Stat label="IPO 的 h → 1/(2τ)" value={f2(hI)} tone="success" />
      </div>
    </VizFrame>
  )
}
