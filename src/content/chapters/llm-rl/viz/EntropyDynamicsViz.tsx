/**
 * 熵的动力学：单状态 softmax 策略做精确策略梯度（∂J/∂z_a = π_a·A_a）。
 * 一阶近似 ΔH ≈ −η·Cov_{a~π}(log π(a), π(a)·A(a))（Cui et al. 2025 对 softmax + PG 的结论）。
 * 洞见：奖励落在高概率动作上 → 协方差为正 → 熵下降；落在低概率动作上 → 熵先升。
 */
import { useMemo, useState } from 'react'
import { Legend, ProbBars, VizFrame, toneColor } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { Axes, Line, Svg, Stat, useScales, type Axis } from './plot'

const TOK = ['So', 'Let', 'We', 'Wait', 'Hmm', 'Check']
const Z0 = [2.0, 1.2, 0.5, 0.0, -0.5, -1.0]
const STEPS = 60
type Case = 'exploit' | 'explore'
const REWARD: Record<Case, number[]> = { exploit: [1, 0, 0, 0, 0, 0], explore: [0, 0, 0, 0, 1, 0] }

const softmax = (z: number[]) => {
  const m = Math.max(...z)
  const e = z.map((v) => Math.exp(v - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}
const ent = (p: number[]) => -p.reduce((s, v) => s + (v > 0 ? v * Math.log(v) : 0), 0)

export function EntropyDynamicsViz() {
  const [cs, setCs] = useState<Case>('exploit')
  const [eta, setEta] = useState(1)
  const [step, setStep] = useState(0)
  const traj = useMemo(() => {
    const r = REWARD[cs]
    let z = Z0.slice()
    const out: { p: number[]; H: number; pred: number }[] = []
    for (let t = 0; t <= STEPS; t++) {
      const p = softmax(z)
      const Er = p.reduce((s, v, i) => s + v * r[i], 0)
      const A = r.map((v) => v - Er)
      const lp = p.map(Math.log)
      const Elp = p.reduce((s, v, i) => s + v * lp[i], 0)
      const g = p.map((v, i) => v * A[i])
      const Eg = p.reduce((s, v, i) => s + v * g[i], 0)
      const cov = p.reduce((s, v, i) => s + v * (lp[i] - Elp) * (g[i] - Eg), 0)
      out.push({ p, H: ent(p), pred: -eta * cov })
      z = z.map((v, i) => v + eta * g[i])
    }
    return out
  }, [cs, eta])
  const cur = traj[step]
  const x: Axis = { min: 0, max: STEPS, ticks: [0, 10, 20, 30, 40, 50, 60], label: '训练步' }
  const y: Axis = { min: 0, max: 1.85, ticks: [0, 0.5, 1, 1.5], label: '熵 H(π)', fmt: (v) => v.toFixed(1) }
  const { sx, sy } = useScales(x, y)
  const actual = step < STEPS ? traj[step + 1].H - cur.H : 0

  return (
    <VizFrame
      title="熵坍缩的机制：奖励落在高概率 token 上，熵就下降"
      caption={
        <>
          第 {step} 步：H = <b className="font-mono">{cur.H.toFixed(3)}</b>。一阶预测 −η·Cov(log π, π·A) = <b className="font-mono">{cur.pred.toFixed(4)}</b>，
          实际下一步 ΔH = <b className="font-mono">{actual.toFixed(4)}</b>。
          {cs === 'exploit'
            ? ' 正确答案恰是模型最自信的 token：协方差为正，熵单调下降——这就是 RLVR 中最常见的情形。'
            : ' 正确答案是低概率的 “Hmm”：起初协方差为负、熵上升（探索），等它成为主导后熵再下降。'}
        </>
      }
      legend={<Legend items={[{ label: '熵曲线', tone: 'accent', variant: 'line' }, { label: '当前步', tone: 'gold' }]} />}
      controls={
        <>
          <Segmented
            size="sm"
            value={cs}
            onChange={setCs}
            options={[
              { value: 'exploit', label: '奖励给高概率 “So”' },
              { value: 'explore', label: '奖励给低概率 “Hmm”' },
            ]}
          />
          <Slider className="w-40" label="学习率 η" value={eta} min={0.2} max={3} step={0.1} onChange={setEta} format={(v) => v.toFixed(1)} />
          <Slider className="min-w-[180px] flex-1" label="训练步" value={step} min={0} max={STEPS} onChange={setStep} />
        </>
      }
    >
      <div className="grid items-center gap-5 @3xl:grid-cols-2">
        <Svg>
          <Axes x={x} y={y} />
          <Line pts={traj.map((t, i) => [i, t.H])} x={x} y={y} tone="accent" />
          <circle cx={sx(step)} cy={sy(cur.H)} r={6} fill={toneColor('gold')} stroke="var(--surface)" strokeWidth={2} />
        </Svg>
        <div className="min-w-0 space-y-3">
          <ProbBars items={TOK.map((t, i) => ({ label: t, p: cur.p[i], tone: REWARD[cs][i] ? 'success' : 'accent', note: REWARD[cs][i] ? 'r=1' : '' }))} max={1} />
          <div className="grid grid-cols-2 gap-2">
            <Stat label="熵 H" value={cur.H.toFixed(3)} />
            <Stat label="预测 ΔH" value={cur.pred.toFixed(4)} tone={cur.pred < 0 ? 'danger' : 'success'} />
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
