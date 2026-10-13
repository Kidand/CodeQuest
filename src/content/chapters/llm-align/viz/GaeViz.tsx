/**
 * GAE 反向递推：δ_t = r_t + γV(s_{t+1}) − V(s_t)，A_t = δ_t + γλA_{t+1}。
 * 逐帧从最后一个 token 往前算；下方折线对比 λ = 0 / 当前 λ / λ = 1 的优势估计。
 */
import { useMemo, useState } from 'react'
import { ArrayView, CodeTrace, Legend, StepControls, VizFrame, useStepper } from '@/viz'
import { Slider } from '@/components/ui'
import { Axes, Line, Svg, type Axis } from './plot'

const TOKS = ['猫', '坐', '在', '垫', '子', '上', '。', '<eos>']
const R = [-0.005, -0.06, 0.02, -0.12, 0, -0.03, 0.01, 1.495] // −β·KL_t，末 token 加 RM 分数
const V = [0.6, 0.7, 0.75, 0.9, 1.0, 1.1, 1.3, 1.4] // critic 估计
const T = R.length

const CODE = `last_gae = 0.0
for t in reversed(range(T)):
    next_v = values[t + 1] if t + 1 < T else 0.0   # 终止后价值为 0
    delta = rewards[t] + gamma * next_v - values[t]
    last_gae = delta + gamma * lam * last_gae
    adv[t] = last_gae
returns = adv + values                              # critic 的回归目标`

function gae(gamma: number, lam: number) {
  const adv = new Array(T).fill(0)
  const delta = new Array(T).fill(0)
  let last = 0
  for (let t = T - 1; t >= 0; t--) {
    const nv = t + 1 < T ? V[t + 1] : 0
    delta[t] = R[t] + gamma * nv - V[t]
    last = delta[t] + gamma * lam * last
    adv[t] = last
  }
  return { adv, delta }
}

interface Frame {
  t: number // 当前处理的 t；T 表示尚未开始，-1 表示结束
  line: number | number[]
}

export function GaeViz() {
  const [gamma, setGamma] = useState(1)
  const [lam, setLam] = useState(0.95)
  const res = useMemo(() => gae(gamma, lam), [gamma, lam])
  const frames = useMemo<Frame[]>(() => [{ t: T, line: 1 }, ...Array.from({ length: T }, (_, k) => ({ t: T - 1 - k, line: [3, 4, 5, 6] })), { t: -1, line: 7 }], [])
  const st = useStepper(frames, { intervalMs: 1000 })
  const f = st.frame
  const done = (i: number) => f.t === -1 || (f.t < T && i >= f.t)
  const fmt = (v: number | string) => (typeof v === 'number' ? v.toFixed(2).replace(/^(-?)0\./, '$1.').replace('-', '−') : v)

  const a0 = useMemo(() => gae(gamma, 0).adv, [gamma])
  const a1 = useMemo(() => gae(gamma, 1).adv, [gamma])
  const x: Axis = { min: 0, max: T - 1, ticks: TOKS.map((_, i) => i), fmt: (v) => TOKS[v] ?? '' }
  const all = [...a0, ...a1, ...res.adv]
  const lo = Math.floor(Math.min(...all) * 4) / 4 - 0.05
  const hi = Math.ceil(Math.max(...all) * 4) / 4 + 0.05
  const y: Axis = { min: lo, max: hi, ticks: [lo, 0, hi].map((v) => Math.round(v * 100) / 100), fmt: (v) => v.toFixed(2), label: 'A_t' }

  const t = f.t
  let caption: React.ReactNode
  if (t === T) caption = <>从最后一个 token 开始往前算。奖励（−β·KL，末尾 +RM）与 critic 的 V(s_t) 已给出；γ = {gamma.toFixed(2)}、λ = {lam.toFixed(2)}。</>
  else if (t === -1)
    caption = (
      <>
        完成。returns = A + V 作为 critic 的回归目标。λ 越大，越依赖真实奖励（偏差小、方差大）；λ = 0 时只看一步 TD 误差（方差小，但完全信任 critic）。
      </>
    )
  else
    caption = (
      <>
        t = {t}：δ = r + γ·V(s_{t + 1}) − V(s_t) = {R[t].toFixed(3)} + {gamma.toFixed(2)}×{(t + 1 < T ? V[t + 1] : 0).toFixed(2)} − {V[t].toFixed(2)} ={' '}
        <b className="font-mono">{res.delta[t].toFixed(3)}</b>；A = δ + γλ·A_{t + 1} = <b className="font-mono">{res.adv[t].toFixed(3)}</b>。
      </>
    )

  const rowLabel = (s: string) => <div className="mb-0.5 font-mono text-[11.5px] text-muted">{s}</div>
  const cur = t >= 0 && t < T ? { [t]: 'accent' as const } : {}

  return (
    <VizFrame
      title="GAE：从后往前累积 TD 误差"
      caption={caption}
      legend={
        <Legend
          items={[
            { label: 'λ = 0（一步 TD）', tone: 'muted', variant: 'line' },
            { label: `λ = ${lam.toFixed(2)}（当前）`, tone: 'accent', variant: 'line' },
            { label: 'λ = 1（蒙特卡洛 − V）', tone: 'warning', variant: 'line' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Slider className="min-w-[150px] flex-1" label="折扣 γ" value={gamma} min={0.8} max={1} step={0.01} onChange={setGamma} format={(v) => v.toFixed(2)} />
          <Slider className="min-w-[150px] flex-1" label="λ" value={lam} min={0} max={1} step={0.01} onChange={setLam} format={(v) => v.toFixed(2)} />
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-[minmax(0,6fr)_minmax(0,5fr)]">
        <div className="flex min-w-0 flex-col gap-2">
          <div>
            {rowLabel('token')}
            <ArrayView values={TOKS} cellSize={42} showIndex={false} highlights={cur} />
          </div>
          <div>
            {rowLabel('rewards r_t')}
            <ArrayView values={R} cellSize={42} showIndex={false} format={fmt} highlights={{ ...cur, [T - 1]: t === T - 1 ? 'accent' : 'success' }} />
          </div>
          <div>
            {rowLabel('values V(s_t)')}
            <ArrayView values={V} cellSize={42} showIndex={false} format={fmt} highlights={t + 1 < T && t >= 0 ? { [t]: 'accent', [t + 1]: 'info' } : cur} />
          </div>
          <div>
            {rowLabel('δ_t')}
            <ArrayView values={res.delta.map((d, i) => (done(i) ? d : '·'))} cellSize={42} showIndex={false} format={fmt} highlights={cur} />
          </div>
          <div>
            {rowLabel(t === -1 ? 'A_t（returns = A_t + V）' : 'A_t')}
            <ArrayView values={res.adv.map((a, i) => (done(i) ? (t === -1 ? a : a) : '·'))} cellSize={42} showIndex={false} format={fmt} highlights={t + 1 < T && t >= 0 ? { [t]: 'accent', [t + 1]: 'warning' } : cur} />
          </div>
        </div>
        <div className="flex min-w-0 flex-col gap-3">
          <CodeTrace code={CODE} active={f.line} vars={{ t: t === T ? '—' : t === -1 ? 'done' : t, gamma: gamma.toFixed(2), lam: lam.toFixed(2) }} />
          <Svg>
            <Axes x={x} y={y} />
            <Line pts={a0.map((v, i) => [i, v])} x={x} y={y} tone="muted" dashed />
            <Line pts={a1.map((v, i) => [i, v])} x={x} y={y} tone="warning" />
            <Line pts={res.adv.map((v, i) => [i, v])} x={x} y={y} tone="accent" />
          </Svg>
        </div>
      </div>
    </VizFrame>
  )
}
