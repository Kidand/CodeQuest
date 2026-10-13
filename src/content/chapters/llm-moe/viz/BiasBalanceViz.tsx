/**
 * 无辅助损失均衡的确定性小模拟：8 个专家（天然冷热不均）、每步一个新 batch（512 token）、top-2。
 * 第 t 步用「上一步更新后的」bias 路由，统计负载，再按 b ← b + γ·sign(均值 − 负载) 更新。
 * 洞见：γ 太小收敛慢，太大来回震荡；bias 从不进入门控权重。
 */
import { useMemo, useState } from 'react'
import { Legend, MatrixView, VizFrame, toneColor } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { normals } from './moeMath'
import { Axes, Line, LoadBars, Stat, Svg, useScales, type Axis } from './plot'

const E = 8
const K = 2
const T = 512
const STEPS = 150
const POP = [0.9, 0.6, 0.3, 0.1, -0.1, -0.3, -0.6, -0.9] // 各专家的「天然热度」（logit 偏移）
const MEAN = (T * K) / E

let NOISE: Float32Array | null = null
function scoresAt(t: number): Float32Array {
  if (!NOISE) {
    NOISE = new Float32Array(STEPS * T * E)
    for (let s = 0; s < STEPS; s++) {
      const z = normals(4000 + s, T * E)
      for (let i = 0; i < T * E; i++) NOISE[s * T * E + i] = 1 / (1 + Math.exp(-(POP[i % E] + z[i])))
    }
  }
  return NOISE.subarray(t * T * E, (t + 1) * T * E)
}

type Rule = 'sign' | 'prop'
function simulate(gamma: number, rule: Rule) {
  const b = new Array(E).fill(0)
  const out: { load: number[]; bias: number[]; vio: number }[] = []
  for (let t = 0; t < STEPS; t++) {
    const s = scoresAt(t)
    const load = new Array(E).fill(0)
    for (let i = 0; i < T; i++) {
      // top-2 of s + b
      let a = -1, av = -Infinity, c = -1, cv = -Infinity
      for (let e = 0; e < E; e++) {
        const v = s[i * E + e] + b[e]
        if (v > av) { c = a; cv = av; a = e; av = v } else if (v > cv) { c = e; cv = v }
      }
      load[a]++
      load[c]++
    }
    out.push({ load, bias: b.slice(), vio: (Math.max(...load) - MEAN) / MEAN })
    for (let e = 0; e < E; e++) {
      const err = MEAN - load[e]
      b[e] += rule === 'sign' ? gamma * Math.sign(err) : (gamma * err) / MEAN
    }
  }
  return out
}

const GAMMAS = [0.001, 0.003, 0.01, 0.03, 0.1]

export function BiasBalanceViz() {
  const [gamma, setGamma] = useState(0.01)
  const [rule, setRule] = useState<Rule>('sign')
  const [step, setStep] = useState(60)
  const sim = useMemo(() => simulate(gamma, rule), [gamma, rule])
  const base = useMemo(() => simulate(0, 'sign'), [])
  const cur = sim[step]
  const tail = sim.slice(-40).reduce((a, x) => a + x.vio, 0) / 40

  const x: Axis = { min: 0, max: STEPS - 1, ticks: [0, 30, 60, 90, 120], label: '训练步' }
  const y: Axis = { min: 0, max: 1.6, ticks: [0, 0.4, 0.8, 1.2, 1.6], label: 'MaxVio', fmt: (v) => v.toFixed(1) }
  const { sx, sy } = useScales(x, y)

  const verdict =
    gamma <= 0.001
      ? '步长太小：150 步内还没追上冷热差距（V3 用 0.001，但它的训练步数以十万计）。'
      : gamma >= 0.1 && rule === 'sign'
        ? '步长太大：每步 bias 都跨过平衡点，负载来回震荡，MaxVio 降不下去。'
        : '步长合适：几十步内把热门专家「压」下去，之后在噪声水平附近小幅波动。'

  return (
    <VizFrame
      title="偏置均衡：每步按负载微调 bias"
      caption={
        <>
          第 {step} 步：MaxVio = <b className="font-mono">{cur.vio.toFixed(2)}</b>（γ=0 时约 {base[step].vio.toFixed(2)}），最后 40 步平均{' '}
          <b className="font-mono">{tail.toFixed(2)}</b>。{verdict}
          {rule === 'prop' && ' 比例更新（按误差大小调整）更平滑——Loss-Free Balancing 论文中它的均衡度略好，但效果没有更好，最终仍用 sign 版本。'}
        </>
      }
      legend={
        <Legend
          items={[
            { label: `当前 γ=${gamma}`, tone: 'accent', variant: 'line' },
            { label: '不做均衡（γ=0）', tone: 'muted', variant: 'line' },
            { label: '红虚线 = 均值负载', tone: 'danger', variant: 'line' },
          ]}
        />
      }
      controls={
        <>
          <div className="flex items-center gap-2 text-[13px] text-muted">
            γ
            <Segmented size="sm" value={gamma} onChange={setGamma} options={GAMMAS.map((g) => ({ value: g, label: String(g) }))} />
          </div>
          <Segmented
            size="sm"
            value={rule}
            onChange={setRule}
            options={[
              { value: 'sign', label: 'γ·sign(误差)' },
              { value: 'prop', label: 'γ·误差/均值' },
            ]}
          />
          <Slider className="min-w-[180px] flex-1" label="查看第几步" value={step} min={0} max={STEPS - 1} onChange={setStep} />
        </>
      }
    >
      <div className="grid items-start gap-5 @3xl:grid-cols-2">
        <Svg>
          <Axes x={x} y={y} />
          <Line pts={base.map((f, i) => [i, f.vio])} x={x} y={y} tone="muted" dashed width={1.6} />
          <Line pts={sim.map((f, i) => [i, f.vio])} x={x} y={y} tone="accent" />
          <circle cx={sx(step)} cy={sy(Math.min(1.6, cur.vio))} r={5.5} fill={toneColor('gold')} stroke="var(--surface)" strokeWidth={2} />
        </Svg>
        <div className="min-w-0 space-y-3">
          <div className="text-xs font-medium text-muted">第 {step} 步各专家负载（E0 天然最热，E7 最冷）</div>
          <LoadBars values={cur.load} labels={cur.load.map((_, e) => `E${e}`)} tones={cur.load.map((v) => (v > MEAN * 1.2 ? 'danger' : v < MEAN * 0.8 ? 'info' : 'success'))} refValue={MEAN} refLabel={`均值 ${MEAN}`} max={300} />
          <div className="text-xs font-medium text-muted">此时的 bias（只加在选择分数上）</div>
          <MatrixView data={[cur.bias]} colLabels={cur.bias.map((_, e) => `E${e}`)} scale="div" domain={[-0.5, 0.5]} cellSize={46} format={(v) => (Math.abs(v) < 5e-4 ? '0' : v.toFixed(2))} />
          <div className="grid grid-cols-2 gap-2">
            <Stat label="最大负载 / 均值" value={(Math.max(...cur.load) / MEAN).toFixed(2)} tone={cur.vio > 0.3 ? 'danger' : 'success'} />
            <Stat label="bias 范围" value={`${Math.min(...cur.bias).toFixed(2)} ~ ${Math.max(...cur.bias).toFixed(2)}`} />
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
