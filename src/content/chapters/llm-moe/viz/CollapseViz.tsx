/**
 * 专家坍缩的确定性玩具模拟（top-1 路由，8 个专家，256 个固定 token）。
 *  - 主任务代理：router 被推向「当前选中的专家」，且专家拿到的数据越多越强（富者愈富）；
 *  - 辅助损失：L_aux = α·N·Σ f_i P_i，对 logits 的梯度 (αN/T)·p_tj·(f_j − Σ_i f_i p_ti)（f 视为常数）。
 * 洞见：没有均衡项时负载迅速集中到 1–2 个专家；α 太小只能延缓坍缩。
 * 注：玩具模型里的 α 数值与真实模型不可直接类比。
 */
import { useMemo, useState } from 'react'
import { Legend, VizFrame, toneColor, type VizTone } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { normals, uniforms } from './moeMath'
import { Axes, Line, LoadBars, Stat, Svg, useScales, type Axis } from './plot'

const E = 8
const T = 256
const D = 4
const STEPS = 120
const X = (() => {
  const u = uniforms(31, T * D)
  return Array.from({ length: T }, (_, t) => u.slice(t * D, t * D + D))
})()
const W0 = (() => {
  const z = normals(37, D * E).map((v) => v * 0.3)
  return Array.from({ length: D }, (_, i) => z.slice(i * E, i * E + E))
})()

function simulate(alpha: number) {
  const beta = 0.03
  const lr = 4
  const etaQ = 0.03
  const W = W0.map((r) => r.slice())
  const c = new Array(E).fill(0)
  let q = new Array(E).fill(1)
  const out: { share: number[]; vio: number; alive: number; aux: number }[] = []
  const p = Array.from({ length: T }, () => new Array(E).fill(0))
  for (let s = 0; s < STEPS; s++) {
    const cnt = new Array(E).fill(0)
    const P = new Array(E).fill(0)
    const arg = new Array(T).fill(0)
    for (let t = 0; t < T; t++) {
      const z = new Array(E)
      let m = -Infinity
      for (let e = 0; e < E; e++) {
        let v = c[e]
        for (let i = 0; i < D; i++) v += X[t][i] * W[i][e]
        z[e] = v
        if (v > m) m = v
      }
      let sum = 0
      for (let e = 0; e < E; e++) sum += (z[e] = Math.exp(z[e] - m))
      let best = 0
      for (let e = 0; e < E; e++) {
        p[t][e] = z[e] / sum
        P[e] += p[t][e] / T
        if (p[t][e] > p[t][best]) best = e
      }
      arg[t] = best
      cnt[best]++
    }
    const f = cnt.map((v) => v / T)
    out.push({
      share: f,
      vio: Math.max(...f) * E - 1,
      alive: cnt.filter((v) => v > 0).length,
      aux: alpha * E * f.reduce((a, v, i) => a + v * P[i], 0),
    })
    // 梯度：对 logits 的梯度 g (T×E)，再链式到 W、c
    const gW = W.map((r) => r.map(() => 0))
    const gc = new Array(E).fill(0)
    for (let t = 0; t < T; t++) {
      const e0 = arg[t]
      const pe = p[t][e0]
      const fp = f.reduce((a, v, i) => a + v * p[t][i], 0)
      for (let j = 0; j < E; j++) {
        const lm = (-beta * q[e0] * pe * ((j === e0 ? 1 : 0) - p[t][j])) / T
        const ax = ((alpha * E) / T) * p[t][j] * (f[j] - fp)
        const g = lm + ax
        gc[j] += g
        for (let i = 0; i < D; i++) gW[i][j] += X[t][i] * g
      }
    }
    for (let j = 0; j < E; j++) {
      c[j] -= lr * gc[j]
      for (let i = 0; i < D; i++) W[i][j] -= lr * gW[i][j]
    }
    // 拿到更多数据的专家变得更强（富者愈富）
    q = q.map((v, e) => Math.min(3, Math.max(0.2, v + etaQ * (f[e] * E - 1))))
  }
  return out
}

const ALPHAS = [0, 0.01, 0.03, 0.1]

export function CollapseViz() {
  const [alpha, setAlpha] = useState(0)
  const [step, setStep] = useState(STEPS - 1)
  const sim = useMemo(() => simulate(alpha), [alpha])
  const base = useMemo(() => simulate(0), [])
  const cur = sim[step]
  const x: Axis = { min: 0, max: STEPS - 1, ticks: [0, 30, 60, 90], label: '训练步' }
  const y: Axis = { min: 0, max: 7, ticks: [0, 1, 2, 3, 4, 5, 6, 7], label: 'MaxVio' }
  const { sx, sy } = useScales(x, y)
  const tone = (v: number): VizTone => (v === 0 ? 'muted' : v > 2 / E ? 'danger' : 'accent')

  return (
    <VizFrame
      title="专家坍缩：富者愈富的正反馈"
      caption={
        <>
          第 {step} 步：最忙的专家拿到 <b className="font-mono">{(Math.max(...cur.share) * 100).toFixed(0)}%</b> 的 token（均匀时应为 12.5%），
          仍有 token 的专家 <b>{cur.alive}</b> / {E} 个。
          {alpha === 0
            ? ' 没有均衡项：被选中的专家得到更多训练、变得更强，router 于是更偏爱它——几十步后几乎所有 token 都挤进一两个专家，其余专家「饿死」。'
            : alpha < 0.03
              ? ' α 太小：辅助损失只能延缓坍缩，压不住正反馈。'
              : ' α 足够：辅助损失持续把过载专家的路由概率往下压，负载保持接近均匀。'}
        </>
      }
      legend={
        <Legend
          items={[
            { label: `当前 α=${alpha}`, tone: 'accent', variant: 'line' },
            { label: '无辅助损失（α=0）', tone: 'muted', variant: 'line' },
            { label: '红柱 = 超过 2 倍均值', tone: 'danger' },
          ]}
        />
      }
      controls={
        <>
          <div className="flex items-center gap-2 text-[13px] text-muted">
            aux loss 系数 α
            <Segmented size="sm" value={alpha} onChange={setAlpha} options={ALPHAS.map((a) => ({ value: a, label: a === 0 ? '关' : String(a) }))} />
          </div>
          <Slider className="min-w-[180px] flex-1" label="查看第几步" value={step} min={0} max={STEPS - 1} onChange={setStep} />
        </>
      }
    >
      <div className="grid items-start gap-5 @3xl:grid-cols-2">
        <Svg>
          <Axes x={x} y={y} />
          <Line pts={base.map((f, i) => [i, f.vio])} x={x} y={y} tone="muted" dashed width={1.6} />
          <Line pts={sim.map((f, i) => [i, f.vio])} x={x} y={y} tone="accent" />
          <circle cx={sx(step)} cy={sy(cur.vio)} r={5.5} fill={toneColor('gold')} stroke="var(--surface)" strokeWidth={2} />
        </Svg>
        <div className="min-w-0 space-y-3">
          <div className="text-xs font-medium text-muted">第 {step} 步各专家拿到的 token 比例 f_i</div>
          <LoadBars
            values={cur.share.map((v) => v * 100)}
            labels={cur.share.map((_, e) => `E${e}`)}
            tones={cur.share.map(tone)}
            refValue={100 / E}
            refLabel="均匀 12.5%"
            max={100}
            fmt={(v) => `${v.toFixed(0)}%`}
          />
          <div className="grid grid-cols-3 gap-2">
            <Stat label="存活专家" value={`${cur.alive}/${E}`} tone={cur.alive < E ? 'danger' : 'success'} />
            <Stat label="MaxVio" value={cur.vio.toFixed(2)} />
            <Stat label="L_aux" value={cur.aux.toFixed(4)} />
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
