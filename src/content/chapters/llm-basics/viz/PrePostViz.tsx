/**
 * Pre-LN vs Post-LN：用随机线性子层（d = 64，W ~ N(0, 1/d)，固定种子）实际跑一遍前向。
 *  - Pre-LN：x ← x + f(LN(x))，每层加入一个范数约 √d 的近似独立向量 → 残差流 RMS ≈ √(1 + l)
 *  - Post-LN：x ← LN(x + f(x))，每层都被拉回 RMS = 1
 * 洞见：Pre-LN 的恒等路径让训练稳定，但越深的层“改动占比”越小（≈ 1/√l）。
 */
import { useMemo, useState } from 'react'
import { Legend, VizFrame } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { rng } from '@/lib/utils'
import { Axes, Dot, Line, Plot, Stat, type Axis } from './plot'

const D = 64
const LMAX = 64

function gauss(r: () => number) {
  const u = Math.max(1e-12, r())
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r())
}

const WS: Float64Array[] = (() => {
  const r = rng(2024)
  return Array.from({ length: LMAX }, () => {
    const w = new Float64Array(D * D)
    for (let i = 0; i < w.length; i++) w[i] = gauss(r) / Math.sqrt(D)
    return w
  })
})()
const X0 = (() => {
  const r = rng(7)
  return Array.from({ length: D }, () => gauss(r))
})()

const matvec = (w: Float64Array, h: number[]) => {
  const out = new Array(D).fill(0)
  for (let i = 0; i < D; i++) {
    let s = 0
    for (let j = 0; j < D; j++) s += w[i * D + j] * h[j]
    out[i] = s
  }
  return out
}
const rmsOf = (v: number[]) => Math.sqrt(v.reduce((a, b) => a + b * b, 0) / v.length)
const ln = (v: number[]) => {
  const m = v.reduce((a, b) => a + b, 0) / v.length
  const s = Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length + 1e-5)
  return v.map((x) => (x - m) / s)
}

function simulate(L: number) {
  const pre: { rms: number; ratio: number }[] = []
  const post: { rms: number; ratio: number }[] = []
  let a = ln(X0)
  let b = ln(X0)
  pre.push({ rms: rmsOf(a), ratio: NaN })
  post.push({ rms: rmsOf(b), ratio: NaN })
  for (let l = 0; l < L; l++) {
    const fa = matvec(WS[l], ln(a))
    const ratioA = rmsOf(fa) / rmsOf(a)
    a = a.map((v, i) => v + fa[i])
    pre.push({ rms: rmsOf(a), ratio: ratioA })
    const fb = matvec(WS[l], b)
    const ratioB = rmsOf(fb) / rmsOf(b)
    b = ln(b.map((v, i) => v + fb[i]))
    post.push({ rms: rmsOf(b), ratio: ratioB })
  }
  return { pre, post }
}

type Metric = 'rms' | 'ratio'

export function PrePostViz() {
  const [L, setL] = useState(32)
  const [metric, setMetric] = useState<Metric>('rms')
  const sim = useMemo(() => simulate(L), [L])
  const x: Axis = { min: 0, max: L, ticks: Array.from({ length: 5 }, (_, i) => Math.round((i * L) / 4)), label: '层号 l' }
  const ymax = metric === 'rms' ? Math.max(2, Math.ceil(Math.sqrt(L + 1) + 1)) : 1.6
  const y: Axis =
    metric === 'rms'
      ? { min: 0, max: ymax, ticks: Array.from({ length: ymax + 1 }, (_, i) => i).filter((v) => ymax <= 6 || v % 2 === 0), label: '残差流 RMS' }
      : { min: 0, max: 1.6, ticks: [0, 0.4, 0.8, 1.2, 1.6], label: '本层更新 / 残差流', fmt: (v) => v.toFixed(1) }
  const pick = (arr: { rms: number; ratio: number }[]) => arr.map((p, i) => [i, metric === 'rms' ? p.rms : p.ratio] as [number, number])
  const last = sim.pre[L]

  return (
    <VizFrame
      title="Pre-LN：残差流越来越“胖”，深层的改动占比越来越小"
      caption={
        metric === 'rms' ? (
          <>
            {L} 层后，Pre-LN 残差流的 RMS 涨到 <b className="font-mono">{last.rms.toFixed(2)}</b>（虚线 √(1+l) = {Math.sqrt(L + 1).toFixed(2)}）；Post-LN 每层都被归一化回 1。
            Pre-LN 每层只往残差流里“加东西”，从不缩放，所以范数随深度近似按 √l 增长。
          </>
        ) : (
          <>
            第 {L} 层的更新只占残差流的 <b className="font-mono">{(last.ratio * 100).toFixed(0)}%</b>（≈ 1/√l）。这意味着 Pre-LN 的深层对输出的影响天然变小——
            一些工作把“深层冗余、剪掉也不太掉点”归因于此。Post-LN 每层的改动占比恒定，但梯度要穿过每一个 LN，深了就难训练。
          </>
        )
      }
      legend={
        <Legend
          items={[
            { label: 'Pre-LN', tone: 'accent', variant: 'line' },
            { label: 'Post-LN', tone: 'warning', variant: 'line' },
            ...(metric === 'rms' ? [{ label: '√(1 + l)', tone: 'muted' as const, variant: 'line' as const }] : [{ label: '1/√(1 + l)', tone: 'muted' as const, variant: 'line' as const }]),
          ]}
        />
      }
      controls={
        <>
          <Segmented
            size="sm"
            value={metric}
            onChange={setMetric}
            options={[
              { value: 'rms', label: '残差流 RMS' },
              { value: 'ratio', label: '每层更新占比' },
            ]}
          />
          <Slider className="min-w-[180px] flex-1" label="层数 L" value={L} min={4} max={LMAX} step={4} onChange={setL} />
        </>
      }
    >
      <div className="grid items-center gap-4 @3xl:grid-cols-[1fr_200px]">
        <Plot height={250}>
          {(box) => (
            <>
              <Axes x={x} y={y} box={box} />
              <Line pts={Array.from({ length: L + 1 }, (_, i) => [i, metric === 'rms' ? Math.sqrt(1 + i) : 1 / Math.sqrt(1 + i)] as [number, number]).slice(metric === 'rms' ? 0 : 1)} x={x} y={y} box={box} tone="muted" dashed width={1.5} />
              <Line pts={pick(sim.post)} x={x} y={y} box={box} tone="warning" />
              <Line pts={pick(sim.pre)} x={x} y={y} box={box} tone="accent" />
              <Dot at={[L, metric === 'rms' ? last.rms : last.ratio]} x={x} y={y} box={box} tone="accent" />
            </>
          )}
        </Plot>
        <div className="grid grid-cols-2 gap-2 @3xl:grid-cols-1">
          <Stat label={`Pre-LN 第 ${L} 层 RMS`} value={last.rms.toFixed(2)} tone="accent" />
          <Stat label={`Pre-LN 第 ${L} 层更新占比`} value={`${(last.ratio * 100).toFixed(0)}%`} tone="accent" />
          <Stat label="Post-LN RMS" value={sim.post[L].rms.toFixed(2)} tone="warning" />
          <Stat label="子层" value="f(h) = W h" sub="W ~ N(0, 1/d)，d = 64" />
        </div>
      </div>
    </VizFrame>
  )
}
