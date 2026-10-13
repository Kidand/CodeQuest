/**
 * AdamW vs Adam + L2：两个参数的梯度都是零均值噪声（没有任何“有用信号”），只是噪声尺度不同。
 * 理想的权重衰减应让两者以相同速率衰减。
 *  - Adam + L2：衰减项 λθ 混进梯度，再被 √v̂ 除——梯度大的参数几乎不衰减，梯度小的参数被猛烈衰减（每步约 lr）
 *  - AdamW：θ ← θ − lr·λ·θ 与自适应更新解耦，两者都按 (1 − lr·λ)^t 衰减
 * 梯度取符号交替的 ±σ（均值为 0），两种优化器使用完全相同的梯度序列。
 */
import { useMemo, useState } from 'react'
import { Legend, VizFrame } from '@/viz'
import { Slider } from '@/components/ui'
import { Axes, Line, Plot, Stat, type Axis } from './plot'

const T = 400
const LR = 0.01
const B1 = 0.9
const B2 = 0.999
const EPS = 1e-8

/** 符号来回翻转的零均值“纯噪声”梯度（不含任何有用信号）。用确定性的交替符号，避免随机游走掩盖衰减效应 */
const NOISE = [Array.from({ length: T }, (_, i) => (i % 2 ? 1 : -1)), Array.from({ length: T }, (_, i) => (i % 2 ? -1 : 1))]

function run(sigma: number, noise: number[], wd: number, decoupled: boolean) {
  let th = 1
  let m = 0
  let v = 0
  const out: [number, number][] = [[0, th]]
  for (let t = 1; t <= T; t++) {
    let g = sigma * noise[t - 1]
    if (!decoupled) g += wd * th // L2：衰减项进入梯度
    else th *= 1 - LR * wd // AdamW：解耦的权重衰减
    m = B1 * m + (1 - B1) * g
    v = B2 * v + (1 - B2) * g * g
    const mh = m / (1 - B1 ** t)
    const vh = v / (1 - B2 ** t)
    th -= (LR * mh) / (Math.sqrt(vh) + EPS)
    out.push([t, th])
  }
  return out
}

export function DecayCompareViz() {
  const [logA, setLogA] = useState(0.5) // σ_A = 10^logA
  const [logB, setLogB] = useState(-2)
  const [wd, setWd] = useState(0.1)
  const sA = 10 ** logA
  const sB = 10 ** logB
  const res = useMemo(
    () => ({
      aL2: run(sA, NOISE[0], wd, false),
      bL2: run(sB, NOISE[1], wd, false),
      aW: run(sA, NOISE[0], wd, true),
      bW: run(sB, NOISE[1], wd, true),
    }),
    [sA, sB, wd],
  )
  const ideal = (1 - LR * wd) ** T
  const x: Axis = { min: 0, max: T, ticks: [0, 100, 200, 300, 400], label: '步数' }
  const y: Axis = { min: -0.2, max: 1.2, ticks: [0, 0.25, 0.5, 0.75, 1], label: '参数 θ（初值 1）', fmt: (v) => v.toFixed(2) }
  const fin = (s: [number, number][]) => s[s.length - 1][1]

  return (
    <VizFrame
      title="Adam + L2 ≠ AdamW：L2 的衰减被 √v̂ 除掉了"
      caption={
        <>
          两个参数的梯度都是符号来回翻转的零均值噪声（A 的幅度 {sA.toFixed(2)}，B 的幅度 {sB.toFixed(3)}），理想的衰减应当一视同仁：(1 − lr·λ)^{T} = {ideal.toFixed(3)}。
          <b> Adam + L2</b>：A 最终 {fin(res.aL2).toFixed(3)}、B 最终 {fin(res.bL2).toFixed(3)}——梯度大的 A 几乎不衰减，梯度小的 B 每步被推动约 lr，
          很快冲到 0 附近。<b>AdamW</b>：A {fin(res.aW).toFixed(3)}、B {fin(res.bW).toFixed(3)}，都贴着理想曲线。
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'A · Adam+L2', tone: 1, variant: 'line' },
            { label: 'B · Adam+L2', tone: 3, variant: 'line' },
            { label: 'A · AdamW（虚线）', tone: 1, variant: 'soft' },
            { label: 'B · AdamW（虚线）', tone: 3, variant: 'soft' },
            { label: '理想 (1−lr·λ)^t', tone: 'muted', variant: 'line' },
          ]}
        />
      }
      controls={
        <>
          <Slider className="min-w-[150px] flex-1" label="A 的梯度尺度" value={logA} min={-1} max={1} step={0.05} onChange={setLogA} format={() => sA.toFixed(2)} />
          <Slider className="min-w-[150px] flex-1" label="B 的梯度尺度" value={logB} min={-3} max={0} step={0.05} onChange={setLogB} format={() => sB.toFixed(3)} />
          <Slider className="min-w-[150px] flex-1" label="衰减系数 λ" value={wd} min={0.02} max={0.3} step={0.01} onChange={setWd} format={(v) => v.toFixed(2)} />
        </>
      }
    >
      <div className="grid items-center gap-4 @3xl:grid-cols-[1fr_200px]">
        <Plot height={260}>
          {(box) => (
            <>
              <Axes x={x} y={y} box={box} />
              <Line pts={Array.from({ length: T + 1 }, (_, t) => [t, (1 - LR * wd) ** t] as [number, number])} x={x} y={y} box={box} tone="muted" width={3} opacity={0.5} />
              <Line pts={res.aW} x={x} y={y} box={box} tone={1} dashed width={1.8} />
              <Line pts={res.bW} x={x} y={y} box={box} tone={3} dashed width={1.8} />
              <Line pts={res.aL2} x={x} y={y} box={box} tone={1} />
              <Line pts={res.bL2} x={x} y={y} box={box} tone={3} />
            </>
          )}
        </Plot>
        <div className="grid grid-cols-2 gap-2 @3xl:grid-cols-1">
          <Stat label="A · Adam+L2" value={fin(res.aL2).toFixed(3)} tone={1} />
          <Stat label="B · Adam+L2" value={fin(res.bL2).toFixed(3)} tone={3} />
          <Stat label="A / B · AdamW" value={`${fin(res.aW).toFixed(2)} / ${fin(res.bW).toFixed(2)}`} />
          <Stat label="设置" value={`lr=${LR}`} sub="β=(0.9, 0.999)，400 步" />
        </div>
      </div>
    </VizFrame>
  )
}
