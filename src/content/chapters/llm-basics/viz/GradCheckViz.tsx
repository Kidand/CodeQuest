/**
 * 数值梯度的误差随步长 h 的变化：截断误差（h 太大）与舍入误差（h 太小）之间存在最优点。
 *   前向差分 (f(x+h) − f(x)) / h：截断误差 O(h)
 *   中心差分 (f(x+h) − f(x−h)) / 2h：截断误差 O(h²)
 *   舍入误差 ≈ ε_machine · |f| / h  →  本例 float64 最优 h ≈ 6e-6、误差约 1e-11；float32 最优 h ≈ 6e-3、误差约 1e-5
 * 测试函数 f(x) = sin(x)·eˣ，在 x = 1 处，解析导数 (cos x + sin x)·eˣ。float32 用 Math.fround 逐步模拟。
 */
import { useMemo, useState } from 'react'
import { Legend, VizFrame } from '@/viz'
import { Slider } from '@/components/ui'
import { Axes, Dot, Line, Plot, Stat, VLine, type Axis } from './plot'

const X0 = 1
const f64 = (x: number) => Math.sin(x) * Math.exp(x)
const fr = Math.fround
const f32 = (x: number) => fr(fr(Math.sin(x)) * fr(Math.exp(x)))
const EXACT = (Math.cos(X0) + Math.sin(X0)) * Math.exp(X0)

function relErr(num: number) {
  const e = Math.abs(num - EXACT) / Math.abs(EXACT)
  return Math.max(1e-16, e)
}
const fwd64 = (h: number) => relErr((f64(X0 + h) - f64(X0)) / h)
const cen64 = (h: number) => relErr((f64(X0 + h) - f64(X0 - h)) / (2 * h))
const cen32 = (h: number) => {
  const x = fr(X0)
  const hh = fr(h)
  const num = fr(fr(f32(fr(x + hh)) - f32(fr(x - hh))) / fr(2 * hh))
  return relErr(num)
}

export function GradCheckViz() {
  const [logH, setLogH] = useState(-5)
  const h = 10 ** logH
  const curves = useMemo(() => {
    const mk = (fn: (h: number) => number) => {
      const out: [number, number][] = []
      for (let i = 0; i <= 220; i++) {
        const hh = 10 ** (-12 + (11 * i) / 220)
        out.push([hh, fn(hh)])
      }
      return out
    }
    return { f: mk(fwd64), c: mk(cen64), c32: mk(cen32) }
  }, [])
  /** 舍入误差是“噪声”，个别 h 会碰巧很准；取 ±0.25 个数量级窗口内的最大值作为包络，再找包络的最低点 */
  const best = (arr: [number, number][]) => {
    const env = arr.map((p, i) => [p[0], Math.max(...arr.slice(Math.max(0, i - 5), i + 6).map((q) => q[1]))] as [number, number])
    return env.reduce((a, b) => (b[1] < a[1] ? b : a))
  }
  const bc = best(curves.c)
  const bc32 = best(curves.c32)
  const x: Axis = { min: 1e-12, max: 1e-1, log: true, ticks: [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2], label: '步长 h', fmt: (v) => `1e${Math.round(Math.log10(v))}` }
  const y: Axis = { min: 1e-12, max: 10, log: true, ticks: [1e-12, 1e-9, 1e-6, 1e-3, 1], label: '相对误差', fmt: (v) => (v === 1 ? '1' : `1e${Math.round(Math.log10(v))}`) }

  return (
    <VizFrame
      title="数值梯度：h 不是越小越好，float32 根本不够用"
      caption={
        <>
          h = <b className="font-mono">{h.toExponential(0)}</b>：中心差分（float64）相对误差 {cen64(h).toExponential(1)}，前向差分 {fwd64(h).toExponential(1)}，
          中心差分（float32）{cen32(h).toExponential(1)}。h 大时误差由截断主导（中心差分斜率 2、前向差分斜率 1），h 小时由舍入误差 ε/h 主导。
          float64 中心差分稳定地能到约 {bc[1].toExponential(0)}（h ≈ {bc[0].toExponential(0)}）；float32 最好只有约 {bc32[1].toExponential(0)}，而且要把 h 放大到 {bc32[0].toExponential(0)} 附近——真实网络的 loss 是大量项的和，舍入误差更大，所以梯度检查必须用 float64。
        </>
      }
      legend={
        <Legend
          items={[
            { label: '中心差分 · float64', tone: 'accent', variant: 'line' },
            { label: '前向差分 · float64', tone: 'info', variant: 'line' },
            { label: '中心差分 · float32', tone: 'danger', variant: 'line' },
          ]}
        />
      }
      controls={<Slider className="min-w-[200px] flex-1" label="步长 h（对数）" value={logH} min={-12} max={-1} step={0.1} onChange={setLogH} format={() => h.toExponential(1)} />}
    >
      <div className="grid items-center gap-4 @3xl:grid-cols-[1fr_200px]">
        <Plot height={270}>
          {(box) => (
            <>
              <Axes x={x} y={y} box={box} />
              <VLine x={x} y={y} box={box} at={h} />
              <Line pts={curves.f} x={x} y={y} box={box} tone="info" width={1.8} />
              <Line pts={curves.c32} x={x} y={y} box={box} tone="danger" width={1.8} />
              <Line pts={curves.c} x={x} y={y} box={box} tone="accent" />
              <Dot x={x} y={y} box={box} at={[h, cen64(h)]} tone="accent" />
              <Dot x={x} y={y} box={box} at={[h, cen32(h)]} tone="danger" />
            </>
          )}
        </Plot>
        <div className="grid grid-cols-2 gap-2 @3xl:grid-cols-1">
          <Stat label="中心 · f64" value={cen64(h).toExponential(1)} tone="accent" />
          <Stat label="前向 · f64" value={fwd64(h).toExponential(1)} tone="info" />
          <Stat label="中心 · f32" value={cen32(h).toExponential(1)} tone="danger" />
          <Stat label="f(x)=sin(x)·eˣ" value={`f′(1)=${EXACT.toFixed(4)}`} />
        </div>
      </div>
    </VizFrame>
  )
}
