/**
 * Adam 的偏差修正：m、v 从 0 开始做 EMA，早期被严重低估。
 * 对恒定梯度 g：m_t = (1−β1^t)·g，v_t = (1−β2^t)·g²。
 * 不做修正时，实际步长 / 目标步长 = (1−β1^t) / √(1−β2^t)——β1 = 0.9、β2 = 0.999 时在第 12 步附近达到约 6.6 倍，
 * 到第 1000 步仍有约 1.26 倍。
 */
import { useMemo, useState } from 'react'
import { Legend, VizFrame } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { Axes, Dot, HBand, Line, Plot, Stat, VLine, type Axis } from './plot'

const B2S = [0.95, 0.99, 0.999, 0.9999] as const

export function BiasCorrectionViz() {
  const [b1, setB1] = useState(0.9)
  const [b2, setB2] = useState<(typeof B2S)[number]>(0.999)
  const [logT, setLogT] = useState(0) // t = 10^logT
  const t = Math.max(1, Math.round(10 ** logT))
  const T_MAX = 20000
  const x: Axis = { min: 1, max: T_MAX, log: true, ticks: [1, 10, 100, 1000, 10000], label: '步数 t（对数刻度）', fmt: (v) => (v >= 1000 ? `${v / 1000}k` : String(v)) }

  const curves = useMemo(() => {
    const pts = (f: (t: number) => number) => {
      const out: [number, number][] = []
      for (let i = 0; i <= 240; i++) {
        const tt = 10 ** ((i / 240) * Math.log10(T_MAX))
        out.push([tt, f(tt)])
      }
      return out
    }
    return {
      m: pts((tt) => 1 - b1 ** tt),
      v: pts((tt) => Math.sqrt(1 - b2 ** tt)),
      ratio: pts((tt) => (1 - b1 ** tt) / Math.sqrt(1 - b2 ** tt)),
    }
  }, [b1, b2])
  const ratioAll = curves.ratio.map((p) => p[1])
  const peak = Math.max(...ratioAll)
  const yMax = Math.max(1.5, Math.ceil(peak * 2) / 2 + 0.5)
  const y: Axis = { min: 0, max: yMax, ticks: Array.from({ length: Math.floor(yMax / 0.5) + 1 }, (_, i) => i * 0.5).filter((v, _, a) => a.length <= 9 || v % 1 === 0), fmt: (v) => v.toFixed(1), label: '相对 1 的倍数' }
  const mt = 1 - b1 ** t
  const vt = Math.sqrt(1 - b2 ** t)
  const r = mt / vt

  return (
    <VizFrame
      title="不做偏差修正，Adam 的早期步长会被放大或缩小"
      caption={
        <>
          第 t = <b className="font-mono">{t}</b> 步（恒定梯度 g）：m_t = {mt.toFixed(3)}·g，√v_t = {vt.toFixed(3)}·|g|，
          未修正的步长是目标步长 lr 的 <b className="font-mono">{r.toFixed(2)}</b> 倍。
          {r > 1.05
            ? ' 此刻步长被放大：v 的“记忆”（β2）比 m 长，被低估得更严重。'
            : r < 0.95
              ? ' 此刻步长被缩小：m 还没“热起来”。'
              : ' 此刻两者的低估程度接近，几乎不受影响。'}
          {` 整条曲线的峰值约 ${peak.toFixed(2)} 倍——β2 越接近 1，放大越严重、持续越久，训练初期最容易因此发散。`}
          偏差修正 m̂ = m/(1−β1^t)、v̂ = v/(1−β2^t) 后，比值恒为 1。
        </>
      }
      legend={
        <Legend
          items={[
            { label: '(1−β1^t)：m 的缩水比例', tone: 'info', variant: 'line' },
            { label: '√(1−β2^t)：√v 的缩水比例', tone: 'warning', variant: 'line' },
            { label: '未修正步长 / lr', tone: 'danger', variant: 'line' },
            { label: '修正后 = 1', tone: 'success', variant: 'line' },
          ]}
        />
      }
      controls={
        <>
          <Slider className="min-w-[150px] flex-1" label="β1" value={b1} min={0.5} max={0.99} step={0.01} onChange={setB1} format={(v) => v.toFixed(2)} />
          <div className="flex flex-col gap-1.5">
            <span className="text-[13px] text-muted">β2</span>
            <Segmented size="sm" value={b2} onChange={setB2} options={B2S.map((v) => ({ value: v, label: String(v) }))} />
          </div>
          <Slider className="min-w-[170px] flex-1" label="查看第 t 步" value={logT} min={0} max={Math.log10(T_MAX)} step={0.01} onChange={setLogT} format={() => String(t)} />
        </>
      }
    >
      <div className="grid items-center gap-4 @3xl:grid-cols-[1fr_200px]">
        <Plot height={260}>
          {(box) => (
            <>
              <HBand x={x} y={y} box={box} from={0.98} to={1.02} tone="success" />
              <Axes x={x} y={y} box={box} />
              <Line pts={[[1, 1], [T_MAX, 1]]} x={x} y={y} box={box} tone="success" width={2} />
              <Line pts={curves.m} x={x} y={y} box={box} tone="info" width={1.8} dashed />
              <Line pts={curves.v} x={x} y={y} box={box} tone="warning" width={1.8} dashed />
              <Line pts={curves.ratio} x={x} y={y} box={box} tone="danger" />
              <VLine x={x} y={y} box={box} at={t} />
              <Dot x={x} y={y} box={box} at={[t, r]} tone="danger" />
            </>
          )}
        </Plot>
        <div className="grid grid-cols-2 gap-2 @3xl:grid-cols-1">
          <Stat label="1/(1−β1^t)" value={(1 / mt).toFixed(2)} sub="m 的修正因子" tone="info" />
          <Stat label="1/(1−β2^t)" value={(1 / (vt * vt)).toFixed(1)} sub="v 的修正因子" tone="warning" />
          <Stat label="未修正步长倍数" value={r.toFixed(2)} tone="danger" />
          <Stat label="峰值倍数" value={peak.toFixed(2)} />
        </div>
      </div>
    </VizFrame>
  )
}
