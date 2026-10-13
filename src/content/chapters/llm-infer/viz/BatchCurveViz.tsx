/**
 * batch 的作用：吞吐上界与 TPOT 随 batch 的变化。
 *   t(B) = max(FLOPs(B) / peak, Bytes(B) / BW)，吞吐 = B / t(B)
 * 两条竖线：拐点 B*（计算时间 = 访存时间）与 KV 显存上限 B_max。
 */
import { useMemo, useState } from 'react'
import { VizFrame, toneColor } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { PlotFrame, Stat, pathOf, type Axis } from './common'
import { GPUS, MODELS, decodeCost, fmtMs, fmtNum, kvPerToken, minTP } from './hw'

export function BatchCurveViz() {
  const [mid, setMid] = useState('8b')
  const [gid, setGid] = useState('h100')
  const [ce, setCe] = useState(11)
  const m = MODELS.find((x) => x.id === mid)!
  const g = GPUS.find((x) => x.id === gid)!
  const S = 2 ** ce

  const data = useMemo(() => {
    const k = minTP(m, g)
    const split = Math.min(k, m.kvHeads)
    const step = (B: number) => {
      const d = decodeCost(m, B, S)
      const tC = d.flops / k / (g.tflops * 1e12)
      const tM = (d.wB / k + d.kvB / split) / (g.tbs * 1e12)
      return { t: Math.max(tC, tM), tC, tM }
    }
    const Bs = Array.from({ length: 121 }, (_, i) => 2 ** (i / 12)) // 1 .. 1024
    const pts = Bs.map((B) => ({ B, ...step(B) }))
    const knee = pts.find((p) => p.tC >= p.tM)?.B ?? Infinity
    const kvPer = (S * kvPerToken(m) * k) / split
    const bMax = Math.floor((k * g.gb * 1e9 * 0.9 - m.params * 2) / kvPer)
    return { k, pts, knee, bMax, step }
  }, [m, g, S])

  const xA: Axis = { min: 1, max: 1024, log: true, ticks: [1, 4, 16, 64, 256, 1024], label: 'batch（对数）' }
  const thrMax = Math.max(...data.pts.map((p) => p.B / p.t))
  const yT: Axis = {
    min: 10,
    max: 10 ** Math.ceil(Math.log10(thrMax * 1.2)),
    log: true,
    ticks: [10, 100, 1e3, 1e4, 1e5, 1e6].filter((v) => v <= 10 ** Math.ceil(Math.log10(thrMax * 1.2))),
    fmt: (v) => fmtNum(v, 0),
    label: '吞吐上界（token/s）',
  }
  const tMax = Math.max(...data.pts.map((p) => p.t)) * 1e3
  const yL: Axis = { min: 0, max: Math.ceil(tMax * 1.15), ticks: [0, Math.ceil(tMax * 1.15) / 2, Math.ceil(tMax * 1.15)].map((v) => Math.round(v * 10) / 10), fmt: (v) => `${v}`, label: '单步时间 / TPOT 下界（ms）' }
  const bm = Math.max(1, data.bMax)
  const at = (B: number) => data.step(B)

  const marks = (s: { sx: (v: number) => number; t: number; H: number; b: number; W: number; r: number }) => (
    <g>
      {data.bMax < 1024 && (
        <rect x={s.sx(Math.max(1, data.bMax))} y={s.t} width={Math.max(0, s.W - s.r - s.sx(Math.max(1, data.bMax)))} height={s.H - s.b - s.t} fill={toneColor('danger')} opacity={0.08} />
      )}
      {data.knee <= 1024 && (
        <g>
          <line x1={s.sx(data.knee)} x2={s.sx(data.knee)} y1={s.t} y2={s.H - s.b} stroke={toneColor('success')} strokeDasharray="4 4" />
          <text x={s.sx(data.knee) - 4} y={s.t + 12} fontSize={11} textAnchor="end" fill={toneColor('success')}>
            拐点 B*≈{Math.round(data.knee)}
          </text>
        </g>
      )}
      {data.bMax >= 1 && data.bMax < 1024 && (
        <text x={s.sx(data.bMax) + 4} y={s.t + 26} fontSize={11} fill={toneColor('danger')}>
          KV 放不下 →
        </text>
      )}
    </g>
  )

  return (
    <VizFrame
      title="batch 越大，吞吐越高、单步越慢"
      caption={
        <>
          {m.name} on {g.name}
          {data.k > 1 && ` ×${data.k}`}，上下文 {fmtNum(S, 0)}：batch 从 1 增到 {Math.min(bm, 1024)}，吞吐上界从 {fmtNum(1 / at(1).t, 0)} 涨到 {fmtNum(Math.min(bm, 1024) / at(Math.min(bm, 1024)).t, 0)} token/s，
          而 TPOT 下界只从 {fmtMs(at(1).t)} 变为 {fmtMs(at(Math.min(bm, 1024)).t)}。
          {data.knee > Math.min(bm, 1024)
            ? ' 在 KV 显存耗尽之前都没走到算力拐点：上下文越长，KV 读取越多，decode 越难变成算力瓶颈。'
            : ` 超过拐点 B*≈${Math.round(data.knee)} 后单步时间随 batch 线性增长，吞吐趋于平台。`}
        </>
      }
      controls={
        <>
          <Segmented size="sm" value={mid} onChange={setMid} options={MODELS.map((x) => ({ value: x.id, label: x.name.replace('Llama-3.1-', '') }))} />
          <Segmented size="sm" value={gid} onChange={setGid} options={GPUS.map((x) => ({ value: x.id, label: x.name.split(' ')[0] }))} />
          <Slider className="min-w-[180px] flex-1" label="上下文 S" value={ce} min={8} max={15} onChange={setCe} format={(v) => fmtNum(2 ** v, 0)} />
        </>
      }
    >
      <div className="grid gap-4 @3xl:grid-cols-2">
        <PlotFrame x={xA} y={yT} height={220} pad={{ l: 46, r: 12, t: 20, b: 36 }}>
          {(s) => (
            <g>
              {marks(s)}
              <path d={pathOf(data.pts.map((p) => [p.B, p.B / p.t]), s, yT)} fill="none" stroke={toneColor('accent')} strokeWidth={2.4} />
            </g>
          )}
        </PlotFrame>
        <PlotFrame x={xA} y={yL} height={220} pad={{ l: 46, r: 12, t: 20, b: 36 }}>
          {(s) => (
            <g>
              {marks(s)}
              <path d={pathOf(data.pts.map((p) => [p.B, p.t * 1e3]), s, yL)} fill="none" stroke={toneColor('warning')} strokeWidth={2.4} />
              <path d={pathOf(data.pts.map((p) => [p.B, p.tM * 1e3]), s, yL)} fill="none" stroke="var(--muted)" strokeWidth={1.2} strokeDasharray="3 3" />
            </g>
          )}
        </PlotFrame>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2 @xl:grid-cols-4">
        <Stat label="张量并行" value={`TP${data.k}`} sub="按权重 ≤ 72% 显存估算" />
        <Stat label="拐点 B*" value={data.knee > 1024 ? '> 1024' : Math.round(data.knee)} tone="success" sub="计算时间 = 访存时间" />
        <Stat label="KV 容量上限" value={data.bMax < 1 ? '0' : data.bMax} tone="danger" sub="90% 显存 − 权重" />
        <Stat label="每序列 KV" value={`${fmtNum(S * kvPerToken(m), 1)}B`} sub={`${(kvPerToken(m) / 1024).toFixed(0)} KiB / token`} />
      </div>
    </VizFrame>
  )
}
