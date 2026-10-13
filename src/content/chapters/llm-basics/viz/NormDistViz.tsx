/**
 * 一个 token 的隐藏向量（d = 12）经过 LayerNorm / RMSNorm 前后的对比。
 * 洞见：LayerNorm 对输入的“整体平移 + 整体缩放”都不变；RMSNorm 只对缩放不变——
 * 均值偏移大时 RMSNorm 的输出被“共同分量”占满，原本的起伏被压扁。
 */
import { useMemo, useState } from 'react'
import { Legend, VizFrame, toneColor } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { Plot, Stat, scales, type Axis } from './plot'

const BASE = [0.9, -1.4, 0.3, 2.1, -0.6, 0.1, -2.2, 1.2, 0.5, -0.3, 1.7, -0.9]
type Mode = 'ln' | 'rms'

const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length
const std = (v: number[]) => {
  const m = mean(v)
  return Math.sqrt(mean(v.map((x) => (x - m) ** 2)))
}
const rms = (v: number[]) => Math.sqrt(mean(v.map((x) => x * x)))
/** 两位小数，去掉 -0.00 */
const f2 = (v: number) => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2))

function Bars({ vals, title, tone, lim }: { vals: number[]; title: string; tone: 'accent' | 'success'; lim: number }) {
  const x: Axis = { min: -0.5, max: vals.length - 0.5, ticks: [] }
  const ticks = lim > 6 ? [-lim, -lim / 2, 0, lim / 2, lim].map((v) => Math.round(v)) : [-lim, -lim / 2, 0, lim / 2, lim]
  const y: Axis = { min: -lim, max: lim, ticks, fmt: (v) => (Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1)) }
  const m = mean(vals)
  return (
    <div className="min-w-0">
      <div className="mb-1 text-xs font-medium text-muted">{title}</div>
      <Plot height={200} maxW={460} pad={{ b: 14, t: 10 }}>
        {(box) => {
          const { sx, sy } = scales(x, y, box)
          const bw = Math.max(4, ((box.W - box.pad.l - box.pad.r) / vals.length) * 0.66)
          return (
            <>
              <g fontSize={11} fill="var(--muted)">
                {y.ticks.map((t) => (
                  <g key={t}>
                    <line x1={box.pad.l} x2={box.W - box.pad.r} y1={sy(t)} y2={sy(t)} stroke="var(--border)" strokeWidth={t === 0 ? 1.3 : 0.7} />
                    <text x={box.pad.l - 5} y={sy(t) + 3.5} textAnchor="end">
                      {y.fmt!(t)}
                    </text>
                  </g>
                ))}
              </g>
              {vals.map((v, i) => {
                const cl = Math.max(-lim, Math.min(lim, v))
                const y0 = sy(0)
                const y1 = sy(cl)
                return (
                  <rect
                    key={i}
                    x={sx(i) - bw / 2}
                    y={Math.min(y0, y1)}
                    width={bw}
                    height={Math.max(1, Math.abs(y1 - y0))}
                    rx={2}
                    fill={toneColor(tone)}
                    fillOpacity={0.85}
                    style={{ transition: 'all 0.25s' }}
                  />
                )
              })}
              <line x1={box.pad.l} x2={box.W - box.pad.r} y1={sy(Math.max(-lim, Math.min(lim, m)))} y2={sy(Math.max(-lim, Math.min(lim, m)))} stroke={toneColor('danger')} strokeWidth={1.6} strokeDasharray="5 3" />
            </>
          )
        }}
      </Plot>
    </div>
  )
}

export function NormDistViz({ initial = 'ln', initialShift = 0 }: { initial?: Mode; initialShift?: number }) {
  const [mode, setMode] = useState<Mode>(initial)
  const [shift, setShift] = useState(initialShift)
  const [scale, setScale] = useState(1)
  const [gamma, setGamma] = useState(1)
  const x = useMemo(() => BASE.map((v) => v * scale + shift), [shift, scale])
  const y = useMemo(() => {
    const eps = 1e-5
    if (mode === 'ln') {
      const m = mean(x)
      const s = Math.sqrt(std(x) ** 2 + eps)
      return x.map((v) => (gamma * (v - m)) / s)
    }
    const r = Math.sqrt(rms(x) ** 2 + eps)
    return x.map((v) => (gamma * v) / r)
  }, [x, mode, gamma])
  const limIn = Math.max(3, Math.ceil(Math.max(...x.map(Math.abs)) / 2) * 2)
  // 输出“起伏”的大小：去掉均值后的标准差
  const spreadOut = std(y)

  return (
    <VizFrame
      title={mode === 'ln' ? 'LayerNorm：不管输入怎么平移、缩放，输出都是均值 0、方差 1' : 'RMSNorm：只除以均方根，不减均值'}
      caption={
        mode === 'ln' ? (
          <>
            输入均值 {f2(mean(x))}、标准差 {f2(std(x))} → 输出均值 {f2(mean(y))}、标准差 {f2(std(y))}（再乘以 γ）。
            拖动“平移”和“缩放”，右图纹丝不动：LayerNorm 对 <b>x → a·x + b</b> 不变（a &gt; 0）。
          </>
        ) : (
          <>
            输入 RMS {rms(x).toFixed(2)} → 输出 RMS {rms(y).toFixed(2)}（再乘以 γ）。缩放不改变右图；
            {Math.abs(shift) >= 2
              ? ` 但平移 ${shift} 后，输出被共同的偏移量占据，各维“起伏”（去均值后的标准差）只剩 ${spreadOut.toFixed(2)}——RMSNorm 不具备平移不变性。`
              : ' 平移为 0 附近时，RMSNorm 与 LayerNorm 几乎一样——实际网络中激活的均值本来就接近 0，这正是 RMSNorm 可以省掉减均值的依据。'}
          </>
        )
      }
      legend={<Legend items={[{ label: '均值', tone: 'danger', variant: 'line' }, { label: '输入 x', tone: 'accent' }, { label: '输出 y', tone: 'success' }]} />}
      controls={
        <>
          <Segmented
            size="sm"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'ln', label: 'LayerNorm' },
              { value: 'rms', label: 'RMSNorm' },
            ]}
          />
          <Slider className="min-w-[140px] flex-1" label="整体平移 b" value={shift} min={-6} max={6} step={0.5} onChange={setShift} />
          <Slider className="min-w-[140px] flex-1" label="整体缩放 a" value={scale} min={0.2} max={5} step={0.1} onChange={setScale} format={(v) => v.toFixed(1)} />
          <Slider className="min-w-[140px] flex-1" label="γ（统一取值）" value={gamma} min={0.5} max={2} step={0.1} onChange={setGamma} format={(v) => v.toFixed(1)} />
        </>
      }
    >
      <div className="grid gap-4 @2xl:grid-cols-2">
        <Bars vals={x} title={`① 输入 x = a·base + b（d = ${BASE.length}）`} tone="accent" lim={limIn} />
        <Bars vals={y} title={`② ${mode === 'ln' ? 'γ ⊙ (x − μ) / σ' : 'γ ⊙ x / RMS(x)'}`} tone="success" lim={3} />
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2 @xl:grid-cols-4">
        <Stat label="输入 μ / σ" value={`${f2(mean(x))} / ${f2(std(x))}`} />
        <Stat label="输入 RMS" value={rms(x).toFixed(2)} />
        <Stat label="输出 μ / σ" value={`${f2(mean(y))} / ${f2(std(y))}`} tone="success" />
        <Stat label="输出 RMS" value={rms(y).toFixed(2)} tone="success" />
      </div>
    </VizFrame>
  )
}

export const NormDistRmsViz = () => <NormDistViz initial="rms" initialShift={3} />
