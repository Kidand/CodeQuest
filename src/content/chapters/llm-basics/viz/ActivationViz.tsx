/**
 * 激活函数对比：ReLU / GELU（精确）/ GELU（tanh 近似）/ Swish_β（β=1 即 SiLU）。
 * 洞见：GELU 与 SiLU 都是“光滑的 ReLU”，负半轴有一小段负值（非单调），导数在 0 附近连续；
 * Swish_β 随 β 增大逼近 ReLU，β ≈ 1.702 时与 GELU 几乎重合。
 */
import { useMemo, useState } from 'react'
import { Legend, VizFrame, toneColor } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { Axes, Dot, Line, Plot, VLine, sample, type Axis } from './plot'

/** erf：Abramowitz–Stegun 7.1.26，最大误差约 1.5e-7，足够画图 */
function erf(x: number) {
  const s = Math.sign(x)
  const a = Math.abs(x)
  const t = 1 / (1 + 0.3275911 * a)
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a)
  return s * y
}
const sig = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)))
const relu = (x: number) => Math.max(0, x)
const gelu = (x: number) => 0.5 * x * (1 + erf(x / Math.SQRT2))
const geluTanh = (x: number) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)))
const swish = (b: number) => (x: number) => x * sig(b * x)
const deriv = (f: (x: number) => number) => (x: number) => (f(x + 1e-4) - f(x - 1e-4)) / 2e-4

type View = 'f' | 'df'

export function ActivationViz() {
  const [view, setView] = useState<View>('f')
  const [beta, setBeta] = useState(1)
  const [x0, setX0] = useState(-1)
  const fns = useMemo(
    () => [
      { key: 'relu', label: 'ReLU', f: relu, tone: 'muted' as const, dashed: false },
      { key: 'gelu', label: 'GELU（erf）', f: gelu, tone: 1 as const, dashed: false },
      { key: 'gtanh', label: 'GELU（tanh 近似）', f: geluTanh, tone: 5 as const, dashed: true },
      { key: 'swish', label: beta === 1 ? 'SiLU = Swish₁' : `Swish（β=${beta.toFixed(2)}）`, f: swish(beta), tone: 3 as const, dashed: false },
    ],
    [beta],
  )
  const x: Axis = { min: -5, max: 5, ticks: [-4, -2, 0, 2, 4] }
  const y: Axis = view === 'f' ? { min: -1, max: 5, ticks: [-1, 0, 1, 2, 3, 4, 5], label: 'f(x)' } : { min: -0.3, max: 1.3, ticks: [0, 0.5, 1], label: "f′(x)", fmt: (v) => v.toFixed(1) }
  const g = (f: (x: number) => number) => (view === 'f' ? f : deriv(f))
  const maxDiff = useMemo(() => {
    let m = 0
    for (let i = 0; i <= 2000; i++) {
      const t = -6 + (12 * i) / 2000
      m = Math.max(m, Math.abs(gelu(t) - geluTanh(t)))
    }
    return m
  }, [])

  return (
    <VizFrame
      title="GELU 与 SiLU：光滑、非单调的“软 ReLU”"
      caption={
        <>
          x = <b className="font-mono">{x0.toFixed(2)}</b> 处：ReLU = {relu(x0).toFixed(3)}，GELU = {gelu(x0).toFixed(3)}，SiLU/Swish = {swish(beta)(x0).toFixed(3)}。
          {x0 < 0 && ' 负半轴上 GELU / SiLU 并不为 0，而是一小段负值（GELU 最小约 −0.17，SiLU 最小约 −0.28），所以负输入仍有梯度。'}
          {beta > 4 && ' β 越大，Swish 越接近 ReLU。'}
          {Math.abs(beta - 1.7) < 0.05 && ' β ≈ 1.702 时 x·σ(1.702x) 是 GELU 的另一种常用近似。'}
          {' '}tanh 近似与精确 GELU 在 [−6, 6] 上的最大差约 {maxDiff.toExponential(1)}。
        </>
      }
      legend={<Legend items={fns.map((f) => ({ label: f.label, tone: f.tone, variant: 'line' as const }))} />}
      controls={
        <>
          <Segmented
            size="sm"
            value={view}
            onChange={setView}
            options={[
              { value: 'f', label: '函数值' },
              { value: 'df', label: '导数' },
            ]}
          />
          <Slider className="min-w-[150px] flex-1" label="Swish 的 β" value={beta} min={0.25} max={10} step={0.05} onChange={setBeta} format={(v) => v.toFixed(2)} />
          <Slider className="min-w-[150px] flex-1" label="探针位置 x" value={x0} min={-5} max={5} step={0.05} onChange={setX0} format={(v) => v.toFixed(2)} />
        </>
      }
    >
      <div className="grid items-center gap-4 @3xl:grid-cols-[1fr_200px]">
        <Plot height={270}>
          {(box) => (
            <>
              <Axes x={x} y={y} box={box} />
              <VLine x={x} y={y} box={box} at={x0} />
              {fns.map((f) => (
                <Line key={f.key} pts={sample(g(f.f), x, 240)} x={x} y={y} box={box} tone={f.tone} dashed={f.dashed} width={f.key === 'relu' ? 2.6 : 2.2} />
              ))}
              {fns.map((f) => (
                <Dot key={f.key} x={x} y={y} box={box} at={[x0, g(f.f)(x0)]} tone={f.tone} r={3.5} />
              ))}
            </>
          )}
        </Plot>
        <div className="grid grid-cols-2 gap-2 @3xl:grid-cols-1">
          {fns.map((f) => (
            <div key={f.key} className="min-w-0 rounded-xl border border-border bg-surface px-3 py-2">
              <div className="truncate text-[11px]" style={{ color: toneColor(f.tone) }}>
                {f.label}
              </div>
              <div className="font-mono text-[14px] font-semibold tabular-nums text-fg">
                {view === 'f' ? 'f' : "f′"}({x0.toFixed(1)}) = {g(f.f)(x0).toFixed(3)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </VizFrame>
  )
}
