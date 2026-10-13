/**
 * 学习率调度：线性 warmup + {cosine / linear / WSD / constant}。
 * 洞见：cosine 的 lr 从一开始就在下降，总步数必须事先确定；WSD 在稳定段保持峰值，
 * 任何时刻都可以从稳定段的 checkpoint 拉出一段短的衰减得到“训练完”的模型，也便于继续训练。
 */
import { useMemo, useState } from 'react'
import { Legend, VizFrame } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { Axes, Dot, Line, Plot, Stat, VLine, type Axis } from './plot'

type Kind = 'cosine' | 'linear' | 'wsd' | 'const'
const N = 10000
const MAX = 3e-4

function lrAt(t: number, kind: Kind, warm: number, minRatio: number, decayFrac: number) {
  const min = MAX * minRatio
  if (t < warm) return (MAX * (t + 1)) / warm
  if (kind === 'const') return MAX
  if (kind === 'wsd') {
    const start = Math.max(warm, N * (1 - decayFrac))
    if (t < start) return MAX
    const p = (t - start) / Math.max(1, N - start)
    return min + 0.5 * (MAX - min) * (1 + Math.cos(Math.PI * Math.min(1, p)))
  }
  const p = Math.min(1, (t - warm) / Math.max(1, N - warm))
  return kind === 'cosine' ? min + 0.5 * (MAX - min) * (1 + Math.cos(Math.PI * p)) : MAX - (MAX - min) * p
}

export function LRScheduleViz() {
  const [kind, setKind] = useState<Kind>('cosine')
  const [warm, setWarm] = useState(500)
  const [minRatio, setMinRatio] = useState(0.1)
  const [decayFrac, setDecayFrac] = useState(0.2)
  const [step, setStep] = useState(3000)
  const pts = useMemo(() => {
    const out: [number, number][] = []
    for (let t = 0; t <= N; t += 20) out.push([t, lrAt(t, kind, warm, minRatio, decayFrac) * 1e4])
    return out
  }, [kind, warm, minRatio, decayFrac])
  const cos = useMemo(() => {
    const out: [number, number][] = []
    for (let t = 0; t <= N; t += 40) out.push([t, lrAt(t, 'cosine', warm, minRatio, decayFrac) * 1e4])
    return out
  }, [warm, minRatio, decayFrac])
  const x: Axis = { min: 0, max: N, ticks: [0, 2500, 5000, 7500, 10000], label: '训练步', fmt: (v) => (v ? `${v / 1000}k` : '0') }
  const y: Axis = { min: 0, max: 3.3, ticks: [0, 1, 2, 3], label: 'lr（×1e-4）' }
  const cur = lrAt(step, kind, warm, minRatio, decayFrac)
  const area = pts.reduce((s, p) => s + p[1], 0) / pts.length / (MAX * 1e4)

  return (
    <VizFrame
      title="warmup + 衰减：学习率曲线决定每个阶段“迈多大步”"
      caption={
        <>
          峰值 3e-4，warmup {warm} 步，最低 {(minRatio * 100).toFixed(0)}% 峰值。第 {step} 步的 lr = <b className="font-mono">{cur.toExponential(2)}</b>，
          全程平均 lr 为峰值的 {(area * 100).toFixed(0)}%。
          {kind === 'cosine' && ' cosine：一过 warmup 就开始缓慢下降，中段下降最快，末端趋平；总步数必须事先定好。'}
          {kind === 'linear' && ' 线性衰减：与 cosine 相比前期 lr 更低、后期更高。'}
          {kind === 'wsd' && ` WSD：大部分时间停在峰值，最后 ${(decayFrac * 100).toFixed(0)}% 才衰减；从稳定段任一 checkpoint 接一段衰减就能得到可用模型，也方便续训。`}
          {kind === 'const' && ' 常数学习率：loss 会停在一个“噪声地板”上，需要衰减段才能继续下降。'}
        </>
      }
      legend={
        <Legend
          items={[
            { label: '当前调度', tone: 'accent', variant: 'line' },
            ...(kind !== 'cosine' ? [{ label: 'cosine 对照', tone: 'muted' as const, variant: 'line' as const }] : []),
          ]}
        />
      }
      controls={
        <>
          <Segmented
            size="sm"
            value={kind}
            onChange={setKind}
            options={[
              { value: 'cosine', label: 'cosine' },
              { value: 'linear', label: '线性' },
              { value: 'wsd', label: 'WSD' },
              { value: 'const', label: '常数' },
            ]}
          />
          <Slider className="min-w-[140px] flex-1" label="warmup 步数" value={warm} min={0} max={2000} step={50} onChange={setWarm} />
          <Slider className="min-w-[140px] flex-1" label="最低 lr / 峰值" value={minRatio} min={0} max={0.5} step={0.05} onChange={setMinRatio} format={(v) => `${(v * 100).toFixed(0)}%`} />
          {kind === 'wsd' && <Slider className="min-w-[140px] flex-1" label="衰减段占比" value={decayFrac} min={0.05} max={0.4} step={0.05} onChange={setDecayFrac} format={(v) => `${(v * 100).toFixed(0)}%`} />}
          <Slider className="min-w-[140px] flex-1" label="查看第 t 步" value={step} min={0} max={N} step={50} onChange={setStep} />
        </>
      }
    >
      <div className="grid items-center gap-4 @3xl:grid-cols-[1fr_190px]">
        <Plot height={240}>
          {(box) => (
            <>
              <Axes x={x} y={y} box={box} />
              {warm > 0 && <VLine x={x} y={y} box={box} at={warm} tone="warning" label="warmup 结束" />}
              {kind !== 'cosine' && <Line pts={cos} x={x} y={y} box={box} tone="muted" dashed width={1.6} />}
              <Line pts={pts} x={x} y={y} box={box} tone="accent" />
              <Dot x={x} y={y} box={box} at={[step, cur * 1e4]} tone="accent" />
            </>
          )}
        </Plot>
        <div className="grid grid-cols-2 gap-2 @3xl:grid-cols-1">
          <Stat label={`第 ${step} 步 lr`} value={cur.toExponential(2)} tone="accent" />
          <Stat label="平均 lr / 峰值" value={`${(area * 100).toFixed(0)}%`} />
          <Stat label="第 0 步 lr" value={lrAt(0, kind, warm, minRatio, decayFrac).toExponential(2)} sub={warm ? '= 峰值 / warmup' : '无 warmup'} />
        </div>
      </div>
    </VizFrame>
  )
}
