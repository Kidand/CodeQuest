/**
 * Bradley-Terry：P(y_w ≻ y_l) = σ(r_w − r_l)，loss = −log σ(r_w − r_l − m)。
 * 拖动整体平移 c 验证“只看差值”；切换 Llama 2 的 margin 方案看 loss 曲线右移。
 */
import { useState } from 'react'
import { Legend, VizFrame } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { Axes, Dot, Line, Stat, Svg, f3, logSigmoid, sample, sigmoid, type Axis } from './plot'

type Scheme = 'none' | 'small' | 'large'
type Rating = 'sig' | 'better' | 'slight' | 'neg'
const MARGIN: Record<Scheme, Record<Rating, number>> = {
  none: { sig: 0, better: 0, slight: 0, neg: 0 },
  small: { sig: 1, better: 2 / 3, slight: 1 / 3, neg: 0 },
  large: { sig: 3, better: 2, slight: 1, neg: 0 },
}

export function BtViz() {
  const [rw, setRw] = useState(1.2)
  const [rl, setRl] = useState(0.4)
  const [c, setC] = useState(0)
  const [scheme, setScheme] = useState<Scheme>('small')
  const [rating, setRating] = useState<Rating>('sig')
  const m = MARGIN[scheme][rating]
  const d = rw - rl
  const loss = -logSigmoid(d - m)
  const g = sigmoid(m - d)

  const x: Axis = { min: -6, max: 6, ticks: [-6, -4, -2, 0, 2, 4, 6], label: 'Δ = r(x, y_w) − r(x, y_l)' }
  const y: Axis = { min: 0, max: 3, ticks: [0, 0.5, 1, 2, 3], label: '' }

  return (
    <VizFrame
      title="Bradley-Terry：分数差决定偏好概率"
      caption={
        <>
          两个分数整体加上 c = {c.toFixed(1)} 后为 ({(rw + c).toFixed(1)}, {(rl + c).toFixed(1)})，差值仍是 <b className="font-mono">{d.toFixed(2)}</b>，概率与 loss 都不变——
          RM 的绝对值没有意义，只有同一 prompt 下的差值有意义（所以 InstructGPT 在 RL 前给 RM 加一个偏置，让人类示范的平均分为 0）。
          {m > 0 ? <> 当前 margin m = {f3(m)}：要求分差超过 m，loss 才会降到 ln2 以下。</> : null}
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'P(y_w ≻ y_l) = σ(Δ)', tone: 'accent', variant: 'line' },
            { label: '−log σ(Δ − m)', tone: 'danger', variant: 'line' },
            { label: '无 margin 的 loss', tone: 'muted', variant: 'line' },
          ]}
        />
      }
      controls={
        <>
          <Slider className="min-w-[120px] flex-1" label="r(y_w)" value={rw} min={-4} max={4} step={0.1} onChange={setRw} format={(v) => v.toFixed(1)} />
          <Slider className="min-w-[120px] flex-1" label="r(y_l)" value={rl} min={-4} max={4} step={0.1} onChange={setRl} format={(v) => v.toFixed(1)} />
          <Slider className="min-w-[120px] flex-1" label="整体平移 c" value={c} min={-5} max={5} step={0.5} onChange={setC} format={(v) => v.toFixed(1)} />
          <Segmented
            size="sm"
            value={scheme}
            onChange={setScheme}
            options={[
              { value: 'none', label: '无 margin' },
              { value: 'small', label: 'Margin Small' },
              { value: 'large', label: 'Margin Large' },
            ]}
          />
          <Segmented
            size="sm"
            value={rating}
            onChange={setRating}
            options={[
              { value: 'sig', label: '显著更好' },
              { value: 'better', label: '更好' },
              { value: 'slight', label: '略好' },
              { value: 'neg', label: '几乎一样' },
            ]}
          />
        </>
      }
    >
      <div className="grid items-center gap-4 @3xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <Svg>
          <Axes x={x} y={y} />
          <Line pts={sample((v) => -logSigmoid(v), x)} x={x} y={y} tone="muted" dashed width={1.6} />
          <Line pts={sample((v) => -logSigmoid(v - m), x)} x={x} y={y} tone="danger" />
          <Line pts={sample(sigmoid, x)} x={x} y={y} tone="accent" />
          <Dot x={x} y={y} px={d} py={loss} tone="danger" />
          <Dot x={x} y={y} px={d} py={sigmoid(d)} tone="accent" />
        </Svg>
        <div className="grid grid-cols-2 gap-2">
          <Stat label="Δ = r_w − r_l" value={f3(d)} />
          <Stat label="P(y_w ≻ y_l)" value={f3(sigmoid(d))} tone="accent" />
          <Stat label="loss" value={f3(loss)} tone="danger" sub={`margin m = ${f3(m)}`} />
          <Stat label="∂loss/∂r_w = −σ(m − Δ)" value={f3(-g)} sub="∂loss/∂r_l = +σ(m − Δ)" />
        </div>
      </div>
    </VizFrame>
  )
}
