/**
 * DeepSeek-V4 把亲和度从 Sigmoid 换成 Sqrt(Softplus)。论文没有给出动机，这里只做函数形状上的对比：
 *  - x ≫ 0：sigmoid → 1 饱和（导数 → 0），sqrt(softplus) ≈ √x 继续增长；
 *  - x ≪ 0：sigmoid ≈ e^x，sqrt(softplus) ≈ e^{x/2}，衰减更慢。
 * 右侧：同一组 logits 整体平移后，两种打分在 top-k 内归一化得到的权重。
 */
import { useMemo, useState } from 'react'
import { Legend, ProbBars, VizFrame, toneColor } from '@/viz'
import { Slider } from '@/components/ui'
import { sigmoid, sqrtSoftplus, softplus, topk } from './moeMath'
import { Axes, Line, Stat, Svg, useScales, type Axis } from './plot'

const REL = [1.5, 1.0, 0.5, 0.0, -0.5, -1.0]
const K = 3
const dSig = (x: number) => sigmoid(x) * (1 - sigmoid(x))
const dSsp = (x: number) => sigmoid(x) / (2 * Math.sqrt(softplus(x)))

export function ActivationFnViz() {
  const [shift, setShift] = useState(4)
  const x: Axis = { min: -6, max: 6, ticks: [-6, -4, -2, 0, 2, 4, 6], label: 'logit x' }
  const y: Axis = { min: 0, max: 2.6, ticks: [0, 0.5, 1, 1.5, 2, 2.5], fmt: (v) => v.toFixed(1) }
  const { sx, sy } = useScales(x, y)
  const pts = (f: (v: number) => number) => Array.from({ length: 121 }, (_, i) => { const v = -6 + i * 0.1; return [v, f(v)] as [number, number] })
  const logits = REL.map((r) => r + shift)
  const w = useMemo(() => {
    const calc = (f: (v: number) => number) => {
      const s = logits.map(f)
      const idx = topk(s, K)
      const sum = idx.reduce((a, i) => a + s[i], 0)
      return idx.map((i) => ({ i, w: s[i] / sum }))
    }
    return { sig: calc(sigmoid), ssp: calc(sqrtSoftplus) }
  }, [shift]) // eslint-disable-line react-hooks/exhaustive-deps
  const ratio = (a: { w: number }[]) => a[0].w / a[a.length - 1].w

  return (
    <VizFrame
      title="Sigmoid vs Sqrt(Softplus)：形状差在哪"
      caption={
        <>
          logits 整体平移到 {shift >= 0 ? '+' : ''}{shift.toFixed(1)}：top-{K} 归一化后最大/最小权重之比，sigmoid 为 <b className="font-mono">{ratio(w.sig).toFixed(3)}</b>，
          sqrt(softplus) 为 <b className="font-mono">{ratio(w.ssp).toFixed(3)}</b>。
          {shift > 2
            ? ' logits 偏大时 sigmoid 饱和，被选中的专家权重几乎一样；sqrt(softplus) 仍保留强弱差别，且导数不会衰减到 0。'
            : shift < -1
              ? ' logits 偏小时 sigmoid ≈ eˣ 把差别放大，sqrt(softplus) ≈ e^{x/2} 更平缓。'
              : ' logits 在 0 附近时二者差别不大。'}
          （论文未说明换函数的动机，此为函数形状分析。）
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'sigmoid', tone: 'warning', variant: 'line' },
            { label: 'sqrt(softplus)', tone: 'accent', variant: 'line' },
            { label: '虚线 = 导数', tone: 'muted', variant: 'line' },
          ]}
        />
      }
      controls={<Slider className="min-w-[220px] flex-1" label="logits 整体平移" value={shift} min={-4} max={6} step={0.1} onChange={setShift} format={(v) => v.toFixed(1)} />}
    >
      <div className="grid items-start gap-5 @3xl:grid-cols-2">
        <Svg>
          <Axes x={x} y={y} />
          <Line pts={pts(sigmoid)} x={x} y={y} tone="warning" />
          <Line pts={pts(sqrtSoftplus)} x={x} y={y} tone="accent" />
          <Line pts={pts(dSig)} x={x} y={y} tone="warning" dashed width={1.5} />
          <Line pts={pts(dSsp)} x={x} y={y} tone="accent" dashed width={1.5} />
          {logits.map((l, i) =>
            l >= -6 && l <= 6 ? (
              <g key={i}>
                <circle cx={sx(l)} cy={sy(sigmoid(l))} r={3.5} fill={toneColor('warning')} />
                <circle cx={sx(l)} cy={sy(sqrtSoftplus(l))} r={3.5} fill={toneColor('accent')} />
              </g>
            ) : null,
          )}
        </Svg>
        <div className="min-w-0 space-y-3">
          <div className="text-xs font-medium text-muted">6 个专家的 logits = 平移量 + [1.5, 1, 0.5, 0, −0.5, −1]，选 top-{K} 后归一化的权重</div>
          <div className="text-xs text-muted">sigmoid（V3）</div>
          <ProbBars items={w.sig.map((o) => ({ label: `E${o.i}`, p: o.w, tone: 'warning' as const }))} max={0.6} />
          <div className="text-xs text-muted">sqrt(softplus)（V4）</div>
          <ProbBars items={w.ssp.map((o) => ({ label: `E${o.i}`, p: o.w, tone: 'accent' as const }))} max={0.6} />
          <div className="grid grid-cols-2 gap-2">
            <Stat label={`x=${logits[0].toFixed(1)} 时 sigmoid 导数`} value={dSig(logits[0]).toFixed(4)} tone="warning" />
            <Stat label={`x=${logits[0].toFixed(1)} 时 √softplus 导数`} value={dSsp(logits[0]).toFixed(4)} tone="accent" />
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
