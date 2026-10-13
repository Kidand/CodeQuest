/**
 * 数值溢出 / 下溢的反例：朴素 softmax 在大 logit 上 inf/inf = NaN，在很负的 logit 上 0/0 = NaN；
 * 减去最大值后，指数的最大项恰为 exp(0) = 1，分母 ≥ 1，两种失败都消失。
 * dtype 只模拟“数值范围”（溢出到 inf、下溢到 0），不模拟尾数舍入。
 */
import { useMemo, useState } from 'react'
import { VizFrame, toneColor, toneSoft } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { cn } from '@/lib/utils'
import { C, fnum } from './plot'

type DType = 'f16' | 'f32' | 'f64'
const BASE = [1.0, 0.8, 0.3, -0.5]
const LIMIT: Record<DType, { max: number; minSub: number; lnMax: string }> = {
  f16: { max: 65504, minSub: 2 ** -24, lnMax: '11.09' },
  f32: { max: 3.4028234663852886e38, minSub: 1.401298464324817e-45, lnMax: '88.72' },
  f64: { max: Number.MAX_VALUE, minSub: 5e-324, lnMax: '709.78' },
}

/** 把 float64 结果“存进”目标 dtype：超范围 → inf，小于最小次正规数的一半 → 0 */
function store(v: number, dt: DType) {
  if (Number.isNaN(v)) return NaN
  const L = LIMIT[dt]
  const a = Math.abs(v)
  if (a > L.max * (dt === 'f64' ? 1 : 1.0000001)) return v > 0 ? Infinity : -Infinity
  if (a !== 0 && a < L.minSub / 2) return 0
  return dt === 'f32' ? Math.fround(v) : v
}

function run(z: number[], dt: DType) {
  const e1 = z.map((v) => store(Math.exp(v), dt))
  const s1 = store(e1.reduce((a, b) => a + b, 0), dt)
  const p1 = e1.map((v) => store(v / s1, dt))
  const m = Math.max(...z)
  const d = z.map((v) => v - m)
  const e2 = d.map((v) => store(Math.exp(v), dt))
  const s2 = store(e2.reduce((a, b) => a + b, 0), dt)
  const p2 = e2.map((v) => store(v / s2, dt))
  const lse = m + Math.log(s2)
  const logp = z.map((v) => v - lse)
  return { e1, s1, p1, m, d, e2, s2, p2, logp }
}

const bad = (v: number) => !Number.isFinite(v)

function Cell({ v, danger, warn }: { v: number; danger?: boolean; warn?: boolean }) {
  const tone = danger ? 'danger' : warn ? 'warning' : null
  return (
    <td
      className={cn('px-1.5 py-1 text-right font-mono text-[12px] tabular-nums')}
      style={tone ? { background: toneSoft(tone, 22), color: toneColor(tone), fontWeight: 600 } : undefined}
    >
      {fnum(v, 4)}
    </td>
  )
}

export function OverflowViz() {
  const [scale, setScale] = useState(1)
  const [shift, setShift] = useState(100)
  const [dt, setDt] = useState<DType>('f32')
  const z = useMemo(() => BASE.map((b) => b * scale + shift), [scale, shift])
  const r = useMemo(() => run(z, dt), [z, dt])

  const naiveOk = !r.p1.some(bad)
  const stableOk = !r.p2.some(bad)
  const overflow = r.e1.some((v) => v === Infinity)
  const underflow = r.s1 === 0

  return (
    <VizFrame
      title="朴素 softmax 会 NaN：减去最大值就好了"
      caption={
        <>
          logits = <C>[{z.map((v) => fnum(v, 2)).join(', ')}]</C>，dtype = <C>{dt === 'f16' ? 'float16' : dt === 'f32' ? 'float32' : 'float64'}</C>
          （exp 在 x &gt; {LIMIT[dt].lnMax} 时溢出）。
          {overflow && <> 朴素版：<b style={{ color: toneColor('danger') }}>exp 溢出为 inf，inf / inf = NaN</b>。</>}
          {!overflow && underflow && <> 朴素版：<b style={{ color: toneColor('danger') }}>全部 exp 下溢为 0，0 / 0 = NaN</b>。</>}
          {naiveOk && ' 朴素版此时还能算对。'}
          {stableOk ? ' 减最大值后：最大项恰为 exp(0)=1，分母 ≥ 1，永远不会溢出也不会 0/0。' : ''}
        </>
      }
      controls={
        <>
          <Segmented
            size="sm"
            value={dt}
            onChange={setDt}
            options={[
              { value: 'f16', label: 'float16' },
              { value: 'f32', label: 'float32' },
              { value: 'f64', label: 'float64' },
            ]}
          />
          <Slider className="min-w-[170px] flex-1" label="logit 尺度 s" value={scale} min={1} max={150} step={1} onChange={setScale} format={(v) => `×${v}`} />
          <Slider className="min-w-[170px] flex-1" label="整体平移 b" value={shift} min={-1200} max={1200} step={10} onChange={setShift} />
        </>
      }
    >
      <div className="grid gap-4 @2xl:grid-cols-2">
        <div className="min-w-0 rounded-xl border border-border bg-surface p-3">
          <div className="mb-2 flex items-center justify-between text-xs">
            <span className="font-medium text-fg">朴素：exp(z) / Σexp(z)</span>
            <span className="font-mono font-semibold" style={{ color: toneColor(naiveOk ? 'success' : 'danger') }}>
              {naiveOk ? 'OK' : 'NaN!'}
            </span>
          </div>
          <table className="w-full border-collapse">
            <thead>
              <tr className="text-[11px] text-muted">
                <th className="px-1.5 text-right font-normal">z</th>
                <th className="px-1.5 text-right font-normal">exp(z)</th>
                <th className="px-1.5 text-right font-normal">p</th>
              </tr>
            </thead>
            <tbody>
              {z.map((v, i) => (
                <tr key={i} className="border-t border-border">
                  <Cell v={v} />
                  <Cell v={r.e1[i]} danger={r.e1[i] === Infinity} warn={r.e1[i] === 0} />
                  <Cell v={r.p1[i]} danger={bad(r.p1[i])} />
                </tr>
              ))}
              <tr className="border-t border-border-strong text-[11px]">
                <td className="px-1.5 py-1 text-right text-muted">Σ</td>
                <Cell v={r.s1} danger={bad(r.s1) || r.s1 === 0} />
                <td />
              </tr>
            </tbody>
          </table>
        </div>
        <div className="min-w-0 rounded-xl border border-border bg-surface p-3">
          <div className="mb-2 flex items-center justify-between text-xs">
            <span className="font-medium text-fg">稳定：m = max(z)，exp(z − m)</span>
            <span className="font-mono font-semibold" style={{ color: toneColor(stableOk ? 'success' : 'danger') }}>
              {stableOk ? 'OK' : 'NaN!'}
            </span>
          </div>
          <table className="w-full border-collapse">
            <thead>
              <tr className="text-[11px] text-muted">
                <th className="px-1.5 text-right font-normal">z − m</th>
                <th className="px-1.5 text-right font-normal">exp</th>
                <th className="px-1.5 text-right font-normal">p</th>
                <th className="px-1.5 text-right font-normal">log p</th>
              </tr>
            </thead>
            <tbody>
              {z.map((_, i) => (
                <tr key={i} className="border-t border-border">
                  <Cell v={r.d[i]} />
                  <Cell v={r.e2[i]} warn={r.e2[i] === 0} />
                  <Cell v={r.p2[i]} danger={bad(r.p2[i])} />
                  <Cell v={r.logp[i]} />
                </tr>
              ))}
              <tr className="border-t border-border-strong text-[11px]">
                <td className="px-1.5 py-1 text-right text-muted">Σ</td>
                <Cell v={r.s2} />
                <td />
                <td />
              </tr>
            </tbody>
          </table>
          <div className="mt-2 text-[11.5px] leading-relaxed text-muted">
            log p 用 <C>z − m − log Σexp(z − m)</C> 直接算：即使某个 p 下溢成 0（黄色），log p 仍是有限值。
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
