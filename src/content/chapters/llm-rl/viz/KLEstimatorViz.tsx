/**
 * Schulman 的三种单样本 KL 估计 k1 / k2 / k3（x = π_ref/π_θ，样本来自 π_θ）。
 * 左：三条曲线；右：在一个 6-token 的小分布上精确计算每个估计量的期望与标准差。
 * 洞见：k3 恒非负且无偏；两分布接近时方差远小于 k1，偏离大时重尾反而更大；k2 有偏但小偏离时很接近。
 */
import { useMemo, useState } from 'react'
import { Legend, VizFrame, toneColor } from '@/viz'
import { Slider } from '@/components/ui'
import { Axes, Line, Svg, f3, sample, useScales, type Axis } from './plot'

const TOK = ['So', 'Let', 'Wait', 'Thus', 'x', '<eos>']
const REF_LOGITS = [2.0, 1.5, 0.8, 0.3, -0.5, -1.0]
const DELTA = [-1.2, 0.4, 1.0, 0.2, 0.6, -0.3]

const softmax = (z: number[]) => {
  const m = Math.max(...z)
  const e = z.map((v) => Math.exp(v - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}
const k1 = (x: number) => -Math.log(x)
const k2 = (x: number) => 0.5 * Math.log(x) ** 2
const k3 = (x: number) => x - 1 - Math.log(x)

export function KLEstimatorViz() {
  const [lam, setLam] = useState(1)
  const { pt, pr, kl, rows } = useMemo(() => {
    const pr = softmax(REF_LOGITS)
    const pt = softmax(REF_LOGITS.map((z, i) => z + lam * DELTA[i]))
    const kl = pt.reduce((s, p, i) => s + p * Math.log(p / pr[i]), 0)
    const rows = (
      [
        ['k1 = −log x', k1, 1],
        ['k2 = ½(log x)²', k2, 4],
        ['k3 = x − 1 − log x', k3, 'accent'],
      ] as const
    ).map(([name, f, tone]) => {
      const vals = pt.map((_, i) => f(pr[i] / pt[i]))
      const mean = vals.reduce((s, v, i) => s + pt[i] * v, 0)
      const sd = Math.sqrt(vals.reduce((s, v, i) => s + pt[i] * (v - mean) ** 2, 0))
      return { name, mean, sd, neg: Math.min(...vals) < -1e-12, tone, vals }
    })
    return { pt, pr, kl, rows }
  }, [lam])

  const x: Axis = { min: 0.1, max: 10, log: true, ticks: [0.1, 0.2, 0.5, 1, 2, 5, 10], label: 'x = π_ref / π_θ（对数轴）', fmt: (v) => String(v) }
  const y: Axis = { min: -1.5, max: 3, ticks: [-1, 0, 1, 2, 3], label: '单样本估计值' }
  const { sx, sy } = useScales(x, y)

  return (
    <VizFrame
      title="KL 的单样本估计：k3 无偏且恒 ≥ 0"
      caption={
        <>
          真实 KL(π_θ‖π_ref) = <b className="font-mono">{f3(kl)}</b>。k1、k3 的期望都等于它（无偏），但 k1 的单样本值可以是负数；
          k3 由 log x ≤ x−1 保证每个样本非负。把 λ 拉小：k3 的标准差远小于 k1；把 λ 拉大（π_θ ≪ π_ref 的 token 出现，x 很大）：
          k3 的重尾让标准差反超 k1——「k3 方差小」只在两分布接近时成立。
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'k1', tone: 1, variant: 'line' },
            { label: 'k2', tone: 4, variant: 'line' },
            { label: 'k3（GRPO 采用）', tone: 'accent', variant: 'line' },
            { label: '6 个 token 的 x 值', tone: 'gold' },
          ]}
        />
      }
      controls={<Slider className="min-w-[220px] flex-1" label="策略偏离 λ（π_θ = softmax(z_ref + λ·δ)）" value={lam} min={0} max={2.5} step={0.05} onChange={setLam} format={(v) => v.toFixed(2)} />}
    >
      <div className="grid items-start gap-5 @3xl:grid-cols-2">
        <Svg>
          <Axes x={x} y={y} />
          <Line pts={sample(k1, x)} x={x} y={y} tone={1} width={1.6} />
          <Line pts={sample(k2, x)} x={x} y={y} tone={4} width={1.6} />
          <Line pts={sample(k3, x)} x={x} y={y} tone="accent" />
          {pt.map((p, i) => {
            const xv = Math.min(10, Math.max(0.1, pr[i] / p))
            return <circle key={i} cx={sx(xv)} cy={sy(Math.min(3, k3(xv)))} r={2.5 + 9 * p} fill={toneColor('gold')} opacity={0.85} stroke="var(--surface)" />
          })}
        </Svg>
        <div className="min-w-0 space-y-3">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[300px] text-[12.5px]">
              <thead className="text-muted">
                <tr>
                  <th className="py-1 text-left font-medium">token</th>
                  {TOK.map((t) => (
                    <th key={t} className="py-1 text-right font-mono font-medium">
                      {t}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="font-mono tabular-nums">
                <tr>
                  <td className="py-0.5 text-muted">π_θ</td>
                  {pt.map((p, i) => (
                    <td key={i} className="text-right">
                      {p.toFixed(2)}
                    </td>
                  ))}
                </tr>
                <tr>
                  <td className="py-0.5 text-muted">π_ref</td>
                  {pr.map((p, i) => (
                    <td key={i} className="text-right">
                      {p.toFixed(2)}
                    </td>
                  ))}
                </tr>
                <tr>
                  <td className="py-0.5 text-muted">k3</td>
                  {rows[2].vals.map((v, i) => (
                    <td key={i} className="text-right">
                      {v.toFixed(2)}
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
          <table className="w-full text-[12.5px]">
            <thead className="text-muted">
              <tr>
                <th className="text-left font-medium">估计量</th>
                <th className="text-right font-medium">E_πθ[k]</th>
                <th className="text-right font-medium">标准差</th>
                <th className="text-right font-medium">样本可为负</th>
              </tr>
            </thead>
            <tbody className="font-mono tabular-nums">
              {rows.map((r) => (
                <tr key={r.name}>
                  <td className="py-1" style={{ color: toneColor(r.tone) }}>
                    {r.name}
                  </td>
                  <td className="text-right">{f3(r.mean)}</td>
                  <td className="text-right">{f3(r.sd)}</td>
                  <td className="text-right">{r.neg ? '是' : '否'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </VizFrame>
  )
}
