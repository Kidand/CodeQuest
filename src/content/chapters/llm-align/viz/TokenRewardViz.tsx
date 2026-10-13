/**
 * RLHF 的逐 token 奖励：r_t = −β·KL_t，最后一个 token 再加上 RM 分数。
 * 对比 k1（log π/π_ref，可正可负）与 k3（(π_ref/π) − 1 − log(π_ref/π)，恒 ≥ 0）两种逐 token KL 估计。
 */
import { useState } from 'react'
import { Legend, VizFrame, toneColor } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { Stat, f3 } from './plot'

const TOKS = ['猫', '坐', '在', '垫', '子', '上', '。', '<eos>']
const K1 = [0.05, 0.6, -0.2, 1.2, 0.0, 0.3, -0.1, 0.05] // log π_θ − log π_ref（采样 token 处）

export function TokenRewardViz() {
  const [beta, setBeta] = useState(0.1)
  const [rm, setRm] = useState(1.5)
  const [est, setEst] = useState<'k1' | 'k3'>('k1')
  const kl = K1.map((k) => (est === 'k1' ? k : Math.exp(-k) - 1 + k))
  const pen = kl.map((k) => -beta * k)
  const T = TOKS.length
  const r = pen.map((p, t) => p + (t === T - 1 ? rm : 0))
  const lim = Math.max(0.3, ...r.map(Math.abs), Math.abs(rm)) * 1.15

  const W = 420
  const H = 220
  const padL = 34
  const padB = 34
  const padT = 18
  const bw = (W - padL - 10) / T
  const y = (v: number) => padT + ((lim - v) / (2 * lim)) * (H - padT - padB)

  return (
    <VizFrame
      title="奖励只在末 token，KL 每步都扣"
      caption={
        <>
          总回报（γ = 1）= RM 分数 {f3(rm)} + KL 惩罚 {f3(pen.reduce((a, b) => a + b, 0))} = <b className="font-mono">{f3(r.reduce((a, b) => a + b, 0))}</b>。
          {est === 'k1'
            ? ' k1 = log(π_θ/π_ref) 是无偏估计，但单个 token 上可以为负（如“在”“。”），负 KL 反而变成奖励。'
            : ' k3 = (π_ref/π_θ) − 1 − log(π_ref/π_θ) 逐项恒 ≥ 0，方差更小；GRPO 把它直接放进 loss（见「推理强化学习」）。'}
          RM 分数只出现在 &lt;eos&gt; 上，前面 token 的“功劳”全靠 critic + GAE 往回分配。
        </>
      }
      legend={
        <Legend
          items={[
            { label: '−β·KL_t（逐 token）', tone: 'danger', variant: 'soft' },
            { label: 'RM 分数（只在最后一个 token）', tone: 'success', variant: 'soft' },
            { label: '合计 r_t', tone: 'accent', variant: 'line' },
          ]}
        />
      }
      controls={
        <>
          <Slider className="min-w-[150px] flex-1" label="KL 系数 β" value={beta} min={0} max={0.5} step={0.01} onChange={setBeta} format={(v) => v.toFixed(2)} />
          <Slider className="min-w-[150px] flex-1" label="RM 分数" value={rm} min={-3} max={3} step={0.1} onChange={setRm} format={(v) => v.toFixed(1)} />
          <Segmented
            size="sm"
            value={est}
            onChange={setEst}
            options={[
              { value: 'k1', label: 'k1 = log ratio' },
              { value: 'k3', label: 'k3（恒非负）' },
            ]}
          />
        </>
      }
    >
      <div className="grid items-center gap-4 @3xl:grid-cols-[minmax(0,3fr)_minmax(0,1fr)]">
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: W * 1.35, display: 'block', margin: '0 auto' }}>
          <line x1={padL} x2={W - 6} y1={y(0)} y2={y(0)} stroke="var(--border-strong)" strokeWidth={1.2} />
          <text x={padL - 5} y={y(0) + 4} textAnchor="end" fontSize={12} fill="var(--muted)">
            0
          </text>
          {TOKS.map((tok, t) => {
            const x0 = padL + t * bw + bw * 0.15
            const w = bw * 0.7
            const p = pen[t]
            const isLast = t === T - 1
            return (
              <g key={t}>
                {/* KL 惩罚 */}
                <rect x={x0} width={w} y={Math.min(y(0), y(p))} height={Math.abs(y(p) - y(0))} fill={toneColor('danger')} opacity={0.55} rx={2} />
                {isLast && (
                  <rect x={x0} width={w} y={Math.min(y(p), y(p + rm))} height={Math.abs(y(p + rm) - y(p))} fill={toneColor('success')} opacity={0.6} rx={2} />
                )}
                <line x1={x0 - 2} x2={x0 + w + 2} y1={y(r[t])} y2={y(r[t])} stroke={toneColor('accent')} strokeWidth={2.5} />
                <text x={x0 + w / 2} y={r[t] >= 0 ? y(r[t]) - 5 : y(r[t]) + 14} textAnchor="middle" fontSize={11.5} fill="var(--fg)" className="font-mono">
                  {r[t].toFixed(2)}
                </text>
                <text x={x0 + w / 2} y={H - 18} textAnchor="middle" fontSize={13} fill="var(--fg)">
                  {tok}
                </text>
                <text x={x0 + w / 2} y={H - 4} textAnchor="middle" fontSize={10.5} fill="var(--muted)" className="font-mono">
                  {kl[t].toFixed(2)}
                </text>
              </g>
            )
          })}
          <text x={4} y={H - 4} fontSize={10.5} fill="var(--muted)">
            KL
          </text>
        </svg>
        <div className="grid grid-cols-2 gap-2 @3xl:grid-cols-1">
          <Stat label="Σ KL_t" value={f3(kl.reduce((a, b) => a + b, 0))} tone="danger" />
          <Stat label="回报 Σ r_t" value={f3(r.reduce((a, b) => a + b, 0))} tone="accent" />
        </div>
      </div>
    </VizFrame>
  )
}
