/**
 * PPO/GRPO 的 clip vs CISPO 的 clipped IS weight：每个 token 对 ∇log π 的系数。
 * PPO：越界 token 系数为 0（被丢弃）；CISPO：系数 = sg(clip(r, ·, 1+ε_high))·A，永不为 0。
 */
import { useState } from 'react'
import { Legend, ProbBars, VizFrame } from '@/viz'
import { Slider } from '@/components/ui'

const TOKENS: { tok: string; r: number; A: number }[] = [
  { tok: 'So', r: 1.02, A: 1 },
  { tok: 'Wait', r: 1.9, A: 1 },
  { tok: 'However', r: 1.45, A: 1 },
  { tok: 'Recheck', r: 2.4, A: 1 },
  { tok: '= 7', r: 0.7, A: -1 },
  { tok: 'thus', r: 1.6, A: -1 },
]

export function CispoViz() {
  const [eps, setEps] = useState(0.2)
  const [epsIS, setEpsIS] = useState(1.0)
  const ppo = TOKENS.map(({ r, A }) => {
    const clipped = A > 0 ? r > 1 + eps : r < 1 - eps
    return clipped ? 0 : r * A
  })
  const cispo = TOKENS.map(({ r, A }) => Math.min(r, 1 + epsIS) * A) // MiniMax-M1：不设下界，只调 ε_high
  const m = Math.max(...ppo.map(Math.abs), ...cispo.map(Math.abs))
  const dropped = ppo.filter((v) => v === 0).length

  return (
    <VizFrame
      title="CISPO：裁剪权重，而不是丢弃 token"
      caption={
        <>
          PPO-clip（ε = {eps.toFixed(2)}）下有 <b>{dropped}</b> 个 token 的梯度系数为 0——多数是 “Wait / Recheck” 这类低概率、被大幅推高的反思 token，
          在多轮 off-policy 更新的后几轮里它们彻底不再贡献梯度。CISPO 把 IS 权重截到 1+ε_high = <b className="font-mono">{(1 + epsIS).toFixed(2)}</b> 并 stop-gradient，
          每个 token 仍以有界的权重参与更新。
        </>
      }
      legend={
        <Legend
          items={[
            { label: '正系数：推高', tone: 'success' },
            { label: '负系数：压低', tone: 'danger' },
            { label: '系数为 0（被丢弃）', tone: 'muted' },
          ]}
        />
      }
      controls={
        <>
          <Slider className="w-44" label="PPO 的 ε" value={eps} min={0.1} max={0.4} step={0.01} onChange={setEps} format={(v) => v.toFixed(2)} />
          <Slider className="w-44" label="CISPO 的 ε_high^IS" value={epsIS} min={0.1} max={2} step={0.05} onChange={setEpsIS} format={(v) => v.toFixed(2)} />
        </>
      }
    >
      <div className="grid gap-5 @3xl:grid-cols-2">
        <div className="min-w-0">
          <div className="mb-2 text-xs font-medium text-muted">PPO / GRPO：∂L/∂log π 的系数（未裁剪时 = r·A）</div>
          <ProbBars
            items={TOKENS.map((t, i) => ({ label: t.tok, p: ppo[i], tone: ppo[i] === 0 ? 'muted' : ppo[i] > 0 ? 'success' : 'danger', note: `r=${t.r}` }))}
            max={m}
            format={(v) => (v === 0 ? '0（丢弃）' : (v > 0 ? '+' : '') + v.toFixed(2))}
          />
        </div>
        <div className="min-w-0">
          <div className="mb-2 text-xs font-medium text-muted">CISPO：sg(clip(r, 1−ε_low, 1+ε_high))·A</div>
          <ProbBars
            items={TOKENS.map((t, i) => ({ label: t.tok, p: cispo[i], tone: cispo[i] > 0 ? 'success' : 'danger', note: `A=${t.A > 0 ? '+1' : '−1'}` }))}
            max={m}
            format={(v) => (v > 0 ? '+' : '') + v.toFixed(2)}
          />
        </div>
      </div>
    </VizFrame>
  )
}
