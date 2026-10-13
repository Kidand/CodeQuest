/**
 * LLM-as-a-Judge 的位置偏差与长度偏差（玩具模型）：
 *   judge 认为“第一个更好”的 logit = 质量差(第一 − 第二) + 位置偏差 b + 长度偏差 ℓ·sign(第一更长)
 * 交换顺序各评一次：位置偏差可以抵消，长度偏差不能。
 */
import { useState } from 'react'
import { VizFrame, toneColor, toneSoft } from '@/viz'
import { Slider } from '@/components/ui'
import { f2, sigmoid } from './plot'

export function JudgeBiasViz() {
  const [q, setQ] = useState(0.6) // 质量(A) − 质量(B)，A 短、B 长
  const [b, setB] = useState(1.0)
  const [l, setL] = useState(0.0)

  const pA1 = sigmoid(q + b - l) // 顺序 (A, B)：A 在前、A 更短
  const pA2 = 1 - sigmoid(-q + b + l) // 顺序 (B, A)：B 在前、B 更长
  const win = (p: number) => (p > 0.5 ? 'A' : 'B')
  const consistent = win(pA1) === win(pA2)
  const avg = (pA1 + pA2) / 2
  const truth = q > 0 ? 'A' : q < 0 ? 'B' : '平局'

  const card = (title: string, p: number) => (
    <div className="min-w-0 rounded-xl border border-border bg-surface p-3">
      <div className="mb-1 text-[12.5px] font-medium text-muted">{title}</div>
      <div className="flex h-5 overflow-hidden rounded-md">
        <div style={{ width: `${p * 100}%`, background: toneSoft('info', 70) }} className="flex items-center pl-1 font-mono text-[11px] text-fg">
          {p > 0.25 ? `A ${f2(p)}` : ''}
        </div>
        <div style={{ width: `${(1 - p) * 100}%`, background: toneSoft('warning', 70) }} className="flex items-center justify-end pr-1 font-mono text-[11px] text-fg">
          {p < 0.75 ? `B ${f2(1 - p)}` : ''}
        </div>
      </div>
      <div className="mt-1 text-[12.5px] text-fg">
        判定：<b style={{ color: toneColor(win(p) === 'A' ? 'info' : 'warning') }}>{win(p)} 更好</b>
      </div>
    </div>
  )

  return (
    <VizFrame
      title="LLM 裁判的位置偏差与长度偏差"
      caption={
        <>
          真实情况：<b>{truth}</b> 更好（A 短、B 长）。两种顺序{consistent ? '给出一致结论' : <b style={{ color: toneColor('danger') }}>结论矛盾</b>}；
          交换顺序取平均后 P(A 更好) = <b className="font-mono">{f2(avg)}</b> → {avg > 0.5 ? 'A' : avg < 0.5 ? 'B' : '平局'}。
          {l > 0 && (avg > 0.5 ? 'A' : 'B') !== truth
            ? ' 交换顺序抵消了位置偏差，但抵消不了长度偏差：更长的 B 被系统性高估。'
            : ' 交换顺序能抵消（对称的）位置偏差；把长度偏差调大试试。'}
        </>
      }
      controls={
        <>
          <Slider className="min-w-[150px] flex-1" label="真实质量差 A − B" value={q} min={-2} max={2} step={0.1} onChange={setQ} format={(v) => v.toFixed(1)} />
          <Slider className="min-w-[150px] flex-1" label="位置偏差（偏爱第一个）" value={b} min={0} max={2} step={0.1} onChange={setB} format={(v) => v.toFixed(1)} />
          <Slider className="min-w-[150px] flex-1" label="长度偏差（偏爱更长的）" value={l} min={0} max={2} step={0.1} onChange={setL} format={(v) => v.toFixed(1)} />
        </>
      }
    >
      <div className="grid gap-3 @xl:grid-cols-3">
        {card('顺序 ①：A 在前，B 在后', pA1)}
        {card('顺序 ②：B 在前，A 在后', pA2)}
        {card('交换顺序取平均', avg)}
      </div>
    </VizFrame>
  )
}
