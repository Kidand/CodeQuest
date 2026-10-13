/**
 * SimPO：奖励 = β·（平均每 token logp），无 reference；目标 margin γ。
 * 对比“序列和”与“长度平均”两种奖励：长回复在求和下天然吃亏，平均则与解码时的似然排序一致。
 */
import { useState } from 'react'
import { Legend, VizFrame, toneColor } from '@/viz'
import { Slider } from '@/components/ui'
import { Stat, f2, f3, logSigmoid } from './plot'

export function SimpoViz() {
  const [nw, setNw] = useState(120)
  const [nl, setNl] = useState(30)
  const [aw, setAw] = useState(-0.6)
  const [al, setAl] = useState(-0.9)
  const [beta, setBeta] = useState(2.5)
  const [gamma, setGamma] = useState(1.4)

  const sumW = nw * aw
  const sumL = nl * al
  const z = beta * (aw - al) - gamma
  const loss = -logSigmoid(z)
  const sumPrefersW = sumW > sumL
  const avgPrefersW = aw > al

  const bar = (label: string, v: number, lim: number, tone: 'success' | 'danger') => (
    <div className="flex items-center gap-2 text-[12.5px]">
      <span className="w-16 shrink-0 text-right text-muted">{label}</span>
      <div className="relative h-4 min-w-0 flex-1 rounded bg-surface-2">
        <div className="absolute inset-y-0 right-0 rounded" style={{ width: `${Math.min(1, Math.abs(v) / lim) * 100}%`, background: toneColor(tone), opacity: 0.75 }} />
      </div>
      <span className="w-16 shrink-0 font-mono tabular-nums">{f2(v)}</span>
    </div>
  )
  const limSum = Math.max(Math.abs(sumW), Math.abs(sumL), 1)

  return (
    <VizFrame
      title="SimPO：平均 logp 当奖励，再要一个 margin"
      caption={
        <>
          按<b>序列和</b> Σ log π，{sumPrefersW ? 'chosen 胜出' : <b style={{ color: toneColor('danger') }}>rejected 胜出</b>}
          （{f2(sumW)} vs {f2(sumL)}）——长回复每多一个 token 就多一个负数；按<b>平均 logp</b>，{avgPrefersW ? 'chosen 胜出' : 'rejected 胜出'}
          （{f2(aw)} vs {f2(al)}），这与 beam search（长度归一化打分）比较候选、多选题评测按平均 logp 打分的口径一致。SimPO 要求 β·(ā_w − ā_l) 超过 γ：当前 {f3(beta * (aw - al))} − {gamma.toFixed(1)} = {f3(z)}，
          loss = <b className="font-mono">{f3(loss)}</b>{z < 0 ? '，还没达到目标 margin，梯度权重 σ(−z) 较大。' : '。'}
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'chosen y_w', tone: 'success', variant: 'soft' },
            { label: 'rejected y_l', tone: 'danger', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <Slider className="min-w-[140px] flex-1" label="|y_w|" value={nw} min={5} max={300} step={5} onChange={setNw} />
          <Slider className="min-w-[140px] flex-1" label="|y_l|" value={nl} min={5} max={300} step={5} onChange={setNl} />
          <Slider className="min-w-[140px] flex-1" label="ā_w（chosen）" value={aw} min={-2} max={-0.1} step={0.05} onChange={setAw} format={(v) => v.toFixed(2)} />
          <Slider className="min-w-[140px] flex-1" label="ā_l（rejected）" value={al} min={-2} max={-0.1} step={0.05} onChange={setAl} format={(v) => v.toFixed(2)} />
          <Slider className="min-w-[140px] flex-1" label="β" value={beta} min={0.5} max={10} step={0.1} onChange={setBeta} format={(v) => v.toFixed(1)} />
          <Slider className="min-w-[140px] flex-1" label="γ" value={gamma} min={0} max={3} step={0.1} onChange={setGamma} format={(v) => v.toFixed(1)} />
        </>
      }
    >
      <div className="grid gap-4 @xl:grid-cols-2">
        <div className="flex flex-col gap-1.5">
          <div className="text-xs font-medium text-muted">① 序列和 Σ log π（越靠右越负）</div>
          {bar('chosen', sumW, limSum, 'success')}
          {bar('rejected', sumL, limSum, 'danger')}
        </div>
        <div className="flex flex-col gap-1.5">
          <div className="text-xs font-medium text-muted">② SimPO 奖励 β/|y|·log π = β·ā</div>
          {bar('chosen', beta * aw, Math.max(beta * 2, 1), 'success')}
          {bar('rejected', beta * al, Math.max(beta * 2, 1), 'danger')}
        </div>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2 @xl:grid-cols-4">
        <Stat label="z = β(ā_w − ā_l) − γ" value={f3(z)} tone={z >= 0 ? 'success' : 'danger'} />
        <Stat label="SimPO loss" value={f3(loss)} tone="accent" />
        <Stat label="求和口径谁赢" value={sumPrefersW ? 'chosen' : 'rejected'} tone={sumPrefersW ? 'success' : 'danger'} />
        <Stat label="平均口径谁赢" value={avgPrefersW ? 'chosen' : 'rejected'} tone={avgPrefersW ? 'success' : 'danger'} />
      </div>
    </VizFrame>
  )
}
