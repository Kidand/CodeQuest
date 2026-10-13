/**
 * SFT loss 的三种归一化：按 token 平均、按样本平均、朴素梯度累积（每个 micro-batch 内 token 平均再平均）。
 * 看每条样本在总梯度中的占比如何随回复长度变化。
 */
import { useState } from 'react'
import { ProbBars, VizFrame } from '@/viz'
import { Slider } from '@/components/ui'
import { f3 } from './plot'

const L = { A: 2.0, B: 1.0, C: 0.5 } // 每条样本的平均 token loss（短样本难、长样本易，仅示意）
const NB = 16

export function LossAvgViz() {
  const [na, setNa] = useState(4)
  const [nc, setNc] = useState(96)
  const n = { A: na, B: NB, C: nc }
  const N = na + NB + nc

  const tok = { A: na / N, B: NB / N, C: nc / N }
  const tokL = (na * L.A + NB * L.B + nc * L.C) / N
  const smp = { A: 1 / 3, B: 1 / 3, C: 1 / 3 }
  const smpL = (L.A + L.B + L.C) / 3
  const mb1 = na + NB
  const ga = { A: (0.5 * na) / mb1, B: (0.5 * NB) / mb1, C: 0.5 }
  const gaL = 0.5 * ((na * L.A + NB * L.B) / mb1) + 0.5 * L.C

  const block = (title: string, loss: number, share: { A: number; B: number; C: number }, note: string) => (
    <div className="min-w-0 rounded-xl border border-border bg-surface p-3">
      <div className="mb-0.5 text-[13px] font-semibold text-fg">{title}</div>
      <div className="mb-2 font-mono text-xs text-muted">
        loss = {f3(loss)} · {note}
      </div>
      <ProbBars
        className="!min-w-0"
        max={1}
        format={(p) => `${(p * 100).toFixed(1)}%`}
        items={(['A', 'B', 'C'] as const).map((k, i) => ({ label: `${k} n=${n[k]}`, p: share[k], tone: (i + 1) as 1 | 2 | 3 }))}
      />
    </div>
  )

  return (
    <VizFrame
      title="Loss 归一化：谁的梯度占大头"
      caption={
        <>
          同一批数据：按 token 平均时长样本 C 占 <b className="font-mono">{(tok.C * 100).toFixed(1)}%</b> 的梯度，短样本 A 的每个 token 与 C 的 token 等权；
          按样本平均时三条各占 1/3，A 的每个 token 权重是 C 的 <b className="font-mono">{(nc / na).toFixed(1)}</b> 倍；
          朴素梯度累积（[A, B] 与 [C] 两个 micro-batch 各自 token 平均再平均）既不是前者也不是后者，结果取决于 batch 怎么切。
        </>
      }
      controls={
        <>
          <Slider className="min-w-[160px] flex-1" label="样本 A 回复长度" value={na} min={1} max={32} onChange={setNa} />
          <Slider className="min-w-[160px] flex-1" label="样本 C 回复长度" value={nc} min={8} max={256} step={8} onChange={setNc} />
        </>
      }
    >
      <div className="grid gap-3 @3xl:grid-cols-3">
        {block('① 全局按 token 平均', tokL, tok, '每 token 权重 1/ΣN')}
        {block('② 先样本内、再样本间平均', smpL, smp, '每 token 权重 1/(B·nᵢ)')}
        {block('③ 朴素梯度累积', gaL, ga, '[A,B] | [C] 各自平均')}
      </div>
      <div className="mt-2 text-[12px] text-muted">柱长 = 该样本所有 token 在总梯度中的权重之和。示意数据：A、B、C 的平均 token loss 分别为 2.0、1.0、0.5，B 固定 16 个 token。</div>
    </VizFrame>
  )
}
