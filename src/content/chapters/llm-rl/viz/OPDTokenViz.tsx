/**
 * On-policy 蒸馏的逐 token 信号：学生采样一条序列，教师对每个位置打分。
 * 采样估计：log p_s(y_t) − log p_t(y_t)（Thinking Machines 的做法，作为负优势）；
 * 全词表：Σ_v p_s(v)·(log p_s(v) − log p_t(v))（DeepSeek-V4 的做法，方差更小）。
 */
import { useMemo, useState } from 'react'
import { Legend, ProbBars, TokenRow, VizFrame } from '@/viz'
import { Stat } from './plot'

interface Pos {
  cand: string[]
  s: number[] // 学生 logits
  t: number[] // 教师 logits
  pick: number // 学生采样到的候选下标
}

const SEQ: Pos[] = [
  { cand: ['Let', 'We', 'First', 'So', 'The'], s: [2.0, 1.2, 0.8, 0.2, 0.0], t: [2.1, 1.0, 0.9, 0.0, 0.1], pick: 0 },
  { cand: [' n', ' x', ' k', ' it', ' a'], s: [1.5, 1.4, 0.3, 0.2, -0.5], t: [1.6, 1.3, 0.4, 0.0, -0.4], pick: 1 },
  { cand: [' =', ' be', ' >', ' :', ' ≥'], s: [2.2, 0.5, 0.3, 0.1, -0.2], t: [2.4, 0.4, 0.2, 0.0, -0.3], pick: 0 },
  { cand: [' 7', ' 5', ' 6', ' 8', ' 3'], s: [1.2, 1.0, 0.9, 0.4, 0.1], t: [0.1, 2.8, 0.3, -0.5, -0.8], pick: 0 },
  { cand: [',', '.', ' so', ' then', ' and'], s: [1.8, 1.0, 0.6, 0.2, 0.0], t: [1.0, 1.9, 0.7, 0.1, -0.2], pick: 0 },
  { cand: [' so', ' Wait', ' then', ' hence', ' thus'], s: [2.0, -0.5, 1.0, 0.3, 0.2], t: [0.2, 2.2, 0.4, 0.1, 0.0], pick: 0 },
  { cand: [' answer', ' x²', ' result', ' sum', ' n'], s: [1.6, 1.2, 0.8, 0.1, 0.0], t: [1.5, 1.0, 0.9, 0.2, 0.0], pick: 0 },
]

const logsm = (z: number[]) => {
  const m = Math.max(...z)
  const l = Math.log(z.reduce((a, v) => a + Math.exp(v - m), 0)) + m
  return z.map((v) => v - l)
}

export function OPDTokenViz() {
  const [sel, setSel] = useState(3)
  const rows = useMemo(
    () =>
      SEQ.map((p) => {
        const ls = logsm(p.s)
        const lt = logsm(p.t)
        const sampled = ls[p.pick] - lt[p.pick]
        const full = ls.reduce((a, l, i) => a + Math.exp(l) * (l - lt[i]), 0)
        return { ls, lt, sampled, full, adv: -sampled }
      }),
    [],
  )
  const r = rows[sel]
  const p = SEQ[sel]

  return (
    <VizFrame
      title="教师给学生采样的每个 token 打分"
      caption={
        <>
          位置 {sel + 1}，学生采样了 “<b className="font-mono">{p.cand[p.pick].trim()}</b>”：log p_s − log p_t = <b className="font-mono">{r.sampled.toFixed(3)}</b>，
          作为优势的值为 <b className="font-mono">{r.adv.toFixed(3)}</b>（{r.adv < 0 ? '教师不认同，压低' : '教师认同，推高'}）。
          全词表反向 KL = <b className="font-mono">{r.full.toFixed(3)}</b>：单个采样 token 的估计在期望上等于它，但方差大；DeepSeek-V4 改用全词表 KL 以获得更稳定的梯度。
        </>
      }
      legend={
        <Legend
          items={[
            { label: '优势 > 0（教师更认同）', tone: 'success', variant: 'soft' },
            { label: '优势 < 0（教师不认同）', tone: 'danger', variant: 'soft' },
          ]}
        />
      }
    >
      <div className="space-y-4">
        <div>
          <div className="mb-2 text-xs font-medium text-muted">学生采样的序列（点击 token 查看两边的分布；下方数字为逐 token 优势 −(log p_s − log p_t)）</div>
          <TokenRow
            tokens={SEQ.map((q, i) => ({
              text: q.cand[q.pick],
              sub: rows[i].adv.toFixed(2),
              tone: rows[i].adv < -0.3 ? 'danger' : rows[i].adv > 0.05 ? 'success' : undefined,
              dashed: i !== sel,
            }))}
            onClick={setSel}
          />
        </div>
        <div className="grid gap-4 @3xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_200px]">
          <div className="min-w-0">
            <div className="mb-1 text-xs text-muted">学生 p_s</div>
            <ProbBars items={p.cand.map((c, i) => ({ label: c.trim(), p: Math.exp(r.ls[i]), tone: i === p.pick ? 'accent' : 'muted' }))} max={1} />
          </div>
          <div className="min-w-0">
            <div className="mb-1 text-xs text-muted">教师 p_t</div>
            <ProbBars items={p.cand.map((c, i) => ({ label: c.trim(), p: Math.exp(r.lt[i]), tone: i === p.pick ? 'accent' : 'info' }))} max={1} />
          </div>
          <div className="grid grid-cols-3 gap-2 @3xl:grid-cols-1">
            <Stat label="采样估计" value={r.sampled.toFixed(3)} />
            <Stat label="全词表 KL" value={r.full.toFixed(3)} tone="info" />
            <Stat label="优势" value={r.adv.toFixed(3)} tone={r.adv < 0 ? 'danger' : 'success'} />
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
