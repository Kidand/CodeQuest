/**
 * 序列 log-prob：只累加 response token 的 log π。
 * 打开两个“错误开关”看会发生什么：
 *   - prompt token 也计入：chosen/rejected 的 prompt 完全相同 → 在 margin 里精确相消，但单条 reward 被平移
 *   - padding 也计入：两条回答 pad 数不同 → margin 被垃圾值污染
 */
import { useMemo, useState } from 'react'
import { Legend, TokenRow, VizFrame, type TokenItem, type VizTone } from '@/viz'
import { Switch } from '@/components/ui'
import { Stat, f3, logSigmoid } from './plot'

type Kind = 'prompt' | 'resp' | 'pad'
interface Tok {
  text: string
  kind: Kind
  pi: number | null // log π_θ(token | 前文)；第一个 token 没有条件概率
  ref: number | null
}

const PROMPT: Tok[] = [
  { text: '<user>', kind: 'prompt', pi: null, ref: null },
  { text: '1+1=?', kind: 'prompt', pi: -3.1, ref: -2.6 },
  { text: '<asst>', kind: 'prompt', pi: -0.2, ref: -0.3 },
]
const CHOSEN: Tok[] = [
  ...PROMPT,
  { text: '答案', kind: 'resp', pi: -1.1, ref: -1.3 },
  { text: '是', kind: 'resp', pi: -0.4, ref: -0.5 },
  { text: '2', kind: 'resp', pi: -0.3, ref: -0.9 },
  { text: '<eos>', kind: 'resp', pi: -0.2, ref: -0.3 },
]
const REJECTED: Tok[] = [
  ...PROMPT,
  { text: '3', kind: 'resp', pi: -2.0, ref: -1.6 },
  { text: '<eos>', kind: 'resp', pi: -0.5, ref: -0.4 },
  { text: '<pad>', kind: 'pad', pi: -9.2, ref: -6.5 },
  { text: '<pad>', kind: 'pad', pi: -8.1, ref: -7.0 },
]
const BETA = 0.1

export function SeqLogpViz() {
  const [withPrompt, setWithPrompt] = useState(false)
  const [withPad, setWithPad] = useState(false)
  const [norm, setNorm] = useState(false)

  const counted = (t: Tok) => t.pi != null && (t.kind === 'resp' || (t.kind === 'prompt' && withPrompt) || (t.kind === 'pad' && withPad))

  const stat = useMemo(() => {
    const agg = (seq: Tok[]) => {
      const used = seq.filter(counted)
      const n = Math.max(1, used.length)
      const s = used.reduce((a, t) => a + (t.pi! - t.ref!), 0)
      const sp = used.reduce((a, t) => a + t.pi!, 0)
      return { ratio: norm ? s / n : s, logp: norm ? sp / n : sp, n: used.length }
    }
    const c = agg(CHOSEN)
    const r = agg(REJECTED)
    const margin = c.ratio - r.ratio
    return { c, r, margin, loss: -logSigmoid(BETA * margin) }
  }, [withPrompt, withPad, norm]) // eslint-disable-line react-hooks/exhaustive-deps

  const toTokens = (seq: Tok[], good: boolean): TokenItem[] =>
    seq.map((t) => {
      const on = counted(t)
      const tone: VizTone = t.kind === 'prompt' ? 'info' : t.kind === 'pad' ? 'muted' : good ? 'success' : 'danger'
      return {
        text: t.text,
        tone,
        dim: !on,
        dashed: t.kind === 'pad',
        sub: t.pi == null ? '—' : `${t.pi - t.ref! >= 0 ? '+' : ''}${(t.pi - t.ref!).toFixed(1)}`,
      }
    })

  const wrong = withPad
  return (
    <VizFrame
      title="序列 log-prob：只累加回复 token"
      caption={
        <>
          margin Δ = <b className="font-mono">{f3(stat.margin)}</b>，loss（β = 0.1）= <b className="font-mono">{f3(stat.loss)}</b>。
          {withPad
            ? ' padding 的垃圾 logp 只出现在较短的 rejected 里，margin 被凭空放大，loss 虚低——模型以为已经学会了。'
            : withPrompt
              ? ' prompt 部分在 chosen 和 rejected 中逐 token 相同，在 margin 里精确抵消；但单条 rewards/chosen、rewards/rejected 都被平移了。'
              : norm
                ? ' 长度归一化后 reward 变成“平均每 token 的 log-ratio”，这时把 prompt 算进来就不再抵消了。'
                : ' 正确做法：只对回复（含 <eos>）求和，prompt 与 padding 的 label 都是 -100。'}
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'prompt', tone: 'info', variant: 'soft' },
            { label: 'chosen 回复', tone: 'success', variant: 'soft' },
            { label: 'rejected 回复', tone: 'danger', variant: 'soft' },
            { label: 'padding', tone: 'muted', variant: 'soft' },
            { label: '小字 = log π_θ − log π_ref', tone: 'muted', variant: 'line' },
          ]}
        />
      }
      controls={
        <>
          <Switch checked={withPrompt} onChange={setWithPrompt} label="把 prompt 也算进去" />
          <Switch checked={withPad} onChange={setWithPad} label="忘了 mask padding" />
          <Switch checked={norm} onChange={setNorm} label="长度归一化（平均）" />
        </>
      }
    >
      <div className="flex flex-col gap-3">
        <div>
          <div className="mb-1 text-xs font-medium text-muted">chosen y_w（{stat.c.n} 个 token 计入）</div>
          <TokenRow tokens={toTokens(CHOSEN, true)} />
        </div>
        <div>
          <div className="mb-1 text-xs font-medium text-muted">rejected y_l（{stat.r.n} 个 token 计入，右侧补了 2 个 pad）</div>
          <TokenRow tokens={toTokens(REJECTED, false)} />
        </div>
        <div className="grid grid-cols-2 gap-2 @xl:grid-cols-4">
          <Stat label="rewards/chosen" value={f3(BETA * stat.c.ratio)} tone="success" sub="β·log(π/π_ref)" />
          <Stat label="rewards/rejected" value={f3(BETA * stat.r.ratio)} tone="danger" />
          <Stat label="margin Δ" value={f3(stat.margin)} tone={wrong ? 'danger' : 'accent'} />
          <Stat label="loss" value={f3(stat.loss)} tone={wrong ? 'danger' : undefined} />
        </div>
      </div>
    </VizFrame>
  )
}
