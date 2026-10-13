/**
 * SFT 的 loss mask 与归约方式：
 *  - labels 中 prompt 与 padding 位置设为 -100（ignore_index），不计入分子也不计入分母
 *  - “batch 内 token 平均”（F.cross_entropy 默认）让长回复的样本占更大权重；
 *    “先样本内平均再跨样本平均”让每条样本权重相同
 * 每个 token 下方的数字是“预测该 token 时的 −log p”（由给定的 p 计算）。
 */
import { useMemo, useState } from 'react'
import { Legend, TokenRow, VizFrame, toneColor } from '@/viz'
import { Segmented, Switch } from '@/components/ui'
import { Stat } from './plot'

type Role = 'prompt' | 'reply' | 'pad'
interface Tok {
  t: string
  role: Role
  /** 模型给“这个 token”的预测概率（第一个 token 没有被预测） */
  p?: number
}
const PAD: Tok = { t: '<pad>', role: 'pad' }
const A: Tok[] = [
  { t: '<user>', role: 'prompt' },
  { t: '1+1=?', role: 'prompt', p: 0.002 },
  { t: '<asst>', role: 'prompt', p: 0.45 },
  { t: '2', role: 'reply', p: 0.55 },
  { t: '<eos>', role: 'reply', p: 0.9 },
  ...Array(8).fill(PAD),
]
const B: Tok[] = [
  { t: '<user>', role: 'prompt' },
  { t: '写', role: 'prompt', p: 0.01 },
  { t: '首', role: 'prompt', p: 0.05 },
  { t: '诗', role: 'prompt', p: 0.2 },
  { t: '<asst>', role: 'prompt', p: 0.5 },
  { t: '春', role: 'reply', p: 0.12 },
  { t: '风', role: 'reply', p: 0.3 },
  { t: '又', role: 'reply', p: 0.25 },
  { t: '绿', role: 'reply', p: 0.4 },
  { t: '江', role: 'reply', p: 0.7 },
  { t: '南', role: 'reply', p: 0.85 },
  { t: '岸', role: 'reply', p: 0.8 },
  { t: '<eos>', role: 'reply', p: 0.6 },
]
type Red = 'token' | 'seq'

export function LossMaskViz() {
  const [maskPrompt, setMaskPrompt] = useState(true)
  const [red, setRed] = useState<Red>('token')

  const res = useMemo(() => {
    const valid = (k: Tok) => k.p != null && k.role !== 'pad' && (k.role === 'reply' || !maskPrompt)
    const seqs = [A, B]
    const nValid = seqs.map((s) => s.filter(valid).length)
    const N = nValid[0] + nValid[1]
    const rows = seqs.map((s, si) =>
      s.map((k) => {
        const ok = valid(k)
        const l = k.p != null ? -Math.log(k.p) : 0
        const w = !ok ? 0 : red === 'token' ? 1 / N : 1 / (2 * nValid[si])
        return { k, ok, l, w }
      }),
    )
    const total = rows.flat().reduce((acc, r) => acc + r.w * r.l, 0)
    const share = rows.map((r) => r.reduce((acc, x) => acc + x.w, 0))
    return { rows, total, share, nValid, N }
  }, [maskPrompt, red])

  const tone = (k: Tok, ok: boolean) => (k.role === 'pad' ? undefined : !ok ? 'muted' : k.role === 'reply' ? 'success' : 'warning')

  return (
    <VizFrame
      title="SFT 的 loss：哪些 token 算、怎么平均"
      caption={
        <>
          参与计算的 token：A 有 {res.nValid[0]} 个、B 有 {res.nValid[1]} 个。
          {red === 'token'
            ? ` 按 batch 内 token 平均（分母 = ${res.N}）：每个 token 权重相同，B 因为回复更长，占总梯度的 ${(res.share[1] * 100).toFixed(0)}%。`
            : ' 先在样本内平均、再跨样本平均：A、B 各占 50%，A 的每个 token 权重远大于 B 的 token。'}
          {!maskPrompt && ' 关闭 mask 后 prompt 也被算进 loss——模型在学“复述用户问题”，而且 prompt 的 loss 通常很大，会稀释回复部分的信号。'}
        </>
      }
      legend={
        <Legend
          items={[
            { label: '回复 token（算 loss）', tone: 'success', variant: 'soft' },
            { label: 'prompt 也算 loss', tone: 'warning', variant: 'soft' },
            { label: 'label = −100（忽略）', tone: 'muted', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <Switch checked={maskPrompt} onChange={setMaskPrompt} label="prompt 位置 label 设为 −100" />
          <Segmented
            size="sm"
            value={red}
            onChange={setRed}
            options={[
              { value: 'token', label: 'batch 内 token 平均' },
              { value: 'seq', label: '先样本内平均再跨样本平均' },
            ]}
          />
        </>
      }
    >
      <div className="space-y-4">
        {res.rows.map((row, si) => (
          <div key={si} className="min-w-0">
            <div className="mb-1.5 flex flex-wrap items-baseline gap-x-3 text-xs text-muted">
              <span className="font-medium text-fg">样本 {si === 0 ? 'A（短回复）' : 'B（长回复）'}</span>
              <span>
                总权重 <b className="font-mono" style={{ color: toneColor(si === 0 ? 'info' : 'llm') }}>{(res.share[si] * 100).toFixed(1)}%</b>
              </span>
              <span>token 下方：−log p ／ 在总 loss 中的权重</span>
            </div>
            <TokenRow
              tokens={row.map(({ k, ok, l, w }) => ({
                text: k.t,
                tone: tone(k, ok),
                dim: k.role === 'pad' || (k.p != null && !ok),
                dashed: k.role === 'pad',
                sub: k.p == null ? (k.role === 'pad' ? '-100' : '—') : ok ? `${l.toFixed(2)}／${(w * 100).toFixed(1)}%` : `${l.toFixed(2)}／-100`,
              }))}
            />
          </div>
        ))}
        <div className="grid grid-cols-2 gap-2 @xl:grid-cols-4">
          <Stat label="batch loss" value={res.total.toFixed(3)} tone="accent" />
          <Stat label="有效 token 数" value={res.N} />
          <Stat label="A 的权重" value={`${(res.share[0] * 100).toFixed(1)}%`} tone="info" />
          <Stat label="B 的权重" value={`${(res.share[1] * 100).toFixed(1)}%`} tone="llm" />
        </div>
      </div>
    </VizFrame>
  )
}
