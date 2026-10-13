/**
 * 投机解码逐步动画：草稿模型自回归起草 γ 个 token → target 一次前向验证 → 从左到右接受 / 拒绝 →
 * 拒绝时从残差分布重采样，全部接受时送一个 bonus token。连续播放 3 个回合并累计统计。
 *
 * toy 模型：8 个 token 的词表。target 的 logits 由「上一个 token 的二元表 + 与位置相关的扰动」决定；
 * draft 的 logits = (1-λ)·target + λ·另一张无关的表，λ = 1 - 草稿质量。随机数来自固定种子。
 */
import { useMemo, useState } from 'react'
import { Shuffle } from 'lucide-react'
import { CodeTrace, Legend, StepControls, TokenRow, VizFrame, useStepper, type TokenItem } from '@/viz'
import { Button, Slider } from '@/components/ui'
import { rng } from '@/lib/utils'
import { C, PQBars, sampleIdx, softmax } from './common'

const VOCAB = ['我', '想', '吃', '苹果', '和', '香蕉', '面包', '。']
const V = VOCAB.length
const LOW = -2
// BIGRAM[上一个 token][下一个 token]
const BIGRAM: number[][] = VOCAB.map(() => Array(V).fill(LOW))
const set = (a: number, row: Record<number, number>) => Object.entries(row).forEach(([b, v]) => (BIGRAM[a][Number(b)] = v))
set(0, { 1: 3.2, 2: 1.6 })
set(1, { 2: 3.4, 0: -0.5 })
set(2, { 3: 2.4, 5: 2.0, 6: 1.6 })
set(3, { 4: 2.3, 7: 1.9 })
set(4, { 5: 2.3, 6: 2.0, 3: 1.0 })
set(5, { 7: 2.6, 4: 1.5 })
set(6, { 7: 2.6, 4: 1.6 })
set(7, { 0: 3.0, 1: 0.5 })

function targetLogits(ctx: number[]) {
  const last = ctx[ctx.length - 1]
  return BIGRAM[last].map((v, j) => v + 0.35 * Math.sin(1.7 * ctx.length + 2.3 * j))
}
const targetP = (ctx: number[]) => softmax(targetLogits(ctx))
function draftQ(ctx: number[], quality: number) {
  const lam = 1 - quality
  const last = ctx[ctx.length - 1]
  const t = targetLogits(ctx)
  return softmax(t.map((v, j) => (1 - lam) * v + lam * 2.4 * Math.cos(3.1 * last + 1.3 * j + 0.7)))
}

type Status = 'pending' | 'accepted' | 'rejected'
interface Frame {
  committed: number[]
  drafts: { tok: number; status: Status }[]
  extra?: { tok: number; kind: 'resample' | 'bonus' }
  line: number
  caption: React.ReactNode
  panel: { title: string; kind?: 'q' | 'pq' | 'res' | 'p'; p?: number[]; q?: number[]; hl?: number }
  stats: { tokens: number; target: number; draft: number }
}

const CODE = `def speculative_round(prefix, gamma):
    drafts, qs = [], []
    for i in range(gamma):                  # ① 草稿：小模型自回归 γ 步
        q = draft_model(prefix + drafts)
        drafts.append(sample(q)); qs.append(q)
    ps = target_model(prefix, drafts)       # ② 1 次前向 → γ+1 个分布
    for i, x in enumerate(drafts):          # ③ 从左到右逐个检查
        if rand() < min(1, ps[i][x] / qs[i][x]):
            continue                        #    接受
        r = norm(max(0, ps[i] - qs[i]))     #    拒绝 → 残差分布
        return drafts[:i] + [sample(r)]
    return drafts + [sample(ps[gamma])]     # ④ 全部接受：bonus token`

const ROUNDS = 3

export function buildFrames(gamma: number, quality: number, seed: number): Frame[] {
  const r = rng(seed)
  const frames: Frame[] = []
  let committed = [0]
  const stats = { tokens: 0, target: 0, draft: 0 }
  const w = (ids: number[]) => ids.map((i) => VOCAB[i]).join('')
  for (let round = 1; round <= ROUNDS; round++) {
    const drafts: { tok: number; status: Status }[] = []
    const qs: number[][] = []
    const snap = (extra?: Frame['extra']) => ({ committed: [...committed], drafts: drafts.map((d) => ({ ...d })), extra, stats: { ...stats } })
    frames.push({
      ...snap(),
      line: 1,
      caption: <>回合 {round}：已提交前缀「{w(committed)}」。草稿模型准备一次起草 γ = {gamma} 个 token。</>,
      panel: { title: '—' },
    })
    for (let i = 0; i < gamma; i++) {
      const ctx = [...committed, ...drafts.map((d) => d.tok)]
      const q = draftQ(ctx, quality)
      const x = sampleIdx(q, r())
      drafts.push({ tok: x, status: 'pending' })
      qs.push(q)
      stats.draft++
      frames.push({
        ...snap(),
        line: i === 0 ? 4 : 5,
        caption: (
          <>
            ① 草稿第 {i + 1} 步：在「{w(ctx)}」之后按 q 采样得到 <b>{VOCAB[x]}</b>（q = {q[x].toFixed(3)}）。草稿模型很小，这一步很便宜。
          </>
        ),
        panel: { title: `草稿分布 q（第 ${i + 1} 个位置）`, kind: 'q', p: q, hl: x },
      })
    }
    const ps = Array.from({ length: gamma + 1 }, (_, i) => targetP([...committed, ...drafts.slice(0, i).map((d) => d.tok)]))
    stats.target++
    frames.push({
      ...snap(),
      line: 6,
      caption: (
        <>
          ② target 只做 <b>1 次</b>前向：输入前缀 + {gamma} 个草稿，并行得到 {gamma + 1} 个位置的分布 p₁…p<sub>{gamma + 1}</sub>。decode 受带宽限制，
          这次前向和只算 1 个 token 的耗时几乎一样。
        </>
      ),
      panel: { title: '—' },
    })
    let rejected = false
    for (let i = 0; i < gamma; i++) {
      const x = drafts[i].tok
      const p = ps[i]
      const q = qs[i]
      const a = Math.min(1, p[x] / q[x])
      const u = r()
      if (u < a) {
        drafts[i].status = 'accepted'
        frames.push({
          ...snap(),
          line: 9,
          caption: (
            <>
              ③ 位置 {i + 1}「{VOCAB[x]}」：p = {p[x].toFixed(3)}，q = {q[x].toFixed(3)}，接受概率 min(1, p/q) = <b>{a.toFixed(3)}</b>；
              随机数 u = {u.toFixed(3)} &lt; {a.toFixed(3)} → <b className="text-success">接受</b>。
            </>
          ),
          panel: { title: `位置 ${i + 1}：target p vs 草稿 q`, kind: 'pq', p, q, hl: x },
        })
      } else {
        drafts[i].status = 'rejected'
        for (let k = i + 1; k < gamma; k++) drafts[k].status = 'rejected'
        frames.push({
          ...snap(),
          line: 8,
          caption: (
            <>
              ③ 位置 {i + 1}「{VOCAB[x]}」：接受概率 min(1, p/q) = {p[x].toFixed(3)}/{q[x].toFixed(3)} = <b>{a.toFixed(3)}</b>，u = {u.toFixed(3)} ≥ {a.toFixed(3)} →{' '}
              <b className="text-danger">拒绝</b>。它后面的草稿都以它为条件，一并作废。
            </>
          ),
          panel: { title: `位置 ${i + 1}：target p vs 草稿 q`, kind: 'pq', p, q, hl: x },
        })
        const res = p.map((v, j) => Math.max(0, v - q[j]))
        const sum = res.reduce((s, v) => s + v, 0)
        const rn = res.map((v) => v / sum)
        const y = sampleIdx(rn, r())
        const accepted = drafts.slice(0, i).map((d) => d.tok)
        stats.tokens += accepted.length + 1
        frames.push({
          ...snap({ tok: y, kind: 'resample' }),
          line: 11,
          caption: (
            <>
              拒绝后从残差分布 norm(max(0, p − q)) 重采样得到 <b>{VOCAB[y]}</b>：残差只保留「target 比草稿更想要」的 token。本回合产出 {accepted.length} 个接受 + 1 个修正 = <b>{accepted.length + 1}</b> 个 token。
            </>
          ),
          panel: { title: '残差分布 r = norm(max(0, p − q))，对照 p', kind: 'res', p: rn, q: p, hl: y },
        })
        committed = [...committed, ...accepted, y]
        rejected = true
        break
      }
    }
    if (!rejected) {
      const pb = ps[gamma]
      const y = sampleIdx(pb, r())
      stats.tokens += gamma + 1
      frames.push({
        ...snap({ tok: y, kind: 'bonus' }),
        line: 12,
        caption: (
          <>
            ④ {gamma} 个草稿全部接受！target 那次前向顺带算出了 p<sub>{gamma + 1}</sub>，直接从中采样 bonus token <b>{VOCAB[y]}</b>，本回合产出 <b>{gamma + 1}</b> 个 token。
          </>
        ),
        panel: { title: `bonus 位置的 target 分布 p${gamma + 1}`, kind: 'p', p: pb, hl: y },
      })
      committed = [...committed, ...drafts.map((d) => d.tok), y]
    }
  }
  frames.push({
    committed: [...committed],
    stats: { ...stats },
    line: 1,
    drafts: [],
    extra: undefined,
    caption: (
      <>
        {ROUNDS} 个回合共产出 <b>{stats.tokens}</b> 个 token，target 只前向了 <b>{stats.target}</b> 次（普通解码要 {stats.tokens} 次），草稿模型前向 {stats.draft} 次。调低草稿质量或调大 γ，再看看比值怎么变。
      </>
    ),
    panel: { title: '—' },
  })
  return frames
}

export function SpecDecodeStepViz() {
  const [gamma, setGamma] = useState(4)
  const [quality, setQuality] = useState(0.6)
  const [seed, setSeed] = useState(5)
  const frames = useMemo(() => buildFrames(gamma, quality, seed), [gamma, quality, seed])
  const st = useStepper(frames, { intervalMs: 1300 })
  const f = st.frame

  const tokens: TokenItem[] = [
    ...f.committed.map((t, i) => ({ text: VOCAB[t], sub: i === 0 ? 'prompt' : undefined })),
    ...f.drafts.map((d, i) => ({
      text: VOCAB[d.tok],
      dashed: d.status === 'pending',
      tone: d.status === 'accepted' ? ('success' as const) : d.status === 'rejected' ? ('danger' as const) : ('info' as const),
      dim: d.status === 'rejected' && f.extra != null,
      sub: `d${i + 1}`,
    })),
    ...(f.extra ? [{ text: VOCAB[f.extra.tok], tone: f.extra.kind === 'bonus' ? ('gold' as const) : ('warning' as const), sub: f.extra.kind === 'bonus' ? 'bonus' : '修正' }] : []),
  ]
  const pn = f.panel
  const ratio = pn.kind === 'pq' && pn.p && pn.q && pn.hl != null ? Math.min(1, pn.p[pn.hl] / pn.q[pn.hl]) : null

  return (
    <VizFrame
      title="草稿 → 验证：一次前向产出多个 token"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '待验证草稿（虚线）', tone: 'info', variant: 'soft' },
            { label: '接受', tone: 'success', variant: 'soft' },
            { label: '拒绝 / 作废', tone: 'danger', variant: 'soft' },
            { label: '残差重采样', tone: 'warning', variant: 'soft' },
            { label: 'bonus', tone: 'gold', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Slider className="min-w-[150px] flex-1" label="草稿长度 γ" value={gamma} min={1} max={6} onChange={setGamma} />
          <Slider className="min-w-[180px] flex-1" label="草稿质量（1 = 完美）" value={quality} min={0} max={1} step={0.05} onChange={setQuality} format={(v) => v.toFixed(2)} />
          <Button size="sm" variant="ghost" icon={<Shuffle size={14} />} onClick={() => setSeed((s) => s + 1)}>
            换一组随机数
          </Button>
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
        <div className="min-w-0 space-y-3">
          <div className="min-h-[64px] rounded-xl border border-border bg-surface p-3">
            <TokenRow tokens={tokens} />
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-xs text-muted">
            <span>
              已产出 <b className="text-fg">{f.stats.tokens}</b> token
            </span>
            <span>
              target 前向 <b className="text-fg">{f.stats.target}</b> 次
            </span>
            <span>
              draft 前向 <b className="text-fg">{f.stats.draft}</b> 次
            </span>
            {f.stats.target > 0 && (
              <span>
                每次 target 前向 ≈ <b className="text-accent">{(f.stats.tokens / f.stats.target).toFixed(2)}</b> token
              </span>
            )}
          </div>
          <CodeTrace code={CODE} active={f.line} maxHeight={300} />
        </div>
        <div className="min-w-0 rounded-xl border border-border bg-surface p-3">
          {pn.p && <div className="mb-2 text-xs font-medium text-muted">{pn.title}</div>}
          {pn.p ? (
            <>
              <PQBars
                labels={VOCAB}
                p={pn.p}
                q={pn.q}
                highlight={pn.hl}
                pLabel={pn.kind === 'q' ? 'q' : pn.kind === 'res' ? 'r' : 'p'}
                qLabel={pn.kind === 'res' ? 'p' : 'q'}
                pTone={pn.kind === 'q' ? 'info' : pn.kind === 'res' ? 'warning' : 'accent'}
                qTone={pn.kind === 'res' ? 'accent' : 'info'}
              />
              {ratio != null && (
                <div className="mt-2 text-[12px] text-muted">
                  高亮行：接受概率 <C>min(1, p/q) = {ratio.toFixed(3)}</C>
                  {ratio >= 1 ? '（p ≥ q：草稿没有高估，必然接受）' : '（p < q：草稿高估了它，只按比例保留）'}
                </div>
              )}
            </>
          ) : (
            <div className="flex h-16 items-center justify-center text-center text-[13px] text-subtle @3xl:h-40">播放或单步执行，这里会显示当前位置的分布</div>
          )}
        </div>
      </div>
    </VizFrame>
  )
}
