/**
 * Top-K：维护一个大小为 k 的**小顶堆**，堆顶是当前第 k 大，也是新元素要超过的“门槛”。
 * 场景：从一串 token 的 logits 中选出 top-k（top-k 采样的候选集）。
 */
import { useMemo, useState } from 'react'
import { Legend, StepControls, TokenRow, VizFrame, useStepper, type VizTone } from '@/viz'
import { Slider } from '@/components/ui'
import { HeapDual } from './HeapDual'
import { C } from './common'

const TOKENS = ['the', 'a', 'cat', 'dog', 'sat', 'ran', 'on', 'mat', 'big', '<eos>']
const LOGITS = [2.1, 1.3, 0.4, 2.8, -0.5, 1.9, 0.9, 3.2, 0.1, 1.6]

interface Frame {
  heap: number[] // 存 token 下标；按 logit 构成小顶堆
  ids: number[]
  t: number
  action?: 'push' | 'replace' | 'skip'
  caption: React.ReactNode
}

function siftUp(h: number[], ids: number[], i: number) {
  while (i > 0) {
    const p = (i - 1) >> 1
    if (LOGITS[h[p]] <= LOGITS[h[i]]) break
    ;[h[p], h[i]] = [h[i], h[p]]
    ;[ids[p], ids[i]] = [ids[i], ids[p]]
    i = p
  }
}
function siftDown(h: number[], ids: number[], i: number) {
  const n = h.length
  while (2 * i + 1 < n) {
    let c = 2 * i + 1
    if (c + 1 < n && LOGITS[h[c + 1]] < LOGITS[h[c]]) c++
    if (LOGITS[h[i]] <= LOGITS[h[c]]) break
    ;[h[i], h[c]] = [h[c], h[i]]
    ;[ids[i], ids[c]] = [ids[c], ids[i]]
    i = c
  }
}

function build(k: number): Frame[] {
  const h: number[] = []
  const ids: number[] = []
  const fr: Frame[] = [{ heap: [], ids: [], t: -1, caption: <>依次读入 {TOKENS.length} 个 token 的 logit，维护大小为 <C>k = {k}</C> 的小顶堆。</> }]
  TOKENS.forEach((tok, t) => {
    const x = LOGITS[t]
    if (h.length < k) {
      h.push(t)
      ids.push(t)
      siftUp(h, ids, h.length - 1)
      fr.push({ heap: [...h], ids: [...ids], t, action: 'push', caption: <>堆未满，直接 push <C>{tok} ({x})</C>。</> })
    } else if (x > LOGITS[h[0]]) {
      const out = h[0]
      h[0] = t
      ids[0] = t
      siftDown(h, ids, 0)
      fr.push({ heap: [...h], ids: [...ids], t, action: 'replace', caption: <><C>{tok} ({x})</C> &gt; 堆顶 <C>{TOKENS[out]} ({LOGITS[out]})</C>：替换堆顶并下沉（<C>heapreplace</C>），{TOKENS[out]} 被淘汰。</> })
    } else {
      fr.push({ heap: [...h], ids: [...ids], t, action: 'skip', caption: <><C>{tok} ({x})</C> ≤ 堆顶 <C>{TOKENS[h[0]]} ({LOGITS[h[0]]})</C>：连当前第 {k} 大都比不过，O(1) 跳过。</> })
    }
  })
  const top = [...h].sort((a, b) => LOGITS[b] - LOGITS[a])
  fr.push({ heap: [...h], ids: [...ids], t: TOKENS.length, caption: <>结束：top-{k} = <C>{top.map((i) => TOKENS[i]).join(', ')}</C>（堆内无序，需要排序时再对 k 个元素排序）。总复杂度 <C>O(n log k)</C>，额外空间 <C>O(k)</C>。</> })
  return fr
}

export function TopKStreamViz() {
  const [k, setK] = useState(3)
  const frames = useMemo(() => build(k), [k])
  const st = useStepper(frames, { intervalMs: 1000 })
  const f = st.frame
  const inHeap = new Set(f.heap)
  const tones: Record<number, VizTone> = { 0: 'gold' }

  return (
    <VizFrame
      title="Top-K：大小为 k 的小顶堆，堆顶就是门槛"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '当前读入', tone: 'accent', variant: 'soft' },
            { label: '在堆中（候选 top-k）', tone: 'success', variant: 'soft' },
            { label: '堆顶 = 当前第 k 大', tone: 'gold', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Slider className="min-w-[160px] flex-1" label="k" value={k} min={1} max={6} onChange={setK} />
        </>
      }
    >
      <div className="space-y-4">
        <div>
          <div className="mb-1.5 text-xs font-medium text-muted">token 流（下标注 logit）</div>
          <TokenRow
            tokens={TOKENS.map((tok, t) => ({
              text: tok,
              sub: LOGITS[t].toFixed(1),
              tone: t === f.t ? 'accent' : inHeap.has(t) ? (f.heap[0] === t ? 'gold' : 'success') : undefined,
              dim: t > f.t,
            }))}
          />
        </div>
        <div>
          <div className="mb-1.5 text-xs font-medium text-muted">小顶堆（按 logit；数组 ↔ 树）</div>
          <HeapDual vals={f.heap.map((t) => LOGITS[t])} ids={f.ids} tones={f.heap.length ? tones : {}} nodeSize={38} />
        </div>
      </div>
    </VizFrame>
  )
}
