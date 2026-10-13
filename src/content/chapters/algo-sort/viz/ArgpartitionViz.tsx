/**
 * 大模型岗场景：从 logits 中取 top-k 的下标。
 * argsort 全排序 O(V log V)；argpartition（introselect）期望 O(V)，只保证“前 k 个是 top-k、第 k 个位置正确”，组内无序。
 */
import { useMemo, useState } from 'react'
import { Legend, TokenRow, VizFrame } from '@/viz'
import { Slider } from '@/components/ui'
import { C } from './common'

const TOKENS = ['the', 'a', 'cat', 'dog', 'sat', 'ran', 'on', 'mat', 'big', '<eos>']
const LOGITS = [2.1, 1.3, 0.4, 2.8, -0.5, 1.9, 0.9, 3.2, 0.1, 1.6]

/** 用 Lomuto 快速选择对下标做“按 -logit 的 partition”，返回一种合法的 argpartition 输出（顺序取决于实现） */
function argpartitionDesc(k: number) {
  const idx = LOGITS.map((_, i) => i)
  const key = (i: number) => -LOGITS[i]
  const target = k - 1
  let lo = 0
  let hi = idx.length - 1
  while (lo < hi) {
    const pv = key(idx[hi])
    let s = lo
    for (let j = lo; j < hi; j++) {
      if (key(idx[j]) < pv) {
        ;[idx[s], idx[j]] = [idx[j], idx[s]]
        s++
      }
    }
    ;[idx[s], idx[hi]] = [idx[hi], idx[s]]
    if (s === target) break
    if (s < target) lo = s + 1
    else hi = s - 1
  }
  return idx
}

export function ArgpartitionViz() {
  const [k, setK] = useState(3)
  const part = useMemo(() => argpartitionDesc(k), [k])
  const sorted = useMemo(() => LOGITS.map((_, i) => i).sort((a, b) => LOGITS[b] - LOGITS[a]), [])
  const topSet = new Set(sorted.slice(0, k))

  return (
    <VizFrame
      title="top-k 下标：argpartition 只保证“分组”，不保证组内顺序"
      caption={
        <>
          <C>idx = np.argpartition(-logits, k - 1)[:k]</C>（等价写法 <C>np.argpartition(logits, -k)[-k:]</C>）：前 {k} 个就是 top-{k}，但组内顺序未定义；
          需要有序时再对这 k 个排序：<C>idx[np.argsort(-logits[idx])]</C>，总代价 <C>O(V + k log k)</C>，而 <C>argsort</C> 是 <C>O(V log V)</C>。
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'top-k（集合正确）', tone: 'success', variant: 'soft' },
            { label: '第 k 个位置（值保证正确）', tone: 'gold', variant: 'soft' },
          ]}
        />
      }
      controls={<Slider className="min-w-[180px] flex-1" label="k" value={k} min={1} max={6} onChange={setK} />}
    >
      <div className="space-y-4">
        <div>
          <div className="mb-1.5 text-xs font-medium text-muted">① 原始 logits（下标 0–9）</div>
          <TokenRow tokens={TOKENS.map((t, i) => ({ text: t, sub: `${i}: ${LOGITS[i]}`, tone: topSet.has(i) ? 'success' : undefined }))} />
        </div>
        <div>
          <div className="mb-1.5 text-xs font-medium text-muted">② argpartition 之后的一种可能顺序（期望 O(V)）</div>
          <TokenRow
            tokens={part.map((i, pos) => ({
              text: TOKENS[i],
              sub: LOGITS[i].toFixed(1),
              tone: pos === k - 1 ? 'gold' : pos < k ? 'success' : undefined,
              dim: pos >= k,
            }))}
          />
        </div>
        <div>
          <div className="mb-1.5 text-xs font-medium text-muted">③ argsort 全排序（O(V log V)）</div>
          <TokenRow tokens={sorted.map((i, pos) => ({ text: TOKENS[i], sub: LOGITS[i].toFixed(1), tone: pos < k ? 'success' : undefined, dim: pos >= k }))} />
        </div>
      </div>
    </VizFrame>
  )
}
