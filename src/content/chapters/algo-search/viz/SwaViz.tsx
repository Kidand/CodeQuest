/**
 * 大模型岗场景：滑动窗口注意力（SWA）。
 *  - 掩码：query i 只看 [i-W+1, i]（每层 W 个 token，含自身）
 *  - 感受野：堆叠 k 层后，最后一个 token 能间接看到 k·(W−1)+1 个 token（Mistral 7B 论文称 W×k ≈ 131K）
 *  - Rolling buffer：KV cache 只保留最近 W 个，位置 i 写入槽 i mod W
 */
import { useState } from 'react'
import { ArrayView, Legend, MatrixView, TokenRow, VizFrame, type VizTone } from '@/viz'
import { Slider } from '@/components/ui'
import { C } from './common'

const TOKENS = ['The', 'cat', 'sat', 'on', 'the', 'mat', 'and', 'saw', 'a', 'dog']
const N = TOKENS.length
const Q = N - 1

export function SwaViz() {
  const [W, setW] = useState(3)
  const [k, setK] = useState(2)
  const data = TOKENS.map((_, i) => TOKENS.map((__, j) => (j <= i && i - j < W ? 1 : null)))
  const reach = (j: number) => {
    const d = Q - j
    if (d === 0) return 0
    if (W <= 1) return Infinity
    return Math.ceil(d / (W - 1))
  }
  const field = Math.min(N, k * (W - 1) + 1)
  const tones: VizTone[] = [1, 2, 3, 4]
  const slots = Array.from({ length: W }, (_, s) => {
    // 处理到第 Q 个 token 时，槽 s 里存的是最近 W 个位置中 p mod W == s 的那个
    for (let p = Q; p > Q - W && p >= 0; p--) if (p % W === s) return `t${p}`
    return '—'
  })

  return (
    <VizFrame
      title="滑动窗口注意力：单层只看 W 个，层数叠加扩大感受野"
      caption={
        <>
          W = {W}、k = {k} 层：最后一个 token <C>{TOKENS[Q]}</C> 的感受野 = <C>k·(W−1)+1 = {k * (W - 1) + 1}</C>
          {k * (W - 1) + 1 > N ? `（本例只有 ${N} 个 token）` : ''} 个 token；而 KV cache 每层只需保留最近 <C>W = {W}</C> 个。
          Mistral 7B：W = 4096、32 层，理论感受野约 131K。
        </>
      }
      legend={
        <Legend
          items={[
            { label: '第 1 层直接可见', tone: 1 },
            { label: '第 2 层间接可见', tone: 2 },
            { label: '第 3 / 4 层', tone: 3 },
            { label: 'query 行（最后一个 token）', tone: 'accent', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <Slider className="min-w-[150px] flex-1" label="窗口 W（含自身）" value={W} min={1} max={6} onChange={setW} />
          <Slider className="min-w-[150px] flex-1" label="层数 k" value={k} min={1} max={4} onChange={setK} />
        </>
      }
    >
      <div className="grid items-start gap-5 @3xl:grid-cols-2">
        <div className="min-w-0">
          <div className="mb-1 text-xs text-muted">掩码（行 = query，列 = key，1 = 可见）</div>
          <MatrixView
            data={data}
            rowLabels={TOKENS}
            colLabels={TOKENS.map((_, j) => `t${j}`)}
            scale="none"
            cellSize={30}
            format={() => '1'}
            highlight={(i, j) => (data[i][j] == null ? null : i === Q ? 'accent' : 'info')}
          />
        </div>
        <div className="min-w-0 space-y-5">
          <div>
            <div className="mb-1.5 text-xs font-medium text-muted">
              {TOKENS[Q]} 经过 k 层能看到的 token（下标注：第几层起可见）
            </div>
            <TokenRow
              tokens={TOKENS.map((t, j) => {
                const L = reach(j)
                const ok = L <= k
                return { text: t, tone: L === 0 ? 'accent' : ok ? tones[Math.min(3, L - 1)] : undefined, dim: !ok, sub: L === 0 ? '自身' : ok ? `L${L}` : '不可见' }
              })}
            />
            <div className="mt-2 text-xs text-muted">
              感受野 {field} / {N} 个 token
            </div>
          </div>
          <div>
            <div className="mb-1.5 text-xs font-medium text-muted">Rolling buffer KV cache（W 个槽，位置 p 写入槽 p mod W）</div>
            <ArrayView values={slots} cellSize={44} showIndex />
            <div className="mt-1 text-xs text-muted">更早的位置已被覆盖：缓存大小恒为 W，与序列长度无关。</div>
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
