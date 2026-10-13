/**
 * 多头的意义：不同头在各自的 d_h 维子空间里学到不同的「关注模式」。
 * 四个头的 logits 由简单规则生成（前一个 token / 首 token / 指代 / 局部平均），再经因果 softmax 实算。
 */
import { useMemo, useState } from 'react'
import { MatrixView, VizFrame } from '@/viz'
import { Segmented } from '@/components/ui'
import { cn } from '@/lib/utils'
import { softmax } from './math'

const TOKENS = ['<s>', '小猫', '坐在', '垫子', '上', '它', '很', '软']
const T = TOKENS.length
// 指代头：「它」「软」→「垫子」，「坐在」→「小猫」
const COREF: Record<number, number> = { 2: 1, 5: 3, 7: 3, 6: 5, 4: 3 }

const HEADS = [
  { name: '头 0 · 前一个 token', logit: (i: number, j: number) => (i === 0 ? 0 : -3 * Math.abs(i - 1 - j)), note: '位置型：几乎只看 i−1，像一个「移位」算子（归纳头的组成部分）。' },
  { name: '头 1 · 首 token', logit: (_i: number, j: number) => (j === 0 ? 3 : 0), note: '把多余注意力倒给 <s>：即 attention sink 现象，常见于大量真实模型的深层。' },
  { name: '头 2 · 指代 / 句法', logit: (i: number, j: number) => (COREF[i] === j ? 4 : j === i ? 1 : 0), note: '内容型：「它」「软」指向「垫子」，「坐在」找主语「小猫」。' },
  { name: '头 3 · 局部平均', logit: (i: number, j: number) => -0.6 * (i - j), note: '对近邻做指数衰减的平滑，类似局部 n-gram 特征。' },
]

function headWeights(h: number) {
  return Array.from({ length: T }, (_, i) => softmax(Array.from({ length: T }, (_, j) => (j > i ? -Infinity : HEADS[h].logit(i, j)))))
}

export function HeadPatternsViz() {
  const [head, setHead] = useState(2)
  const all = useMemo(() => HEADS.map((_, h) => headWeights(h)), [])

  return (
    <VizFrame
      title="同一句话，不同的头看到不同的关系"
      caption={
        <>
          <b>{HEADS[head].name}</b>：{HEADS[head].note} 每个头只有 <span className="font-mono">d_h = D / h</span> 维，四个头的输出拼接后由{' '}
          <span className="font-mono">W_O</span> 混合。
        </>
      }
      controls={
        <Segmented size="sm" value={head} onChange={setHead} options={HEADS.map((h, i) => ({ value: i, label: h.name.split(' · ')[1] }))} />
      }
    >
      <div className="grid items-start gap-5 @3xl:grid-cols-[1.4fr_1fr]">
        <MatrixView
          data={all[head].map((r) => r.map((v) => v))}
          rowLabels={TOKENS}
          colLabels={TOKENS}
          domain={[0, 1]}
          cellSize={38}
          format={(v) => (v < 0.005 ? '0' : v.toFixed(2).replace(/^0/, ''))}
          masked={(r, c) => c > r}
        />
        <div className="grid grid-cols-2 gap-3">
          {HEADS.map((h, k) => (
            <button
              key={k}
              onClick={() => setHead(k)}
              className={cn('rounded-xl border p-2 text-left transition-colors', k === head ? 'border-accent bg-accent-soft' : 'border-border bg-surface hover:bg-surface-2')}
            >
              <div className="mb-1 truncate text-[11px] font-medium text-muted">{h.name}</div>
              <MatrixView data={all[k]} domain={[0, 1]} showValues={false} cellSize={14} fit={false} masked={(r, c) => c > r} />
            </button>
          ))}
        </div>
      </div>
    </VizFrame>
  )
}
