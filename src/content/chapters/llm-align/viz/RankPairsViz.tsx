/**
 * K 个回答的排序 → C(K,2) 个成对比较。InstructGPT：同一 prompt 的全部比较放进一个 batch 元素，
 * 每个回答只前向一次（K 次而不是 2·C(K,2) 次），loss 按 1/C(K,2) 归一化。
 */
import { useState } from 'react'
import { Legend, MatrixView, VizFrame } from '@/viz'
import { Slider } from '@/components/ui'
import { Stat, f3, logSigmoid } from './plot'

// 按人类排序给出（第 0 个最好）的 RM 打分；第 1、2 名被 RM 排反了
const SCORES = [2.1, 1.4, 1.6, 0.2, -0.3, -0.8, -1.1, -1.9, -2.4]
const NAMES = 'ABCDEFGHI'.split('')

export function RankPairsViz() {
  const [K, setK] = useState(4)
  const s = SCORES.slice(0, K)
  const pairs = (K * (K - 1)) / 2
  const losses: (number | null)[][] = s.map((ri, i) => s.map((rj, j) => (i < j ? -logSigmoid(ri - rj) : null)))
  const total = losses.flat().reduce<number>((a, v) => a + (v ?? 0), 0)
  const wrong = s.reduce((n, ri, i) => n + s.filter((rj, j) => i < j && ri < rj).length, 0)

  return (
    <VizFrame
      title="K 路排序：一次前向，C(K,2) 个比较"
      caption={
        <>
          标注员对 K = {K} 个回答排序（A 最好），得到 {pairs} 个成对比较。每个回答出现在 {K - 1} 个比较里：若把比较打散成独立样本，
          一个 epoch 内同一回答要被前向 {K - 1} 次、梯度高度相关，InstructGPT 发现这样训一遍就过拟合；把同一 prompt 的全部比较放进一个 batch 元素，
          只需前向 {K} 次（而不是 {2 * pairs} 次），loss 取 {pairs} 项的平均。红框是 RM 排反的对（{wrong} 个），它们的 loss 最大。
        </>
      }
      legend={
        <Legend
          items={[
            { label: '格子 = −log σ(r_行 − r_列)，行排名高于列', tone: 'accent', variant: 'soft' },
            { label: 'RM 排反（r_行 < r_列）', tone: 'danger', variant: 'soft' },
          ]}
        />
      }
      controls={<Slider className="min-w-[200px] flex-1" label="每个 prompt 的回答数 K（InstructGPT：4–9）" value={K} min={2} max={9} onChange={setK} />}
    >
      <div className="grid items-center gap-4 @3xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <MatrixView
          data={losses}
          rowLabels={NAMES.slice(0, K).map((n, i) => `${n} ${s[i].toFixed(1)}`)}
          colLabels={NAMES.slice(0, K)}
          domain={[0, 1.5]}
          format={(v) => v.toFixed(2)}
          cellSize={42}
          masked={(r, c) => r >= c}
          highlight={(r, c) => (r < c && s[r] < s[c] ? 'danger' : null)}
        />
        <div className="grid grid-cols-2 gap-2">
          <Stat label="成对比较数 C(K,2)" value={pairs} />
          <Stat label="平均 loss" value={f3(total / pairs)} tone="accent" />
          <Stat label="前向次数：打散 → 合并" value={`${2 * pairs} → ${K}`} tone="success" />
          <Stat label="排反的对" value={wrong} tone={wrong ? 'danger' : 'success'} />
        </div>
      </div>
    </VizFrame>
  )
}
