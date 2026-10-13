/**
 * 因果掩码 / padding 掩码 / 二者叠加；左 padding + 因果时，pad query 行整行被遮 → softmax 得到 NaN。
 */
import { useMemo, useState } from 'react'
import { Legend, MatrixView, TokenRow, VizFrame } from '@/viz'
import { Segmented } from '@/components/ui'
import { randMatrix, softmax } from './math'

type Mode = 'none' | 'causal' | 'pad' | 'both'
type Side = 'right' | 'left'

const REAL = ['<s>', '我', '爱', '你']
const N = 6
const LOGITS = randMatrix(N, N, 11, 1.2)

export function MaskViz() {
  const [mode, setMode] = useState<Mode>('both')
  const [side, setSide] = useState<Side>('left')
  const tokens = side === 'right' ? [...REAL, '<pad>', '<pad>'] : ['<pad>', '<pad>', ...REAL]
  const isPad = (j: number) => tokens[j] === '<pad>'
  const blocked = (i: number, j: number) => ((mode === 'causal' || mode === 'both') && j > i) || ((mode === 'pad' || mode === 'both') && isPad(j))

  const weights = useMemo(
    () => LOGITS.map((row, i) => softmax(row.map((s, j) => (blocked(i, j) ? -Infinity : s)))),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [mode, side],
  )
  const nanRows = weights.map((r, i) => (Number.isNaN(r[0]) ? i : -1)).filter((i) => i >= 0)

  return (
    <VizFrame
      title="掩码：softmax 之前把禁看位置设为 −∞"
      caption={
        nanRows.length ? (
          <>
            ⚠️ 第 {nanRows.join('、')} 行的 query 是 <span className="font-mono">&lt;pad&gt;</span>，因果掩码只允许它看左边——而左边也全是 pad，于是整行都是 −∞，
            <span className="font-mono">softmax = 0/0 = NaN</span>。NaN 会经过 <span className="font-mono">P @ V</span> 的「0 × NaN」传染给下一层的所有行。
          </>
        ) : mode === 'none' ? (
          <>不加掩码：每个 token 都能看到全部 6 个位置，包括未来 token 与 pad——训练语言模型时这等于「偷看答案」。</>
        ) : (
          <>
            斜线格 = 被遮位置（加 −∞，softmax 后权重恰为 0）。每一行在未被遮的位置上重新归一化，和仍为 1。
            {mode !== 'causal' && side === 'right' && ' 右 padding 时 pad 行仍能看到真实 token，输出有限，最后被 loss mask 丢掉即可。'}
          </>
        )
      }
      legend={
        <Legend
          items={[
            { label: '注意力权重', tone: 'accent', variant: 'soft' },
            { label: '被遮（−∞）', tone: 'muted', variant: 'soft' },
            { label: '整行 NaN', tone: 'danger', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <Segmented
            size="sm"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'none', label: '无掩码' },
              { value: 'causal', label: '因果' },
              { value: 'pad', label: 'padding' },
              { value: 'both', label: '因果 + padding' },
            ]}
          />
          <Segmented
            size="sm"
            value={side}
            onChange={setSide}
            options={[
              { value: 'right', label: '右 padding' },
              { value: 'left', label: '左 padding' },
            ]}
          />
        </>
      }
    >
      <div className="space-y-4">
        <TokenRow tokens={tokens.map((t, i) => ({ text: t, sub: `pos ${i}`, dashed: isPad(i), dim: isPad(i), tone: isPad(i) ? undefined : 'info' }))} />
        <MatrixView
          data={weights}
          rowLabels={tokens.map((t, i) => `${t} ${i}`)}
          colLabels={tokens.map((_, i) => `k${i}`)}
          domain={[0, 1]}
          cellSize={46}
          format={(v) => (Number.isNaN(v) ? 'NaN' : v.toFixed(2))}
          masked={(r, c) => !nanRows.includes(r) && blocked(r, c)}
          highlight={(r) => (nanRows.includes(r) ? 'danger' : null)}
        />
      </div>
    </VizFrame>
  )
}
