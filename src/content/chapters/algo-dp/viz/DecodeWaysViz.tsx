/**
 * LC 91 解码方法：带条件的转移。
 * 洞见：dp[i] 的两个来源（单字符 / 两字符）各自有合法性条件，'0' 只能作为 "10"/"20" 的后半。
 */
import { useMemo, useState } from 'react'
import { CodeTrace, Legend, StepControls, VizFrame, useStepper } from '@/viz'
import { Segmented } from '@/components/ui'
import { DpGrid, type DpArrow } from './DpGrid'
import { C } from './common'

const PRESETS = ['226', '11106', '2101', '1224', '06']
const CODE = `def num_decodings(s):
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1                    # 空前缀：1 种（乘法单位元）
    for i in range(1, n + 1):
        if s[i-1] != '0':        # 最后一段是 1 个字符
            dp[i] += dp[i-1]
        if i >= 2 and '10' <= s[i-2:i] <= '26':   # 最后一段是 2 个字符
            dp[i] += dp[i-2]
    return dp[n]`

interface Frame {
  dp: (number | null)[]
  i: number
  one?: boolean
  two?: boolean | null
  line: number | number[]
  caption: React.ReactNode
}

function build(s: string): Frame[] {
  const n = s.length
  const dp: (number | null)[] = Array(n + 1).fill(null)
  dp[0] = 1
  const fr: Frame[] = [{ dp: [...dp], i: 0, line: 4, caption: <>初始化 <C>dp[0] = 1</C>：空前缀有 1 种解码方式（什么都不选）。它不是“答案”，而是让转移公式成立的边界。</> }]
  for (let i = 1; i <= n; i++) {
    const ch = s[i - 1]
    const one = ch !== '0'
    const pair = i >= 2 ? s.slice(i - 2, i) : null
    const two = pair == null ? null : pair >= '10' && pair <= '26'
    const v = (one ? dp[i - 1]! : 0) + (two ? dp[i - 2]! : 0)
    dp[i] = v
    const parts: React.ReactNode[] = []
    parts.push(one ? <>单字符 <C>'{ch}'</C> 合法 → 加 <C>dp[{i - 1}]={dp[i - 1]}</C></> : <>单字符 <C>'0'</C> 不能单独解码</>)
    if (pair != null) parts.push(two ? <>两字符 <C>"{pair}"</C>∈[10,26] → 加 <C>dp[{i - 2}]={dp[i - 2]}</C></> : <>两字符 <C>"{pair}"</C> 不在 [10,26]{pair[0] === '0' ? '（前导 0）' : ''}</>)
    fr.push({
      dp: [...dp],
      i,
      one,
      two,
      line: one && two ? [7, 9] : one ? 7 : two ? 9 : [6, 8],
      caption: (
        <>
          i={i}：{parts[0]}；{parts[1] ?? '没有前一位'}。<C>dp[{i}] = {v}</C>
          {v === 0 ? <b className="text-danger">（为 0：之后全部为 0，整串无法解码）</b> : null}
        </>
      ),
    })
  }
  fr.push({ dp: [...dp], i: n, line: 10, caption: <>答案 <C>dp[{n}] = {dp[n]}</C>。</> })
  return fr
}

export function DecodeWaysViz() {
  const [s, setS] = useState(PRESETS[1])
  const frames = useMemo(() => build(s), [s])
  const st = useStepper(frames, { intervalMs: 1300 })
  const f = st.frame
  const n = s.length
  const active = f.i >= 1 && f.one !== undefined
  const arrows: DpArrow[] = []
  if (active) {
    arrows.push({ from: [0, f.i - 1], to: [0, f.i], tone: f.one ? 'success' : 'danger', dashed: !f.one, label: f.one ? `'${s[f.i - 1]}'` : '✗' })
    if (f.two != null) arrows.push({ from: [0, f.i - 2], to: [0, f.i], tone: f.two ? 'success' : 'danger', dashed: !f.two, label: f.two ? `"${s.slice(f.i - 2, f.i)}"` : '✗' })
  }
  return (
    <VizFrame
      title="解码方法：两条转移各有合法条件，'0' 是全部难点"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '合法转移（累加）', tone: 'success', variant: 'line' },
            { label: '非法转移（跳过）', tone: 'danger', variant: 'line' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Segmented size="sm" value={s} onChange={setS} options={PRESETS.map((p) => ({ value: p, label: `"${p}"` }))} />
        </>
      }
    >
      <div className="space-y-4">
        <DpGrid
          data={[f.dp, ['∅', ...s.split('')]]}
          rowLabels={['dp', 's']}
          colLabels={Array.from({ length: n + 1 }, (_, i) => `${i}`)}
          cellSize={52}
          arrows={arrows}
          ring={(r, c) => (r === 0 && c === f.i && active ? 'accent' : r === 1 && active && (c === f.i || (c === f.i - 1 && f.two)) ? 'accent' : undefined)}
          fill={(r, c) => (r === 0 && c === f.i && active ? 'accent' : r === 0 && f.dp[c] === 0 ? 'danger' : undefined)}
        />
        <CodeTrace code={CODE} active={f.line} vars={{ i: f.i, [`dp[${f.i}]`]: f.dp[f.i] ?? '—' }} />
      </div>
    </VizFrame>
  )
}
