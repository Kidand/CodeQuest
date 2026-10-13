/**
 * LC 198 打家劫舍：一维 DP 的「选 / 不选」转移 + 滚动变量。
 * 洞见：dp[i] 只依赖 dp[i-1]、dp[i-2]，所以整张表只有两个格子是「活」的。
 */
import { useMemo, useState } from 'react'
import { CodeTrace, Legend, StepControls, VizFrame, useStepper } from '@/viz'
import { Segmented } from '@/components/ui'
import { DpGrid, type Cell, type DpArrow } from './DpGrid'
import { C } from './common'

const NUMS = [2, 7, 9, 3, 1, 5, 8, 4]
type Mode = 'array' | 'rolling'

const CODE: Record<Mode, string> = {
  array: `def rob(nums):
    n = len(nums)
    dp = [0] * (n + 1)        # dp[i]: 前 i 间房能偷到的最大金额
    dp[1] = nums[0]
    for i in range(2, n + 1):
        dp[i] = max(dp[i - 1],                # 不偷第 i 间
                    dp[i - 2] + nums[i - 1])  # 偷第 i 间
    return dp[n]`,
  rolling: `def rob(nums):
    prev2, prev1 = 0, 0       # dp[i-2], dp[i-1]
    for x in nums:
        cur = max(prev1,      # 不偷
                  prev2 + x)  # 偷
        prev2, prev1 = prev1, cur
    return prev1`,
}

interface Frame {
  dp: (number | null)[]
  i: number
  take?: boolean
  line: number
  caption: React.ReactNode
  chosen?: Set<number>
}

function build(): Frame[] {
  const n = NUMS.length
  const dp: (number | null)[] = Array(n + 1).fill(null)
  const fr: Frame[] = []
  dp[0] = 0
  fr.push({ dp: [...dp], i: 0, line: 3, caption: <>初始化 <C>dp[0] = 0</C>：一间房都没有，金额为 0。</> })
  dp[1] = NUMS[0]
  fr.push({ dp: [...dp], i: 1, line: 4, caption: <>只有第 1 间房时只能偷它：<C>dp[1] = nums[0] = {NUMS[0]}</C>。</> })
  for (let i = 2; i <= n; i++) {
    const skip = dp[i - 1]!
    const take = dp[i - 2]! + NUMS[i - 1]
    dp[i] = Math.max(skip, take)
    fr.push({
      dp: [...dp],
      i,
      take: take > skip,
      line: 6,
      caption: (
        <>
          第 {i} 间（{NUMS[i - 1]} 元）：不偷 → <C>dp[{i - 1}] = {skip}</C>；偷 → <C>dp[{i - 2}] + {NUMS[i - 1]} = {take}</C>。
          取较大者 <C>dp[{i}] = {dp[i]}</C>（{take > skip ? '偷' : '不偷'}）。
        </>
      ),
    })
  }
  // 回溯：dp[i] != dp[i-1] 说明偷了第 i 间
  const chosen = new Set<number>()
  let i = n
  while (i >= 1) {
    if (i === 1 || dp[i] !== dp[i - 1]) {
      chosen.add(i)
      i -= 2
    } else i -= 1
  }
  fr.push({
    dp: [...dp],
    i: n,
    line: 8,
    chosen,
    caption: (
      <>
        答案 <C>dp[{n}] = {dp[n]}</C>。回溯：若 <C>dp[i] ≠ dp[i-1]</C> 说明第 i 间被偷，跳到 i-2；否则跳到 i-1。偷的是第{' '}
        {[...chosen].sort((a, b) => a - b).join('、')} 间：{[...chosen].sort((a, b) => a - b).map((k) => NUMS[k - 1]).join(' + ')} = {dp[n]}。
      </>
    ),
  })
  return fr
}

export function RobberViz() {
  const frames = useMemo(build, [])
  const [mode, setMode] = useState<Mode>('array')
  const st = useStepper(frames, { intervalMs: 1300 })
  const f = st.frame
  const n = NUMS.length
  const data: Cell[][] = [f.dp, ['—', ...NUMS]]
  const computing = f.i >= 2 && !f.chosen
  const arrows: DpArrow[] = computing
    ? [
        { from: [0, f.i - 1], to: [0, f.i], tone: f.take ? 'muted' : 'success', label: '不偷', dashed: f.take },
        { from: [0, f.i - 2], to: [0, f.i], tone: f.take ? 'success' : 'muted', label: '偷', dashed: !f.take },
        { from: [1, f.i], to: [0, f.i], tone: f.take ? 'success' : 'muted', dashed: !f.take },
      ]
    : []
  // 滚动模式下只有 dp[i-2], dp[i-1]（以及正在算的 dp[i]）是活的
  const live = (c: number) => c >= f.i - 2 && c <= f.i
  const line = mode === 'array' ? f.line : f.chosen ? 7 : f.i <= 1 ? 2 : 4
  const rollVars =
    f.i >= 2 && !f.chosen
      ? { prev2: f.dp[f.i - 2] ?? 0, prev1: f.dp[f.i - 1] ?? 0, cur: f.dp[f.i] ?? 0 }
      : { prev2: f.i >= 1 ? (f.dp[f.i - 1] ?? 0) : 0, prev1: f.dp[f.i] ?? 0 }

  return (
    <VizFrame
      title="打家劫舍：dp[i] 只看前两格，「偷 / 不偷」二选一"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '当前格 dp[i]', tone: 'accent', variant: 'soft' },
            { label: '被选中的转移', tone: 'success', variant: 'line' },
            { label: '落选的转移', tone: 'muted', variant: 'line' },
            ...(mode === 'rolling' ? [{ label: '变暗 = 滚动后已丢弃', tone: 'muted' as const, variant: 'soft' as const }] : []),
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Segmented
            size="sm"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'array', label: 'dp 数组 O(n) 空间' },
              { value: 'rolling', label: '两个变量 O(1) 空间' },
            ]}
          />
        </>
      }
    >
      <div className="space-y-4">
        <DpGrid
          data={data}
          rowLabels={['dp', 'nums']}
          colLabels={Array.from({ length: n + 1 }, (_, i) => `i=${i}`)}
          cellSize={50}
          arrows={arrows}
          ring={(r, c) => (r === 0 && c === f.i && !f.chosen ? 'accent' : r === 1 && f.chosen?.has(c) ? 'success' : undefined)}
          fill={(r, c) => (r === 1 && f.chosen?.has(c) ? 'success' : r === 0 && c === f.i && !f.chosen ? 'accent' : undefined)}
          dim={(r, c) => mode === 'rolling' && r === 0 && !f.chosen && f.dp[c] != null && !live(c)}
        />
        <CodeTrace
          code={CODE[mode]}
          active={mode === 'array' ? (f.line === 6 ? [6, 7] : f.line) : line === 4 ? [4, 5, 6] : line}
          vars={
            mode === 'array'
              ? { i: f.i, [`dp[${f.i}]`]: f.dp[f.i] ?? '—' }
              : rollVars
          }
        />
      </div>
    </VizFrame>
  )
}
