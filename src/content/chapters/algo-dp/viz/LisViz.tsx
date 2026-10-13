/**
 * LIS 两种解法：
 *  - LisN2Viz：O(n²)，dp[i] = 以 nums[i] 结尾的 LIS 长度，箭头指向所有可接的 j
 *  - TailsViz：O(n log n)，tails[k] = 长度 k+1 的递增子序列的最小结尾，每个元素在 tails 上做一次 lower_bound
 */
import { useMemo, useState } from 'react'
import { ArrayView, CodeTrace, Legend, StepControls, VizFrame, useStepper, type ArrayPointer } from '@/viz'
import { Segmented } from '@/components/ui'
import { DpGrid, type DpArrow } from './DpGrid'
import { C, Sub } from './common'

/* ------------------------------------------------------------------ */
/* O(n²)                                                               */
/* ------------------------------------------------------------------ */

const N2_NUMS = [10, 9, 2, 5, 3, 7, 101, 18]
const N2_CODE = `def length_of_lis(nums):
    n = len(nums)
    dp = [1] * n              # dp[i]: 以 nums[i] 结尾的 LIS 长度
    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i]:          # nums[i] 能接在 nums[j] 后面
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)            # 答案不一定以最后一个元素结尾！`

interface N2Frame {
  dp: (number | null)[]
  i: number
  valid: number[]
  best: number
  line: number | number[]
  caption: React.ReactNode
  path?: Set<number>
}

function buildN2(): N2Frame[] {
  const a = N2_NUMS
  const n = a.length
  const dp: (number | null)[] = Array(n).fill(null)
  const prev = Array(n).fill(-1)
  const fr: N2Frame[] = []
  for (let i = 0; i < n; i++) {
    const valid: number[] = []
    let v = 1
    let best = -1
    for (let j = 0; j < i; j++)
      if (a[j] < a[i]) {
        valid.push(j)
        if (dp[j]! + 1 > v) {
          v = dp[j]! + 1
          best = j
        }
      }
    dp[i] = v
    prev[i] = best
    fr.push({
      dp: [...dp],
      i,
      valid,
      best,
      line: valid.length ? [6, 7] : 3,
      caption: valid.length ? (
        <>
          <C>nums[{i}] = {a[i]}</C> 可以接在 {valid.map((j) => `${a[j]}`).join('、')} 后面；其中 <C>dp[{best}] = {dp[best]}</C> 最大，所以{' '}
          <C>dp[{i}] = {dp[best]} + 1 = {v}</C>。
        </>
      ) : (
        <>
          <C>nums[{i}] = {a[i]}</C> 前面没有比它小的数，只能自成一个序列：<C>dp[{i}] = 1</C>。
        </>
      ),
    })
  }
  const mx = Math.max(...(dp as number[]))
  let end = (dp as number[]).indexOf(mx)
  const path = new Set<number>()
  while (end !== -1) {
    path.add(end)
    end = prev[end]
  }
  const seq = [...path].sort((x, y) => x - y).map((k) => a[k])
  fr.push({
    dp: [...dp],
    i: -1,
    valid: [],
    best: -1,
    line: 8,
    path,
    caption: (
      <>
        答案是 <C>max(dp) = {mx}</C>（注意不是 <C>dp[n-1]</C>）。沿着每格记录的“前驱”回溯可得一条 LIS：<C>{seq.join(' → ')}</C>。
      </>
    ),
  })
  return fr
}

export function LisN2Viz() {
  const frames = useMemo(buildN2, [])
  const st = useStepper(frames, { intervalMs: 1400 })
  const f = st.frame
  const arrows: DpArrow[] = f.valid.map((j) => ({
    from: [0, j] as [number, number],
    to: [0, f.i] as [number, number],
    tone: j === f.best ? 'success' : 'muted',
    dashed: j !== f.best,
    label: j === f.best ? '+1' : undefined,
    arc: true,
  }))
  return (
    <VizFrame
      title="O(n²) 的 LIS：dp[i] 枚举所有能接上的前驱"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '当前 i', tone: 'accent', variant: 'soft' },
            { label: '取最大值的前驱', tone: 'success', variant: 'line' },
            { label: '其它合法前驱', tone: 'muted', variant: 'line' },
            { label: '回溯出的 LIS', tone: 'success', variant: 'soft' },
          ]}
        />
      }
      controls={<StepControls stepper={st} />}
    >
      <div className="space-y-4">
        <DpGrid
          data={[f.dp, N2_NUMS]}
          rowLabels={['dp', 'nums']}
          colLabels={N2_NUMS.map((_, i) => `${i}`)}
          cellSize={50}
          arrows={arrows}
          ring={(r, c) => (c === f.i ? 'accent' : f.path?.has(c) ? 'success' : r === 1 && f.valid.includes(c) ? (c === f.best ? 'success' : 'muted') : undefined)}
          fill={(_r, c) => (c === f.i ? 'accent' : f.path?.has(c) ? 'success' : undefined)}
        />
        <CodeTrace code={N2_CODE} active={f.line} vars={f.i >= 0 ? { i: f.i, [`dp[${f.i}]`]: f.dp[f.i] ?? '—' } : { 'max(dp)': Math.max(...(f.dp as number[])) }} />
      </div>
    </VizFrame>
  )
}

/* ------------------------------------------------------------------ */
/* O(n log n)：tails + 二分                                            */
/* ------------------------------------------------------------------ */

const T_NUMS = [2, 6, 3, 4, 1, 4, 7, 4, 8, 5]
type Kind = 'strict' | 'nonstrict'

const tailsCode = (k: Kind) => `def length_of_lis(nums):
    tails = []        # tails[k]: 长度为 k+1 的${k === 'strict' ? '递增' : '非降'}子序列的最小结尾
    for x in nums:
        lo, hi = 0, len(tails)      # 找第一个 ${k === 'strict' ? '>= x' : '> x'} 的位置
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] ${k === 'strict' ? '<' : '<='} x:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(tails):
            tails.append(x)         # 比所有结尾都大：LIS 变长
        else:
            tails[lo] = x           # 同样长度，结尾更小 → 更有潜力
    return len(tails)`

interface TFrame {
  xi: number
  tails: number[]
  lo?: number
  hi?: number
  mid?: number
  changed?: number
  appended?: boolean
  line: number | number[]
  caption: React.ReactNode
}

function buildTails(kind: Kind): TFrame[] {
  const tails: number[] = []
  const fr: TFrame[] = []
  const less = (a: number, b: number) => (kind === 'strict' ? a < b : a <= b)
  T_NUMS.forEach((x, xi) => {
    let lo = 0
    let hi = tails.length
    fr.push({ xi, tails: [...tails], lo, hi, line: 4, caption: <>处理 <C>x = {x}</C>：在 tails 上二分，找第一个 {kind === 'strict' ? '≥' : '>'} {x} 的位置，区间 <C>[{lo}, {hi})</C>。</> })
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      const goRight = less(tails[mid], x)
      fr.push({
        xi,
        tails: [...tails],
        lo,
        hi,
        mid,
        line: goRight ? 8 : 10,
        caption: goRight ? (
          <>
            <C>tails[{mid}] = {tails[mid]}</C> {kind === 'strict' ? '<' : '≤'} {x}：答案在右边，<C>lo = {mid + 1}</C>。
          </>
        ) : (
          <>
            <C>tails[{mid}] = {tails[mid]}</C> {kind === 'strict' ? '≥' : '>'} {x}：mid 可能就是答案，<C>hi = {mid}</C>。
          </>
        ),
      })
      if (goRight) lo = mid + 1
      else hi = mid
    }
    if (lo === tails.length) {
      tails.push(x)
      fr.push({ xi, tails: [...tails], lo, changed: lo, appended: true, line: 12, caption: <><C>lo = {lo}</C> 等于 tails 长度：{x} 比所有结尾都大，<b>追加</b>，当前 LIS 长度变为 {tails.length}。</> })
    } else {
      const old = tails[lo]
      tails[lo] = x
      fr.push({
        xi,
        tails: [...tails],
        lo,
        changed: lo,
        line: 14,
        caption: (
          <>
            <b>替换</b> <C>tails[{lo}]: {old} → {x}</C>：长度为 {lo + 1} 的序列现在可以以更小的 {x} 结尾，后面的数更容易接上。长度不变。
          </>
        ),
      })
    }
  })
  fr.push({
    xi: -1,
    tails: [...tails],
    line: 15,
    caption: (
      <>
        结束：LIS 长度 = <C>len(tails) = {tails.length}</C>。注意 tails = <C>[{tails.join(', ')}]</C> <b>不一定是</b>一个真实存在的子序列——它的各项来自不同时刻。
      </>
    ),
  })
  return fr
}

export function TailsViz() {
  const [kind, setKind] = useState<Kind>('strict')
  const frames = useMemo(() => buildTails(kind), [kind])
  const st = useStepper(frames, { intervalMs: 1000 })
  const f = st.frame
  const tp: ArrayPointer[] = []
  if (f.lo != null && f.changed == null) {
    tp.push({ index: f.lo, label: 'lo', tone: 'info' })
    if (f.hi != null) tp.push({ index: f.hi, label: 'hi', tone: 'warning' })
    if (f.mid != null) tp.push({ index: f.mid, label: 'mid', tone: 'accent', side: 'above' })
  }
  const tailsShown = f.tails.length ? f.tails : ['∅']
  return (
    <VizFrame
      title="tails 数组：贪心保持“最小结尾”，每个数只做一次二分"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '当前处理的 x', tone: 'accent', variant: 'soft' },
            { label: '追加（变长）', tone: 'success', variant: 'soft' },
            { label: '替换（结尾变小）', tone: 'warning', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Segmented
            size="sm"
            value={kind}
            onChange={setKind}
            options={[
              { value: 'strict', label: '严格递增（bisect_left）' },
              { value: 'nonstrict', label: '非降（bisect_right）' },
            ]}
          />
        </>
      }
    >
      <div className="space-y-4">
        <div>
          <Sub>nums</Sub>
          <ArrayView
            values={T_NUMS}
            cellSize={40}
            highlights={f.xi >= 0 ? { [f.xi]: 'accent' } : {}}
            dim={(i) => f.xi >= 0 && i > f.xi}
            pointers={f.xi >= 0 ? [{ index: f.xi, label: 'x', tone: 'accent' }] : []}
          />
        </div>
        <div>
          <Sub>tails（长度 = 当前 LIS 长度；下标 k 表示长度 k+1）</Sub>
          <div className="min-h-[110px]">
            <ArrayView
              values={tailsShown}
              cellSize={44}
              fit={false}
              pointers={tp}
              highlights={f.changed != null ? { [f.changed]: f.appended ? 'success' : 'warning' } : f.mid != null ? { [f.mid]: 'accent' } : {}}
              dim={(i) => f.lo != null && f.hi != null && f.changed == null && (i < f.lo || i >= f.hi)}
            />
          </div>
        </div>
        <CodeTrace code={tailsCode(kind)} active={f.line} vars={{ x: f.xi >= 0 ? T_NUMS[f.xi] : '—', lo: f.lo ?? '—', hi: f.hi ?? '—', 'len(tails)': f.tails.length }} />
      </div>
    </VizFrame>
  )
}
