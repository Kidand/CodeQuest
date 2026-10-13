/**
 * Kadane（LC 53 最大子数组和）与 LC 152 乘积最大子数组。
 * 洞见：状态是「以 i 结尾」；求积时负数会让最小值翻身成最大值，所以要同时维护 max 和 min。
 */
import { useMemo, useState } from 'react'
import { CodeTrace, Legend, StepControls, VizFrame, useStepper } from '@/viz'
import { Segmented } from '@/components/ui'
import { DpGrid, type Cell, type DpArrow } from './DpGrid'
import { C, Stat } from './common'

type Mode = 'sum' | 'prod'
const NUMS: Record<Mode, number[]> = { sum: [-2, 1, -3, 4, -1, 2, 1, -5, 4], prod: [2, -5, -2, -4, 3] }

const CODE: Record<Mode, string> = {
  sum: `def max_sub_array(nums):
    cur = best = nums[0]           # cur: 以 i 结尾的最大和
    for x in nums[1:]:
        cur = max(x, cur + x)      # 重新开始 vs 接在前面
        best = max(best, cur)
    return best`,
  prod: `def max_product(nums):
    hi = lo = best = nums[0]       # 以 i 结尾的最大积 / 最小积
    for x in nums[1:]:
        cands = (x, hi * x, lo * x)
        hi, lo = max(cands), min(cands)   # 同时更新！
        best = max(best, hi)
    return best`,
}

interface Frame {
  i: number
  rows: (number | null)[][] // sum: [cur]；prod: [hi, lo]
  start: number // 当前 cur（或 hi）对应子数组的起点
  best: number
  bestRange: [number, number]
  from: ('restart' | 'extend' | 'hi' | 'lo')[] // 每行的来源
  caption: React.ReactNode
}

function build(mode: Mode): Frame[] {
  const a = NUMS[mode]
  const n = a.length
  const fr: Frame[] = []
  if (mode === 'sum') {
    const cur: (number | null)[] = Array(n).fill(null)
    cur[0] = a[0]
    let start = 0
    let best = a[0]
    let bestRange: [number, number] = [0, 0]
    fr.push({ i: 0, rows: [[...cur]], start, best, bestRange, from: ['restart'], caption: <>初始化 <C>cur = best = nums[0] = {a[0]}</C>。</> })
    for (let i = 1; i < n; i++) {
      const ext = cur[i - 1]! + a[i]
      const restart = a[i] > ext
      cur[i] = restart ? a[i] : ext
      if (restart) start = i
      if (cur[i]! > best) {
        best = cur[i]!
        bestRange = [start, i]
      }
      fr.push({
        i,
        rows: [[...cur]],
        start,
        best,
        bestRange,
        from: [restart ? 'restart' : 'extend'],
        caption: restart ? (
          <>
            前面的和 <C>{cur[i - 1]}</C> 是负担：<C>{cur[i - 1]} + {a[i]} = {ext}</C> &lt; <C>{a[i]}</C>，<b>从 {a[i]} 重新开始</b>。
          </>
        ) : (
          <>
            接在前面：<C>{cur[i - 1]} + {a[i]} = {ext}</C> ≥ <C>{a[i]}</C>，cur = {ext}。best = {best}。
          </>
        ),
      })
    }
  } else {
    const hi: (number | null)[] = Array(n).fill(null)
    const lo: (number | null)[] = Array(n).fill(null)
    hi[0] = lo[0] = a[0]
    let best = a[0]
    let bestRange: [number, number] = [0, 0]
    // 记录 hi/lo 的起点用于标注最佳子数组
    const hs = [0]
    const ls = [0]
    fr.push({ i: 0, rows: [[...hi], [...lo]], start: 0, best, bestRange, from: ['restart', 'restart'], caption: <>初始化 <C>hi = lo = best = {a[0]}</C>。</> })
    for (let i = 1; i < n; i++) {
      const x = a[i]
      const c: [number, 'restart' | 'hi' | 'lo', number][] = [
        [x, 'restart', i],
        [hi[i - 1]! * x, 'hi', hs[i - 1]],
        [lo[i - 1]! * x, 'lo', ls[i - 1]],
      ]
      const mx = c.reduce((p, q) => (q[0] > p[0] ? q : p))
      const mn = c.reduce((p, q) => (q[0] < p[0] ? q : p))
      hi[i] = mx[0]
      lo[i] = mn[0]
      hs[i] = mx[2]
      ls[i] = mn[2]
      if (mx[0] > best) {
        best = mx[0]
        bestRange = [mx[2], i]
      }
      fr.push({
        i,
        rows: [[...hi], [...lo]],
        start: mx[2],
        best,
        bestRange,
        from: [mx[1], mn[1]],
        caption: (
          <>
            候选 <C>x={x}</C>、<C>hi·x={hi[i - 1]! * x}</C>、<C>lo·x={lo[i - 1]! * x}</C> → <C>hi={mx[0]}</C>
            {mx[1] === 'lo' ? <b>（最小值乘负数翻身成最大值！）</b> : ''}，<C>lo={mn[0]}</C>。best = {best}。
          </>
        ),
      })
    }
  }
  const last = fr[fr.length - 1]
  last.caption = (
    <>
      {last.caption} 结束：答案 <C>{last.best}</C>，对应子数组 <C>nums[{last.bestRange[0]}..{last.bestRange[1]}]</C>。
    </>
  )
  return fr
}

export function KadaneViz() {
  const [mode, setMode] = useState<Mode>('sum')
  const frames = useMemo(() => build(mode), [mode])
  const st = useStepper(frames, { intervalMs: 1200 })
  const f = st.frame
  const a = NUMS[mode]
  const data: Cell[][] = [a, ...f.rows]
  const labels = mode === 'sum' ? ['nums', 'cur'] : ['nums', 'hi', 'lo']
  const arrows: DpArrow[] = []
  if (f.i >= 1) {
    f.from.forEach((src, k) => {
      const row = k + 1
      if (src === 'restart') arrows.push({ from: [0, f.i], to: [row, f.i], tone: 'warning', label: '重开' })
      else if (src === 'extend') arrows.push({ from: [row, f.i - 1], to: [row, f.i], tone: 'success' })
      else arrows.push({ from: [src === 'hi' ? 1 : 2, f.i - 1], to: [row, f.i], tone: src === 'lo' && row === 1 ? 'danger' : 'success' })
    })
  }
  const done = st.atEnd
  const inBest = (c: number) => done && c >= f.bestRange[0] && c <= f.bestRange[1]
  const inCur = (c: number) => !done && c >= f.start && c <= f.i
  return (
    <VizFrame
      title={mode === 'sum' ? 'Kadane：以 i 结尾的最大和，负的前缀直接丢掉' : '乘积最大子数组：负数让 min 翻身成 max，两个都要记'}
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '当前「以 i 结尾」的子数组', tone: 'info', variant: 'soft' },
            { label: '接在前面', tone: 'success', variant: 'line' },
            { label: '重新开始', tone: 'warning', variant: 'line' },
            ...(mode === 'prod' ? [{ label: 'min × 负数 → max', tone: 'danger' as const, variant: 'line' as const }] : []),
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
              { value: 'sum', label: 'LC 53 最大和' },
              { value: 'prod', label: 'LC 152 最大积' },
            ]}
          />
        </>
      }
    >
      <div className="space-y-4">
        <div className="flex flex-wrap gap-2">
          <Stat label="best" value={f.best} tone="success" />
          <Stat label="最佳区间" value={`[${f.bestRange[0]}, ${f.bestRange[1]}]`} />
        </div>
        <DpGrid
          data={data}
          rowLabels={labels}
          colLabels={a.map((_, i) => `${i}`)}
          cellSize={50}
          arrows={arrows}
          fill={(r, c) => (r === 0 && inBest(c) ? 'success' : r === 0 && inCur(c) ? 'info' : r > 0 && c === f.i && !done ? 'accent' : undefined)}
          ring={(r, c) => (r > 0 && c === f.i && !done ? 'accent' : undefined)}
        />
        <CodeTrace code={CODE[mode]} active={f.i === 0 ? 2 : mode === 'sum' ? [4, 5] : [4, 5, 6]} vars={{ i: f.i, x: a[f.i], best: f.best }} />
      </div>
    </VizFrame>
  )
}
