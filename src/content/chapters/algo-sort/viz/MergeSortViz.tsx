/**
 * 归并排序：自顶向下（递归拆分 → 回溯合并）与自底向上（宽度 1, 2, 4… 逐轮合并）。
 * 上方的区间条是当前递归栈（每行一层）；ids 让合并时元素的移动有动画。
 */
import { useMemo, useState } from 'react'
import { ArrayView, CodeTrace, Legend, StepControls, VizFrame, useStepper, type ArrayRange, type VizTone } from '@/viz'
import { Segmented } from '@/components/ui'
import { C } from './common'

type Mode = 'top' | 'bottom'
const DATA = [38, 27, 43, 3, 9, 82, 10, 19]

const CODE: Record<Mode, string> = {
  top: `def merge_sort(a, lo, hi):          # 排序 a[lo..hi]
    if lo >= hi:
        return                          # 单个元素天然有序
    mid = (lo + hi) // 2
    merge_sort(a, lo, mid)              # 拆分：左半
    merge_sort(a, mid + 1, hi)          # 拆分：右半
    merge(a, lo, mid, hi)               # 合并两个有序段`,
  bottom: `def merge_sort(a):
    n, width = len(a), 1
    while width < n:                    # 宽度 1, 2, 4, ...
        for lo in range(0, n, 2 * width):
            mid = min(lo + width, n) - 1
            hi = min(lo + 2 * width, n) - 1
            if mid < hi:
                merge(a, lo, mid, hi)   # 合并相邻两段
        width *= 2`,
}

interface Frame {
  vals: number[]
  ids: number[]
  stack: [number, number][]
  left?: [number, number]
  right?: [number, number]
  merged?: [number, number]
  line: number
  caption: React.ReactNode
}

function mergeInto(vals: number[], ids: number[], lo: number, mid: number, hi: number) {
  const L = vals.slice(lo, mid + 1).map((v, k) => [v, ids[lo + k]] as const)
  const R = vals.slice(mid + 1, hi + 1).map((v, k) => [v, ids[mid + 1 + k]] as const)
  let i = 0
  let j = 0
  let k = lo
  while (i < L.length && j < R.length) {
    const take = L[i][0] <= R[j][0] ? L[i++] : R[j++]
    vals[k] = take[0]
    ids[k++] = take[1]
  }
  for (const t of [...L.slice(i), ...R.slice(j)]) {
    vals[k] = t[0]
    ids[k++] = t[1]
  }
}

function buildTop(): Frame[] {
  const vals = [...DATA]
  const ids = DATA.map((_, k) => k)
  const fr: Frame[] = []
  const stack: [number, number][] = []
  const rec = (lo: number, hi: number) => {
    stack.push([lo, hi])
    if (lo >= hi) {
      fr.push({ vals: [...vals], ids: [...ids], stack: [...stack], line: 3, caption: <>区间 <C>[{lo}]</C> 只有一个元素，直接返回。</> })
      stack.pop()
      return
    }
    const mid = (lo + hi) >> 1
    fr.push({ vals: [...vals], ids: [...ids], stack: [...stack], line: 5, caption: <>拆分 <C>[{lo}, {hi}]</C> → <C>[{lo}, {mid}]</C> 与 <C>[{mid + 1}, {hi}]</C>（第 {stack.length} 层）。</> })
    rec(lo, mid)
    rec(mid + 1, hi)
    fr.push({ vals: [...vals], ids: [...ids], stack: [...stack], left: [lo, mid], right: [mid + 1, hi], line: 7, caption: <>两半已各自有序：<C>[{vals.slice(lo, mid + 1).join(', ')}]</C> 与 <C>[{vals.slice(mid + 1, hi + 1).join(', ')}]</C>，合并。</> })
    mergeInto(vals, ids, lo, mid, hi)
    fr.push({ vals: [...vals], ids: [...ids], stack: [...stack], merged: [lo, hi], line: 7, caption: <>合并得到 <C>[{vals.slice(lo, hi + 1).join(', ')}]</C>，共 {hi - lo + 1} 个元素，线性时间。</> })
    stack.pop()
  }
  rec(0, vals.length - 1)
  fr.push({ vals: [...vals], ids: [...ids], stack: [], merged: [0, vals.length - 1], line: 7, caption: <>完成。递归深度 <C>⌈log₂n⌉ = 3</C>，每层合并总共 O(n)：<C>O(n log n)</C>，任何输入都一样。</> })
  return fr
}

function buildBottom(): Frame[] {
  const vals = [...DATA]
  const ids = DATA.map((_, k) => k)
  const n = vals.length
  const fr: Frame[] = [{ vals: [...vals], ids: [...ids], stack: [], line: 2, caption: <>不递归：把每个元素看成长度为 1 的有序段，逐轮把相邻两段合并，段长 1 → 2 → 4 → 8。</> }]
  for (let width = 1; width < n; width *= 2) {
    for (let lo = 0; lo < n; lo += 2 * width) {
      const mid = Math.min(lo + width, n) - 1
      const hi = Math.min(lo + 2 * width, n) - 1
      if (mid >= hi) continue
      fr.push({ vals: [...vals], ids: [...ids], stack: [], left: [lo, mid], right: [mid + 1, hi], line: 8, caption: <>宽度 {width}：合并 <C>[{lo}, {mid}]</C> 与 <C>[{mid + 1}, {hi}]</C>。</> })
      mergeInto(vals, ids, lo, mid, hi)
      fr.push({ vals: [...vals], ids: [...ids], stack: [], merged: [lo, hi], line: 8, caption: <>得到有序段 <C>[{vals.slice(lo, hi + 1).join(', ')}]</C>。</> })
    }
  }
  fr.push({ vals: [...vals], ids: [...ids], stack: [], merged: [0, n - 1], line: 9, caption: <>共 {Math.ceil(Math.log2(n))} 轮，每轮 O(n)。自底向上不需要递归栈——对<b>链表</b>可以做到 O(1) 额外空间（LC148 的进阶要求）。</> })
  return fr
}

export function MergeSortViz({ initialMode = 'top' }: { initialMode?: Mode }) {
  const [mode, setMode] = useState<Mode>(initialMode)
  const frames = useMemo(() => (mode === 'top' ? buildTop() : buildBottom()), [mode])
  const st = useStepper(frames, { intervalMs: 1000 })
  const f = st.frame
  const ranges: ArrayRange[] = f.stack.map(([lo, hi], d) => ({ from: lo, to: hi, tone: ((d % 8) + 1) as VizTone, label: `第 ${d + 1} 层` }))
  const highlights: Record<number, VizTone> = {}
  const paint = (r: [number, number] | undefined, t: VizTone) => {
    if (!r) return
    for (let k = r[0]; k <= r[1]; k++) highlights[k] = t
  }
  paint(f.left, 'info')
  paint(f.right, 'warning')
  paint(f.merged, 'success')

  return (
    <VizFrame
      title={mode === 'top' ? '归并排序：先拆到底，再一路合并上来' : '自底向上归并：没有递归，按段长倍增'}
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '左半有序段', tone: 'info' },
            { label: '右半有序段', tone: 'warning' },
            { label: '合并结果', tone: 'success' },
            ...(mode === 'top' ? [{ label: '上方色条 = 当前递归栈', tone: 1 as VizTone, variant: 'soft' as const }] : []),
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
              { value: 'top', label: '自顶向下（递归）' },
              { value: 'bottom', label: '自底向上（迭代）' },
            ]}
          />
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="min-w-0 py-2">
          <ArrayView values={f.vals} ids={f.ids} mode="bars" barHeight={130} cellSize={40} ranges={ranges} highlights={highlights} />
        </div>
        <CodeTrace code={CODE[mode]} active={f.line} />
      </div>
    </VizFrame>
  )
}

export const MergeSortBottomUpViz = () => <MergeSortViz initialMode="bottom" />
