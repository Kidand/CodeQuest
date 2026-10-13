/**
 * 快排的核心：一次划分（partition）。Lomuto 与 Hoare 两种写法对照。
 * 柱状图 + ids 让每次交换都有位移动画；颜色编码循环不变量。
 */
import { useMemo, useState } from 'react'
import { ArrayView, CodeTrace, Legend, StepControls, VizFrame, useStepper, type ArrayPointer, type VizTone } from '@/viz'
import { Segmented } from '@/components/ui'
import { C } from './common'

type Mode = 'lomuto' | 'hoare'
const DATA = [6, 3, 9, 1, 7, 4, 8, 2, 5]

const CODE: Record<Mode, string> = {
  lomuto: `def partition(nums, lo, hi):        # Lomuto
    pivot = nums[hi]                 # 取最后一个元素为 pivot
    i = lo                           # [lo, i) < pivot
    for j in range(lo, hi):          # [i, j) >= pivot，[j, hi) 未处理
        if nums[j] < pivot:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
    nums[i], nums[hi] = nums[hi], nums[i]   # pivot 归位
    return i                         # pivot 的最终位置`,
  hoare: `def partition(nums, lo, hi):        # Hoare
    pivot = nums[(lo + hi) // 2]
    i, j = lo - 1, hi + 1
    while True:
        i += 1
        while nums[i] < pivot: i += 1    # 左边找 >= pivot 的
        j -= 1
        while nums[j] > pivot: j -= 1    # 右边找 <= pivot 的
        if i >= j:
            return j                     # [lo, j] <= pivot <= [j+1, hi]
        nums[i], nums[j] = nums[j], nums[i]`,
}

interface Frame {
  vals: number[]
  ids: number[]
  i: number
  j: number
  line: number
  pivotVal: number
  pivotId?: number
  done?: number
  caption: React.ReactNode
}

function buildLomuto(): Frame[] {
  const vals = [...DATA]
  const ids = DATA.map((_, k) => k)
  const lo = 0
  const hi = vals.length - 1
  const pivot = vals[hi]
  const pivotId = ids[hi]
  const fr: Frame[] = []
  const push = (f: Omit<Frame, 'vals' | 'ids' | 'pivotVal' | 'pivotId'>) => fr.push({ ...f, vals: [...vals], ids: [...ids], pivotVal: pivot, pivotId })
  let i = lo
  push({ i, j: lo, line: 3, caption: <>pivot = 最后一个元素 <C>{pivot}</C>。不变量：<C>[lo, i)</C> 都 &lt; pivot，<C>[i, j)</C> 都 ≥ pivot。</> })
  for (let j = lo; j < hi; j++) {
    if (vals[j] < pivot) {
      const a = vals[j]
      ;[vals[i], vals[j]] = [vals[j], vals[i]]
      ;[ids[i], ids[j]] = [ids[j], ids[i]]
      i++
      push({ i, j: j + 1, line: 7, caption: <><C>{a} &lt; {pivot}</C>：与 <C>nums[i]</C> 交换，放进“小于区”，<C>i = {i}</C>。</> })
    } else {
      push({ i, j: j + 1, line: 5, caption: <><C>{vals[j]} ≥ {pivot}</C>：留在原地，“≥ 区” <C>[i, j)</C> 自然扩大。</> })
    }
  }
  ;[vals[i], vals[hi]] = [vals[hi], vals[i]]
  ;[ids[i], ids[hi]] = [ids[hi], ids[i]]
  push({ i, j: hi, line: 9, done: i, caption: <>最后把 pivot 换到 <C>i = {i}</C>：左边都 &lt; {pivot}，右边都 ≥ {pivot}，<b>pivot 已在最终位置</b>。共比较 {hi - lo} 次。</> })
  return fr
}

function buildHoare(): Frame[] {
  const vals = [...DATA]
  const ids = DATA.map((_, k) => k)
  const lo = 0
  const hi = vals.length - 1
  const pivot = vals[(lo + hi) >> 1]
  const fr: Frame[] = []
  const push = (f: Omit<Frame, 'vals' | 'ids' | 'pivotVal'>) => fr.push({ ...f, vals: [...vals], ids: [...ids], pivotVal: pivot })
  let i = lo - 1
  let j = hi + 1
  push({ i: lo, j: hi, line: 3, caption: <>pivot = 中间元素的<b>值</b> <C>{pivot}</C>。i 从左找 ≥ pivot 的元素，j 从右找 ≤ pivot 的元素，找到就交换。</> })
  for (;;) {
    i++
    while (vals[i] < pivot) i++
    j--
    while (vals[j] > pivot) j--
    if (i >= j) {
      push({ i, j, line: 10, done: j, caption: <><C>i = {i} ≥ j = {j}</C>：指针相遇，返回 <C>{j}</C>。<C>[0, {j}]</C> ≤ {pivot} ≤ <C>[{j + 1}, {hi}]</C>，但 pivot <b>不一定</b>在最终位置，递归区间是 <C>[lo, j]</C> 与 <C>[j+1, hi]</C>。</> })
      return fr
    }
    push({ i, j, line: 8, caption: <>i 停在 <C>{vals[i]}</C>（≥ {pivot}），j 停在 <C>{vals[j]}</C>（≤ {pivot}），两者都“站错了边”。</> })
    ;[vals[i], vals[j]] = [vals[j], vals[i]]
    ;[ids[i], ids[j]] = [ids[j], ids[i]]
    push({ i, j, line: 11, caption: <>交换后 <C>{vals[i]}</C> 到左边、<C>{vals[j]}</C> 到右边，继续向中间推进。Hoare 的交换次数平均约为 Lomuto 的 1/3。</> })
  }
}

export function PartitionViz() {
  const [mode, setMode] = useState<Mode>('lomuto')
  const frames = useMemo(() => (mode === 'lomuto' ? buildLomuto() : buildHoare()), [mode])
  const st = useStepper(frames, { intervalMs: 1000 })
  const f = st.frame
  const n = f.vals.length

  const highlights: Record<number, VizTone> = {}
  if (mode === 'lomuto') {
    for (let k = 0; k < n; k++) {
      if (f.ids[k] === f.pivotId) highlights[k] = 'gold'
      else if (f.done != null) highlights[k] = k < f.done ? 'info' : 'warning'
      else if (k < f.i) highlights[k] = 'info'
      else if (k < f.j) highlights[k] = 'warning'
    }
  } else {
    for (let k = 0; k < n; k++) {
      if (f.done != null) highlights[k] = k <= f.done ? 'info' : 'warning'
      else if (k < f.i) highlights[k] = 'info'
      else if (k > f.j) highlights[k] = 'warning'
      if (f.vals[k] === f.pivotVal && f.done == null) highlights[k] = 'gold'
    }
  }
  const pointers: ArrayPointer[] =
    mode === 'lomuto'
      ? [
          { index: f.i, label: 'i', tone: 'info' },
          ...(f.done == null ? [{ index: f.j, label: 'j', tone: 'accent' as const, side: 'above' as const }] : []),
        ]
      : [
          { index: Math.max(0, Math.min(n - 1, f.i)), label: 'i', tone: 'info' },
          { index: Math.max(0, Math.min(n - 1, f.j)), label: 'j', tone: 'warning', side: 'above' },
        ]

  return (
    <VizFrame
      title="一次划分：把 pivot 放到“左小右大”的分界处"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: mode === 'lomuto' ? '< pivot' : '≤ pivot（已归左）', tone: 'info' },
            { label: mode === 'lomuto' ? '≥ pivot' : '≥ pivot（已归右）', tone: 'warning' },
            { label: 'pivot', tone: 'gold' },
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
              { value: 'lomuto', label: 'Lomuto（末尾 pivot）' },
              { value: 'hoare', label: 'Hoare（双向扫描）' },
            ]}
          />
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="min-w-0 py-2">
          <ArrayView values={f.vals} ids={f.ids} mode="bars" barHeight={140} cellSize={38} highlights={highlights} pointers={pointers} />
        </div>
        <CodeTrace code={CODE[mode]} active={f.line} vars={{ pivot: f.pivotVal, i: f.i, j: f.j }} />
      </div>
    </VizFrame>
  )
}
