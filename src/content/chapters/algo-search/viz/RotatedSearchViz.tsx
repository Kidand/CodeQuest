/**
 * LC33 搜索旋转排序数组：每一轮先判断「哪半边有序」，再看 target 是否落在有序那半。
 * 柱状图让“两段上坡 + 一个断崖”的形状一目了然；有序半段用区间条标出。
 */
import { useMemo, useState } from 'react'
import { ArrayView, CodeTrace, Legend, StepControls, VizFrame, useStepper, type ArrayPointer, type ArrayRange } from '@/viz'
import { Slider } from '@/components/ui'
import { C } from './common'

const BASE = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32]

const CODE = `def search(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        if nums[lo] <= nums[mid]:          # 左半 [lo, mid] 有序
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:                              # 右半 [mid, hi] 有序
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1`

interface Frame {
  lo: number
  hi: number
  mid?: number
  line: number
  sorted?: [number, number]
  found?: number
  caption: React.ReactNode
}

function build(nums: number[], target: number): Frame[] {
  const fr: Frame[] = []
  let lo = 0
  let hi = nums.length - 1
  fr.push({ lo, hi, line: 2, caption: <>闭区间 <C>[{lo}, {hi}]</C>，找 <C>target = {target}</C>。数组被旋转过，整体不有序，但<b>任意切一刀，至少有一半有序</b>。</> })
  let round = 0
  while (lo <= hi) {
    round++
    const mid = (lo + hi) >> 1
    fr.push({ lo, hi, mid, line: 4, caption: <>第 {round} 轮：<C>mid = ({lo}+{hi})//2 = {mid}</C>，<C>nums[mid] = {nums[mid]}</C>。</> })
    if (nums[mid] === target) {
      fr.push({ lo, hi, mid, line: 6, found: mid, caption: <>✅ <C>nums[{mid}] == {target}</C>，返回 <C>{mid}</C>。共 {round} 轮。</> })
      return fr
    }
    if (nums[lo] <= nums[mid]) {
      const s: [number, number] = [lo, mid]
      fr.push({ lo, hi, mid, line: 7, sorted: s, caption: <><C>nums[lo]={nums[lo]} ≤ nums[mid]={nums[mid]}</C> ⇒ 左半 <C>[{lo}, {mid}]</C> 有序（断崖不在这里）。</> })
      if (nums[lo] <= target && target < nums[mid]) {
        hi = mid - 1
        fr.push({ lo, hi, mid, line: 9, sorted: s, caption: <><C>{nums[lo]} ≤ {target} &lt; {nums[mid]}</C>：target 落在<b>有序的左半</b>，<C>hi = mid - 1 = {hi}</C>。</> })
      } else {
        lo = mid + 1
        fr.push({ lo, hi, mid, line: 11, sorted: s, caption: <>target 不在 <C>[{nums[s[0]]}, {nums[mid]})</C> 这个有序范围内，只能在右半：<C>lo = mid + 1 = {lo}</C>。</> })
      }
    } else {
      const s: [number, number] = [mid, hi]
      fr.push({ lo, hi, mid, line: 12, sorted: s, caption: <><C>nums[lo]={nums[lo]} &gt; nums[mid]={nums[mid]}</C> ⇒ 断崖在左半，所以右半 <C>[{mid}, {hi}]</C> 有序。</> })
      if (nums[mid] < target && target <= nums[hi]) {
        lo = mid + 1
        fr.push({ lo, hi, mid, line: 14, sorted: s, caption: <><C>{nums[mid]} &lt; {target} ≤ {nums[hi]}</C>：target 落在<b>有序的右半</b>，<C>lo = mid + 1 = {lo}</C>。</> })
      } else {
        hi = mid - 1
        fr.push({ lo, hi, mid, line: 16, sorted: s, caption: <>target 不在 <C>({nums[mid]}, {nums[s[1]]}]</C> 这个有序范围内，只能在左半：<C>hi = mid - 1 = {hi}</C>。</> })
      }
    }
  }
  fr.push({ lo, hi, line: 17, caption: <>区间已空，<C>{target}</C> 不在数组中，返回 <C>-1</C>。共 {round} 轮——依然是 <C>O(log n)</C>。</> })
  return fr
}

export function RotatedSearchViz() {
  const [k, setK] = useState(4)
  const [target, setTarget] = useState(5)
  const nums = useMemo(() => [...BASE.slice(k), ...BASE.slice(0, k)], [k])
  const frames = useMemo(() => build(nums, target), [nums, target])
  const st = useStepper(frames, { intervalMs: 1100 })
  const f = st.frame

  const pointers: ArrayPointer[] = [
    { index: f.lo, label: 'lo', tone: 'info' },
    { index: f.hi, label: 'hi', tone: 'warning' },
  ]
  if (f.mid != null) pointers.push({ index: f.mid, label: 'mid', tone: 'accent', side: 'above' })
  const ranges: ArrayRange[] = f.sorted ? [{ from: f.sorted[0], to: f.sorted[1], tone: 'success', label: '有序' }] : []
  const highlights: Record<number, 'success' | 'accent'> = {}
  if (f.found != null) highlights[f.found] = 'success'
  else if (f.mid != null) highlights[f.mid] = 'accent'

  return (
    <VizFrame
      title="旋转数组：切一刀，至少一半有序"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '有序半段', tone: 'success', variant: 'soft' },
            { label: '当前 mid', tone: 'accent' },
            { label: '已排除（变暗）', tone: 'muted', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Slider className="min-w-[150px] flex-1" label="旋转点 k（把前 k 个挪到末尾）" value={k} min={0} max={BASE.length - 1} onChange={setK} />
          <Slider className="min-w-[150px] flex-1" label="target" value={target} min={0} max={34} onChange={setTarget} />
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="min-w-0 py-2">
          <ArrayView
            values={nums}
            mode="bars"
            barHeight={130}
            cellSize={36}
            pointers={pointers}
            ranges={ranges}
            highlights={highlights}
            dim={(i) => (i < f.lo || i > f.hi) && f.found !== i}
          />
        </div>
        <CodeTrace code={CODE} active={f.line} vars={{ lo: f.lo, hi: f.hi, mid: f.mid ?? '—', target }} />
      </div>
    </VizFrame>
  )
}
