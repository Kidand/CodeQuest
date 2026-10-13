/**
 * 「不找 target 的二分」：LC153 旋转数组最小值 / LC162 寻找峰值。
 * 两者都只用 lo < hi、lo = mid + 1、hi = mid 的“找第一个 True”骨架；
 * 区别在于 153 有真正单调的谓词，而 162 依赖的是“区间内必有峰”的不变量。
 */
import { useMemo, useState } from 'react'
import { Shuffle } from 'lucide-react'
import { ArrayView, CodeTrace, Legend, StepControls, VizFrame, useStepper, type ArrayPointer, type VizTone } from '@/viz'
import { Button, Segmented, Slider } from '@/components/ui'
import { rng } from '@/lib/utils'
import { C } from './common'

type Mode = 'min' | 'peak'

const CODE: Record<Mode, string> = {
  min: `def find_min(nums):
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] > nums[hi]:   # mid 在左边那段（较大的一段）
            lo = mid + 1
        else:                      # mid 在右段：最小值在 [lo, mid]
            hi = mid
    return nums[lo]`,
  peak: `def find_peak(nums):
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] < nums[mid + 1]:   # 上坡：右边一定有峰
            lo = mid + 1
        else:                           # 下坡：[lo, mid] 一定有峰
            hi = mid
    return lo`,
}

const BASE = [3, 4, 5, 7, 9, 10, 12, 15, 18, 20]
const PEAK_DEFAULT = [1, 3, 2, 4, 7, 6, 5, 8, 10, 9, 4]

interface Frame {
  lo: number
  hi: number
  mid?: number
  line: number
  caption: React.ReactNode
  done?: boolean
}

function buildMin(nums: number[]): Frame[] {
  const fr: Frame[] = []
  let lo = 0
  let hi = nums.length - 1
  fr.push({ lo, hi, line: 2, caption: <>谓词 <C>P(i) = nums[i] ≤ nums[-1]</C> 把数组分成 F…F（左段，大）和 T…T（右段，小）。最小值 = <b>第一个 T</b>。</> })
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (nums[mid] > nums[hi]) {
      fr.push({ lo, hi, mid, line: 6, caption: <><C>nums[{mid}]={nums[mid]} &gt; nums[hi]={nums[hi]}</C>：mid 在左段（F），最小值在它右边，<C>lo = {mid + 1}</C>。</> })
      lo = mid + 1
    } else {
      fr.push({ lo, hi, mid, line: 8, caption: <><C>nums[{mid}]={nums[mid]} ≤ nums[hi]={nums[hi]}</C>：mid 在右段（T），它自己可能就是最小值，<C>hi = {mid}</C>。</> })
      hi = mid
    }
  }
  fr.push({ lo, hi, line: 9, done: true, caption: <><C>lo == hi == {lo}</C>，最小值 <C>{nums[lo]}</C>（也就是旋转点）。</> })
  return fr
}

function buildPeak(nums: number[]): Frame[] {
  const fr: Frame[] = []
  let lo = 0
  let hi = nums.length - 1
  fr.push({ lo, hi, line: 2, caption: <>不变量：<C>nums[lo-1] &lt; nums[lo]</C> 且 <C>nums[hi] &gt; nums[hi+1]</C>（越界视为 −∞），所以 <C>[lo, hi]</C> 内<b>一定有峰</b>。</> })
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (nums[mid] < nums[mid + 1]) {
      fr.push({ lo, hi, mid, line: 6, caption: <><C>{nums[mid]} &lt; {nums[mid + 1]}</C> 上坡：一直往右走，要么遇到下坡（峰），要么走到末尾（末尾 &gt; −∞ 也是峰）。<C>lo = {mid + 1}</C>，此时 <C>nums[lo-1] &lt; nums[lo]</C> 仍成立。</> })
      lo = mid + 1
    } else {
      fr.push({ lo, hi, mid, line: 8, caption: <><C>{nums[mid]} &gt; {nums[mid + 1]}</C> 下坡：<C>[lo, mid]</C> 内必有峰（mid 本身可能是）。<C>hi = {mid}</C>，此时 <C>nums[hi] &gt; nums[hi+1]</C> 仍成立。</> })
      hi = mid
    }
  }
  fr.push({ lo, hi, line: 9, done: true, caption: <>区间缩成一个点 <C>{lo}</C>：左邻比它小、右邻比它小 ⇒ 峰值 <C>nums[{lo}] = {nums[lo]}</C>。注意它<b>不一定是最大的峰</b>，题目只要求任意一个。</> })
  return fr
}

function randomPeakArray(seed: number, n = 11) {
  const r = rng(seed)
  const out: number[] = []
  while (out.length < n) {
    const v = 1 + Math.floor(r() * 12)
    if (out.length && out[out.length - 1] === v) continue
    out.push(v)
  }
  return out
}

export function PredicateHalfViz() {
  const [mode, setMode] = useState<Mode>('min')
  const [k, setK] = useState(6)
  const [seed, setSeed] = useState(0)
  const nums = useMemo(
    () => (mode === 'min' ? [...BASE.slice(k), ...BASE.slice(0, k)] : seed === 0 ? PEAK_DEFAULT : randomPeakArray(seed)),
    [mode, k, seed],
  )
  const frames = useMemo(() => (mode === 'min' ? buildMin(nums) : buildPeak(nums)), [mode, nums])
  const st = useStepper(frames, { intervalMs: 1200 })
  const f = st.frame
  const n = nums.length

  const highlights: Record<number, VizTone> = {}
  if (mode === 'min') {
    nums.forEach((v, i) => (highlights[i] = v <= nums[n - 1] ? 'success' : 'muted'))
  } else {
    nums.forEach((v, i) => {
      const l = i === 0 ? -Infinity : nums[i - 1]
      const r = i === n - 1 ? -Infinity : nums[i + 1]
      if (v > l && v > r) highlights[i] = 'gold'
    })
  }
  if (f.done) highlights[f.lo] = mode === 'min' ? 'accent' : 'danger'

  const pointers: ArrayPointer[] = [
    { index: f.lo, label: 'lo', tone: 'info' },
    { index: f.hi, label: 'hi', tone: 'warning' },
  ]
  if (f.mid != null) pointers.push({ index: f.mid, label: 'mid', tone: 'accent', side: 'above' })

  return (
    <VizFrame
      title={mode === 'min' ? '找最小值：在 F…F T…T 上找第一个 T' : '找峰值：没有单调性，也能二分'}
      caption={f.caption}
      legend={
        <Legend
          items={
            mode === 'min'
              ? [
                  { label: 'P(i) = T（右段）', tone: 'success' },
                  { label: 'P(i) = F（左段）', tone: 'muted' },
                  { label: '结果', tone: 'accent' },
                ]
              : [
                  { label: '所有峰（任一个都对）', tone: 'gold' },
                  { label: '二分找到的峰', tone: 'danger' },
                ]
          }
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
              { value: 'min', label: 'LC153 最小值' },
              { value: 'peak', label: 'LC162 峰值' },
            ]}
          />
          {mode === 'min' ? (
            <Slider className="min-w-[160px] flex-1" label="旋转点 k" value={k} min={0} max={BASE.length - 1} onChange={setK} />
          ) : (
            <Button size="sm" variant="ghost" icon={<Shuffle size={14} />} onClick={() => setSeed((s) => s + 1)}>
              换一组数据
            </Button>
          )}
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="min-w-0 py-2">
          <ArrayView values={nums} mode="bars" barHeight={130} cellSize={36} highlights={highlights} pointers={pointers} dim={(i) => i < f.lo || i > f.hi} />
        </div>
        <CodeTrace code={CODE[mode]} active={f.line} vars={{ lo: f.lo, hi: f.hi, mid: f.mid ?? '—' }} />
      </div>
    </VizFrame>
  )
}
