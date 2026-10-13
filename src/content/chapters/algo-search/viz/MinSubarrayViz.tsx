/**
 * 滑动窗口模板（LC209 长度最小的子数组）：右扩 → 满足条件就左缩。
 * 开关「含负数」给出反例：窗口和不再随左缩单调变化，滑窗会漏解。
 */
import { useMemo, useState } from 'react'
import { ArrayView, CodeTrace, Legend, StepControls, VizFrame, useStepper, type ArrayPointer, type ArrayRange } from '@/viz'
import { Slider, Switch } from '@/components/ui'
import { C } from './common'

const POS = [2, 3, 1, 2, 4, 3, 2, 1, 5, 1]
const NEG = [1, 2, -5, 4, 3, 1]

const CODE = `def min_subarray_len(target, nums):
    best = float('inf')
    s = l = 0
    for r, x in enumerate(nums):
        s += x                      # 右扩：纳入 nums[r]
        while s >= target:          # 满足条件 → 记录并尝试左缩
            best = min(best, r - l + 1)
            s -= nums[l]
            l += 1
    return 0 if best == float('inf') else best`

interface Frame {
  l: number
  r: number
  s: number
  best: number
  line: number
  bestRange?: [number, number]
  caption: React.ReactNode
}

function brute(nums: number[], target: number) {
  let best = Infinity
  let at: [number, number] | undefined
  for (let i = 0; i < nums.length; i++) {
    let s = 0
    for (let j = i; j < nums.length; j++) {
      s += nums[j]
      if (s >= target && j - i + 1 < best) {
        best = j - i + 1
        at = [i, j]
      }
    }
  }
  return { best, at }
}

function build(nums: number[], target: number): Frame[] {
  const fr: Frame[] = []
  let best = Infinity
  let bestRange: [number, number] | undefined
  let s = 0
  let l = 0
  fr.push({ l: 0, r: -1, s, best, line: 3, caption: <>窗口 <C>[l, r]</C> 初始为空。目标：和 ≥ <C>{target}</C> 的最短子数组。</> })
  for (let r = 0; r < nums.length; r++) {
    s += nums[r]
    fr.push({ l, r, s, best, bestRange, line: 5, caption: <>右扩：纳入 <C>nums[{r}] = {nums[r]}</C>，窗口和 <C>s = {s}</C>{s >= target ? '，已满足条件' : `，还不够 ${target}`}。</> })
    while (s >= target) {
      if (r - l + 1 < best) {
        best = r - l + 1
        bestRange = [l, r]
      }
      fr.push({ l, r, s, best, bestRange, line: 7, caption: <><C>s = {s} ≥ {target}</C>：记录长度 <C>{r - l + 1}</C>（best = {best}），再试着从左边缩短。</> })
      s -= nums[l]
      l++
      fr.push({ l, r, s, best, bestRange, line: 9, caption: <>左缩：移出 <C>nums[{l - 1}] = {nums[l - 1]}</C>，<C>s = {s}</C>。{s >= target ? '仍满足，继续缩。' : '不再满足，回到右扩。'}</> })
    }
  }
  const bf = brute(nums, target)
  const ans = best === Infinity ? 0 : best
  const truth = bf.best === Infinity ? 0 : bf.best
  fr.push({
    l, r: nums.length - 1, s, best, bestRange, line: 10,
    caption:
      ans === truth ? (
        <>结束：最短长度 <C>{ans}</C>。每个元素最多进窗一次、出窗一次，<C>O(n)</C>。</>
      ) : (
        <>⚠️ 滑窗返回 <C>{ans}</C>，但暴力枚举的正确答案是 <C>{truth}</C>（区间 <C>[{bf.at?.[0]}, {bf.at?.[1]}]</C>）。负数让“左缩一定让和变小”不再成立，窗口在错误的时机停止收缩/扩张，于是漏解。</>
      ),
  })
  return fr
}

export function MinSubarrayViz() {
  const [neg, setNeg] = useState(false)
  const [target, setTarget] = useState(8)
  const nums = neg ? NEG : POS
  const t = neg ? 7 : target
  const frames = useMemo(() => build(nums, t), [nums, t])
  const st = useStepper(frames, { intervalMs: 900 })
  const f = st.frame
  const ranges: ArrayRange[] = []
  if (f.r >= f.l) ranges.push({ from: f.l, to: f.r, tone: 'accent', label: `窗口 s=${f.s}` })
  if (f.bestRange) ranges.push({ from: f.bestRange[0], to: f.bestRange[1], tone: 'success', label: `best=${f.best}` })
  const pointers: ArrayPointer[] = [{ index: f.l, label: 'l', tone: 'info' }]
  if (f.r >= 0) pointers.push({ index: f.r, label: 'r', tone: 'warning' })

  return (
    <VizFrame
      title={neg ? '反例：有负数时滑窗失效' : '滑动窗口：右扩找可行，左缩找最优'}
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '当前窗口 [l, r]', tone: 'accent', variant: 'soft' },
            { label: '目前最短的可行窗口', tone: 'success', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          {!neg && <Slider className="min-w-[160px] flex-1" label="target" value={target} min={3} max={20} onChange={setTarget} />}
          <Switch checked={neg} onChange={setNeg} label="含负数（反例，target = 7）" />
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="flex min-h-[150px] min-w-0 items-center py-2">
          <ArrayView values={nums} cellSize={40} ranges={ranges} pointers={pointers} dim={(i) => i < f.l || i > f.r} />
        </div>
        <CodeTrace code={CODE} active={f.line} vars={{ l: f.l, r: f.r, s: f.s, best: f.best === Infinity ? '∞' : f.best }} />
      </div>
    </VizFrame>
  )
}
