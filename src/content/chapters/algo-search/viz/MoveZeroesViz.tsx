/**
 * 同向双指针（读写指针 / 快慢指针）：LC283 移动零。
 * 不变量：[0, slow) 是已经处理好的非零元素（保持原相对顺序），[slow, fast) 全是 0。
 * ArrayView 的 ids 让交换有位移动画。
 */
import { useMemo } from 'react'
import { ArrayView, CodeTrace, Legend, StepControls, VizFrame, useStepper, type ArrayPointer, type ArrayRange, type VizTone } from '@/viz'
import { C } from './common'

const INIT = [0, 1, 0, 3, 12, 0, 5, 7]

const CODE = `def move_zeroes(nums):
    slow = 0                     # [0, slow)：已就位的非零元素
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1
    # 循环结束：[0, slow) 非零且保序，[slow, n) 全为 0`

interface Frame {
  vals: number[]
  ids: number[]
  slow: number
  fast: number
  line: number
  caption: React.ReactNode
}

function build(): Frame[] {
  const vals = [...INIT]
  const ids = INIT.map((_, i) => i)
  const fr: Frame[] = []
  let slow = 0
  fr.push({ vals: [...vals], ids: [...ids], slow, fast: 0, line: 2, caption: <>读指针 <C>fast</C> 扫描每个元素，写指针 <C>slow</C> 指向下一个非零元素该放的位置。</> })
  for (let fast = 0; fast < vals.length; fast++) {
    if (vals[fast] !== 0) {
      const before = slow
      ;[vals[slow], vals[fast]] = [vals[fast], vals[slow]]
      ;[ids[slow], ids[fast]] = [ids[fast], ids[slow]]
      slow++
      fr.push({
        vals: [...vals], ids: [...ids], slow, fast, line: 5,
        caption: before === fast
          ? <><C>nums[{fast}] = {vals[before]}</C> 非零，且 slow == fast，原地“交换”自己，<C>slow = {slow}</C>。</>
          : <><C>nums[{fast}] = {vals[before]}</C> 非零：与 <C>nums[{before}] = 0</C> 交换，非零元素前移，<C>slow = {slow}</C>。</>,
      })
    } else {
      fr.push({ vals: [...vals], ids: [...ids], slow, fast, line: 4, caption: <><C>nums[{fast}] = 0</C>：跳过，0 留在 <C>[slow, fast]</C> 区间里等待被换到后面。</> })
    }
  }
  fr.push({ vals: [...vals], ids: [...ids], slow, fast: vals.length - 1, line: 7, caption: <>完成：前 {slow} 个是非零元素（相对顺序不变），后面全是 0。每个元素只被读一次，<C>O(n)</C> 时间、<C>O(1)</C> 空间。</> })
  return fr
}

export function MoveZeroesViz() {
  const frames = useMemo(build, [])
  const st = useStepper(frames, { intervalMs: 900 })
  const f = st.frame
  const highlights: Record<number, VizTone> = {}
  f.vals.forEach((v, i) => {
    if (i < f.slow) highlights[i] = 'success'
    else if (i <= f.fast && v === 0) highlights[i] = 'muted'
  })
  const pointers: ArrayPointer[] = [
    { index: f.slow, label: 'slow', tone: 'accent' },
    { index: f.fast, label: 'fast', tone: 'warning', side: 'above' },
  ]
  const ranges: ArrayRange[] = f.slow > 0 ? [{ from: 0, to: f.slow - 1, tone: 'success', label: '非零·已就位' }] : []

  return (
    <VizFrame
      title="同向双指针：[0, slow) 永远是处理好的结果"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '[0, slow) 非零已就位', tone: 'success', variant: 'soft' },
            { label: '[slow, fast] 中的 0', tone: 'muted', variant: 'soft' },
          ]}
        />
      }
      controls={<StepControls stepper={st} />}
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="flex min-h-[150px] min-w-0 items-center py-2">
          <ArrayView values={f.vals} ids={f.ids} cellSize={42} highlights={highlights} pointers={pointers} ranges={ranges} />
        </div>
        <CodeTrace code={CODE} active={f.line} vars={{ slow: f.slow, fast: f.fast }} />
      </div>
    </VizFrame>
  )
}
