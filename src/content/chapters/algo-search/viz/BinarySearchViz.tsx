/**
 * 二分查找逐帧可视化（样板：帧预计算 + useStepper + ArrayView + CodeTrace 联动）。
 *
 * 写法要点（供其它可视化参考）：
 *  1. 纯函数 buildFrames() 把算法完整执行一遍，每一步产出一帧「可渲染状态」
 *  2. 参数（数组、target、模板）变化 → useMemo 重新生成帧 → stepper 自动归零
 *  3. 帧里带上 line（当前代码行）与 caption（文字说明），让图、码、字三者同步
 */
import { useMemo, useState } from 'react'
import { Shuffle } from 'lucide-react'
import { ArrayView, CodeTrace, Legend, StepControls, VizFrame, useStepper, type ArrayPointer } from '@/viz'
import { Button, Segmented, Slider } from '@/components/ui'
import { rng } from '@/lib/utils'

type Mode = 'closed' | 'half'

const CODE: Record<Mode, string> = {
  closed: `def binary_search(nums, target):
    lo, hi = 0, len(nums) - 1        # 闭区间 [lo, hi]
    while lo <= hi:                  # 区间非空
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            lo = mid + 1             # mid 已检查，排除
        else:
            hi = mid - 1
    return -1`,
  half: `def binary_search(nums, target):
    lo, hi = 0, len(nums)            # 左闭右开 [lo, hi)
    while lo < hi:                   # 区间非空
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid                 # hi 本身不在区间内
    return -1`,
}

interface Frame {
  lo: number
  hi: number
  mid?: number
  line: number
  caption: React.ReactNode
  found?: number
  done?: boolean
}

const C = ({ children }: { children: React.ReactNode }) => (
  <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[12.5px]">{children}</code>
)

function buildFrames(nums: number[], target: number, mode: Mode): Frame[] {
  const frames: Frame[] = []
  let lo = 0
  let hi = mode === 'closed' ? nums.length - 1 : nums.length
  const range = () => (mode === 'closed' ? `[${lo}, ${hi}]` : `[${lo}, ${hi})`)
  frames.push({ lo, hi, line: 2, caption: <>初始化搜索区间 <C>{range()}</C>，目标 <C>target = {target}</C>。</> })
  let round = 0
  while (mode === 'closed' ? lo <= hi : lo < hi) {
    round++
    const mid = (lo + hi) >> 1
    frames.push({
      lo, hi, mid, line: 4,
      caption: <>第 {round} 轮：区间 <C>{range()}</C> 非空，<C>mid = ({lo}+{hi})//2 = {mid}</C>，<C>nums[mid] = {nums[mid]}</C>。</>,
    })
    if (nums[mid] === target) {
      frames.push({ lo, hi, mid, line: 6, found: mid, done: true, caption: <>✅ <C>nums[{mid}] == {target}</C>，找到目标，返回下标 <C>{mid}</C>。共比较 {round} 轮。</> })
      return frames
    }
    if (nums[mid] < target) {
      lo = mid + 1
      frames.push({ lo, hi, mid, line: 8, caption: <><C>{nums[mid]} &lt; {target}</C>：目标只可能在右侧，<C>lo = mid + 1 = {lo}</C>，区间变为 <C>{range()}</C>。</> })
    } else {
      hi = mode === 'closed' ? mid - 1 : mid
      frames.push({
        lo, hi, mid, line: 10,
        caption:
          mode === 'closed'
            ? <><C>{nums[mid]} &gt; {target}</C>：目标只可能在左侧，<C>hi = mid - 1 = {hi}</C>，区间变为 <C>{range()}</C>。</>
            : <><C>{nums[mid]} &gt; {target}</C>：目标只可能在左侧，<C>hi = mid = {hi}</C>（开区间端点本就不含 mid）。</>,
      })
    }
  }
  frames.push({ lo, hi, line: 11, done: true, caption: <>区间 <C>{range()}</C> 已空，<C>{target}</C> 不存在，返回 <C>-1</C>。注意此时 <C>lo = {lo}</C> 恰好是它的<b>插入位置</b>。</> })
  return frames
}

function randomSorted(seed: number, n = 11) {
  const r = rng(seed)
  const set = new Set<number>()
  while (set.size < n) set.add(1 + Math.floor(r() * 60))
  return [...set].sort((a, b) => a - b)
}

export function BinarySearchViz({ initialMode = 'closed' }: { initialMode?: Mode }) {
  const [seed, setSeed] = useState(7)
  const nums = useMemo(() => randomSorted(seed), [seed])
  const [target, setTarget] = useState(() => nums[7])
  const [mode, setMode] = useState<Mode>(initialMode)
  const frames = useMemo(() => buildFrames(nums, target, mode), [nums, target, mode])
  const st = useStepper(frames, { intervalMs: 1100 })
  const f = st.frame

  const inRange = (i: number) => (mode === 'closed' ? i >= f.lo && i <= f.hi : i >= f.lo && i < f.hi)
  const pointers: ArrayPointer[] = [
    { index: f.lo, label: 'lo', tone: 'info', side: 'below' },
    // 左闭右开时 hi 可能等于 n，指针会画在数组外侧
    { index: f.hi, label: 'hi', tone: 'warning', side: 'below' },
  ]
  if (f.mid != null) pointers.push({ index: f.mid, label: 'mid', tone: 'accent', side: 'above' })

  return (
    <VizFrame
      title="二分查找：每一轮把搜索区间砍掉一半"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '当前 mid', tone: 'accent', variant: 'soft' },
            { label: '找到目标', tone: 'success', variant: 'soft' },
            { label: '已排除（变暗）', tone: 'muted', variant: 'soft' },
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
              { value: 'closed', label: '闭区间 [lo, hi]' },
              { value: 'half', label: '左闭右开 [lo, hi)' },
            ]}
          />
          <Slider className="w-44" label="target" value={target} min={0} max={62} onChange={setTarget} />
          <Button size="sm" variant="ghost" icon={<Shuffle size={14} />} onClick={() => setSeed((s) => s + 1)}>
            换一组数据
          </Button>
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="flex min-h-[150px] items-center overflow-x-auto py-2">
          <ArrayView
            values={nums}
            cellSize={38}
            pointers={pointers}
            dim={(i) => !inRange(i) && f.found !== i}
            highlights={f.found != null ? { [f.found]: 'success' } : f.mid != null ? { [f.mid]: 'accent' } : {}}
          />
        </div>
        <CodeTrace code={CODE[mode]} active={f.line} vars={{ lo: f.lo, hi: f.hi, mid: f.mid ?? '—', target }} />
      </div>
    </VizFrame>
  )
}

/** 第二节使用：默认展示左闭右开 */
export const BinarySearchHalfOpenViz = () => <BinarySearchViz initialMode="half" />
