/**
 * LC15 三数之和：排序 + 固定 i + 对撞指针；重点看两处去重为什么能保证“不重不漏”。
 */
import { useMemo } from 'react'
import { ArrayView, CodeTrace, Legend, StepControls, VizFrame, useStepper, type ArrayPointer, type VizTone } from '@/viz'
import { C } from './common'

const NUMS = [-4, -2, -2, 0, 0, 1, 2, 2, 3, 4]

const CODE = `def three_sum(nums):
    nums.sort()
    res = []
    for i in range(len(nums) - 2):
        if nums[i] > 0:
            break                     # 最小的数都 > 0，不可能凑出 0
        if i > 0 and nums[i] == nums[i - 1]:
            continue                  # 外层去重
        l, r = i + 1, len(nums) - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s < 0:
                l += 1
            elif s > 0:
                r -= 1
            else:
                res.append([nums[i], nums[l], nums[r]])
                while l < r and nums[l] == nums[l + 1]:
                    l += 1            # 内层去重
                l += 1
                r -= 1
    return res`

interface Frame {
  i: number
  l?: number
  r?: number
  line: number
  res: number[][]
  hit?: [number, number, number]
  skip?: number[]
  caption: React.ReactNode
}

function build(): Frame[] {
  const a = NUMS
  const n = a.length
  const res: number[][] = []
  const fr: Frame[] = []
  const snap = (x: Omit<Frame, 'res'>) => fr.push({ ...x, res: res.map((t) => [...t]) })
  for (let i = 0; i < n - 2; i++) {
    if (a[i] > 0) {
      snap({ i, line: 6, caption: <><C>nums[{i}] = {a[i]} &gt; 0</C>：后面都更大，三数之和不可能为 0，提前结束。</> })
      break
    }
    if (i > 0 && a[i] === a[i - 1]) {
      snap({ i, line: 8, skip: [i], caption: <><C>nums[{i}] == nums[{i - 1}] = {a[i]}</C>：以 {a[i]} 为最小数的三元组<b>上一轮已经全部找过</b>，跳过，否则会重复。</> })
      continue
    }
    let l = i + 1
    let r = n - 1
    snap({ i, l, r, line: 9, caption: <>固定 <C>nums[i] = {a[i]}</C>，在 <C>[{l}, {r}]</C> 上找两数之和 = <C>{-a[i]}</C>——就是上一节的对撞指针。</> })
    while (l < r) {
      const s = a[i] + a[l] + a[r]
      if (s < 0) {
        snap({ i, l, r, line: 13, caption: <><C>{a[i]} + {a[l]} + {a[r]} = {s} &lt; 0</C>，需要更大：<C>l += 1</C>。</> })
        l++
      } else if (s > 0) {
        snap({ i, l, r, line: 15, caption: <><C>{a[i]} + {a[l]} + {a[r]} = {s} &gt; 0</C>，需要更小：<C>r -= 1</C>。</> })
        r--
      } else {
        res.push([a[i], a[l], a[r]])
        const hit: [number, number, number] = [i, l, r]
        snap({ i, l, r, line: 17, hit, caption: <>✅ 找到 <C>[{a[i]}, {a[l]}, {a[r]}]</C>。</> })
        const skipped: number[] = []
        while (l < r && a[l] === a[l + 1]) {
          l++
          skipped.push(l)
        }
        if (skipped.length)
          snap({ i, l, r, line: 19, skip: skipped, caption: <>内层去重：<C>nums[l]</C> 后面还有相同的 <C>{a[l]}</C>，跳到最后一个，避免再次凑出同一个三元组。</> })
        l++
        r--
        snap({ i, l, r, line: 21, caption: <>两端同时收缩：只动一端不可能再得到 0（另一端的值已固定，且 <C>nums[l]</C> 已换成更大的值）。</> })
      }
    }
  }
  snap({ i: n - 3, line: 22, caption: <>结束，共 {res.length} 个不重复三元组。时间 <C>O(n²)</C>：外层 n 次 × 内层对撞 O(n)，排序的 <C>O(n log n)</C> 被吸收。</> })
  return fr
}

export function ThreeSumViz() {
  const frames = useMemo(build, [])
  const st = useStepper(frames, { intervalMs: 1000 })
  const f = st.frame
  const highlights: Record<number, VizTone> = {}
  highlights[f.i] = 'gold'
  if (f.l != null && f.l < NUMS.length) highlights[f.l] = 'info'
  if (f.r != null && f.r >= 0) highlights[f.r] = 'warning'
  f.skip?.forEach((k) => (highlights[k] = 'muted'))
  f.hit?.forEach((k) => (highlights[k] = 'success'))
  const pointers: ArrayPointer[] = [{ index: f.i, label: 'i', tone: 'gold', side: 'above' }]
  if (f.l != null && f.r != null && f.l <= f.r) {
    pointers.push({ index: f.l, label: 'l', tone: 'info' }, { index: f.r, label: 'r', tone: 'warning' })
  }

  return (
    <VizFrame
      title="三数之和：固定一个，剩下的交给对撞指针"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '固定的 i', tone: 'gold', variant: 'soft' },
            { label: 'l / r', tone: 'info', variant: 'soft' },
            { label: '命中', tone: 'success', variant: 'soft' },
            { label: '去重跳过', tone: 'muted', variant: 'soft' },
          ]}
        />
      }
      controls={<StepControls stepper={st} />}
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="min-w-0 space-y-4">
          <ArrayView values={NUMS} cellSize={38} highlights={highlights} pointers={pointers} dim={(k) => k < f.i} />
          <div>
            <div className="mb-1.5 text-xs font-medium text-muted">res（{f.res.length}）</div>
            <div className="flex min-h-[28px] flex-wrap gap-1.5">
              {f.res.map((t, k) => (
                <span key={k} className="rounded-md border border-success/60 bg-success-soft px-1.5 py-0.5 font-mono text-[12px]">
                  [{t.join(', ')}]
                </span>
              ))}
            </div>
          </div>
        </div>
        <CodeTrace code={CODE} active={f.line} maxHeight={320} vars={{ i: f.i, l: f.l ?? '—', r: f.r ?? '—' }} />
      </div>
    </VizFrame>
  )
}
