/**
 * lower_bound / upper_bound：把二分理解为「在 F F F … T T T 的单调谓词上找第一个 T」。
 * 样板要点：用颜色直接编码谓词结果，让“为什么二分正确”一眼可见。
 */
import { useMemo, useState } from 'react'
import { ArrayView, CodeTrace, Legend, StepControls, VizFrame, useStepper, type ArrayPointer } from '@/viz'
import { Segmented, Slider } from '@/components/ui'

const NUMS = [1, 2, 2, 2, 3, 5, 5, 8, 9, 9, 12]
type Kind = 'lower' | 'upper'

const code = (k: Kind) => `def ${k}_bound(nums, target):
    lo, hi = 0, len(nums)            # 答案在 [0, n] 中
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] ${k === 'lower' ? '<' : '<='} target:      # 谓词为 False
            lo = mid + 1
        else:                         # 谓词为 True
            hi = mid
    return lo                         # 第一个 True 的位置`

interface Frame {
  lo: number
  hi: number
  mid?: number
  line: number
  text: string
}

function build(nums: number[], target: number, kind: Kind): Frame[] {
  const pred = (x: number) => (kind === 'lower' ? x >= target : x > target)
  const fr: Frame[] = []
  let lo = 0
  let hi = nums.length
  fr.push({ lo, hi, line: 2, text: `答案区间 [${lo}, ${hi}]：第一个满足「nums[i] ${kind === 'lower' ? '≥' : '>'} ${target}」的位置一定在其中（全不满足时为 n=${hi}）。` })
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (!pred(nums[mid])) {
      fr.push({ lo, hi, mid, line: 6, text: `nums[${mid}]=${nums[mid]} 谓词为 False，第一个 True 在 mid 右边：lo = ${mid + 1}` })
      lo = mid + 1
    } else {
      fr.push({ lo, hi, mid, line: 8, text: `nums[${mid}]=${nums[mid]} 谓词为 True，mid 可能就是答案，保留它：hi = ${mid}` })
      hi = mid
    }
  }
  fr.push({ lo, hi, line: 9, text: `lo == hi == ${lo}，收敛！${kind}_bound(${target}) = ${lo}` })
  return fr
}

export function BoundaryViz() {
  const [target, setTarget] = useState(5)
  const [kind, setKind] = useState<Kind>('lower')
  const frames = useMemo(() => build(NUMS, target, kind), [target, kind])
  const st = useStepper(frames, { intervalMs: 1000 })
  const f = st.frame
  const pred = (x: number) => (kind === 'lower' ? x >= target : x > target)
  const highlights = Object.fromEntries(NUMS.map((x, i) => [i, pred(x) ? 'success' : 'muted'] as const))
  const pointers: ArrayPointer[] = [
    { index: f.lo, label: 'lo', tone: 'info' },
    { index: f.hi, label: 'hi', tone: 'warning' },
  ]
  if (f.mid != null) pointers.push({ index: f.mid, label: 'mid', tone: 'accent', side: 'above' })

  return (
    <VizFrame
      title="边界二分：在单调谓词上找第一个 True"
      caption={f.text}
      legend={
        <Legend
          items={[
            { label: `谓词 True（nums[i] ${kind === 'lower' ? '≥' : '>'} target）`, tone: 'success', variant: 'soft' },
            { label: '谓词 False', tone: 'muted', variant: 'soft' },
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
              { value: 'lower', label: 'lower_bound（≥）' },
              { value: 'upper', label: 'upper_bound（>）' },
            ]}
          />
          <Slider className="w-40" label="target" value={target} min={0} max={13} onChange={setTarget} />
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="flex min-h-[150px] items-center overflow-x-auto py-2">
          <ArrayView values={NUMS} cellSize={36} highlights={highlights} pointers={pointers} dim={(i) => i < f.lo || i >= f.hi} />
        </div>
        <CodeTrace code={code(kind)} active={f.line} vars={{ lo: f.lo, hi: f.hi, mid: f.mid ?? '—' }} />
      </div>
    </VizFrame>
  )
}
