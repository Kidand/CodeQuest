/**
 * 三路划分（荷兰国旗问题，LC75）：[0, lt) < p，[lt, i) == p，[i, gt] 未处理，(gt, n) > p。
 */
import { useMemo } from 'react'
import { ArrayView, CodeTrace, Legend, StepControls, VizFrame, useStepper, type ArrayPointer, type VizTone } from '@/viz'
import { C } from './common'

const DATA = [2, 0, 2, 1, 1, 0, 2, 1, 0, 1]
const P = 1

const CODE = `def sort_colors(nums):          # 三路划分，pivot = 1
    lt, i, gt = 0, 0, len(nums) - 1
    while i <= gt:
        if nums[i] < 1:
            nums[lt], nums[i] = nums[i], nums[lt]
            lt += 1
            i += 1
        elif nums[i] > 1:
            nums[i], nums[gt] = nums[gt], nums[i]
            gt -= 1             # 换来的元素没看过，i 不动
        else:
            i += 1`

interface Frame {
  vals: number[]
  ids: number[]
  lt: number
  i: number
  gt: number
  line: number
  caption: React.ReactNode
}

function build(): Frame[] {
  const vals = [...DATA]
  const ids = DATA.map((_, k) => k)
  let lt = 0
  let i = 0
  let gt = vals.length - 1
  const fr: Frame[] = []
  const push = (line: number, caption: React.ReactNode) => fr.push({ vals: [...vals], ids: [...ids], lt, i, gt, line, caption })
  push(2, <>四段不变量：<C>[0, lt)</C> 为 0，<C>[lt, i)</C> 为 1，<C>[i, gt]</C> 未处理，<C>(gt, n)</C> 为 2。初始全部未处理。</>)
  while (i <= gt) {
    const v = vals[i]
    if (v < P) {
      ;[vals[lt], vals[i]] = [vals[i], vals[lt]]
      ;[ids[lt], ids[i]] = [ids[i], ids[lt]]
      lt++
      i++
      push(7, <><C>nums[i] = 0</C>：与 <C>nums[lt]</C>（一定是 1 或就是自己）交换，<C>lt</C>、<C>i</C> 都前进。</>)
    } else if (v > P) {
      ;[vals[i], vals[gt]] = [vals[gt], vals[i]]
      ;[ids[i], ids[gt]] = [ids[gt], ids[i]]
      gt--
      push(10, <><C>nums[i] = 2</C>：与 <C>nums[gt]</C> 交换，<C>gt -= 1</C>。换过来的 <C>{vals[i]}</C> 还没检查过，<b>i 不能动</b>。</>)
    } else {
      i++
      push(12, <><C>nums[i] = 1</C>：本来就在中间段，<C>i += 1</C>。</>)
    }
  }
  push(3, <><C>i &gt; gt</C>，未处理区为空，一趟完成：时间 <C>O(n)</C>、空间 <C>O(1)</C>。用在快排里，等于 pivot 的整段不再参与递归。</>)
  return fr
}

const TONE: Record<number, VizTone> = { 0: 'info', 1: 'success', 2: 'warning' }

export function ThreeWayViz() {
  const frames = useMemo(build, [])
  const st = useStepper(frames, { intervalMs: 900 })
  const f = st.frame
  const highlights: Record<number, VizTone> = {}
  f.vals.forEach((v, k) => {
    if (k < f.i || k > f.gt) highlights[k] = TONE[v] // 未处理区 [i, gt] 保持中性色
  })
  const pointers: ArrayPointer[] = [
    { index: f.lt, label: 'lt', tone: 'info' },
    { index: Math.min(f.i, f.vals.length - 1), label: 'i', tone: 'accent', side: 'above' },
    { index: Math.max(0, f.gt), label: 'gt', tone: 'warning' },
  ]

  return (
    <VizFrame
      title="三路划分：一趟扫描分出 < = > 三段"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '0（< pivot）', tone: 'info', variant: 'soft' },
            { label: '1（== pivot）', tone: 'success', variant: 'soft' },
            { label: '2（> pivot）', tone: 'warning', variant: 'soft' },
            { label: '未处理区 [i, gt]（无色）', tone: 'muted', variant: 'soft' },
          ]}
        />
      }
      controls={<StepControls stepper={st} />}
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="flex min-h-[150px] min-w-0 items-center py-2">
          <ArrayView values={f.vals} ids={f.ids} cellSize={40} highlights={highlights} pointers={pointers} />
        </div>
        <CodeTrace code={CODE} active={f.line} vars={{ lt: f.lt, i: f.i, gt: f.gt }} />
      </div>
    </VizFrame>
  )
}
