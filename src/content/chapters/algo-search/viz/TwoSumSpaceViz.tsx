/**
 * 对撞指针的正确性：把所有 (i, j) 对画成“搜索空间矩阵”（上三角）。
 * 每一步要么排除一整行（l += 1），要么排除一整列（r -= 1），所以 O(n) 步就能看完 O(n²) 个候选且不漏解。
 */
import { useMemo, useState } from 'react'
import { ArrayView, CodeTrace, Legend, MatrixView, StepControls, VizFrame, useStepper, type ArrayPointer } from '@/viz'
import { Slider } from '@/components/ui'
import { C } from './common'

const NUMS = [1, 3, 4, 6, 8, 11, 13, 15]

const CODE = `def two_sum(nums, target):        # nums 升序
    l, r = 0, len(nums) - 1
    while l < r:
        s = nums[l] + nums[r]
        if s == target:
            return [l, r]
        if s < target:
            l += 1      # 第 l 行整行排除
        else:
            r -= 1      # 第 r 列整列排除
    return []`

interface Frame {
  l: number
  r: number
  line: number
  found?: boolean
  caption: React.ReactNode
}

function build(target: number): Frame[] {
  const fr: Frame[] = []
  let l = 0
  let r = NUMS.length - 1
  fr.push({ l, r, line: 2, caption: <>右上角的矩阵列出所有 <C>i &lt; j</C> 的和 <C>nums[i] + nums[j]</C>：行从左到右递增、列从下到上递增。从右上角 <C>(0, {r})</C> 出发。</> })
  while (l < r) {
    const s = NUMS[l] + NUMS[r]
    if (s === target) {
      fr.push({ l, r, line: 6, found: true, caption: <>✅ <C>{NUMS[l]} + {NUMS[r]} = {target}</C>，返回 <C>[{l}, {r}]</C>。</> })
      return fr
    }
    if (s < target) {
      fr.push({ l, r, line: 8, caption: <><C>{NUMS[l]} + {NUMS[r]} = {s} &lt; {target}</C>：<C>nums[{l}]</C> 配最大的 <C>nums[{r}]</C> 都不够，配更小的更不够 ⇒ <b>第 {l} 行全部排除</b>，<C>l = {l + 1}</C>。</> })
      l++
    } else {
      fr.push({ l, r, line: 10, caption: <><C>{NUMS[l]} + {NUMS[r]} = {s} &gt; {target}</C>：<C>nums[{r}]</C> 配最小的 <C>nums[{l}]</C> 都太大 ⇒ <b>第 {r} 列全部排除</b>，<C>r = {r - 1}</C>。</> })
      r--
    }
  }
  fr.push({ l, r, line: 11, caption: <><C>l == r</C>，矩阵中所有格子都已被排除，确认<b>无解</b>。只走了 {fr.length - 1} 步，却覆盖了全部 {(NUMS.length * (NUMS.length - 1)) / 2} 个数对。</> })
  return fr
}

export function TwoSumSpaceViz() {
  const [target, setTarget] = useState(17)
  const frames = useMemo(() => build(target), [target])
  const st = useStepper(frames, { intervalMs: 1100 })
  const f = st.frame
  const data = NUMS.map((a, i) => NUMS.map((b, j) => (j > i ? a + b : null)))
  const pointers: ArrayPointer[] = [
    { index: f.l, label: 'l', tone: 'info' },
    { index: f.r, label: 'r', tone: 'warning' },
  ]

  return (
    <VizFrame
      title="对撞指针：每一步排除一整行或一整列"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '当前 (l, r)', tone: 'accent', variant: 'soft' },
            { label: '找到', tone: 'success', variant: 'soft' },
            { label: '已排除', tone: 'muted', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Slider className="min-w-[180px] flex-1" label="target" value={target} min={4} max={28} onChange={setTarget} />
        </>
      }
    >
      <div className="space-y-4">
        <ArrayView values={NUMS} cellSize={40} pointers={pointers} highlights={{ [f.l]: f.found ? 'success' : 'info', [f.r]: f.found ? 'success' : 'warning' }} />
        <div className="grid items-start gap-4 @3xl:grid-cols-2">
          <div className="min-w-0">
            <div className="mb-1 text-xs text-muted">行 = nums[i]，列 = nums[j]，格子 = 两数之和</div>
            <MatrixView
              data={data}
              rowLabels={NUMS.map(String)}
              colLabels={NUMS.map(String)}
              scale="none"
              cellSize={38}
              format={(v) => String(v)}
              highlight={(i, j) =>
                i === f.l && j === f.r ? (f.found ? 'success' : 'accent') : j > i && (i < f.l || j > f.r) ? 'muted' : null
              }
            />
          </div>
          <CodeTrace code={CODE} active={f.line} vars={{ l: f.l, r: f.r, 's': f.l < f.r ? NUMS[f.l] + NUMS[f.r] : '—', target }} />
        </div>
      </div>
    </VizFrame>
  )
}
