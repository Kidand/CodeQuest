/**
 * 小顶堆的两个基本操作：push（末尾加入 + sift up）与 pop（末尾补到堆顶 + sift down）。
 * 数组 ↔ 树双视图同步；「以结果为新堆」可以连续操作。
 */
import { useMemo, useState } from 'react'
import { CodeTrace, Legend, StepControls, VizFrame, useStepper, type VizTone } from '@/viz'
import { Button, Segmented, Slider } from '@/components/ui'
import { HeapDual } from './HeapDual'
import { C } from './common'

type Op = 'push' | 'pop'
const INIT = [2, 4, 3, 8, 5, 6, 7, 10, 9]

const CODE: Record<Op, string> = {
  push: `def heappush(h, x):
    h.append(x)                    # 放到末尾（完全二叉树的下一个位置）
    i = len(h) - 1
    while i > 0:
        p = (i - 1) // 2           # 父节点
        if h[p] <= h[i]:
            break                  # 堆序满足，停止
        h[p], h[i] = h[i], h[p]    # 比父亲小：上浮
        i = p`,
  pop: `def heappop(h):
    last = h.pop()
    if not h:
        return last
    top, h[0] = h[0], last         # 末尾元素补到堆顶
    i, n = 0, len(h)
    while 2 * i + 1 < n:
        c = 2 * i + 1              # 左孩子
        if c + 1 < n and h[c + 1] < h[c]:
            c += 1                 # 选较小的孩子
        if h[i] <= h[c]:
            break
        h[i], h[c] = h[c], h[i]    # 下沉
        i = c
    return top`,
}

interface Heap {
  vals: number[]
  ids: number[]
  next: number
}
interface Frame {
  vals: number[]
  ids: number[]
  cur?: number
  cmp?: number
  line: number
  popped?: number
  caption: React.ReactNode
}

function buildPush(h: Heap, x: number): { frames: Frame[]; result: Heap } {
  const vals = [...h.vals, x]
  const ids = [...h.ids, h.next]
  const fr: Frame[] = []
  let i = vals.length - 1
  fr.push({ vals: [...vals], ids: [...ids], cur: i, line: 2, caption: <><C>{x}</C> 放到数组末尾 <C>[{i}]</C>，也就是树的最后一层最左边的空位。可能破坏了与父节点的堆序。</> })
  while (i > 0) {
    const p = (i - 1) >> 1
    if (vals[p] <= vals[i]) {
      fr.push({ vals: [...vals], ids: [...ids], cur: i, cmp: p, line: 7, caption: <>父节点 <C>h[{p}] = {vals[p]} ≤ {vals[i]}</C>：堆序满足，停止。</> })
      break
    }
    fr.push({ vals: [...vals], ids: [...ids], cur: i, cmp: p, line: 6, caption: <>父节点 <C>h[{p}] = {vals[p]} &gt; {vals[i]}</C>：违反小顶堆性质。</> })
    ;[vals[p], vals[i]] = [vals[i], vals[p]]
    ;[ids[p], ids[i]] = [ids[i], ids[p]]
    i = p
    fr.push({ vals: [...vals], ids: [...ids], cur: i, line: 8, caption: <>交换，<C>{x}</C> 上浮到 <C>[{i}]</C>。{i === 0 ? '已到堆顶。' : ''}</> })
  }
  fr.push({ vals: [...vals], ids: [...ids], cur: i, line: 9, caption: <>完成。上浮最多走树高 <C>⌊log₂n⌋</C> 步：<C>O(log n)</C>。</> })
  return { frames: fr, result: { vals, ids, next: h.next + 1 } }
}

function buildPop(h: Heap): { frames: Frame[]; result: Heap } {
  const vals = [...h.vals]
  const ids = [...h.ids]
  const fr: Frame[] = []
  if (!vals.length) return { frames: [{ vals, ids, line: 1, caption: <>空堆，无法 pop。</> }], result: h }
  const top = vals[0]
  fr.push({ vals: [...vals], ids: [...ids], cur: 0, line: 2, caption: <>要弹出堆顶 <C>{top}</C>（最小值）。直接删掉会留下空洞，所以先拿<b>末尾</b>元素来补。</> })
  const lastV = vals.pop()!
  const lastId = ids.pop()!
  if (!vals.length) {
    fr.push({ vals, ids, line: 4, popped: top, caption: <>堆只剩一个元素，直接返回 <C>{top}</C>。</> })
    return { frames: fr, result: { vals, ids, next: h.next } }
  }
  vals[0] = lastV
  ids[0] = lastId
  fr.push({ vals: [...vals], ids: [...ids], cur: 0, line: 5, popped: top, caption: <>末尾的 <C>{lastV}</C> 补到堆顶，现在要让它下沉到合适的位置。</> })
  let i = 0
  const n = vals.length
  while (2 * i + 1 < n) {
    let c = 2 * i + 1
    if (c + 1 < n && vals[c + 1] < vals[c]) c++
    if (vals[i] <= vals[c]) {
      fr.push({ vals: [...vals], ids: [...ids], cur: i, cmp: c, line: 12, popped: top, caption: <>较小的孩子 <C>h[{c}] = {vals[c]} ≥ {vals[i]}</C>：堆序满足，停止。</> })
      break
    }
    fr.push({ vals: [...vals], ids: [...ids], cur: i, cmp: c, line: 10, popped: top, caption: <>两个孩子中较小的是 <C>h[{c}] = {vals[c]}</C>，它比 <C>{vals[i]}</C> 小——必须和<b>较小</b>的孩子换，否则换上去的较大孩子会比另一个孩子大。</> })
    ;[vals[i], vals[c]] = [vals[c], vals[i]]
    ;[ids[i], ids[c]] = [ids[c], ids[i]]
    i = c
    fr.push({ vals: [...vals], ids: [...ids], cur: i, line: 14, popped: top, caption: <>交换，<C>{vals[i]}</C> 下沉到 <C>[{i}]</C>。</> })
  }
  fr.push({ vals: [...vals], ids: [...ids], cur: i, line: 15, popped: top, caption: <>完成，返回 <C>{top}</C>。下沉每层 2 次比较，最多 <C>⌊log₂n⌋</C> 层：<C>O(log n)</C>。</> })
  return { frames: fr, result: { vals, ids, next: h.next } }
}

export function HeapOpsViz() {
  const [heap, setHeap] = useState<Heap>({ vals: INIT, ids: INIT.map((_, k) => k), next: INIT.length })
  const [op, setOp] = useState<Op>('push')
  const [x, setX] = useState(1)
  const built = useMemo(() => (op === 'push' ? buildPush(heap, x) : buildPop(heap)), [heap, op, x])
  const st = useStepper(built.frames, { intervalMs: 1000 })
  const f = st.frame
  const tones: Record<number, VizTone> = {}
  if (f.cmp != null) tones[f.cmp] = 'warning'
  if (f.cur != null) tones[f.cur] = 'accent'
  const full = op === 'push' && heap.vals.length >= 15

  return (
    <VizFrame
      title="push 上浮、pop 下沉：每次只沿一条路径走 O(log n) 步"
      caption={
        <>
          {f.caption}
          {f.popped != null && <span className="ml-2 text-muted">（已弹出：{f.popped}）</span>}
        </>
      }
      legend={
        <Legend
          items={[
            { label: '正在移动的元素', tone: 'accent', variant: 'soft' },
            { label: '比较对象（父 / 较小孩子）', tone: 'warning', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Segmented
            size="sm"
            value={op}
            onChange={setOp}
            options={[
              { value: 'push', label: 'push(x)' },
              { value: 'pop', label: 'pop()' },
            ]}
          />
          {op === 'push' && <Slider className="min-w-[140px] flex-1" label="x" value={x} min={0} max={12} onChange={setX} />}
          <Button size="sm" variant="secondary" disabled={!st.atEnd || full || (op === 'pop' && !heap.vals.length)} onClick={() => setHeap(built.result)}>
            以结果为新堆
          </Button>
          <Button size="sm" variant="ghost" onClick={() => setHeap({ vals: INIT, ids: INIT.map((_, k) => k), next: INIT.length })}>
            重置堆
          </Button>
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <HeapDual vals={f.vals} ids={f.ids} tones={tones} />
        <CodeTrace code={CODE[op]} active={f.line} maxHeight={330} />
      </div>
    </VizFrame>
  )
}
