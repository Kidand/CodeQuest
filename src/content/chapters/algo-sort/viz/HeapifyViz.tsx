/**
 * O(n) 建堆：从最后一个非叶子节点 n//2-1 开始，倒序对每个节点做 sift down。
 * 对比“逐个 push”的比较次数，直观感受 O(n) 与 O(n log n) 的差别。
 */
import { useMemo, useState } from 'react'
import { CodeTrace, Legend, StepControls, VizFrame, useStepper, type VizTone } from '@/viz'
import { Segmented } from '@/components/ui'
import { HeapDual } from './HeapDual'
import { C } from './common'

type Input = 'random' | 'desc'
const INPUTS: Record<Input, number[]> = {
  random: [9, 4, 7, 1, 8, 2, 6, 3, 5, 0],
  desc: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
}

const CODE = `def heapify(h):
    n = len(h)
    for i in range(n // 2 - 1, -1, -1):   # 从最后一个非叶子节点倒序
        while 2 * i + 1 < n:              # 对 i 做 sift down
            c = 2 * i + 1
            if c + 1 < n and h[c + 1] < h[c]:
                c += 1
            if h[i] <= h[c]:
                break
            h[i], h[c] = h[c], h[i]
            i = c`

interface Frame {
  vals: number[]
  ids: number[]
  root?: number
  cur?: number
  cmp?: number
  done: number // 下标 >= done 的子树已经是堆
  cmps: number
  line: number
  caption: React.ReactNode
}

function pushCount(a: number[]) {
  // 逐个 push 建堆的比较次数（对照）
  const h: number[] = []
  let cmps = 0
  for (const x of a) {
    h.push(x)
    let i = h.length - 1
    while (i > 0) {
      const p = (i - 1) >> 1
      cmps++
      if (h[p] <= h[i]) break
      ;[h[p], h[i]] = [h[i], h[p]]
      i = p
    }
  }
  return cmps
}

function build(input: number[]): Frame[] {
  const vals = [...input]
  const ids = input.map((_, k) => k)
  const n = vals.length
  const fr: Frame[] = []
  let cmps = 0
  const start = (n >> 1) - 1
  fr.push({ vals: [...vals], ids: [...ids], done: start + 1, cmps, line: 3, caption: <>下标 <C>≥ n//2 = {n >> 1}</C> 的都是叶子，本身就是合法的堆（绿色）。从 <C>i = {start}</C> 开始倒序 sift down。</> })
  for (let s = start; s >= 0; s--) {
    let i = s
    fr.push({ vals: [...vals], ids: [...ids], root: s, cur: i, done: s + 1, cmps, line: 3, caption: <>处理 <C>i = {s}</C>（值 {vals[s]}）：它的两棵子树已经是堆，只需让它下沉。</> })
    while (2 * i + 1 < n) {
      let c = 2 * i + 1
      if (c + 1 < n) {
        cmps++
        if (vals[c + 1] < vals[c]) c++
      }
      cmps++
      if (vals[i] <= vals[c]) {
        fr.push({ vals: [...vals], ids: [...ids], root: s, cur: i, cmp: c, done: s, cmps, line: 9, caption: <><C>{vals[i]} ≤ 较小孩子 {vals[c]}</C>，停止。以 <C>[{s}]</C> 为根的子树已成堆。</> })
        break
      }
      ;[vals[i], vals[c]] = [vals[c], vals[i]]
      ;[ids[i], ids[c]] = [ids[c], ids[i]]
      fr.push({ vals: [...vals], ids: [...ids], root: s, cur: c, cmp: i, done: s + 1, cmps, line: 10, caption: <>与较小孩子交换：<C>{vals[c]}</C> 下沉到 <C>[{c}]</C>。</> })
      i = c
    }
  }
  fr.push({ vals: [...vals], ids: [...ids], done: 0, cmps, line: 3, caption: <>建堆完成，共 <C>{cmps}</C> 次比较；逐个 push 需要 <C>{pushCount(input)}</C> 次。n 越大差距越明显：前者 <C>≤ 2n</C>，后者最坏 <C>≈ n log₂n</C>。</> })
  return fr
}

export function HeapifyViz() {
  const [input, setInput] = useState<Input>('random')
  const frames = useMemo(() => build(INPUTS[input]), [input])
  const st = useStepper(frames, { intervalMs: 900 })
  const f = st.frame
  const tones: Record<number, VizTone> = {}
  for (let k = f.done; k < f.vals.length; k++) tones[k] = 'success'
  if (f.cmp != null) tones[f.cmp] = 'warning'
  if (f.cur != null) tones[f.cur] = 'accent'

  return (
    <VizFrame
      title="自底向上建堆：O(n)，不是 O(n log n)"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '已成堆的子树（含叶子）', tone: 'success', variant: 'soft' },
            { label: '正在下沉', tone: 'accent', variant: 'soft' },
            { label: '比较的孩子', tone: 'warning', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Segmented
            size="sm"
            value={input}
            onChange={setInput}
            options={[
              { value: 'random', label: '随机输入' },
              { value: 'desc', label: '降序输入（逐个 push 的最坏情况）' },
            ]}
          />
          <span className="font-mono text-xs text-muted">比较次数 {f.cmps}</span>
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <HeapDual vals={f.vals} ids={f.ids} tones={tones} />
        <CodeTrace code={CODE} active={f.line} vars={{ i: f.root ?? '—', 比较: f.cmps }} />
      </div>
    </VizFrame>
  )
}
