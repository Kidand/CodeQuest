/**
 * 反转链表（LC206）：迭代三指针 / 递归 两种写法逐帧对照。
 * 洞见：迭代版每轮只改一条边，改之前必须先用 nxt 保存后继；递归版在“归”的阶段让后继指回自己，并断开旧边防环。
 */
import { useMemo, useState } from 'react'
import { CodeTrace, Legend, StepControls, VizFrame, useStepper, type VizTone } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { C, ListLane, type LaneId, type LanePtr } from './common'

type Mode = 'iter' | 'rec'

const CODE: Record<Mode, string> = {
  iter: `def reverse_list(head):
    prev, cur = None, head
    while cur:
        nxt = cur.next      # ① 先保存后继
        cur.next = prev     # ② 改向
        prev = cur          # ③ prev 前进
        cur = nxt           # ④ cur 前进
    return prev             # prev 是新头`,
  rec: `def reverse_list(head):
    if head is None or head.next is None:
        return head             # 到底：最后一个节点就是新头
    new_head = reverse_list(head.next)
    head.next.next = head       # 让后继指回自己
    head.next = None            # 断开旧边（否则成环）
    return new_head`,
}

interface Frame {
  next: Record<string, LaneId | null>
  ptrs: LanePtr[]
  line: number
  done: Set<number>
  focus?: number
  edge?: number
  caption: React.ReactNode
  vars: Record<string, string>
}

const name = (vals: number[], p: LaneId | null | undefined) => (p == null || p === 'L' || p === 'R' ? 'None' : String(vals[p as number]))

function buildIter(vals: number[]): Frame[] {
  const n = vals.length
  const next: Record<string, LaneId | null> = {}
  vals.forEach((_, i) => (next[i] = i + 1 < n ? i + 1 : null))
  const frames: Frame[] = []
  let prev: LaneId = 'L'
  let cur: LaneId = 0
  let nxt: LaneId | undefined
  const done = new Set<number>()
  const snap = (line: number, caption: React.ReactNode, edge?: number) => {
    const ptrs: LanePtr[] = [
      { at: prev, label: 'prev', tone: 'success' },
      { at: cur, label: 'cur', tone: 'accent' },
    ]
    if (nxt !== undefined) ptrs.push({ at: nxt, label: 'nxt', tone: 'warning' })
    frames.push({
      next: { ...next },
      ptrs,
      line,
      done: new Set(done),
      focus: typeof cur === 'number' ? cur : undefined,
      edge,
      caption,
      vars: { prev: name(vals, prev), cur: name(vals, cur), nxt: nxt === undefined ? '—' : name(vals, nxt) },
    })
  }
  snap(2, <>初始化：<C>prev = None</C>，<C>cur = head</C>。prev 指向“已反转部分”的头，一开始为空。</>)
  while (typeof cur === 'number') {
    const c: number = cur
    nxt = next[c] === null ? 'R' : (next[c] as number)
    snap(4, <>① 先存后继 <C>nxt = {name(vals, nxt)}</C>：下一步改向后，<C>cur.next</C> 就丢了，必须先存。</>)
    next[c] = prev === 'L' ? null : prev
    snap(5, <>② 改向 <C>cur.next = prev</C>：节点 {vals[c]} 现在指向 <C>{name(vals, prev)}</C>。每轮只改这一条边。</>, c)
    done.add(c)
    prev = c
    snap(6, <>③ <C>prev = cur</C>：已反转部分扩大到以 {vals[c]} 为头。</>)
    cur = nxt
    snap(7, <>④ <C>cur = nxt</C>：处理下一个节点{cur === 'R' ? '；cur 变成 None，循环结束' : ''}。</>)
  }
  nxt = undefined
  snap(8, <>返回 <C>prev</C>：新头是 {name(vals, prev)}。共 {n} 轮、每轮 O(1)，总 <b>O(n) 时间、O(1) 空间</b>。</>)
  return frames
}

function buildRec(vals: number[]): Frame[] {
  const n = vals.length
  const next: Record<string, LaneId | null> = {}
  vals.forEach((_, i) => (next[i] = i + 1 < n ? i + 1 : null))
  const frames: Frame[] = []
  const done = new Set<number>()
  const stack = (d: number) => vals.slice(0, d + 1).map((v) => `f(${v})`).join(' → ')
  const snap = (line: number, head: number, newHead: number | undefined, caption: React.ReactNode, edge?: number) => {
    const ptrs: LanePtr[] = [{ at: head, label: 'head', tone: 'accent' }]
    if (newHead !== undefined) ptrs.push({ at: newHead, label: 'new_head', tone: 'success' })
    frames.push({
      next: { ...next },
      ptrs,
      line,
      done: new Set(done),
      focus: head,
      edge,
      caption,
      vars: { 调用栈深度: String(head + 1), head: String(vals[head]), new_head: newHead === undefined ? '—' : String(vals[newHead]) },
    })
  }
  for (let d = 0; d < n; d++) {
    snap(2, d, undefined, <>“递”：调用 <C>reverse_list({vals[d]})</C>，栈：{stack(d)}。{d < n - 1 ? '还有后继，继续向下递归。' : ''}</>)
  }
  done.add(n - 1)
  snap(3, n - 1, n - 1, <>到达最后一个节点 {vals[n - 1]}：<C>head.next is None</C>，它就是新头，原样返回。</>)
  for (let d = n - 2; d >= 0; d--) {
    snap(4, d, n - 1, <>“归”到 <C>head = {vals[d]}</C>：子链表 {vals[d + 1]}…{vals[n - 1]} 已经反转好，new_head = {vals[n - 1]}。注意 <C>head.next</C> 仍指向 {vals[d + 1]}——它现在是子链表的<b>尾</b>。</>)
    next[d + 1] = d
    done.add(d)
    snap(5, d, n - 1, <><C>head.next.next = head</C>：让 {vals[d + 1]} 指回 {vals[d]}。此刻 {vals[d]} ⇄ {vals[d + 1]} 互指，形成一个二元环！</>, d + 1)
    next[d] = null
    snap(6, d, n - 1, <><C>head.next = None</C>：断开旧边，{vals[d]} 成为新的尾。漏掉这一行，原头节点会和第二个节点成环。</>, d)
    snap(7, d, n - 1, <>返回 new_head = {vals[n - 1]}（每一层都原样向上传递同一个新头）。</>)
  }
  return frames
}

export function ReverseListViz() {
  const [n, setN] = useState(5)
  const [mode, setMode] = useState<Mode>('iter')
  const vals = useMemo(() => Array.from({ length: n }, (_, i) => i + 1), [n])
  const frames = useMemo(() => (mode === 'iter' ? buildIter(vals) : buildRec(vals)), [vals, mode])
  const st = useStepper(frames, { intervalMs: 1000 })
  const f = st.frame
  const tone = (i: number): VizTone | undefined => (i === f.focus ? 'accent' : f.done.has(i) ? 'success' : undefined)

  return (
    <VizFrame
      title="反转链表：每轮只改一条边"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '当前节点', tone: 'accent', variant: 'soft' },
            { label: '已反转部分', tone: 'success', variant: 'soft' },
            { label: '本步改动的边', tone: 'danger', variant: 'line' },
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
              { value: 'iter', label: '迭代（三指针）' },
              { value: 'rec', label: '递归' },
            ]}
          />
          <Slider className="w-40" label="节点数" value={n} min={3} max={7} onChange={setN} />
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="min-w-0 py-2">
          <ListLane
            items={vals.map((v, i) => ({ id: i, label: String(v), tone: tone(i) }))}
            next={f.next}
            pointers={f.ptrs}
            ghosts={mode === 'iter' ? [true, true] : [false, false]}
            edgeTones={f.edge != null ? { [f.edge]: 'danger' } : {}}
          />
        </div>
        <CodeTrace code={CODE[mode]} active={f.line} vars={f.vars} />
      </div>
    </VizFrame>
  )
}
