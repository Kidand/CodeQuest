/**
 * K 个一组翻转链表（LC25）：哑节点 + 「找尾 → 断开 → 组内反转 → 接回 → pre 前移」。
 * 洞见：每组只需要记住 4 个指针 pre / start / end / nxt；哑节点让第一组和其它组的处理完全一致。
 */
import { useMemo, useState } from 'react'
import { CodeTrace, Legend, StepControls, VizFrame, useStepper, type VizTone } from '@/viz'
import { Slider } from '@/components/ui'
import { C, ListLane, type LaneId, type LanePtr } from './common'

const CODE = `def reverse_k_group(head, k):
    dummy = ListNode(0, head)
    pre = dummy
    while True:
        end = pre
        for _ in range(k):          # 向前走 k 步，找本组尾
            end = end.next
            if end is None:
                return dummy.next   # 不足 k 个：保持原样
        start, nxt = pre.next, end.next
        end.next = None             # ① 断开本组
        pre.next = reverse(start)   # ② 组内反转，新头是 end
        start.next = nxt            # ③ 原组头变组尾，接回后面
        pre = start                 # ④ pre 移到下一组之前`

interface Frame {
  order: LaneId[]
  next: Record<string, LaneId | null>
  ptrs: LanePtr[]
  line: number
  group: Set<LaneId>
  done: Set<LaneId>
  edge?: LaneId
  caption: React.ReactNode
}

function build(n: number, k: number): Frame[] {
  const D = 'd'
  const ids: LaneId[] = Array.from({ length: n }, (_, i) => i)
  const next: Record<string, LaneId | null> = { [D]: 0 }
  ids.forEach((i) => (next[i] = (i as number) + 1 < n ? (i as number) + 1 : null))
  const label = (x: LaneId | null | undefined) => (x == null || x === 'R' ? 'None' : x === D ? 'dummy' : String((x as number) + 1))
  const order = (): LaneId[] => {
    const out: LaneId[] = []
    let p: LaneId | null = D
    while (p != null && out.length <= n + 1) {
      out.push(p)
      p = next[String(p)] ?? null
    }
    return out
  }
  const frames: Frame[] = []
  const done = new Set<LaneId>()
  let group = new Set<LaneId>()
  let curOrder = order()
  const snap = (line: number, ptrs: LanePtr[], caption: React.ReactNode, edge?: LaneId) =>
    frames.push({ order: [...curOrder], next: { ...next }, ptrs, line, group: new Set(group), done: new Set(done), edge, caption })

  let pre: LaneId = D
  snap(3, [{ at: pre, label: 'pre', tone: 'info' }], <>哑节点 <C>dummy</C> 接在头前面，<C>pre = dummy</C>。这样第一组的“前驱”也存在，不用特判换头。</>)
  let g = 0
  while (true) {
    // 找本组尾
    let end: LaneId | null = pre
    const members: LaneId[] = []
    for (let s = 0; s < k; s++) {
      end = end == null ? null : (next[String(end)] ?? null)
      if (end == null) break
      members.push(end)
    }
    if (end == null) {
      group = new Set(members)
      snap(
        9,
        [
          { at: pre, label: 'pre', tone: 'info' },
          { at: 'R', label: 'end', tone: 'warning' },
        ],
        <>剩余 {members.length} 个节点不足 k = {k}，<C>end</C> 走到了 None：保持原样，返回 <C>dummy.next</C>。</>,
      )
      break
    }
    g++
    group = new Set(members)
    snap(
      7,
      [
        { at: pre, label: 'pre', tone: 'info' },
        { at: end, label: 'end', tone: 'warning' },
      ],
      <>第 {g} 组：从 pre 向前走 k = {k} 步，<C>end = {label(end)}</C>。凑满 k 个才翻转。</>,
    )
    const start = next[String(pre)] as LaneId
    const nxt = next[String(end)]
    const nxtAt: LaneId = nxt == null ? 'R' : nxt
    const base: LanePtr[] = [
      { at: pre, label: 'pre', tone: 'info' },
      { at: start, label: 'start', tone: 'accent' },
      { at: end, label: 'end', tone: 'warning' },
      { at: nxtAt, label: 'nxt', tone: 'success' },
    ]
    snap(10, base, <>记下 <C>start = {label(start)}</C>（本组头）和 <C>nxt = {label(nxt)}</C>（下一组头），断开前必须先保存 nxt。</>)
    next[String(end)] = null
    snap(11, base, <>① <C>end.next = None</C>：把本组切成一条独立链表，才能直接复用 206 的 <C>reverse</C>。</>, end)
    // 组内反转
    for (let j = members.length - 1; j > 0; j--) next[String(members[j])] = members[j - 1]
    next[String(members[0])] = null
    next[String(pre)] = end
    curOrder = order()
    // 补上尚未接回的后半段，保持显示顺序
    const rest: LaneId[] = []
    let p: LaneId | null | undefined = nxt
    while (p != null) {
      rest.push(p)
      p = next[String(p)]
    }
    curOrder = [...curOrder, ...rest]
    snap(12, base, <>② 组内反转：{members.map(label).join('→')} 变成 {[...members].reverse().map(label).join('→')}，<C>pre.next</C> 指向新头 <C>end</C>。此刻 start.next 还是 None。</>, pre)
    next[String(start)] = nxt
    curOrder = order()
    snap(13, base, <>③ <C>start.next = nxt</C>：原来的组头 {label(start)} 变成组尾，接回下一组 {label(nxt)}。</>, start)
    members.forEach((m) => done.add(m))
    group = new Set()
    pre = start
    snap(14, [{ at: pre, label: 'pre', tone: 'info' }], <>④ <C>pre = start</C>：pre 停在下一组之前，循环继续。</>)
  }
  return frames
}

export function KGroupViz() {
  const n = 8
  const [k, setK] = useState(3)
  const frames = useMemo(() => build(n, k), [k])
  const st = useStepper(frames, { intervalMs: 1200 })
  const f = st.frame
  const tone = (id: LaneId): VizTone | undefined => (f.group.has(id) ? 'accent' : f.done.has(id) ? 'success' : undefined)

  return (
    <VizFrame
      title="K 个一组翻转：找尾、断开、反转、接回"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '本组', tone: 'accent', variant: 'soft' },
            { label: '已翻转的组', tone: 'success', variant: 'soft' },
            { label: '本步改动的边', tone: 'danger', variant: 'line' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Slider className="w-40" label="k" value={k} min={2} max={4} onChange={setK} />
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="min-w-0 py-2">
          <ListLane
            items={f.order.map((id) => ({ id, label: id === 'd' ? 'D' : String((id as number) + 1), tone: id === 'd' ? undefined : tone(id), dashed: id === 'd' }))}
            next={f.next}
            pointers={f.ptrs}
            ghosts={[false, true]}
            edgeTones={f.edge != null ? { [String(f.edge)]: 'danger' } : {}}
          />
        </div>
        <CodeTrace code={CODE} active={f.line} vars={{ k, 已翻转组数: f.done.size / k }} />
      </div>
    </VizFrame>
  )
}
