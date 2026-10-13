/**
 * 统一的显式栈写法（标记法）：栈里放 (node, visited)。
 * 第一次弹出“展开”，按访问顺序的逆序压栈；第二次弹出（visited=True）才输出。
 * 三种遍历只差 (node, True) 这一行压栈的位置。
 */
import { useMemo, useState } from 'react'
import { CodeTrace, Legend, StepControls, TreeView, VizFrame, useStepper } from '@/viz'
import { Segmented } from '@/components/ui'
import { C, OutRow, StackList, fromLevel, toView, type BT } from './common'

type Order = 'pre' | 'in' | 'post'
const SAMPLE = [1, 2, 3, 4, 5, null, 6, null, null, 7]

const CODE = `def traverse(root, order):
    res, stack = [], [(root, False)] if root else []
    while stack:
        node, visited = stack.pop()
        if visited:                  # 第二次见到：输出
            res.append(node.val)
            continue
        # 第一次见到：按“访问顺序的逆序”压栈
        if order == "post": stack.append((node, True))
        if node.right: stack.append((node.right, False))
        if order == "in":   stack.append((node, True))
        if node.left:  stack.append((node.left, False))
        if order == "pre":  stack.append((node, True))
    return res`

const ORDER_LINE: Record<Order, number> = { post: 9, in: 11, pre: 13 }
const ORDER_SEQ: Record<Order, string> = { pre: '根 → 左 → 右', in: '左 → 根 → 右', post: '左 → 右 → 根' }

interface Item {
  key: number
  node: BT
  visited: boolean
}
interface Frame {
  stack: Item[]
  cur?: BT
  out: BT[]
  line: number | number[]
  caption: React.ReactNode
  maxDepth: number
}

function buildFrames(root: BT, order: Order): Frame[] {
  let key = 0
  const stack: Item[] = [{ key: key++, node: root, visited: false }]
  const out: BT[] = []
  const frames: Frame[] = []
  let maxDepth = 1
  frames.push({ stack: [...stack], out: [], line: 2, maxDepth, caption: <>初始化：栈里只有 <C>({root.val}, False)</C>。想要的顺序是 {ORDER_SEQ[order]}。</> })
  while (stack.length) {
    const it = stack.pop()!
    const n = it.node
    if (it.visited) {
      out.push(n)
      frames.push({ stack: [...stack], cur: n, out: [...out], line: [4, 6], maxDepth, caption: <>弹出 <C>({n.val}, True)</C>：这是<b>第二次</b>见到它，输出 <b>{n.val}</b>。</> })
      continue
    }
    const pushed: string[] = []
    const push = (x: BT | null, v: boolean) => {
      if (!x) return
      stack.push({ key: key++, node: x, visited: v })
      pushed.push(`(${x.val}, ${v ? 'True' : 'False'})`)
    }
    if (order === 'post') push(n, true)
    push(n.right, false)
    if (order === 'in') push(n, true)
    push(n.left, false)
    if (order === 'pre') push(n, true)
    maxDepth = Math.max(maxDepth, stack.length)
    const lines = [4, ORDER_LINE[order]]
    if (n.right) lines.push(10)
    if (n.left) lines.push(12)
    frames.push({
      stack: [...stack],
      cur: n,
      out: [...out],
      line: lines,
      maxDepth,
      caption: (
        <>
          弹出 <C>({n.val}, False)</C>：第一次见到，<b>不输出</b>，只“展开”。依次压入 {pushed.map((p, i) => <span key={i}><C>{p}</C>{i < pushed.length - 1 ? ' ' : ''}</span>)}
          ——后压的先弹，所以弹出顺序正好是 {ORDER_SEQ[order]}。
        </>
      ),
    })
  }
  frames.push({
    stack: [],
    out: [...out],
    line: 14,
    maxDepth,
    caption: <>栈空，结束：<C>[{out.map((x) => x.val).join(', ')}]</C>。每个节点恰好入栈 2 次、出栈 2 次，时间 O(n)；栈的最大长度 {maxDepth}，与树高同阶。</>,
  })
  return frames
}

export function StackTraversalViz() {
  const root = useMemo(() => fromLevel(SAMPLE)!, [])
  const [order, setOrder] = useState<Order>('in')
  const frames = useMemo(() => buildFrames(root, order), [root, order])
  const st = useStepper(frames, { intervalMs: 1100 })
  const f = st.frame
  const inStack = new Map(f.stack.map((it) => [it.node.id, it.visited]))
  const done = new Set(f.out.map((x) => x.id))
  const view = toView(root, {
    tone: (n) => (n.id === f.cur?.id ? 'accent' : done.has(n.id) ? 'success' : inStack.get(n.id) ? 'warning' : inStack.has(n.id) ? 'info' : undefined),
  })
  return (
    <VizFrame
      title="统一显式栈：只差一行的位置"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '刚弹出', tone: 'accent', variant: 'soft' },
            { label: '栈中 (x, False)：待展开', tone: 'info', variant: 'soft' },
            { label: '栈中 (x, True)：待输出', tone: 'warning', variant: 'soft' },
            { label: '已输出', tone: 'success', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Segmented
            size="sm"
            value={order}
            onChange={setOrder}
            options={[
              { value: 'pre', label: '前序' },
              { value: 'in', label: '中序' },
              { value: 'post', label: '后序' },
            ]}
          />
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="min-w-0 space-y-3">
          <div className="flex items-start gap-3">
            <div className="thin-scroll min-w-0 flex-1 overflow-x-auto py-1">
              <TreeView root={view} nodeSize={32} levelGap={54} siblingGap={10} />
            </div>
            <StackList
              className="w-[92px] shrink-0"
              title={<>stack（最大 {f.maxDepth}）</>}
              items={f.stack.map((it) => ({ key: it.key, text: `${it.node.val}${it.visited ? ' ✓' : ''}`, tone: it.visited ? 'warning' : 'info' }))}
            />
          </div>
          <OutRow label="res =" values={f.out.map((x) => x.val)} lastTone={f.line instanceof Array && f.line.includes(6) ? 'success' : 'muted'} />
        </div>
        <CodeTrace code={CODE} active={f.line} vars={{ order: `"${order}"`, 'len(stack)': f.stack.length }} />
      </div>
    </VizFrame>
  )
}
