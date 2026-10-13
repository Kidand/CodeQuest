/**
 * 递归三序：同一条“访问路线”，每个节点被经过 3 次，三种遍历只是“在第几次经过时输出”。
 */
import { useMemo, useState } from 'react'
import { CodeTrace, Legend, StepControls, TreeView, VizFrame, useStepper } from '@/viz'
import { Segmented } from '@/components/ui'
import { C, OutRow, allNodes, fromLevel, toView, type BT } from './common'

type Order = 'pre' | 'in' | 'post'
const SAMPLE = [1, 2, 3, 4, 5, null, 6, null, null, 7]
const MARK = ['①', '②', '③']
const POS_NAME = ['前序', '中序', '后序']
const ORDER_IDX: Record<Order, number> = { pre: 0, in: 1, post: 2 }

function code(order: Order) {
  const k = ORDER_IDX[order]
  const lines = [
    '# ① 前序位置：第一次到达 node',
    '# ② 中序位置：左子树刚处理完',
    '# ③ 后序位置：左右子树都处理完',
  ].map((c, i) => (i === k ? `res.append(node.val)   ${c}` : c))
  return `def dfs(node):
    if node is None:
        return
    ${lines[0]}
    dfs(node.left)
    ${lines[1]}
    dfs(node.right)
    ${lines[2]}`
}

interface Frame {
  cur: number | null
  path: number[]
  passed: Record<number, number>
  out: number[]
  emit?: boolean
  line: number
  caption: React.ReactNode
}

function buildFrames(root: BT, order: Order): Frame[] {
  const frames: Frame[] = []
  const passed: Record<number, number> = {}
  const out: number[] = []
  const path: number[] = []
  const k = ORDER_IDX[order]
  frames.push({ cur: null, path: [], passed: {}, out: [], line: 1, caption: <>调用 <C>dfs(root)</C>。注意观察：每个节点都会被<b>经过 3 次</b>。</> })
  const visit = (n: BT) => {
    path.push(n.id)
    for (let p = 0; p < 3; p++) {
      passed[n.id] = p + 1
      const emit = p === k
      if (emit) out.push(n.val)
      const why =
        p === 0 ? <>第 1 次到达 <C>{n.val}</C></> : p === 1 ? <>左子树处理完，回到 <C>{n.val}</C>（第 2 次）</> : <>右子树处理完，回到 <C>{n.val}</C>（第 3 次）</>
      frames.push({
        cur: n.id,
        path: [...path],
        passed: { ...passed },
        out: [...out],
        emit,
        line: 4 + p * 2,
        caption: (
          <>
            {why}，处于{MARK[p]}{POS_NAME[p]}位置
            {emit ? <>：<b>输出 {n.val}</b>。</> : <>：{POS_NAME[k]}遍历不在这里输出。</>}
            {p === 0 && !n.left && <> 左孩子为空，<C>dfs(None)</C> 直接返回。</>}
            {p === 1 && !n.right && <> 右孩子为空，直接返回。</>}
          </>
        ),
      })
      if (p === 0 && n.left) visit(n.left)
      if (p === 1 && n.right) visit(n.right)
    }
    path.pop()
  }
  visit(root)
  frames.push({
    cur: null,
    path: [],
    passed: { ...passed },
    out: [...out],
    line: 8,
    caption: (
      <>
        结束：{POS_NAME[k]}遍历结果 <C>[{out.join(', ')}]</C>。路线完全相同，换一种遍历只是把 <C>res.append</C> 挪到另一个位置。
      </>
    ),
  })
  return frames
}

export function TraversalViz() {
  const root = useMemo(() => fromLevel(SAMPLE)!, [])
  const [order, setOrder] = useState<Order>('pre')
  const frames = useMemo(() => buildFrames(root, order), [root, order])
  const st = useStepper(frames, { intervalMs: 900 })
  const f = st.frame
  const onPath = new Set(f.path)
  const emitted = new Set(allNodes(root).filter((n) => f.out.includes(n.val)).map((n) => n.id))
  const view = toView(root, {
    tone: (n) => (n.id === f.cur ? 'accent' : onPath.has(n.id) ? 'info' : emitted.has(n.id) ? 'success' : undefined),
    sub: (n) => (f.passed[n.id] ? MARK.slice(0, f.passed[n.id]).join('') : undefined),
  })
  return (
    <VizFrame
      title="递归三序：同一路线，三个输出时机"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '当前节点', tone: 'accent', variant: 'soft' },
            { label: '调用栈上的祖先', tone: 'info', variant: 'soft' },
            { label: '已输出', tone: 'success', variant: 'soft' },
            { label: '①②③ = 已经过的次数', tone: 'muted', variant: 'soft' },
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
              { value: 'pre', label: '前序 根左右' },
              { value: 'in', label: '中序 左根右' },
              { value: 'post', label: '后序 左右根' },
            ]}
          />
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="min-w-0 space-y-3">
          <div className="thin-scroll overflow-x-auto py-1">
            <TreeView root={view} nodeSize={34} levelGap={58} siblingGap={12} />
          </div>
          <OutRow label="res =" values={f.out} lastTone={f.emit ? 'success' : 'muted'} />
          <div className="font-mono text-xs text-muted">
            调用栈：{f.path.length ? f.path.map((id) => allNodes(root).find((n) => n.id === id)!.val).join(' → ') : '（空）'}
          </div>
        </div>
        <CodeTrace code={code(order)} active={f.line} vars={{ node: f.cur == null ? '—' : allNodes(root).find((n) => n.id === f.cur)!.val }} />
      </div>
    </VizFrame>
  )
}
