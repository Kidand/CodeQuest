/**
 * algo-graph 章节内复用的小组件与工具（仅本章使用）。
 *  - C：行内代码胶囊（caption 里用）
 *  - BT：二叉树节点；fromLevel / toLevel：LeetCode 层序列表 ↔ 树；toView：转成 TreeView 的数据
 *  - StackList：竖向的栈 / 队列展示（栈顶在上）
 */
import type { ReactNode } from 'react'
import { motion } from 'motion/react'
import { cn } from '@/lib/utils'
import { toneColor, toneSoft, type TreeNode, type VizTone } from '@/viz'

export const C = ({ children }: { children: ReactNode }) => (
  <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[12.5px]">{children}</code>
)

export interface BT {
  /** 稳定 id（层序列表中的下标），用作动画 key */
  id: number
  val: number
  left: BT | null
  right: BT | null
}

/** LeetCode 层序列表 → 树（None 只给“存在的节点”的空孩子占位） */
export function fromLevel(list: (number | null)[]): BT | null {
  if (!list.length || list[0] == null) return null
  const root: BT = { id: 0, val: list[0], left: null, right: null }
  const q: BT[] = [root]
  let i = 1
  while (q.length && i < list.length) {
    const node = q.shift()!
    for (const side of ['left', 'right'] as const) {
      if (i < list.length && list[i] != null) {
        const child: BT = { id: i, val: list[i]!, left: null, right: null }
        node[side] = child
        q.push(child)
      }
      i++
    }
  }
  return root
}

/** 树 → LeetCode 层序列表（去掉末尾 None） */
export function toLevel(root: BT | null): (number | null)[] {
  const out: (number | null)[] = []
  const q: (BT | null)[] = [root]
  while (q.length) {
    const x = q.shift()!
    if (!x) {
      out.push(null)
      continue
    }
    out.push(x.val)
    q.push(x.left, x.right)
  }
  while (out.length && out[out.length - 1] == null) out.pop()
  return out
}

/** 前序收集全部节点 */
export function allNodes(root: BT | null): BT[] {
  const out: BT[] = []
  const st: BT[] = root ? [root] : []
  while (st.length) {
    const x = st.pop()!
    out.push(x)
    if (x.right) st.push(x.right)
    if (x.left) st.push(x.left)
  }
  return out
}

export interface ViewOpts {
  tone?: (n: BT) => VizTone | undefined
  sub?: (n: BT) => string | undefined
  dim?: (n: BT) => boolean
  edgeTone?: (n: BT) => VizTone | undefined
  label?: (n: BT) => ReactNode
}

/** BT → TreeView 数据；单孩子时用 null 占位，保证左右位置正确 */
export function toView(root: BT, o: ViewOpts = {}): TreeNode {
  const kids = root.left || root.right ? [root.left ? toView(root.left, o) : null, root.right ? toView(root.right, o) : null] : undefined
  return {
    id: root.id,
    label: o.label ? o.label(root) : root.val,
    tone: o.tone?.(root),
    sub: o.sub?.(root),
    dim: o.dim?.(root),
    edgeTone: o.edgeTone?.(root),
    children: kids,
  }
}

/** 竖向栈 / 横向队列。items 从底到顶（栈）或从队头到队尾（队列） */
export function StackList({
  items,
  title,
  kind = 'stack',
  empty = '空',
  className,
}: {
  items: { key: string | number; text: ReactNode; tone?: VizTone }[]
  title: ReactNode
  kind?: 'stack' | 'queue'
  empty?: string
  className?: string
}) {
  const shown = kind === 'stack' ? [...items].reverse() : items
  return (
    <div className={cn('min-w-0', className)}>
      <div className="mb-1.5 text-xs text-muted">{title}</div>
      <div className={cn('flex gap-1.5', kind === 'stack' ? 'flex-col items-stretch' : 'flex-wrap items-center')}>
        {shown.length === 0 && <span className="text-xs text-subtle">{empty}</span>}
        {shown.map((it, i) => (
          <motion.div
            key={it.key}
            layout
            initial={{ opacity: 0, y: kind === 'stack' ? -6 : 0, x: kind === 'queue' ? 6 : 0 }}
            animate={{ opacity: 1, y: 0, x: 0 }}
            transition={{ duration: 0.18 }}
            className="relative whitespace-nowrap rounded-lg border px-2 py-1 text-center font-mono text-[12.5px] leading-tight"
            style={{
              background: it.tone ? toneSoft(it.tone, 24) : 'var(--surface)',
              borderColor: it.tone ? toneColor(it.tone) : 'var(--border-strong)',
              color: 'var(--fg)',
            }}
          >
            {it.text}
            {kind === 'stack' && i === 0 && <span className="ml-1.5 text-[10.5px] text-muted">← 栈顶</span>}
          </motion.div>
        ))}
      </div>
    </div>
  )
}

/** 一行小标签：输出序列（值 + 可选高亮最后一个） */
export function OutRow({ label, values, lastTone = 'success' }: { label: ReactNode; values: ReactNode[]; lastTone?: VizTone }) {
  return (
    <div className="flex min-w-0 flex-wrap items-center gap-1.5 text-[13px]">
      <span className="shrink-0 text-muted">{label}</span>
      {values.length === 0 && <span className="text-xs text-subtle">（空）</span>}
      {values.map((v, i) => (
        <span
          key={i}
          className="rounded-md border px-1.5 py-0.5 font-mono text-[12.5px]"
          style={
            i === values.length - 1
              ? { background: toneSoft(lastTone, 26), borderColor: toneColor(lastTone) }
              : { background: 'var(--surface)', borderColor: 'var(--border-strong)' }
          }
        >
          {v}
        </span>
      ))}
    </div>
  )
}
