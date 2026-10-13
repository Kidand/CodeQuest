/**
 * 堆的「数组 ↔ 完全二叉树」双视图：同一份数据，上面是数组（ids 让交换有动画），下面是树。
 * 下标关系（0 起）：parent = (i-1)//2，children = 2i+1、2i+2。
 */
import { ArrayView, TreeView, type ArrayPointer, type TreeNode, type VizTone } from '@/viz'

export function toTree(vals: number[], ids: number[], tones: Record<number, VizTone>, dimFrom = Infinity, i = 0): TreeNode | null {
  if (i >= vals.length) return null
  const l = toTree(vals, ids, tones, dimFrom, 2 * i + 1)
  const r = toTree(vals, ids, tones, dimFrom, 2 * i + 2)
  const children = l || r ? [l, r] : undefined
  return { id: ids[i], label: vals[i], tone: tones[i], sub: `[${i}]`, dim: i >= dimFrom, children }
}

export function HeapDual({
  vals,
  ids,
  tones,
  pointers = [],
  dimFrom,
  nodeSize = 32,
}: {
  vals: number[]
  ids: number[]
  tones: Record<number, VizTone>
  pointers?: ArrayPointer[]
  /** 下标 >= dimFrom 的元素变暗（如堆排序中已出堆的部分） */
  dimFrom?: number
  nodeSize?: number
}) {
  const root = toTree(vals, ids, tones, dimFrom)
  return (
    <div className="min-w-0 space-y-3">
      <ArrayView values={vals} ids={ids} cellSize={38} highlights={tones} pointers={pointers} dim={dimFrom != null ? (k) => k >= dimFrom : undefined} />
      <div className="thin-scroll overflow-x-auto">
        {root ? <TreeView root={root} nodeSize={nodeSize} levelGap={52} siblingGap={8} /> : <div className="py-6 text-center text-sm text-muted">空堆</div>}
      </div>
    </div>
  )
}
