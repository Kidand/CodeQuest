import { useMemo } from 'react'
import { motion } from 'motion/react'
import { cn } from '@/lib/utils'
import { useElementWidth } from './useWidth'
import { toneColor, toneSoft, type VizTone } from './tones'

export interface TreeNode {
  id: string | number
  label: React.ReactNode
  children?: (TreeNode | null)[]
  tone?: VizTone
  /** 节点下方小字 */
  sub?: string
  /** 指向该节点的边的标签 */
  edgeLabel?: string
  edgeTone?: VizTone
  dim?: boolean
}

/**
 * 树（二叉树、递归树、搜索树、beam search 树、Trie、radix tree…）。
 * 自动布局：叶子等距，父节点居中。children 中的 null 表示二叉树空位（保留位置）。
 */
export function TreeView({
  root,
  nodeSize = 36,
  levelGap = 64,
  siblingGap = 14,
  shape = 'circle',
  className,
  onNodeClick,
}: {
  root: TreeNode
  nodeSize?: number
  levelGap?: number
  siblingGap?: number
  shape?: 'circle' | 'rect'
  className?: string
  onNodeClick?: (id: TreeNode['id']) => void
}) {
  const layout = useMemo(() => {
    type P = { node: TreeNode; x: number; y: number; parent?: P }
    const out: P[] = []
    let cursor = 0
    const unit = nodeSize + siblingGap
    const walk = (n: TreeNode | null, depth: number, parent?: P): number => {
      if (!n) {
        const x = cursor
        cursor += unit * 0.6
        return x
      }
      const kids = n.children ?? []
      const realKids = kids.filter(Boolean)
      if (realKids.length === 0) {
        const p: P = { node: n, x: cursor, y: depth * levelGap, parent }
        out.push(p)
        cursor += unit
        return p.x
      }
      const p: P = { node: n, x: 0, y: depth * levelGap, parent }
      out.push(p)
      const xs = kids.map((k) => walk(k, depth + 1, p))
      p.x = (xs[0] + xs[xs.length - 1]) / 2
      return p.x
    }
    walk(root, 0)
    const minX = Math.min(...out.map((p) => p.x))
    out.forEach((p) => (p.x -= minX))
    const width = Math.max(...out.map((p) => p.x)) + nodeSize
    const height = Math.max(...out.map((p) => p.y)) + nodeSize + 18
    return { nodes: out, width, height }
  }, [root, nodeSize, levelGap, siblingGap])

  const r = nodeSize / 2
  // 树比容器宽时等比缩小（最多缩到 55%），文字反向补偿以保持可读
  const [wrapRef, avail] = useElementWidth()
  const W = layout.width + 8
  const H = layout.height + 8
  const k = avail > 0 ? Math.max(0.55, Math.min(1, avail / W)) : 1
  const fz = (base: number) => base / Math.max(k, 0.7)
  return (
    <div ref={wrapRef} className="w-full min-w-0 overflow-x-auto">
    <div className={cn('mx-auto w-max', className)}>
      <svg width={W * k} height={H * k} viewBox={`-4 -4 ${W} ${H}`} className="overflow-visible">
        {layout.nodes.map((p) =>
          p.parent ? (
            <g key={`e-${p.node.id}`} opacity={p.node.dim ? 0.35 : 1}>
              <motion.line
                initial={false}
                animate={{ x1: p.parent.x + r, y1: p.parent.y + r, x2: p.x + r, y2: p.y + r }}
                stroke={p.node.edgeTone ? toneColor(p.node.edgeTone) : 'var(--border-strong)'}
                strokeWidth={p.node.edgeTone ? 2.2 : 1.5}
              />
              {p.node.edgeLabel && (
                <text
                  x={(p.parent.x + p.x) / 2 + r}
                  y={(p.parent.y + p.y) / 2 + r - 4}
                  textAnchor="middle"
                  fontSize={fz(10.5)}
                  fill="var(--muted)"
                  className="font-mono"
                  paintOrder="stroke"
                  stroke="var(--surface)"
                  strokeWidth={3}
                >
                  {p.node.edgeLabel}
                </text>
              )}
            </g>
          ) : null,
        )}
        {layout.nodes.map((p) => {
          const tone = p.node.tone
          return (
            <motion.g
              key={`n-${p.node.id}`}
              initial={false}
              animate={{ x: p.x, y: p.y, opacity: p.node.dim ? 0.35 : 1 }}
              transition={{ type: 'spring', stiffness: 300, damping: 30 }}
              onClick={onNodeClick ? () => onNodeClick(p.node.id) : undefined}
              style={{ cursor: onNodeClick ? 'pointer' : undefined }}
            >
              {shape === 'circle' ? (
                <circle cx={r} cy={r} r={r - 1} fill={tone ? toneSoft(tone, 30) : 'var(--surface)'} stroke={tone ? toneColor(tone) : 'var(--border-strong)'} strokeWidth={tone ? 2.2 : 1.5} />
              ) : (
                <rect x={1} y={1} width={nodeSize - 2} height={nodeSize - 2} rx={8} fill={tone ? toneSoft(tone, 30) : 'var(--surface)'} stroke={tone ? toneColor(tone) : 'var(--border-strong)'} strokeWidth={tone ? 2.2 : 1.5} />
              )}
              <text x={r} y={r} textAnchor="middle" dominantBaseline="central" fontSize={fz(nodeSize > 34 ? 13 : 11)} fontWeight={600} fill="var(--fg)" className="font-mono">
                {p.node.label}
              </text>
              {p.node.sub && (
                <text x={r} y={nodeSize + 11} textAnchor="middle" fontSize={fz(10)} fill="var(--muted)" className="font-mono">
                  {p.node.sub}
                </text>
              )}
            </motion.g>
          )
        })}
      </svg>
    </div>
    </div>
  )
}
