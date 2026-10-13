import { motion } from 'motion/react'
import { cn } from '@/lib/utils'
import { useElementWidth } from './useWidth'
import { toneColor, toneSoft, type VizTone } from './tones'

export interface GraphNode {
  id: string | number
  label: React.ReactNode
  /** 坐标（viewBox 单位） */
  x: number
  y: number
  tone?: VizTone
  sub?: string
  dim?: boolean
  r?: number
}
export interface GraphEdge {
  from: string | number
  to: string | number
  label?: string
  tone?: VizTone
  directed?: boolean
  dashed?: boolean
  dim?: boolean
  /** 弧度弯曲（双向边时避免重叠），默认 0 */
  curve?: number
}

/**
 * 通用图（BFS/DFS、拓扑、Dijkstra、并查集、MoE 路由、设备拓扑、Ring All-Reduce…）。
 * 节点坐标由调用方给出（viewBox 坐标系），组件按 width 自适应缩放。
 */
export function GraphView({
  nodes,
  edges,
  width,
  height,
  nodeR = 18,
  className,
  onNodeClick,
}: {
  nodes: GraphNode[]
  edges: GraphEdge[]
  width: number
  height: number
  nodeR?: number
  className?: string
  onNodeClick?: (id: GraphNode['id']) => void
}) {
  const byId = new Map(nodes.map((n) => [n.id, n]))
  // viewBox 缩放会让文字在窄屏上变小：按实测宽度反向放大字号（最多 1.8 倍）
  const [wrapRef, avail] = useElementWidth()
  const k = avail > 0 ? Math.min(1, avail / width) : 1
  const fs = (base: number) => base / Math.max(k, 1 / 1.8)
  return (
    <div ref={wrapRef} className="w-full min-w-0">
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" style={{ maxWidth: width }} className={cn('mx-auto block overflow-visible', className)}>
      <defs>
        {(['accent', 'muted', 'success', 'danger', 'warning', 'info', 1, 2, 3, 4, 5, 6, 7, 8] as VizTone[]).map((t) => (
          <marker key={String(t)} id={`cq-arrow-${t}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill={toneColor(t)} />
          </marker>
        ))}
      </defs>
      {edges.map((e, k) => {
        const a = byId.get(e.from)
        const b = byId.get(e.to)
        if (!a || !b) return null
        const ra = a.r ?? nodeR
        const rb = b.r ?? nodeR
        const dx = b.x - a.x
        const dy = b.y - a.y
        const len = Math.hypot(dx, dy) || 1
        const ux = dx / len
        const uy = dy / len
        const x1 = a.x + ux * ra
        const y1 = a.y + uy * ra
        const x2 = b.x - ux * (rb + (e.directed ? 3 : 0))
        const y2 = b.y - uy * (rb + (e.directed ? 3 : 0))
        const c = e.curve ?? 0
        const mx = (x1 + x2) / 2 - uy * c * len
        const my = (y1 + y2) / 2 + ux * c * len
        const tone = e.tone ?? 'muted'
        const d = c ? `M${x1},${y1} Q${mx},${my} ${x2},${y2}` : `M${x1},${y1} L${x2},${y2}`
        const lx = c ? (x1 + 2 * mx + x2) / 4 : (x1 + x2) / 2
        const ly = c ? (y1 + 2 * my + y2) / 4 : (y1 + y2) / 2
        return (
          <g key={k} opacity={e.dim ? 0.25 : 1}>
            <motion.path
              initial={false}
              animate={{ d }}
              fill="none"
              stroke={e.tone ? toneColor(tone) : 'var(--border-strong)'}
              strokeWidth={e.tone ? 2.2 : 1.5}
              strokeDasharray={e.dashed ? '5 4' : undefined}
              markerEnd={e.directed ? `url(#cq-arrow-${e.tone ? tone : 'muted'})` : undefined}
            />
            {e.label && (
              <text x={lx} y={ly - 4} textAnchor="middle" fontSize={fs(11)} fill="var(--muted)" className="font-mono" paintOrder="stroke" stroke="var(--surface)" strokeWidth={3}>
                {e.label}
              </text>
            )}
          </g>
        )
      })}
      {nodes.map((n) => {
        const r = n.r ?? nodeR
        return (
          <motion.g
            key={n.id}
            initial={false}
            animate={{ x: n.x, y: n.y, opacity: n.dim ? 0.35 : 1 }}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            onClick={onNodeClick ? () => onNodeClick(n.id) : undefined}
            style={{ cursor: onNodeClick ? 'pointer' : undefined }}
          >
            <circle r={r} fill={n.tone ? toneSoft(n.tone, 30) : 'var(--surface)'} stroke={n.tone ? toneColor(n.tone) : 'var(--border-strong)'} strokeWidth={n.tone ? 2.2 : 1.5} />
            <text textAnchor="middle" dominantBaseline="central" fontSize={fs(r > 16 ? 12.5 : 10.5)} fontWeight={600} fill="var(--fg)" className="font-mono">
              {n.label}
            </text>
            {n.sub && (
              <text y={r + 12 * (fs(1))} textAnchor="middle" fontSize={fs(10.5)} fill="var(--muted)" className="font-mono">
                {n.sub}
              </text>
            )}
          </motion.g>
        )
      })}
    </svg>
    </div>
  )
}
