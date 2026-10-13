/**
 * EAGLE-2 的动态草稿树：草稿模型的置信度近似接受率，节点的「全局分数」= 根到它路径上置信度的乘积。
 *  1) 扩展：每一层只挑全局分数最高的 top-k 个节点继续展开（每个节点取 top-k 个孩子）
 *  2) 重排：在全部生成的节点里选全局分数最高的 m 个作为送去验证的草稿
 * 若置信度校准良好，被选节点分数之和 ≈ 期望接受的草稿数（每个节点被接受 ⇔ 整条路径被接受）。
 * toy 置信度由固定种子生成。
 */
import { useMemo, useState } from 'react'
import { Legend, TreeView, VizFrame, type TreeNode } from '@/viz'
import { Slider } from '@/components/ui'
import { rng } from '@/lib/utils'
import { Stat } from './common'

const VOCAB = ['的', '是', '在', '我们', '可以', '一个', '模型', '推理', '加速', '。', '和', '更']

interface N {
  id: number
  parent: number
  depth: number
  tok: string
  conf: number
  score: number
}

/** 某个节点的候选孩子（按置信度降序），由节点 id 决定，结果确定 */
function childrenOf(id: number, depth: number) {
  const r = rng(1000 + id * 37 + depth * 7)
  const top = 0.35 + 0.6 * r()
  const second = (1 - top) * (0.35 + 0.5 * r())
  const third = (1 - top - second) * (0.3 + 0.5 * r())
  const confs = [top, second, third]
  return confs.map((c, k) => ({ conf: c, tok: VOCAB[Math.floor(r() * VOCAB.length + k) % VOCAB.length] }))
}

function grow(k: number, D: number) {
  const nodes: N[] = []
  let frontier: { id: number; score: number; depth: number }[] = [{ id: -1, score: 1, depth: 0 }]
  let nid = 0
  const expanded = new Set<number>()
  for (let d = 1; d <= D; d++) {
    const layer: N[] = []
    frontier.forEach((p) => {
      expanded.add(p.id)
      childrenOf(p.id + 1, d)
        .slice(0, k)
        .forEach((c) => {
          const nd: N = { id: nid++, parent: p.id, depth: d, tok: c.tok, conf: c.conf, score: p.score * c.conf }
          layer.push(nd)
        })
    })
    nodes.push(...layer)
    frontier = [...layer].sort((a, b) => b.score - a.score).slice(0, k)
  }
  return { nodes, expanded }
}

export function DynamicTreeViz() {
  const [k, setK] = useState(2)
  const [D, setD] = useState(4)
  const [m, setM] = useState(6)
  const { nodes, expanded } = useMemo(() => grow(k, D), [k, D])
  const mm = Math.min(m, nodes.length)
  const chosen = useMemo(() => {
    const order = [...nodes].sort((a, b) => b.score - a.score || a.depth - b.depth)
    return new Set(order.slice(0, mm).map((x) => x.id))
  }, [nodes, mm])
  const expTree = nodes.filter((x) => chosen.has(x.id)).reduce((s, x) => s + x.score, 0)
  // 对照：同样 m 个 token 排成一条链（每步只沿置信度最高的孩子走）
  const chain = useMemo(() => {
    let cur = -1
    let score = 1
    let sum = 0
    for (let d = 1; d <= mm; d++) {
      const top = childrenOf(cur + 1, d)[0]
      score *= top.conf
      sum += score
      const hit = nodes.find((x) => x.parent === cur && x.depth === d && x.conf === top.conf)
      cur = hit ? hit.id : 900 + d
    }
    return sum
  }, [mm, nodes])

  const root: TreeNode = useMemo(() => {
    const build = (x: N): TreeNode => ({
      id: x.id,
      label: x.tok,
      sub: x.score.toFixed(2),
      tone: chosen.has(x.id) ? 'success' : undefined,
      dim: !chosen.has(x.id),
      edgeTone: chosen.has(x.id) ? 'success' : undefined,
      children: nodes.filter((c) => c.parent === x.id).map(build),
    })
    return { id: 'r', label: '根', tone: 'muted', children: nodes.filter((c) => c.parent === -1).map(build) }
  }, [nodes, chosen])

  return (
    <VizFrame
      title="EAGLE-2：按置信度长出动态树"
      caption={
        <>
          每层只展开全局分数（路径置信度之积）最高的 {k} 个节点，共生成 {nodes.length} 个候选；重排后取分数最高的 {mm} 个送去验证（绿色）。
          若置信度≈接受率，期望每轮接受 <b>{expTree.toFixed(2)}</b> 个草稿（+1 个 bonus）；同样 {mm} 个 token 排成一条单链（每步取置信度最高者）则期望 {chain.toFixed(2)} 个。
          节点下方是全局分数；被选节点的置信度见下表。
        </>
      }
      legend={
        <Legend
          items={[
            { label: '被选中验证的草稿', tone: 'success', variant: 'soft' },
            { label: '生成了但未选中', tone: 'muted', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <Slider className="min-w-[120px] flex-1" label="每层展开 top-k" value={k} min={1} max={3} onChange={setK} />
          <Slider className="min-w-[120px] flex-1" label="草稿深度" value={D} min={2} max={5} onChange={setD} />
          <Slider className="min-w-[140px] flex-1" label="验证 token 数 m" value={m} min={2} max={16} onChange={setM} />
        </>
      }
    >
      <div className="space-y-3">
        <div className="min-w-0 overflow-x-auto pb-1">
          <TreeView root={root} nodeSize={32} levelGap={62} siblingGap={6} />
        </div>
        <div className="grid grid-cols-2 gap-2 @xl:grid-cols-4">
          <Stat label="生成的候选数" value={nodes.length} sub={`${D} 次草稿前向`} />
          <Stat label="送去验证" value={mm} sub="目标模型 1 次前向" />
          <Stat label="期望接受（树）" value={expTree.toFixed(2)} tone="success" sub="Σ 被选节点分数" />
          <Stat label="期望接受（单链）" value={chain.toFixed(2)} sub="同样 m 个 token" />
        </div>
        <div className="flex flex-wrap gap-1.5 font-mono text-[11px]">
          {nodes
            .filter((x) => chosen.has(x.id))
            .sort((a, b) => b.score - a.score)
            .map((x) => (
              <span key={x.id} className="rounded border border-border bg-surface px-1.5 py-0.5 text-muted">
                <b className="text-fg">{x.tok}</b> 深度{x.depth} · 置信 {x.conf.toFixed(2)} · 分数 {x.score.toFixed(2)}
              </span>
            ))}
        </div>
        <div className="text-[11.5px] text-subtle">
          已展开的节点数：{expanded.size - 1}。分数沿路径单调不增，所以「取分数最高的 m 个」天然构成一棵连通的子树。实际 EAGLE-2 由草稿模型给出置信度，这里用固定种子的 toy 数字演示。
        </div>
      </div>
    </VizFrame>
  )
}
