/**
 * 树注意力掩码：把草稿树按 BFS 展平成一个序列，一次前向验证所有分支。
 * 节点 i 只能看到：已提交前缀（全部）+ 自己的祖先链 + 自己。位置编号 = 前缀长度 + 深度。
 * 点击树节点或掩码的行，查看该节点能看到谁。
 */
import { useMemo, useState } from 'react'
import { Legend, MatrixView, TreeView, VizFrame, type TreeNode } from '@/viz'
import { Segmented } from '@/components/ui'
import { C } from './common'

interface DraftTree {
  tokens: string[]
  parents: number[]
}

/** Medusa：3 个头各取 top-2，笛卡尔积 → 2 + 4 + 8 = 14 个节点 */
function medusaTree(): DraftTree {
  const heads = [
    ['天气', '今天'],
    ['很', '真'],
    ['好', '冷'],
  ]
  const tokens: string[] = []
  const parents: number[] = []
  let layer: number[] = [-1]
  heads.forEach((cands) => {
    const next: number[] = []
    layer.forEach((p) => {
      cands.forEach((t) => {
        tokens.push(t)
        parents.push(p)
        next.push(tokens.length - 1)
      })
    })
    layer = next
  })
  return { tokens, parents }
}

/** 动态树（EAGLE-2 风格）：不规则，置信度高的分支更深 */
const DYN: DraftTree = {
  tokens: ['天气', '今天', '很', '不', '真', '好', '错', '冷', '啊', '。'],
  parents: [-1, -1, 0, 0, 1, 2, 3, 2, 5, 5],
}

const PREFIX = 5

function depthOf(parents: number[]) {
  const d: number[] = []
  parents.forEach((p, i) => (d[i] = p < 0 ? 0 : d[p] + 1))
  return d
}
function ancestorsMask(parents: number[]) {
  const n = parents.length
  const m = Array.from({ length: n }, () => Array(n).fill(0))
  for (let i = 0; i < n; i++) for (let j = i; j !== -1; j = parents[j]) m[i][j] = 1
  return m
}

export function TreeMaskViz() {
  const [kind, setKind] = useState<'dyn' | 'medusa'>('dyn')
  const tree = useMemo(() => (kind === 'dyn' ? DYN : medusaTree()), [kind])
  const [sel, setSel] = useState(5)
  const n = tree.tokens.length
  const s = Math.min(sel, n - 1)
  const mask = useMemo(() => ancestorsMask(tree.parents), [tree])
  const depth = useMemo(() => depthOf(tree.parents), [tree])
  const path: number[] = []
  for (let j = s; j !== -1; j = tree.parents[j]) path.unshift(j)

  const root: TreeNode = useMemo(() => {
    const build = (i: number): TreeNode => ({
      id: i,
      label: tree.tokens[i],
      sub: String(i),
      tone: i === s ? 'gold' : path.includes(i) ? 'accent' : undefined,
      children: tree.parents.map((p, j) => (p === i ? j : -2)).filter((j) => j >= 0).map(build),
    })
    return {
      id: 'root',
      label: '根',
      sub: '已提交',
      tone: 'muted',
      children: tree.parents.map((p, j) => (p === -1 ? j : -2)).filter((j) => j >= 0).map(build),
    }
  }, [tree, s, path.join(',')])

  return (
    <VizFrame
      title="树注意力：一次前向验证所有分支"
      caption={
        <>
          节点 {s}「{tree.tokens[s]}」能看到：前缀的全部 {PREFIX} 个 token + 祖先链 <C>{path.map((j) => tree.tokens[j]).join(' → ')}</C>，
          看不到兄弟和其它分支。它的位置编号是 <C>{PREFIX} + 深度 {depth[s]} = {PREFIX + depth[s]}</C>——同一深度的节点位置编号相同，
          而不是按展平后的下标递增。共 {n} 个节点，一次前向就得到每个节点之后的目标分布。
        </>
      }
      legend={
        <Legend
          items={[
            { label: '选中节点 / 选中行', tone: 'gold', variant: 'soft' },
            { label: '祖先链 / 掩码中可见（=1）', tone: 'accent', variant: 'soft' },
            { label: '掩掉（斜线）', tone: 'muted', variant: 'soft' },
          ]}
        />
      }
      controls={
        <Segmented
          size="sm"
          value={kind}
          onChange={(v) => {
            setKind(v)
            setSel(v === 'dyn' ? 5 : 9)
          }}
          options={[
            { value: 'dyn', label: '动态树（10 节点）' },
            { value: 'medusa', label: 'Medusa 笛卡尔积（2×2×2）' },
          ]}
        />
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="min-w-0 overflow-x-auto">
          <TreeView root={root} nodeSize={34} levelGap={58} siblingGap={kind === 'medusa' ? 4 : 10} onNodeClick={(id) => typeof id === 'number' && setSel(id)} />
          <div className="mt-1 text-center text-[11px] text-subtle">节点下方数字 = 展平后的下标（BFS 顺序）；点击节点切换</div>
        </div>
        <div className="min-w-0">
          <div className="mb-1 text-xs font-medium text-muted">树内部的掩码（前缀列全为 1，未画出）</div>
          <MatrixView
            data={mask}
            scale="none"
            showValues={false}
            cellSize={26}
            rowLabels={tree.tokens.map((t, i) => `${i} ${t}`)}
            colLabels={tree.tokens.map((_, i) => String(i))}
            masked={(r, c) => !mask[r][c]}
            highlight={(r, c) => (mask[r][c] ? (r === s ? 'gold' : 'accent') : undefined)}
            onCellClick={(r) => setSel(r)}
          />
          <div className="mt-2 overflow-x-auto font-mono text-[11px] text-muted">
            position_ids = [{depth.map((d) => PREFIX + d).join(', ')}]
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
