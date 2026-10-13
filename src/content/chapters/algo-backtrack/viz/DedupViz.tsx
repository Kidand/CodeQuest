/**
 * 含重复元素的回溯：排序 + 同层去重（LC90 子集 II / LC47 全排列 II）。
 * 洞见：重复答案来自“同一层”选了值相同的不同元素；同一条路径上（不同层）选相同值是合法的。
 */
import { useMemo, useState } from 'react'
import { Legend, VizFrame, type VizTone } from '@/viz'
import { Segmented, Switch } from '@/components/ui'
import { C, RecTree, Stat, Sub, type RNode } from './common'

type Mode = 'sub' | 'perm'

interface Built {
  root: RNode
  results: { label: string; dup: boolean }[]
  calls: number
  pruned: number
}

const LABELS: Record<Mode, string[]> = { sub: ['1', '2', '2′'], perm: ['1', '1′', '2'] }
const VALUES: Record<Mode, number[]> = { sub: [1, 2, 2], perm: [1, 1, 2] }

function build(mode: Mode, dedup: boolean): Built {
  const nums = VALUES[mode]
  const lab = LABELS[mode]
  const n = nums.length
  let id = 0
  let calls = 0
  let pruned = 0
  const seen = new Set<string>()
  const results: Built['results'] = []
  const pathL: string[] = []
  const pathV: number[] = []
  const collect = (node: RNode) => {
    const key = pathV.join(',')
    const dup = seen.has(key)
    seen.add(key)
    results.push({ label: `[${pathL.join(',')}]`, dup })
    node.tone = (dup ? 'warning' : 'success') as VizTone
    node.sub = dup ? '重复' : '✓'
    node.subTone = dup ? 'warning' : 'success'
  }
  const mk = (edge: string): RNode => ({ id: id++, label: pathL.length ? pathL.join(',') : '∅', edgeLabel: edge, children: [] })
  const prunedNode = (edge: string, label: string): RNode => {
    pruned++
    return { id: id++, label, edgeLabel: edge, dashed: true, dim: true, tone: 'danger', sub: '✂', subTone: 'danger', edgeTone: 'danger' }
  }
  if (mode === 'sub') {
    const dfs = (start: number, node: RNode) => {
      calls++
      collect(node)
      for (let i = start; i < n; i++) {
        if (dedup && i > start && nums[i] === nums[i - 1]) {
          node.children!.push(prunedNode(`+${lab[i]}`, [...pathL, lab[i]].join(',')))
          continue
        }
        pathL.push(lab[i])
        pathV.push(nums[i])
        const c = mk(`+${lab[i]}`)
        node.children!.push(c)
        dfs(i + 1, c)
        pathL.pop()
        pathV.pop()
      }
    }
    const root = mk('')
    dfs(0, root)
    return { root, results, calls, pruned }
  }
  const used = nums.map(() => false)
  const dfs = (node: RNode) => {
    calls++
    if (pathL.length === n) {
      collect(node)
      return
    }
    for (let i = 0; i < n; i++) {
      if (used[i]) continue
      if (dedup && i > 0 && nums[i] === nums[i - 1] && !used[i - 1]) {
        node.children!.push(prunedNode(`+${lab[i]}`, [...pathL, lab[i]].join(',')))
        continue
      }
      used[i] = true
      pathL.push(lab[i])
      pathV.push(nums[i])
      const c = mk(`+${lab[i]}`)
      node.children!.push(c)
      dfs(c)
      used[i] = false
      pathL.pop()
      pathV.pop()
    }
  }
  const root = mk('')
  dfs(root)
  return { root, results, calls, pruned }
}

export function DedupViz() {
  const [mode, setMode] = useState<Mode>('sub')
  const [dedup, setDedup] = useState(false)
  const b = useMemo(() => build(mode, dedup), [mode, dedup])
  const dupCount = b.results.filter((r) => r.dup).length

  const caption =
    mode === 'sub' ? (
      dedup ? (
        <>
          规则 <C>if i &gt; start and nums[i] == nums[i-1]: continue</C>：同一层（同一个 for 循环）里，值相同的元素只让<b>第一个</b>展开。
          注意 <C>[1,2] → [1,2,2′]</C> 没被剪：2′ 和 2 在<b>不同层</b>（同一条路径上），这是合法的“选两个 2”。
        </>
      ) : (
        <>没有去重时，根节点下的 <C>+2</C> 与 <C>+2′</C> 两棵子树完全相同，于是 [2]、[1,2] 各出现两次。打开开关看看剪掉了哪里。</>
      )
    ) : dedup ? (
      <>
        规则 <C>if i &gt; 0 and nums[i] == nums[i-1] and not used[i-1]: continue</C>：前一个相同值<b>没被用</b>，说明它刚在本层被撤销，现在选 1′ 就会重复。
        而在 [1] 的子树里 1 已在路径上（used），选 1′ 是合法的。
      </>
    ) : (
      <>没有去重时 1 和 1′ 互换得到的排列被当成不同答案，6 个排列里只有 3 个不同。</>
    )

  return (
    <VizFrame
      title="排序 + 同层去重：剪掉重复子树"
      caption={caption}
      legend={
        <Legend
          items={[
            { label: '新答案', tone: 'success', variant: 'soft' },
            { label: '重复答案', tone: 'warning', variant: 'soft' },
            { label: '被剪掉的分支', tone: 'danger', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <Segmented
            size="sm"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'sub', label: '子集 II [1,2,2′]' },
              { value: 'perm', label: '全排列 II [1,1′,2]' },
            ]}
          />
          <Switch checked={dedup} onChange={setDedup} label="同层去重" />
        </>
      }
    >
      <div className="flex min-w-0 flex-col gap-3">
        <RecTree root={b.root} levelGap={54} />
        <div>
          <Sub>结果（{b.results.length} 个）</Sub>
          <div className="flex flex-wrap gap-1">
            {b.results.map((r, i) => (
              <span
                key={i}
                className={`rounded-md border px-1.5 py-0.5 font-mono text-[11.5px] text-fg ${r.dup ? 'border-warning bg-warning-soft line-through' : 'border-success/60 bg-success-soft'}`}
              >
                {r.label}
              </span>
            ))}
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <Stat label="递归调用" value={b.calls} />
          <Stat label="重复答案" value={dupCount} tone={dupCount ? 'warning' : 'success'} />
          <Stat label="剪掉的分支" value={b.pruned} tone="danger" />
        </div>
      </div>
    </VizFrame>
  )
}
