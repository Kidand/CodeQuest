/**
 * 回溯 = 在决策树上做 DFS：三种模板（选/不选、枚举 start、used[] 排列）逐帧展开递归树。
 * 洞见：path 就是“从根到当前节点的边”；做选择 = 走下一条边，撤销 = 回到父节点。三种模板只是树的形状不同。
 */
import { useMemo, useState } from 'react'
import { CodeTrace, Legend, StepControls, VizFrame, useStepper, type VizTone } from '@/viz'
import { Segmented } from '@/components/ui'
import { C, RecTree, Stat, Sub, type RNode } from './common'

type Mode = 'pick' | 'start' | 'perm'

const CODE: Record<Mode, string> = {
  pick: `def subsets(nums):
    res, path = [], []
    def dfs(i):
        if i == len(nums):
            res.append(path[:])      # 叶子：一组完整决策
            return
        path.append(nums[i])         # 选 nums[i]
        dfs(i + 1)
        path.pop()                   # 撤销选择
        dfs(i + 1)                   # 不选 nums[i]
    dfs(0)
    return res`,
  start: `def subsets(nums):
    res, path = [], []
    def dfs(start):
        res.append(path[:])          # 每个节点都是一个子集
        for i in range(start, len(nums)):
            path.append(nums[i])     # 做选择
            dfs(i + 1)               # 只往后选：不重不漏
            path.pop()               # 撤销选择
    dfs(0)
    return res`,
  perm: `def permute(nums):
    res, path = [], []
    used = [False] * len(nums)
    def dfs():
        if len(path) == len(nums):
            res.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]:
                continue             # 已在路径上，跳过
            used[i] = True; path.append(nums[i])
            dfs()
            used[i] = False; path.pop()   # 撤销
    dfs()
    return res`,
}

interface TN {
  id: number
  label: string
  parent: number | null
  edge: string
  children: number[]
}
interface Ev {
  kind: 'enter' | 'collect' | 'back'
  node: number
  line: number[]
  path: number[]
}

function simulate(mode: Mode, nums: number[]) {
  const nodes: TN[] = []
  const evs: Ev[] = []
  const path: number[] = []
  const lab = () => (path.length ? path.join(',') : '∅')
  const add = (parent: number | null, edge: string) => {
    const id = nodes.length
    nodes.push({ id, label: lab(), parent, edge, children: [] })
    if (parent != null) nodes[parent].children.push(id)
    return id
  }
  const n = nums.length
  if (mode === 'pick') {
    const dfs = (i: number, id: number, line: number[]) => {
      evs.push({ kind: 'enter', node: id, line, path: [...path] })
      if (i === n) {
        evs.push({ kind: 'collect', node: id, line: [4, 5], path: [...path] })
        return
      }
      path.push(nums[i])
      const a = add(id, `选${nums[i]}`)
      dfs(i + 1, a, [7, 8])
      path.pop()
      evs.push({ kind: 'back', node: id, line: [9], path: [...path] })
      const b = add(id, '不选')
      dfs(i + 1, b, [10])
    }
    dfs(0, add(null, ''), [11])
  } else if (mode === 'start') {
    const dfs = (start: number, id: number, line: number[]) => {
      evs.push({ kind: 'enter', node: id, line, path: [...path] })
      evs.push({ kind: 'collect', node: id, line: [4], path: [...path] })
      for (let i = start; i < n; i++) {
        path.push(nums[i])
        const c = add(id, `+${nums[i]}`)
        dfs(i + 1, c, [6, 7])
        path.pop()
        evs.push({ kind: 'back', node: id, line: [8], path: [...path] })
      }
    }
    dfs(0, add(null, ''), [9])
  } else {
    const used = nums.map(() => false)
    const dfs = (id: number, line: number[]) => {
      evs.push({ kind: 'enter', node: id, line, path: [...path] })
      if (path.length === n) {
        evs.push({ kind: 'collect', node: id, line: [5, 6], path: [...path] })
        return
      }
      for (let i = 0; i < n; i++) {
        if (used[i]) continue
        used[i] = true
        path.push(nums[i])
        const c = add(id, `+${nums[i]}`)
        dfs(c, [11, 12])
        used[i] = false
        path.pop()
        evs.push({ kind: 'back', node: id, line: [13], path: [...path] })
      }
    }
    dfs(add(null, ''), [14])
  }
  return { nodes, evs }
}

const NUMS = [1, 2, 3]

interface Frame {
  visible: Set<number>
  collected: number[]
  cur: number
  ev: Ev
  calls: number
}

export function BacktrackTreeViz() {
  const [mode, setMode] = useState<Mode>('start')
  const { nodes, frames } = useMemo(() => {
    const { nodes, evs } = simulate(mode, NUMS)
    const visible = new Set<number>()
    const collected: number[] = []
    let calls = 0
    const frames: Frame[] = evs.map((ev) => {
      if (ev.kind === 'enter') {
        visible.add(ev.node)
        calls++
      }
      if (ev.kind === 'collect') collected.push(ev.node)
      return { visible: new Set(visible), collected: [...collected], cur: ev.node, ev, calls }
    })
    return { nodes, frames }
  }, [mode])
  const st = useStepper(frames, { intervalMs: 700 })
  const f = st.frame

  const onPath = new Set<number>()
  for (let p: number | null = f.cur; p != null; p = nodes[p].parent) onPath.add(p)
  const collectedSet = new Set(f.collected)

  const toTree = (id: number): RNode => {
    const nd = nodes[id]
    let tone: VizTone | undefined
    if (id === f.cur) tone = 'accent'
    else if (onPath.has(id)) tone = 'info'
    else if (collectedSet.has(id)) tone = 'success'
    return {
      id,
      label: nd.label,
      tone,
      edgeLabel: nd.edge,
      edgeTone: onPath.has(id) && nd.parent != null ? 'info' : undefined,
      hidden: !f.visible.has(id),
      sub: collectedSet.has(id) ? '✓' : undefined,
      subTone: 'success',
      children: nd.children.map(toTree),
    }
  }

  const path = f.ev.path
  const caption =
    f.ev.kind === 'enter' ? (
      <>
        进入节点 <C>path = [{path.join(', ')}]</C>
        {f.ev.node === 0 ? '：从空路径出发。' : '：做了一次选择，向下递归一层。'}
      </>
    ) : f.ev.kind === 'collect' ? (
      <>
        收集答案 <C>res.append(path[:])</C> → [{path.join(', ')}]。必须拷贝：path 之后还会被修改。
      </>
    ) : (
      <>
        子树探索完毕，<C>path.pop()</C> 撤销选择，回到 <C>[{path.join(', ')}]</C>，尝试下一个分支。
      </>
    )

  const total = nodes.length
  return (
    <VizFrame
      title="回溯 = 决策树上的 DFS"
      caption={caption}
      legend={
        <Legend
          items={[
            { label: '当前节点', tone: 'accent', variant: 'soft' },
            { label: '当前路径 path', tone: 'info', variant: 'soft' },
            { label: '已收集的答案', tone: 'success', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Segmented
            size="sm"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'pick', label: '子集·选/不选' },
              { value: 'start', label: '子集·枚举 start' },
              { value: 'perm', label: '排列·used[]' },
            ]}
          />
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-[3fr_2fr]">
        <div className="flex min-w-0 flex-col gap-3">
          <RecTree root={toTree(0)} levelGap={mode === 'pick' ? 50 : 56} />
          <div>
            <Sub>res（{f.collected.length} 个）</Sub>
            <div className="flex min-h-[26px] flex-wrap gap-1">
              {f.collected.map((id) => (
                <span key={id} className="rounded-md border border-success/60 bg-success-soft px-1.5 py-0.5 font-mono text-[11.5px] text-fg">
                  [{nodes[id].label === '∅' ? '' : nodes[id].label}]
                </span>
              ))}
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <Stat label="递归调用" value={`${f.calls} / ${total}`} />
            <Stat label="树的形状" value={mode === 'pick' ? '二叉，答案在叶子' : mode === 'start' ? '多叉，每个节点都是答案' : '多叉，答案在叶子'} />
          </div>
        </div>
        <CodeTrace code={CODE[mode]} active={f.ev.line} vars={{ path: `[${path.join(', ')}]` }} />
      </div>
    </VizFrame>
  )
}
