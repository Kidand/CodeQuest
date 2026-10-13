/**
 * LeetCode 的层序列表表示：None 只给“存在的节点”的空孩子占位，末尾 None 截掉。
 * 点击节点剪掉 / 恢复它的子树，观察列表如何变化；打开开关对比“堆式数组”（2i+1 / 2i+2）。
 */
import { useMemo, useState } from 'react'
import { Legend, TreeView, VizFrame, toneColor, toneSoft } from '@/viz'
import { Button, Segmented, Switch } from '@/components/ui'
import { C, fromLevel, toLevel, toView, type BT } from './common'

const PRESETS = {
  sample: [1, 2, 3, 4, 5, null, 6, null, null, 7],
  lc102: [3, 9, 20, null, null, 15, 7],
  chain: [1, null, 2, null, 3, null, 4],
} as const
type Preset = keyof typeof PRESETS

function prune(n: BT | null, cut: Set<number>): BT | null {
  if (!n || cut.has(n.id)) return null
  return { ...n, left: prune(n.left, cut), right: prune(n.right, cut) }
}

/** 堆式数组：根在 0，孩子在 2i+1、2i+2 */
function heapArray(root: BT | null): (number | null)[] {
  const out: (number | null)[] = []
  const walk = (n: BT | null, i: number) => {
    if (!n) return
    while (out.length <= i) out.push(null)
    out[i] = n.val
    walk(n.left, 2 * i + 1)
    walk(n.right, 2 * i + 2)
  }
  walk(root, 0)
  return out
}

function Cells({ vals, trimmed = 0 }: { vals: (number | null)[]; trimmed?: number }) {
  const all = [...vals, ...Array(trimmed).fill(null)]
  return (
    <div className="flex flex-wrap gap-1">
      {all.map((v, i) => {
        const isTrim = i >= vals.length
        return (
          <div key={i} className="flex flex-col items-center" style={{ opacity: isTrim ? 0.35 : 1 }}>
            <span
              className="min-w-[30px] rounded-md border px-1 py-0.5 text-center font-mono text-[12px]"
              style={
                v == null
                  ? { borderStyle: 'dashed', borderColor: 'var(--border-strong)', color: 'var(--muted)', textDecoration: isTrim ? 'line-through' : undefined }
                  : { background: toneSoft('info', 20), borderColor: toneColor('info') }
              }
            >
              {v == null ? 'None' : v}
            </span>
            <span className="font-mono text-[10px] text-subtle">{i}</span>
          </div>
        )
      })}
    </div>
  )
}

export function LevelListViz() {
  const [preset, setPreset] = useState<Preset>('sample')
  const [cut, setCut] = useState<Set<number>>(new Set())
  const [heap, setHeap] = useState(false)
  const full = useMemo(() => fromLevel([...PRESETS[preset]])!, [preset])
  const tree = useMemo(() => prune(full, cut), [full, cut])
  const list = useMemo(() => toLevel(tree), [tree])
  // 统计被截掉的末尾 None 个数（不截时列表多长）
  const untrimmed = useMemo(() => {
    const out: (number | null)[] = []
    const q: (BT | null)[] = [tree]
    while (q.length) {
      const x = q.shift()!
      out.push(x ? x.val : null)
      if (x) q.push(x.left, x.right)
    }
    return out.length - list.length
  }, [tree, list])
  const harr = useMemo(() => heapArray(tree), [tree])
  const toggle = (id: number | string) => {
    if (id === 0) return
    setCut((s) => {
      const t = new Set(s)
      if (t.has(Number(id))) t.delete(Number(id))
      else t.add(Number(id))
      return t
    })
  }
  const view = tree ? toView(tree, { tone: () => 'info' }) : null
  return (
    <VizFrame
      title="层序列表 ↔ 树：None 怎么占位"
      caption={
        <>
          点击节点<b>剪掉 / 恢复</b>它的子树。规则：按层从左到右，<b>只为存在的节点</b>写出左右孩子（空孩子写 <C>None</C>）；
          不存在的节点不再展开，列表末尾的 <C>None</C> 全部截掉。当前列表长 {list.length}
          {heap && <>，堆式数组长 {harr.length}</>}。
        </>
      }
      legend={<Legend items={[{ label: '节点值', tone: 'info', variant: 'soft' }, { label: '空孩子占位 None（虚线）', tone: 'muted', variant: 'line' }, { label: '被截掉的末尾 None（删除线）', tone: 'muted', variant: 'soft' }]} />}
      controls={
        <>
          <Segmented
            size="sm"
            value={preset}
            onChange={(p) => {
              setPreset(p)
              setCut(new Set())
            }}
            options={[
              { value: 'sample', label: '示例树' },
              { value: 'lc102', label: 'LC 102 示例' },
              { value: 'chain', label: '右斜链' },
            ]}
          />
          <Switch checked={heap} onChange={setHeap} label="对比堆式数组（2i+1, 2i+2）" />
          <Button size="sm" variant="ghost" onClick={() => setCut(new Set())}>
            恢复整棵树
          </Button>
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="thin-scroll min-w-0 overflow-x-auto py-1">
          {view ? <TreeView root={view} nodeSize={32} levelGap={52} siblingGap={12} onNodeClick={toggle} /> : <div className="py-6 text-center text-sm text-muted">空树</div>}
        </div>
        <div className="min-w-0 space-y-3">
          <div>
            <div className="mb-1 text-xs text-muted">LeetCode 层序列表（list_to_tree 的输入）</div>
            <Cells vals={list} trimmed={untrimmed} />
          </div>
          {heap && (
            <div>
              <div className="mb-1 text-xs text-muted">堆式数组：下标 i 的孩子在 2i+1、2i+2（{harr.filter((v) => v == null).length} 个空槽）</div>
              <Cells vals={harr} />
            </div>
          )}
        </div>
      </div>
    </VizFrame>
  )
}
