/**
 * 层序遍历：进入每一层前先记下 len(q)，本轮只弹这么多个——这就是“按层切分”。
 * 打开锯齿开关：奇数层整体反转（LC 103）。
 */
import { useMemo, useState } from 'react'
import { CodeTrace, Legend, StepControls, TreeView, VizFrame, toneColor, useStepper } from '@/viz'
import { Switch } from '@/components/ui'
import { C, StackList, fromLevel, toView, type BT } from './common'

const SAMPLE = [1, 2, 3, 4, 5, null, 6, null, null, 7]

const CODE = `def level_order(root, zigzag=False):
    if root is None:
        return []
    res, q = [], deque([root])
    while q:
        level = []
        for _ in range(len(q)):      # 只处理“当前层”
            node = q.popleft()
            level.append(node.val)
            if node.left:  q.append(node.left)
            if node.right: q.append(node.right)
        if zigzag and len(res) % 2 == 1:
            level.reverse()          # 奇数层反过来
        res.append(level)
    return res`

interface Frame {
  q: BT[]
  cur?: BT
  level: number[]
  res: number[][]
  levelIds: number[]
  line: number | number[]
  caption: React.ReactNode
  quota?: number
}

function buildFrames(root: BT, zigzag: boolean): Frame[] {
  const frames: Frame[] = []
  const q: BT[] = [root]
  const res: number[][] = []
  frames.push({ q: [...q], level: [], res: [], levelIds: [], line: 4, caption: <>队列初始化为 <C>[{root.val}]</C>。</> })
  while (q.length) {
    const size = q.length
    const level: number[] = []
    const ids: number[] = []
    frames.push({
      q: [...q], level: [], res: res.map((r) => [...r]), levelIds: [], line: [6, 7], quota: size,
      caption: <>第 {res.length} 层开始：此刻队列里<b>恰好是这一层的全部节点</b>，记下 <C>len(q) = {size}</C>，本轮只弹 {size} 个。</>,
    })
    for (let k = 0; k < size; k++) {
      const n = q.shift()!
      level.push(n.val)
      ids.push(n.id)
      const kids = [n.left, n.right].filter(Boolean) as BT[]
      q.push(...kids)
      frames.push({
        q: [...q], cur: n, level: [...level], res: res.map((r) => [...r]), levelIds: [...ids], line: [8, 9, 10, 11], quota: size - k - 1,
        caption: (
          <>
            弹出 <C>{n.val}</C> 放进 level；{kids.length ? <>孩子 {kids.map((c) => c.val).join('、')} 入队（排在<b>下一层</b>）。</> : '没有孩子。'}
            {size - k - 1 > 0 ? ` 本层还剩 ${size - k - 1} 个。` : ' 本层弹完。'}
          </>
        ),
      })
    }
    const rev = zigzag && res.length % 2 === 1
    const lv = rev ? [...level].reverse() : level
    res.push(lv)
    frames.push({
      q: [...q], level: lv, res: res.map((r) => [...r]), levelIds: ids, line: rev ? [12, 13, 14] : [12, 14],
      caption: rev ? <>奇数层且开启锯齿：把 <C>[{level.join(', ')}]</C> 反转成 <C>[{lv.join(', ')}]</C> 再放进结果。</> : <>本层结束，<C>[{lv.join(', ')}]</C> 放进结果。</>,
    })
  }
  frames.push({ q: [], level: [], res, levelIds: [], line: 15, caption: <>队列为空，返回 <C>{JSON.stringify(res)}</C>。每个节点入队、出队各一次：O(n) 时间，队列最长为最宽一层的节点数。</> })
  return frames
}

export function LevelOrderViz() {
  const root = useMemo(() => fromLevel(SAMPLE)!, [])
  const [zigzag, setZigzag] = useState(false)
  const frames = useMemo(() => buildFrames(root, zigzag), [root, zigzag])
  const st = useStepper(frames, { intervalMs: 1000 })
  const f = st.frame
  const inQ = new Set(f.q.map((x) => x.id))
  const inLevel = new Set(f.levelIds)
  const doneVals = new Set(f.res.flat())
  const view = toView(root, {
    tone: (n) => (n.id === f.cur?.id ? 'accent' : inLevel.has(n.id) ? 'warning' : inQ.has(n.id) ? 'info' : doneVals.has(n.val) ? 'success' : undefined),
  })
  return (
    <VizFrame
      title="层序遍历：用 len(q) 切出每一层"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '刚出队', tone: 'accent', variant: 'soft' },
            { label: '本层已出队', tone: 'warning', variant: 'soft' },
            { label: '在队列中', tone: 'info', variant: 'soft' },
            { label: '已放进结果', tone: 'success', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Switch checked={zigzag} onChange={setZigzag} label="锯齿形（LC 103）" />
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="min-w-0 space-y-3">
          <div className="thin-scroll overflow-x-auto py-1">
            <TreeView root={view} nodeSize={32} levelGap={54} siblingGap={12} />
          </div>
          <StackList kind="queue" title={<>队列 q（左为队头）{f.quota != null && <span className="ml-2 text-subtle">本层还需弹 {f.quota} 个</span>}</>} items={f.q.map((x) => ({ key: x.id, text: x.val, tone: 'info' }))} />
          <div className="space-y-1 font-mono text-[12.5px]">
            <div className="text-muted">res =</div>
            {f.res.map((r, i) => (
              <div key={i} className="text-fg">
                <span className="text-subtle">第 {i} 层 </span>[{r.join(', ')}]
              </div>
            ))}
            {f.quota != null && <div style={{ color: toneColor('warning') }}>level = [{f.level.join(', ')}]</div>}
          </div>
        </div>
        <CodeTrace code={CODE} active={f.line} vars={{ 'len(q)': f.q.length, 'len(res)': f.res.length }} />
      </div>
    </VizFrame>
  )
}
