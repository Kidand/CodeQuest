/**
 * 岛屿数量（LC 200）：扫描到未访问的陆地 → 计数 +1 → BFS/DFS 把整座岛“淹掉”。
 * 开关“出队时才标记”展示经典 bug：同一格被重复入队。
 */
import { useMemo, useState } from 'react'
import { CodeTrace, Legend, StepControls, VizFrame, useStepper, type VizTone } from '@/viz'
import { Segmented, Switch } from '@/components/ui'
import { C, StackList } from './common'
import { GridBoard } from './GridBoard'

const GRID = ['11000110', '11010010', '00011000', '10110011', '10000110', '11010000'].map((r) => r.split('').map(Number))
const M = GRID.length
const N = GRID[0].length
const DIRS = [[1, 0], [-1, 0], [0, 1], [0, -1]] as const
const ISLAND_TONES: VizTone[] = [1, 2, 3, 5, 6, 7, 8, 4]

type Mode = 'bfs' | 'dfs'

const code = (mode: Mode, late: boolean) => `def num_islands(grid):
    m, n = len(grid), len(grid[0])
    count = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] != "1":
                continue
            count += 1                    # 发现一座新岛
            ${late ? '# （出队时才标记）' : 'grid[i][j] = "0"              # 入队即标记'}
            q = deque([(i, j)])
            while q:
                r, c = ${mode === 'bfs' ? 'q.popleft()      # BFS：队头' : 'q.pop()          # DFS：栈顶'}
                ${late ? 'grid[r][c] = "0"              # ✗ 出队才标记' : '# 已在入队时标记过'}
                for nr, nc in neighbors(r, c):
                    if inside(nr, nc) and grid[nr][nc] == "1":
                        ${late ? '# ✗ 没标记，可能被重复入队' : 'grid[nr][nc] = "0"'}
                        q.append((nr, nc))
    return count`

interface Frame {
  owner: number[][] // -1 未访问；k = 第 k 座岛（已出队处理）
  queued: Set<string>
  q: [number, number][]
  cur?: [number, number]
  count: number
  pushes: number
  dup: number
  line: number | number[]
  caption: React.ReactNode
}

function buildFrames(mode: Mode, late: boolean): Frame[] {
  const g = GRID.map((r) => [...r])
  const owner = GRID.map((r) => r.map(() => -1))
  const frames: Frame[] = []
  let count = 0
  let pushes = 0
  let dup = 0
  const snap = (q: [number, number][], extra: Partial<Frame> & { line: Frame['line']; caption: React.ReactNode }) =>
    frames.push({ owner: owner.map((r) => [...r]), queued: new Set(q.map(([r, c]) => `${r},${c}`)), q: [...q], count, pushes, dup, ...extra })
  snap([], { line: [4, 5], caption: <>从左上角开始逐格扫描，遇到<b>还没被淹掉的陆地</b>就说明发现了一座新岛。</> })
  for (let i = 0; i < M; i++)
    for (let j = 0; j < N; j++) {
      if (g[i][j] !== 1) continue
      count++
      const isl = count - 1
      if (!late) g[i][j] = 0
      const q: [number, number][] = [[i, j]]
      pushes++
      snap(q, { cur: [i, j], line: [8, 9, 10], caption: <>扫描到 <C>({i}, {j})</C> 是陆地：<b>岛屿数 = {count}</b>，从这里开始 {mode === 'bfs' ? 'BFS' : 'DFS'} 把整座岛淹掉。</> })
      while (q.length) {
        const [r, c] = mode === 'bfs' ? q.shift()! : q.pop()!
        if (late) {
          if (owner[r][c] >= 0) {
            dup++
            snap(q, { cur: [r, c], line: [12, 13], caption: <>弹出 <C>({r}, {c})</C>：它<b>已经处理过</b>，却又一次出现在{mode === 'bfs' ? '队列' : '栈'}里——这就是“出队才标记”造成的重复入队（累计 {dup} 次）。</> })
            continue
          }
          g[r][c] = 0
        }
        owner[r][c] = isl
        const added: string[] = []
        for (const [dr, dc] of DIRS) {
          const nr = r + dr
          const nc = c + dc
          if (nr >= 0 && nr < M && nc >= 0 && nc < N && g[nr][nc] === 1) {
            if (!late) g[nr][nc] = 0
            q.push([nr, nc])
            pushes++
            added.push(`(${nr}, ${nc})`)
          }
        }
        snap(q, {
          cur: [r, c],
          line: [12, 14, 15, 16, 17],
          caption: (
            <>
              弹出 <C>({r}, {c})</C>，检查上下左右：{added.length ? <>把 {added.join('、')} {late ? '入队（尚未标记）' : '标记并入队'}。</> : '没有新的陆地邻居。'}
            </>
          ),
        })
      }
      snap([], { line: 11, caption: <>{mode === 'bfs' ? '队列' : '栈'}空了：第 {count} 座岛已全部淹掉，继续扫描。</> })
    }
  snap([], {
    line: 18,
    caption: (
      <>
        扫描结束，共 <b>{count}</b> 座岛。入{mode === 'bfs' ? '队' : '栈'} {pushes} 次{late ? <>，其中重复 {dup} 次——陆地格只有 {GRID.flat().filter(Boolean).length} 个</> : <>，恰好等于陆地格数：每格只进一次</>}。时间 O(mn)。
      </>
    ),
  })
  return frames
}

export function FloodFillViz() {
  const [mode, setMode] = useState<Mode>('bfs')
  const [late, setLate] = useState(false)
  const frames = useMemo(() => buildFrames(mode, late), [mode, late])
  const st = useStepper(frames, { intervalMs: 700 })
  const f = st.frame
  return (
    <VizFrame
      title="岛屿数量：发现一块陆地就淹掉整座岛"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '已处理（颜色 = 所属岛屿）', tone: 1, variant: 'soft' },
            { label: '在队列 / 栈中（虚线）', tone: 'warning', variant: 'line' },
            { label: '当前弹出', tone: 'accent', variant: 'line' },
            { label: '水', tone: 'muted', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Segmented size="sm" value={mode} onChange={setMode} options={[{ value: 'bfs', label: 'BFS（队列）' }, { value: 'dfs', label: 'DFS（显式栈）' }]} />
          <Switch checked={late} onChange={setLate} label="出队时才标记（错误）" />
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="min-w-0 space-y-3">
          <GridBoard
            rows={M}
            cols={N}
            max={40}
            cell={(r, c) => {
              const own = f.owner[r][c]
              const isCur = f.cur && f.cur[0] === r && f.cur[1] === c
              const inQ = f.queued.has(`${r},${c}`)
              if (!GRID[r][c]) return { fill: 'water', text: '0' }
              return {
                fill: own >= 0 ? ISLAND_TONES[own % 8] : undefined,
                ring: isCur ? 'accent' : inQ ? 'warning' : undefined,
                dashed: !isCur && inQ,
                text: '1',
              }
            }}
          />
          <div className="flex flex-wrap gap-x-5 gap-y-1 font-mono text-[12.5px]">
            <span>
              count = <b>{f.count}</b>
            </span>
            <span className="text-muted">入{mode === 'bfs' ? '队' : '栈'}次数 {f.pushes}</span>
            {late && <span className="text-danger">重复 {f.dup}</span>}
          </div>
          <StackList kind="queue" title={mode === 'bfs' ? '队列（左为队头）' : '栈（右为栈顶）'} items={f.q.map(([r, c], k) => ({ key: `${r},${c},${k}`, text: `${r},${c}`, tone: 'warning' }))} />
        </div>
        <CodeTrace code={code(mode, late)} active={f.line} maxHeight={330} />
      </div>
    </VizFrame>
  )
}
