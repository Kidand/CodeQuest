/**
 * 二维 DP：LC 64 最小路径和 / LC 62 不同路径，逐格填表 + 依赖箭头 + 遍历顺序对错对比。
 * 洞见：遍历顺序必须保证「算 dp[i][j] 时，它依赖的格子都已算好」；顺序反了就会读到未初始化的 0。
 */
import { useMemo, useState } from 'react'
import { CodeTrace, Legend, StepControls, VizFrame, useStepper } from '@/viz'
import { Segmented } from '@/components/ui'
import { DpGrid, type Cell, type DpArrow } from './DpGrid'
import { C, Sub } from './common'

const GRID = [
  [1, 3, 1, 2, 4],
  [1, 5, 1, 3, 1],
  [4, 2, 1, 2, 2],
  [2, 1, 3, 1, 1],
]
const M = GRID.length
const N = GRID[0].length
type Prob = '64' | '62'
type Order = 'row' | 'reverse'

const code = (p: Prob, o: Order) => {
  const loop = o === 'row' ? 'for i in range(m):' : 'for i in range(m - 1, -1, -1):   # ✗ 自下而上'
  return p === '64'
    ? `def min_path_sum(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    ${loop}
        for j in range(n):
            if i == 0 and j == 0:
                dp[i][j] = grid[0][0]
            elif i == 0:                      # 第一行只能从左边来
                dp[i][j] = dp[i][j-1] + grid[i][j]
            elif j == 0:                      # 第一列只能从上边来
                dp[i][j] = dp[i-1][j] + grid[i][j]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
    return dp[m-1][n-1]`
    : `def unique_paths(m, n):
    dp = [[0] * n for _ in range(m)]
    ${loop}
        for j in range(n):
            if i == 0 or j == 0:
                dp[i][j] = 1                  # 边界：只有一条路
            else:
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[m-1][n-1]`
}

interface Frame {
  dp: (number | null)[][]
  wrong: boolean[][]
  cur?: [number, number]
  deps: { r: number; c: number; ok: boolean; win: boolean }[]
  line: number | number[]
  caption: React.ReactNode
  path?: Set<string>
}

function build(p: Prob, o: Order): Frame[] {
  const dp: (number | null)[][] = GRID.map((row) => row.map(() => null))
  const wrong = GRID.map((row) => row.map(() => false))
  const fr: Frame[] = []
  const rowsOrder = o === 'row' ? [...Array(M).keys()] : [...Array(M).keys()].reverse()
  const snap = () => ({ dp: dp.map((r) => [...r]), wrong: wrong.map((r) => [...r]) })
  const get = (r: number, c: number) => dp[r][c] ?? 0 // Python 里初始化为 0：未计算的格子读出来是 0
  for (const i of rowsOrder) {
    for (let j = 0; j < N; j++) {
      const deps: Frame['deps'] = []
      if (i > 0 && (p === '64' || j > 0)) deps.push({ r: i - 1, c: j, ok: dp[i - 1][j] != null, win: false })
      if (j > 0 && (p === '64' || i > 0)) deps.push({ r: i, c: j - 1, ok: dp[i][j - 1] != null, win: false })
      let v: number
      let line: number | number[]
      let text: React.ReactNode
      if (p === '64') {
        if (i === 0 && j === 0) {
          v = GRID[0][0]
          line = 7
          text = <>起点 <C>dp[0][0] = grid[0][0] = {v}</C>。</>
        } else if (i === 0) {
          v = get(0, j - 1) + GRID[i][j]
          line = 9
          text = <>第一行只能从左边来：<C>dp[0][{j}] = {get(0, j - 1)} + {GRID[i][j]} = {v}</C>。</>
        } else if (j === 0) {
          v = get(i - 1, 0) + GRID[i][j]
          line = 11
          text = <>第一列只能从上边来：<C>dp[{i}][0] = {get(i - 1, 0)} + {GRID[i][j]} = {v}</C>。</>
        } else {
          const up = get(i - 1, j)
          const left = get(i, j - 1)
          v = Math.min(up, left) + GRID[i][j]
          deps[0].win = up <= left
          deps[1].win = !(up <= left)
          line = 13
          text = (
            <>
              <C>dp[{i}][{j}] = min(上 {up}, 左 {left}) + {GRID[i][j]} = {v}</C>。
            </>
          )
        }
        if (deps.length === 1) deps[0].win = true
      } else {
        if (i === 0 || j === 0) {
          v = 1
          line = 6
          text = <>边界格 <C>dp[{i}][{j}] = 1</C>：只能一路向右或一路向下。</>
        } else {
          v = get(i - 1, j) + get(i, j - 1)
          deps.forEach((d) => (d.win = true))
          line = 8
          text = <>到达 ({i},{j}) 的最后一步要么从上来、要么从左来：<C>dp[{i}][{j}] = {get(i - 1, j)} + {get(i, j - 1)} = {v}</C>。</>
        }
      }
      const bad = deps.some((d) => !d.ok || wrong[d.r][d.c])
      dp[i][j] = v
      wrong[i][j] = bad
      if (deps.some((d) => !d.ok))
        text = (
          <>
            {text} <b className="text-danger">✗ 依赖的格子还没算（读到初始值 0），结果是错的！</b>
          </>
        )
      fr.push({ ...snap(), cur: [i, j], deps, line, caption: text })
    }
  }
  const last = dp[M - 1][N - 1]!
  if (o === 'reverse') {
    const correct = solve(p)
    fr.push({
      ...snap(),
      deps: [],
      line: p === '64' ? 14 : 9,
      caption: (
        <>
          返回 <C>dp[{M - 1}][{N - 1}] = {last}</C>，正确答案是 <C>{correct}</C>。红色格子都受到了「先读后写」的污染——<b>遍历顺序必须和依赖方向一致</b>。
        </>
      ),
    })
  } else if (p === '64') {
    // 回溯最优路径
    const path = new Set<string>()
    let i = M - 1
    let j = N - 1
    path.add(`${i},${j}`)
    while (i > 0 || j > 0) {
      if (i === 0) j--
      else if (j === 0) i--
      else if (dp[i - 1][j]! <= dp[i][j - 1]!) i--
      else j--
      path.add(`${i},${j}`)
    }
    fr.push({
      ...snap(),
      deps: [],
      line: 14,
      path,
      caption: <>答案 <C>dp[{M - 1}][{N - 1}] = {last}</C>。从终点沿「较小的那个来源」倒着走回起点，就得到一条最优路径（绿色）。</>,
    })
  } else {
    fr.push({ ...snap(), deps: [], line: 9, caption: <>答案 <C>dp[{M - 1}][{N - 1}] = {last}</C> = C(7, 3) = 35：共 7 步中选 3 步向下。</> })
  }
  return fr
}

function solve(p: Prob) {
  const dp = GRID.map((r) => r.map(() => 0))
  for (let i = 0; i < M; i++)
    for (let j = 0; j < N; j++) {
      if (p === '64') dp[i][j] = GRID[i][j] + (i === 0 && j === 0 ? 0 : Math.min(i > 0 ? dp[i - 1][j] : Infinity, j > 0 ? dp[i][j - 1] : Infinity))
      else dp[i][j] = i === 0 || j === 0 ? 1 : dp[i - 1][j] + dp[i][j - 1]
    }
  return dp[M - 1][N - 1]
}

export function GridPathViz() {
  const [p, setP] = useState<Prob>('64')
  const [o, setO] = useState<Order>('row')
  const frames = useMemo(() => build(p, o), [p, o])
  const st = useStepper(frames, { intervalMs: 800 })
  const f = st.frame
  const arrows: DpArrow[] = f.cur
    ? f.deps.map((d) => ({ from: [d.r, d.c] as [number, number], to: f.cur!, tone: !d.ok ? 'danger' : d.win ? 'success' : 'muted', dashed: !d.win && d.ok }))
    : []
  const isCur = (r: number, c: number) => !!f.cur && f.cur[0] === r && f.cur[1] === c

  return (
    <VizFrame
      title="二维 DP：每格只看「上」和「左」，遍历顺序要跟着依赖走"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '当前格', tone: 'accent', variant: 'soft' },
            { label: '取用的依赖', tone: 'success', variant: 'line' },
            { label: '读到未计算格', tone: 'danger', variant: 'line' },
            ...(p === '64' ? [{ label: '最优路径', tone: 'success' as const, variant: 'soft' as const }] : []),
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Segmented
            size="sm"
            value={p}
            onChange={setP}
            options={[
              { value: '64', label: 'LC 64 最小路径和' },
              { value: '62', label: 'LC 62 不同路径' },
            ]}
          />
          <Segmented
            size="sm"
            value={o}
            onChange={setO}
            options={[
              { value: 'row', label: '行优先 ✓' },
              { value: 'reverse', label: '自下而上 ✗' },
            ]}
          />
        </>
      }
    >
      <div className="space-y-4">
        <div className={p === '64' ? 'grid items-start gap-4 @xl:grid-cols-[2fr_3fr]' : ''}>
          {p === '64' && (
            <div>
              <Sub>grid（每格的代价）</Sub>
              <DpGrid data={GRID} cellSize={38} ring={(r, c) => (isCur(r, c) ? 'accent' : f.path?.has(`${r},${c}`) ? 'success' : undefined)} />
            </div>
          )}
          <div>
            <Sub>{p === '64' ? 'dp[i][j]：从起点到 (i,j) 的最小路径和' : 'dp[i][j]：从起点到 (i,j) 的路径条数'}</Sub>
            <DpGrid
              data={f.dp as Cell[][]}
              rowLabels={GRID.map((_, i) => `i=${i}`)}
              colLabels={GRID[0].map((_, j) => `j=${j}`)}
              cellSize={50}
              arrows={arrows}
              ring={(r, c) => (isCur(r, c) ? 'accent' : f.path?.has(`${r},${c}`) ? 'success' : undefined)}
              fill={(r, c) => (f.wrong[r][c] ? 'danger' : f.path?.has(`${r},${c}`) ? 'success' : isCur(r, c) ? 'accent' : undefined)}
            />
          </div>
        </div>
        <CodeTrace code={code(p, o)} active={f.line} vars={f.cur ? { i: f.cur[0], j: f.cur[1], [`dp[${f.cur[0]}][${f.cur[1]}]`]: f.dp[f.cur[0]][f.cur[1]] ?? '—' } : undefined} />
      </div>
    </VizFrame>
  )
}
