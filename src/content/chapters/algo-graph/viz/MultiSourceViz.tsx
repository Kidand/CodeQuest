/**
 * 腐烂的橘子（LC 994）：多源 BFS = 把所有源点一起放进第 0 层。
 * 每一层就是一分钟；格子里的数字是它腐烂的时刻。
 */
import { useMemo, useState } from 'react'
import { Legend, StepControls, VizFrame, useStepper } from '@/viz'
import { Segmented } from '@/components/ui'
import { C } from './common'
import { GridBoard } from './GridBoard'

const PRESETS = {
  ok: ['211011', '110011', '011110', '110112', '010011'],
  island: ['211011', '110011', '011100', '110112', '010011'],
} as const
type Preset = keyof typeof PRESETS
const DIRS = [[1, 0], [-1, 0], [0, 1], [0, -1]] as const

interface Frame {
  t: number
  time: number[][] // -1 = 新鲜未腐烂
  frontier: Set<string>
  fresh: number
  caption: React.ReactNode
}

function buildFrames(preset: Preset): Frame[] {
  const g = PRESETS[preset].map((r) => r.split('').map(Number))
  const m = g.length
  const n = g[0].length
  const time: number[][] = g.map((r) => r.map((v) => (v === 2 ? 0 : -1)))
  let q: [number, number][] = []
  let fresh = 0
  g.forEach((row, i) => row.forEach((v, j) => (v === 2 ? q.push([i, j]) : v === 1 ? fresh++ : 0)))
  const frames: Frame[] = []
  const key = (q: [number, number][]) => new Set(q.map(([r, c]) => `${r},${c}`))
  frames.push({
    t: 0, time: time.map((r) => [...r]), frontier: key(q), fresh,
    caption: <>第 0 分钟：<b>所有</b>烂橘子（{q.length} 个）同时入队，相当于从一个连向它们的“超级源点”出发做 BFS。新鲜橘子 {fresh} 个。</>,
  })
  let t = 0
  while (q.length && fresh) {
    t++
    const nq: [number, number][] = []
    for (const [r, c] of q)
      for (const [dr, dc] of DIRS) {
        const nr = r + dr
        const nc = c + dc
        if (nr >= 0 && nr < m && nc >= 0 && nc < n && g[nr][nc] === 1) {
          g[nr][nc] = 2
          time[nr][nc] = t
          fresh--
          nq.push([nr, nc])
        }
      }
    q = nq
    frames.push({
      t, time: time.map((r) => [...r]), frontier: key(q), fresh,
      caption: <>第 {t} 分钟：上一层的每个烂橘子感染四邻，新腐烂 {q.length} 个（高亮），剩余新鲜 {fresh} 个。</>,
    })
  }
  frames.push({
    t, time: time.map((r) => [...r]), frontier: new Set(), fresh,
    caption:
      fresh === 0 ? (
        <>全部腐烂，答案 <b>{t}</b> 分钟 = BFS 的层数。每个格子入队一次，总时间 O(mn)；如果对每个源点分别 BFS 再取最小，会变成 O(k·mn)。</>
      ) : (
        <>
          队列空了但还剩 <b>{fresh}</b> 个新鲜橘子（红框）——它们和任何烂橘子都不连通，返回 <C>-1</C>。
        </>
      ),
  })
  return frames
}

export function MultiSourceViz() {
  const [preset, setPreset] = useState<Preset>('ok')
  const frames = useMemo(() => buildFrames(preset), [preset])
  const st = useStepper(frames, { intervalMs: 1100 })
  const f = st.frame
  const g = PRESETS[preset].map((r) => r.split('').map(Number))
  const end = st.atEnd
  return (
    <VizFrame
      title="多源 BFS：所有源点一起出发"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '已腐烂（数字 = 第几分钟）', tone: 'warning', variant: 'soft' },
            { label: '本分钟新腐烂', tone: 'danger', variant: 'line' },
            { label: '新鲜', tone: 'success', variant: 'soft' },
            { label: '空格', tone: 'muted', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Segmented size="sm" value={preset} onChange={setPreset} options={[{ value: 'ok', label: '全部可达' }, { value: 'island', label: '有孤立橘子' }]} />
        </>
      }
    >
      <div className="mx-auto max-w-md space-y-3">
        <GridBoard
          rows={g.length}
          cols={g[0].length}
          max={46}
          cell={(r, c) => {
            if (g[r][c] === 0) return { fill: 'water', text: '' }
            const tm = f.time[r][c]
            if (tm >= 0) return { fill: 'warning', text: tm, ring: f.frontier.has(`${r},${c}`) && f.t > 0 ? 'danger' : undefined }
            return { fill: 'success', text: '🍊', ring: end && f.fresh > 0 ? 'danger' : undefined, dashed: end && f.fresh > 0 }
          }}
        />
        <div className="text-center font-mono text-[12.5px] text-muted">
          minute = <b className="text-fg">{f.t}</b> · fresh = <b className="text-fg">{f.fresh}</b>
        </div>
      </div>
    </VizFrame>
  )
}
