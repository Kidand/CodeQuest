/**
 * 静态批处理 vs 迭代级调度（Orca 的 continuous batching）的甘特图。
 * 8 个请求同时到达，输出长度不同，batch 最多 4 个槽位；每次迭代每个运行中的请求生成 1 个 token。
 */
import { useMemo, useState } from 'react'
import { Legend, StepControls, VizFrame, catColor, toneColor, useStepper } from '@/viz'
import { Segmented } from '@/components/ui'
import { Stat } from './common'

const LENS = [5, 12, 3, 8, 10, 4, 7, 6]
const SLOTS = 4

interface Seg {
  r: number
  slot: number
  start: number // 第一个 token 所在的迭代（0 起）
  end: number // 最后一个 token 所在迭代 + 1
  idleEnd?: number // 静态批：占着槽位空等到这里
}

export function schedule(lens: number[], slots: number, mode: 'static' | 'cont') {
  const segs: Seg[] = []
  if (mode === 'static') {
    let t = 0
    for (let i = 0; i < lens.length; i += slots) {
      const group = lens.slice(i, i + slots)
      const m = Math.max(...group)
      group.forEach((L, k) => segs.push({ r: i + k, slot: k, start: t, end: t + L, idleEnd: t + m }))
      t += m
    }
    return { segs, total: t }
  }
  const free = Array(slots).fill(0) // 每个槽位下一次空闲的迭代
  lens.forEach((L, r) => {
    let k = 0
    for (let j = 1; j < slots; j++) if (free[j] < free[k]) k = j
    segs.push({ r, slot: k, start: free[k], end: free[k] + L })
    free[k] += L
  })
  return { segs, total: Math.max(...free) }
}

export function BatchingViz() {
  const [mode, setMode] = useState<'static' | 'cont'>('static')
  const plan = useMemo(() => schedule(LENS, SLOTS, mode), [mode])
  const other = useMemo(() => schedule(LENS, SLOTS, mode === 'static' ? 'cont' : 'static'), [mode])
  const T = Math.max(plan.total, other.total)
  const frames = useMemo(() => Array.from({ length: plan.total + 1 }, (_, t) => t), [plan])
  const st = useStepper(frames, { intervalMs: 450 })
  const t = st.frame
  const work = LENS.reduce((a, b) => a + b, 0)
  const util = work / (SLOTS * plan.total)
  const avgDone = plan.segs.reduce((a, s) => a + s.end, 0) / LENS.length
  const running = plan.segs.filter((s) => s.start <= t && t < s.end)
  const waiting = plan.segs.filter((s) => s.start > t).length
  const doneN = plan.segs.filter((s) => s.end <= t).length
  const idleNow = plan.segs.filter((s) => s.idleEnd && s.end <= t && t < s.idleEnd).length

  return (
    <VizFrame
      title="静态批 vs 迭代级调度"
      caption={
        mode === 'static' ? (
          <>
            静态批处理：一个 batch 必须等<b>最长</b>的请求生成完才能整体退出，短请求早早结束后槽位空转（斜线）。第 {t} 次迭代：运行 {running.length} 个、空转 {idleNow} 个槽位、排队 {waiting} 个。
            共 {plan.total} 次迭代，槽位利用率 {(util * 100).toFixed(0)}%。
          </>
        ) : (
          <>
            迭代级调度：每次迭代结束都检查一遍，有请求完成就<b>立刻</b>让排队的请求补进来。第 {t} 次迭代：运行 {running.length} 个、排队 {waiting} 个。共 {plan.total} 次迭代（静态批要{' '}
            {other.total} 次），利用率 {(util * 100).toFixed(0)}%。
          </>
        )
      }
      legend={
        <Legend
          items={[
            { label: '生成中', tone: 1, variant: 'solid' },
            { label: '已完成但占着槽位（空转）', tone: 'muted', variant: 'soft' },
            { label: '当前迭代', tone: 'danger', variant: 'line' },
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
              { value: 'static', label: '静态批处理' },
              { value: 'cont', label: '迭代级调度（Orca）' },
            ]}
          />
        </>
      }
    >
      <div className="space-y-3">
        <div className="relative">
          {Array.from({ length: SLOTS }, (_, k) => (
            <div key={k} className="mb-1.5 grid items-center gap-2" style={{ gridTemplateColumns: '2.6rem minmax(0,1fr)' }}>
              <span className="font-mono text-[11px] text-muted">槽 {k}</span>
              <div className="relative h-7 rounded bg-surface-2">
                {plan.segs
                  .filter((s) => s.slot === k)
                  .map((s) => {
                    const vis = Math.min(s.end, Math.max(s.start, t + 1))
                    const L = (x: number) => `${(x / T) * 100}%`
                    return (
                      <div key={s.r}>
                        {s.idleEnd && s.idleEnd > s.end && t >= s.end && (
                          <div
                            className="absolute inset-y-0.5 rounded"
                            style={{
                              left: L(s.end),
                              width: L(Math.min(s.idleEnd, t + 1) - s.end),
                              backgroundImage: `repeating-linear-gradient(135deg, transparent 0 3px, color-mix(in oklab, var(--subtle) 60%, transparent) 3px 4.5px)`,
                            }}
                          />
                        )}
                        {vis > s.start && (
                          <div
                            className="absolute inset-y-0.5 flex items-center overflow-hidden rounded px-1 font-mono text-[10.5px] font-semibold"
                            style={{
                              left: L(s.start),
                              width: L(vis - s.start),
                              background: `color-mix(in oklab, ${catColor(s.r)} ${t < s.end ? 70 : 40}%, var(--surface))`,
                              color: 'var(--fg)',
                            }}
                          >
                            R{s.r + 1}
                          </div>
                        )}
                      </div>
                    )
                  })}
                <div className="absolute -inset-y-1 w-0.5" style={{ left: `${((t + 0.5) / T) * 100}%`, background: toneColor('danger') }} />
              </div>
            </div>
          ))}
          <div className="grid gap-2" style={{ gridTemplateColumns: '2.6rem minmax(0,1fr)' }}>
            <span />
            <div className="relative h-4 font-mono text-[10px] text-subtle">
              {Array.from({ length: Math.floor(T / 5) + 1 }, (_, i) => i * 5).map((x) => (
                <span key={x} className="absolute -translate-x-1/2" style={{ left: `${(x / T) * 100}%` }}>
                  {x}
                </span>
              ))}
            </div>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-2 @xl:grid-cols-4">
          <Stat label="总迭代数" value={plan.total} tone={mode === 'cont' ? 'success' : 'danger'} />
          <Stat label="槽位利用率" value={`${(util * 100).toFixed(0)}%`} />
          <Stat label="平均完成时刻" value={avgDone.toFixed(1)} sub="越小 → 平均延迟越低" />
          <Stat label="已完成" value={`${doneN} / ${LENS.length}`} />
        </div>
        <div className="font-mono text-[11px] text-muted">输出长度：{LENS.map((L, i) => `R${i + 1}=${L}`).join(' · ')}（全部在第 0 次迭代到达，FIFO）</div>
      </div>
    </VizFrame>
  )
}
