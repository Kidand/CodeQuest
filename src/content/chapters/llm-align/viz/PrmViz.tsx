/**
 * ORM vs PRM：ORM 对整条解答给一个分；PRM 对每一步给“这一步正确”的概率，再聚合（Let's Verify 用连乘）。
 * 打开“第 4 步算错”，看 PRM 如何定位第一个错误步骤。
 */
import { useState } from 'react'
import { motion } from 'motion/react'
import { Legend, VizFrame, toneColor, toneSoft } from '@/viz'
import { Segmented, Switch } from '@/components/ui'
import { Stat, f3 } from './plot'

type Agg = 'prod' | 'min' | 'last'
const GOOD = [
  { text: '设所求的数为 x', p: 0.99 },
  { text: '由题意：2x + 3 = 11', p: 0.97 },
  { text: '移项：2x = 8', p: 0.96 },
  { text: '两边除以 2：x = 4', p: 0.95 },
  { text: '代回检验：2·4 + 3 = 11 ✓', p: 0.93 },
  { text: '答案：4', p: 0.94 },
]
const BAD = [
  GOOD[0],
  GOOD[1],
  GOOD[2],
  { text: '两边除以 2：x = 5', p: 0.08 },
  { text: '所以 x = 5 满足条件', p: 0.21 },
  { text: '答案：5', p: 0.1 },
]

export function PrmViz() {
  const [err, setErr] = useState(true)
  const [agg, setAgg] = useState<Agg>('prod')
  const steps = err ? BAD : GOOD
  const orm = err ? 0.07 : 0.91
  const ps = steps.map((s) => s.p)
  const score = agg === 'prod' ? ps.reduce((a, b) => a * b, 1) : agg === 'min' ? Math.min(...ps) : ps[ps.length - 1]
  const first = ps.findIndex((p) => p < 0.5)

  return (
    <VizFrame
      title="ORM 只给总分，PRM 能指出错在哪一步"
      caption={
        err ? (
          <>
            ORM 给整条解答 {f3(orm)}，只知道“错了”；PRM 在第 <b>{first + 1}</b> 步给出 {f3(ps[first])}，定位了第一个错误——前 3 步其实是对的，
            这对 RL 的信用分配和搜索时的剪枝都更有用。代价是需要步骤级标注（PRM800K 由人工标了 80 万个步骤标签）或昂贵的自动标注。
          </>
        ) : (
          <>解答正确时 ORM 与 PRM 都给高分。注意按连乘聚合时，步数越多分数天然越低——长解答会被系统性地压分，这是连乘聚合的一个偏置。</>
        )
      }
      legend={
        <Legend
          items={[
            { label: 'PRM：该步正确的概率', tone: 'success', variant: 'soft' },
            { label: '第一个错误步骤', tone: 'danger', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <Switch checked={err} onChange={setErr} label="第 4 步算错" />
          <Segmented
            size="sm"
            value={agg}
            onChange={setAgg}
            options={[
              { value: 'prod', label: '连乘（Let’s Verify）' },
              { value: 'min', label: '取最小' },
              { value: 'last', label: '最后一步' },
            ]}
          />
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-[minmax(0,3fr)_minmax(0,1fr)]">
        <div className="flex flex-col gap-1.5">
          {steps.map((s, i) => {
            const isFirst = i === first
            return (
              <div
                key={i}
                className="flex items-center gap-2 rounded-lg border px-2 py-1.5 text-[13px]"
                style={{ borderColor: isFirst ? toneColor('danger') : 'var(--border)', background: isFirst ? toneSoft('danger', 12) : 'var(--surface)' }}
              >
                <span className="w-5 shrink-0 font-mono text-xs text-subtle">{i + 1}</span>
                <span className="min-w-0 flex-1 truncate text-fg">{s.text}</span>
                <div className="relative h-3 w-16 shrink-0 overflow-hidden rounded bg-surface-2 @md:w-24">
                  <motion.div
                    className="absolute inset-y-0 left-0 rounded"
                    initial={false}
                    animate={{ width: `${s.p * 100}%` }}
                    style={{ background: toneColor(s.p < 0.5 ? 'danger' : 'success') }}
                  />
                </div>
                <span className="w-10 shrink-0 text-right font-mono text-xs tabular-nums text-muted">{s.p.toFixed(2)}</span>
              </div>
            )
          })}
        </div>
        <div className="grid grid-cols-2 gap-2 @3xl:grid-cols-1">
          <Stat label="ORM（整条）" value={f3(orm)} tone="info" />
          <Stat label={`PRM（${agg === 'prod' ? '连乘' : agg === 'min' ? '最小' : '最后一步'}）`} value={f3(score)} tone="success" />
          <Stat label="第一个错误步骤" value={first >= 0 ? `第 ${first + 1} 步` : '无'} tone={first >= 0 ? 'danger' : 'success'} />
        </div>
      </div>
    </VizFrame>
  )
}
