/** ④ 实战：多道题用标签切换；工作台占满剩余视口高度 */
import { useState } from 'react'
import { ArrowRight, CheckCircle2, Circle } from 'lucide-react'
import { ChallengeWorkspace } from '@/components/code/ChallengeWorkspace'
import { DIFF_META } from '@/components/code/panels'
import { useProgress } from '@/lib/store'
import { cn } from '@/lib/utils'
import { useLevel } from './context'
import { EmptyStage } from './QuizStage'
import { STAGE_META } from './status'

export function ChallengeStage() {
  const { levelId, L, next, goStage } = useLevel()
  const records = useProgress((s) => s.levels[levelId]?.challenges)
  const [cur, setCur] = useState(() => {
    const i = L.challenges.findIndex((c) => !records?.[c.id]?.passed)
    return i < 0 ? 0 : i
  })
  if (L.challenges.length === 0) return <EmptyStage text="本关没有实战题" />
  const c = L.challenges[Math.min(cur, L.challenges.length - 1)]
  const allPassed = L.challenges.every((x) => records?.[x.id]?.passed)

  return (
    <div className="flex h-[calc(100dvh-var(--lv-top,56px))] flex-col lg:px-4 lg:pb-4">
      <div className="no-scrollbar flex shrink-0 items-center gap-2 overflow-x-auto px-3 py-2 lg:px-0">
        {L.challenges.map((x, i) => {
          const ok = records?.[x.id]?.passed
          return (
            <button
              key={x.id}
              type="button"
              onClick={() => setCur(i)}
              data-challenge-tab={x.id}
              className={cn(
                'inline-flex h-9 max-w-[16rem] shrink-0 items-center gap-1.5 rounded-xl border px-3 text-[13px] font-medium transition-colors',
                i === cur ? 'border-accent bg-accent-soft text-fg' : 'border-border bg-surface text-muted hover:text-fg',
              )}
              title={x.title}
            >
              {ok ? <CheckCircle2 size={15} className="shrink-0 text-success" /> : <Circle size={15} className="shrink-0 text-subtle" />}
              <span className="truncate">{x.title}</span>
              <span
                className={cn(
                  'shrink-0 text-[11px]',
                  x.difficulty === 'easy' ? 'text-success' : x.difficulty === 'medium' ? 'text-warning' : 'text-danger',
                )}
              >
                {DIFF_META[x.difficulty].label}
              </span>
            </button>
          )
        })}
        {next && (
          <button
            type="button"
            onClick={() => goStage(next)}
            className={cn(
              'ml-auto inline-flex h-9 shrink-0 items-center gap-1 rounded-xl px-3 text-[13px] font-semibold transition-colors',
              allPassed ? 'bg-accent text-accent-fg hover:bg-accent-hover' : 'text-muted hover:bg-surface-2 hover:text-fg',
            )}
            data-action="next-stage"
          >
            {STAGE_META[next].label}
            <ArrowRight size={15} />
          </button>
        )}
      </div>
      <div className="min-h-0 flex-1 max-lg:border-t max-lg:border-border">
        <ChallengeWorkspace key={c.id} levelId={levelId} challenge={c} className="max-lg:rounded-none max-lg:border-0" />
      </div>
    </div>
  )
}
