import { forwardRef } from 'react'
import { Link } from 'react-router'
import { Check, ChevronLeft, NotebookText, Star } from 'lucide-react'
import { STAGES, TIER_LABEL, type LevelContent, type Stage } from '@/content/types'
import type { LevelRef } from '@/content/catalog'
import type { LevelProgress } from '@/lib/game'
import { starsOf } from '@/lib/game'
import { Badge, ThemeToggle } from '@/components/ui'
import { cn } from '@/lib/utils'
import { STAGE_META, stageState } from './status'

function Stepper({
  L,
  p,
  stage,
  onGo,
  compact,
}: {
  L: LevelContent
  p: LevelProgress | undefined
  stage: Stage
  onGo: (s: Stage) => void
  compact?: boolean
}) {
  return (
    <nav aria-label="关卡阶段" className={cn('flex items-center', compact ? 'w-full' : 'gap-0.5')}>
      {STAGES.map((s, i) => {
        const st = stageState(s, L, p)
        const meta = STAGE_META[s]
        const Icon = meta.icon
        const cur = s === stage
        return (
          <div key={s} className={cn('flex items-center', compact && 'flex-1')}>
            {i > 0 && !compact && <span className={cn('mx-0.5 h-px w-3 xl:w-5', st.done && !st.empty ? 'bg-success/60' : 'bg-border-strong')} />}
            <button
              type="button"
              onClick={() => onGo(s)}
              data-stage={s}
              aria-current={cur ? 'step' : undefined}
              title={`${meta.label}：${st.text}`}
              className={cn(
                'group relative flex items-center gap-1.5 rounded-xl transition-colors',
                compact ? 'h-11 w-full flex-col justify-center gap-0.5 px-0.5' : 'h-9 px-2',
                cur ? 'bg-accent-soft text-accent' : 'text-muted hover:bg-surface-2 hover:text-fg',
                st.empty && 'opacity-45',
              )}
            >
              <span
                className={cn(
                  'relative inline-flex shrink-0 items-center justify-center rounded-full',
                  compact ? 'h-5 w-5' : 'h-6 w-6',
                  st.done && !st.empty ? 'bg-success text-surface' : cur ? 'bg-accent text-accent-fg' : 'bg-surface-3 text-muted',
                )}
              >
                {st.done && !st.empty ? <Check size={compact ? 12 : 14} strokeWidth={3} /> : <Icon size={compact ? 11 : 13} />}
                {!st.done && st.ratio > 0 && !cur && (
                  <svg viewBox="0 0 24 24" className="absolute -inset-[3px] h-[calc(100%+6px)] w-[calc(100%+6px)] -rotate-90">
                    <circle cx="12" cy="12" r="11" fill="none" stroke="var(--accent)" strokeWidth="2" strokeDasharray={`${st.ratio * 69.1} 69.1`} />
                  </svg>
                )}
              </span>
              <span className={cn('whitespace-nowrap font-medium', compact ? 'text-[11px] leading-none' : 'text-[13px] max-xl:hidden')}>
                {compact ? meta.short : meta.label}
              </span>
            </button>
          </div>
        )
      })}
    </nav>
  )
}

export const LevelTopBar = forwardRef<
  HTMLElement,
  {
    levelRef: LevelRef
    L: LevelContent
    p: LevelProgress | undefined
    stage: Stage
    onGo: (s: Stage) => void
    onCheatsheet: () => void
  }
>(function LevelTopBar({ levelRef, L, p, stage, onGo, onCheatsheet }, ref) {
  const stars = starsOf(p)
  const { chapter, meta } = levelRef
  return (
    <header ref={ref} className="sticky top-0 z-40 border-b border-border bg-surface/90 pt-[env(safe-area-inset-top)] backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-[1600px] items-center gap-2 px-2 sm:px-4">
        <Link
          to={`/chapter/${chapter.id}`}
          className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-xl text-muted hover:bg-surface-2 hover:text-fg"
          aria-label={`返回章节：${chapter.title}`}
          title={`返回「${chapter.title}」`}
        >
          <ChevronLeft size={20} />
        </Link>
        <div className="min-w-0 flex-1 lg:flex-none lg:max-w-[22rem] xl:max-w-[28rem]">
          <div className="truncate text-[11px] leading-tight text-subtle">{chapter.title}</div>
          <div className="flex min-w-0 items-center gap-1.5">
            <h1 className="truncate text-[15px] font-semibold leading-tight text-fg">{meta.title}</h1>
            <Badge tone={chapter.track === 'llm' ? 'llm' : 'algo'} className="shrink-0 px-1.5 py-0 text-[10.5px] max-sm:hidden">
              {TIER_LABEL[meta.tier]}
            </Badge>
          </div>
        </div>
        <div className="hidden min-w-0 flex-1 justify-center lg:flex">
          <Stepper L={L} p={p} stage={stage} onGo={onGo} />
        </div>
        <div className="flex shrink-0 items-center gap-0.5">
          <span className="mr-1 inline-flex items-center gap-0.5" title={`本关 ${stars} 星`} data-stars={stars}>
            {[1, 2, 3].map((k) => (
              <Star key={k} size={14} className={k <= stars ? 'text-gold' : 'text-border-strong'} fill={k <= stars ? 'currentColor' : 'none'} />
            ))}
          </span>
          <button
            type="button"
            onClick={onCheatsheet}
            className="inline-flex h-9 items-center gap-1.5 rounded-xl px-2.5 text-muted hover:bg-surface-2 hover:text-fg"
            title="速记卡"
            data-action="cheatsheet"
          >
            <NotebookText size={18} />
            <span className="text-sm max-md:hidden">速记卡</span>
          </button>
          <ThemeToggle />
        </div>
      </div>
      <div className="border-t border-border/60 px-1 lg:hidden">
        <Stepper L={L} p={p} stage={stage} onGo={onGo} compact />
      </div>
    </header>
  )
})
