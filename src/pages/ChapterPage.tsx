/** 章节页 /chapter/:chapterId：导语、关卡列表（阶段完成度 / 星级 / 代表题）、速记入口 */
import { Link, useParams } from 'react-router'
import { ArrowRight, Check, ChevronRight, Clock, Crown, Lightbulb, Lock, Map as MapIcon, Play, RotateCcw, ScrollText, Sparkles } from 'lucide-react'
import { useCatalog, type Catalog } from '@/content/catalog'
import type { ChapterDef } from '@/content/types'
import { useProgress } from '@/lib/store'
import { chapterStatuses, pendingPrerequisites, type LevelStatus } from '@/lib/unlock'
import { TRACK_META, chapterNext, chapterStats, levelHref } from '@/lib/selectors'
import { Card } from '@/components/ui'
import { Markdown } from '@/components/Markdown'
import { PageSpinner } from '@/components/PageSpinner'
import { ChapterIcon } from '@/components/ChapterIcon'
import { Stars } from '@/components/game/Stars'
import { BossBadge, FreqBadge, TierBadge } from '@/components/game/TierBadge'
import { Bar } from '@/components/game/ProgressRing'
import { DifficultyDots, Minutes, StagePips } from '@/components/game/LevelBits'
import { cn } from '@/lib/utils'
import NotFound from './NotFound'

export default function ChapterPage() {
  const { chapterId } = useParams()
  const cat = useCatalog()
  if (!cat) return <PageSpinner />
  const chapter = chapterId ? cat.chapterById.get(chapterId) : undefined
  if (!chapter) return <NotFound title="章节不存在">没有找到章节「{chapterId}」，它可能还在制作中。</NotFound>
  return <ChapterInner cat={cat} chapter={chapter} />
}

function ChapterInner({ cat, chapter }: { cat: Catalog; chapter: ChapterDef }) {
  const levels = useProgress((s) => s.levels)
  const prereqs = pendingPrerequisites(cat, chapter, levels)
  const freeMode = useProgress((s) => s.settings.freeMode)
  const sts = chapterStatuses(chapter, levels, freeMode)
  const st = chapterStats(chapter, levels)
  const next = chapterNext(chapter, levels, freeMode)
  const totalMin = chapter.levels.reduce((n, l) => n + l.estMinutes, 0)
  const started = st.cleared > 0 || chapter.levels.some((l) => levels[l.id])
  const allDone = st.levels > 0 && st.cleared === st.levels

  return (
    <div className={cn('space-y-6', `track-${chapter.track}`)}>
      <nav className="flex items-center gap-1 text-[13px] text-muted" aria-label="面包屑">
        <Link to={`/track/${chapter.track}`} className="hover:text-fg">
          {TRACK_META[chapter.track].title}
        </Link>
        <ChevronRight size={14} className="text-subtle" />
        <span>第 {chapter.order} 章</span>
      </nav>

      <Card className="relative overflow-hidden">
        <div className="pointer-events-none absolute -right-16 -top-16 h-56 w-56 rounded-full bg-track-soft opacity-70" aria-hidden />
        <div className="relative p-5 md:p-7">
          <div className="flex items-start gap-4">
            <div className="grid h-14 w-14 shrink-0 place-items-center rounded-2xl bg-track text-accent-fg shadow-sm dark:text-bg md:h-16 md:w-16">
              <ChapterIcon name={chapter.icon} size={28} />
            </div>
            <div className="min-w-0 flex-1">
              <h1 className="text-xl font-bold leading-snug tracking-tight md:text-2xl">{chapter.title}</h1>
              <p className="mt-1 text-sm text-muted">{chapter.subtitle}</p>
            </div>
          </div>
          <div className="mt-5 grid grid-cols-3 gap-2 sm:max-w-md">
            <MiniStat label="通关" value={`${st.cleared}/${st.levels}`} />
            <MiniStat label="星数" value={`${st.stars}/${st.maxStars}`} />
            <MiniStat label="预计" value={totalMin >= 60 ? `${(totalMin / 60).toFixed(1)}h` : `${totalMin}m`} />
          </div>
          <Bar value={st.levels ? st.cleared / st.levels : 0} color="var(--track)" className="mt-4 sm:max-w-md" height={6} />
          <div className="mt-5 flex flex-wrap gap-2">
            {next && (
              <Link
                to={next.unlocked ? levelHref(next.meta.id) : `/track/${chapter.track}?focus=${next.meta.id}`}
                className="inline-flex h-11 items-center gap-2 rounded-xl bg-track px-5 text-sm font-semibold text-accent-fg shadow-sm transition-[filter] hover:brightness-110 dark:text-bg"
              >
                {allDone ? <RotateCcw size={16} /> : <Play size={16} className="fill-current" />}
                {allDone ? '复习本章' : started ? '继续本章' : '开始本章'}
                {!allDone && <span className="hidden font-normal opacity-80 sm:inline">· {next.meta.title}</span>}
              </Link>
            )}
            <Link
              to={`/cheatsheet?chapter=${chapter.id}`}
              className="inline-flex h-11 items-center gap-2 rounded-xl border border-border bg-surface px-4 text-sm font-medium transition-colors hover:bg-surface-2"
            >
              <ScrollText size={16} /> 本章速记
            </Link>
            <Link
              to={`/track/${chapter.track}`}
              className="inline-flex h-11 items-center gap-2 rounded-xl px-3 text-sm font-medium text-muted transition-colors hover:bg-surface-2 hover:text-fg"
            >
              <MapIcon size={16} /> 地图
            </Link>
          </div>
        </div>
        {prereqs.length > 0 && (
          <div className="relative flex items-start gap-2 border-t border-border bg-surface-2/50 px-5 py-3 text-[13px] text-muted md:px-7">
            <Lightbulb size={15} className="mt-0.5 shrink-0 ink-warning" />
            <span>
              建议先学：
              {prereqs.map((p, i) => (
                <span key={p.id}>
                  {i > 0 && '、'}
                  <Link className="font-medium text-fg underline-offset-2 hover:underline" to={`/chapter/${p.id}`}>
                    {p.title}
                  </Link>
                </span>
              ))}
              （只是建议，不影响解锁）
            </span>
          </div>
        )}
      </Card>

      {chapter.intro && (
        <section>
          <h2 className="mb-2 text-[15px] font-semibold">本章导语</h2>
          <Card className="p-5 md:p-6">
            <Markdown>{chapter.intro}</Markdown>
          </Card>
        </section>
      )}

      <section>
        <div className="mb-3 flex items-baseline justify-between">
          <h2 className="text-[15px] font-semibold">关卡 · {st.levels}</h2>
          <span className="flex items-center gap-1 text-xs text-subtle">
            <Clock size={12} /> 共约 {totalMin} 分钟
          </span>
        </div>
        {sts.length === 0 ? (
          <Card className="p-6 text-center text-sm text-muted">本章关卡正在制作中。</Card>
        ) : (
          <ol className="space-y-3">
            {sts.map((s) => (
              <LevelRow key={s.meta.id} s={s} track={chapter.track} />
            ))}
          </ol>
        )}
      </section>
    </div>
  )
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl bg-surface-2/80 px-3 py-2">
      <div className="text-[11px] text-muted">{label}</div>
      <div className="text-[17px] font-bold tabular-nums">{value}</div>
    </div>
  )
}

function LevelRow({ s, track }: { s: LevelStatus; track: string }) {
  const p = useProgress((st) => st.levels[s.meta.id])
  const m = s.meta
  const inProgress = !!p && (p.sectionsRead.length > 0 || p.quizAttempts > 0)
  const href = s.unlocked ? levelHref(m.id) : `/track/${track}?focus=${m.id}`
  return (
    <li>
      <Link
        to={href}
        className={cn(
          'group flex gap-3.5 rounded-2xl border bg-surface p-4 shadow-sm transition-[border-color,box-shadow] hover:shadow-md md:gap-4 md:p-5',
          s.current ? 'border-track' : 'border-border hover:border-border-strong',
          s.isBoss && 'bg-gradient-to-br from-gold-soft/50 to-surface',
        )}
      >
        <div
          className={cn(
            'grid h-11 w-11 shrink-0 place-items-center rounded-full text-[15px] font-black tabular-nums md:h-12 md:w-12',
            s.state === 'cleared' && 'bg-track text-accent-fg dark:text-bg',
            (s.state === 'current' || s.state === 'available') && 'bg-track-soft text-track ring-2 ring-inset ring-[var(--track)]',
            s.state === 'locked' && 'bg-surface-3 text-subtle',
          )}
        >
          {s.state === 'locked' ? <Lock size={17} /> : s.state === 'cleared' ? <Check size={20} strokeWidth={3} /> : s.isBoss ? <Crown size={19} /> : s.index + 1}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-1.5">
            <TierBadge tier={m.tier} size="xs" />
            {m.frequency && <FreqBadge freq={m.frequency} size="xs" />}
            {s.isBoss && <BossBadge size="xs" />}
            {s.current && <span className="text-[11px] font-semibold ink-track">{inProgress ? '进行中' : '下一关'}</span>}
          </div>
          <h3 className={cn('mt-1.5 font-semibold leading-snug md:text-[16px]', s.state === 'locked' && 'text-muted')}>{m.title}</h3>
          <p className="mt-0.5 text-[13px] leading-relaxed text-muted">{m.summary}</p>
          {!!m.problems?.length && (
            <div className="mt-2 flex flex-wrap gap-1">
              {m.problems.map((x) => (
                <span key={x} className="rounded-md bg-surface-2 px-1.5 py-0.5 text-[11.5px] text-muted">
                  {x}
                </span>
              ))}
            </div>
          )}
          {m.llmLink && (
            <div className="mt-2 flex items-start gap-1.5 text-[12.5px] text-muted">
              <Sparkles size={13} className="mt-0.5 shrink-0 ink-llm" />
              <span>{m.llmLink}</span>
            </div>
          )}
          <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2">
            <Minutes value={m.estMinutes} />
            <DifficultyDots value={m.difficulty} />
            {s.cleared && <Stars value={s.stars} size={13} />}
            {!s.unlocked && <span className="text-xs text-subtle">{s.lockReason}</span>}
          </div>
          {p && <StagePips progress={p} showLabels className="mt-3 max-w-sm" />}
        </div>
        <ArrowRight size={18} className="mt-1 hidden shrink-0 self-center text-subtle transition-transform group-hover:translate-x-0.5 sm:block" />
      </Link>
    </li>
  )
}
