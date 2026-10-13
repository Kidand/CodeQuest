/**
 * 闯关地图 /track/:track —— Duolingo 式蜿蜒路径。
 * 每章一个区块；节点显示 锁定 / 可玩 / 当前 / 已通关 + 星级，BOSS 更大带皇冠；
 * 点击节点弹出卡片；顶部 tier chips 高亮筛选；?focus=<levelId> 自动定位并弹出该节点。
 */
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router'
import { ArrowRight, BookOpen, Check, Crown, Lightbulb, Lock, Map as MapIcon, Play, RotateCcw, Sparkles, Star, Swords, X } from 'lucide-react'
import { useCatalog, type Catalog } from '@/content/catalog'
import { TIER_LABEL, type ChapterDef, type Tier, type Track } from '@/content/types'
import { useProgress } from '@/lib/store'
import { chapterStatuses, pendingPrerequisites, type LevelStatus } from '@/lib/unlock'
import { TRACK_META, TRACKS, chapterStats, levelHref, rememberTrack, trackStats } from '@/lib/selectors'
import { useElementWidth } from '@/viz/useWidth'
import { Button, Segmented } from '@/components/ui'
import { PageSpinner } from '@/components/PageSpinner'
import { ChapterIcon } from '@/components/ChapterIcon'
import { Stars } from '@/components/game/Stars'
import { BossBadge, FreqBadge, TierBadge, tierDotClass } from '@/components/game/TierBadge'
import { Bar } from '@/components/game/ProgressRing'
import { DifficultyDots, Minutes, StagePips } from '@/components/game/LevelBits'
import { EmptyState } from '@/components/game/EmptyState'
import { cn } from '@/lib/utils'
import NotFound from './NotFound'

export default function MapPage() {
  const { track } = useParams()
  const cat = useCatalog()
  if (track !== 'algo' && track !== 'llm') return <NotFound />
  if (!cat) return <PageSpinner />
  return <MapInner key={track} cat={cat} track={track} />
}

function MapInner({ cat, track }: { cat: Catalog; track: Track }) {
  const navigate = useNavigate()
  const [params, setParams] = useSearchParams()
  const levels = useProgress((s) => s.levels)
  const meta = TRACK_META[track]
  const chapters = cat.byTrack[track]
  const stats = trackStats(cat, track, levels)
  const [tier, setTier] = useState<Tier | 'all'>('all')
  const focus = params.get('focus')
  const [selected, setSelected] = useState<string | null>(focus)

  useEffect(() => rememberTrack(track), [track])
  useEffect(() => {
    if (!focus) return
    setSelected(focus)
    const t = setTimeout(() => document.getElementById(`node-${focus}`)?.scrollIntoView({ block: 'center', behavior: 'smooth' }), 120)
    // 消费掉 focus 参数，避免返回时重复弹出
    setParams((p) => {
      p.delete('focus')
      return p
    }, { replace: true })
    return () => clearTimeout(t)
  }, [focus, setParams])

  return (
    <div className={cn(`track-${track}`)}>
      <header className="mb-5 flex flex-col gap-4 md:mb-7 md:flex-row md:items-end md:justify-between">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-[13px] font-semibold ink-track">
            <MapIcon size={15} />
            闯关地图
          </div>
          <h1 className="mt-1 text-2xl font-bold tracking-tight md:text-[28px]">{meta.title}</h1>
          <p className="mt-1 max-w-xl text-sm text-muted">{meta.desc}</p>
        </div>
        <Segmented<Track>
          value={track}
          onChange={(t) => navigate(`/track/${t}`)}
          options={TRACKS.map((t) => ({ value: t, label: TRACK_META[t].title }))}
          className="self-start md:self-auto"
        />
      </header>

      {stats.levels > 0 && (
        <div className="sticky top-[calc(env(safe-area-inset-top)+56px)] z-20 -mx-4 mb-6 border-b border-border bg-bg/85 px-4 py-2.5 backdrop-blur-md md:top-0 md:-mx-8 md:px-8">
          <div className="flex items-center gap-2 overflow-x-auto no-scrollbar">
            <TierChip active={tier === 'all'} onClick={() => setTier('all')} label="全部" count={`${stats.cleared}/${stats.levels}`} />
            {meta.tiers.map((t) => {
              const c = stats.tiers[t]
              return (
                <TierChip
                  key={t}
                  active={tier === t}
                  onClick={() => setTier(tier === t ? 'all' : t)}
                  label={TIER_LABEL[t]}
                  dot={tierDotClass(t)}
                  count={c ? `${c.cleared}/${c.total}` : '0'}
                  disabled={!c}
                />
              )
            })}
            <span className="ml-auto hidden shrink-0 items-center gap-1.5 pl-3 text-xs text-muted sm:flex">
              <Star size={13} className="fill-gold text-gold" /> {stats.stars} / {stats.maxStars}
            </span>
          </div>
        </div>
      )}

      {chapters.length === 0 ? (
        <EmptyState
          icon={MapIcon}
          title="这条路线的章节正在制作中"
          action={
            <Button variant="secondary" onClick={() => navigate(`/track/${track === 'algo' ? 'llm' : 'algo'}`)}>
              先看看{TRACK_META[track === 'algo' ? 'llm' : 'algo'].title}
            </Button>
          }
        >
          内容会按「先高频、后前沿」的顺序陆续上线。
        </EmptyState>
      ) : (
        <div className="space-y-10 md:space-y-14">
          {chapters.map((c) => (
            <ChapterBlock key={c.id} cat={cat} chapter={c} tier={tier} selected={selected} onSelect={setSelected} />
          ))}
          <div className="flex flex-col items-center gap-2 pb-4 text-center text-sm text-subtle">
            <Sparkles size={18} />
            更多章节陆续上线中
          </div>
        </div>
      )}
    </div>
  )
}

function TierChip({ active, onClick, label, count, dot, disabled }: { active: boolean; onClick: () => void; label: string; count: string; dot?: string; disabled?: boolean }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-pressed={active}
      className={cn(
        'flex h-8 shrink-0 items-center gap-1.5 rounded-full border px-3 text-[13px] font-medium transition-colors',
        active ? 'border-track bg-track-soft text-fg' : 'border-border bg-surface text-muted hover:border-border-strong hover:text-fg',
      )}
    >
      {dot && <i className={cn('h-1.5 w-1.5 rounded-full', dot)} />}
      {label}
      <span className="text-xs text-subtle tabular-nums">{count}</span>
    </button>
  )
}

/* ------------------------------------------------------------------ */
/* 章节区块                                                            */
/* ------------------------------------------------------------------ */

const STEP = 128
const NODE = 68
const BOSS = 86
const TOP = 76

function ChapterBlock({
  cat,
  chapter,
  tier,
  selected,
  onSelect,
}: {
  cat: Catalog
  chapter: ChapterDef
  tier: Tier | 'all'
  selected: string | null
  onSelect: (id: string | null) => void
}) {
  const levels = useProgress((s) => s.levels)
  const freeMode = useProgress((s) => s.settings.freeMode)
  const sts = chapterStatuses(chapter, levels, freeMode)
  const st = chapterStats(chapter, levels)
  const prereqs = pendingPrerequisites(cat, chapter, levels)
  const tierCounts = TRACK_META[chapter.track].tiers.map((t) => [t, st.tiers[t]?.total ?? 0] as const).filter(([, n]) => n > 0)
  const done = st.levels > 0 && st.cleared === st.levels

  return (
    <section id={`chapter-${chapter.id}`} aria-labelledby={`ch-title-${chapter.id}`}>
      <div className="relative overflow-hidden rounded-3xl border border-border bg-surface shadow-sm">
        <div className="absolute inset-y-0 left-0 w-1.5 bg-track" aria-hidden />
        <div className="flex flex-col gap-4 p-5 pl-6 sm:flex-row sm:items-center md:p-6 md:pl-7">
          <div className="flex min-w-0 flex-1 items-start gap-4">
            <div className="grid h-12 w-12 shrink-0 place-items-center rounded-2xl bg-track-soft text-track">
              <ChapterIcon name={chapter.icon} size={24} />
            </div>
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2 text-xs font-semibold text-subtle">
                第 {chapter.order} 章
                {done && (
                  <span className="inline-flex items-center gap-1 rounded-full bg-success-soft px-2 py-0.5 ink-success">
                    <Check size={11} strokeWidth={3} /> 已完成
                  </span>
                )}
              </div>
              <h2 id={`ch-title-${chapter.id}`} className="mt-0.5 text-lg font-bold leading-snug md:text-xl">
                {chapter.title}
              </h2>
              <p className="mt-0.5 text-[13px] text-muted">{chapter.subtitle}</p>
              <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted">
                {tierCounts.map(([t, n]) => (
                  <span key={t} className="inline-flex items-center gap-1">
                    <i className={cn('h-1.5 w-1.5 rounded-full', tierDotClass(t))} />
                    {TIER_LABEL[t]} {n}
                  </span>
                ))}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-4 sm:w-56 sm:flex-col sm:items-stretch sm:gap-2">
            <div className="flex flex-1 items-center justify-between text-xs text-muted sm:flex-none">
              <span className="tabular-nums">
                通关 <b className="text-fg">{st.cleared}</b> / {st.levels}
              </span>
              <span className="inline-flex items-center gap-1 tabular-nums">
                <Star size={12} className="fill-gold text-gold" /> {st.stars}/{st.maxStars}
              </span>
            </div>
            <Bar value={st.levels ? st.cleared / st.levels : 0} color="var(--track)" className="hidden sm:block" height={6} />
            <Link
              to={`/chapter/${chapter.id}`}
              className="inline-flex h-8 shrink-0 items-center justify-center gap-1 rounded-lg border border-border px-3 text-[13px] font-medium text-muted transition-colors hover:border-border-strong hover:text-fg"
            >
              章节详情 <ArrowRight size={14} />
            </Link>
          </div>
        </div>
        {prereqs.length > 0 && (
          <div className="flex items-start gap-2 border-t border-border bg-surface-2/50 px-6 py-2.5 text-[13px] text-muted md:px-7">
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
            </span>
          </div>
        )}
      </div>

      {sts.length === 0 ? (
        <div className="py-8 text-center text-sm text-subtle">本章关卡正在制作中</div>
      ) : (
        <PathView statuses={sts} chapter={chapter} tier={tier} selected={selected} onSelect={onSelect} />
      )}
    </section>
  )
}

/* ------------------------------------------------------------------ */
/* 路径                                                                */
/* ------------------------------------------------------------------ */

function PathView({
  statuses,
  chapter,
  tier,
  selected,
  onSelect,
}: {
  statuses: LevelStatus[]
  chapter: ChapterDef
  tier: Tier | 'all'
  selected: string | null
  onSelect: (id: string | null) => void
}) {
  const [ref, avail] = useElementWidth<HTMLDivElement>()
  const W = Math.max(300, Math.min(avail || 340, 560))
  const amp = Math.min(110, W * 0.24)
  const cx = W / 2
  const pts = statuses.map((s, i) => {
    const off = Math.sin((i * Math.PI) / 4) * amp
    return { x: cx + off, y: TOP + i * STEP, off, size: s.isBoss ? BOSS : NODE }
  })
  const last = pts[pts.length - 1]
  const H = (last?.y ?? TOP) + (last?.size ?? NODE) / 2 + 44
  const sel = statuses.findIndex((s) => s.meta.id === selected)

  return (
    <div ref={ref} className="relative mt-2">
      <div className="relative mx-auto" style={{ width: W, height: H }}>
        <svg width={W} height={H} className="absolute inset-0" aria-hidden>
          {pts.slice(0, -1).map((p, i) => {
            const q = pts[i + 1]
            const d = `M ${p.x} ${p.y} C ${p.x} ${p.y + STEP / 2}, ${q.x} ${q.y - STEP / 2}, ${q.x} ${q.y}`
            const doneSeg = statuses[i].cleared && statuses[i + 1].unlocked
            return (
              <path
                key={i}
                d={d}
                fill="none"
                strokeWidth={doneSeg ? 8 : 6}
                strokeLinecap="round"
                strokeDasharray={doneSeg ? undefined : '2 14'}
                className={doneSeg ? 'cq-path-done' : 'cq-path-todo'}
                opacity={doneSeg ? 0.45 : 1}
              />
            )
          })}
        </svg>
        {statuses.map((s, i) => (
          <PathNode
            key={s.meta.id}
            s={s}
            p={pts[i]}
            W={W}
            dim={tier !== 'all' && s.meta.tier !== tier}
            selected={sel === i}
            onClick={() => onSelect(sel === i ? null : s.meta.id)}
          />
        ))}
        {sel >= 0 && <NodePopover s={statuses[sel]} chapter={chapter} p={pts[sel]} W={W} onClose={() => onSelect(null)} />}
      </div>
    </div>
  )
}

function PathNode({
  s,
  p,
  W,
  dim,
  selected,
  onClick,
}: {
  s: LevelStatus
  p: { x: number; y: number; off: number; size: number }
  W: number
  dim: boolean
  selected: boolean
  onClick: () => void
}) {
  const levelP = useProgress((st) => st.levels[s.meta.id])
  const inProgress = !!levelP && (levelP.sectionsRead.length > 0 || levelP.quizAttempts > 0)
  // 标题放在节点的另一侧
  const labelRight = p.off < -4 || (Math.abs(p.off) <= 4 && s.index % 8 === 0)
  const space = labelRight ? W - (p.x + p.size / 2 + 14) : p.x - p.size / 2 - 14
  const labelW = Math.max(80, Math.min(200, space - 2))
  const Icon = s.state === 'locked' ? Lock : s.state === 'cleared' ? Check : s.isBoss ? Swords : s.state === 'current' ? Play : null
  return (
    <div className="absolute" style={{ left: p.x, top: p.y, width: 0, height: 0 }}>
      <button
        type="button"
        id={`node-${s.meta.id}`}
        onClick={onClick}
        data-state={s.state}
        data-dim={dim}
        data-selected={selected}
        aria-expanded={selected}
        aria-label={`${s.meta.title}，${s.state === 'locked' ? '未解锁' : s.state === 'cleared' ? `已通关 ${s.stars} 星` : '可挑战'}`}
        className="cq-node absolute -translate-x-1/2 -translate-y-1/2"
        style={{ width: p.size, height: p.size, left: 0, top: 0 }}
      >
        {s.state === 'current' && !dim && <span className="cq-node-halo" />}
        {s.state === 'current' && !dim && <span className="cq-bubble">{inProgress ? '继续' : '开始'}</span>}
        {s.isBoss && (
          <span className="absolute -top-4 left-1/2 grid h-7 w-7 -translate-x-1/2 place-items-center rounded-full border-2 border-surface bg-gold text-accent-fg shadow-sm dark:text-bg">
            <Crown size={14} className="fill-current" />
          </span>
        )}
        {Icon ? (
          <Icon size={s.isBoss ? 32 : 28} strokeWidth={s.state === 'cleared' ? 3.2 : 2.4} className={s.state === 'current' ? 'fill-current' : ''} />
        ) : (
          <span className="text-[22px] font-black tabular-nums">{s.index + 1}</span>
        )}
      </button>
      {s.cleared && (
        <span className={cn('absolute -translate-x-1/2', dim && 'opacity-30')} style={{ top: p.size / 2 + 8, left: 0 }}>
          <Stars value={s.stars} size={13} gap={1} />
        </span>
      )}
      <span
        className={cn(
          'absolute -translate-y-1/2 text-[12.5px] font-medium leading-snug',
          labelRight ? 'text-left' : 'text-right',
          s.state === 'locked' ? 'text-subtle' : 'text-fg',
          dim && 'opacity-30',
        )}
        style={{ top: 0, width: labelW, ...(labelRight ? { left: p.size / 2 + 14 } : { right: p.size / 2 + 14 }) }}
      >
        <span className="line-clamp-2">{s.meta.title}</span>
        {s.isBoss && <span className="block text-[11px] font-semibold ink-gold">BOSS 关</span>}
      </span>
    </div>
  )
}

function NodePopover({
  s,
  chapter,
  p,
  W,
  onClose,
}: {
  s: LevelStatus
  chapter: ChapterDef
  p: { x: number; y: number; size: number }
  W: number
  onClose: () => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  const levelP = useProgress((st) => st.levels[s.meta.id])
  const freeMode = useProgress((st) => st.settings.freeMode)
  const m = s.meta
  const cardW = Math.min(340, W + 24)
  const left = Math.max(-12, Math.min(W - cardW + 12, p.x - cardW / 2))
  const top = p.y + p.size / 2 + (s.cleared ? 34 : 16)
  const inProgress = !!levelP && (levelP.sectionsRead.length > 0 || levelP.quizAttempts > 0)

  useEffect(() => {
    const onDown = (e: PointerEvent) => {
      const t = e.target as HTMLElement
      if (ref.current?.contains(t) || t.closest(`#node-${CSS.escape(m.id)}`)) return
      onClose()
    }
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    document.addEventListener('pointerdown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('pointerdown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [m.id, onClose])

  // 弹出后保证卡片可见
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const r = el.getBoundingClientRect()
    const vb = window.innerHeight - (window.innerWidth < 768 ? 84 : 16)
    if (r.bottom > vb) window.scrollBy({ top: r.bottom - vb + 12, behavior: 'smooth' })
  }, [m.id])

  return (
    <div
      ref={ref}
      role="dialog"
      aria-label={m.title}
      className="absolute z-30 animate-pop rounded-2xl border border-border bg-surface p-4 shadow-lg"
      style={{ left, top, width: cardW }}
    >
      <span
        className="absolute -top-[7px] h-3 w-3 rotate-45 border-l border-t border-border bg-surface"
        style={{ left: Math.max(16, Math.min(cardW - 28, p.x - left - 6)) }}
        aria-hidden
      />
      <div className="flex items-start gap-2">
        <div className="flex min-w-0 flex-1 flex-wrap items-center gap-1.5">
          <TierBadge tier={m.tier} size="xs" />
          {m.frequency && <FreqBadge freq={m.frequency} size="xs" />}
          {s.isBoss && <BossBadge size="xs" />}
        </div>
        <button type="button" onClick={onClose} aria-label="关闭" className="-mr-1.5 -mt-1.5 grid h-7 w-7 place-items-center rounded-lg text-subtle hover:bg-surface-2 hover:text-fg">
          <X size={15} />
        </button>
      </div>
      <h3 className="mt-2 text-[16px] font-bold leading-snug">{m.title}</h3>
      <p className="mt-1 text-[13px] leading-relaxed text-muted">{m.summary}</p>
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1">
        <Minutes value={m.estMinutes} />
        <DifficultyDots value={m.difficulty} />
        {s.stars > 0 && <Stars value={s.stars} size={13} />}
      </div>
      {!!m.problems?.length && (
        <div className="mt-3">
          <div className="mb-1 text-[11px] font-semibold text-subtle">代表题</div>
          <div className="flex flex-wrap gap-1">
            {m.problems.map((x) => (
              <span key={x} className="rounded-md bg-surface-2 px-1.5 py-0.5 text-[11.5px] text-muted">
                {x}
              </span>
            ))}
          </div>
        </div>
      )}
      {m.llmLink && (
        <div className="mt-3 flex gap-2 rounded-xl bg-llm-soft/70 px-3 py-2 text-[12.5px] leading-relaxed">
          <Sparkles size={14} className="mt-0.5 shrink-0 ink-llm" />
          <span>
            <b className="font-semibold ink-llm">大模型岗场景：</b>
            <span className="text-fg/85">{m.llmLink}</span>
          </span>
        </div>
      )}
      {levelP && <StagePips progress={levelP} showLabels className="mt-3" />}
      {!s.unlocked ? (
        <div className="mt-4">
          <div className="flex items-start gap-2 rounded-xl bg-surface-2 px-3 py-2 text-[12.5px] text-muted">
            <Lock size={14} className="mt-0.5 shrink-0" />
            {s.lockReason}
          </div>
          <div className="mt-2 flex items-center justify-between gap-2">
            <Link to={`/chapter/${chapter.id}`} className="text-xs text-muted underline-offset-2 hover:text-fg hover:underline">
              查看本章
            </Link>
            {!freeMode && (
              <Link to="/profile#settings" className="text-xs font-medium ink-accent underline-offset-2 hover:underline">
                开启自由模式可直接进入
              </Link>
            )}
          </div>
        </div>
      ) : (
        <div className="mt-4 flex gap-2">
          <Link
            to={levelHref(m.id)}
            className="inline-flex h-10 flex-1 items-center justify-center gap-1.5 rounded-xl bg-track text-sm font-semibold text-accent-fg shadow-sm transition-[filter] hover:brightness-110 dark:text-bg"
          >
            {s.cleared ? <RotateCcw size={15} /> : <Play size={15} className="fill-current" />}
            {s.cleared ? '复习本关' : inProgress ? '继续' : '开始'}
          </Link>
          {s.cleared && s.stars < 3 && (
            <Link
              to={levelHref(m.id, 'challenge')}
              className="inline-flex h-10 items-center justify-center gap-1.5 rounded-xl border border-border px-3 text-sm font-medium transition-colors hover:bg-surface-2"
            >
              <Swords size={15} /> 冲三星
            </Link>
          )}
          {!s.cleared && (
            <Link
              to={`/chapter/${chapter.id}`}
              className="inline-flex h-10 items-center justify-center rounded-xl border border-border px-3 text-muted transition-colors hover:bg-surface-2 hover:text-fg"
              aria-label="章节详情"
              title="章节详情"
            >
              <BookOpen size={16} />
            </Link>
          )}
        </div>
      )}
    </div>
  )
}
