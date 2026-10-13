/** 首页：问候 / 继续学习 / 今日目标 / 两条路线 / 推荐关卡 / 分层说明 */
import { Link } from 'react-router'
import { ArrowRight, Flame, Map as MapIcon, Play, Repeat, ScrollText, Sparkles, Target, Timer } from 'lucide-react'
import { useCatalog, type Catalog } from '@/content/catalog'
import { STAGES, TIER_LABEL, type Track } from '@/content/types'
import { useProgress } from '@/lib/store'
import { levelFromXp, starsOf, titleForLevel, today } from '@/lib/game'
import {
  STAGE_LABEL,
  TIER_DESC,
  TRACK_META,
  TRACKS,
  greeting,
  levelHref,
  recommendLevels,
  trackStats,
  useDueCount,
  type Recommendation,
} from '@/lib/selectors'
import { chapterStatuses } from '@/lib/unlock'
import { Card } from '@/components/ui'
import { PageSpinner } from '@/components/PageSpinner'
import { ChapterIcon } from '@/components/ChapterIcon'
import { Stars } from '@/components/game/Stars'
import { BossBadge, FreqBadge, TierBadge, tierDotClass } from '@/components/game/TierBadge'
import { Bar, ProgressRing } from '@/components/game/ProgressRing'
import { Minutes, StagePips } from '@/components/game/LevelBits'
import { cn } from '@/lib/utils'

export default function Home() {
  const cat = useCatalog()
  if (!cat) return <PageSpinner />
  return <HomeInner cat={cat} />
}

function HomeInner({ cat }: { cat: Catalog }) {
  const xp = useProgress((s) => s.xp)
  const levels = useProgress((s) => s.levels)
  const freeMode = useProgress((s) => s.settings.freeMode)
  const lastVisited = useProgress((s) => s.lastVisited)
  const lv = levelFromXp(xp)
  const lastRef = lastVisited ? cat.levelById.get(lastVisited.levelId) : undefined
  const recs = recommendLevels(cat, levels, freeMode, 4, lastRef?.meta.id)
  const now = new Date()

  return (
    <div className="space-y-6 md:space-y-8">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <div className="text-[13px] text-muted">
            {now.getMonth() + 1} 月 {now.getDate()} 日 · 星期{'日一二三四五六'[now.getDay()]}
          </div>
          <h1 className="mt-1 text-2xl font-bold tracking-tight md:text-[28px]">
            {greeting(now)}，{titleForLevel(lv.level)}
          </h1>
          <p className="mt-1 text-sm text-muted">看懂 → 练熟 → 手撕 → 讲清。今天也往前推进一关。</p>
        </div>
      </header>

      <div className="grid gap-4 lg:grid-cols-3">
        <ContinueCard cat={cat} className="lg:col-span-2" />
        <TodayCard />
      </div>

      <section>
        <SectionTitle title="学习路线" />
        <div className="grid gap-4 md:grid-cols-2">
          {TRACKS.map((t) => (
            <TrackCard key={t} cat={cat} track={t} />
          ))}
        </div>
      </section>

      <section>
        <SectionTitle title="推荐关卡" hint="优先未完成的必考 / 新必会" />
        {recs.length ? (
          <div className="grid gap-3 md:grid-cols-2">
            {recs.map((r) => (
              <RecCard key={r.status.meta.id} r={r} />
            ))}
          </div>
        ) : (
          <Card className="p-5 text-sm text-muted">
            {cat.chapters.length ? '已解锁的关卡都通关了。去地图上挑战新的章节，或开一场模拟面试检验成果。' : '关卡内容正在制作中，稍后再来看看。'}
          </Card>
        )}
      </section>

      <section className="grid gap-3 sm:grid-cols-3">
        <QuickLink to="/mock" icon={Timer} title="模拟面试" desc="限时手撕 + 八股，出报告" tone="accent" />
        <QuickLink to="/review" icon={Repeat} title="间隔复习" desc="错题与闪卡按遗忘曲线回顾" tone="info" />
        <QuickLink to="/cheatsheet" icon={ScrollText} title="速查手册" desc="全站速记卡，可搜索、可打印" tone="success" />
      </section>

      <section>
        <SectionTitle title="分层说明" hint="关卡徽章的含义" />
        <div className="grid gap-4 md:grid-cols-2">
          {TRACKS.map((t) => (
            <Card key={t} className={cn('p-4 md:p-5', `track-${t}`)}>
              <div className="mb-3 flex items-center gap-2 text-sm font-semibold">
                <i className="h-2 w-2 rounded-full bg-track" />
                {TRACK_META[t].title}
                <span className="font-normal text-subtle">· {t === 'algo' ? '按面试频率分级' : '按知识代际分层'}</span>
              </div>
              <ul className="space-y-2.5">
                {TRACK_META[t].tiers.map((tier) => (
                  <li key={tier} className="flex items-start gap-3">
                    <TierBadge tier={tier} className="mt-px w-[84px] justify-center" />
                    <span className="text-[13.5px] leading-relaxed text-muted">{TIER_DESC[tier]}</span>
                  </li>
                ))}
              </ul>
            </Card>
          ))}
        </div>
      </section>
    </div>
  )
}

function SectionTitle({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="mb-3 flex items-baseline gap-2">
      <h2 className="text-[17px] font-semibold">{title}</h2>
      {hint && <span className="text-xs text-subtle">{hint}</span>}
    </div>
  )
}

function ContinueCard({ cat, className }: { cat: Catalog; className?: string }) {
  const lastVisited = useProgress((s) => s.lastVisited)
  const levels = useProgress((s) => s.levels)
  const freeMode = useProgress((s) => s.settings.freeMode)
  const ref = lastVisited ? cat.levelById.get(lastVisited.levelId) : undefined
  // 没有上次记录：取第一条推荐（通常是第一章第一关）
  const fallback = !ref ? recommendLevels(cat, levels, freeMode, 1)[0] : undefined
  const chapter = ref?.chapter ?? fallback?.chapter
  const meta = ref?.meta ?? fallback?.status.meta
  if (!chapter || !meta) {
    return (
      <Card className={cn('flex flex-col justify-center p-6', className)}>
        <div className="text-lg font-semibold">欢迎来到 CodeQuest</div>
        <p className="mt-1 text-sm text-muted">关卡内容正在陆续上线，先去个人中心看看设置吧。</p>
      </Card>
    )
  }
  const p = levels[meta.id]
  const stars = starsOf(p)
  const stage = ref && lastVisited && (STAGES as string[]).includes(lastVisited.stage) ? lastVisited.stage : undefined
  const st = chapterStatuses(chapter, levels, freeMode)[cat.levelById.get(meta.id)?.index ?? 0]
  return (
    <Card className={cn('relative overflow-hidden p-5 md:p-6', `track-${chapter.track}`, className)}>
      <div className="pointer-events-none absolute -right-10 -top-10 h-44 w-44 rounded-full bg-track-soft opacity-80" aria-hidden />
      <div className="relative flex flex-col gap-4 sm:flex-row sm:items-center">
        <div className="grid h-14 w-14 shrink-0 place-items-center rounded-2xl bg-track text-accent-fg shadow-sm dark:text-bg">
          <ChapterIcon name={chapter.icon} size={26} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-xs font-semibold ink-track">{ref ? '继续学习' : '从这里开始'}</div>
          <div className="mt-0.5 truncate text-lg font-bold md:text-xl">{meta.title}</div>
          <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-[13px] text-muted">
            <span className="truncate">{chapter.title}</span>
            {stage && <span>· 上次停在「{STAGE_LABEL[stage as keyof typeof STAGE_LABEL]}」</span>}
            {stars > 0 && <Stars value={stars} size={13} />}
          </div>
        </div>
        <Link
          to={st && !st.unlocked ? `/track/${chapter.track}?focus=${meta.id}` : levelHref(meta.id, stage)}
          className="inline-flex h-11 shrink-0 items-center justify-center gap-2 rounded-xl bg-track px-5 text-sm font-semibold text-accent-fg shadow-sm transition-[filter] hover:brightness-110 dark:text-bg"
        >
          <Play size={16} className="fill-current" />
          {ref ? '继续' : '开始第一关'}
        </Link>
      </div>
      <StagePips progress={p} showLabels className="relative mt-5 max-w-md" />
    </Card>
  )
}

function TodayCard() {
  const daily = useProgress((s) => s.daily)
  const goal = useProgress((s) => s.settings.dailyGoal)
  const streak = useProgress((s) => s.streak)
  const due = useDueCount()
  const todayXp = daily[today()]?.xp ?? 0
  const pct = Math.min(1, todayXp / Math.max(1, goal))
  const doneToday = streak.lastDate === today()
  return (
    <Card className="p-5">
      <div className="flex items-center gap-4">
        <ProgressRing value={pct} size={84} stroke={9} color="var(--gold)" label={`今日 XP ${todayXp}/${goal}`}>
          <div className="text-center leading-none">
            <div className="text-lg font-bold tabular-nums">{todayXp}</div>
            <div className="mt-1 text-[10px] text-subtle tabular-nums">/ {goal}</div>
          </div>
        </ProgressRing>
        <div className="min-w-0">
          <div className="flex items-center gap-1.5 text-sm font-semibold">
            <Target size={15} className="ink-gold" />
            今日目标
          </div>
          <div className="mt-1 text-[13px] leading-relaxed text-muted">
            {pct >= 1 ? '今日目标已达成，状态不错！' : `再拿 ${goal - todayXp} XP 达成今日目标`}
          </div>
        </div>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-2">
        <div className="rounded-xl bg-surface-2 p-3">
          <div className={cn('flex items-center gap-1 text-lg font-bold tabular-nums', streak.current > 0 ? 'ink-warning' : 'text-subtle')}>
            <Flame size={18} className={streak.current > 0 ? 'fill-current' : ''} />
            {streak.current}
            <span className="text-xs font-medium text-muted">天</span>
          </div>
          <div className="mt-0.5 text-[11.5px] text-muted">{doneToday ? '今日已打卡' : streak.current > 0 ? '今天还没学习' : '连续学习'} · 最长 {streak.best}</div>
        </div>
        <Link to="/review" className="group rounded-xl bg-surface-2 p-3 transition-colors hover:bg-surface-3">
          <div className={cn('flex items-center gap-1 text-lg font-bold tabular-nums', due > 0 ? 'ink-info' : 'text-subtle')}>
            <Repeat size={17} />
            {due}
            <span className="text-xs font-medium text-muted">项</span>
          </div>
          <div className="mt-0.5 flex items-center gap-1 text-[11.5px] text-muted">
            {due > 0 ? '今日待复习' : '暂无待复习'}
            <ArrowRight size={12} className="transition-transform group-hover:translate-x-0.5" />
          </div>
        </Link>
      </div>
    </Card>
  )
}

function TrackCard({ cat, track }: { cat: Catalog; track: Track }) {
  const levels = useProgress((s) => s.levels)
  const meta = TRACK_META[track]
  const st = trackStats(cat, track, levels)
  const ratio = st.levels ? st.cleared / st.levels : 0
  return (
    <Link to={`/track/${track}`} className={cn('group block', `track-${track}`)}>
      <Card className="h-full overflow-hidden transition-[border-color,box-shadow] group-hover:border-border-strong group-hover:shadow-md">
        <div className="flex items-start gap-4 bg-track-soft/70 p-5">
          <div className="grid h-12 w-12 shrink-0 place-items-center rounded-2xl bg-track text-accent-fg shadow-sm dark:text-bg">
            {track === 'algo' ? <MapIcon size={22} /> : <Sparkles size={22} />}
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <h3 className="text-lg font-bold">{meta.title}</h3>
              <ArrowRight size={16} className="text-track transition-transform group-hover:translate-x-1" />
            </div>
            <p className="mt-0.5 text-[13px] leading-relaxed text-muted">{meta.desc}</p>
          </div>
        </div>
        <div className="space-y-4 p-5 pt-4">
          {st.levels === 0 ? (
            <div className="text-sm text-muted">本路线的关卡正在制作中，敬请期待。</div>
          ) : (
            <>
              <div className="grid grid-cols-3 gap-2 text-center">
                <Stat value={`${st.cleared}/${st.levels}`} label="已通关" />
                <Stat value={`${st.stars}`} label={`星（满 ${st.maxStars}）`} />
                <Stat value={`${st.chapters}`} label="章节" />
              </div>
              <Bar value={ratio} color={`var(--${track})`} />
              <div className="space-y-1.5">
                {meta.tiers.map((tier) => {
                  const c = st.tiers[tier]
                  return (
                    <div key={tier} className="flex items-center gap-2 text-xs">
                      <i className={cn('h-1.5 w-1.5 shrink-0 rounded-full', tierDotClass(tier))} />
                      <span className="w-14 shrink-0 text-muted">{TIER_LABEL[tier]}</span>
                      <Bar value={c ? c.cleared / c.total : 0} height={5} color={`var(--${track})`} className="flex-1" />
                      <span className="w-10 shrink-0 text-right text-muted tabular-nums">{c ? `${c.cleared}/${c.total}` : '—'}</span>
                    </div>
                  )
                })}
              </div>
            </>
          )}
        </div>
      </Card>
    </Link>
  )
}

function Stat({ value, label }: { value: string; label: string }) {
  return (
    <div className="rounded-xl bg-surface-2 px-2 py-2">
      <div className="text-[17px] font-bold tabular-nums">{value}</div>
      <div className="text-[11px] text-muted">{label}</div>
    </div>
  )
}

function RecCard({ r }: { r: Recommendation }) {
  const m = r.status.meta
  return (
    <Link to={levelHref(m.id)} className={cn('group block', `track-${r.chapter.track}`)}>
      <Card className="flex h-full gap-3.5 p-4 transition-[border-color,box-shadow] group-hover:border-border-strong group-hover:shadow-md">
        <div className="grid h-11 w-11 shrink-0 place-items-center rounded-xl bg-track-soft text-track">
          <ChapterIcon name={r.chapter.icon} size={20} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-1.5">
            <TierBadge tier={m.tier} size="xs" />
            {m.frequency && <FreqBadge freq={m.frequency} size="xs" />}
            {r.status.isBoss && <BossBadge size="xs" />}
            {r.inProgress && <span className="text-[11px] font-medium ink-accent">进行中</span>}
          </div>
          <div className="mt-1.5 truncate font-semibold">{m.title}</div>
          <p className="mt-0.5 line-clamp-2 text-[13px] leading-relaxed text-muted">{m.summary}</p>
          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="truncate text-xs text-subtle">{r.chapter.title}</span>
            <Minutes value={m.estMinutes} />
          </div>
          {!!m.problems?.length && (
            <div className="mt-2 flex flex-wrap gap-1">
              {m.problems.slice(0, 3).map((p) => (
                <span key={p} className="rounded-md bg-surface-2 px-1.5 py-0.5 text-[11px] text-muted">
                  {p}
                </span>
              ))}
            </div>
          )}
        </div>
      </Card>
    </Link>
  )
}

function QuickLink({ to, icon: Icon, title, desc, tone }: { to: string; icon: typeof Timer; title: string; desc: string; tone: 'accent' | 'info' | 'success' }) {
  const cls = { accent: 'bg-accent-soft ink-accent', info: 'bg-info-soft ink-info', success: 'bg-success-soft ink-success' }[tone]
  return (
    <Link to={to} className="group flex items-center gap-3 rounded-2xl border border-border bg-surface p-4 shadow-sm transition-colors hover:border-border-strong">
      <span className={cn('grid h-10 w-10 shrink-0 place-items-center rounded-xl', cls)}>
        <Icon size={19} />
      </span>
      <span className="min-w-0 flex-1">
        <span className="block text-sm font-semibold">{title}</span>
        <span className="block truncate text-xs text-muted">{desc}</span>
      </span>
      <ArrowRight size={16} className="shrink-0 text-subtle transition-transform group-hover:translate-x-0.5" />
    </Link>
  )
}
