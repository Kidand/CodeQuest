/**
 * 全站布局（P1）：
 *  - 手机（<768）：顶部精简栏（logo / 连续天数 / 等级 XP / 搜索）+ 底部 Tab（首页 / 地图 / 复习 / 我的）
 *  - 平板（768–1023）：图标侧栏
 *  - 桌面（≥1024）：展开侧栏（logo、导航、复习角标、迷你等级卡、主题切换）
 * 游戏事件浮层与 ⌘K 面板挂在 Root（App.tsx），关卡页同样可用。
 */
import { useEffect } from 'react'
import { Link, NavLink, Outlet, useLocation } from 'react-router'
import { Flame, House, Map as MapIcon, Repeat, ScrollText, Search, Timer, UserRound, type LucideIcon } from 'lucide-react'
import '@/styles/app.css'
import { useProgress } from '@/lib/store'
import { levelFromXp, titleForLevel } from '@/lib/game'
import { recallTrack, useDueCount } from '@/lib/selectors'
import { applyTheme } from '@/lib/theme'
import { ThemeToggle } from '@/components/ui'
import { Bar } from '@/components/game/ProgressRing'
import { openCommandPalette, modKey } from './CommandPalette'
import { cn } from '@/lib/utils'

interface NavDef {
  to: string
  label: string
  icon: LucideIcon
  match: (path: string) => boolean
  badge?: 'review'
  track?: 'algo' | 'llm'
}

const SIDE_NAV: NavDef[] = [
  { to: '/', label: '首页', icon: House, match: (p) => p === '/' },
  { to: '/track/algo', label: '经典算法', icon: MapIcon, match: (p) => p.startsWith('/track/algo'), track: 'algo' },
  { to: '/track/llm', label: '大模型', icon: MapIcon, match: (p) => p.startsWith('/track/llm'), track: 'llm' },
  { to: '/review', label: '复习', icon: Repeat, match: (p) => p.startsWith('/review'), badge: 'review' },
  { to: '/mock', label: '模拟面试', icon: Timer, match: (p) => p.startsWith('/mock') },
  { to: '/cheatsheet', label: '速查手册', icon: ScrollText, match: (p) => p.startsWith('/cheatsheet') },
  { to: '/profile', label: '我的', icon: UserRound, match: (p) => p.startsWith('/profile') },
]

export function Logo({ compact }: { compact?: boolean }) {
  return (
    <Link to="/" className="flex items-center gap-2.5 rounded-xl" aria-label="CodeQuest 首页">
      <span className="grid h-9 w-9 shrink-0 place-items-center rounded-[11px] bg-gradient-to-br from-accent to-algo text-accent-fg shadow-sm">
        <svg viewBox="0 0 64 64" className="h-5 w-5" aria-hidden>
          <path d="M24 20l-11 12 11 12M40 20l11 12-11 12" stroke="currentColor" strokeWidth="6" fill="none" strokeLinecap="round" strokeLinejoin="round" />
          <circle cx="32" cy="32" r="4" fill="currentColor" />
        </svg>
      </span>
      {!compact && (
        <span className="leading-none">
          <span className="block text-[16px] font-extrabold tracking-tight">CodeQuest</span>
          <span className="mt-0.5 block text-[10.5px] font-medium text-subtle">大模型算法面试闯关</span>
        </span>
      )}
    </Link>
  )
}

function DueBadge({ n, dot }: { n: number; dot?: boolean }) {
  if (n <= 0) return null
  if (dot) return <span className="absolute right-1.5 top-1.5 h-2.5 w-2.5 rounded-full border-2 border-surface bg-danger" aria-label={`${n} 项待复习`} />
  return (
    <span className="ml-auto min-w-5 rounded-full bg-danger px-1.5 text-center text-[11px] font-bold leading-5 text-accent-fg tabular-nums">
      {n > 99 ? '99+' : n}
    </span>
  )
}

function Sidebar() {
  const { pathname } = useLocation()
  const due = useDueCount()
  const xp = useProgress((s) => s.xp)
  const streak = useProgress((s) => s.streak)
  const lv = levelFromXp(xp)
  return (
    <aside className="app-chrome sticky top-0 hidden h-dvh shrink-0 flex-col border-r border-border bg-surface md:flex md:w-[76px] lg:w-[var(--app-sidebar)]">
      <div className="flex h-16 items-center px-5 max-lg:justify-center max-lg:px-0">
        <span className="lg:hidden">
          <Logo compact />
        </span>
        <span className="hidden lg:block">
          <Logo />
        </span>
      </div>
      <button
        type="button"
        onClick={openCommandPalette}
        className="mx-3 mb-3 flex h-10 items-center gap-2 rounded-xl border border-border bg-surface-2 px-3 text-sm text-subtle transition-colors hover:border-border-strong hover:text-muted max-lg:justify-center max-lg:px-0"
        title={`搜索（${modKey} K）`}
        aria-label="搜索关卡"
      >
        <Search size={16} className="shrink-0" />
        <span className="hidden flex-1 text-left lg:inline">搜索关卡…</span>
        <kbd className="hidden rounded-md border border-border bg-surface px-1.5 text-[11px] font-medium text-muted lg:inline">{modKey} K</kbd>
      </button>
      <nav className="flex flex-1 flex-col gap-1 overflow-y-auto px-3 no-scrollbar" aria-label="主导航">
        {SIDE_NAV.map((n) => {
          const active = n.match(pathname)
          const Icon = n.icon
          return (
            <NavLink
              key={n.to}
              to={n.to}
              title={n.label}
              className={cn(
                'relative flex h-11 items-center gap-3 rounded-xl px-3 text-[14px] font-medium transition-colors max-lg:justify-center max-lg:px-0',
                active ? 'bg-accent-soft text-fg' : 'text-muted hover:bg-surface-2 hover:text-fg',
                n.track && `track-${n.track}`,
              )}
            >
              <Icon
                size={19}
                className={cn('shrink-0', active && (n.track ? 'text-track' : 'text-accent'))}
                strokeWidth={active ? 2.3 : 2}
              />
              <span className="hidden lg:inline">{n.label}</span>
              {n.track && <i className={cn('hidden h-1.5 w-1.5 rounded-full lg:block', n.track === 'algo' ? 'bg-algo' : 'bg-llm')} />}
              {n.badge === 'review' && (
                <>
                  <span className="hidden lg:contents">
                    <DueBadge n={due} />
                  </span>
                  <span className="lg:hidden">
                    <DueBadge n={due} dot />
                  </span>
                </>
              )}
            </NavLink>
          )
        })}
      </nav>
      <div className="border-t border-border p-3">
        <Link
          to="/profile"
          className="hidden rounded-2xl border border-border bg-surface-2/60 p-3 transition-colors hover:border-border-strong lg:block"
        >
          <div className="flex items-center gap-2.5">
            <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-gradient-to-br from-accent to-llm text-sm font-black text-accent-fg tabular-nums">
              {lv.level}
            </span>
            <div className="min-w-0 flex-1">
              <div className="truncate text-[13px] font-semibold">{titleForLevel(lv.level)}</div>
              <div className="text-[11px] text-muted tabular-nums">
                {lv.intoLevel} / {lv.levelSpan} XP
              </div>
            </div>
            <span className={cn('flex items-center gap-0.5 text-[13px] font-bold tabular-nums', streak.current > 0 ? 'ink-warning' : 'text-subtle')} title="连续学习天数">
              <Flame size={15} className={streak.current > 0 ? 'fill-current' : ''} />
              {streak.current}
            </span>
          </div>
          <Bar value={lv.progress} className="mt-2.5" height={6} color="var(--gold)" />
        </Link>
        <div className="mt-2 flex items-center justify-between max-lg:justify-center">
          <ThemeToggle showLabel className="max-lg:hidden" />
          <ThemeToggle className="lg:hidden" />
          <Link to="/profile" className="hidden text-xs text-subtle hover:text-muted lg:inline">
            设置
          </Link>
        </div>
      </div>
    </aside>
  )
}

function MobileTopBar() {
  const xp = useProgress((s) => s.xp)
  const streak = useProgress((s) => s.streak)
  const lv = levelFromXp(xp)
  return (
    <header className="app-chrome sticky top-0 z-40 border-b border-border bg-surface/85 pt-safe backdrop-blur-md md:hidden">
      <div className="flex h-14 items-center gap-2 px-4">
        <Logo />
        <div className="ml-auto flex items-center gap-1.5">
          <span
            className={cn(
              'flex h-8 items-center gap-1 rounded-full border border-border px-2.5 text-[13px] font-bold tabular-nums',
              streak.current > 0 ? 'ink-warning' : 'text-subtle',
            )}
            title="连续学习天数"
          >
            <Flame size={15} className={streak.current > 0 ? 'fill-current' : ''} />
            {streak.current}
          </span>
          <Link to="/profile" className="flex h-8 items-center gap-1.5 rounded-full border border-border pl-1 pr-2.5" title={`${xp} XP`}>
            <span className="grid h-6 w-6 place-items-center rounded-full bg-gradient-to-br from-accent to-llm text-[11px] font-black text-accent-fg">{lv.level}</span>
            <span className="text-[12px] font-semibold tabular-nums ink-gold">{xp} XP</span>
          </Link>
          <button type="button" onClick={openCommandPalette} aria-label="搜索关卡" className="grid h-9 w-9 place-items-center rounded-full text-muted hover:bg-surface-2">
            <Search size={19} />
          </button>
        </div>
      </div>
    </header>
  )
}

function MobileTabBar() {
  const { pathname } = useLocation()
  const due = useDueCount()
  const tabs: NavDef[] = [
    { to: '/', label: '首页', icon: House, match: (p) => p === '/' },
    { to: `/track/${recallTrack()}`, label: '地图', icon: MapIcon, match: (p) => p.startsWith('/track') || p.startsWith('/chapter') },
    { to: '/review', label: '复习', icon: Repeat, match: (p) => p.startsWith('/review'), badge: 'review' },
    {
      to: '/profile',
      label: '我的',
      icon: UserRound,
      match: (p) => p.startsWith('/profile') || p.startsWith('/mock') || p.startsWith('/cheatsheet'),
    },
  ]
  return (
    <nav
      className="app-chrome fixed inset-x-0 bottom-0 z-40 border-t border-border bg-surface/90 pb-safe backdrop-blur-md md:hidden"
      aria-label="底部导航"
    >
      <div className="grid h-16 grid-cols-4">
        {tabs.map((t) => {
          const active = t.match(pathname)
          const Icon = t.icon
          return (
            <NavLink
              key={t.label}
              to={t.to}
              className={cn('relative flex flex-col items-center justify-center gap-1 text-[11px] font-medium', active ? 'text-accent' : 'text-subtle')}
            >
              <span className={cn('relative grid h-7 w-12 place-items-center rounded-full transition-colors', active && 'bg-accent-soft')}>
                <Icon size={20} strokeWidth={active ? 2.4 : 2} />
                {t.badge === 'review' && due > 0 && (
                  <span className="absolute -right-1 -top-1 min-w-[18px] rounded-full border-2 border-surface bg-danger px-1 text-center text-[10px] font-bold leading-[14px] text-accent-fg tabular-nums">
                    {due > 99 ? '99+' : due}
                  </span>
                )}
              </span>
              {t.label}
            </NavLink>
          )
        })}
      </div>
    </nav>
  )
}

/** 打印时临时切换到浅色主题 */
function usePrintLight() {
  useEffect(() => {
    const before = () => document.documentElement.classList.remove('dark')
    const after = () => applyTheme(useProgress.getState().settings.theme)
    window.addEventListener('beforeprint', before)
    window.addEventListener('afterprint', after)
    return () => {
      window.removeEventListener('beforeprint', before)
      window.removeEventListener('afterprint', after)
    }
  }, [])
}

export default function AppLayout() {
  usePrintLight()
  return (
    <div className="app-shell flex min-h-dvh">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <MobileTopBar />
        <main className="app-main-pad print-full mx-auto w-full max-w-[1100px] flex-1 px-4 pt-5 md:px-8 md:pt-8">
          <Outlet />
        </main>
      </div>
      <MobileTabBar />
    </div>
  )
}
