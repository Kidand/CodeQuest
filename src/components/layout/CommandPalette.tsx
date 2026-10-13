/**
 * ⌘K / Ctrl+K 命令面板：按标题、标签、代表题、章节搜索关卡；也可跳转页面、切换主题。
 * 其它组件可调用 openCommandPalette() 打开。
 */
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { useNavigate } from 'react-router'
import { AnimatePresence, motion } from 'motion/react'
import {
  BookOpen,
  CornerDownLeft,
  House,
  Lock,
  Map as MapIcon,
  Moon,
  Repeat,
  ScrollText,
  Search,
  Timer,
  UserRound,
  type LucideIcon,
} from 'lucide-react'
import { useCatalog, type Catalog } from '@/content/catalog'
import { useProgress } from '@/lib/store'
import { starsOf } from '@/lib/game'
import { levelStatusById } from '@/lib/unlock'
import { levelHref } from '@/lib/selectors'
import { ChapterIcon } from '@/components/ChapterIcon'
import { Stars } from '@/components/game/Stars'
import { TierBadge } from '@/components/game/TierBadge'
import { cn } from '@/lib/utils'

const OPEN_EVENT = 'cq:open-palette'
export function openCommandPalette() {
  window.dispatchEvent(new Event(OPEN_EVENT))
}
export const isMac = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent)
export const modKey = isMac ? '⌘' : 'Ctrl'

interface Entry {
  id: string
  group: '页面' | '章节' | '关卡' | '操作'
  title: string
  subtitle?: string
  /** 用于匹配的文本（小写） */
  hay: string[]
  icon: ReactNode
  right?: ReactNode
  run: () => void
  boost?: number
}

function score(e: Entry, tokens: string[]): number {
  if (!tokens.length) return 1
  let total = 0
  const title = e.title.toLowerCase()
  for (const t of tokens) {
    let best = 0
    if (title.startsWith(t)) best = 12
    else if (title.includes(t)) best = 9
    e.hay.forEach((h, i) => {
      if (h.includes(t)) best = Math.max(best, i === 0 ? 7 : i === 1 ? 5 : 3)
    })
    if (!best) return 0
    total += best
  }
  return total + (e.boost ?? 0)
}

function highlight(text: string, tokens: string[]) {
  if (!tokens.length) return text
  const lower = text.toLowerCase()
  const marks: [number, number][] = []
  for (const t of tokens) {
    const i = lower.indexOf(t)
    if (i >= 0) marks.push([i, i + t.length])
  }
  if (!marks.length) return text
  marks.sort((a, b) => a[0] - b[0])
  const out: ReactNode[] = []
  let pos = 0
  marks.forEach(([s, e], k) => {
    if (s < pos) return
    out.push(text.slice(pos, s))
    out.push(
      <mark key={k} className="cq-mark">
        {text.slice(s, e)}
      </mark>,
    )
    pos = e
  })
  out.push(text.slice(pos))
  return out
}

export function CommandPalette() {
  const [open, setOpen] = useState(false)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && !e.altKey && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setOpen((o) => !o)
      }
    }
    const onOpen = () => setOpen(true)
    window.addEventListener('keydown', onKey)
    window.addEventListener(OPEN_EVENT, onOpen)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener(OPEN_EVENT, onOpen)
    }
  }, [])
  return createPortal(
    <AnimatePresence>{open && <Palette onClose={() => setOpen(false)} />}</AnimatePresence>,
    document.body,
  )
}

function Palette({ onClose }: { onClose: () => void }) {
  const cat = useCatalog()
  const navigate = useNavigate()
  const levels = useProgress((s) => s.levels)
  const freeMode = useProgress((s) => s.settings.freeMode)
  const theme = useProgress((s) => s.settings.theme)
  const updateSettings = useProgress((s) => s.updateSettings)
  const [q, setQ] = useState('')
  const [active, setActive] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    inputRef.current?.focus()
    const overflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = overflow
    }
  }, [])

  const entries = useMemo(() => {
    const go = (to: string) => () => {
      onClose()
      navigate(to)
    }
    const page = (id: string, title: string, to: string, Icon: LucideIcon, kw: string): Entry => ({
      id,
      group: '页面',
      title,
      hay: [kw.toLowerCase()],
      icon: <Icon size={17} />,
      run: go(to),
    })
    const list: Entry[] = [
      page('p-home', '首页', '/', House, 'home 首页 主页 shouye'),
      page('p-algo', '经典算法地图', '/track/algo', MapIcon, 'map algo 地图 算法 路线'),
      page('p-llm', '大模型地图', '/track/llm', MapIcon, 'map llm 地图 大模型 路线'),
      page('p-review', '复习', '/review', Repeat, 'review 复习 错题 闪卡'),
      page('p-mock', '模拟面试', '/mock', Timer, 'mock interview 模拟 面试'),
      page('p-cheat', '速查手册', '/cheatsheet', ScrollText, 'cheatsheet 速查 手册 速记'),
      page('p-profile', '我的', '/profile', UserRound, 'profile 我的 设置 成就 settings'),
      {
        id: 'a-theme',
        group: '操作',
        title: `切换主题（当前：${theme === 'light' ? '浅色' : theme === 'dark' ? '深色' : '跟随系统'}）`,
        hay: ['theme dark light 主题 深色 浅色 暗色'],
        icon: <Moon size={17} />,
        run: () => updateSettings({ theme: theme === 'light' ? 'dark' : theme === 'dark' ? 'system' : 'light' }),
      },
    ]
    if (cat) list.push(...catalogEntries(cat, levels, freeMode, go))
    return list
  }, [cat, levels, freeMode, theme, navigate, onClose, updateSettings])

  const tokens = useMemo(() => q.trim().toLowerCase().split(/\s+/).filter(Boolean), [q])
  const results = useMemo(() => {
    if (!tokens.length) {
      // 空查询：页面 + 操作 + 前几个关卡
      return [...entries.filter((e) => e.group !== '关卡' && e.group !== '章节'), ...entries.filter((e) => e.group === '关卡').slice(0, 6)]
    }
    const ranked = entries
      .map((e) => ({ e, s: score(e, tokens) }))
      .filter((x) => x.s > 0)
      .sort((a, b) => b.s - a.s)
      .slice(0, 40)
      .map((x) => x.e)
    // 按分组聚合（分组顺序取该组最高分出现的先后），避免组标题重复
    const groups = new Map<string, Entry[]>()
    for (const e of ranked) groups.set(e.group, [...(groups.get(e.group) ?? []), e])
    return [...groups.values()].flat()
  }, [entries, tokens])

  useEffect(() => setActive(0), [q])
  useEffect(() => {
    listRef.current?.querySelector(`[data-idx="${active}"]`)?.scrollIntoView({ block: 'nearest' })
  }, [active])

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActive((a) => Math.min(results.length - 1, a + 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActive((a) => Math.max(0, a - 1))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      results[active]?.run()
    } else if (e.key === 'Escape') {
      e.preventDefault()
      onClose()
    }
  }

  // 分组渲染（保持排序）
  let lastGroup = ''
  return (
    <div className="fixed inset-0 z-[85] flex items-start justify-center px-3 pt-[max(12px,env(safe-area-inset-top))] sm:pt-[12vh]" onKeyDown={onKeyDown}>
      <motion.div className="absolute inset-0 bg-bg/70 backdrop-blur-[3px]" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose} />
      <motion.div
        role="dialog"
        aria-modal="true"
        aria-label="搜索关卡与命令"
        initial={{ opacity: 0, y: -12, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: -8, scale: 0.98 }}
        transition={{ type: 'spring', stiffness: 520, damping: 36 }}
        className="relative z-10 flex max-h-[min(640px,calc(100dvh-24px))] w-full max-w-[640px] flex-col overflow-hidden rounded-2xl border border-border bg-surface shadow-lg"
      >
        <div className="flex items-center gap-3 border-b border-border px-4">
          <Search size={18} className="shrink-0 text-subtle" />
          <input
            ref={inputRef}
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="搜索关卡、标签、代表题（如 LRU、MLA、LC 146）"
            className="h-14 min-w-0 flex-1 bg-transparent text-[15px] outline-none placeholder:text-subtle"
            aria-label="搜索"
            role="combobox"
            aria-expanded="true"
            aria-controls="cq-palette-list"
            aria-activedescendant={results[active] ? `cq-pal-${results[active].id}` : undefined}
          />
          <kbd className="hidden rounded-md border border-border bg-surface-2 px-1.5 py-0.5 text-[11px] text-muted sm:inline">Esc</kbd>
          <button type="button" onClick={onClose} className="text-sm text-muted sm:hidden">
            取消
          </button>
        </div>
        <div ref={listRef} id="cq-palette-list" role="listbox" className="thin-scroll min-h-0 flex-1 overflow-y-auto p-2">
          {!cat && <div className="px-3 py-6 text-center text-sm text-muted">正在加载目录…</div>}
          {cat && results.length === 0 && (
            <div className="px-3 py-10 text-center text-sm text-muted">
              没有找到与「{q}」相关的内容
              <div className="mt-1 text-xs text-subtle">试试关卡名、知识点标签（如「二分」「KV cache」）或题号（如「146」）</div>
            </div>
          )}
          {results.map((e, i) => {
            const header = e.group !== lastGroup ? e.group : null
            lastGroup = e.group
            return (
              <div key={e.id}>
                {header && <div className="px-3 pb-1 pt-2.5 text-[11px] font-semibold tracking-wide text-subtle">{header}</div>}
                <button
                  type="button"
                  id={`cq-pal-${e.id}`}
                  role="option"
                  aria-selected={i === active}
                  data-idx={i}
                  onMouseMove={() => i !== active && setActive(i)}
                  onClick={() => e.run()}
                  className={cn(
                    'flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition-colors',
                    i === active ? 'bg-accent-soft' : 'hover:bg-surface-2',
                  )}
                >
                  <span className={cn('grid h-8 w-8 shrink-0 place-items-center rounded-lg', i === active ? 'bg-surface text-accent' : 'bg-surface-2 text-muted')}>
                    {e.icon}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-medium">{highlight(e.title, tokens)}</span>
                    {e.subtitle && <span className="block truncate text-xs text-muted">{highlight(e.subtitle, tokens)}</span>}
                  </span>
                  {e.right}
                  {i === active && <CornerDownLeft size={15} className="hidden shrink-0 text-subtle sm:block" />}
                </button>
              </div>
            )
          })}
        </div>
        <div className="hidden items-center gap-4 border-t border-border bg-surface-2/50 px-4 py-2 text-[11px] text-subtle sm:flex">
          <span>↑ ↓ 选择</span>
          <span>Enter 打开</span>
          <span>Esc 关闭</span>
          <span className="ml-auto">{modKey} K 随时唤起</span>
        </div>
      </motion.div>
    </div>
  )
}

function catalogEntries(
  cat: Catalog,
  levels: ReturnType<typeof useProgress.getState>['levels'],
  freeMode: boolean,
  go: (to: string) => () => void,
): Entry[] {
  const out: Entry[] = []
  for (const ch of cat.chapters) {
    out.push({
      id: `c-${ch.id}`,
      group: '章节',
      title: ch.title,
      subtitle: `${ch.track === 'algo' ? '经典算法' : '大模型'} · 第 ${ch.order} 章 · ${ch.subtitle}`,
      hay: [ch.subtitle.toLowerCase(), ch.id.toLowerCase()],
      icon: <ChapterIcon name={ch.icon} size={17} />,
      run: go(`/chapter/${ch.id}`),
    })
    for (const m of ch.levels) {
      const st = levelStatusById(cat, m.id, levels, freeMode)
      const locked = st ? !st.unlocked : false
      out.push({
        id: `l-${m.id}`,
        group: '关卡',
        title: m.title,
        subtitle: [ch.title, ...(m.problems ?? []).slice(0, 3)].join(' · '),
        hay: [
          [...m.tags, ...(m.problems ?? [])].join(' ').toLowerCase(),
          `${ch.title} ${m.id}`.toLowerCase(),
          `${m.summary} ${m.llmLink ?? ''}`.toLowerCase(),
        ],
        icon: locked ? <Lock size={15} /> : <BookOpen size={16} />,
        right: (
          <span className="hidden shrink-0 items-center gap-2 sm:flex">
            <TierBadge tier={m.tier} size="xs" />
            {starsOf(levels[m.id]) > 0 && <Stars value={starsOf(levels[m.id])} size={11} />}
          </span>
        ),
        // 锁定关卡跳到地图并弹出节点卡（显示解锁条件）
        run: locked ? go(`/track/${ch.track}?focus=${m.id}`) : go(levelHref(m.id)),
        boost: st?.current ? 1 : 0,
      })
    }
  }
  return out
}
