/**
 * 速查手册 /cheatsheet：按路线 / 章节分组；展开章节时懒加载该章各关的 cheatsheet；
 * 支持搜索（元信息即时匹配，速记正文在加载后匹配，可一键全文加载）；打印样式见 app.css。
 * ?chapter=<id> 自动展开并定位到该章。
 */
import { useEffect, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router'
import { ArrowUpRight, ChevronDown, ChevronsDownUp, ChevronsUpDown, FileSearch, Loader2, Printer, ScrollText, Search, X } from 'lucide-react'
import { useCatalog, type Catalog } from '@/content/catalog'
import type { ChapterDef, LevelMeta, Track } from '@/content/types'
import { TRACK_META, TRACKS, levelHref, useLevelContents } from '@/lib/selectors'
import { cn } from '@/lib/utils'
import { Button, Card, Segmented } from '@/components/ui'
import { Markdown } from '@/components/Markdown'
import { PageSpinner } from '@/components/PageSpinner'
import { ChapterIcon } from '@/components/ChapterIcon'
import { EmptyState } from '@/components/game/EmptyState'
import { TierBadge } from '@/components/game/TierBadge'

export default function Cheatsheet() {
  const cat = useCatalog()
  if (!cat) return <PageSpinner />
  return <CheatInner cat={cat} />
}

const metaText = (c: ChapterDef, m: LevelMeta) => [m.title, c.title, m.summary, ...m.tags, ...(m.problems ?? [])].join(' ').toLowerCase()

function CheatInner({ cat }: { cat: Catalog }) {
  const [params] = useSearchParams()
  const focus = params.get('chapter')
  const [track, setTrack] = useState<'all' | Track>(() => (focus ? (cat.chapterById.get(focus)?.track ?? 'all') : 'all'))
  const [q, setQ] = useState('')
  const [open, setOpen] = useState<Set<string>>(() => new Set(focus ? [focus] : []))
  const [loadAll, setLoadAll] = useState(false)
  const [printing, setPrinting] = useState(false)

  const chapters = cat.chapters.filter((c) => track === 'all' || c.track === track)
  const needIds = useMemo(
    () => chapters.filter((c) => loadAll || open.has(c.id)).flatMap((c) => c.levels.map((l) => l.id)),
    [chapters, loadAll, open],
  )
  const contents = useLevelContents(needIds)
  const allLoaded = needIds.every((id) => contents.has(id))
  const tokens = q.trim().toLowerCase().split(/\s+/).filter(Boolean)

  const matchLevel = (c: ChapterDef, m: LevelMeta) => {
    if (!tokens.length) return true
    const hay = metaText(c, m) + ' ' + (contents.get(m.id)?.cheatsheet ?? '').toLowerCase()
    return tokens.every((t) => hay.includes(t))
  }

  useEffect(() => {
    if (!focus) return
    const t = setTimeout(() => document.getElementById(`cs-${focus}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 150)
    return () => clearTimeout(t)
  }, [focus])

  // 打印：全部加载并展开后再调用 print
  useEffect(() => {
    if (!printing || !allLoaded) return
    const t = setTimeout(() => {
      window.print()
      setPrinting(false)
    }, 400)
    return () => clearTimeout(t)
  }, [printing, allLoaded])

  const toggle = (id: string) =>
    setOpen((s) => {
      const n = new Set(s)
      if (n.has(id)) n.delete(id)
      else n.add(id)
      return n
    })
  const allOpen = chapters.length > 0 && chapters.every((c) => open.has(c.id))

  const groups = TRACKS.filter((t) => track === 'all' || t === track).map((t) => ({
    track: t,
    chapters: chapters
      .filter((c) => c.track === t)
      .map((c) => ({ c, levels: c.levels.filter((m) => matchLevel(c, m)) }))
      .filter((g) => !tokens.length || g.levels.length > 0),
  }))
  const hits = groups.reduce((n, g) => n + g.chapters.reduce((k, x) => k + x.levels.length, 0), 0)

  return (
    <div className="space-y-5">
      <header className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight md:text-[28px]">速查手册</h1>
          <p className="mt-1 text-sm text-muted">全站速记卡汇总：核心公式、模板代码、关键数字与易错点。面试前过一遍。</p>
        </div>
        <div className="no-print flex gap-2">
          <Button
            variant="secondary"
            size="sm"
            icon={allOpen ? <ChevronsDownUp size={15} /> : <ChevronsUpDown size={15} />}
            onClick={() => setOpen(allOpen ? new Set() : new Set(chapters.map((c) => c.id)))}
          >
            {allOpen ? '全部收起' : '全部展开'}
          </Button>
          <Button
            variant="secondary"
            size="sm"
            icon={printing ? <Loader2 size={15} className="animate-spin" /> : <Printer size={15} />}
            disabled={printing}
            onClick={() => {
              setLoadAll(true)
              setOpen(new Set(chapters.map((c) => c.id)))
              setPrinting(true)
            }}
          >
            打印
          </Button>
        </div>
      </header>

      <div className="no-print sticky top-[calc(env(safe-area-inset-top)+56px)] z-20 -mx-4 flex flex-col gap-2 border-b border-border bg-bg/90 px-4 py-3 backdrop-blur-md sm:flex-row sm:items-center md:top-0 md:-mx-8 md:px-8">
        <label className="flex h-10 min-w-0 flex-1 items-center gap-2 rounded-xl border border-border bg-surface px-3 focus-within:border-accent">
          <Search size={16} className="shrink-0 text-subtle" />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="搜索关卡、标签、代表题或速记内容"
            className="min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-subtle"
            aria-label="搜索速查手册"
          />
          {q && (
            <button type="button" onClick={() => setQ('')} aria-label="清空搜索" className="text-subtle hover:text-fg">
              <X size={15} />
            </button>
          )}
        </label>
        <Segmented
          size="sm"
          value={track}
          onChange={setTrack}
          options={[{ value: 'all' as const, label: '全部' }, ...TRACKS.map((t) => ({ value: t, label: TRACK_META[t].title }))]}
        />
      </div>

      {tokens.length > 0 && (
        <div className="no-print flex flex-wrap items-center gap-2 text-[13px] text-muted">
          找到 {hits} 关
          {!loadAll && (
            <button type="button" onClick={() => setLoadAll(true)} className="inline-flex items-center gap-1 font-medium ink-accent hover:underline">
              <FileSearch size={14} /> 在全部速记正文中搜索
            </button>
          )}
          {loadAll && !allLoaded && (
            <span className="inline-flex items-center gap-1">
              <Loader2 size={13} className="animate-spin" /> 正在加载速记…
            </span>
          )}
        </div>
      )}

      {cat.chapters.length === 0 ? (
        <EmptyState icon={ScrollText} title="速查内容正在制作中" />
      ) : hits === 0 && tokens.length ? (
        <EmptyState icon={Search} title={`没有找到「${q}」`}>
          {loadAll ? '换个关键词试试。' : '目前只匹配了关卡标题、标签与已展开章节的速记，可以点击上方「在全部速记正文中搜索」。'}
        </EmptyState>
      ) : (
        groups.map(
          (g) =>
            g.chapters.length > 0 && (
              <section key={g.track} className={cn('space-y-3', `track-${g.track}`)}>
                <h2 className="flex items-center gap-2 text-[15px] font-semibold">
                  <i className="h-2 w-2 rounded-full bg-track" />
                  {TRACK_META[g.track].title}
                </h2>
                {g.chapters.map(({ c, levels }) => {
                  const isOpen = open.has(c.id) || tokens.length > 0
                  return (
                    <Card key={c.id} id={`cs-${c.id}`} className="print-shadow-none scroll-mt-40 overflow-hidden md:scroll-mt-24">
                      <button
                        type="button"
                        onClick={() => toggle(c.id)}
                        aria-expanded={isOpen}
                        className="flex w-full items-center gap-3 px-4 py-3.5 text-left transition-colors hover:bg-surface-2/60 md:px-5"
                      >
                        <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-track-soft text-track">
                          <ChapterIcon name={c.icon} size={18} />
                        </span>
                        <span className="min-w-0 flex-1">
                          <span className="block truncate font-semibold">{c.title}</span>
                          <span className="block truncate text-xs text-muted">
                            第 {c.order} 章 · {levels.length} 关
                          </span>
                        </span>
                        <ChevronDown size={18} className={cn('no-print shrink-0 text-subtle transition-transform', isOpen && 'rotate-180')} />
                      </button>
                      {isOpen && (
                        <div className="grid gap-2.5 border-t border-border bg-surface-2/30 p-2 sm:p-3 md:grid-cols-2 md:gap-3 md:p-4">
                          {levels.map((m) => (
                            <SheetCard key={m.id} meta={m} content={contents.get(m.id)} loading={!contents.has(m.id) && (loadAll || open.has(c.id))} />
                          ))}
                        </div>
                      )}
                    </Card>
                  )
                })}
              </section>
            ),
        )
      )}
    </div>
  )
}

function SheetCard({ meta, content, loading }: { meta: LevelMeta; content: ReturnType<typeof useLevelContents> extends Map<string, infer V> ? V | undefined : never; loading: boolean }) {
  return (
    <article className="print-break-avoid flex min-w-0 flex-col rounded-xl border border-border bg-surface p-3 sm:p-4">
      <header className="mb-2 flex items-start gap-2">
        <div className="min-w-0 flex-1">
          <h3 className="font-semibold leading-snug">{meta.title}</h3>
          <div className="mt-1 flex flex-wrap gap-1">
            <TierBadge tier={meta.tier} size="xs" />
          </div>
        </div>
        <Link to={levelHref(meta.id)} className="no-print inline-flex shrink-0 items-center gap-0.5 text-xs text-muted hover:text-fg" title="进入关卡">
          关卡 <ArrowUpRight size={13} />
        </Link>
      </header>
      {content === undefined ? (
        loading ? (
          <div className="flex items-center gap-2 py-4 text-sm text-subtle">
            <Loader2 size={14} className="animate-spin" /> 加载中…
          </div>
        ) : (
          <div className="py-2 text-sm text-subtle">展开章节以加载速记</div>
        )
      ) : content === null ? (
        <div className="py-2 text-sm ink-danger">加载失败</div>
      ) : content.cheatsheet ? (
        <Markdown className="text-[14px] [&_pre]:text-[12px]">{content.cheatsheet}</Markdown>
      ) : (
        <div className="py-2 text-sm text-subtle">本关暂无速记卡</div>
      )}
    </article>
  )
}
