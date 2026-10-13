/**
 * 复习 /review：今日待复习（题目 + 闪卡混合，逐个作答）/ 错题本（筛选、单题重做）/ 已掌握。
 * 作答结果统一调用 reviewResult(key, correct)。
 */
import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router'
import { BookOpen, CalendarClock, Check, CircleHelp, Eye, ListChecks, MessageSquareQuote, PartyPopper, Play, RotateCcw, Trash2, X } from 'lucide-react'
import { useCatalog, type Catalog } from '@/content/catalog'
import { QUESTION_TYPE_LABEL, type LevelContent } from '@/content/types'
import { useProgress } from '@/lib/store'
import { XP, daysBetween, today } from '@/lib/game'
import { findQA, findQuestion, levelHref, plainPreview, reviewBuckets, useLevelContents, type ReviewEntry } from '@/lib/selectors'
import { cn, shuffle } from '@/lib/utils'
import { Button, Card, Segmented } from '@/components/ui'
import { Markdown } from '@/components/Markdown'
import { PageSpinner } from '@/components/PageSpinner'
import { QuestionView } from '@/components/quiz/QuestionView'
import { EmptyState } from '@/components/game/EmptyState'
import { Modal } from '@/components/game/Modal'
import { TrackBadge } from '@/components/game/TierBadge'
import { Bar } from '@/components/game/ProgressRing'

type Tab = 'due' | 'wrong' | 'mastered'

const seedOf = (s: string) => {
  let h = 2166136261
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619)
  return h >>> 0
}

export default function Review() {
  const cat = useCatalog()
  if (!cat) return <PageSpinner />
  return <ReviewInner cat={cat} />
}

function ReviewInner({ cat }: { cat: Catalog }) {
  const review = useProgress((s) => s.review)
  const buckets = useMemo(() => reviewBuckets(review), [review])
  const [tab, setTab] = useState<Tab>('due')
  const [session, setSession] = useState<ReviewEntry[] | null>(null)

  if (session) return <ReviewSession cat={cat} queue={session} onExit={() => setSession(null)} />

  return (
    <div className="space-y-5">
      <header>
        <h1 className="text-2xl font-bold tracking-tight md:text-[28px]">复习</h1>
        <p className="mt-1 text-sm text-muted">答错的题与自评「模糊 / 不会」的闪卡会进入复习池，按 1 → 2 → 5 → 11 … 天的间隔回顾，连续答对 3 次视为掌握。</p>
      </header>
      <Segmented<Tab>
        value={tab}
        onChange={setTab}
        options={[
          { value: 'due', label: `今日待复习 ${buckets.due.length}` },
          { value: 'wrong', label: `错题本 ${buckets.pending.length}` },
          { value: 'mastered', label: `已掌握 ${buckets.mastered.length}` },
        ]}
      />
      {tab === 'due' && <DueTab cat={cat} due={buckets.due} onStart={(q) => setSession(q)} />}
      {tab === 'wrong' && <ListTab cat={cat} entries={buckets.pending} mode="wrong" />}
      {tab === 'mastered' && <ListTab cat={cat} entries={buckets.mastered} mode="mastered" />}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* 今日待复习                                                          */
/* ------------------------------------------------------------------ */

function DueTab({ cat, due, onStart }: { cat: Catalog; due: ReviewEntry[]; onStart: (q: ReviewEntry[]) => void }) {
  const nQ = due.filter((d) => d.kind === 'question').length
  const nQA = due.length - nQ
  if (!due.length)
    return (
      <EmptyState
        icon={PartyPopper}
        title="今天没有待复习的内容"
        action={
          <Link to="/" className="inline-flex h-10 items-center rounded-xl bg-accent px-4 text-sm font-medium text-accent-fg shadow-sm hover:bg-accent-hover">
            去闯下一关
          </Link>
        }
      >
        随堂练习答错的题、面试追问里自评「模糊 / 不会」的闪卡，会按间隔自动出现在这里。
      </EmptyState>
    )
  const start = () => {
    // 题目与闪卡交错，当天顺序固定
    const qs = shuffle(due.filter((d) => d.kind === 'question'), seedOf(today()))
    const cards = shuffle(due.filter((d) => d.kind === 'qa'), seedOf(today()) + 1)
    const out: ReviewEntry[] = []
    while (qs.length || cards.length) {
      if (qs.length) out.push(qs.shift()!)
      if (qs.length) out.push(qs.shift()!)
      if (cards.length) out.push(cards.shift()!)
    }
    onStart(out)
  }
  return (
    <div className="space-y-4">
      <Card className="flex flex-col gap-4 p-5 sm:flex-row sm:items-center md:p-6">
        <div className="grid h-14 w-14 shrink-0 place-items-center rounded-2xl bg-info-soft ink-info">
          <CalendarClock size={26} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-lg font-bold">今天有 {due.length} 项待复习</div>
          <div className="mt-0.5 text-sm text-muted">
            {nQ} 道题目 · {nQA} 张闪卡 · 预计 {Math.max(1, Math.round(nQ * 1.2 + nQA * 0.6))} 分钟 · 每答对一项 +{XP.review} XP
          </div>
        </div>
        <Button variant="primary" size="lg" icon={<Play size={16} className="fill-current" />} onClick={start}>
          开始复习
        </Button>
      </Card>
      <EntryList cat={cat} entries={due.slice(0, 12)} />
      {due.length > 12 && <div className="text-center text-xs text-subtle">还有 {due.length - 12} 项…</div>}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* 复习会话                                                            */
/* ------------------------------------------------------------------ */

function ReviewSession({ cat, queue, onExit }: { cat: Catalog; queue: ReviewEntry[]; onExit: () => void }) {
  const [idx, setIdx] = useState(0)
  const [results, setResults] = useState<Record<string, boolean>>({})
  const contents = useLevelContents(queue.map((q) => q.levelId))
  const reviewResult = useProgress((s) => s.reviewResult)
  const done = idx >= queue.length
  const correct = Object.values(results).filter(Boolean).length
  const answered = Object.keys(results).length

  if (done)
    return (
      <div className="mx-auto max-w-lg py-6">
        <Card className="p-6 text-center md:p-8">
          <div className="mx-auto grid h-16 w-16 place-items-center rounded-3xl bg-success-soft ink-success">
            <Check size={30} strokeWidth={3} />
          </div>
          <h1 className="mt-4 text-xl font-bold">本轮复习完成</h1>
          <p className="mt-1 text-sm text-muted">
            作答 {answered} 项，答对 {correct} 项{correct > 0 && `，获得 ${correct * XP.review} XP`}。答错的会在明天再次出现。
          </p>
          <div className="mt-5 grid grid-cols-3 gap-2 text-center">
            <SumStat label="作答" value={answered} />
            <SumStat label="答对" value={correct} tone="ink-success" />
            <SumStat label="待加强" value={answered - correct} tone={answered - correct ? 'ink-danger' : undefined} />
          </div>
          <Button variant="primary" size="lg" className="mt-6 w-full" onClick={onExit}>
            返回复习
          </Button>
        </Card>
      </div>
    )

  const e = queue[idx]
  const ref = cat.levelById.get(e.levelId)
  const content = contents.get(e.levelId)
  const next = () => setIdx((i) => i + 1)
  const onResult = (ok: boolean) => {
    if (e.key in results) return
    reviewResult(e.key, ok)
    setResults((r) => ({ ...r, [e.key]: ok }))
  }

  return (
    <div className="mx-auto max-w-3xl space-y-4">
      <div className="flex items-center gap-3">
        <Button variant="ghost" size="icon-sm" onClick={onExit} aria-label="退出复习">
          <X size={18} />
        </Button>
        <Bar value={idx / queue.length} className="flex-1" color="var(--info)" />
        <span className="text-sm font-medium text-muted tabular-nums">
          {idx + 1} / {queue.length}
        </span>
      </div>
      <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
        {ref && <TrackBadge track={ref.chapter.track} size="xs" />}
        <span className="truncate">{ref ? `${ref.chapter.title} · ${ref.meta.title}` : e.levelId}</span>
        <span className="text-subtle">· 已错 {e.wrongCount} 次</span>
      </div>
      <ReviewItemBody key={e.key} entry={e} content={content} onResult={onResult} onNext={next} last={idx === queue.length - 1} />
    </div>
  )
}

function SumStat({ label, value, tone }: { label: string; value: number; tone?: string }) {
  return (
    <div className="rounded-xl bg-surface-2 py-2.5">
      <div className={cn('text-xl font-bold tabular-nums', tone)}>{value}</div>
      <div className="text-xs text-muted">{label}</div>
    </div>
  )
}

/** 单个复习项：题目 → QuestionView；闪卡 → 翻面自评；内容缺失 → 可移出复习池 */
function ReviewItemBody({
  entry,
  content,
  onResult,
  onNext,
  last,
}: {
  entry: ReviewEntry
  content: LevelContent | null | undefined
  onResult: (ok: boolean) => void
  onNext: () => void
  last?: boolean
}) {
  const dismiss = useProgress((s) => s.dismissReview)
  if (content === undefined) return <PageSpinner label="正在加载题目…" />
  const q = content && entry.kind === 'question' ? findQuestion(content, entry.itemId) : undefined
  const qa = content && entry.kind === 'qa' ? findQA(content, entry.itemId) : undefined
  if (!q && !qa)
    return (
      <Card className="p-6 text-center">
        <CircleHelp size={28} className="mx-auto text-subtle" />
        <div className="mt-3 font-semibold">这道题已经下线或被修改</div>
        <p className="mt-1 text-sm text-muted">关卡内容更新后，原题可能被删除或换了编号。</p>
        <div className="mt-4 flex justify-center gap-2">
          <Button
            variant="secondary"
            icon={<Trash2 size={15} />}
            onClick={() => {
              dismiss(entry.key)
              onNext()
            }}
          >
            移出复习池
          </Button>
          <Button variant="primary" onClick={onNext}>
            跳过
          </Button>
        </div>
      </Card>
    )
  if (q)
    return (
      <QuestionView question={q} seed={seedOf(entry.key)} onAnswered={onResult} onNext={onNext} nextLabel={last ? '完成' : '下一题'} />
    )
  return <Flashcard q={qa!.q} a={qa!.a} onRate={(ok) => onResult(ok)} onNext={onNext} last={last} />
}

function Flashcard({ q, a, onRate, onNext, last }: { q: string; a: string; onRate: (ok: boolean) => void; onNext: () => void; last?: boolean }) {
  const [flipped, setFlipped] = useState(false)
  const [rated, setRated] = useState<null | 'know' | 'fuzzy' | 'dont'>(null)
  const rate = (r: 'know' | 'fuzzy' | 'dont') => {
    setRated(r)
    onRate(r === 'know')
  }
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest('input,textarea')) return
      if (!flipped && (e.key === ' ' || e.key === 'Enter')) {
        e.preventDefault()
        setFlipped(true)
      } else if (flipped && !rated && ['1', '2', '3'].includes(e.key)) rate((['know', 'fuzzy', 'dont'] as const)[Number(e.key) - 1])
      else if (rated && e.key === 'Enter') onNext()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })
  return (
    <Card className="overflow-hidden">
      <div className="flex items-center gap-2 border-b border-border px-5 py-3 text-xs font-semibold text-subtle">
        <MessageSquareQuote size={14} className="ink-accent" /> 面试追问
      </div>
      <div className="p-5 md:p-6">
        <Markdown className="text-[16px] font-medium">{q}</Markdown>
        {!flipped ? (
          <Button variant="primary" size="lg" className="mt-6 w-full" icon={<Eye size={16} />} onClick={() => setFlipped(true)}>
            先在心里回答，再翻看参考答案
          </Button>
        ) : (
          <div className="mt-5 animate-fade-in rounded-xl border border-border bg-surface-2/60 p-4">
            <div className="mb-2 text-xs font-semibold text-subtle">参考回答</div>
            <Markdown>{a}</Markdown>
          </div>
        )}
      </div>
      {flipped && (
        <div className="border-t border-border bg-surface-2/40 p-4">
          {!rated ? (
            <div className="grid grid-cols-3 gap-2">
              <Button variant="outline" className="border-success/50 ink-success" onClick={() => rate('know')}>
                会了
              </Button>
              <Button variant="outline" className="border-warning/50 ink-warning" onClick={() => rate('fuzzy')}>
                模糊
              </Button>
              <Button variant="outline" className="border-danger/50 ink-danger" onClick={() => rate('dont')}>
                不会
              </Button>
            </div>
          ) : (
            <div className="flex items-center justify-between gap-3">
              <span className="text-sm text-muted">{rated === 'know' ? '很好，间隔会拉长。' : '已记下，明天再来一次。'}</span>
              <Button variant="primary" onClick={onNext} data-autofocus>
                {last ? '完成' : '下一项'}
              </Button>
            </div>
          )}
        </div>
      )}
    </Card>
  )
}

/* ------------------------------------------------------------------ */
/* 错题本 / 已掌握                                                     */
/* ------------------------------------------------------------------ */

function ListTab({ cat, entries, mode }: { cat: Catalog; entries: ReviewEntry[]; mode: 'wrong' | 'mastered' }) {
  const [track, setTrack] = useState<'all' | 'algo' | 'llm'>('all')
  const [chapter, setChapter] = useState('all')
  const chapterIds = useMemo(() => {
    const s = new Set<string>()
    entries.forEach((e) => {
      const r = cat.levelById.get(e.levelId)
      if (r && (track === 'all' || r.chapter.track === track)) s.add(r.chapter.id)
    })
    return cat.chapters.filter((c) => s.has(c.id))
  }, [entries, cat, track])
  const filtered = entries.filter((e) => {
    const r = cat.levelById.get(e.levelId)
    if (track !== 'all' && r?.chapter.track !== track) return false
    if (chapter !== 'all' && r?.chapter.id !== chapter) return false
    return true
  })
  if (!entries.length)
    return (
      <EmptyState icon={mode === 'wrong' ? ListChecks : BookOpen} title={mode === 'wrong' ? '错题本是空的' : '还没有掌握的复习项'}>
        {mode === 'wrong' ? '答错的题会自动收录到这里。保持下去！' : '在复习中连续答对 3 次，题目就会出现在这里。'}
      </EmptyState>
    )
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <Segmented
          size="sm"
          value={track}
          onChange={(v) => {
            setTrack(v)
            setChapter('all')
          }}
          options={[
            { value: 'all', label: '全部' },
            { value: 'algo', label: '经典算法' },
            { value: 'llm', label: '大模型' },
          ]}
        />
        <select
          value={chapter}
          onChange={(e) => setChapter(e.target.value)}
          className="h-8 max-w-[60vw] rounded-xl border border-border bg-surface px-2.5 text-xs text-fg"
          aria-label="按章节筛选"
        >
          <option value="all">全部章节</option>
          {chapterIds.map((c) => (
            <option key={c.id} value={c.id}>
              {c.title}
            </option>
          ))}
        </select>
        <span className="ml-auto text-xs text-subtle">{filtered.length} 项</span>
      </div>
      <EntryList cat={cat} entries={filtered} actions mode={mode} />
    </div>
  )
}

function EntryList({ cat, entries, actions, mode }: { cat: Catalog; entries: ReviewEntry[]; actions?: boolean; mode?: 'wrong' | 'mastered' }) {
  const contents = useLevelContents(entries.map((e) => e.levelId))
  const [redo, setRedo] = useState<ReviewEntry | null>(null)
  const d0 = today()
  return (
    <>
      <ul className="divide-y divide-border overflow-hidden rounded-2xl border border-border bg-surface shadow-sm">
        {entries.map((e) => {
          const ref = cat.levelById.get(e.levelId)
          const c = contents.get(e.levelId)
          const q = c && e.kind === 'question' ? findQuestion(c, e.itemId) : undefined
          const qa = c && e.kind === 'qa' ? findQA(c, e.itemId) : undefined
          const text = q ? q.prompt : qa ? qa.q : c === undefined ? '加载中…' : '（题目已下线）'
          const dueIn = daysBetween(d0, e.due)
          return (
            <li key={e.key} className="flex items-start gap-3 px-4 py-3.5">
              <span
                className={cn('mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-lg', e.kind === 'qa' ? 'bg-accent-soft ink-accent' : 'bg-danger-soft ink-danger')}
                title={e.kind === 'qa' ? '闪卡' : '题目'}
              >
                {e.kind === 'qa' ? <MessageSquareQuote size={15} /> : <ListChecks size={15} />}
              </span>
              <div className="min-w-0 flex-1">
                <div className="line-clamp-2 text-sm"><Markdown>{text}</Markdown></div>
                <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11.5px] text-subtle">
                  <span>{e.kind === 'qa' ? '面试追问' : q ? QUESTION_TYPE_LABEL[q.type] : '题目'}</span>
                  {ref && (
                    <Link to={levelHref(e.levelId)} className="truncate hover:text-muted hover:underline">
                      {ref.meta.title}
                    </Link>
                  )}
                  <span>· 错 {e.wrongCount} 次</span>
                  {mode !== 'mastered' && <span>· {dueIn <= 0 ? '今天复习' : `${dueIn} 天后复习`}</span>}
                  {mode === 'mastered' && <span className="ink-success">· 已掌握</span>}
                </div>
              </div>
              {actions && (
                <Button variant="secondary" size="sm" icon={mode === 'mastered' ? <Eye size={14} /> : <RotateCcw size={14} />} onClick={() => setRedo(e)}>
                  {mode === 'mastered' ? '查看' : '重做'}
                </Button>
              )}
            </li>
          )
        })}
      </ul>
      <RedoModal entry={redo} content={redo ? contents.get(redo.levelId) : undefined} onClose={() => setRedo(null)} reveal={mode === 'mastered'} />
    </>
  )
}

function RedoModal({ entry, content, onClose, reveal }: { entry: ReviewEntry | null; content: LevelContent | null | undefined; onClose: () => void; reveal?: boolean }) {
  const reviewResult = useProgress((s) => s.reviewResult)
  const [doneKey, setDoneKey] = useState<string | null>(null)
  const q = entry && content && entry.kind === 'question' ? findQuestion(content, entry.itemId) : undefined
  return (
    <Modal open={!!entry} onClose={onClose} title={reveal ? '查看题目' : '重做'} size="lg">
      {entry && reveal && q ? (
        <QuestionView question={q} reveal seed={seedOf(entry.key)} />
      ) : entry ? (
        <ReviewItemBody
          key={entry.key}
          entry={entry}
          content={content}
          onResult={(ok) => {
            if (doneKey === entry.key) return
            setDoneKey(entry.key)
            if (!reveal) reviewResult(entry.key, ok)
          }}
          onNext={onClose}
          last
        />
      ) : null}
    </Modal>
  )
}
