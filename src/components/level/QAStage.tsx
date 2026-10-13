/** ⑤ 面试追问：闪卡翻面 → 自评（会 / 模糊 / 不会）→ rateQA（模糊 / 不会进入间隔复习） */
import { useEffect, useState } from 'react'
import { Link } from 'react-router'
import { AnimatePresence, motion } from 'motion/react'
import { ArrowLeft, ArrowRight, Check, Eye, HelpCircle, List, Layers, RotateCcw, Star, Trophy, X } from 'lucide-react'
import { Markdown } from '@/components/Markdown'
import { Button } from '@/components/ui'
import { Kbd } from '@/components/quiz/marks'
import { useProgress } from '@/lib/store'
import { cn } from '@/lib/utils'
import { useLevel } from './context'
import { EmptyStage } from './QuizStage'

type Rating = 'know' | 'fuzzy' | 'dont'
const RATINGS: { r: Rating; label: string; icon: typeof Check; cls: string; key: string }[] = [
  { r: 'dont', label: '不会', icon: X, cls: 'border-danger/40 bg-danger-soft text-danger', key: '1' },
  { r: 'fuzzy', label: '模糊', icon: HelpCircle, cls: 'border-warning/40 bg-warning-soft text-warning', key: '2' },
  { r: 'know', label: '会', icon: Check, cls: 'border-success/40 bg-success-soft text-success', key: '3' },
]

function Freq({ n }: { n?: number }) {
  if (!n) return null
  return (
    <span className="inline-flex items-center gap-0.5" title={`出现频率 ${n}/3`}>
      {[1, 2, 3].map((k) => (
        <Star key={k} size={11} className={k <= n ? 'text-gold' : 'text-border-strong'} fill={k <= n ? 'currentColor' : 'none'} />
      ))}
    </span>
  )
}

export function QAStage() {
  const { levelId, L, openSummary } = useLevel()
  const cards = L.interview
  const rateQA = useProgress((s) => s.rateQA)
  const rated = useProgress((s) => s.levels[levelId]?.qaRated) ?? []
  const [idx, setIdx] = useState(0)
  const [flipped, setFlipped] = useState(false)
  const [ratings, setRatings] = useState<Record<string, Rating>>({})
  const [list, setList] = useState(false)
  const [finished, setFinished] = useState(false)

  const card = cards[Math.min(idx, cards.length - 1)]

  const go = (i: number) => {
    setIdx(Math.max(0, Math.min(cards.length - 1, i)))
    setFlipped(false)
  }
  const rate = (r: Rating) => {
    rateQA(levelId, card.id, r)
    setRatings((x) => ({ ...x, [card.id]: r }))
    if (idx < cards.length - 1) go(idx + 1)
    else setFinished(true)
  }

  useEffect(() => {
    if (list || finished || cards.length === 0) return
    const on = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return
      const t = e.target as HTMLElement
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.closest('[role="dialog"]'))) return
      if (e.key === ' ' || (e.key === 'Enter' && !flipped)) {
        if (t?.tagName === 'BUTTON' && e.key === 'Enter') return
        e.preventDefault()
        setFlipped((f) => !f)
      } else if (flipped && ['1', '2', '3'].includes(e.key)) {
        e.preventDefault()
        rate(RATINGS[Number(e.key) - 1].r)
      } else if (e.key === 'ArrowRight') go(idx + 1)
      else if (e.key === 'ArrowLeft') go(idx - 1)
    }
    window.addEventListener('keydown', on)
    return () => window.removeEventListener('keydown', on)
  })

  if (cards.length === 0) return <EmptyStage text="本关没有面试追问" />

  if (finished) {
    const cnt = (r: Rating) => Object.values(ratings).filter((x) => x === r).length
    return (
      <div className="mx-auto max-w-xl px-4 pb-16 pt-10">
        <div className="animate-pop rounded-3xl border border-border bg-surface p-6 text-center shadow-sm sm:p-8" data-qa-done>
          <Trophy size={36} className="mx-auto text-gold" />
          <h2 className="mt-3 text-xl font-bold">追问过完一轮！</h2>
          <div className="mt-4 grid grid-cols-3 gap-2">
            {RATINGS.slice()
              .reverse()
              .map((x) => (
                <div key={x.r} className={cn('rounded-2xl border px-2 py-3', x.cls)}>
                  <div className="text-2xl font-bold tabular-nums">{cnt(x.r)}</div>
                  <div className="text-xs font-medium">{x.label}</div>
                </div>
              ))}
          </div>
          <p className="mt-4 text-sm text-muted">「模糊」「不会」的卡片已加入间隔复习，会在 1 → 2 → 5 → 11… 天后再次出现。</p>
          <div className="mt-6 flex flex-col gap-2 sm:flex-row sm:justify-center">
            <Button variant="primary" size="lg" onClick={openSummary} icon={<Trophy size={18} />} data-action="summary">
              通关结算
            </Button>
            <Link to="/review" className="inline-flex h-12 items-center justify-center rounded-xl border border-border px-5 text-sm font-medium text-fg hover:bg-surface-2">
              去复习
            </Link>
            <Button
              variant="ghost"
              size="lg"
              icon={<RotateCcw size={17} />}
              onClick={() => {
                setFinished(false)
                setRatings({})
                go(0)
              }}
            >
              再来一遍
            </Button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-2xl px-4 pb-16 pt-5 sm:pt-8">
      <div className="mb-4 flex items-center gap-3">
        <div className="flex flex-1 gap-1">
          {cards.map((q, i) => (
            <button
              key={q.id}
              type="button"
              onClick={() => go(i)}
              aria-label={`第 ${i + 1} 张`}
              className={cn(
                'h-1.5 flex-1 rounded-full transition-colors',
                i === idx
                  ? 'bg-accent'
                  : ratings[q.id] === 'know'
                    ? 'bg-success'
                    : ratings[q.id] === 'fuzzy'
                      ? 'bg-warning'
                      : ratings[q.id] === 'dont'
                        ? 'bg-danger'
                        : rated.includes(q.id)
                          ? 'bg-border-strong'
                          : 'bg-surface-3',
              )}
            />
          ))}
        </div>
        <span className="shrink-0 text-xs text-muted tabular-nums">
          {idx + 1}/{cards.length}
        </span>
        <button
          type="button"
          onClick={() => setList((l) => !l)}
          className="inline-flex h-8 items-center gap-1 rounded-lg px-2 text-xs text-muted hover:bg-surface-2 hover:text-fg"
          aria-pressed={list}
        >
          {list ? <Layers size={14} /> : <List size={14} />}
          {list ? '闪卡' : '列表'}
        </button>
      </div>

      {list ? (
        <div className="space-y-3">
          {cards.map((q, i) => (
            <div key={q.id} className="rounded-2xl border border-border bg-surface p-4 sm:p-5">
              <div className="mb-1.5 flex items-center gap-2 text-xs text-subtle">
                Q{i + 1} <Freq n={q.freq} />
              </div>
              <Markdown className="font-semibold">{q.q}</Markdown>
              <div className="mt-3 border-t border-border pt-3">
                <Markdown className="!text-[14.5px]">{q.a}</Markdown>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <>
          <div style={{ perspective: 1400 }}>
            <AnimatePresence mode="wait" initial={false}>
              <motion.div
                key={`${card.id}-${flipped}`}
                initial={{ rotateY: flipped ? -90 : 0, opacity: flipped ? 0.4 : 0, y: flipped ? 0 : 12 }}
                animate={{ rotateY: 0, opacity: 1, y: 0 }}
                exit={{ rotateY: flipped ? 0 : 90, opacity: 0.4 }}
                transition={{ duration: 0.22, ease: 'easeOut' }}
                className={cn(
                  'rounded-3xl border bg-surface p-5 shadow-md sm:p-7',
                  flipped ? 'border-accent/40' : 'border-border',
                )}
                data-qa-card={card.id}
                data-flipped={flipped}
              >
                <div className="mb-3 flex items-center gap-2 text-xs font-semibold text-accent">
                  面试官问 · Q{idx + 1}
                  <Freq n={card.freq} />
                  {rated.includes(card.id) && <span className="ml-auto font-normal text-subtle">已自评过</span>}
                </div>
                <Markdown className={cn('font-semibold', flipped ? '!text-[15px] text-muted' : '!text-[17px]')}>{card.q}</Markdown>
                {flipped ? (
                  <div className="mt-4 border-t border-border pt-4">
                    <div className="mb-2 text-xs font-semibold text-success">参考回答</div>
                    <Markdown className="!text-[15px]">{card.a}</Markdown>
                  </div>
                ) : (
                  <div className="mt-6 rounded-2xl border border-dashed border-border-strong px-4 py-6 text-center text-sm text-muted">
                    先在心里（或出声）组织回答：结论 → 原因 → 延伸，
                    <br className="max-sm:hidden" />
                    再翻面对照参考回答。
                  </div>
                )}
              </motion.div>
            </AnimatePresence>
          </div>

          <div className="sticky bottom-0 mt-4 bg-bg/90 pb-[calc(0.75rem+env(safe-area-inset-bottom))] pt-2 backdrop-blur">
            {!flipped ? (
              <div className="flex items-center gap-2">
                <Button variant="ghost" size="icon" onClick={() => go(idx - 1)} disabled={idx === 0} aria-label="上一张">
                  <ArrowLeft size={18} />
                </Button>
                <Button variant="primary" size="lg" className="flex-1" onClick={() => setFlipped(true)} icon={<Eye size={18} />} data-action="flip">
                  翻面看参考回答
                  <Kbd className="ml-1 border-accent-fg/30 bg-transparent text-accent-fg max-md:hidden">空格</Kbd>
                </Button>
                <Button variant="ghost" size="icon" onClick={() => go(idx + 1)} disabled={idx === cards.length - 1} aria-label="下一张">
                  <ArrowRight size={18} />
                </Button>
              </div>
            ) : (
              <div>
                <p className="mb-2 text-center text-xs text-muted">你能讲到这个程度吗？</p>
                <div className="grid grid-cols-3 gap-2">
                  {RATINGS.map((x) => (
                    <button
                      key={x.r}
                      type="button"
                      onClick={() => rate(x.r)}
                      data-rate={x.r}
                      className={cn('relative flex h-14 flex-col items-center justify-center gap-0.5 rounded-2xl border text-sm font-semibold transition-transform active:scale-95', x.cls)}
                    >
                      <x.icon size={18} strokeWidth={2.5} />
                      {x.label}
                      <span className="absolute right-2 top-1.5 font-mono text-[10px] opacity-60 max-md:hidden">{x.key}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}
