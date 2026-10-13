/**
 * ① 图解：各节纵向排成故事流。
 *  - 桌面（≥1280）左侧目录 + 阅读进度；其它宽度顶部进度条 + 可展开目录
 *  - 可视化比正文更宽（最多约 1000px）
 *  - 「读完一节」= 本节末尾进入过视口 且 在本节停留足够时间（按字数估算），或答完节末小测，或点击「下一节」
 *  - 全部读完 → completeLearn
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ArrowDown, ArrowRight, Check, ChevronDown, Sparkles, Star, Trophy } from 'lucide-react'
import type { LearnSection } from '@/content/types'
import { Markdown } from '@/components/Markdown'
import { QuestionView } from '@/components/quiz/QuestionView'
import { Button } from '@/components/ui'
import { useProgress } from '@/lib/store'
import { XP } from '@/lib/game'
import { preloadWhenIdle } from '@/runner/client'
import { cn } from '@/lib/utils'
import { useLevel } from './context'
import { ErrorBoundary } from './Overlay'
import { STAGE_META } from './status'

/** 本节最少停留秒数：按正文长度估算（中文约 8 字/秒，取 35% 作为“扫读”下限） */
function minDwell(s: LearnSection) {
  const chars = (s.body + (s.after ?? '')).length
  return Math.min(20, Math.max(4, Math.round((chars / 8) * 0.35))) + (s.viz ? 3 : 0)
}

function Checkpoint({ s, levelId, onContinue }: { s: LearnSection; levelId: string; onContinue: () => void }) {
  const answerCheckpoint = useProgress((st) => st.answerCheckpoint)
  const markSectionRead = useProgress((st) => st.markSectionRead)
  const lit = useProgress((st) => st.levels[levelId]?.checkpointsCorrect.includes(s.id) ?? false)
  const [attempt, setAttempt] = useState(0)
  const [last, setLast] = useState<boolean | null>(null)
  const q = s.checkpoint!
  return (
    <div className="mt-6">
      <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-fg">
        <Sparkles size={16} className="text-accent" />
        节末小测
        {lit ? (
          <span className="inline-flex items-center gap-1 rounded-full bg-gold-soft px-2 py-0.5 text-xs text-gold">
            <Star size={12} fill="currentColor" /> 已点亮
          </span>
        ) : (
          <span className="text-xs font-normal text-subtle">答对点亮本节 · 答错不扣分</span>
        )}
      </div>
      <QuestionView
        key={attempt}
        question={q}
        seed={attempt}
        onAnswered={(ok) => {
          setLast(ok)
          answerCheckpoint(levelId, s.id, ok, q.id)
          markSectionRead(levelId, s.id)
        }}
        nextLabel={last === false ? '再试一次' : '继续阅读'}
        onNext={() => {
          if (last === false) {
            setAttempt((a) => a + 1)
            setLast(null)
          } else onContinue()
        }}
      />
    </div>
  )
}

export function LearnStage() {
  const { levelId, L, next, goStage } = useLevel()
  const sections = L.learn
  const read = useProgress((s) => s.levels[levelId]?.sectionsRead) ?? []
  const learnDone = useProgress((s) => s.levels[levelId]?.learnDone ?? false)
  const markSectionRead = useProgress((s) => s.markSectionRead)
  const completeLearn = useProgress((s) => s.completeLearn)

  const [current, setCurrent] = useState(0)
  const [progress, setProgress] = useState(0)
  const [tocOpen, setTocOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)
  const secRefs = useRef<(HTMLElement | null)[]>([])
  const endRefs = useRef<(HTMLElement | null)[]>([])
  const dwell = useRef<number[]>(sections.map(() => 0))
  const endSeen = useRef<boolean[]>(sections.map(() => false))

  const readSet = useMemo(() => new Set(read), [read])
  const readCount = sections.filter((s) => readSet.has(s.id)).length
  const allRead = readCount === sections.length

  // 空闲时预热 Python（后续跟练 / 实战要用）
  useEffect(() => preloadWhenIdle(6000), [])

  // 全部读完 → 完成图解
  useEffect(() => {
    if (allRead && !learnDone && sections.length > 0) completeLearn(levelId)
  }, [allRead, learnDone, completeLearn, levelId, sections.length])

  const tryMark = useCallback(
    (i: number) => {
      const s = sections[i]
      if (!s || readSet.has(s.id)) return
      if (endSeen.current[i] && dwell.current[i] >= minDwell(s)) markSectionRead(levelId, s.id)
    },
    [sections, readSet, markSectionRead, levelId],
  )

  // 末尾哨兵进入视口
  useEffect(() => {
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (!e.isIntersecting) continue
          const i = Number((e.target as HTMLElement).dataset.end)
          endSeen.current[i] = true
          tryMark(i)
        }
      },
      { rootMargin: '0px 0px -10% 0px' },
    )
    endRefs.current.forEach((el) => el && io.observe(el))
    return () => io.disconnect()
  }, [tryMark])

  // 当前节 + 停留计时 + 阅读进度
  const locate = useCallback(() => {
    const line = window.innerHeight * 0.4
    let cur = 0
    secRefs.current.forEach((el, i) => {
      if (el && el.getBoundingClientRect().top <= line) cur = i
    })
    setCurrent(cur)
    const root = rootRef.current
    if (root) {
      const r = root.getBoundingClientRect()
      const total = r.height - window.innerHeight * 0.6
      setProgress(Math.max(0, Math.min(1, total > 0 ? -r.top / total : 1)))
    }
    return cur
  }, [])
  useEffect(() => {
    locate()
    const onScroll = () => locate()
    window.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('resize', onScroll)
    const timer = setInterval(() => {
      if (document.visibilityState !== 'visible') return
      const cur = locate()
      dwell.current[cur] = (dwell.current[cur] ?? 0) + 1
      // 短节：末尾与下一节同屏可见时，停留也计入
      const el = secRefs.current[cur + 1]
      if (el && el.getBoundingClientRect().top < window.innerHeight * 0.85) dwell.current[cur + 1] = (dwell.current[cur + 1] ?? 0) + 0.5
      tryMark(cur)
      tryMark(cur + 1)
    }, 1000)
    return () => {
      window.removeEventListener('scroll', onScroll)
      window.removeEventListener('resize', onScroll)
      clearInterval(timer)
    }
  }, [locate, tryMark])

  const scrollToSection = (i: number) => {
    const el = i >= sections.length ? document.getElementById('learn-end') : secRefs.current[i]
    if (!el) return
    const top = el.getBoundingClientRect().top + window.scrollY - (parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--lv-top')) || 64) - 56
    window.scrollTo({ top, behavior: 'smooth' })
    setTocOpen(false)
  }

  /** 「下一节」：显式确认读完当前节 */
  const goNextSection = () => {
    const s = sections[current]
    if (s && !readSet.has(s.id)) markSectionRead(levelId, s.id)
    scrollToSection(current + 1)
  }

  const nextMeta = next ? STAGE_META[next] : null

  const toc = (
    <ol className="space-y-0.5">
      {sections.map((s, i) => {
        const isRead = readSet.has(s.id)
        return (
          <li key={s.id}>
            <button
              type="button"
              onClick={() => scrollToSection(i)}
              className={cn(
                'flex w-full items-start gap-2.5 rounded-lg px-2 py-1.5 text-left text-[13px] leading-snug transition-colors',
                i === current ? 'bg-accent-soft font-semibold text-fg' : 'text-muted hover:bg-surface-2 hover:text-fg',
              )}
            >
              <span
                className={cn(
                  'mt-px inline-flex h-[18px] w-[18px] shrink-0 items-center justify-center rounded-full text-[10px] font-bold',
                  isRead ? 'bg-success text-surface' : i === current ? 'bg-accent text-accent-fg' : 'border border-border-strong text-subtle',
                )}
              >
                {isRead ? <Check size={11} strokeWidth={3} /> : i + 1}
              </span>
              <span className="min-w-0">{s.title}</span>
            </button>
          </li>
        )
      })}
    </ol>
  )

  return (
    <div ref={rootRef} className="relative pb-28">
      {/* 顶部进度（< xl） */}
      <div className="sticky top-[var(--lv-top,56px)] z-20 border-b border-border bg-bg/90 backdrop-blur xl:hidden">
        <div className="h-0.5 w-full bg-surface-3">
          <div className="cq-progress-bar h-full w-full bg-accent" style={{ transform: `scaleX(${progress})` }} />
        </div>
        <button
          type="button"
          onClick={() => setTocOpen((o) => !o)}
          className="mx-auto flex w-full max-w-[46rem] items-center gap-2 px-4 py-2 text-left text-[13px]"
          aria-expanded={tocOpen}
        >
          <span className="shrink-0 font-semibold text-accent tabular-nums">
            {Math.min(current + 1, sections.length)}/{sections.length}
          </span>
          <span className="min-w-0 flex-1 truncate text-fg">{sections[current]?.title}</span>
          <span className="shrink-0 text-xs text-subtle">已读 {readCount}</span>
          <ChevronDown size={16} className={cn('shrink-0 text-muted transition-transform', tocOpen && 'rotate-180')} />
        </button>
        {tocOpen && <div className="mx-auto max-w-[46rem] animate-fade-in border-t border-border px-2 py-2">{toc}</div>}
      </div>

      <div className="mx-auto flex max-w-[1320px] gap-8 px-4 sm:px-6">
        {/* 左侧目录（≥ xl） */}
        <aside className="sticky top-[calc(var(--lv-top,56px)+24px)] hidden h-fit w-56 shrink-0 pt-8 xl:block">
          <div className="mb-3 text-xs font-semibold tracking-wide text-subtle">本关图解</div>
          {toc}
          <div className="mt-4 px-2">
            <div className="mb-1 flex justify-between text-xs text-muted">
              <span>阅读进度</span>
              <span className="tabular-nums">
                {readCount}/{sections.length} 节
              </span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-surface-3">
              <div className="h-full rounded-full bg-success transition-[width] duration-500" style={{ width: `${(readCount / Math.max(1, sections.length)) * 100}%` }} />
            </div>
          </div>
        </aside>

        <div className="min-w-0 flex-1 xl:pr-16">
          {sections.map((s, i) => {
            const isRead = readSet.has(s.id)
            return (
              <section
                key={s.id}
                id={`sec-${s.id}`}
                data-section={s.id}
                ref={(el) => {
                  secRefs.current[i] = el
                }}
                className="border-b border-border/70 py-8 last:border-b-0 sm:py-10"
              >
                <div className="mx-auto max-w-[46rem]">
                  <div className="mb-1 flex items-center gap-2 text-xs font-semibold text-accent">
                    第 {i + 1} 节
                    {isRead && (
                      <span className="inline-flex items-center gap-0.5 rounded-full bg-success-soft px-1.5 py-0.5 text-[11px] text-success">
                        <Check size={11} strokeWidth={3} /> 已读
                      </span>
                    )}
                  </div>
                  <h2 className="mb-4 text-[1.35rem] font-bold leading-snug tracking-tight text-fg sm:text-2xl">{s.title}</h2>
                  <Markdown>{s.body}</Markdown>
                </div>
                {s.viz && (
                  <div className="mx-auto my-6 max-w-[1000px]" data-viz={s.id}>
                    <ErrorBoundary name={s.title}>
                      <s.viz />
                    </ErrorBoundary>
                  </div>
                )}
                <div className="mx-auto max-w-[46rem]">
                  {s.after && <Markdown>{s.after}</Markdown>}
                  {s.checkpoint && <Checkpoint s={s} levelId={levelId} onContinue={() => scrollToSection(i + 1)} />}
                  <div
                    data-end={i}
                    ref={(el) => {
                      endRefs.current[i] = el
                    }}
                    className="h-px"
                  />
                </div>
              </section>
            )
          })}

          {/* 完成卡片 */}
          <div id="learn-end" className="mx-auto mt-6 max-w-[46rem]">
            {learnDone ? (
              <div className="animate-pop rounded-2xl border border-success/40 bg-success-soft/60 p-5 text-center sm:p-6">
                <Trophy size={32} className="mx-auto text-gold" />
                <h3 className="mt-2 text-lg font-bold text-fg">图解完成！</h3>
                <p className="mt-1 text-sm text-muted">已获得 +{XP.learnDone} XP。趁热打铁，去做随堂练习检验一下理解。</p>
                {nextMeta && (
                  <Button variant="primary" size="lg" className="mt-4" onClick={() => goStage(next!)} icon={<nextMeta.icon size={18} />}>
                    进入{nextMeta.label}
                    <ArrowRight size={18} />
                  </Button>
                )}
              </div>
            ) : (
              <div className="rounded-2xl border border-dashed border-border-strong p-5 text-center text-sm text-muted">
                还有 {sections.length - readCount} 节未读完。读到每节末尾（或答完节末小测）即算读完。
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 底部主操作栏 */}
      <div className="fixed inset-x-0 bottom-0 z-30 border-t border-border bg-surface/92 backdrop-blur pb-[env(safe-area-inset-bottom)]">
        <div className="mx-auto flex max-w-[46rem] items-center gap-3 px-4 py-2.5">
          <div className="min-w-0 flex-1">
            <div className="flex items-center justify-between text-xs text-muted">
              <span>
                已读 <b className="text-fg tabular-nums">{readCount}</b>/{sections.length} 节
              </span>
              {learnDone && <span className="font-semibold text-success">图解完成 ✓</span>}
            </div>
            <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-surface-3">
              <div className="h-full rounded-full bg-success transition-[width] duration-500" style={{ width: `${(readCount / Math.max(1, sections.length)) * 100}%` }} />
            </div>
          </div>
          {learnDone || (allRead && sections.length > 0) ? (
            next ? (
              <Button variant="primary" onClick={() => goStage(next)} data-action="next-stage">
                下一阶段：{STAGE_META[next].short}
                <ArrowRight size={16} />
              </Button>
            ) : null
          ) : current < sections.length - 1 ? (
            <Button variant="primary" onClick={goNextSection} data-action="next-section">
              下一节
              <ArrowDown size={16} />
            </Button>
          ) : (
            <Button variant="primary" onClick={goNextSection} data-action="finish-learn">
              读完了
              <Check size={16} />
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
