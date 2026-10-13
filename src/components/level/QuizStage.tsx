/**
 * ② 随堂练习：一次一题 + 进度条；每题 recordAnswer；结束页显示得分、星级变化、XP、错题列表，可重做。
 * 会话状态按关卡缓存在内存中：切换阶段再回来不会丢失进度。
 */
import { useEffect, useState } from 'react'
import { ArrowRight, ChevronDown, Flame, RotateCcw, Star, Target, Trophy, XCircle } from 'lucide-react'
import { QuestionView } from '@/components/quiz/QuestionView'
import { InlineMarkdown } from '@/components/Markdown'
import { Button } from '@/components/ui'
import { QUESTION_TYPE_LABEL } from '@/content/types'
import { useProgress } from '@/lib/store'
import { starsOf } from '@/lib/game'
import { cn } from '@/lib/utils'
import { useLevel } from './context'
import { STAGE_META } from './status'

interface Session {
  mode: 'full' | 'wrong'
  /** 本轮题目（quiz 下标） */
  items: number[]
  idx: number
  answers: (boolean | null)[]
  seed: number
  finished: boolean
  /** 正式一轮的结算 */
  result?: { correct: number; total: number; xp: number; starsBefore: number; starsAfter: number; wrong: number[] }
}

const sessions = new Map<string, Session>()

const newSession = (items: number[], mode: Session['mode'], seed: number): Session => ({
  mode,
  items,
  idx: 0,
  answers: items.map(() => null),
  seed,
  finished: false,
})

function StarsRow({ before, after, size = 30 }: { before: number; after: number; size?: number }) {
  return (
    <div className="flex items-center justify-center gap-2">
      {[1, 2, 3].map((k) => {
        const on = k <= after
        const fresh = on && k > before
        return (
          <Star
            key={k}
            size={size}
            strokeWidth={1.8}
            className={cn(on ? 'text-gold' : 'text-border-strong', fresh && 'cq-star-in')}
            style={fresh ? { animationDelay: `${0.25 + (k - before - 1) * 0.25}s` } : undefined}
            fill={on ? 'currentColor' : 'none'}
          />
        )
      })}
    </div>
  )
}

export function QuizStage() {
  const { levelId, L, next, goStage } = useLevel()
  const quiz = L.quiz
  const recordAnswer = useProgress((s) => s.recordAnswer)
  const completeQuiz = useProgress((s) => s.completeQuiz)
  const best = useProgress((s) => s.levels[levelId]?.quizBest ?? 0)
  const attempts = useProgress((s) => s.levels[levelId]?.quizAttempts ?? 0)

  const [sess, setSessRaw] = useState<Session>(() => sessions.get(levelId) ?? newSession(quiz.map((_, i) => i), 'full', 1))
  const setSess = (s: Session) => {
    sessions.set(levelId, s)
    setSessRaw(s)
  }
  const [openWrong, setOpenWrong] = useState<number | null>(null)
  /** 从缓存恢复时，当前题若已作答则只读展示，避免重复记分 */
  const [restoredIdx] = useState(() => (sess.answers[sess.idx] != null ? sess.idx : -1))

  useEffect(() => {
    window.scrollTo({ top: 0 })
  }, [sess.idx, sess.finished])

  if (quiz.length === 0) return <EmptyStage />

  const qi = sess.items[sess.idx]
  const q = quiz[qi]
  const answeredCount = sess.answers.filter((a) => a !== null).length
  const correctCount = sess.answers.filter((a) => a === true).length

  const onAnswered = (ok: boolean) => {
    recordAnswer(levelId, q.id, ok, 'quiz')
    const answers = sess.answers.slice()
    answers[sess.idx] = ok
    setSess({ ...sess, answers })
  }

  const onNext = () => {
    if (sess.idx < sess.items.length - 1) return setSess({ ...sess, idx: sess.idx + 1 })
    // 一轮结束
    if (sess.mode === 'full') {
      const correct = sess.answers.filter((a) => a === true).length
      const before = starsOf(useProgress.getState().levels[levelId])
      const xp = completeQuiz(levelId, correct, quiz.length)
      const after = starsOf(useProgress.getState().levels[levelId])
      const wrong = sess.items.filter((_, k) => sess.answers[k] !== true)
      setSess({ ...sess, finished: true, result: { correct, total: quiz.length, xp, starsBefore: before, starsAfter: after, wrong } })
    } else {
      const prev = sessions.get(`${levelId}::last`) ?? sess
      setSess({ ...prev, finished: true, result: prev.result, mode: 'full', items: prev.items, answers: prev.answers })
    }
  }

  const restart = () => setSess(newSession(quiz.map((_, i) => i), 'full', sess.seed + 1))
  const practiceWrong = (wrong: number[]) => {
    sessions.set(`${levelId}::last`, sess)
    setSess(newSession(wrong, 'wrong', sess.seed + 1))
  }

  /* ---------------- 结算页 ---------------- */
  if (sess.finished && sess.result) {
    const r = sess.result
    const pct = Math.round((r.correct / r.total) * 100)
    const R = 52
    const C = 2 * Math.PI * R
    const nextMeta = next ? STAGE_META[next] : null
    return (
      <div className="mx-auto max-w-2xl px-4 pb-16 pt-6 sm:pt-10">
        <div className="animate-pop rounded-3xl border border-border bg-surface p-6 text-center shadow-sm sm:p-8" data-quiz-result>
          <div className="relative mx-auto h-36 w-36">
            <svg viewBox="0 0 120 120" className="h-full w-full -rotate-90">
              <circle cx="60" cy="60" r={R} fill="none" stroke="var(--surface-3)" strokeWidth="10" />
              <circle
                cx="60"
                cy="60"
                r={R}
                fill="none"
                stroke={pct >= 80 ? 'var(--success)' : pct >= 60 ? 'var(--accent)' : 'var(--warning)'}
                strokeWidth="10"
                strokeLinecap="round"
                strokeDasharray={C}
                strokeDashoffset={C * (1 - r.correct / r.total)}
                style={{ transition: 'stroke-dashoffset 1s ease-out' }}
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-3xl font-bold tabular-nums text-fg">{pct}%</span>
              <span className="text-xs text-muted">
                答对 {r.correct}/{r.total}
              </span>
            </div>
          </div>
          <h2 className="mt-4 text-xl font-bold text-fg">
            {pct === 100 ? '全对！完美通关 🎉' : pct >= 80 ? '掌握得很好！' : pct >= 60 ? '达标了，再巩固一下' : '还需要再看看图解'}
          </h2>
          <div className="mt-4">
            <StarsRow before={r.starsBefore} after={r.starsAfter} />
            <p className="mt-2 text-xs text-muted">
              {r.starsAfter > r.starsBefore
                ? `星级提升：${r.starsBefore} → ${r.starsAfter} 星`
                : r.starsAfter === 0
                  ? '★ 需要：图解完成 且 随堂 ≥ 60%'
                  : r.starsAfter === 1
                    ? '★★ 需要：随堂 ≥ 80% 且 跟练全部完成'
                    : r.starsAfter === 2
                      ? '★★★ 需要：实战全部通过（未看题解）'
                      : '本关已三星'}
            </p>
          </div>
          <div className="mt-4 flex flex-wrap items-center justify-center gap-2 text-sm">
            <span className="inline-flex items-center gap-1 rounded-full bg-gold-soft px-3 py-1 font-semibold text-gold">
              <Flame size={14} /> +{r.xp} XP
            </span>
            <span className="inline-flex items-center gap-1 rounded-full bg-surface-2 px-3 py-1 text-muted">
              <Target size={14} /> 历史最佳 {Math.round(best * 100)}% · 第 {attempts} 次
            </span>
          </div>
          {r.xp === 0 && <p className="mt-2 text-xs text-subtle">XP 只按「新增答对数」发放，超过历史最佳才会再得分</p>}
          <div className="mt-6 flex flex-col gap-2 sm:flex-row sm:justify-center">
            {nextMeta && (
              <Button variant="primary" size="lg" onClick={() => goStage(next!)} data-action="next-stage">
                下一阶段：{nextMeta.label}
                <ArrowRight size={18} />
              </Button>
            )}
            {r.wrong.length > 0 && (
              <Button variant="secondary" size="lg" onClick={() => practiceWrong(r.wrong)} icon={<XCircle size={17} />}>
                练习错题（{r.wrong.length}）
              </Button>
            )}
            <Button variant="ghost" size="lg" onClick={restart} icon={<RotateCcw size={17} />} data-action="restart-quiz">
              重新挑战
            </Button>
          </div>
        </div>

        {r.wrong.length > 0 && (
          <div className="mt-6">
            <h3 className="mb-2 text-sm font-semibold text-muted">错题回顾（已加入复习）</h3>
            <div className="space-y-2">
              {r.wrong.map((i) => (
                <div key={i} className="overflow-hidden rounded-2xl border border-border bg-surface">
                  <button
                    type="button"
                    onClick={() => setOpenWrong(openWrong === i ? null : i)}
                    className="flex w-full items-center gap-3 px-4 py-3 text-left text-sm hover:bg-surface-2"
                  >
                    <span className="shrink-0 rounded-full bg-danger-soft px-2 py-0.5 text-xs font-semibold text-danger">
                      {QUESTION_TYPE_LABEL[quiz[i].type]}
                    </span>
                    <span className="line-clamp-1 min-w-0 flex-1 text-fg">
                      <InlineMarkdown>{quiz[i].prompt.split('\n')[0]}</InlineMarkdown>
                    </span>
                    <ChevronDown size={16} className={cn('shrink-0 text-muted transition-transform', openWrong === i && 'rotate-180')} />
                  </button>
                  {openWrong === i && (
                    <div className="border-t border-border p-2 sm:p-3">
                      <QuestionView question={quiz[i]} reveal seed={sess.seed} className="border-0 shadow-none" />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    )
  }

  /* ---------------- 答题中 ---------------- */
  return (
    <div className="mx-auto max-w-2xl px-4 pb-16 pt-4 sm:pt-8">
      <div className="mb-4">
        <div className="mb-2 flex items-center justify-between text-sm">
          <span className="font-semibold text-fg">
            {sess.mode === 'wrong' ? '错题练习 · ' : ''}第 <span className="tabular-nums">{sess.idx + 1}</span> / {sess.items.length} 题
          </span>
          <span className="flex items-center gap-3 text-xs text-muted">
            {sess.mode === 'full' && attempts > 0 && <span>最佳 {Math.round(best * 100)}%</span>}
            <span className="inline-flex items-center gap-1">
              <Trophy size={13} className="text-success" />
              <span className="tabular-nums">
                {correctCount}/{answeredCount}
              </span>
            </span>
          </span>
        </div>
        <div className="flex gap-1" aria-hidden>
          {sess.items.map((_, k) => (
            <div
              key={k}
              className={cn(
                'h-1.5 flex-1 rounded-full transition-colors duration-300',
                sess.answers[k] === true
                  ? 'bg-success'
                  : sess.answers[k] === false
                    ? 'bg-danger'
                    : k === sess.idx
                      ? 'bg-accent'
                      : 'bg-surface-3',
              )}
            />
          ))}
        </div>
      </div>
      <QuestionView
        key={`${sess.mode}-${sess.seed}-${q.id}`}
        question={q}
        seed={sess.seed}
        className="cq-qv-sticky"
        reveal={sess.idx === restoredIdx && sess.answers[sess.idx] !== null}
        onAnswered={onAnswered}
        onNext={onNext}
        nextLabel={sess.idx < sess.items.length - 1 ? '下一题' : sess.mode === 'full' ? '查看结果' : '完成练习'}
      />
    </div>
  )
}

export function EmptyStage({ text = '本关没有这一阶段的内容' }: { text?: string }) {
  const { next, goStage } = useLevel()
  return (
    <div className="mx-auto max-w-md px-4 py-20 text-center">
      <p className="text-muted">{text}</p>
      {next && (
        <Button variant="primary" className="mt-4" onClick={() => goStage(next)}>
          前往{STAGE_META[next].label}
          <ArrowRight size={16} />
        </Button>
      )}
    </div>
  )
}
