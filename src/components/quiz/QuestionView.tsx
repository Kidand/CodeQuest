/**
 * 【P2 平台-关卡 负责】单题渲染器（10 种题型）。随堂练习、节末小测、错题复习、模拟面试共用。
 * ⚠️ 接口已冻结：P1 的复习页与模拟面试页依赖此签名。
 *
 * 交互：
 *  - 全部题型支持点选（触屏），排序 / 拼图额外支持手柄拖拽；
 *  - 键盘：1-9 / A-J 选择选项（判断题 1=正确 2=错误），Enter 提交 / 继续；
 *    页面上只有一个 QuestionView 时全局生效，多个时作用于焦点所在或最近操作过的那一题；
 *  - 提交后逐项标注对错，展示解析与「继续」按钮（提供 onNext 时）。
 */
import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'
import { ArrowRight, BookOpen, CheckCircle2, Lightbulb, XCircle } from 'lucide-react'
import { QUESTION_TYPE_LABEL, type Question } from '@/content/types'
import { Markdown } from '@/components/Markdown'
import { CodeBlock } from '@/components/CodeBlock'
import { cn } from '@/lib/utils'
import { correctAnswer, initialAnswer, isCorrect, makeLayout, submitBlocker, type AnyAnswer } from './grade'
import { Kbd, type BodyProps } from './marks'
import { ChoiceBody, JudgeBody, LETTERS } from './ChoiceBody'
import { FillBody, NumericBody, fillPrompt } from './FillBody'
import { OrderBody } from './OrderBody'
import { ParsonsBody } from './ParsonsBody'
import { MatchBody } from './MatchBody'
import { BugBody, PickBody } from './BugPickBody'
import '@/styles/level.css'

export interface QuestionViewProps {
  question: Question
  /** 用户提交后回调一次（correct=是否答对） */
  onAnswered?: (correct: boolean) => void
  /** 提交后点击“继续/下一题” */
  onNext?: () => void
  /** 下一题按钮文案，默认“继续” */
  nextLabel?: string
  /** 只读展示答案（例如复习时查看解析） */
  reveal?: boolean
  /** 洗牌种子（保证同一题在重渲染时选项顺序稳定） */
  seed?: number
  className?: string
}

/* ---------------- 键盘作用域：多个实例同时存在时，只有一个响应全局按键 ---------------- */
const mounted = new Set<string>()
let lastActive: string | null = null

function inViewport(el: HTMLElement) {
  const r = el.getBoundingClientRect()
  return r.bottom > 0 && r.top < window.innerHeight
}
function isTypingTarget(el: Element | null) {
  if (!el) return false
  const tag = el.tagName
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (el as HTMLElement).isContentEditable
}

const DIFF_LABEL = ['', '基础', '进阶', '压轴'] as const

function Body(props: BodyProps<Question>) {
  const q = props.q
  switch (q.type) {
    case 'single':
    case 'multi':
      return <ChoiceBody {...(props as BodyProps<typeof q>)} />
    case 'judge':
      return <JudgeBody {...(props as BodyProps<typeof q>)} />
    case 'fill':
      return <FillBody {...(props as BodyProps<typeof q>)} />
    case 'numeric':
      return <NumericBody {...(props as BodyProps<typeof q>)} />
    case 'order':
      return <OrderBody {...(props as BodyProps<typeof q>)} />
    case 'match':
      return <MatchBody {...(props as BodyProps<typeof q>)} />
    case 'parsons':
      return <ParsonsBody {...(props as BodyProps<typeof q>)} />
    case 'bug':
      return <BugBody {...(props as BodyProps<typeof q>)} />
    case 'pick':
      return <PickBody {...(props as BodyProps<typeof q>)} />
  }
}

export function QuestionView({ question: q, onAnswered, onNext, nextLabel = '继续', reveal = false, seed = 0, className }: QuestionViewProps) {
  const uid = useId()
  const rootRef = useRef<HTMLDivElement>(null)
  const feedbackRef = useRef<HTMLDivElement>(null)
  const layout = useMemo(() => makeLayout(q, seed), [q, seed])

  const fresh = useCallback(() => (reveal ? correctAnswer(q) : initialAnswer(q, layout)), [q, layout, reveal])
  const [value, setValue] = useState<AnyAnswer>(fresh)
  const [result, setResult] = useState<boolean | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [showHint, setShowHint] = useState(false)
  const [anim, setAnim] = useState<{ kind: 'pop' | 'shake'; n: number } | null>(null)

  // 题目 / 种子 / 模式变化：重置（父组件通常会用 key，这里兜底）
  const [ident, setIdent] = useState({ q, layout, reveal })
  if (ident.q !== q || ident.layout !== layout || ident.reveal !== reveal) {
    setIdent({ q, layout, reveal })
    setValue(reveal ? correctAnswer(q) : initialAnswer(q, layout))
    setResult(null)
    setNotice(null)
    setShowHint(false)
  }

  const submitted = reveal || result !== null

  const change = useCallback((v: AnyAnswer) => {
    setValue(v)
    setNotice(null)
  }, [])

  const submit = useCallback(() => {
    if (submitted) return
    const block = submitBlocker(q, value)
    if (block) {
      setNotice(block)
      setAnim((a) => ({ kind: 'shake', n: (a?.n ?? 0) + 1 }))
      return
    }
    const ok = isCorrect(q, value)
    setResult(ok)
    setNotice(null)
    setAnim((a) => ({ kind: ok ? 'pop' : 'shake', n: (a?.n ?? 0) + 1 }))
    onAnswered?.(ok)
  }, [submitted, q, value, onAnswered])

  // 提交后把解析滚入视野
  useEffect(() => {
    if (result === null) return
    const t = setTimeout(() => {
      const el = feedbackRef.current
      if (!el) return
      const r = el.getBoundingClientRect()
      if (r.top > window.innerHeight - 120) el.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }, 120)
    return () => clearTimeout(t)
  }, [result])

  // 键盘
  const stateRef = useRef({ submitted, submit, onNext, q, value, change })
  stateRef.current = { submitted, submit, onNext, q, value, change }
  useEffect(() => {
    mounted.add(uid)
    if (mounted.size === 1) lastActive = uid
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey || e.isComposing) return
      const root = rootRef.current
      if (!root) return
      const target = e.target as Element | null
      const inside = !!target && root.contains(target)
      const ae = document.activeElement
      const focusInside = !!ae && ae !== document.body && root.contains(ae)
      if (!inside) {
        // 焦点在其它可交互元素 / 对话框里时不劫持
        if (target && target !== document.body && (isTypingTarget(target) || target.closest('[role="dialog"]'))) return
        if (target && target !== document.body && /^(BUTTON|A|SUMMARY)$/.test(target.tagName) && e.key === 'Enter') return
      }
      if (inside && isTypingTarget(target)) return // 输入框自行处理 Enter
      const active = focusInside || mounted.size === 1 || (lastActive === uid && inViewport(root))
      if (!active) return
      const s = stateRef.current
      if (e.key === 'Enter') {
        e.preventDefault()
        if (!s.submitted) s.submit()
        else s.onNext?.()
        return
      }
      if (s.submitted) return
      const qq = s.q
      const key = e.key.toLowerCase()
      let idx = -1
      if (/^[1-9]$/.test(key)) idx = Number(key) - 1
      else if (qq.type === 'single' || qq.type === 'multi') idx = LETTERS.toLowerCase().indexOf(key)
      if (idx < 0) return
      if (qq.type === 'single' && idx < qq.options.length) {
        e.preventDefault()
        s.change(idx)
      } else if (qq.type === 'multi' && idx < qq.options.length) {
        e.preventDefault()
        const cur = s.value as number[]
        s.change(cur.includes(idx) ? cur.filter((x) => x !== idx) : [...cur, idx].sort((a, b) => a - b))
      } else if (qq.type === 'judge' && idx < 2) {
        e.preventDefault()
        s.change(idx === 0)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('keydown', onKey)
      mounted.delete(uid)
      if (lastActive === uid) lastActive = [...mounted].pop() ?? null
    }
  }, [uid])

  const markActive = () => {
    lastActive = uid
  }

  const prompt = q.type === 'fill' ? fillPrompt(q.prompt, value as string[]) : q.prompt
  const hasKeys = q.type === 'single' || q.type === 'multi' || q.type === 'judge'

  return (
    <div
      ref={rootRef}
      onPointerDownCapture={markActive}
      onFocusCapture={markActive}
      data-question={q.id}
      data-qtype={q.type}
      data-state={reveal ? 'reveal' : result === null ? 'answering' : result ? 'correct' : 'wrong'}
      className={cn('cq-qv relative rounded-2xl border border-border bg-surface shadow-sm', className)}
    >
      <div
        key={anim?.n ?? 0}
        className={cn('p-4 sm:p-5', anim?.kind === 'pop' && 'animate-pop', anim?.kind === 'shake' && 'animate-shake')}
      >
        {/* 题头 */}
        <div className="mb-2.5 flex flex-wrap items-center gap-2 text-xs">
          <span className="rounded-full bg-accent-soft px-2 py-0.5 font-semibold text-accent">{QUESTION_TYPE_LABEL[q.type]}</span>
          {q.type === 'multi' && !submitted && <span className="text-muted">选出全部正确项</span>}
          {q.difficulty && (
            <span className="inline-flex items-center gap-1 text-subtle" title={`难度 ${q.difficulty}/3`}>
              {[1, 2, 3].map((k) => (
                <span key={k} className={cn('h-1.5 w-1.5 rounded-full', k <= q.difficulty! ? 'bg-warning' : 'bg-surface-3')} />
              ))}
              <span className="ml-0.5">{DIFF_LABEL[q.difficulty]}</span>
            </span>
          )}
          {q.hint && !submitted && (
            <button
              type="button"
              onClick={() => setShowHint((x) => !x)}
              className="ml-auto inline-flex items-center gap-1 rounded-full px-2 py-0.5 font-medium text-warning transition-colors hover:bg-warning-soft"
              aria-expanded={showHint}
            >
              <Lightbulb size={13} />
              {showHint ? '收起提示' : '提示'}
            </button>
          )}
        </div>

        <Markdown className="mb-4 text-[15.5px]">{prompt}</Markdown>

        {showHint && q.hint && !submitted && (
          <div className="mb-4 flex animate-fade-in gap-2 rounded-xl border border-warning/40 bg-warning-soft/70 px-3 py-2 text-sm text-fg">
            <Lightbulb size={16} className="mt-0.5 shrink-0 text-warning" />
            <Markdown className="!text-sm">{q.hint}</Markdown>
          </div>
        )}

        <Body q={q} value={value} onChange={change} submitted={submitted} reveal={reveal} layout={layout} onSubmitRequest={submit} />

        {/* 反馈 + 解析 */}
        {submitted && (
          <div ref={feedbackRef} className="mt-4 animate-fade-in space-y-3">
            {reveal ? (
              <div className="flex items-center gap-2 text-sm font-semibold text-success">
                <BookOpen size={18} /> 参考答案
              </div>
            ) : result ? (
              <div className="flex items-center gap-2 text-[15px] font-semibold text-success" role="status">
                <CheckCircle2 size={20} /> 回答正确！
              </div>
            ) : (
              <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[15px] font-semibold text-danger" role="status">
                <XCircle size={20} /> 回答错误
                {q.type === 'single' && (
                  <span className="text-sm font-normal text-muted">
                    正确答案是 <b className="text-success">{LETTERS[q.answer]}</b>
                  </span>
                )}
                {q.type === 'judge' && (
                  <span className="text-sm font-normal text-muted">
                    正确答案是 <b className="text-success">{q.answer ? '正确' : '错误'}</b>
                  </span>
                )}
                {q.type === 'multi' && (
                  <span className="text-sm font-normal text-muted">
                    正确答案是 <b className="text-success">{q.answer.map((i) => LETTERS[i]).join('、')}</b>
                  </span>
                )}
              </div>
            )}
            {q.type === 'parsons' && !result && !reveal && (
              <div>
                <div className="mb-1 text-xs font-semibold text-muted">参考代码</div>
                <CodeBlock code={q.lines.join('\n')} />
              </div>
            )}
            <div className="rounded-xl border border-border bg-surface-2/70 px-4 py-3">
              <div className="mb-1 flex items-center gap-1.5 text-xs font-semibold text-muted">
                <BookOpen size={13} /> 解析
              </div>
              <Markdown className="!text-[14.5px]">{q.explanation}</Markdown>
            </div>
          </div>
        )}

        {notice && !submitted && (
          <div className="mt-3 text-sm font-medium text-warning" role="alert">
            {notice}
          </div>
        )}
      </div>

      {/* 操作栏 */}
      {(!submitted || onNext) && (
        <div className="cq-qv-actions flex items-center justify-end gap-3 rounded-b-2xl border-t border-border bg-surface/95 px-4 py-3 backdrop-blur sm:px-5">
          {!submitted ? (
            <>
              <span className="mr-auto hidden items-center gap-1.5 text-xs text-subtle md:flex">
                {hasKeys && (
                  <>
                    <Kbd>{q.type === 'judge' ? '1' : '1'}</Kbd>
                    {q.type === 'judge' ? <Kbd>2</Kbd> : <>–<Kbd>{Math.min(9, (q as { options: string[] }).options.length)}</Kbd></>}
                    选择 ·
                  </>
                )}
                <Kbd>Enter</Kbd> 提交
              </span>
              <button
                type="button"
                onClick={submit}
                data-action="submit"
                className={cn(
                  'inline-flex h-11 min-w-28 items-center justify-center gap-2 rounded-xl px-6 text-[15px] font-semibold shadow-sm transition-[background,transform,opacity] active:scale-[0.97] max-md:flex-1',
                  submitBlocker(q, value) ? 'bg-surface-3 text-muted' : 'bg-accent text-accent-fg hover:bg-accent-hover',
                )}
              >
                提交
              </button>
            </>
          ) : (
            <>
              <span className="mr-auto hidden items-center gap-1.5 text-xs text-subtle md:flex">
                <Kbd>Enter</Kbd> {nextLabel}
              </span>
              <button
                type="button"
                onClick={onNext}
                data-action="next"
                autoFocus={!reveal}
                className={cn(
                  'inline-flex h-11 min-w-28 items-center justify-center gap-2 rounded-xl px-6 text-[15px] font-semibold text-white shadow-sm transition-[filter,transform] hover:brightness-110 active:scale-[0.97] max-md:flex-1 dark:text-bg',
                  reveal || result ? 'bg-success' : 'bg-accent',
                )}
              >
                {nextLabel}
                <ArrowRight size={17} />
              </button>
            </>
          )}
        </div>
      )}
    </div>
  )
}
export default QuestionView
