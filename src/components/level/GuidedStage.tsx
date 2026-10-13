/**
 * ③ 跟练：代码骨架中的空位渲染为内联输入框，逐步填写。
 * 判定：normalizeAnswer 比对 answer / accept → 不匹配则把用户答案代入（≤k 用用户答案、>k 用参考答案）跑测试判等价。
 * 全部填完后用用户的完整代码跑全部测试，通过则 completeGuided。进度保存在 useDrafts。
 */
import { Fragment, useEffect, useMemo, useRef, useState } from 'react'
import {
  ArrowRight,
  Check,
  CheckCircle2,
  Eye,
  Flame,
  Lightbulb,
  Loader2,
  Play,
  RotateCcw,
  Sparkles,
  XCircle,
} from 'lucide-react'
import type { GuidedExercise } from '@/content/types'
import { Markdown } from '@/components/Markdown'
import { highlight } from '@/components/CodeBlock'
import { Button } from '@/components/ui'
import { draftKey, useDrafts, useProgress } from '@/lib/store'
import { XP } from '@/lib/game'
import { cn, normalizeAnswer } from '@/lib/utils'
import { assembleTemplate, type RunResult } from '@/runner/harness'
import { pyRunner, usePyStatus } from '@/runner/client'
import { useLevel } from './context'
import { EmptyStage } from './QuizStage'
import { STAGE_META } from './status'

type Kind = 'correct' | 'equivalent' | 'revealed'
interface GState {
  values: Record<number, string>
  status: Record<number, Kind>
  active: number
  done: boolean
}
const emptyState = (): GState => ({ values: {}, status: {}, active: 0, done: false })

const CIRCLED = '①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳'
const circled = (n: number) => CIRCLED[n - 1] ?? `(${n})`

type Seg = { kind: 'text'; text: string } | { kind: 'blank'; n: number }
function parseTemplate(t: string): Seg[][] {
  return t.split('\n').map((line) => {
    const segs: Seg[] = []
    let last = 0
    for (const m of line.matchAll(/\{\{(\d+)\}\}/g)) {
      if (m.index! > last) segs.push({ kind: 'text', text: line.slice(last, m.index) })
      segs.push({ kind: 'blank', n: Number(m[1]) })
      last = m.index! + m[0].length
    }
    if (last < line.length || segs.length === 0) segs.push({ kind: 'text', text: line.slice(last) })
    return segs
  })
}

type Feedback =
  | { type: 'ok'; step: number; kind: Kind }
  | { type: 'fail'; step: number; run?: RunResult; reason?: string }
  | { type: 'final-fail'; run: RunResult }

function RunFailure({ run }: { run: RunResult }) {
  if (run.timedOut) return <p>运行超时（可能写出了死循环），已自动重启运行环境。</p>
  if (run.infraError) return <p>Python 运行时不可用：{run.infraError}。可以先「揭晓答案」继续。</p>
  if (run.compileError)
    return (
      <div>
        <p className="mb-1">代码无法执行：</p>
        <pre className="thin-scroll overflow-x-auto whitespace-pre-wrap rounded-lg bg-surface px-2.5 py-2 font-mono text-xs text-danger">{run.compileError}</pre>
      </div>
    )
  const fails = run.results.map((r, i) => ({ r, i })).filter((x) => !x.r.passed)
  let hiddenNo = 0
  const hiddenIdx = new Map<number, number>()
  run.results.forEach((r, i) => r.hidden && hiddenIdx.set(i, ++hiddenNo))
  return (
    <div className="space-y-1.5">
      <p>
        代入后 {fails.length}/{run.results.length} 个测试未通过：
      </p>
      {fails.slice(0, 3).map(({ r, i }) => (
        <div key={i} className="rounded-lg bg-surface px-2.5 py-2">
          <div className="text-xs font-semibold text-fg">{r.hidden ? `隐藏用例 #${hiddenIdx.get(i)}` : r.name}</div>
          {r.error && !r.hidden && (
            <pre className="thin-scroll mt-1 overflow-x-auto whitespace-pre-wrap font-mono text-xs text-danger">{r.error.split('\n').slice(-3).join('\n')}</pre>
          )}
        </div>
      ))}
    </div>
  )
}

function Exercise({ g, levelId }: { g: GuidedExercise; levelId: string }) {
  const key = draftKey(levelId, 'guided', g.id)
  const setDraft = useDrafts((s) => s.setDraft)
  const clearDraft = useDrafts((s) => s.clearDraft)
  const completeGuided = useProgress((s) => s.completeGuided)
  const everDone = useProgress((s) => s.levels[levelId]?.guidedDone.includes(g.id) ?? false)
  const py = usePyStatus()
  const { next, goStage } = useLevel()

  const [st, setStRaw] = useState<GState>(() => {
    try {
      const raw = useDrafts.getState().drafts[key]
      if (raw) return { ...emptyState(), ...JSON.parse(raw) }
    } catch {}
    return emptyState()
  })
  const setSt = (fn: (s: GState) => GState) =>
    setStRaw((s) => {
      const n = fn(s)
      setDraft(key, JSON.stringify(n))
      return n
    })

  const [checking, setChecking] = useState<false | 'step' | 'final'>(false)
  const [feedback, setFeedback] = useState<Feedback | null>(null)
  const [hint, setHint] = useState(false)
  const [notice, setNotice] = useState('')
  const [gained, setGained] = useState<number | null>(null)
  const inputRef = useRef<HTMLInputElement & HTMLTextAreaElement>(null)
  const userActed = useRef(false)
  const mounted = useRef(true)
  useEffect(() => { mounted.current = true; return () => { mounted.current = false } }, [])

  const lines = useMemo(() => parseTemplate(g.template), [g.template])
  const steps = g.steps
  const stepOfBlank = useMemo(() => new Map(steps.map((s, i) => [s.blank, i])), [steps])
  const allFilled = steps.every((s) => st.status[s.blank])
  const active = Math.min(st.active, steps.length - 1)
  const step = steps[active]

  useEffect(() => {
    pyRunner.preload().catch(() => {})
  }, [])

  useEffect(() => {
    if (!userActed.current) return
    const el = inputRef.current
    if (el) {
      el.focus({ preventScroll: true })
      el.scrollIntoView({ block: 'nearest', inline: 'nearest' })
    }
  }, [active])

  /** 判等价用的拼装：第 k 步及之前用用户答案，之后用参考答案 */
  const fillsFor = (k: number, v: string) =>
    Object.fromEntries(
      steps.map((s, i) => [
        s.blank,
        i === k ? v : i < k && st.status[s.blank] !== 'revealed' ? (st.values[s.blank] ?? s.answer) : s.answer,
      ]),
    )

  const userCode = (values: Record<number, string>, status: Record<number, Kind>) =>
    assembleTemplate(
      g.template,
      Object.fromEntries(steps.map((s) => [s.blank, status[s.blank] === 'revealed' ? s.answer : (values[s.blank] ?? '')])),
    )

  const runFinal = async (values: Record<number, string>, status: Record<number, Kind>) => {
    setChecking('final')
    const r = await pyRunner.run({ code: userCode(values, status), setup: g.setup, tests: g.tests }, 10000)
    if (!mounted.current) return
    setChecking(false)
    if (r.ok) {
      const xp = completeGuided(levelId, g.id)
      setGained(xp)
      setSt((s) => ({ ...s, done: true }))
      setFeedback(null)
    } else setFeedback({ type: 'final-fail', run: r })
  }

  const complete = (k: number, kind: Kind, v: string) => {
    const s = steps[k]
    const values = { ...st.values, [s.blank]: kind === 'revealed' ? s.answer : v }
    const status = { ...st.status, [s.blank]: kind }
    const firstTodo = steps.findIndex((x) => !status[x.blank])
    userActed.current = true
    setSt((cur) => ({ ...cur, values, status, active: firstTodo < 0 ? k : firstTodo }))
    setFeedback({ type: 'ok', step: k, kind })
    setHint(false)
    setNotice('')
    if (firstTodo < 0) void runFinal(values, status)
  }

  const check = async () => {
    if (checking || st.done) return
    const v = st.values[step.blank] ?? ''
    if (!v.trim()) {
      setNotice('先在代码中的输入框里填写这一空')
      inputRef.current?.focus()
      return
    }
    const n = normalizeAnswer(v)
    if ([step.answer, ...(step.accept ?? [])].some((a) => normalizeAnswer(a) === n)) return complete(active, 'correct', v)
    setChecking('step')
    setNotice('')
    const r = await pyRunner.run({ code: assembleTemplate(g.template, fillsFor(active, v)), setup: g.setup, tests: g.tests }, 8000)
    if (!mounted.current) return
    setChecking(false)
    if (r.ok) complete(active, 'equivalent', v)
    else {
      setFeedback({ type: 'fail', step: active, run: r })
      inputRef.current?.focus()
    }
  }

  const reveal = () => complete(active, 'revealed', step.answer)

  const reset = () => {
    clearDraft(key)
    setStRaw(emptyState())
    setFeedback(null)
    setGained(null)
    setHint(false)
    userActed.current = true
  }

  const setValue = (v: string) => {
    setSt((s) => ({ ...s, values: { ...s.values, [step.blank]: v } }))
    setNotice('')
  }

  const editStep = (k: number) => {
    if (checking || st.done) return
    userActed.current = true
    setSt((s) => ({ ...s, active: k }))
    setFeedback(null)
  }

  const activeLine = lines.findIndex((segs) => segs.some((x) => x.kind === 'blank' && x.n === step.blank))
  const multiLine = step.answer.includes('\n')

  /* ---------- 代码骨架 ---------- */
  const code = (
    <div className="thin-scroll overflow-x-auto rounded-2xl border border-border bg-code-bg py-3 text-[13px] leading-[1.9] sm:text-[13.5px]">
      <div className="min-w-max font-mono">
        {lines.map((segs, li) => (
          <div key={li} className={cn('flex pr-4', !st.done && li === activeLine && 'cq-line-current')}>
            <span className="w-10 shrink-0 select-none pr-3 text-right text-[12px] text-subtle tabular-nums">{li + 1}</span>
            <span className="hljs whitespace-pre">
              {segs.map((sg, k) => {
                if (sg.kind === 'text') return <span key={k} dangerouslySetInnerHTML={{ __html: highlight(sg.text) }} />
                const si = stepOfBlank.get(sg.n)
                if (si == null) return <span key={k}>{`{{${sg.n}}}`}</span>
                const s = steps[si]
                const kind = st.status[sg.n]
                const isActive = !st.done && si === active
                if (isActive) {
                  const v = st.values[sg.n] ?? ''
                  const common = {
                    ref: inputRef,
                    value: v,
                    'aria-label': `第 ${si + 1} 空`,
                    'data-blank': sg.n,
                    'data-state': feedback?.type === 'fail' && feedback.step === si ? 'wrong' : undefined,
                    spellCheck: false,
                    autoCapitalize: 'off',
                    autoCorrect: 'off',
                    autoComplete: 'off',
                    placeholder: circled(sg.n),
                    className: 'cq-blank',
                  } as const
                  return multiLine ? (
                    <textarea
                      key={k}
                      {...common}
                      rows={Math.max(2, v.split('\n').length)}
                      style={{ width: `${Math.max(16, ...v.split('\n').map((l) => l.length + 2))}ch`, resize: 'none', verticalAlign: 'top' }}
                      onChange={(e) => setValue(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                          e.preventDefault()
                          void check()
                        } else if (e.key === 'Tab') {
                          e.preventDefault()
                          const el = e.currentTarget
                          const p = el.selectionStart
                          setValue(v.slice(0, p) + '    ' + v.slice(el.selectionEnd))
                          requestAnimationFrame(() => el.setSelectionRange(p + 4, p + 4))
                        }
                      }}
                    />
                  ) : (
                    <input
                      key={k}
                      {...common}
                      enterKeyHint="go"
                      style={{ width: `${Math.max(8, v.length + 2)}ch` }}
                      onChange={(e) => setValue(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.nativeEvent.isComposing) {
                          e.preventDefault()
                          void check()
                        }
                      }}
                    />
                  )
                }
                if (kind) {
                  const shown = kind === 'revealed' ? s.answer : (st.values[sg.n] ?? '')
                  return (
                    <button
                      key={k}
                      type="button"
                      className="cq-blank-done"
                      data-kind={kind}
                      title={st.done ? undefined : '点击修改这一空'}
                      onClick={() => editStep(si)}
                      dangerouslySetInnerHTML={{ __html: highlight(shown) }}
                    />
                  )
                }
                return (
                  <span key={k} className="cq-blank-todo">
                    {circled(sg.n)}
                  </span>
                )
              })}
            </span>
          </div>
        ))}
      </div>
    </div>
  )

  const legend = (
    <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-subtle">
      <span className="inline-flex items-center gap-1">
        <span className="h-2 w-4 rounded-sm border-b-2 border-success bg-success-soft" /> 标准写法
      </span>
      <span className="inline-flex items-center gap-1">
        <span className="h-2 w-4 rounded-sm border-b-2 border-info bg-info-soft" /> 等价写法（测试判定）
      </span>
      <span className="inline-flex items-center gap-1">
        <span className="h-2 w-4 rounded-sm border-b-2 border-warning bg-warning-soft" /> 揭晓的答案
      </span>
    </div>
  )

  /* ---------- 步骤面板 ---------- */
  const okStep = feedback?.type === 'ok' ? steps[feedback.step] : null
  const panel = st.done ? (
    <div className="animate-pop rounded-2xl border border-success/40 bg-success-soft/50 p-5" data-guided-done>
      <div className="flex items-center gap-2 text-base font-bold text-success">
        <CheckCircle2 size={22} /> 跟练完成！全部测试通过
      </div>
      {(gained ?? 0) > 0 && (
        <div className="mt-2 inline-flex items-center gap-1 rounded-full bg-gold-soft px-3 py-1 text-sm font-semibold text-gold">
          <Flame size={14} /> +{gained} XP
        </div>
      )}
      {g.outro && <Markdown className="mt-3 !text-[14.5px]">{g.outro}</Markdown>}
      <div className="mt-4 flex flex-wrap gap-2">
        {next && (
          <Button variant="primary" onClick={() => goStage(next)}>
            下一阶段：{STAGE_META[next].label}
            <ArrowRight size={16} />
          </Button>
        )}
        <Button variant="ghost" icon={<RotateCcw size={15} />} onClick={reset}>
          重新练习
        </Button>
      </div>
    </div>
  ) : (
    <div className="space-y-3">
      {/* 步骤点 */}
      <div className="flex flex-wrap items-center gap-1.5">
        {steps.map((s, i) => {
          const kind = st.status[s.blank]
          return (
            <button
              key={s.blank}
              type="button"
              onClick={() => (kind || i === active ? editStep(i) : undefined)}
              disabled={!kind && i !== active}
              className={cn(
                'inline-flex h-8 min-w-8 items-center justify-center gap-1 rounded-full px-2 text-xs font-semibold transition-colors',
                i === active
                  ? 'bg-accent text-accent-fg shadow-[0_0_0_3px_var(--ring)]'
                  : kind === 'correct'
                    ? 'bg-success-soft text-success'
                    : kind === 'equivalent'
                      ? 'bg-info-soft text-info'
                      : kind === 'revealed'
                        ? 'bg-warning-soft text-warning'
                        : 'bg-surface-2 text-subtle',
              )}
              aria-label={`第 ${i + 1} 步`}
            >
              {kind && i !== active ? <Check size={13} strokeWidth={3} /> : i + 1}
            </button>
          )
        })}
        <span className="ml-auto text-xs text-muted">
          {Object.keys(st.status).length}/{steps.length} 空
        </span>
      </div>

      {okStep && feedback?.type === 'ok' && (
        <div
          className={cn(
            'animate-fade-in rounded-xl border px-3.5 py-3 text-sm',
            feedback.kind === 'revealed' ? 'border-warning/40 bg-warning-soft/60' : 'border-success/40 bg-success-soft/60',
          )}
        >
          <div className={cn('mb-1 flex items-center gap-1.5 font-semibold', feedback.kind === 'revealed' ? 'text-warning' : 'text-success')}>
            {feedback.kind === 'revealed' ? <Eye size={15} /> : <CheckCircle2 size={15} />}
            第 {feedback.step + 1} 步
            {feedback.kind === 'correct' ? '正确' : feedback.kind === 'equivalent' ? '：等价写法 ✓（测试通过）' : '：已揭晓答案'}
          </div>
          {feedback.kind === 'equivalent' && (
            <p className="mb-1 text-xs text-muted">
              参考写法：<code className="font-mono text-fg">{okStep.answer}</code>
            </p>
          )}
          {okStep.explain && <Markdown className="!text-[14px]">{okStep.explain}</Markdown>}
        </div>
      )}

      {checking === 'final' || (allFilled && !st.done) ? (
        <div className="rounded-2xl border border-border bg-surface p-4">
          {checking === 'final' ? (
            <div className="flex items-center gap-2 text-sm text-muted">
              <Loader2 size={16} className="animate-spin text-accent" />
              全部填完，正在用你的代码跑完整测试…
            </div>
          ) : feedback?.type === 'final-fail' ? (
            <div className="text-sm text-fg">
              <div className="mb-2 flex items-center gap-1.5 font-semibold text-danger">
                <XCircle size={16} /> 组合后的完整代码没有通过测试
              </div>
              <div className="text-muted">
                <RunFailure run={feedback.run} />
              </div>
              <p className="mt-2 text-xs text-muted">点击代码中已填写的空可以修改；也可以重新练习。</p>
              <div className="mt-3 flex gap-2">
                <Button size="sm" variant="primary" icon={<Play size={14} />} onClick={() => runFinal(st.values, st.status)}>
                  重新运行
                </Button>
                <Button size="sm" variant="ghost" icon={<RotateCcw size={14} />} onClick={reset}>
                  重新练习
                </Button>
              </div>
            </div>
          ) : (
            <Button variant="primary" icon={<Play size={15} />} onClick={() => runFinal(st.values, st.status)}>
              运行完整测试
            </Button>
          )}
        </div>
      ) : (
        <div className="rounded-2xl border border-accent/40 bg-surface p-4 shadow-sm" data-step={active}>
          <div className="mb-1.5 flex items-center gap-2 text-xs font-semibold text-accent">
            <Sparkles size={14} />第 {active + 1} 步 · 填写空 {circled(step.blank)}
          </div>
          <Markdown className="!text-[14.5px]">{step.instruction}</Markdown>
          {hint && step.hint && (
            <div className="mt-2 flex animate-fade-in gap-2 rounded-lg bg-warning-soft/70 px-3 py-2 text-sm">
              <Lightbulb size={15} className="mt-0.5 shrink-0 text-warning" />
              <Markdown className="!text-sm">{step.hint}</Markdown>
            </div>
          )}
          {feedback?.type === 'fail' && feedback.step === active && (
            <div className="mt-3 animate-shake rounded-xl border border-danger/40 bg-danger-soft/60 px-3 py-2.5 text-sm text-fg">
              <div className="mb-1 flex items-center gap-1.5 font-semibold text-danger">
                <XCircle size={15} /> 不对，再想想
              </div>
              <div className="text-muted">{feedback.run ? <RunFailure run={feedback.run} /> : feedback.reason}</div>
              {step.hint && !hint && <p className="mt-1 text-xs text-muted">可以看看提示。</p>}
            </div>
          )}
          {checking === 'step' && (
            <div className="mt-3 flex items-center gap-2 text-sm text-muted">
              <Loader2 size={15} className="animate-spin text-accent" />
              {py.status === 'loading'
                ? `写法与参考不同，正在加载 Python 运行时以测试判定等价（${Math.round(py.progress * 100)}%）…`
                : '写法与参考不同，正在跑测试判定是否等价…'}
            </div>
          )}
          {notice && <p className="mt-2 text-sm font-medium text-warning">{notice}</p>}
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <Button variant="primary" onClick={check} disabled={!!checking} data-action="check" icon={checking ? <Loader2 size={15} className="animate-spin" /> : <Check size={15} />}>
              检查
            </Button>
            {step.hint && (
              <Button variant="ghost" size="sm" onClick={() => setHint((h) => !h)} icon={<Lightbulb size={14} />}>
                {hint ? '收起提示' : '提示'}
              </Button>
            )}
            <Button variant="ghost" size="sm" onClick={reveal} disabled={!!checking} data-action="reveal" icon={<Eye size={14} />}>
              揭晓答案
            </Button>
            <span className="ml-auto hidden text-[11px] text-subtle md:inline">{multiLine ? '⌘/Ctrl+Enter 检查' : 'Enter 检查'}</span>
          </div>
        </div>
      )}
    </div>
  )

  return (
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1.3fr)_minmax(320px,1fr)] lg:gap-8">
      <div className="min-w-0">
        <Markdown className="mb-4">{g.intro}</Markdown>
        {code}
        {legend}
        {everDone && !st.done && <p className="mt-2 text-xs text-success">你之前已完成过这个跟练，这次是复习。</p>}
      </div>
      <div className="min-w-0 lg:sticky lg:top-[calc(var(--lv-top,56px)+16px)] lg:h-fit">{panel}</div>
    </div>
  )
}

export function GuidedStage() {
  const { levelId, L } = useLevel()
  const [cur, setCur] = useState(0)
  const doneIds = useProgress((s) => s.levels[levelId]?.guidedDone) ?? []
  if (L.guided.length === 0) return <EmptyStage text="本关没有跟练，直接去实战吧" />
  const g = L.guided[Math.min(cur, L.guided.length - 1)]
  return (
    <div className="mx-auto max-w-[1200px] px-4 pb-16 pt-5 sm:px-6 sm:pt-8">
      <div className="mb-4 flex flex-wrap items-center gap-2">
        {L.guided.length > 1 ? (
          L.guided.map((x, i) => (
            <button
              key={x.id}
              type="button"
              onClick={() => setCur(i)}
              className={cn(
                'inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-sm font-medium transition-colors',
                i === cur ? 'border-accent bg-accent-soft text-accent' : 'border-border text-muted hover:text-fg',
              )}
            >
              {doneIds.includes(x.id) && <CheckCircle2 size={14} className="text-success" />}
              {x.title}
            </button>
          ))
        ) : (
          <h2 className="flex items-center gap-2 text-xl font-bold text-fg">
            {g.title}
            {doneIds.includes(g.id) && <CheckCircle2 size={18} className="text-success" />}
          </h2>
        )}
        <span className="ml-auto text-xs text-subtle">完成 +{XP.guided} XP</span>
      </div>
      <Fragment key={g.id}>
        <Exercise g={g} levelId={levelId} />
      </Fragment>
    </div>
  )
}
