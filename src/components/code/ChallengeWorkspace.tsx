/**
 * 【P2 平台-关卡 负责】实战工作台：题面 / 编辑器 / 测试结果 / 提示 / 题解。
 * ⚠️ 接口已冻结：P1 的模拟面试页依赖此签名。
 *
 * 布局：≥1024 左右分栏（左：题目 / 提示 / 题解；右：编辑器 + 结果）；<1024 「题目 / 代码 / 结果」三个标签。
 * 高度：根元素 h-full（父容器需给定高度；否则最小 560px）。
 * 行为：
 *  - 运行 = 只跑可见用例；提交 = 全部用例（attemptChallenge）；全部通过 → passChallenge + 庆祝 + onPassed
 *  - practice 模式草稿自动保存到 useDrafts；interview 模式隐藏提示与题解、不读写草稿、关闭自动补全
 */
import { lazy, Suspense, useCallback, useEffect, useRef, useState, useSyncExternalStore } from 'react'
import { BookOpenCheck, CheckCircle2, Clock, FileText, Flame, Lightbulb, ListChecks, Play, RotateCcw, Send, Code2, Loader2 } from 'lucide-react'
import type { Challenge } from '@/content/types'
import { draftKey, useDrafts, useProgress } from '@/lib/store'
import { challengeXp } from '@/lib/game'
import { cn, fmtDuration } from '@/lib/utils'
import { pyRunner } from '@/runner/client'
import { ConfirmDialog } from '@/components/level/Overlay'
import { celebrate } from '@/components/level/celebrate'
import { ResultsPanel, type RunRecord } from './ResultsPanel'
import { DescPanel, HintsPanel, PyStatusBadge, SolutionPanel, SymbolBar } from './panels'
import type { CodeEditorHandle } from './CodeEditor'
import '@/styles/level.css'

const CodeEditor = lazy(() => import('./CodeEditor'))

export interface ChallengeWorkspaceProps {
  levelId: string
  challenge: Challenge
  /** 模拟面试模式：隐藏提示与题解、显示倒计时 */
  mode?: 'practice' | 'interview'
  interviewId?: string
  /** 剩余秒数（interview 模式由外部计时） */
  onPassed?: (info: { timeSec: number; hintsUsed: number; usedSolution: boolean }) => void
  className?: string
}

const DESKTOP = '(min-width: 1024px)'
function useMedia(q: string) {
  return useSyncExternalStore(
    (cb) => {
      const m = window.matchMedia(q)
      m.addEventListener('change', cb)
      return () => m.removeEventListener('change', cb)
    },
    () => window.matchMedia(q).matches,
  )
}

type LeftTab = 'desc' | 'hints' | 'solution'
type MobileTab = 'desc' | 'code' | 'result'

function EditorFallback() {
  return (
    <div className="flex h-full items-center justify-center gap-2 bg-code-bg text-sm text-muted">
      <Loader2 size={16} className="animate-spin" /> 加载编辑器…
    </div>
  )
}

export function ChallengeWorkspace({ levelId, challenge: c, mode = 'practice', interviewId, onPassed, className }: ChallengeWorkspaceProps) {
  const interview = mode === 'interview'
  const desktop = useMedia(DESKTOP)
  const dkey = interview ? `mock::${interviewId}::${levelId}::${c.id}` : draftKey(levelId, 'challenge', c.id)
  const fontSize = useProgress((s) => s.settings.editorFontSize)
  const record = useProgress((s) => interview ? undefined : s.levels[levelId]?.challenges[c.id])
  const takeChallengeHint = useProgress((s) => s.takeChallengeHint)
  const viewChallengeSolution = useProgress((s) => s.viewChallengeSolution)
  const attemptChallenge = useProgress((s) => s.attemptChallenge)
  const passChallenge = useProgress((s) => s.passChallenge)
  const setDraft = useDrafts((s) => s.setDraft)
  const clearDraft = useDrafts((s) => s.clearDraft)

  const [code, setCode] = useState(() => useDrafts.getState().drafts[dkey] ?? c.starter)
  const [running, setRunning] = useState<false | 'run' | 'submit'>(false)
  const [last, setLast] = useState<RunRecord | null>(null)
  const [errorLine, setErrorLine] = useState<number | null>(null)
  const [leftTab, setLeftTab] = useState<LeftTab>('desc')
  const [mTab, setMTab] = useState<MobileTab>('desc')
  const [localHints, setLocalHints] = useState(0)
  const [localSolution, setLocalSolution] = useState(false)
  const [confirm, setConfirm] = useState<null | 'reset' | 'solution'>(null)
  const [elapsed, setElapsed] = useState(0)
  const [passedNow, setPassedNow] = useState<null | { xp: number; timeSec: number }>(null)
  const [saved, setSaved] = useState(false)
  const editorRef = useRef<CodeEditorHandle>(null)

  const passed = !!record?.passed
  const hintsShown = Math.max(localHints, record?.hintsUsed ?? 0)
  const solutionUnlocked = !interview && (localSolution || !!record?.usedSolution)

  // 题目切换时重置
  const [ident, setIdent] = useState(c)
  if (ident !== c) {
    setIdent(c)
    setCode(useDrafts.getState().drafts[dkey] ?? c.starter)
    setLast(null)
    setErrorLine(null)
    setLocalHints(0)
    setLocalSolution(false)
    setElapsed(0)
    setPassedNow(null)
    setLeftTab('desc')
    setMTab('desc')
  }

  // 预热 Python
  useEffect(() => {
    pyRunner.preload().catch(() => {})
  }, [])

  // 计时（页面可见时计数；本次通过后停止）
  useEffect(() => {
    if (passedNow) return
    const t = setInterval(() => {
      if (document.visibilityState === 'visible') setElapsed((e) => e + 1)
    }, 1000)
    return () => clearInterval(t)
  }, [passedNow, c.id])

  // 面试草稿按场次隔离，切题、刷新时也保留。
  useEffect(() => {
    const t = setTimeout(() => {
      if (code === c.starter) clearDraft(dkey)
      else setDraft(dkey, code)
    }, 400)
    return () => clearTimeout(t)
  }, [code, interview, dkey, c.starter, setDraft, clearDraft])

  const mounted = useRef(true)
  useEffect(() => { mounted.current = true; return () => { mounted.current = false } }, [])

  const busyRef = useRef(false)
  const doRun = useCallback(
    async (kind: 'run' | 'submit') => {
      if (busyRef.current) return
      busyRef.current = true
      setRunning(kind)
      setErrorLine(null)
      if (!desktop) setMTab('result')
      const tests = kind === 'run' ? c.tests.filter((t) => !t.hidden) : c.tests
      if (kind === 'submit' && !interview) attemptChallenge(levelId, c.id)
      const result = await pyRunner.run({ code, setup: c.setup, tests }, 10000)
      if (!mounted.current) return
      busyRef.current = false
      setRunning(false)
      setLast({ kind, result, tests, at: Date.now() })
      setErrorLine(result.errorLine ?? result.results.find((r) => r.errorLine)?.errorLine ?? null)
      if (kind === 'submit' && result.ok) {
        if (interview) {
          setPassedNow({ xp: 0, timeSec: elapsed })
          onPassed?.({ timeSec: elapsed, hintsUsed: 0, usedSolution: false })
          return
        }
        const before = useProgress.getState().levels[levelId]?.challenges[c.id]
        const hintsUsed = before?.hintsUsed ?? 0
        const usedSolution = before?.usedSolution ?? false
        const wasPassed = !!before?.passed
        const xp = passChallenge(levelId, c.id, c.difficulty, elapsed)
        setPassedNow({ xp: wasPassed ? 0 : xp, timeSec: elapsed })
        void celebrate(wasPassed ? 'small' : 'big')
        onPassed?.({ timeSec: elapsed, hintsUsed, usedSolution })
      }
    },
    [code, c, desktop, levelId, attemptChallenge, passChallenge, elapsed, onPassed, interview],
  )

  const revealHint = () => {
    if (hintsShown >= c.hints.length) return
    if (!passed) takeChallengeHint(levelId, c.id)
    setLocalHints(hintsShown + 1)
  }

  const requestSolution = () => {
    if (passed) {
      viewChallengeSolution(levelId, c.id)
      setLocalSolution(true)
    } else setConfirm('solution')
  }

  const doReset = () => {
    setCode(c.starter)
    clearDraft(dkey)
    setErrorLine(null)
    setConfirm(null)
  }

  const jump = (line: number) => {
    setErrorLine(null)
    requestAnimationFrame(() => setErrorLine(line))
    if (!desktop) setMTab('code')
  }

  const save = () => {
    setDraft(dkey, code)
    setSaved(true)
    setTimeout(() => setSaved(false), 1200)
  }

  const overTime = !!c.timeLimitMin && elapsed > c.timeLimitMin * 60
  const xpPreview = challengeXp(c.difficulty, hintsShown, !!record?.usedSolution || localSolution)

  /* ---------------- 片段 ---------------- */
  const timer = (
    <span
      className={cn('inline-flex items-center gap-1 font-mono text-xs tabular-nums', overTime ? 'text-warning' : 'text-muted')}
      title={c.timeLimitMin ? `建议用时 ${c.timeLimitMin} 分钟` : '用时'}
      data-timer
    >
      <Clock size={12} />
      {fmtDuration(passedNow ? passedNow.timeSec : elapsed)}
      {c.timeLimitMin ? <span className="text-subtle">/{c.timeLimitMin}:00</span> : null}
    </span>
  )

  const runButtons = (
    <>
      <button
        type="button"
        onClick={() => doRun('run')}
        disabled={!!running}
        data-action="run"
        title="运行可见用例（⌘/Ctrl+Enter）"
        className="inline-flex h-9 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-xl border border-border bg-surface-2 px-3 text-sm font-medium text-fg transition-colors hover:bg-surface-3 disabled:opacity-50 max-lg:flex-1 max-lg:justify-center max-lg:h-11"
      >
        {running === 'run' ? <Loader2 size={15} className="animate-spin" /> : <Play size={15} />}
        运行
      </button>
      <button
        type="button"
        onClick={() => doRun('submit')}
        disabled={!!running}
        data-action="submit"
        title="提交：运行全部用例（⌘/Ctrl+Shift+Enter）"
        className="inline-flex h-9 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-xl bg-success px-3.5 text-sm font-semibold text-white shadow-sm transition-[filter] hover:brightness-110 disabled:opacity-50 dark:text-bg max-lg:flex-1 max-lg:justify-center max-lg:h-11"
      >
        {running === 'submit' ? <Loader2 size={15} className="animate-spin" /> : <Send size={15} />}
        提交
      </button>
    </>
  )

  const passedBanner =
    passedNow && last?.kind === 'submit' && last.result.ok ? (
      <div className="animate-pop rounded-2xl border border-success/40 bg-success-soft/60 p-4" data-passed-banner>
        <div className="flex items-center gap-2 text-base font-bold text-success">
          <CheckCircle2 size={22} /> 全部用例通过！
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-2 text-sm">
          {passedNow.xp > 0 && (
            <span className="inline-flex items-center gap-1 rounded-full bg-gold-soft px-2.5 py-0.5 font-semibold text-gold">
              <Flame size={13} /> +{passedNow.xp} XP
            </span>
          )}
          <span className="text-muted">用时 {fmtDuration(passedNow.timeSec)}</span>
          {!interview && !solutionUnlocked && (
            <button type="button" className="font-medium text-accent hover:underline" onClick={() => { requestSolution(); setLeftTab('solution'); setMTab('desc') }}>
              对比参考题解 →
            </button>
          )}
        </div>
      </div>
    ) : null

  const editor = (
    <div className="cq-editor relative min-h-0 flex-1" data-editor>
      <Suspense fallback={<EditorFallback />}>
        <CodeEditor
          ref={editorRef}
          value={code}
          onChange={(value) => { setCode(value); setDraft(dkey, value) }}
          fontSize={fontSize}
          errorLine={errorLine}
          onRun={() => doRun('run')}
          onSubmit={() => doRun('submit')}
          onSave={save}
          autocomplete={!interview}
          className="h-full"
        />
      </Suspense>
      {saved && <span className="absolute right-3 top-2 z-10 animate-fade-in rounded-md bg-surface px-2 py-0.5 text-xs text-success shadow-sm">草稿已保存</span>}
    </div>
  )

  const leftTabs: { id: LeftTab; label: string; icon: typeof FileText; badge?: string }[] = [
    { id: 'desc', label: '题目', icon: FileText },
    ...(!interview
      ? [
          { id: 'hints' as const, label: '提示', icon: Lightbulb, badge: c.hints.length ? `${hintsShown}/${c.hints.length}` : undefined },
          { id: 'solution' as const, label: '题解', icon: BookOpenCheck },
        ]
      : []),
  ]

  const leftContent = (tab: LeftTab) =>
    tab === 'desc' ? (
      <DescPanel c={c} attempts={record?.attempts} />
    ) : tab === 'hints' ? (
      <HintsPanel c={c} shown={hintsShown} passed={passed} onReveal={revealHint} />
    ) : (
      <SolutionPanel c={c} unlocked={solutionUnlocked} passed={passed} onRequest={requestSolution} />
    )

  const dialogs = (
    <>
      <ConfirmDialog
        open={confirm === 'reset'}
        title="重置为初始代码？"
        tone="danger"
        confirmLabel="重置"
        onConfirm={doReset}
        onCancel={() => setConfirm(null)}
      >
        当前编辑器中的代码会被清空，此操作无法撤销。
      </ConfirmDialog>
      <ConfirmDialog
        open={confirm === 'solution'}
        title="确定现在查看题解？"
        confirmLabel="查看题解"
        onConfirm={() => {
          viewChallengeSolution(levelId, c.id)
          setLocalSolution(true)
          setConfirm(null)
        }}
        onCancel={() => setConfirm(null)}
      >
        通过前查看题解：本题奖励减半（当前可得 {xpPreview} XP → {challengeXp(c.difficulty, hintsShown, true)} XP），且本关无法获得 ★★★。
        建议先看看提示。
      </ConfirmDialog>
    </>
  )

  /* ---------------- 桌面：左右分栏 ---------------- */
  if (desktop) {
    return (
      <div
        className={cn('flex h-full min-h-[560px] overflow-hidden rounded-2xl border border-border bg-surface shadow-sm', className)}
        data-workspace={c.id}
      >
        <section className="flex w-[42%] min-w-[340px] max-w-[560px] flex-col border-r border-border">
          <div className="flex items-center gap-1 border-b border-border px-2 py-1.5">
            {leftTabs.map((t) => (
              <button
                key={t.id}
                type="button"
                onClick={() => setLeftTab(t.id)}
                data-tab={t.id}
                className={cn(
                  'inline-flex h-8 items-center gap-1.5 rounded-lg px-3 text-[13px] font-medium transition-colors',
                  leftTab === t.id ? 'bg-accent-soft text-accent' : 'text-muted hover:bg-surface-2 hover:text-fg',
                )}
              >
                <t.icon size={14} />
                {t.label}
                {t.badge && <span className="text-[11px] opacity-70">{t.badge}</span>}
              </button>
            ))}
            {passed && (
              <span className="ml-auto inline-flex items-center gap-1 pr-2 text-xs font-medium text-success">
                <CheckCircle2 size={14} /> 已通过
              </span>
            )}
          </div>
          <div className="thin-scroll min-h-0 flex-1 overflow-y-auto px-5 py-4">{leftContent(leftTab)}</div>
        </section>
        <section className="flex min-w-0 flex-1 flex-col">
          <div className="flex flex-wrap items-center gap-2 border-b border-border px-3 py-1.5">
            <PyStatusBadge className="min-w-0" />
            <span className="ml-auto" />
            {timer}
            <button
              type="button"
              onClick={() => setConfirm('reset')}
              className="inline-flex h-8 shrink-0 items-center gap-1 whitespace-nowrap rounded-lg px-2 text-xs text-muted hover:bg-surface-2 hover:text-fg"
              title="重置为初始代码"
              data-action="reset"
            >
              <RotateCcw size={14} /> 重置
            </button>
            {runButtons}
          </div>
          <div className="flex min-h-0 flex-1 flex-col">
            <div className="flex min-h-0 flex-[3] flex-col">{editor}</div>
            <div className="thin-scroll min-h-[160px] flex-[2] overflow-y-auto border-t border-border bg-surface" data-results>
              <ResultsPanel record={last} running={running} onJump={jump} passedBanner={passedBanner} className="min-h-full" />
            </div>
          </div>
        </section>
        {dialogs}
      </div>
    )
  }

  /* ---------------- 手机 / 平板：三标签 ---------------- */
  const resultBadge = last && last.result.results.length > 0 ? `${last.result.results.filter((r) => r.passed).length}/${last.result.results.length}` : null
  return (
    <div className={cn('flex h-full min-h-[480px] flex-col overflow-hidden bg-surface', className)} data-workspace={c.id}>
      <div className="flex items-center gap-1 border-b border-border px-2 py-1.5">
        {(
          [
            { id: 'desc', label: '题目', icon: FileText },
            { id: 'code', label: '代码', icon: Code2 },
            { id: 'result', label: '结果', icon: ListChecks },
          ] as const
        ).map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => setMTab(t.id)}
            data-mtab={t.id}
            className={cn(
              'inline-flex h-9 flex-1 items-center justify-center gap-1.5 rounded-lg text-[13px] font-medium transition-colors',
              mTab === t.id ? 'bg-accent-soft text-accent' : 'text-muted',
            )}
          >
            <t.icon size={15} />
            {t.label}
            {t.id === 'result' && resultBadge && (
              <span className={cn('rounded-full px-1.5 text-[10px] font-bold', last!.result.ok ? 'bg-success text-surface' : 'bg-danger text-surface')}>
                {resultBadge}
              </span>
            )}
            {t.id === 'result' && running && <Loader2 size={12} className="animate-spin" />}
          </button>
        ))}
      </div>
      <div className="flex min-h-0 flex-1 flex-col">
        {mTab === 'desc' && (
          <div className="thin-scroll min-h-0 flex-1 space-y-6 overflow-y-auto px-4 py-4">
            {leftContent('desc')}
            {!interview && (
              <>
                <section>
                  <h3 className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-fg">
                    <Lightbulb size={15} className="text-warning" /> 提示
                  </h3>
                  {leftContent('hints')}
                </section>
                <section>
                  <h3 className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-fg">
                    <BookOpenCheck size={15} className="text-accent" /> 题解
                  </h3>
                  {leftContent('solution')}
                </section>
              </>
            )}
          </div>
        )}
        {mTab === 'code' && (
          <>
            <SymbolBar
              onInsert={(s) => editorRef.current?.insert(s)}
              onIndent={(d) => editorRef.current?.indent(d)}
              onUndo={() => editorRef.current?.undo()}
            />
            {editor}
          </>
        )}
        {mTab === 'result' && (
          <div className="thin-scroll min-h-0 flex-1 overflow-y-auto" data-results>
            <ResultsPanel record={last} running={running} onJump={jump} passedBanner={passedBanner} className="min-h-full" />
          </div>
        )}
      </div>
      <div className="flex items-center gap-2 border-t border-border bg-surface px-3 py-2 pb-[calc(0.5rem+env(safe-area-inset-bottom))]">
        <div className="flex min-w-0 flex-col gap-0.5">
          {timer}
          <PyStatusBadge className="max-w-[9rem]" />
        </div>
        <button
          type="button"
          onClick={() => setConfirm('reset')}
          className="inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-xl text-muted hover:bg-surface-2"
          aria-label="重置为初始代码"
          data-action="reset"
        >
          <RotateCcw size={17} />
        </button>
        {runButtons}
      </div>
      {dialogs}
    </div>
  )
}
export default ChallengeWorkspace
