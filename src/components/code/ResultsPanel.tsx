import { useState } from 'react'
import { AlertTriangle, CheckCircle2, ChevronRight, Clock, EyeOff, Loader2, PlayCircle, RotateCcw, Terminal, TimerOff, XCircle } from 'lucide-react'
import type { TestCase } from '@/content/types'
import type { RunResult, TestResult } from '@/runner/harness'
import { pyRunner, usePyStatus } from '@/runner/client'
import { CodeBlock } from '@/components/CodeBlock'
import { Button } from '@/components/ui'
import { cn } from '@/lib/utils'

export interface RunRecord {
  kind: 'run' | 'submit'
  result: RunResult & { cancelled?: boolean }
  tests: TestCase[]
  at: number
}

const fmtMs = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${ms < 10 ? ms.toFixed(1) : Math.round(ms)}ms`)

function Pre({ children, tone = 'muted', label }: { children: string; tone?: 'danger' | 'muted'; label?: React.ReactNode }) {
  return (
    <div>
      {label && <div className="mb-1 text-[11px] font-semibold text-subtle">{label}</div>}
      <pre
        className={cn(
          'thin-scroll max-h-56 overflow-auto whitespace-pre-wrap break-words rounded-lg border px-3 py-2 font-mono text-[12.5px] leading-relaxed',
          tone === 'danger' ? 'border-danger/30 bg-danger-soft/50 text-danger' : 'border-border bg-surface-2 text-fg',
        )}
      >
        {children}
      </pre>
    </div>
  )
}

function TestRow({ t, idx, hiddenNo, test, onJump }: { t: TestResult; idx: number; hiddenNo?: number; test?: TestCase; onJump?: (line: number) => void }) {
  const [open, setOpen] = useState(!t.passed && !t.hidden)
  const [showCode, setShowCode] = useState(false)
  const expandable = !t.hidden && (!!t.error || !!t.stdout || !!test)
  return (
    <li className={cn('rounded-xl border', t.passed ? 'border-border bg-surface' : 'border-danger/35 bg-danger-soft/25')} data-test={idx} data-passed={t.passed}>
      <button
        type="button"
        disabled={!expandable && !(t.hidden && t.error)}
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2.5 px-3 py-2.5 text-left text-sm disabled:cursor-default"
      >
        {t.passed ? <CheckCircle2 size={17} className="shrink-0 text-success" /> : <XCircle size={17} className="shrink-0 text-danger" />}
        <span className="min-w-0 flex-1 truncate font-medium text-fg">
          {t.hidden ? (
            <span className="inline-flex items-center gap-1.5">
              <EyeOff size={13} className="text-subtle" />
              隐藏用例 #{hiddenNo}
            </span>
          ) : (
            t.name
          )}
        </span>
        <span className="shrink-0 font-mono text-[11px] text-subtle tabular-nums">{fmtMs(t.ms)}</span>
        {(expandable || (t.hidden && t.error)) && (
          <ChevronRight size={15} className={cn('shrink-0 text-subtle transition-transform', open && 'rotate-90')} />
        )}
      </button>
      {open && (
        <div className="space-y-2 border-t border-border/70 px-3 py-2.5">
          {t.error && (
            <>
              <Pre tone="danger" label={t.hidden ? '错误信息（隐藏用例不展示输入）' : '错误信息'}>
                {t.error}
              </Pre>
              {t.errorLine && onJump && (
                <button type="button" onClick={() => onJump(t.errorLine!)} className="text-xs font-medium text-accent hover:underline">
                  在编辑器中定位第 {t.errorLine} 行 →
                </button>
              )}
            </>
          )}
          {!t.hidden && t.stdout && <Pre label="print 输出">{t.stdout}</Pre>}
          {test && !t.hidden && (
            <div>
              <button type="button" onClick={() => setShowCode((s) => !s)} className="text-xs font-medium text-muted hover:text-fg">
                {showCode ? '收起用例代码' : '查看用例代码'}
              </button>
              {showCode && <CodeBlock code={test.code} className="mt-1.5" />}
            </div>
          )}
        </div>
      )}
    </li>
  )
}

export function ResultsPanel({
  record,
  running,
  onJump,
  passedBanner,
  className,
}: {
  record: RunRecord | null
  running: false | 'run' | 'submit'
  onJump?: (line: number) => void
  passedBanner?: React.ReactNode
  className?: string
}) {
  const py = usePyStatus()

  if (running) {
    return (
      <div className={cn('flex flex-col items-center justify-center gap-3 p-6 text-center text-sm text-muted', className)} aria-live="polite">
        <Loader2 size={26} className="animate-spin text-accent" />
        {py.status === 'loading' ? (
          <>
            <div>{py.message || '加载 Python 运行时…'}</div>
            <div className="h-1.5 w-48 overflow-hidden rounded-full bg-surface-3">
              <div className="h-full rounded-full bg-accent transition-[width] duration-300" style={{ width: `${Math.round(py.progress * 100)}%` }} />
            </div>
            <div className="text-xs text-subtle">首次运行需要加载约 16MB 的 Python 运行时，之后会快很多</div>
          </>
        ) : (
          <div>{running === 'run' ? '正在运行可见用例…' : '正在运行全部用例（含隐藏用例）…'}</div>
        )}
        <Button size="sm" variant="ghost" onClick={() => pyRunner.cancel()} className={py.busy ? '' : 'invisible'}>
          停止运行
        </Button>
      </div>
    )
  }

  if (!record) {
    return (
      <div className={cn('flex flex-col items-center justify-center gap-2 p-6 text-center text-sm text-muted', className)}>
        <PlayCircle size={28} className="text-subtle" />
        <p>
          <b className="text-fg">运行</b> 只跑可见用例；<b className="text-fg">提交</b> 跑全部用例（含隐藏用例）。
        </p>
        <p className="text-xs text-subtle">快捷键：⌘/Ctrl + Enter 运行 · ⌘/Ctrl + Shift + Enter 提交</p>
      </div>
    )
  }

  const r = record.result
  const passed = r.results.filter((t) => t.passed).length
  let hiddenNo = 0
  const hiddenMap = new Map<number, number>()
  r.results.forEach((t, i) => t.hidden && hiddenMap.set(i, ++hiddenNo))

  return (
    <div className={cn('space-y-3 p-3 sm:p-4', className)} data-run-kind={record.kind} data-run-ok={r.ok} data-run-at={record.at}>
      {passedBanner}
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold text-fg">{record.kind === 'run' ? '运行结果（可见用例）' : '提交结果（全部用例）'}</span>
        {r.results.length > 0 && (
          <span
            className={cn(
              'rounded-full px-2 py-0.5 text-xs font-semibold tabular-nums',
              r.ok ? 'bg-success-soft text-success' : 'bg-danger-soft text-danger',
            )}
            data-summary
          >
            通过 {passed}/{record.tests.length}
          </span>
        )}
        {r.durationMs != null && (
          <span className="ml-auto inline-flex items-center gap-1 text-xs text-subtle">
            <Clock size={12} />
            {fmtMs(r.durationMs)}
          </span>
        )}
      </div>

      {r.timedOut && (
        <div className="flex gap-2.5 rounded-xl border border-warning/40 bg-warning-soft/60 px-3 py-2.5 text-sm text-fg" data-timeout>
          <TimerOff size={18} className="mt-0.5 shrink-0 text-warning" />
          <div>
            <div className="font-semibold">运行超时（超过 10 秒）</div>
            <div className="text-muted">可能写出了死循环或复杂度过高。已保留 {r.results.length} 个完成用例的结果，剩余用例未完成。运行环境已自动重启，修改后可再次运行。</div>
          </div>
        </div>
      )}
      {r.cancelled && <div className="rounded-xl bg-surface-2 px-3 py-2.5 text-sm text-muted">已停止运行，运行环境已重启。</div>}
      {r.infraError && (
        <div className="rounded-xl border border-danger/40 bg-danger-soft/50 px-3 py-2.5 text-sm text-fg">
          <div className="flex items-center gap-1.5 font-semibold text-danger">
            <AlertTriangle size={16} /> Python 运行时暂时不可用
          </div>
          <div className="mt-1 break-all text-xs text-muted">{r.infraError}</div>
          <Button size="sm" variant="secondary" className="mt-2" icon={<RotateCcw size={14} />} onClick={() => pyRunner.preload().catch(() => {})}>
            重新加载
          </Button>
        </div>
      )}
      {r.compileError && (
        <div className="space-y-1.5">
          <div className="flex items-center gap-1.5 text-sm font-semibold text-danger">
            <XCircle size={16} /> 代码无法执行
          </div>
          <Pre tone="danger">{r.compileError}</Pre>
          {r.errorLine && onJump && (
            <button type="button" onClick={() => onJump(r.errorLine!)} className="text-xs font-medium text-accent hover:underline">
              在编辑器中定位第 {r.errorLine} 行 →
            </button>
          )}
        </div>
      )}
      {r.stdout && <Pre label={<span className="inline-flex items-center gap-1"><Terminal size={11} /> 顶层 print 输出</span>}>{r.stdout}</Pre>}
      {!r.compileError && !r.timedOut && !r.infraError && !r.cancelled && r.results.length === 0 && (
        <div className="rounded-xl bg-surface-2 px-3 py-2.5 text-sm text-muted">代码可以正常执行。本题没有可见用例，请直接「提交」跑全部测试。</div>
      )}
      {r.results.length > 0 && (
        <ul className="space-y-1.5">
          {r.results.map((t, i) => (
            <TestRow key={`${record.at}-${i}`} t={t} idx={i} hiddenNo={hiddenMap.get(i)} test={record.tests[i]} onJump={onJump} />
          ))}
        </ul>
      )}
    </div>
  )
}
