import { useEffect, useState } from 'react'
import { AlertTriangle, BookOpenCheck, ChevronRight, Clock, Lightbulb, Lock, Loader2, MessageSquareQuote, Square } from 'lucide-react'
import type { Challenge } from '@/content/types'
import { Markdown } from '@/components/Markdown'
import { CodeBlock } from '@/components/CodeBlock'
import { Badge, Button, Segmented } from '@/components/ui'
import { pyRunner, usePyStatus } from '@/runner/client'
import { cn } from '@/lib/utils'

export const DIFF_META = {
  easy: { label: '简单', tone: 'success' as const },
  medium: { label: '中等', tone: 'warning' as const },
  hard: { label: '困难', tone: 'danger' as const },
}

/** 题面 */
export function DescPanel({ c, attempts }: { c: Challenge; attempts?: number }) {
  const visible = c.tests.filter((t) => !t.hidden)
  const hidden = c.tests.length - visible.length
  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-bold leading-snug text-fg">{c.title}</h2>
        <div className="mt-2 flex flex-wrap items-center gap-1.5">
          <Badge tone={DIFF_META[c.difficulty].tone}>{DIFF_META[c.difficulty].label}</Badge>
          {c.source && <Badge>{c.source}</Badge>}
          {c.timeLimitMin && (
            <Badge icon={<Clock size={11} />}>建议 {c.timeLimitMin} 分钟</Badge>
          )}
          {!!attempts && <span className="text-xs text-subtle">已提交 {attempts} 次</span>}
        </div>
      </div>
      <Markdown className="!text-[14.5px]">{c.description}</Markdown>
      <div className="rounded-xl border border-border bg-surface-2/60 px-3 py-2.5 text-xs text-muted">
        测试：{visible.length} 个可见用例{hidden ? ` + ${hidden} 个隐藏用例（提交时运行）` : ''}。运行环境 Python 3.14 + NumPy（Pyodide，浏览器内执行）。
      </div>
      {visible.length > 0 && (
        <details className="group rounded-xl border border-border">
          <summary className="flex cursor-pointer list-none items-center gap-1.5 px-3 py-2 text-sm font-medium text-muted hover:text-fg">
            <ChevronRight size={15} className="transition-transform group-open:rotate-90" />
            可见用例代码
          </summary>
          <div className="space-y-2 border-t border-border p-3">
            {visible.map((t) => (
              <div key={t.name}>
                <div className="mb-1 text-xs font-semibold text-muted">{t.name}</div>
                <CodeBlock code={t.code} />
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  )
}

/** 渐进提示 */
export function HintsPanel({ c, shown, passed, onReveal }: { c: Challenge; shown: number; passed: boolean; onReveal: () => void }) {
  if (c.hints.length === 0) return <p className="text-sm text-muted">本题没有提示。</p>
  return (
    <div className="space-y-3">
      <p className="text-xs text-muted">
        {passed ? '你已通过本题，查看提示不影响奖励。' : '提示由浅入深。每查看一条，本题通过奖励 −10%（最低 50%）。'}
      </p>
      <ol className="space-y-2">
        {c.hints.slice(0, shown).map((h, i) => (
          <li key={i} className="flex animate-fade-in gap-2.5 rounded-xl border border-warning/35 bg-warning-soft/50 px-3 py-2.5">
            <span className="mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-warning text-[11px] font-bold text-surface">
              {i + 1}
            </span>
            <Markdown className="min-w-0 !text-[14px]">{h}</Markdown>
          </li>
        ))}
      </ol>
      {shown < c.hints.length ? (
        <Button variant="secondary" icon={<Lightbulb size={15} className="text-warning" />} onClick={onReveal} data-action="hint">
          {shown === 0 ? '查看第 1 条提示' : `查看下一条提示（${shown + 1}/${c.hints.length}）`}
        </Button>
      ) : (
        <p className="text-xs text-subtle">已显示全部 {c.hints.length} 条提示。还卡住的话可以看题解。</p>
      )}
    </div>
  )
}

/** 题解：讲解 + 参考解 / solutionAlt 标签页 + 复杂度 + 追问 */
export function SolutionPanel({ c, unlocked, passed, onRequest }: { c: Challenge; unlocked: boolean; passed: boolean; onRequest: () => void }) {
  const tabs = [{ label: '参考解（Python）', code: c.solution }, ...(c.solutionAlt ?? [])]
  const [tab, setTab] = useState(0)
  if (!unlocked) {
    return (
      <div className="flex flex-col items-center gap-3 rounded-2xl border border-dashed border-border-strong px-5 py-8 text-center">
        <span className="inline-flex h-11 w-11 items-center justify-center rounded-full bg-surface-2 text-muted">
          <Lock size={20} />
        </span>
        <div className="text-sm text-muted">
          {passed ? (
            '你已通过本题，可以放心查看题解对比思路。'
          ) : (
            <>
              建议先独立思考、用完提示再看题解。
              <br />
              <span className="text-warning">通过前查看题解：奖励减半，且本关无法获得 ★★★。</span>
            </>
          )}
        </div>
        <Button variant={passed ? 'primary' : 'secondary'} icon={<BookOpenCheck size={16} />} onClick={onRequest} data-action="solution">
          查看题解
        </Button>
      </div>
    )
  }
  return (
    <div className="space-y-4" data-solution>
      <Markdown className="!text-[14.5px]">{c.explanation}</Markdown>
      {c.complexity && (
        <div className="flex flex-wrap gap-2 text-xs">
          <span className="rounded-lg bg-surface-2 px-2.5 py-1 text-muted">
            时间 <b className="font-mono text-fg">{c.complexity.time}</b>
          </span>
          <span className="rounded-lg bg-surface-2 px-2.5 py-1 text-muted">
            空间 <b className="font-mono text-fg">{c.complexity.space}</b>
          </span>
        </div>
      )}
      <div>
        {tabs.length > 1 && (
          <Segmented size="sm" className="mb-2" value={tab} onChange={setTab} options={tabs.map((t, i) => ({ value: i, label: t.label }))} />
        )}
        <CodeBlock code={tabs[Math.min(tab, tabs.length - 1)].code} lineNumbers />
      </div>
      {c.followUps && c.followUps.length > 0 && (
        <div>
          <h4 className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-fg">
            <MessageSquareQuote size={15} className="text-accent" /> 面试官追问
          </h4>
          <div className="space-y-2">
            {c.followUps.map((f) => (
              <details key={f.id} className="group rounded-xl border border-border bg-surface">
                <summary className="flex cursor-pointer list-none items-start gap-2 px-3 py-2.5 text-sm hover:bg-surface-2">
                  <ChevronRight size={15} className="mt-1 shrink-0 text-subtle transition-transform group-open:rotate-90" />
                  <Markdown className="min-w-0 !text-[14px] font-medium">{f.q}</Markdown>
                </summary>
                <div className="border-t border-border px-3 py-2.5">
                  <Markdown className="!text-[14px] text-muted">{f.a}</Markdown>
                </div>
              </details>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

/** Python 运行时状态徽标（加载进度 / 就绪 / 失败 / 运行中可停止） */
export function PyStatusBadge({ className }: { className?: string }) {
  const py = usePyStatus()
  const [now, setNow] = useState(() => performance.now())
  useEffect(() => {
    if (py.status !== 'loading' && !py.busy) return
    const t = setInterval(() => setNow(performance.now()), 500)
    return () => clearInterval(t)
  }, [py.status, py.busy])
  const waited = Math.round((now - py.loadStartedAt) / 1000)

  if (py.busy && py.status === 'ready')
    return (
      <span className={cn('inline-flex items-center gap-1.5 text-xs text-muted', className)}>
        <Loader2 size={13} className="animate-spin text-accent" />
        运行中
        <button
          type="button"
          onClick={() => pyRunner.cancel()}
          className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 font-medium text-danger hover:bg-danger-soft"
          title="停止运行"
        >
          <Square size={10} fill="currentColor" /> 停止
        </button>
      </span>
    )
  if (py.status === 'loading')
    return (
      <span className={cn('inline-flex min-w-0 items-center gap-2 text-xs text-muted', className)} title={py.message} data-py="loading">
        <Loader2 size={13} className="shrink-0 animate-spin text-accent" />
        <span className="hidden truncate sm:inline">{waited > 25 ? '网络较慢，仍在加载 Python…' : py.message || '加载 Python…'}</span>
        <span className="sm:hidden">Python</span>
        <span className="h-1 w-12 shrink-0 overflow-hidden rounded-full bg-surface-3">
          <span className="block h-full rounded-full bg-accent transition-[width] duration-300" style={{ width: `${Math.round(py.progress * 100)}%` }} />
        </span>
        <span className="tabular-nums">{Math.round(py.progress * 100)}%</span>
      </span>
    )
  if (py.status === 'error')
    return (
      <button
        type="button"
        onClick={() => pyRunner.preload().catch(() => {})}
        className={cn('inline-flex items-center gap-1.5 rounded-md px-1.5 py-0.5 text-xs font-medium text-danger hover:bg-danger-soft', className)}
        title={py.message}
      >
        <AlertTriangle size={13} /> Python 加载失败 · 重试
      </button>
    )
  if (py.status === 'ready')
    return (
      <span className={cn('inline-flex items-center gap-1.5 text-xs text-muted', className)} data-py="ready">
        <span className="h-2 w-2 rounded-full bg-success" /> Python 就绪
      </span>
    )
  return (
    <button
      type="button"
      onClick={() => pyRunner.preload().catch(() => {})}
      className={cn('inline-flex items-center gap-1.5 text-xs text-subtle hover:text-fg', className)}
    >
      <span className="h-2 w-2 rounded-full bg-border-strong" /> Python 未加载
    </button>
  )
}

/** 手机端符号辅助栏（按下不夺取编辑器焦点，避免收起软键盘） */
export function SymbolBar({ onInsert, onIndent, onUndo }: { onInsert: (s: string) => void; onIndent: (d: 1 | -1) => void; onUndo: () => void }) {
  const keys: { label: string; text?: string; act?: () => void; wide?: boolean }[] = [
    { label: 'Tab', act: () => onIndent(1), wide: true },
    { label: '⇤', act: () => onIndent(-1) },
    { label: '(', text: '(' },
    { label: ')', text: ')' },
    { label: '[', text: '[' },
    { label: ']', text: ']' },
    { label: ':', text: ':' },
    { label: '=', text: '=' },
    { label: '_', text: '_' },
    { label: ',', text: ', ' },
    { label: '.', text: '.' },
    { label: "'", text: "'" },
    { label: '"', text: '"' },
    { label: '+', text: '+' },
    { label: '-', text: '-' },
    { label: '*', text: '*' },
    { label: '/', text: '/' },
    { label: '<', text: '<' },
    { label: '>', text: '>' },
    { label: '#', text: '# ' },
    { label: '↶', act: onUndo },
  ]
  return (
    <div className="no-scrollbar flex gap-1 overflow-x-auto border-b border-border bg-surface-2 px-1.5 py-1.5" role="toolbar" aria-label="符号辅助栏">
      {keys.map((k) => (
        <button
          key={k.label}
          type="button"
          onPointerDown={(e) => e.preventDefault()}
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => (k.act ? k.act() : onInsert(k.text!))}
          className={cn(
            'h-9 shrink-0 rounded-lg border border-border bg-surface font-mono text-[15px] text-fg shadow-sm active:scale-95 active:bg-accent-soft',
            k.wide ? 'px-3 text-xs font-semibold' : 'w-9',
          )}
        >
          {k.label}
        </button>
      ))}
    </div>
  )
}
