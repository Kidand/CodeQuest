import { Bug } from 'lucide-react'
import type { BugQ, PickQ } from '@/content/types'
import { highlight } from '@/components/CodeBlock'
import { InlineMarkdown } from '@/components/Markdown'
import { cn } from '@/lib/utils'
import { MarkIcon, markCls, type BodyProps, type Mark } from './marks'

const toggleIn = (arr: number[], x: number) => (arr.includes(x) ? arr.filter((y) => y !== x) : [...arr, x].sort((a, b) => a - b))

/** 找 Bug：带行号的代码，点击行切换选中 */
export function BugBody({ q, value, onChange, submitted }: BodyProps<BugQ>) {
  const sel = value as number[]
  const lines = q.code.replace(/\n$/, '').split('\n')
  return (
    <div>
      {!submitted && <p className="mb-2 text-xs text-subtle">点击有问题的行（可多选），再次点击取消</p>}
      <div className="thin-scroll overflow-x-auto rounded-xl border border-border bg-code-bg py-1.5">
        <div className="min-w-max">
          {lines.map((line, k) => {
            const n = k + 1
            const isSel = sel.includes(n)
            const isBug = q.bugLines.includes(n)
            let m: Mark | 'none' = isSel ? 'selected' : 'none'
            if (submitted) m = isSel ? (isBug ? 'correct' : 'wrong') : isBug ? 'missed' : 'none'
            return (
              <button
                key={k}
                type="button"
                disabled={submitted}
                aria-pressed={isSel}
                data-line={n}
                onClick={() => onChange(toggleIn(sel, n))}
                className={cn(
                  'group flex w-full items-center gap-0 py-[3px] pr-3 text-left font-mono text-[13px] leading-6 transition-colors disabled:cursor-default',
                  m === 'none' && !submitted && 'hover:bg-surface-2',
                  m === 'selected' && 'bg-danger-soft shadow-[inset_3px_0_0_var(--danger)]',
                  m === 'correct' && 'bg-success-soft shadow-[inset_3px_0_0_var(--success)]',
                  m === 'wrong' && 'bg-danger-soft shadow-[inset_3px_0_0_var(--danger)]',
                  m === 'missed' && 'bg-warning-soft shadow-[inset_3px_0_0_var(--warning)]',
                )}
              >
                <span className="flex w-11 shrink-0 select-none items-center justify-end gap-1 pr-3 text-right text-[12px] text-subtle tabular-nums">
                  {m === 'selected' ? <Bug size={12} className="text-danger" /> : null}
                  {n}
                </span>
                <span className="hljs whitespace-pre" dangerouslySetInnerHTML={{ __html: highlight(line) || ' ' }} />
                {submitted && m !== 'none' && (
                  <span className="ml-3 shrink-0 font-sans">
                    {m === 'missed' ? (
                      <span className="rounded-full bg-warning px-1.5 py-0.5 text-[11px] font-semibold text-surface">漏掉的 bug</span>
                    ) : (
                      <MarkIcon m={m} />
                    )}
                  </span>
                )}
              </button>
            )
          })}
        </div>
      </div>
    </div>
  )
}

/** 点选：chips 或带行列标签的网格 */
export function PickBody({ q, value, onChange, submitted }: BodyProps<PickQ>) {
  const sel = value as number[]
  const markOf = (i: number): Mark => {
    const isSel = sel.includes(i)
    const ok = q.answer.includes(i)
    if (!submitted) return isSel ? 'selected' : 'idle'
    return isSel ? (ok ? 'correct' : 'wrong') : ok ? 'missed' : 'dim'
  }
  const cell = (i: number, grid: boolean) => {
    const m = markOf(i)
    return (
      <button
        key={i}
        type="button"
        disabled={submitted}
        aria-pressed={sel.includes(i)}
        data-item={i}
        onClick={() => onChange(toggleIn(sel, i))}
        className={cn(
          'relative flex items-center justify-center rounded-lg font-mono text-[13px] text-fg disabled:cursor-default',
          grid ? 'aspect-square min-h-9 min-w-9 p-1' : 'min-h-10 min-w-10 px-3 py-1.5',
          markCls(m === 'dim' ? 'idle' : m, !submitted),
          m === 'dim' && 'opacity-60',
          m === 'selected' && 'font-semibold text-accent',
        )}
      >
        <InlineMarkdown>{q.items[i]}</InlineMarkdown>
        {submitted && (m === 'correct' || m === 'wrong') && <MarkIcon m={m} className="absolute -right-1.5 -top-1.5 h-4 w-4" />}
        {submitted && m === 'missed' && (
          <span className="absolute -right-1 -top-1.5 rounded-full bg-success px-1 text-[9px] font-bold leading-4 text-surface">漏</span>
        )}
      </button>
    )
  }

  if (q.cols) {
    const cols = q.cols
    const rows = Math.ceil(q.items.length / cols)
    const hasRow = !!q.rowLabels?.length
    const hasCol = !!q.colLabels?.length
    return (
      <div className="thin-scroll overflow-x-auto pb-1">
        {!submitted && <p className="mb-2 text-xs text-subtle">点击格子选中 / 取消，选出全部满足条件的格子</p>}
        <div
          className="mx-auto grid w-max gap-1.5"
          style={{ gridTemplateColumns: `${hasRow ? 'auto ' : ''}repeat(${cols}, minmax(2.5rem, 3.25rem))` }}
        >
          {hasCol && (
            <>
              {hasRow && <span />}
              {Array.from({ length: cols }, (_, c) => (
                <span key={`c${c}`} className="truncate pb-0.5 text-center font-mono text-[11px] text-muted">
                  {q.colLabels![c] ?? ''}
                </span>
              ))}
            </>
          )}
          {Array.from({ length: rows }, (_, r) => (
            <div key={r} className="contents">
              {hasRow && (
                <span className="flex items-center justify-end pr-1 font-mono text-[11px] text-muted">{q.rowLabels![r] ?? ''}</span>
              )}
              {Array.from({ length: cols }, (_, c) => {
                const i = r * cols + c
                return i < q.items.length ? cell(i, true) : <span key={`e${c}`} />
              })}
            </div>
          ))}
        </div>
      </div>
    )
  }
  return (
    <div>
      {!submitted && <p className="mb-2 text-xs text-subtle">点击选中 / 取消，选出全部满足条件的项</p>}
      <div className="flex flex-wrap gap-2">{q.items.map((_, i) => cell(i, false))}</div>
    </div>
  )
}
