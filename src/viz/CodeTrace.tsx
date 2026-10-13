import { useEffect, useMemo, useRef } from 'react'
import { highlight } from '@/components/CodeBlock'
import { cn } from '@/lib/utils'

/**
 * 代码追踪面板：显示代码并高亮当前执行行（与 stepper 联动）。
 * active 为 1 起始行号（可多行）。vars 可显示当前变量快照。
 */
export function CodeTrace({
  code,
  active,
  lang = 'python',
  vars,
  className,
  maxHeight = 360,
}: {
  code: string
  active?: number | number[]
  lang?: string
  vars?: Record<string, React.ReactNode>
  className?: string
  maxHeight?: number
}) {
  const lines = useMemo(() => code.replace(/\n$/, '').split('\n'), [code])
  const html = useMemo(() => lines.map((l) => highlight(l, lang) || ' '), [lines, lang])
  const act = new Set(Array.isArray(active) ? active : active != null ? [active] : [])
  const first = Math.min(...act)
  const boxRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const box = boxRef.current
    if (!box || !Number.isFinite(first)) return
    const el = box.querySelector<HTMLElement>(`[data-line="${first}"]`)
    if (!el) return
    const top = el.offsetTop - box.clientHeight / 3
    box.scrollTo({ top: Math.max(0, top), behavior: 'smooth' })
  }, [first])
  return (
    <div className={cn('overflow-hidden rounded-xl border border-border bg-code-bg text-[12.5px]', className)}>
      <div ref={boxRef} className="thin-scroll relative overflow-auto py-2" style={{ maxHeight }}>
        <code className="hljs block min-w-max whitespace-pre font-mono leading-[1.7]">
          {html.map((h, i) => {
            const on = act.has(i + 1)
            return (
              <div
                key={i}
                data-line={i + 1}
                className={cn('relative flex pr-4 transition-colors duration-200', on ? 'bg-[var(--line-active)]' : '')}
              >
                <span className={cn('absolute left-0 top-0 h-full w-[3px] transition-colors', on ? 'bg-accent' : 'bg-transparent')} />
                <span className="ml-2 mr-3 inline-block w-5 shrink-0 select-none text-right text-subtle tabular-nums">{i + 1}</span>
                <span dangerouslySetInnerHTML={{ __html: h }} />
              </div>
            )
          })}
        </code>
      </div>
      {vars && Object.keys(vars).length > 0 && (
        <div className="flex flex-wrap gap-x-3 gap-y-1 border-t border-border bg-surface px-3 py-2 font-mono text-xs">
          {Object.entries(vars).map(([k, v]) => (
            <span key={k}>
              <span className="text-muted">{k}</span>
              <span className="text-subtle"> = </span>
              <span className="text-fg">{v}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
