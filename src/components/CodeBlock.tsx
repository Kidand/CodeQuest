import { useMemo, useState } from 'react'
import hljs from 'highlight.js/lib/core'
import python from 'highlight.js/lib/languages/python'
import bash from 'highlight.js/lib/languages/bash'
import json from 'highlight.js/lib/languages/json'
import { Check, Copy } from 'lucide-react'
import { cn } from '@/lib/utils'

hljs.registerLanguage('python', python)
hljs.registerLanguage('py', python)
hljs.registerLanguage('bash', bash)
hljs.registerLanguage('json', json)

export function highlight(code: string, lang = 'python'): string {
  try {
    if (lang && hljs.getLanguage(lang)) return hljs.highlight(code, { language: lang, ignoreIllegals: true }).value
  } catch {}
  return escapeHtml(code)
}

export function escapeHtml(s: string) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

/** 只读代码块（语法高亮 + 复制按钮），可选行号与高亮行 */
export function CodeBlock({
  code,
  lang = 'python',
  lineNumbers = false,
  highlightLines,
  className,
  copyable = true,
  maxHeight,
}: {
  code: string
  lang?: string
  lineNumbers?: boolean
  highlightLines?: number[]
  className?: string
  copyable?: boolean
  maxHeight?: number
}) {
  const [copied, setCopied] = useState(false)
  const lines = useMemo(() => code.replace(/\n$/, '').split('\n'), [code])
  const html = useMemo(() => lines.map((l) => highlight(l, lang) || ' '), [lines, lang])
  const hl = new Set(highlightLines ?? [])
  return (
    <div className={cn('group relative overflow-hidden rounded-xl border border-border bg-code-bg', className)}>
      {copyable && (
        <button
          type="button"
          onClick={() => {
            navigator.clipboard?.writeText(code).then(() => {
              setCopied(true)
              setTimeout(() => setCopied(false), 1200)
            })
          }}
          className="absolute right-2 top-2 z-10 rounded-lg border border-border bg-surface/80 p-1.5 text-muted opacity-0 backdrop-blur transition-opacity hover:text-fg group-hover:opacity-100 focus:opacity-100 max-md:opacity-70"
          aria-label="复制代码"
        >
          {copied ? <Check size={14} /> : <Copy size={14} />}
        </button>
      )}
      <pre className="thin-scroll overflow-auto py-3 text-[13px] leading-[1.65]" style={maxHeight ? { maxHeight } : undefined}>
        <code className="hljs block min-w-max">
          {html.map((h, i) => (
            <div
              key={i}
              className={cn('flex px-4', hl.has(i + 1) && 'bg-[var(--line-active)]')}
            >
              {lineNumbers && (
                <span className="mr-4 inline-block w-6 shrink-0 select-none text-right text-subtle tabular-nums">{i + 1}</span>
              )}
              <span dangerouslySetInnerHTML={{ __html: h }} />
            </div>
          ))}
        </code>
      </pre>
    </div>
  )
}
