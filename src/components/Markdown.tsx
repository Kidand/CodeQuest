import { memo, type ReactNode, Children, isValidElement } from 'react'
import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { AlertTriangle, Info, Lightbulb, MessageSquareQuote, Sparkles } from 'lucide-react'
import { CodeBlock } from './CodeBlock'
import { cn } from '@/lib/utils'

/**
 * 提示块语法（GitHub Alerts 风格）：
 *   > [!TIP] 标题（可选）
 *   > 内容
 * 支持：TIP 技巧 / NOTE 说明 / WARN 易错 / INTERVIEW 面试官视角 / NEW 2026 新进展
 */
const CALLOUTS = {
  TIP: { icon: Lightbulb, label: '技巧', cls: 'border-success/40 bg-success-soft/60', ic: 'text-success' },
  NOTE: { icon: Info, label: '说明', cls: 'border-info/40 bg-info-soft/60', ic: 'text-info' },
  WARN: { icon: AlertTriangle, label: '易错点', cls: 'border-warning/40 bg-warning-soft/60', ic: 'text-warning' },
  INTERVIEW: { icon: MessageSquareQuote, label: '面试官视角', cls: 'border-accent/40 bg-accent-soft/60', ic: 'text-accent' },
  NEW: { icon: Sparkles, label: '前沿进展', cls: 'border-llm/40 bg-llm-soft/60', ic: 'text-llm' },
} as const
type CalloutKind = keyof typeof CALLOUTS

function textOf(node: ReactNode): string {
  if (typeof node === 'string') return node
  if (Array.isArray(node)) return node.map(textOf).join('')
  if (isValidElement(node)) return textOf((node.props as { children?: ReactNode }).children)
  return ''
}

/** 从 blockquote 的第一个段落中剥离 [!KIND] 标记 */
function stripMarker(children: ReactNode): { kind?: CalloutKind; title?: string; rest: ReactNode } {
  const arr = Children.toArray(children).filter((c) => !(typeof c === 'string' && c.trim() === ''))
  const first = arr[0]
  if (!isValidElement(first)) return { rest: children }
  const firstChildren = Children.toArray((first.props as { children?: ReactNode }).children)
  const head = firstChildren[0]
  if (typeof head !== 'string') return { rest: children }
  const m = head.match(/^\s*\[!(TIP|NOTE|WARN|WARNING|INTERVIEW|NEW)\]\s*([^\n]*)\n?/)
  if (!m) return { rest: children }
  const kind = (m[1] === 'WARNING' ? 'WARN' : m[1]) as CalloutKind
  const title = m[2]?.trim() || undefined
  const remainingHead = head.slice(m[0].length)
  const newFirstChildren = [remainingHead, ...firstChildren.slice(1)].filter((c) => !(typeof c === 'string' && c === ''))
  const hasFirst = newFirstChildren.some((c) => textOf(c).trim() !== '' || isValidElement(c))
  const rest = [hasFirst ? <p key="__first">{newFirstChildren}</p> : null, ...arr.slice(1)]
  return { kind, title, rest }
}

const components: Components = {
  pre({ children }) {
    const child = Children.toArray(children)[0]
    if (isValidElement(child)) {
      const props = child.props as { className?: string; children?: ReactNode }
      const lang = /language-(\w+)/.exec(props.className ?? '')?.[1] ?? 'python'
      const code = textOf(props.children).replace(/\n$/, '')
      return <CodeBlock code={code} lang={lang} />
    }
    return <pre>{children}</pre>
  },
  blockquote({ children }) {
    const { kind, title, rest } = stripMarker(children)
    if (!kind) return <blockquote>{children}</blockquote>
    const c = CALLOUTS[kind]
    const Icon = c.icon
    return (
      <div className={cn('my-4 rounded-xl border px-4 py-3', c.cls)}>
        <div className={cn('mb-1 flex items-center gap-1.5 text-[13px] font-semibold', c.ic)}>
          <Icon size={15} />
          <span>{title ?? c.label}</span>
        </div>
        <div className="md-callout text-[14.5px] [&>p:first-child]:mt-0 [&>p:last-child]:mb-0">{rest}</div>
      </div>
    )
  },
  a({ href, children }) {
    return (
      <a href={href} target="_blank" rel="noreferrer">
        {children}
      </a>
    )
  },
}

/** 渲染 markdown（GFM + KaTeX + 代码高亮 + 提示块） */
export const Markdown = memo(function Markdown({ children, className }: { children: string; className?: string }) {
  return (
    <div className={cn('md-body', className)}>
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: false }]]} components={components}>
        {children}
      </ReactMarkdown>
    </div>
  )
})

/** 行内渲染（无段落外边距），用于选项、按钮文字等短文本 */
export const InlineMarkdown = memo(function InlineMarkdown({ children, className }: { children: string; className?: string }) {
  return (
    <span className={cn('md-body md-inline', className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: false }]]}
        components={{ ...components, p: ({ children }) => <>{children}</> }}
      >
        {children}
      </ReactMarkdown>
    </span>
  )
})
