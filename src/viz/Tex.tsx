import { useMemo } from 'react'
import katex from 'katex'

/** 在可视化里渲染公式：<Tex>{String.raw`\frac{a}{b}`}</Tex>，display 为块级 */
export function Tex({ children, display = false, className }: { children: string; display?: boolean; className?: string }) {
  const html = useMemo(() => katex.renderToString(children, { displayMode: display, throwOnError: false, strict: false }), [children, display])
  return <span className={className} dangerouslySetInnerHTML={{ __html: html }} />
}
