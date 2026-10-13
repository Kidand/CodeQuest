import type { ChapterDef, LevelContent } from './types'

/**
 * 去掉公共缩进与首尾空行。
 */
export function dedent(text: string): string {
  const lines = text.replace(/\r\n/g, '\n').split('\n')
  while (lines.length && lines[0].trim() === '') lines.shift()
  while (lines.length && lines[lines.length - 1].trim() === '') lines.pop()
  let min = Infinity
  for (const l of lines) {
    if (l.trim() === '') continue
    const m = l.match(/^[ \t]*/)![0].length
    if (m < min) min = m
  }
  if (!isFinite(min) || min === 0) return lines.join('\n')
  return lines.map((l) => l.slice(Math.min(min, l.match(/^[ \t]*/)![0].length))).join('\n')
}

/**
 * 原样模板标签：反斜杠不转义（LaTeX 的 \frac、\times、\nabla、Python 的 '\n' 都安全），
 * 自动 dedent。唯二的转义：用 \` 表示反引号，用 \${ 表示字面量 ${。
 *
 *   md`公式 $\frac{QK^\top}{\sqrt{d_k}}$，代码 \`x @ W\``
 *   py`print("a\nb")`   // Python 收到的就是 "a\nb"
 */
function raw(strings: TemplateStringsArray, values: unknown[]): string {
  const s = String.raw({ raw: strings.raw }, ...values)
  return dedent(s.replace(/\\`/g, '`').replace(/\\\$\{/g, '${'))
}

/** Markdown 文本（含 LaTeX）请一律使用 md`` */
export const md = (strings: TemplateStringsArray, ...values: unknown[]) => raw(strings, values)
/** Python 代码请一律使用 py`` */
export const py = (strings: TemplateStringsArray, ...values: unknown[]) => raw(strings, values)

/** 类型辅助：章节定义 */
export const defineChapter = (c: ChapterDef) => c
/** 类型辅助：关卡正文 */
export const defineLevel = (l: LevelContent) => l
