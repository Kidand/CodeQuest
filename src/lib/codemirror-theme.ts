/**
 * CodeMirror 6 主题：全部颜色来自 src/styles/index.css 的语义 token（CSS 变量），
 * 因此明暗主题切换无需重建编辑器，只有 `dark` 标志（影响 CM 内置面板的基础样式）随主题变化。
 */
import { EditorView } from '@codemirror/view'
import { HighlightStyle, syntaxHighlighting } from '@codemirror/language'
import { tags as t } from '@lezer/highlight'
import type { Extension } from '@codemirror/state'

const mix = (v: string, pct: number) => `color-mix(in oklab, var(${v}) ${pct}%, transparent)`

const base = (dark: boolean) =>
  EditorView.theme(
    {
      '&': {
        color: 'var(--code-fg)',
        backgroundColor: 'var(--code-bg)',
        height: '100%',
      },
      '&.cm-focused': { outline: 'none' },
      '.cm-scroller': {
        fontFamily: 'var(--font-mono)',
        lineHeight: '1.65',
        overscrollBehavior: 'contain',
      },
      '.cm-content': { caretColor: 'var(--accent)', padding: '10px 0 40px' },
      '.cm-line': { padding: '0 12px 0 6px' },
      '.cm-cursor, .cm-dropCursor': { borderLeftColor: 'var(--accent)', borderLeftWidth: '2px' },
      '&.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground, .cm-selectionBackground, .cm-content ::selection':
        { backgroundColor: mix('--accent', 24) },
      '.cm-activeLine': { backgroundColor: mix('--accent', 6) },
      '.cm-gutters': {
        backgroundColor: 'var(--code-bg)',
        color: 'var(--subtle)',
        border: 'none',
        borderRight: '1px solid var(--border)',
      },
      '.cm-lineNumbers .cm-gutterElement': { padding: '0 8px 0 10px', minWidth: '2.6em', fontVariantNumeric: 'tabular-nums' },
      '.cm-activeLineGutter': { backgroundColor: 'transparent', color: 'var(--fg)' },
      '.cm-foldGutter .cm-gutterElement': { color: 'var(--subtle)' },
      '.cm-matchingBracket, &.cm-focused .cm-matchingBracket': {
        backgroundColor: mix('--success', 22),
        outline: `1px solid ${mix('--success', 55)}`,
        borderRadius: '2px',
      },
      '.cm-nonmatchingBracket, &.cm-focused .cm-nonmatchingBracket': {
        backgroundColor: mix('--danger', 22),
        outline: `1px solid ${mix('--danger', 55)}`,
      },
      '.cm-selectionMatch': { backgroundColor: mix('--warning', 18) },
      '.cm-searchMatch': { backgroundColor: mix('--warning', 30), outline: `1px solid ${mix('--warning', 60)}` },
      '.cm-searchMatch.cm-searchMatch-selected': { backgroundColor: mix('--accent', 30) },
      '.cm-foldPlaceholder': {
        backgroundColor: 'var(--surface-2)',
        border: '1px solid var(--border)',
        color: 'var(--muted)',
        borderRadius: '4px',
        padding: '0 4px',
      },
      '.cm-tooltip': {
        backgroundColor: 'var(--surface)',
        color: 'var(--fg)',
        border: '1px solid var(--border)',
        borderRadius: '10px',
        boxShadow: 'var(--sh-md)',
        overflow: 'hidden',
      },
      '.cm-tooltip-autocomplete > ul': { fontFamily: 'var(--font-mono)', maxHeight: '14em' },
      '.cm-tooltip-autocomplete > ul > li': { padding: '2px 10px !important' },
      '.cm-tooltip-autocomplete > ul > li[aria-selected]': { backgroundColor: 'var(--accent-soft)', color: 'var(--fg)' },
      '.cm-completionIcon': { opacity: '0.6' },
      '.cm-completionDetail': { color: 'var(--subtle)' },
      '.cm-panels': { backgroundColor: 'var(--surface-2)', color: 'var(--fg)' },
      '.cm-panels.cm-panels-top': { borderBottom: '1px solid var(--border)' },
      '.cm-panels.cm-panels-bottom': { borderTop: '1px solid var(--border)' },
      '.cm-panel input, .cm-panel button': { fontSize: '12px' },
      '.cm-textfield': {
        backgroundColor: 'var(--surface)',
        border: '1px solid var(--border-strong)',
        borderRadius: '6px',
        color: 'var(--fg)',
      },
      '.cm-button': {
        backgroundImage: 'none',
        backgroundColor: 'var(--surface)',
        border: '1px solid var(--border-strong)',
        borderRadius: '6px',
        color: 'var(--fg)',
      },
      /* 运行报错行 */
      '.cm-cq-error-line': { backgroundColor: mix('--danger', 14), boxShadow: `inset 3px 0 0 var(--danger)` },
      '.cm-cq-error-gutter': { color: 'var(--danger)', fontWeight: '700' },
      '.cm-placeholder': { color: 'var(--subtle)' },
    },
    { dark },
  )

export const cqHighlightStyle = HighlightStyle.define([
  { tag: [t.keyword, t.controlKeyword, t.definitionKeyword, t.moduleKeyword, t.operatorKeyword], color: 'var(--syn-keyword)' },
  { tag: [t.string, t.special(t.string), t.docString, t.character, t.regexp], color: 'var(--syn-string)' },
  { tag: [t.number, t.bool, t.null, t.atom], color: 'var(--syn-number)' },
  { tag: [t.comment, t.lineComment, t.blockComment], color: 'var(--syn-comment)', fontStyle: 'italic' },
  {
    tag: [t.function(t.variableName), t.function(t.definition(t.variableName)), t.function(t.propertyName), t.definition(t.className)],
    color: 'var(--syn-func)',
  },
  { tag: [t.className, t.typeName, t.self, t.standard(t.variableName), t.namespace], color: 'var(--syn-builtin)' },
  { tag: [t.meta, t.annotation], color: 'var(--syn-func)' },
  { tag: [t.operator, t.punctuation, t.bracket, t.separator], color: 'var(--syn-op)' },
  { tag: [t.propertyName, t.variableName, t.definition(t.variableName)], color: 'var(--code-fg)' },
  { tag: t.invalid, color: 'var(--danger)' },
  { tag: t.strong, fontWeight: '700' },
  { tag: t.emphasis, fontStyle: 'italic' },
])

/** 编辑器主题（颜色走 token） */
export function cqEditorTheme(dark: boolean): Extension {
  return [base(dark), syntaxHighlighting(cqHighlightStyle)]
}

/** 字号（来自设置 editorFontSize） */
export function cqFontSize(px: number): Extension {
  return EditorView.theme({ '&': { fontSize: `${px}px` }, '.cm-gutters': { fontSize: `${Math.max(11, px - 1)}px` } })
}
