/**
 * CodeMirror 6 Python 编辑器（懒加载模块，控制首屏体积）。
 *  - 主题走 CSS 变量（src/lib/codemirror-theme.ts），字号来自设置
 *  - 4 空格缩进、括号匹配 / 自动闭合、Ctrl/⌘+Enter 运行、Ctrl/⌘+Shift+Enter 提交
 *  - errorLine 行高亮（编辑后自动清除）
 */
import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef } from 'react'
import CodeMirror, { Decoration, EditorState, EditorView, Prec, StateEffect, StateField, keymap, type DecorationSet } from '@uiw/react-codemirror'
import { python } from '@codemirror/lang-python'
import { indentUnit } from '@codemirror/language'
import { undo } from '@codemirror/commands'
import { cqEditorTheme, cqFontSize } from '@/lib/codemirror-theme'
import { useIsDark } from '@/lib/theme'

export interface CodeEditorHandle {
  insert: (text: string) => void
  focus: () => void
  /** 缩进 / 反缩进当前行 */
  indent: (dir: 1 | -1) => void
  undo: () => void
}

export interface CodeEditorProps {
  value: string
  onChange: (v: string) => void
  fontSize: number
  errorLine?: number | null
  onRun?: () => void
  onSubmit?: () => void
  onSave?: () => void
  autocomplete?: boolean
  className?: string
}

const setErrorLine = StateEffect.define<number | null>()
const errorLineField = StateField.define<DecorationSet>({
  create: () => Decoration.none,
  update(deco, tr) {
    if (tr.docChanged) deco = Decoration.none
    for (const e of tr.effects) {
      if (!e.is(setErrorLine)) continue
      const n = e.value
      if (n == null || n < 1 || n > tr.state.doc.lines) deco = Decoration.none
      else deco = Decoration.set([Decoration.line({ class: 'cm-cq-error-line' }).range(tr.state.doc.line(n).from)])
    }
    return deco
  },
  provide: (f) => EditorView.decorations.from(f),
})

const CodeEditor = forwardRef<CodeEditorHandle, CodeEditorProps>(function CodeEditor(
  { value, onChange, fontSize, errorLine, onRun, onSubmit, onSave, autocomplete = true, className },
  ref,
) {
  const viewRef = useRef<EditorView | null>(null)
  const cb = useRef({ onRun, onSubmit, onSave })
  cb.current = { onRun, onSubmit, onSave }
  const dark = useIsDark()

  useImperativeHandle(ref, () => ({
    insert(text) {
      const v = viewRef.current
      if (!v) return
      v.dispatch(v.state.replaceSelection(text), { scrollIntoView: true, userEvent: 'input' })
      v.focus()
    },
    focus() {
      viewRef.current?.focus()
    },
    indent(dir) {
      const v = viewRef.current
      if (!v) return
      const { state } = v
      const changes = []
      const seen = new Set<number>()
      for (const r of state.selection.ranges) {
        for (let pos = r.from; pos <= r.to; ) {
          const line = state.doc.lineAt(pos)
          if (!seen.has(line.number)) {
            seen.add(line.number)
            if (dir === 1) changes.push({ from: line.from, insert: '    ' })
            else {
              const m = /^ {1,4}/.exec(line.text)
              if (m) changes.push({ from: line.from, to: line.from + m[0].length })
            }
          }
          pos = line.to + 1
        }
      }
      v.dispatch({ changes, userEvent: 'input' })
      v.focus()
    },
    undo() {
      const v = viewRef.current
      if (!v) return
      undo(v)
      v.focus()
    },
  }))

  // 报错行：高亮并滚到可见
  useEffect(() => {
    const v = viewRef.current
    if (!v) return
    const effects: StateEffect<unknown>[] = [setErrorLine.of(errorLine ?? null)]
    if (errorLine && errorLine >= 1 && errorLine <= v.state.doc.lines)
      effects.push(EditorView.scrollIntoView(v.state.doc.line(errorLine).from, { y: 'center' }))
    v.dispatch({ effects })
  }, [errorLine])

  const extensions = useMemo(
    () => [
      python(),
      indentUnit.of('    '),
      EditorState.tabSize.of(4),
      errorLineField,
      cqEditorTheme(dark),
      cqFontSize(fontSize),
      EditorView.contentAttributes.of({ autocapitalize: 'off', autocorrect: 'off', spellcheck: 'false', 'aria-label': '代码编辑器' }),
      Prec.highest(
        keymap.of([
          { key: 'Mod-Enter', run: () => (cb.current.onRun?.(), true) },
          { key: 'Mod-Shift-Enter', run: () => (cb.current.onSubmit?.(), true) },
          { key: 'Mod-s', run: () => (cb.current.onSave?.(), true), preventDefault: true },
        ]),
      ),
    ],
    [dark, fontSize],
  )

  return (
    <CodeMirror
      value={value}
      onChange={onChange}
      onCreateEditor={(view) => {
        viewRef.current = view
        if (errorLine) view.dispatch({ effects: setErrorLine.of(errorLine) })
      }}
      theme="none"
      height="100%"
      className={className}
      extensions={extensions}
      indentWithTab
      basicSetup={{
        lineNumbers: true,
        foldGutter: true,
        highlightActiveLine: true,
        highlightActiveLineGutter: true,
        bracketMatching: true,
        closeBrackets: true,
        autocompletion: autocomplete,
        indentOnInput: true,
        highlightSelectionMatches: true,
        searchKeymap: true,
        tabSize: 4,
        dropCursor: true,
        allowMultipleSelections: true,
        rectangularSelection: true,
        crosshairCursor: false,
        history: true,
        drawSelection: true,
        syntaxHighlighting: false,
      }}
    />
  )
})

export default CodeEditor
