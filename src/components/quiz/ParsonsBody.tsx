import { Reorder, useDragControls } from 'motion/react'
import { CornerDownLeft, Plus } from 'lucide-react'
import type { ParsonsQ } from '@/content/types'
import { highlight } from '@/components/CodeBlock'
import { cn } from '@/lib/utils'
import { parsonsText } from './grade'
import { MarkIcon, markCls, type BodyProps, type Mark } from './marks'
import { DragHandle, MoveButtons, moveItem } from './OrderBody'

/** 一行代码：保留缩进（缩进处画出参考线），语法高亮 */
export function CodeLine({ text, className }: { text: string; className?: string }) {
  const indent = text.match(/^[ \t]*/)![0].replace(/\t/g, '    ').length
  const body = text.slice(text.match(/^[ \t]*/)![0].length)
  return (
    <span className={cn('flex min-w-0 font-mono text-[13px] leading-6 whitespace-pre', className)}>
      {indent > 0 && <span className="cq-indent shrink-0" style={{ width: `${indent}ch` }} aria-hidden />}
      <span className="hljs" dangerouslySetInnerHTML={{ __html: highlight(body) || ' ' }} />
    </span>
  )
}

function AnswerRow({
  q,
  id,
  pos,
  count,
  submitted,
  onMove,
  onRemove,
}: {
  q: ParsonsQ
  id: number
  pos: number
  count: number
  submitted: boolean
  onMove: (to: number) => void
  onRemove: () => void
}) {
  const controls = useDragControls()
  const text = parsonsText(q, id)
  const m: Mark = submitted ? (text === q.lines[pos] ? 'correct' : 'wrong') : 'idle'
  return (
    <Reorder.Item
      value={id}
      dragListener={false}
      dragControls={controls}
      drag={submitted ? false : 'y'}
      className={cn('flex items-center gap-1 rounded-lg py-0.5 pl-1 pr-1', markCls(m, false))}
      whileDrag={{ scale: 1.02, boxShadow: 'var(--sh-lg)', zIndex: 10 }}
    >
      <DragHandle onPointerDown={(e) => controls.start(e)} disabled={submitted} />
      <span className="w-5 shrink-0 select-none text-right font-mono text-[11px] text-subtle tabular-nums">{pos + 1}</span>
      <button
        type="button"
        disabled={submitted}
        onClick={onRemove}
        title="点击移回候选区"
        className="thin-scroll min-w-0 flex-1 overflow-x-auto rounded-md px-1.5 py-1 text-left hover:bg-surface-3 disabled:cursor-default disabled:hover:bg-transparent"
      >
        <CodeLine text={text} />
      </button>
      {submitted ? <MarkIcon m={m} className="mr-1" /> : <MoveButtons pos={pos} count={count} onMove={onMove} label="此行" />}
    </Reorder.Item>
  )
}

/** 代码拼图：候选区 ↔ 答案区点选移动；答案区可上下移动 / 拖拽调整顺序 */
export function ParsonsBody({ q, value, onChange, submitted, layout }: BodyProps<ParsonsQ>) {
  const ans = value as number[]
  const pool = (layout.pool ?? []).filter((id) => !ans.includes(id))
  const total = q.lines.length + (q.distractors?.length ?? 0)
  return (
    <div className="flex flex-col gap-3">
      <div>
        <div className="mb-1.5 flex items-center justify-between text-xs text-subtle">
          <span className="font-semibold text-muted">答案区</span>
          {!submitted && <span>点击行移回候选区 · ↑↓ 或拖动手柄调整顺序</span>}
        </div>
        <div
          className={cn(
            'min-h-16 rounded-xl border-2 border-dashed p-1.5 transition-colors',
            ans.length ? 'border-border bg-code-bg' : 'border-border-strong bg-surface-2',
          )}
        >
          {ans.length === 0 ? (
            <div className="flex h-12 items-center justify-center gap-1.5 text-sm text-subtle">
              <CornerDownLeft size={14} /> 点击下方代码行，按顺序放到这里
            </div>
          ) : (
            <Reorder.Group axis="y" values={ans} onReorder={(v) => !submitted && onChange(v)} className="flex flex-col gap-1" as="ol">
              {ans.map((id, pos) => (
                <AnswerRow
                  key={id}
                  q={q}
                  id={id}
                  pos={pos}
                  count={ans.length}
                  submitted={submitted}
                  onMove={(to) => onChange(moveItem(ans, pos, to))}
                  onRemove={() => onChange(ans.filter((x) => x !== id))}
                />
              ))}
            </Reorder.Group>
          )}
        </div>
      </div>
      {!submitted && (
        <div>
          <div className="mb-1.5 flex items-center justify-between text-xs">
            <span className="font-semibold text-muted">候选区</span>
            <span className="text-subtle">
              剩余 {pool.length}/{total} 行{q.distractors?.length ? ` · 含 ${q.distractors.length} 行干扰项` : ''}
            </span>
          </div>
          {pool.length === 0 ? (
            <div className="rounded-xl border border-border bg-surface-2 px-3 py-3 text-center text-sm text-subtle">候选区已空</div>
          ) : (
            <div className="flex flex-col gap-1.5">
              {pool.map((id) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => onChange([...ans, id])}
                  className="thin-scroll group flex items-center gap-2 overflow-x-auto rounded-lg border border-border bg-surface px-2 py-1 text-left shadow-sm transition-colors hover:border-accent hover:bg-accent-soft active:scale-[0.99]"
                >
                  <Plus size={14} className="shrink-0 text-subtle group-hover:text-accent" />
                  <CodeLine text={parsonsText(q, id)} />
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
