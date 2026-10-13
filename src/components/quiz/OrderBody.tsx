import { Reorder, useDragControls } from 'motion/react'
import { ArrowDown, ArrowUp, GripVertical } from 'lucide-react'
import type { OrderQ } from '@/content/types'
import { InlineMarkdown } from '@/components/Markdown'
import { cn } from '@/lib/utils'
import { MarkIcon, markCls, type BodyProps, type Mark } from './marks'

export function moveItem<T>(arr: T[], from: number, to: number): T[] {
  if (to < 0 || to >= arr.length || from === to) return arr
  const a = arr.slice()
  const [x] = a.splice(from, 1)
  a.splice(to, 0, x)
  return a
}

/** 拖拽手柄（仅手柄触发拖拽，避免误触与滚动冲突） */
export function DragHandle({ onPointerDown, disabled }: { onPointerDown: (e: React.PointerEvent) => void; disabled?: boolean }) {
  return (
    <span
      aria-hidden
      onPointerDown={(e) => {
        if (disabled) return
        e.preventDefault()
        onPointerDown(e)
      }}
      className={cn(
        'flex h-9 w-6 shrink-0 items-center justify-center rounded-md text-subtle',
        disabled ? 'opacity-0' : 'cursor-grab touch-none hover:bg-surface-3 hover:text-muted active:cursor-grabbing',
      )}
    >
      <GripVertical size={16} />
    </span>
  )
}

export function MoveButtons({
  pos,
  count,
  onMove,
  label,
}: {
  pos: number
  count: number
  onMove: (to: number) => void
  label: string
}) {
  const btn =
    'inline-flex h-8 w-8 items-center justify-center rounded-lg text-muted transition-colors hover:bg-surface-3 hover:text-fg disabled:opacity-25 active:scale-95'
  return (
    <span className="flex shrink-0 items-center">
      <button type="button" className={btn} disabled={pos === 0} onClick={() => onMove(pos - 1)} aria-label={`上移${label}`}>
        <ArrowUp size={16} />
      </button>
      <button type="button" className={btn} disabled={pos === count - 1} onClick={() => onMove(pos + 1)} aria-label={`下移${label}`}>
        <ArrowDown size={16} />
      </button>
    </span>
  )
}

function OrderRow({
  q,
  idx,
  pos,
  count,
  submitted,
  onMove,
}: {
  q: OrderQ
  idx: number
  pos: number
  count: number
  submitted: boolean
  onMove: (to: number) => void
}) {
  const controls = useDragControls()
  const m: Mark = submitted ? (idx === pos ? 'correct' : 'wrong') : 'idle'
  return (
    <Reorder.Item
      value={idx}
      dragListener={false}
      dragControls={controls}
      drag={submitted ? false : 'y'}
      className={cn('relative flex items-center gap-1.5 rounded-xl py-1.5 pl-1.5 pr-1 text-[15px] text-fg', markCls(m, false))}
      whileDrag={{ scale: 1.02, boxShadow: 'var(--sh-lg)', zIndex: 10 }}
    >
      <DragHandle onPointerDown={(e) => controls.start(e)} disabled={submitted} />
      <span
        className={cn(
          'inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full font-mono text-xs font-semibold',
          m === 'correct' ? 'bg-success text-surface' : m === 'wrong' ? 'bg-danger text-surface' : 'bg-surface-3 text-muted',
        )}
      >
        {pos + 1}
      </span>
      <span className="min-w-0 flex-1 py-1 leading-relaxed">
        <InlineMarkdown>{q.items[idx]}</InlineMarkdown>
        {submitted && idx !== pos && <span className="ml-2 whitespace-nowrap text-xs font-medium text-danger">应为第 {idx + 1} 步</span>}
      </span>
      {submitted ? <MarkIcon m={m} className="mr-1.5" /> : <MoveButtons pos={pos} count={count} onMove={onMove} label="此项" />}
    </Reorder.Item>
  )
}

/** 排序：上下移动按钮（触屏）+ 手柄拖拽（增强） */
export function OrderBody({ q, value, onChange, submitted }: BodyProps<OrderQ>) {
  const order = value as number[]
  return (
    <div>
      <p className="mb-2 text-xs text-subtle">用 ↑ ↓ 调整顺序，或按住左侧手柄拖动</p>
      <Reorder.Group axis="y" values={order} onReorder={(v) => !submitted && onChange(v)} className="flex flex-col gap-2" as="ol">
        {order.map((idx, pos) => (
          <OrderRow
            key={idx}
            q={q}
            idx={idx}
            pos={pos}
            count={order.length}
            submitted={submitted}
            onMove={(to) => onChange(moveItem(order, pos, to))}
          />
        ))}
      </Reorder.Group>
    </div>
  )
}
