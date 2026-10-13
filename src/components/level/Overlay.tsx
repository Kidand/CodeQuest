/**
 * 应用内浮层：对话框 / 二次确认 / 抽屉 / 错误边界。
 * 禁止使用 window.alert / confirm / prompt（会阻塞浏览器自动化）。
 */
import { Component, useEffect, useRef, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { AlertTriangle, RotateCcw, X } from 'lucide-react'
import { Button } from '@/components/ui'
import { cn } from '@/lib/utils'

function useEscape(open: boolean, onClose: () => void) {
  const ref = useRef(onClose)
  ref.current = onClose
  useEffect(() => {
    if (!open) return
    const on = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        ref.current()
      }
    }
    window.addEventListener('keydown', on)
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      window.removeEventListener('keydown', on)
      document.body.style.overflow = prev
    }
  }, [open])
}

export function Modal({
  open,
  onClose,
  children,
  className,
  label,
  dismissable = true,
}: {
  open: boolean
  onClose: () => void
  children: ReactNode
  className?: string
  label?: string
  dismissable?: boolean
}) {
  useEscape(open, () => dismissable && onClose())
  if (!open) return null
  return createPortal(
    <div className="fixed inset-0 z-[70] flex items-end justify-center p-0 sm:items-center sm:p-4" role="dialog" aria-modal="true" aria-label={label}>
      <div className="absolute inset-0 animate-fade-in bg-[color-mix(in_oklab,var(--bg)_72%,transparent)] backdrop-blur-[3px]" onClick={() => dismissable && onClose()} />
      <div
        className={cn(
          'cq-zoom-in relative max-h-[92dvh] w-full overflow-y-auto rounded-t-3xl border border-border bg-surface shadow-lg sm:max-w-md sm:rounded-3xl',
          'pb-[env(safe-area-inset-bottom)]',
          className,
        )}
      >
        {children}
      </div>
    </div>,
    document.body,
  )
}

/** 二次确认对话框 */
export function ConfirmDialog({
  open,
  title,
  children,
  confirmLabel = '确定',
  cancelLabel = '取消',
  tone = 'primary',
  onConfirm,
  onCancel,
}: {
  open: boolean
  title: string
  children?: ReactNode
  confirmLabel?: string
  cancelLabel?: string
  tone?: 'primary' | 'danger'
  onConfirm: () => void
  onCancel: () => void
}) {
  return (
    <Modal open={open} onClose={onCancel} label={title}>
      <div className="p-5 sm:p-6">
        <div className="flex items-start gap-3">
          <span
            className={cn(
              'mt-0.5 inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-full',
              tone === 'danger' ? 'bg-danger-soft text-danger' : 'bg-warning-soft text-warning',
            )}
          >
            <AlertTriangle size={18} />
          </span>
          <div className="min-w-0">
            <h3 className="text-base font-semibold text-fg">{title}</h3>
            {children && <div className="mt-1.5 text-sm leading-relaxed text-muted">{children}</div>}
          </div>
        </div>
        <div className="mt-5 flex justify-end gap-2">
          <Button variant="secondary" onClick={onCancel} data-action="cancel">
            {cancelLabel}
          </Button>
          <Button variant={tone === 'danger' ? 'danger' : 'primary'} onClick={onConfirm} data-action="confirm" autoFocus>
            {confirmLabel}
          </Button>
        </div>
      </div>
    </Modal>
  )
}

/** 抽屉：桌面从右侧滑入，手机从底部升起 */
export function Drawer({
  open,
  onClose,
  title,
  icon,
  children,
  actions,
}: {
  open: boolean
  onClose: () => void
  title: ReactNode
  icon?: ReactNode
  children: ReactNode
  actions?: ReactNode
}) {
  useEscape(open, onClose)
  if (!open) return null
  return createPortal(
    <div className="fixed inset-0 z-[60]" role="dialog" aria-modal="true">
      <div className="absolute inset-0 animate-fade-in bg-[color-mix(in_oklab,var(--bg)_65%,transparent)] backdrop-blur-[2px]" onClick={onClose} />
      <aside
        className={cn(
          'absolute flex flex-col border-border bg-surface shadow-lg',
          'inset-x-0 bottom-0 max-h-[86dvh] rounded-t-3xl border-t cq-drawer-up',
          'md:inset-y-0 md:left-auto md:right-0 md:max-h-none md:w-[min(560px,92vw)] md:rounded-none md:border-l md:border-t-0 md:cq-drawer-right',
        )}
      >
        <div className="mx-auto mt-2 h-1.5 w-10 rounded-full bg-border-strong md:hidden" />
        <header className="flex items-center gap-2 border-b border-border px-5 py-3">
          {icon}
          <h2 className="min-w-0 flex-1 truncate text-base font-semibold">{title}</h2>
          {actions}
          <button type="button" onClick={onClose} className="rounded-lg p-1.5 text-muted hover:bg-surface-2 hover:text-fg" aria-label="关闭">
            <X size={18} />
          </button>
        </header>
        <div className="thin-scroll flex-1 overflow-y-auto px-5 py-4 pb-[calc(1rem+env(safe-area-inset-bottom))]">{children}</div>
      </aside>
    </div>,
    document.body,
  )
}

/** 错误边界：内容组件（可视化等）崩溃时不拖垮整页 */
export class ErrorBoundary extends Component<{ children: ReactNode; name?: string; compact?: boolean }, { err?: Error }> {
  state: { err?: Error } = {}
  static getDerivedStateFromError(err: Error) {
    return { err }
  }
  componentDidCatch(err: Error) {
    console.warn(`[${this.props.name ?? 'ErrorBoundary'}]`, err)
  }
  render() {
    if (!this.state.err) return this.props.children
    return (
      <div className={cn('rounded-2xl border border-danger/40 bg-danger-soft/60 text-sm', this.props.compact ? 'p-3' : 'p-5')}>
        <div className="flex items-center gap-2 font-semibold text-danger">
          <AlertTriangle size={16} /> {this.props.name ? `「${this.props.name}」` : ''}渲染出错
        </div>
        <p className="mt-1 break-all font-mono text-xs text-muted">{String(this.state.err.message ?? this.state.err)}</p>
        <Button size="sm" variant="secondary" className="mt-3" icon={<RotateCcw size={14} />} onClick={() => this.setState({ err: undefined })}>
          重试
        </Button>
      </div>
    )
  }
}
