export function PageSpinner({ label = '加载中…' }: { label?: string }) {
  return (
    <div className="flex min-h-[40vh] items-center justify-center gap-3 text-sm text-muted">
      <span className="h-5 w-5 animate-spin rounded-full border-2 border-border-strong border-t-accent" />
      {label}
    </div>
  )
}
