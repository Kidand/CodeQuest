import { Monitor, Moon, Sun } from 'lucide-react'
import { useProgress, type ThemePref } from '@/lib/store'
import { cn } from '@/lib/utils'

const order: ThemePref[] = ['light', 'dark', 'system']
const meta: Record<ThemePref, { icon: typeof Sun; label: string }> = {
  light: { icon: Sun, label: '浅色' },
  dark: { icon: Moon, label: '深色' },
  system: { icon: Monitor, label: '跟随系统' },
}

/** 点击循环切换 浅色 → 深色 → 跟随系统 */
export function ThemeToggle({ className, showLabel = false }: { className?: string; showLabel?: boolean }) {
  const theme = useProgress((s) => s.settings.theme)
  const update = useProgress((s) => s.updateSettings)
  const next = order[(order.indexOf(theme) + 1) % order.length]
  const Icon = meta[theme].icon
  return (
    <button
      type="button"
      onClick={() => update({ theme: next })}
      title={`主题：${meta[theme].label}（点击切换为${meta[next].label}）`}
      aria-label={`切换主题，当前${meta[theme].label}`}
      className={cn('inline-flex h-9 items-center gap-2 rounded-xl px-2.5 text-muted transition-colors hover:bg-surface-2 hover:text-fg', className)}
    >
      <Icon size={18} />
      {showLabel && <span className="text-sm">{meta[theme].label}</span>}
    </button>
  )
}
