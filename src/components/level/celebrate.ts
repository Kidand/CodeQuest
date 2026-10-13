/** 庆祝动效：confetti 颜色取自设计 token；遵循「减少动效」设置与系统偏好 */
import { useProgress } from '@/lib/store'

export function prefersReducedMotion() {
  if (useProgress.getState().settings.reduceMotion) return true
  try {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches
  } catch {
    return false
  }
}

function tokenColors() {
  const cs = getComputedStyle(document.documentElement)
  return ['--gold', '--accent', '--success', '--viz-3', '--viz-4', '--viz-6']
    .map((v) => cs.getPropertyValue(v).trim())
    .filter(Boolean)
}

export async function celebrate(power: 'small' | 'big' = 'small') {
  if (prefersReducedMotion()) return
  try {
    const { default: confetti } = await import('canvas-confetti')
    const colors = tokenColors()
    const base = { colors, disableForReducedMotion: true, zIndex: 80 }
    if (power === 'small') {
      confetti({ ...base, particleCount: 70, spread: 70, startVelocity: 38, origin: { y: 0.7 } })
    } else {
      confetti({ ...base, particleCount: 90, spread: 80, origin: { x: 0.25, y: 0.65 }, angle: 60 })
      confetti({ ...base, particleCount: 90, spread: 80, origin: { x: 0.75, y: 0.65 }, angle: 120 })
      setTimeout(() => confetti({ ...base, particleCount: 60, spread: 120, startVelocity: 30, origin: { y: 0.5 } }), 280)
    }
  } catch {
    /* 动效失败不影响功能 */
  }
}
