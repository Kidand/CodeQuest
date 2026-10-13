/** 彩带特效：颜色取自设计 token；「减少动效」（设置或系统）时不播放 */
import confetti from 'canvas-confetti'
import { useProgress } from '@/lib/store'

function tokenColors(): string[] {
  const cs = getComputedStyle(document.documentElement)
  return ['--gold', '--accent', '--algo', '--llm', '--success', '--viz-4']
    .map((v) => cs.getPropertyValue(v).trim())
    .filter(Boolean)
}

export function motionReduced() {
  if (useProgress.getState().settings.reduceMotion) return true
  return typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
}

export function burst(kind: 'big' | 'small' = 'small') {
  if (motionReduced()) return
  const colors = tokenColors()
  const base = { colors, zIndex: 120, disableForReducedMotion: true, scalar: 0.9 }
  if (kind === 'small') {
    confetti({ ...base, particleCount: 70, spread: 70, startVelocity: 38, origin: { x: 0.5, y: 0.32 } })
    return
  }
  confetti({ ...base, particleCount: 90, angle: 60, spread: 60, startVelocity: 52, origin: { x: 0, y: 0.7 } })
  confetti({ ...base, particleCount: 90, angle: 120, spread: 60, startVelocity: 52, origin: { x: 1, y: 0.7 } })
  setTimeout(() => confetti({ ...base, particleCount: 80, spread: 100, startVelocity: 30, origin: { x: 0.5, y: 0.35 } }), 260)
}
