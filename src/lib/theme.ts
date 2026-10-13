/**
 * 主题：light / dark / system。偏好同时写入 localStorage('cq-theme')，
 * 供 index.html 内联脚本在首屏前读取，避免闪烁。
 */
import { useEffect, useState } from 'react'
import { useProgress, type ThemePref } from './store'

const mq = () => window.matchMedia('(prefers-color-scheme: dark)')

export function resolveDark(pref: ThemePref) {
  return pref === 'dark' || (pref === 'system' && mq().matches)
}

export function applyTheme(pref: ThemePref) {
  const dark = resolveDark(pref)
  document.documentElement.classList.toggle('dark', dark)
  try {
    localStorage.setItem('cq-theme', pref)
  } catch {}
}

/** 在 App 根部调用一次：同步主题偏好与系统变化 */
export function useThemeSync() {
  const pref = useProgress((s) => s.settings.theme)
  const reduceMotion = useProgress((s) => s.settings.reduceMotion)
  useEffect(() => {
    applyTheme(pref)
    if (pref !== 'system') return
    const m = mq()
    const on = () => applyTheme('system')
    m.addEventListener('change', on)
    return () => m.removeEventListener('change', on)
  }, [pref])
  useEffect(() => {
    document.documentElement.classList.toggle('reduce-motion', reduceMotion)
  }, [reduceMotion])
}

/** 当前是否暗色（可视化里需要根据主题切换绘制逻辑时使用；颜色本身优先用 CSS 变量） */
export function useIsDark() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains('dark'))
  useEffect(() => {
    const obs = new MutationObserver(() => setDark(document.documentElement.classList.contains('dark')))
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
    return () => obs.disconnect()
  }, [])
  return dark
}
