/**
 * 可视化配色：一律返回 CSS 变量表达式，自动适配明暗主题。
 */
export type VizTone =
  | 'accent' | 'success' | 'warning' | 'danger' | 'info' | 'muted' | 'gold' | 'algo' | 'llm'
  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8

const named: Record<string, string> = {
  accent: 'var(--accent)',
  success: 'var(--success)',
  warning: 'var(--warning)',
  danger: 'var(--danger)',
  info: 'var(--info)',
  muted: 'var(--subtle)',
  gold: 'var(--gold)',
  algo: 'var(--algo)',
  llm: 'var(--llm)',
}

/** 实色（描边、文字、图形填充） */
export function toneColor(t: VizTone): string {
  return typeof t === 'number' ? `var(--viz-${((t - 1) % 8) + 1})` : named[t]
}

/** 浅色底（单元格背景等）：amount 为实色占比 0-100 */
export function toneSoft(t: VizTone, amount = 22): string {
  return `color-mix(in oklab, ${toneColor(t)} ${amount}%, var(--surface))`
}

/** 第 i 个分类色（0 起） */
export const catColor = (i: number) => `var(--viz-${(i % 8) + 1})`

/** 顺序热力色：t∈[0,1]，0=背景，1=--heat */
export function heatColor(t: number): string {
  const p = Math.round(Math.max(0, Math.min(1, t)) * 92)
  return `color-mix(in oklab, var(--heat) ${p}%, var(--surface))`
}

/** 发散热力色：v∈[-1,1]，负值 --heat-neg，正值 --heat */
export function divergingColor(v: number): string {
  const p = Math.round(Math.min(1, Math.abs(v)) * 90)
  return `color-mix(in oklab, ${v < 0 ? 'var(--heat-neg)' : 'var(--heat)'} ${p}%, var(--surface))`
}

/** 背景较深时文字应使用反色 */
export const textOnHeat = (t: number) => (Math.abs(t) > 0.55 ? 'var(--accent-fg)' : 'var(--fg)')
