/**
 * 本章可视化共用的小型数值工具（纯函数，确定性）。
 */
import { rng } from '@/lib/utils'

/** 稳定 softmax；-Infinity 视为被遮；整行被遮时返回 NaN（与朴素实现行为一致） */
export function softmax(xs: number[]): number[] {
  const m = Math.max(...xs)
  if (m === -Infinity) return xs.map(() => NaN)
  const e = xs.map((x) => (x === -Infinity ? 0 : Math.exp(x - m)))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

/** 可复现的标准正态随机数生成器（Box–Muller） */
export function gauss(seed: number) {
  const r = rng(seed)
  return () => {
    const u = Math.max(1e-12, r())
    const v = r()
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
  }
}

export const dot = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0)

export function randMatrix(rows: number, cols: number, seed: number, scale = 1): number[][] {
  const g = gauss(seed)
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => g() * scale))
}

export function entropy(p: number[]) {
  return -p.reduce((s, x) => (x > 0 ? s + x * Math.log(x) : s), 0)
}

/** 字节数 → GiB / MiB 文本 */
export function fmtGiB(bytes: number) {
  if (bytes >= 2 ** 30) return `${(bytes / 2 ** 30).toFixed(2)} GiB`
  if (bytes >= 2 ** 20) return `${(bytes / 2 ** 20).toFixed(1)} MiB`
  return `${(bytes / 2 ** 10).toFixed(1)} KiB`
}
