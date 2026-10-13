/**
 * 本章可视化共用的小数学工具（纯函数、确定性）。
 */
import { rng } from '@/lib/utils'

export const softmax = (z: number[]) => {
  const m = Math.max(...z)
  const e = z.map((v) => Math.exp(v - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x))
/** softplus 的数值稳定写法：log(1+e^x) = max(x,0) + log1p(e^{-|x|}) */
export const softplus = (x: number) => Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)))
export const sqrtSoftplus = (x: number) => Math.sqrt(softplus(x))

/** 取最大的 k 个下标（按值降序；值相同时下标小的优先） */
export function topk(v: number[], k: number): number[] {
  return v
    .map((x, i) => [x, i] as const)
    .sort((a, b) => b[0] - a[0] || a[1] - b[1])
    .slice(0, k)
    .map(([, i]) => i)
}

/** 确定性标准正态（Box-Muller） */
export function normals(seed: number, n: number): number[] {
  const r = rng(seed)
  const out: number[] = []
  while (out.length < n) {
    const u = Math.max(1e-12, r())
    const v = r()
    const m = Math.sqrt(-2 * Math.log(u))
    out.push(m * Math.cos(2 * Math.PI * v), m * Math.sin(2 * Math.PI * v))
  }
  return out.slice(0, n)
}

/** 确定性 [lo, hi) 均匀数 */
export function uniforms(seed: number, n: number, lo = -1, hi = 1): number[] {
  const r = rng(seed)
  return Array.from({ length: n }, () => lo + (hi - lo) * r())
}

export const sumArr = (xs: number[]) => xs.reduce((a, b) => a + b, 0)
export const mean = (xs: number[]) => (xs.length ? sumArr(xs) / xs.length : 0)
/** 变异系数 std/mean（负载不均衡度） */
export const cv = (xs: number[]) => {
  const m = mean(xs)
  if (m === 0) return 0
  return Math.sqrt(mean(xs.map((x) => (x - m) ** 2))) / m
}
/** MaxVio = (max load − mean load) / mean load（Loss-Free Balancing 论文的指标） */
export const maxVio = (xs: number[]) => {
  const m = mean(xs)
  return m === 0 ? 0 : (Math.max(...xs) - m) / m
}

/** 组合数 C(n, k)（浮点，n ≤ 1000 足够） */
export function comb(n: number, k: number): number {
  if (k < 0 || k > n) return 0
  k = Math.min(k, n - k)
  let r = 1
  for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i
  return r
}

export const fmtB = (n: number) =>
  n >= 1e12 ? `${(n / 1e12).toFixed(2)}T` : n >= 1e9 ? `${(n / 1e9).toFixed(n >= 1e11 ? 1 : 2)}B` : `${(n / 1e6).toFixed(1)}M`
