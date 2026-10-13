import { clsx, type ClassValue } from 'clsx'

/** className 拼接 */
export const cn = (...inputs: ClassValue[]) => clsx(inputs)

/** 可复现的伪随机数（mulberry32） */
export function rng(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Fisher–Yates 洗牌（返回新数组）；传 seed 则结果可复现 */
export function shuffle<T>(arr: readonly T[], seed?: number): T[] {
  const rand = seed == null ? Math.random : rng(seed)
  const a = arr.slice()
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1))
    ;[a[i], a[j]] = [a[j], a[i]]
  }
  return a
}

/** 洗牌但保证与原顺序不同（长度 ≥ 2 时） */
export function shuffleDifferent<T>(arr: readonly T[], seed?: number): T[] {
  if (arr.length < 2) return arr.slice()
  let s = seed
  for (let k = 0; k < 20; k++) {
    const out = shuffle(arr, s)
    if (out.some((v, i) => v !== arr[i])) return out
    s = s == null ? undefined : s + 1
  }
  return [...arr.slice(1), arr[0]]
}

export const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x))
export const range = (n: number) => Array.from({ length: n }, (_, i) => i)
export const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0)

/** 数值格式化：自动选择小数位 */
export function fmt(x: number, digits = 3) {
  if (!Number.isFinite(x)) return x > 0 ? '∞' : x < 0 ? '-∞' : 'NaN'
  if (x === 0) return '0'
  const ax = Math.abs(x)
  if (ax >= 1e5 || ax < 1e-3) return x.toExponential(2)
  return String(Number(x.toFixed(digits)))
}

/** 字节数 → 人类可读 */
export function fmtBytes(b: number) {
  const u = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
  let i = 0
  while (Math.abs(b) >= 1024 && i < u.length - 1) {
    b /= 1024
    i++
  }
  return `${Number(b.toFixed(2))} ${u[i]}`
}

/** 大数 → 1.2K / 3.4M / 5.6B */
export function fmtCount(n: number) {
  const u: [number, string][] = [
    [1e12, 'T'],
    [1e9, 'B'],
    [1e6, 'M'],
    [1e3, 'K'],
  ]
  for (const [v, s] of u) if (Math.abs(n) >= v) return `${Number((n / v).toFixed(2))}${s}`
  return String(n)
}

/** 秒 → mm:ss */
export function fmtDuration(sec: number) {
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

/** 文本归一化：用于填空 / 跟练答案比较 */
export function normalizeAnswer(s: string) {
  return s
    .replace(/[‘’]/g, "'")
    .replace(/[“”]/g, '"')
    .replace(/"/g, "'")
    .replace(/\s+/g, '')
    .replace(/[;；]+$/, '')
    .toLowerCase()
}

/**
 * 解析用户输入的数值表达式：支持 + - * / ^ ** ( )、科学计数法、
 * 以及后缀 K/M/G/T/B（十进制）与 Ki/Mi/Gi/Ti（二进制）。非法输入返回 NaN。
 */
export function parseNumberExpr(input: string): number {
  const s = input.trim().replace(/，/g, ',').replace(/,/g, '').replace(/×/g, '*').replace(/\*\*/g, '^')
  if (!s) return NaN
  let i = 0
  const peek = () => s[i]
  const skip = () => {
    while (s[i] === ' ') i++
  }
  const suffix = (): number => {
    const rest = s.slice(i)
    const m = rest.match(/^(Ki|Mi|Gi|Ti|K|M|G|T|B|k)(?![a-zA-Z])/)
    if (!m) return 1
    i += m[0].length
    const table: Record<string, number> = {
      k: 1e3, K: 1e3, M: 1e6, G: 1e9, B: 1e9, T: 1e12,
      Ki: 1024, Mi: 1024 ** 2, Gi: 1024 ** 3, Ti: 1024 ** 4,
    }
    return table[m[1]]
  }
  const num = (): number => {
    skip()
    const m = s.slice(i).match(/^(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?/)
    if (!m) return NaN
    i += m[0].length
    return parseFloat(m[0]) * suffix()
  }
  const atom = (): number => {
    skip()
    if (peek() === '(') {
      i++
      const v = expr()
      skip()
      if (peek() !== ')') return NaN
      i++
      return v * suffix()
    }
    if (peek() === '-') {
      i++
      return -atom()
    }
    if (peek() === '+') {
      i++
      return atom()
    }
    return num()
  }
  const power = (): number => {
    const b = atom()
    skip()
    if (peek() === '^') {
      i++
      return Math.pow(b, power())
    }
    return b
  }
  const term = (): number => {
    let v = power()
    for (;;) {
      skip()
      const c = peek()
      if (c === '*' || c === '/') {
        i++
        const r = power()
        v = c === '*' ? v * r : v / r
      } else return v
    }
  }
  const expr = (): number => {
    let v = term()
    for (;;) {
      skip()
      const c = peek()
      if (c === '+' || c === '-') {
        i++
        const r = term()
        v = c === '+' ? v + r : v - r
      } else return v
    }
  }
  const v = expr()
  skip()
  return i === s.length ? v : NaN
}

/** 相对误差比较 */
export function approxEqual(a: number, b: number, rel = 0.01) {
  if (!Number.isFinite(a) || !Number.isFinite(b)) return false
  if (b === 0) return Math.abs(a) <= rel
  return Math.abs(a - b) <= rel * Math.abs(b)
}
