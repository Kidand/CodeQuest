/**
 * 题目判定（纯函数）。QuestionView 与其它需要判分的地方（如模拟面试报告）共用。
 */
import type { FillBlank, Question } from '@/content/types'
import { approxEqual, normalizeAnswer, parseNumberExpr, range, shuffleDifferent } from '@/lib/utils'

/** 每种题型的作答状态 */
export interface AnswerMap {
  single: number | null
  multi: number[]
  judge: boolean | null
  fill: string[]
  numeric: string
  /** 当前显示顺序（元素为 items 的原始下标） */
  order: number[]
  /** 左列 i 配对到的右列原始下标（pairs 下标），未配对为 null */
  match: (number | null)[]
  /** 答案区中的行（池子 id：0..lines-1 为正确行，之后为干扰行） */
  parsons: number[]
  /** 选中的行号（1 起始） */
  bug: number[]
  pick: number[]
}
export type AnswerOf<Q extends Question> = AnswerMap[Q['type']]
export type AnyAnswer = AnswerMap[keyof AnswerMap]

/** 由 seed 决定的展示布局（洗牌结果） */
export interface QLayout {
  /** order：初始顺序 */
  order?: number[]
  /** match：右列显示顺序 */
  right?: number[]
  /** parsons：候选池显示顺序 */
  pool?: number[]
}

/** 题目内容指纹：同一题内容不变时洗牌结果稳定 */
function hashStr(s: string) {
  let h = 2166136261
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619)
  return h >>> 0
}

export function makeLayout(q: Question, seed = 0): QLayout {
  const s = (hashStr(q.id) + seed * 7919) >>> 0
  switch (q.type) {
    case 'order':
      return { order: shuffleDifferent(range(q.items.length), s) }
    case 'match':
      return { right: shuffleDifferent(range(q.pairs.length), s) }
    case 'parsons':
      return { pool: shuffleDifferent(range(q.lines.length + (q.distractors?.length ?? 0)), s) }
    default:
      return {}
  }
}

export function initialAnswer(q: Question, layout: QLayout): AnyAnswer {
  switch (q.type) {
    case 'single':
      return null
    case 'multi':
      return []
    case 'judge':
      return null
    case 'fill':
      return q.blanks.map(() => '')
    case 'numeric':
      return ''
    case 'order':
      return layout.order ?? range(q.items.length)
    case 'match':
      return q.pairs.map(() => null)
    case 'parsons':
      return []
    case 'bug':
      return []
    case 'pick':
      return []
  }
}

/** 正确答案（复习时“只读展示”用） */
export function correctAnswer(q: Question): AnyAnswer {
  switch (q.type) {
    case 'single':
      return q.answer
    case 'multi':
      return [...q.answer]
    case 'judge':
      return q.answer
    case 'fill':
      return q.blanks.map((b) => b.answers[0] ?? '')
    case 'numeric':
      return String(q.answer)
    case 'order':
      return range(q.items.length)
    case 'match':
      return range(q.pairs.length)
    case 'parsons':
      return range(q.lines.length)
    case 'bug':
      return [...q.bugLines]
    case 'pick':
      return [...q.answer]
  }
}

const sameSet = (a: number[], b: number[]) => {
  const A = new Set(a)
  const B = new Set(b)
  return A.size === B.size && [...A].every((x) => B.has(x))
}

export function blankCorrect(b: FillBlank, input: string): boolean {
  if (!input.trim()) return false
  if (b.numeric) {
    const x = parseNumberExpr(input)
    if (!Number.isFinite(x)) return false
    return b.answers.some((a) => approxEqual(x, parseNumberExpr(a), b.tolerance ?? 0.01))
  }
  const n = normalizeAnswer(input)
  return b.answers.some((a) => normalizeAnswer(a) === n)
}

/** parsons：池子 id → 文本 */
export function parsonsText(q: Extract<Question, { type: 'parsons' }>, id: number) {
  return id < q.lines.length ? q.lines[id] : (q.distractors ?? [])[id - q.lines.length]
}

export function isCorrect(q: Question, v: AnyAnswer): boolean {
  switch (q.type) {
    case 'single':
      return v === q.answer
    case 'multi':
      return sameSet(v as number[], q.answer)
    case 'judge':
      return v === q.answer
    case 'fill':
      return q.blanks.every((b, i) => blankCorrect(b, (v as string[])[i] ?? ''))
    case 'numeric':
      return approxEqual(parseNumberExpr(v as string), q.answer, q.tolerance ?? 0.01)
    case 'order':
      return (v as number[]).every((x, i) => x === i)
    case 'match':
      return (v as (number | null)[]).every((x, i) => x === i)
    case 'parsons': {
      const ids = v as number[]
      return ids.length === q.lines.length && ids.every((id, i) => parsonsText(q, id) === q.lines[i])
    }
    case 'bug':
      return sameSet(v as number[], q.bugLines)
    case 'pick':
      return sameSet(v as number[], q.answer)
  }
}

/** 是否已作答到可以提交；返回 null 表示可提交，否则为提示文案 */
export function submitBlocker(q: Question, v: AnyAnswer): string | null {
  switch (q.type) {
    case 'single':
    case 'judge':
      return v == null ? '先选择一个答案' : null
    case 'multi':
      return (v as number[]).length === 0 ? '至少选择一项' : null
    case 'fill':
      return (v as string[]).some((s) => !s.trim()) ? '还有空没填' : null
    case 'numeric': {
      const s = (v as string).trim()
      if (!s) return '先输入一个数值'
      return Number.isFinite(parseNumberExpr(s)) ? null : '无法解析这个数值'
    }
    case 'order':
      return null
    case 'match':
      return (v as (number | null)[]).some((x) => x == null) ? '还有未配对的项' : null
    case 'parsons':
      return (v as number[]).length === 0 ? '先把代码行放进答案区' : null
    case 'bug':
      return (v as number[]).length === 0 ? '先点出有问题的行' : null
    case 'pick':
      return (v as number[]).length === 0 ? '至少选择一项' : null
  }
}

/** 数值预览文案：parseNumberExpr 的结果，便于用户确认 */
export function formatParsed(x: number): string {
  if (!Number.isFinite(x)) return ''
  const ax = Math.abs(x)
  if (Number.isInteger(x) && ax < 1e15) {
    const s = x.toLocaleString('en-US')
    return ax >= 1e6 ? `${s}（≈ ${x.toExponential(3).replace(/\.?0+e/, 'e')}）` : s
  }
  if (ax !== 0 && (ax >= 1e9 || ax < 1e-4)) return x.toExponential(4).replace(/\.?0+e/, 'e')
  return String(Number(x.toPrecision(8)))
}
