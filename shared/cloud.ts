/** Shared, side-effect-free cloud archive contract. No browser or store dependencies. */
import type { ChallengeRecord, LevelProgress, LevelShape } from '../src/lib/game'

// D1 has a 2,000,000-byte row limit. Leave room for the row metadata and encoding.
export const MAX_CLOUD_BYTES = 1_500_000
export interface CloudUser { id: string; username: string; email: string }
export interface CloudSessionResponse { user: CloudUser | null; signInUrl: string; signOutUrl: string }
export interface CloudStateResponse { user: CloudUser; revision: number; updatedAt: string | null; snapshot: CloudSnapshot | null }
export interface ProgressData {
  xp: number
  levels: Record<string, LevelProgress>
  streak: { current: number; best: number; lastDate?: string }
  daily: Record<string, { xp: number; answered: number; correct: number; solved: number }>
  achievements: Record<string, string>
  review: Record<string, { kind: 'question' | 'qa'; levelId: string; itemId: string; source: string; wrongCount: number; streak: number; interval: number; due: string; lastAt: string; mastered: boolean }>
  settings: { theme: 'light' | 'dark' | 'system'; freeMode: boolean; editorFontSize: number; reduceMotion: boolean; dailyGoal: number; sound: boolean }
  lastVisited?: { levelId: string; stage: string }
  createdAt: string
}
export interface CloudMockConfig { durationMin: 30 | 45 | 60; algoChallenges: number; llmChallenges: number; concepts: number; scope: 'unlocked' | 'all'; tiers: ('must' | 'high' | 'advanced' | 'foundation' | 'modern' | 'frontier')[] }
export interface CloudMockItem { kind: 'challenge' | 'question'; levelId: string; itemId: string; track: 'algo' | 'llm'; title: string; difficulty?: 'easy' | 'medium' | 'hard' }
export interface CloudMockResult { status: 'pending' | 'correct' | 'wrong' | 'passed' | 'skipped'; timeSec: number }
export interface CloudMockSession { id: string; config: CloudMockConfig; items: CloudMockItem[]; results: CloudMockResult[]; startedAt: number; deadline: number; active: number; activeSince: number }
export interface CloudMockRecord { id: string; startedAt: string; finishedAt: string; durationMin: number; usedSec: number; score: number; timeUp: boolean; xpAwarded: number; items: (CloudMockItem & CloudMockResult)[] }
export interface CloudSnapshot {
  app: 'codequest'; version: 2; state: ProgressData; drafts: Record<string, string>
  mock: { active: CloudMockSession | null; lastConfig: CloudMockConfig; history: CloudMockRecord[] }
}

export class CloudValidationError extends Error {
  constructor(public readonly code: 'invalid_snapshot' | 'payload_too_large' = 'invalid_snapshot') {
    super(code === 'payload_too_large' ? '存档超过云端容量限制，请先导出备份并清理不需要的代码草稿。' : '存档格式不完整或含有无效数据。')
    this.name = 'CloudValidationError'
  }
}
type Obj = Record<string, unknown>
const fail = (): never => { throw new CloudValidationError() }
const badKeys = new Set(['__proto__', 'prototype', 'constructor'])
function obj(value: unknown, allowed?: string[]): Obj {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return fail()
  const proto = Object.getPrototypeOf(value)
  if (proto !== Object.prototype && proto !== null) return fail()
  if (Object.keys(value).some(k => badKeys.has(k) || (allowed && !allowed.includes(k)))) return fail()
  return value as Obj
}
function str(value: unknown, max = 256, min = 1): string {
  if (typeof value !== 'string' || value.length < min || value.length > max) return fail()
  return value
}
function num(value: unknown, max = 100_000_000, min = 0, integer = true): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max || (integer && !Number.isSafeInteger(value))) return fail()
  return value
}
function bool(value: unknown): boolean { return typeof value === 'boolean' ? value : fail() }
function choice<T extends string | number>(value: unknown, options: readonly T[]): T { return options.includes(value as T) ? value as T : fail() }
function date(value: unknown): string {
  const text = str(value, 10)
  if (!/^\d{4}-\d{2}-\d{2}$/.test(text) || !Number.isFinite(Date.parse(text)) || new Date(text).toISOString().slice(0, 10) !== text) return fail()
  return text
}
function instant(value: unknown): string {
  const text = str(value, 40)
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?Z$/.test(text) || !Number.isFinite(Date.parse(text))) return fail()
  date(text.slice(0, 10))
  return text
}
function arr<T>(value: unknown, max: number, fn: (item: unknown) => T): T[] {
  if (!Array.isArray(value) || value.length > max) return fail()
  return Array.from(value, item => fn(item))
}
const ids = (value: unknown) => arr(value, 2_000, v => str(v))
function map<T>(value: unknown, max: number, fn: (item: unknown) => T, keyMax = 256): Record<string, T> {
  const entries = Object.entries(obj(value))
  if (entries.length > max) return fail()
  return Object.fromEntries(entries.sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([key, item]) => [str(key, keyMax), fn(item)]))
}
const optional = <T>(value: unknown, fn: (value: unknown) => T): T | undefined => value === undefined ? undefined : fn(value)

function challenge(value: unknown): ChallengeRecord {
  const p = obj(value, ['passed', 'usedSolution', 'hintsUsed', 'attempts', 'passedAt', 'bestTimeSec', 'difficulty'])
  return { passed: bool(p.passed), usedSolution: bool(p.usedSolution), hintsUsed: num(p.hintsUsed, 10_000), attempts: num(p.attempts), passedAt: optional(p.passedAt, instant), bestTimeSec: optional(p.bestTimeSec, v => num(v, 31_536_000, 0, false)), difficulty: optional(p.difficulty, v => choice(v, ['easy', 'medium', 'hard'])) }
}
function shape(value: unknown): LevelShape {
  const p = obj(value, ['sections', 'quiz', 'guided', 'challenges', 'qa'])
  return { sections: num(p.sections, 2_000), quiz: num(p.quiz, 2_000), guided: ids(p.guided), challenges: ids(p.challenges), qa: optional(p.qa, v => num(v, 2_000)) }
}
function level(value: unknown): LevelProgress {
  const p = obj(value, ['shape', 'sectionsRead', 'checkpointsCorrect', 'learnDone', 'quizBestCorrect', 'quizBest', 'quizAttempts', 'guidedDone', 'challenges', 'qaRated', 'firstCompletedAt', 'lastVisitedAt'])
  return { shape: optional(p.shape, shape), sectionsRead: ids(p.sectionsRead), checkpointsCorrect: ids(p.checkpointsCorrect), learnDone: bool(p.learnDone), quizBestCorrect: num(p.quizBestCorrect, 2_000), quizBest: num(p.quizBest, 1, 0, false), quizAttempts: num(p.quizAttempts), guidedDone: ids(p.guidedDone), challenges: map(p.challenges, 2_000, challenge), qaRated: ids(p.qaRated), firstCompletedAt: optional(p.firstCompletedAt, instant), lastVisitedAt: optional(p.lastVisitedAt, instant) }
}
function progress(value: unknown): ProgressData {
  const p = obj(value, ['xp', 'levels', 'streak', 'daily', 'achievements', 'review', 'settings', 'lastVisited', 'createdAt'])
  const s = obj(p.streak, ['current', 'best', 'lastDate'])
  const setting = obj(p.settings, ['theme', 'freeMode', 'editorFontSize', 'reduceMotion', 'dailyGoal', 'sound'])
  return {
    xp: num(p.xp), levels: map(p.levels, 2_000, level),
    streak: { current: num(s.current, 100_000), best: num(s.best, 100_000), lastDate: optional(s.lastDate, date) },
    daily: map(p.daily, 36_600, value => { const d = obj(value, ['xp', 'answered', 'correct', 'solved']); return { xp: num(d.xp), answered: num(d.answered), correct: num(d.correct), solved: num(d.solved) } }),
    achievements: map(p.achievements, 2_000, instant),
    review: map(p.review, 20_000, value => { const r = obj(value, ['kind', 'levelId', 'itemId', 'source', 'wrongCount', 'streak', 'interval', 'due', 'lastAt', 'mastered']); return { kind: choice(r.kind, ['question', 'qa']), levelId: str(r.levelId), itemId: str(r.itemId), source: str(r.source, 64), wrongCount: num(r.wrongCount), streak: num(r.streak), interval: num(r.interval, 100_000), due: date(r.due), lastAt: date(r.lastAt), mastered: bool(r.mastered) } }, 768),
    settings: { theme: choice(setting.theme, ['light', 'dark', 'system']), freeMode: bool(setting.freeMode), editorFontSize: num(setting.editorFontSize, 30, 10), reduceMotion: bool(setting.reduceMotion), dailyGoal: num(setting.dailyGoal, 1_000_000, 1), sound: bool(setting.sound) },
    lastVisited: optional(p.lastVisited, value => { const v = obj(value, ['levelId', 'stage']); return { levelId: str(v.levelId), stage: str(v.stage, 64) } }),
    createdAt: instant(p.createdAt),
  }
}
function mockConfig(value: unknown): CloudMockConfig {
  const c = obj(value, ['durationMin', 'algoChallenges', 'llmChallenges', 'concepts', 'scope', 'tiers'])
  return { durationMin: choice(c.durationMin, [30, 45, 60]), algoChallenges: num(c.algoChallenges, 100), llmChallenges: num(c.llmChallenges, 100), concepts: num(c.concepts, 200), scope: choice(c.scope, ['unlocked', 'all']), tiers: arr(c.tiers, 6, v => choice(v, ['must', 'high', 'advanced', 'foundation', 'modern', 'frontier'])) }
}
function mockItem(value: unknown, result = false): CloudMockItem {
  const it = obj(value, ['kind', 'levelId', 'itemId', 'track', 'title', 'difficulty', ...(result ? ['status', 'timeSec'] : [])])
  return { kind: choice(it.kind, ['challenge', 'question']), levelId: str(it.levelId), itemId: str(it.itemId), track: choice(it.track, ['algo', 'llm']), title: str(it.title, 1_000), difficulty: optional(it.difficulty, v => choice(v, ['easy', 'medium', 'hard'])) }
}
function mockResult(value: unknown, item = false): CloudMockResult {
  const r = obj(value, ['status', 'timeSec', ...(item ? ['kind', 'levelId', 'itemId', 'track', 'title', 'difficulty'] : [])])
  return { status: choice(r.status, ['pending', 'correct', 'wrong', 'passed', 'skipped']), timeSec: num(r.timeSec, 31_536_000, 0, false) }
}
function mockSession(value: unknown): CloudMockSession {
  const s = obj(value, ['id', 'config', 'items', 'results', 'startedAt', 'deadline', 'active', 'activeSince'])
  const items = arr(s.items, 400, mockItem)
  const results = arr(s.results, 400, mockResult)
  if (!items.length || results.length !== items.length) return fail()
  const startedAt = num(s.startedAt, 8_640_000_000_000_000)
  const deadline = num(s.deadline, 8_640_000_000_000_000, startedAt)
  return { id: str(s.id), config: mockConfig(s.config), items, results, startedAt, deadline, active: num(s.active, items.length - 1), activeSince: num(s.activeSince, 8_640_000_000_000_000, startedAt) }
}
function mockRecord(value: unknown): CloudMockRecord {
  const r = obj(value, ['id', 'startedAt', 'finishedAt', 'durationMin', 'usedSec', 'score', 'timeUp', 'xpAwarded', 'items'])
  return { id: str(r.id), startedAt: instant(r.startedAt), finishedAt: instant(r.finishedAt), durationMin: num(r.durationMin, 1_440), usedSec: num(r.usedSec, 31_536_000), score: num(r.score, 100), timeUp: bool(r.timeUp), xpAwarded: num(r.xpAwarded), items: arr(r.items, 400, value => ({ ...mockItem(value, true), ...mockResult(value, true) })) }
}

/** Validate and copy only the v2 persisted data contract, with deterministic map order. */
export function normalizeCloudSnapshot(input: unknown): CloudSnapshot {
  const s = obj(input, ['app', 'version', 'state', 'drafts', 'mock'])
  if (s.app !== 'codequest' || s.version !== 2) return fail()
  const m = obj(s.mock, ['active', 'lastConfig', 'history'])
  const result: CloudSnapshot = {
    app: 'codequest', version: 2, state: progress(s.state),
    drafts: map(s.drafts, 2_000, value => str(value, 200_000, 0), 768),
    mock: { active: m.active === null ? null : mockSession(m.active), lastConfig: mockConfig(m.lastConfig), history: arr(m.history, 50, mockRecord) },
  }
  // Dates are keys in daily activity and are used by calendar and streak logic.
  Object.keys(result.state.daily).forEach(date)
  if (new TextEncoder().encode(JSON.stringify(result)).byteLength > MAX_CLOUD_BYTES) throw new CloudValidationError('payload_too_large')
  return result
}
