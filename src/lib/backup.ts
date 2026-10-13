/** Versioned portable learning archive. Password hashes and local account IDs are never exported. */
import { emptyLevelProgress, type LevelProgress } from './game'
import { exportProgress, initialData, useDrafts, useEvents, useProgress, type ProgressState } from './store'
import { DEFAULT_MOCK_CONFIG, useMock } from './mock'
import { setAccountSwitching } from './account-storage'

function object(value: unknown): value is Record<string, any> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}
const number = (v: unknown) => typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1e12
const strings = (v: unknown) => Array.isArray(v) && v.every((x) => typeof x === 'string')
const date = (v: unknown) => typeof v === 'string' && Number.isFinite(Date.parse(v))
function requireValue(condition: unknown, name: string): asserts condition {
  if (!condition) throw new Error(`备份中的「${name}」格式不正确，当前存档未被修改。`)
}
function safeEntries(value: unknown, name: string): [string, any][] {
  requireValue(object(value), name)
  const entries = Object.entries(value)
  requireValue(entries.every(([key]) => !['__proto__', 'constructor', 'prototype'].includes(key)), name)
  return entries
}

export function parseProgress(src: unknown) {
  requireValue(object(src) && number(src.xp), '进度')
  const data = initialData()
  const levels: Record<string, LevelProgress> = {}
  for (const [id, raw] of safeEntries(src.levels, '关卡')) {
    requireValue(object(raw), '关卡')
    const p = { ...emptyLevelProgress(), ...raw }
    requireValue((['sectionsRead', 'checkpointsCorrect', 'guidedDone', 'qaRated'] as const).every((k) => strings(p[k])) && typeof p.learnDone === 'boolean', '关卡阶段')
    requireValue((['quizBestCorrect', 'quizBest', 'quizAttempts'] as const).every((k) => number(p[k])) && p.quizBest <= 1, '关卡分数')
    for (const [, c] of safeEntries(p.challenges, '实战记录')) requireValue(object(c) && typeof c.passed === 'boolean' && typeof c.usedSolution === 'boolean' && number(c.hintsUsed) && number(c.attempts), '实战记录')
    if (p.shape) requireValue(object(p.shape) && number(p.shape.sections) && number(p.shape.quiz) && strings(p.shape.guided) && strings(p.shape.challenges), '关卡结构')
    levels[id] = p as LevelProgress
  }
  const streak = src.streak ?? data.streak
  requireValue(object(streak) && number(streak.current) && number(streak.best) && (!streak.lastDate || date(streak.lastDate)), '连续学习')
  for (const [day, stat] of safeEntries(src.daily ?? {}, '每日记录')) requireValue(date(day) && object(stat) && ['xp', 'answered', 'correct', 'solved'].every((k) => number(stat[k])), '每日记录')
  for (const [, v] of safeEntries(src.achievements ?? {}, '成就')) requireValue(date(v), '成就')
  for (const [, r] of safeEntries(src.review ?? {}, '复习')) {
    requireValue(object(r) && ['question', 'qa'].includes(r.kind) && typeof r.levelId === 'string' && typeof r.itemId === 'string' && typeof r.source === 'string' && date(r.due) && date(r.lastAt) && typeof r.mastered === 'boolean' && ['wrongCount', 'streak', 'interval'].every((k) => number(r[k])), '复习')
  }
  const s = { ...data.settings, ...(object(src.settings) ? src.settings : {}) }
  requireValue(['light', 'dark', 'system'].includes(s.theme) && typeof s.freeMode === 'boolean' && typeof s.reduceMotion === 'boolean' && number(s.dailyGoal) && s.dailyGoal > 0 && number(s.editorFontSize) && s.editorFontSize >= 10 && s.editorFontSize <= 30, '设置')
  requireValue(src.lastVisited === undefined || (object(src.lastVisited) && typeof src.lastVisited.levelId === 'string' && typeof src.lastVisited.stage === 'string'), '最近学习')
  requireValue(src.createdAt === undefined || date(src.createdAt), '创建时间')
  // Explicit whitelist prevents imported JSON from replacing Zustand actions.
  return { ...data, xp: src.xp as number, levels, streak: streak as ProgressState['streak'], daily: src.daily ?? {}, achievements: src.achievements ?? {}, review: src.review ?? {}, settings: s, lastVisited: src.lastVisited, createdAt: src.createdAt ?? data.createdAt }
}

export function exportBackup() {
  const m = useMock.getState()
  return JSON.stringify({ app: 'codequest', version: 2, exportedAt: new Date().toISOString(), state: JSON.parse(exportProgress()).state, drafts: useDrafts.getState().drafts, mock: { active: m.active, history: m.history, lastConfig: m.lastConfig } }, null, 2)
}

export function importBackup(text: string) {
  requireValue(text.length <= 20_000_000, '文件大小（最多 20 MB）')
  const file = JSON.parse(text)
  requireValue(object(file) && (!file.app || file.app === 'codequest') && (file.version === undefined || [1, 2].includes(file.version)), '文件版本')
  const state = parseProgress(file.state ?? file)
  const drafts = file.version === 2 ? file.drafts : useDrafts.getState().drafts
  requireValue(safeEntries(drafts, '代码草稿').every(([, code]) => typeof code === 'string'), '代码草稿')
  const mock = file.version === 2 ? file.mock : null
  if (mock) {
    requireValue(object(mock) && Array.isArray(mock.history) && mock.history.length <= 50, '面试历史')
    const validItem = (v: any) => object(v) && ['challenge', 'question'].includes(v.kind) && typeof v.levelId === 'string' && typeof v.itemId === 'string' && typeof v.title === 'string' && ['algo', 'llm'].includes(v.track)
    const validResult = (v: any) => object(v) && ['pending', 'correct', 'wrong', 'passed', 'skipped'].includes(v.status) && number(v.timeSec)
    const validConfig = (v: any) => object(v) && [30, 45, 60].includes(v.durationMin) && ['unlocked', 'all'].includes(v.scope) && strings(v.tiers) && ['algoChallenges', 'llmChallenges', 'concepts'].every((key) => number(v[key]) && Number.isInteger(v[key]) && v[key] <= 20)
    requireValue(validConfig(mock.lastConfig), '面试设置')
    for (const r of mock.history) requireValue(object(r) && typeof r.id === 'string' && date(r.startedAt) && date(r.finishedAt) && number(r.score) && r.score <= 100 && number(r.usedSec) && number(r.durationMin) && number(r.xpAwarded) && typeof r.timeUp === 'boolean' && Array.isArray(r.items) && r.items.every((x: unknown) => validItem(x) && validResult(x)), '面试历史')
    const a = mock.active
    if (a !== null) requireValue(object(a) && typeof a.id === 'string' && validConfig(a.config) && Array.isArray(a.items) && a.items.length > 0 && a.items.every(validItem) && Array.isArray(a.results) && a.results.length === a.items.length && a.results.every(validResult) && number(a.active) && Number.isInteger(a.active) && a.active < a.items.length && ['startedAt', 'deadline', 'activeSince'].every((k) => typeof a[k] === 'number' && Number.isFinite(a[k])), '进行中的面试')
  }
  // Commit raw storage first; quota failures restore every namespace before any live store changes.
  const writes = [{ key: useProgress.persist.getOptions().name!, value: JSON.stringify({ state, version: 1 }) }]
  if (file.version === 2) {
    writes.push({ key: useDrafts.persist.getOptions().name!, value: JSON.stringify({ state: { drafts }, version: 0 }) })
    writes.push({ key: useMock.persist.getOptions().name!, value: JSON.stringify({ state: mock ? { active: mock.active, history: mock.history, lastConfig: mock.lastConfig } : { active: null, history: [], lastConfig: DEFAULT_MOCK_CONFIG }, version: 1 }) })
  }
  const previous = writes.map(({ key }) => ({ key, value: localStorage.getItem(key) }))
  try {
    for (const { key, value } of writes) localStorage.setItem(key, value)
  } catch {
    for (const { key } of writes) localStorage.removeItem(key)
    for (const { key, value } of previous) if (value !== null) localStorage.setItem(key, value)
    throw new Error('浏览器存储空间不足或不可写，导入已取消，原存档已保留。请减少代码草稿或改用其它浏览器。')
  }
  setAccountSwitching(true)
  try {
    void useProgress.persist.rehydrate()
    if (file.version === 2) { void useDrafts.persist.rehydrate(); void useMock.persist.rehydrate() }
    useEvents.getState().clear()
  } finally {
    setAccountSwitching(false)
  }
}
