/** Cloud account coordinator. Journals are scoped by verified user and tab; CAS never silently merges progress. */
import { create } from 'zustand'
import { createJSONStorage } from 'zustand/middleware'
import { normalizeCloudSnapshot, MAX_CLOUD_BYTES, type CloudSnapshot, type CloudUser, type CloudStateResponse, type CloudSessionResponse } from '../../shared/cloud'
import { useProgress, useDrafts, useEvents, exportProgress, initialData } from './store'
import { useMock, DEFAULT_MOCK_CONFIG } from './mock'
import { useAccount, readAccounts, switchStorageAccount } from './accounts'
import { accountSwitchInProgress, accountKey, currentAccountId, setAccountSwitching } from './account-storage'
import { parseProgress } from './backup'
import { pyRunner } from '@/runner/client'

export type CloudStatus = 'checking' | 'local' | 'saved' | 'syncing' | 'offline' | 'conflict' | 'expired' | 'error'
type Pending = { baseRevision: number; mutationId: string; snapshot: CloudSnapshot }
type Journal = { userId: string; revision: number; updatedAt: string | null; dirty: boolean; snapshot: CloudSnapshot; editedAt: number; pending?: Pending; conflict?: CloudStateResponse }
interface CloudView {
  ready: boolean; user: CloudUser | null; status: CloudStatus; dirty: boolean; revision: number; updatedAt: string | null
  error: string; conflict: CloudStateResponse | null; signInUrl: string; signOutUrl: string
}
export const useCloud = create<CloudView>(() => ({ ready: false, user: null, status: 'checking', dirty: false, revision: 0, updatedAt: null, error: '', conflict: null, signInUrl: '/signin-with-chatgpt?return_to=%2F%23%2Faccount', signOutUrl: '/signout-with-chatgpt?return_to=%2F%23%2Faccount' }))
export const cloudStorageId = (id: string) => `cloud:${encodeURIComponent(id)}`
const cacheKey = (id: string) => `cq-cloud-cache::${encodeURIComponent(id)}`
const journalPrefix = (id: string) => `cq-cloud-journal::${encodeURIComponent(id)}::`
// sessionStorage is cloned by Duplicate Tab/window.open. Every document gets a new writer ID;
// the previous ID is only a recovery pointer, never reused for writes.
const tabId = crypto.randomUUID()
let resumeTabId: string | null = null
try { resumeTabId = sessionStorage.getItem('cq-cloud-tab'); sessionStorage.setItem('cq-cloud-tab', tabId) } catch { /* An in-memory identifier keeps downloads available when storage is blocked. */ }
const journalKey = (id: string) => journalPrefix(id) + tabId
let journal: Journal | null = null
let applying = false
let timer: ReturnType<typeof setTimeout> | undefined
let flight: Promise<boolean> | null = null
let refreshing: Promise<void> | null = null
let initialization: Promise<void> | null = null
let epoch = 0
let inheritedJournal: { key: string; value: string } | null = null
let navigatingAuth = false
function clearInheritedJournal() {
  if (!inheritedJournal) return
  try {
    const original = JSON.parse(inheritedJournal.value) as Journal
    // A cloned tab may diverge from its source. Keep the source's unsynced branch
    // until its exact snapshot is acknowledged; it remains downloadable in recovery.
    if (original.dirty && !same(original.snapshot, journal?.snapshot)) return
    if (localStorage.getItem(inheritedJournal.key) === inheritedJournal.value) localStorage.removeItem(inheritedJournal.key)
    inheritedJournal = null
  } catch { /* Keep an archive that cannot be safely compared or removed. */ }
}

const json = (v: unknown) => JSON.stringify(v)
const stable = (v: unknown) => JSON.stringify(v, (_key, value) => value && typeof value === 'object' && !Array.isArray(value) ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)) : value)
const same = (a: unknown, b: unknown) => stable(a) === stable(b)
export function captureCloudSnapshot(): CloudSnapshot {
  const m = useMock.getState()
  return JSON.parse(json({ app: 'codequest', version: 2, state: JSON.parse(exportProgress()).state, drafts: useDrafts.getState().drafts, mock: { active: m.active, lastConfig: m.lastConfig, history: m.history } }))
}
export function summarizeCloudSnapshot(snapshot: CloudSnapshot | null) {
  return snapshot ? `${snapshot.state.xp} XP · ${Object.keys(snapshot.state.levels).length} 关记录 · ${Object.keys(snapshot.drafts).length} 份代码 · ${snapshot.mock.history.length} 场面试${snapshot.mock.active ? '（另有进行中的面试）' : ''}` : '尚无云端存档'
}
export function downloadCloudBackup(snapshot: CloudSnapshot, label = 'progress') {
  const url = URL.createObjectURL(new Blob([JSON.stringify({ ...snapshot, exportedAt: new Date().toISOString() }, null, 2)], { type: 'application/json' }))
  const a = document.createElement('a'); a.href = url; a.download = `codequest-${label}-${new Date().toISOString().slice(0, 10)}.json`; document.body.appendChild(a); a.click(); a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
function publish(status?: CloudStatus, error = '') {
  if (!journal) return
  useCloud.setState({ dirty: journal.dirty, revision: journal.revision, updatedAt: journal.updatedAt, conflict: journal.conflict ?? null, error, ...(status ? { status } : {}) })
}
function persistJournal() {
  if (!journal) return
  try { localStorage.setItem(journalKey(journal.userId), json(journal)); return true }
  catch { publish('error', '本机存储空间不足，尚未安全保存。请立即下载备份，再清理浏览器空间。'); return false }
}
function saveCache(state: CloudStateResponse) {
  try {
    const previous = JSON.parse(localStorage.getItem(cacheKey(state.user.id)) || 'null') as CloudStateResponse | null
    if (!previous || previous.revision <= state.revision) localStorage.setItem(cacheKey(state.user.id), json(state))
  } catch { /* The active journal still contains the confirmed archive. */ }
}
function isCurrent(id: string, generation: number) { return epoch === generation && useCloud.getState().user?.id === id && journal?.userId === id }
function applySnapshot(snapshot: CloudSnapshot) {
  const state = parseProgress(snapshot.state)
  const writes = [
    { key: useProgress.persist.getOptions().name!, value: json({ state, version: 1 }) },
    { key: useDrafts.persist.getOptions().name!, value: json({ state: { drafts: snapshot.drafts }, version: 0 }) },
    { key: useMock.persist.getOptions().name!, value: json({ state: snapshot.mock, version: 1 }) },
  ]
  const previous = writes.map(({ key }) => ({ key, value: localStorage.getItem(key) }))
  try { for (const { key, value } of writes) localStorage.setItem(key, value) }
  catch {
    try {
      for (const { key } of writes) localStorage.removeItem(key)
      for (const { key, value } of previous) if (value !== null) localStorage.setItem(key, value)
    } catch { /* Raw prior archive also remains in the journal/recovery. */ }
    throw new Error('本机存储空间不足，存档载入已取消。请先下载备份并清理浏览器空间。')
  }
  applying = true
  setAccountSwitching(true)
  pyRunner.cancel()
  const paused = { getItem: () => null, setItem: () => {}, removeItem: () => {} }
  useProgress.persist.setOptions({ storage: paused }); useDrafts.persist.setOptions({ storage: paused }); useMock.persist.setOptions({ storage: paused })
  try {
    useProgress.setState(state)
    useDrafts.setState({ drafts: snapshot.drafts })
    useMock.setState(snapshot.mock)
    useEvents.getState().clear()
    useAccount.setState((s) => ({ revision: s.revision + 1 }))
  } finally {
    useProgress.persist.setOptions({ storage: createJSONStorage(() => localStorage) }); useDrafts.persist.setOptions({ storage: createJSONStorage(() => localStorage) }); useMock.persist.setOptions({ storage: createJSONStorage(() => localStorage) })
    setAccountSwitching(false); applying = false
  }
}

function onEdit() {
  if (applying || accountSwitchInProgress || !journal || !useCloud.getState().user) return
  const snapshot = captureCloudSnapshot()
  if (same(snapshot, journal.snapshot)) return
  journal.snapshot = snapshot; journal.dirty = true; journal.editedAt = Date.now()
  if (!persistJournal()) return
  publish(journal.conflict ? 'conflict' : useCloud.getState().status === 'expired' ? 'expired' : navigator.onLine ? 'syncing' : 'offline')
  clearTimeout(timer)
  timer = setTimeout(() => { void flushCloud() }, 650)
}
export function markCloudDirty() { onEdit() }
function expired(message = '登录已失效。未同步的进度已保留在此设备，请重新登录同一账号。') {
  publish('expired', message)
}
async function request(path: string, init: RequestInit = {}, expectedId?: string): Promise<{ response: Response; body: any }> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 12000)
  try {
    const response = await fetch(path, { ...init, credentials: 'same-origin', cache: 'no-store', signal: controller.signal, headers: { ...(init.body ? { 'Content-Type': 'application/json' } : {}), ...(expectedId ? { 'X-CodeQuest-User': expectedId } : {}), ...init.headers } })
    const body = await response.json()
    return { response, body }
  } finally { clearTimeout(timeout) }
}
function acceptRemote(remote: CloudStateResponse) {
  if (!journal || remote.user.id !== journal.userId || remote.revision < Math.max(journal.revision, journal.conflict?.revision ?? 0)) return
  saveCache(remote)
  if (journal.dirty) {
    // A lost PUT response is retried with its original idempotency key, not replaced by GET.
    if (journal.pending) return
    if (remote.revision !== journal.revision) { journal.conflict = remote; if (persistJournal()) publish('conflict'); return }
    publish(navigator.onLine ? 'syncing' : 'offline'); return
  }
  if (remote.snapshot && (remote.revision !== journal.revision || !same(remote.snapshot, journal.snapshot))) {
    applySnapshot(remote.snapshot)
    journal.snapshot = captureCloudSnapshot()
  }
  journal.revision = remote.revision; journal.updatedAt = remote.updatedAt; journal.conflict = undefined
  if (persistJournal()) { clearInheritedJournal(); publish('saved') }
}
function restoreJournal(user: CloudUser): Journal | null {
  const own = localStorage.getItem(journalKey(user.id))
  const resumeKey = resumeTabId ? journalPrefix(user.id) + resumeTabId : null
  const candidates: { key: string; value: string; data: Journal }[] = []
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i)!
    if (!key.startsWith(journalPrefix(user.id))) continue
    try { const value = localStorage.getItem(key)!; const data = JSON.parse(value) as Journal; if (data.userId === user.id && data.snapshot && (data.dirty || key === journalKey(user.id) || key === resumeKey)) candidates.push({ key, value, data }) } catch { /* Keep unreadable raw archives for recovery. */ }
  }
  const chosen = candidates.find((c) => own && c.key === journalKey(user.id)) ?? candidates.find((c) => c.key === resumeKey) ?? candidates.sort((a, b) => b.data.editedAt - a.data.editedAt)[0]
  if (chosen) { if (chosen.key !== journalKey(user.id)) inheritedJournal = { key: chosen.key, value: chosen.value }; return chosen.data }
  try {
    const cache = JSON.parse(localStorage.getItem(cacheKey(user.id)) || 'null') as CloudStateResponse | null
    if (cache?.user.id === user.id && cache.snapshot) return { userId: user.id, revision: cache.revision, updatedAt: cache.updatedAt, dirty: false, snapshot: cache.snapshot, editedAt: Date.now() }
  } catch { /* Fresh cloud namespace below. */ }
  return null
}
async function activateUser(user: CloudUser) {
  epoch++; clearTimeout(timer); inheritedJournal = null
  let restored: Journal | null = null
  try { restored = restoreJournal(user) } catch { useCloud.setState({ error: '无法读取本机缓存，请检查浏览器存储权限。' }) }
  journal = null
  useCloud.setState({ user, status: 'checking', conflict: null, dirty: false, error: '' })
  await switchStorageAccount(cloudStorageId(user.id))
  journal = restored ?? { userId: user.id, revision: 0, updatedAt: null, dirty: false, snapshot: captureCloudSnapshot(), editedAt: Date.now() }
  if (restored) applySnapshot(restored.snapshot)
  try { localStorage.setItem('cq-cloud-last-user', json(user)); sessionStorage.setItem('cq-cloud-active', user.id) } catch { /* Reported by the journal write below. */ }
  if (persistJournal()) publish(restored?.conflict ? 'conflict' : 'checking')
}
async function activateLocal() {
  epoch++; clearTimeout(timer); journal = null; inheritedJournal = null
  useCloud.setState({ user: null, status: 'local', dirty: false, conflict: null, error: '', revision: 0, updatedAt: null })
  const id = currentAccountId()
  const account = (() => { try { return readAccounts().find((a) => a.id === id) ?? null } catch { return null } })()
  await switchStorageAccount(id, account)
  sessionStorage.removeItem('cq-cloud-active')
}
async function refreshInternal() {
  try {
    const { response, body } = await request('/api/cloud/session?returnTo=%2F%23%2Faccount')
    if (!response.ok) throw new Error(body.message || '暂时无法连接云端')
    const session = body as CloudSessionResponse
    useCloud.setState({ signInUrl: session.signInUrl, signOutUrl: session.signOutUrl })
    if (!session.user) {
      if (useCloud.getState().user) { expired(); return }
      if (!useCloud.getState().ready) await activateLocal()
      else useCloud.setState({ status: 'local', error: '' })
      return
    }
    if (session.user.id !== useCloud.getState().user?.id) await activateUser(session.user)
    else useCloud.setState({ user: session.user })
    const id = session.user.id, generation = epoch
    const remote = await request('/api/cloud/state', {}, id)
    if (!isCurrent(id, generation)) return
    if (remote.response.status === 401 || remote.body.error === 'identity_changed') { expired(); return }
    if (!remote.response.ok) throw new Error(remote.body.message || '读取云端存档失败')
    if (useCloud.getState().status === 'expired') publish('syncing')
    acceptRemote(remote.body)
    if (journal?.dirty && !journal.conflict) await flushCloud()
  } catch (e) {
    if (journal) {
      const message = (e as Error).message || ''
      if (/存储|格式|载入/.test(message)) publish('error', message)
      else publish('offline', '暂时无法连接云端，进度保留在此设备，联网后会重试。')
    }
    else {
      let previous: CloudUser | null = null
      try { previous = JSON.parse(localStorage.getItem('cq-cloud-last-user') || 'null') } catch { /* No cached account. */ }
      if (previous?.id) { await activateUser(previous); publish('offline', '暂时离线，正在使用该账号的设备存档。重新连接后会核对身份并同步。') }
      else useCloud.setState({ status: 'local', error: '暂时无法连接云端，可以先在本机学习，稍后登录迁移。' })
    }
  } finally { useCloud.setState({ ready: true }) }
}
export function refreshCloud() {
  if (!refreshing) refreshing = refreshInternal().finally(() => { refreshing = null })
  return refreshing
}
export function initializeCloud() {
  if (!initialization) {
    useProgress.subscribe(onEdit); useDrafts.subscribe(onEdit); useMock.subscribe(onEdit)
    window.addEventListener('online', () => { void refreshCloud() })
    window.addEventListener('offline', () => { if (journal) publish('offline') })
    window.addEventListener('focus', () => { void refreshCloud() })
    document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') void refreshCloud() })
    window.addEventListener('storage', (e) => { if (e.key === 'cq-cloud-auth-change' || e.key?.startsWith('cq-cloud-cache::')) void refreshCloud() })
    window.addEventListener('beforeunload', (e) => { if (journal?.dirty) { persistJournal(); if (!navigatingAuth) { e.preventDefault(); e.returnValue = '' } } })
    initialization = refreshCloud()
  }
  return initialization
}
async function flushInternal(): Promise<boolean> {
  if (!journal || !useCloud.getState().user) return true
  if (!navigator.onLine) { publish('offline'); return false }
  if (journal.conflict) { publish('conflict'); return false }
  if (useCloud.getState().status === 'expired') return false
  while (journal?.dirty) {
    const id = journal.userId, generation = epoch
    try {
      if (!journal.pending) {
        if (new TextEncoder().encode(json(journal.snapshot)).byteLength > MAX_CLOUD_BYTES) throw new Error(`存档超过云端 ${Math.floor(MAX_CLOUD_BYTES / 1000)} KB 上限。进度已保留本机，请导出备份并精简代码草稿后重试。`)
        journal.pending = { baseRevision: journal.revision, mutationId: crypto.randomUUID(), snapshot: normalizeCloudSnapshot(journal.snapshot) }
        if (!persistJournal()) return false
      }
      const pending = journal.pending
      publish('syncing')
      const { response, body } = await request('/api/cloud/state', { method: 'PUT', body: json(pending) }, id)
      if (!isCurrent(id, generation)) return false
      if (response.status === 401 || body.error === 'identity_changed') { expired(); return false }
      if (response.status === 409 && body.error === 'conflict') {
        journal!.conflict = body; journal!.pending = undefined; if (persistJournal()) publish('conflict'); return false
      }
      if (!response.ok) throw new Error(body.message || (response.status === 413 ? '云端存档过大，请先导出备份并精简代码草稿。' : '云端保存失败，请稍后重试。'))
      const remote = body as CloudStateResponse
      if (remote.user.id !== id) { expired(); return false }
      if (remote.revision !== pending.baseRevision + 1 && !same(remote.snapshot, pending.snapshot)) {
        journal!.conflict = remote; journal!.pending = undefined; if (persistJournal()) publish('conflict'); return false
      }
      const hasNewEdits = !same(normalizeCloudSnapshot(journal!.snapshot), pending.snapshot)
      journal!.revision = remote.revision; journal!.updatedAt = remote.updatedAt; journal!.pending = undefined; journal!.dirty = hasNewEdits
      saveCache(remote); if (!persistJournal()) return false
      if (!hasNewEdits) clearInheritedJournal()
      publish(hasNewEdits ? 'syncing' : 'saved')
    } catch (e) {
      if (!isCurrent(id, generation)) return false
      const offline = e instanceof TypeError || (e instanceof DOMException && e.name === 'AbortError') || !navigator.onLine
      publish(offline ? 'offline' : 'error', offline ? '网络连接中断，进度已保留在此设备，联网后继续同步。' : (e as Error).message)
      return false
    }
  }
  return true
}
export function flushCloud(): Promise<boolean> {
  clearTimeout(timer)
  if (!flight) {
    const startedId = journal?.userId
    flight = flushInternal().finally(() => {
      flight = null
      if (journal?.dirty && journal.userId !== startedId && !journal.conflict && useCloud.getState().status !== 'expired') timer = setTimeout(() => { void flushCloud() }, 0)
    })
  }
  return flight
}
function keepRecovery(value: unknown) {
  if (!journal) return
  try { localStorage.setItem(`cq-cloud-recovery::${encodeURIComponent(journal.userId)}::${Date.now()}`, json(value)) }
  catch { throw new Error('无法保留替换前的恢复备份，操作已取消。请先下载备份并清理本机存储空间，当前存档未被修改。') }
}
export async function resolveCloudConflict(choice: 'local' | 'remote') {
  if (!journal?.conflict) return
  const remote = journal.conflict
  // Persist both complete archives before resolving, even if downloads were skipped.
  keepRecovery({ local: journal.snapshot, remote })
  if (choice === 'remote') {
    const snapshot = remote.snapshot ?? emptyCloudSnapshot()
    applySnapshot(snapshot)
    journal.snapshot = snapshot; journal.dirty = false
    journal.revision = remote.revision; journal.updatedAt = remote.updatedAt; journal.pending = undefined; journal.conflict = undefined
    saveCache(remote); if (persistJournal()) publish('saved')
  } else {
    journal.revision = remote.revision; journal.updatedAt = remote.updatedAt; journal.pending = undefined; journal.conflict = undefined
    journal.dirty = true; if (persistJournal()) await flushCloud()
  }
}
export function emptyCloudSnapshot(): CloudSnapshot { return { app: 'codequest', version: 2, state: initialData(), drafts: {}, mock: { active: null, history: [], lastConfig: DEFAULT_MOCK_CONFIG } } }
export async function migrateCloudSnapshot(input: CloudSnapshot) {
  if (!journal || !useCloud.getState().user) throw new Error('请先登录云端账号。')
  if (journal.conflict) throw new Error('请先处理同步冲突，再迁移存档。')
  const id = journal.userId, generation = epoch
  const snapshot = normalizeCloudSnapshot(input)
  if (new TextEncoder().encode(json(snapshot)).byteLength > MAX_CLOUD_BYTES) throw new Error(`此备份超过云端 ${Math.floor(MAX_CLOUD_BYTES / 1000)} KB 上限，请先精简代码草稿。原存档未被修改。`)
  // Do not abandon an in-flight/unacknowledged request; retry it before replacement.
  if (journal.pending && !await flushCloud()) throw new Error('请先恢复同步并处理待保存的进度，再迁移存档。')
  if (!isCurrent(id, generation)) throw new Error('账号已切换，迁移已取消。请在当前账号重新选择存档。')
  keepRecovery({ local: journal.snapshot, imported: snapshot })
  applySnapshot(snapshot)
  journal.snapshot = captureCloudSnapshot(); journal.dirty = true; journal.editedAt = Date.now()
  if (persistJournal()) { publish('syncing'); await flushCloud() }
}
export function parseCloudBackup(text: string): CloudSnapshot {
  const file = JSON.parse(text)
  if (!file || typeof file !== 'object' || Array.isArray(file) || (file.app !== undefined && file.app !== 'codequest') || (file.version !== undefined && ![1, 2].includes(file.version))) throw new Error('这不是支持的 CodeQuest 备份，请选择版本 1 或版本 2 的 JSON。')
  if (file.version === 2) return normalizeCloudSnapshot({ app: file.app, version: file.version, state: file.state, drafts: file.drafts, mock: file.mock })
  return normalizeCloudSnapshot({ ...emptyCloudSnapshot(), state: parseProgress(file.state ?? file) })
}
export interface LocalMigrationSource { id: string; label: string; snapshot: CloudSnapshot | null; error?: string }
export function localMigrationSources(): LocalMigrationSource[] {
  let accounts: ReturnType<typeof readAccounts> = []
  try { accounts = readAccounts() } catch { /* The guest and exported JSON can still be recovered. */ }
  return [{ id: null, username: '游客存档' }, ...accounts].flatMap<LocalMigrationSource>((account) => {
    const keys = ['cq-progress', 'cq-drafts', 'cq-mock'].map((base) => accountKey(base, account.id))
    try {
      if (!keys.some((key) => localStorage.getItem(key))) return []
      const read = (key: string) => JSON.parse(localStorage.getItem(key) || 'null')?.state
      const snapshot = normalizeCloudSnapshot({ app: 'codequest', version: 2, state: parseProgress(read(keys[0]) ?? initialData()), drafts: read(keys[1])?.drafts ?? {}, mock: read(keys[2]) ?? emptyCloudSnapshot().mock })
      return [{ id: account.id ?? 'guest', label: account.username, snapshot }]
    } catch (e) { return [{ id: account.id ?? 'guest', label: account.username, snapshot: null, error: (e as Error).message }] }
  })
}
export const cloudStatusText: Record<CloudStatus, string> = { checking: '正在连接云端', local: '仅本机保存', saved: '已保存到云端', syncing: '正在同步', offline: '离线 · 等待同步', conflict: '存档冲突 · 需要选择', expired: '登录已失效', error: '同步需要处理' }
export function announceCloudAuthChange() { navigatingAuth = true; try { localStorage.setItem('cq-cloud-auth-change', String(Date.now())) } catch { /* Top-level authentication can proceed without browser storage. */ } }

export function cloudRecoveryBackups(): { id: string; label: string; snapshot: CloudSnapshot }[] {
  const id = useCloud.getState().user?.id
  if (!id) return []
  try {
    const prefix = `cq-cloud-recovery::${encodeURIComponent(id)}::`
    const keys = Object.keys(localStorage).filter((key) => key.startsWith(prefix)).sort().reverse().slice(0, 8)
    const recoveries = keys.flatMap((key) => {
      try {
        const item = JSON.parse(localStorage.getItem(key)!)
        const time = new Date(Number(key.slice(prefix.length))).toLocaleString('zh-CN')
        return [
          ...(item.local ? [{ id: `${key}:local`, label: `${time} · 替换前的设备版本`, snapshot: item.local as CloudSnapshot }] : []),
          ...(item.remote?.snapshot ? [{ id: `${key}:remote`, label: `${time} · 冲突时的云端版本`, snapshot: item.remote.snapshot as CloudSnapshot }] : []),
        ]
      } catch { return [] }
    })
    const branches = Object.keys(localStorage).filter((key) => key.startsWith(journalPrefix(id)) && key !== journalKey(id)).flatMap((key) => {
      try {
        const branch = JSON.parse(localStorage.getItem(key)!) as Journal
        if (!branch.dirty || same(branch.snapshot, journal?.snapshot)) return []
        return [{ id: key, label: `${new Date(branch.editedAt).toLocaleString('zh-CN')} · 其他标签未同步的版本`, snapshot: branch.snapshot }]
      } catch { return [] }
    })
    return [...branches, ...recoveries]
  } catch { return [] }
}
