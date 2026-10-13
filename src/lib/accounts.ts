import { create } from 'zustand'
import { createJSONStorage } from 'zustand/middleware'
import { accountKey, currentAccountId, setAccountSwitching } from './account-storage'
import { initialData, useDrafts, useEvents, useProgress } from './store'
import { DEFAULT_MOCK_CONFIG, useMock } from './mock'
import { pyRunner } from '@/runner/client'

interface LocalAccount { id: string; username: string; salt: string; hash: string; iterations: number; createdAt: string }
const ITERATIONS = 600_000
const stores = ['cq-progress', 'cq-drafts', 'cq-mock'] as const

function readAccounts(): LocalAccount[] {
  const value = JSON.parse(localStorage.getItem('cq-accounts') || '[]')
  if (!Array.isArray(value) || value.some((a) => !a || typeof a.id !== 'string' || typeof a.username !== 'string' || typeof a.salt !== 'string' || typeof a.hash !== 'string' || a.iterations !== ITERATIONS)) {
    throw new Error('本机账号数据无法读取，请先备份浏览器数据。游客学习仍可使用。')
  }
  return value
}

const hex = (bytes: Uint8Array) => Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')
async function passwordHash(password: string, salt: string, iterations = ITERATIONS) {
  if (!crypto.subtle) throw new Error('本地账号需要 HTTPS 安全连接，请使用正式站点地址。')
  const key = await crypto.subtle.importKey('raw', new TextEncoder().encode(password), 'PBKDF2', false, ['deriveBits'])
  const bits = await crypto.subtle.deriveBits({ name: 'PBKDF2', salt: new TextEncoder().encode(salt), iterations, hash: 'SHA-256' }, key, 256)
  return hex(new Uint8Array(bits))
}

function activeAccount() {
  try { return readAccounts().find((a) => a.id === currentAccountId()) ?? null } catch { return null }
}

export const useAccount = create<{ account: LocalAccount | null; revision: number; switching: boolean }>(() => ({ account: activeAccount(), revision: 0, switching: false }))

/** Reset memory with persistence paused, then hydrate the target. Never write the previous user into a new key. */
export async function switchAccount(account: LocalAccount | null) {
  setAccountSwitching(true)
  useAccount.setState({ switching: true })
  pyRunner.cancel()
  useEvents.getState().clear()
  try {
    if (account) sessionStorage.setItem('cq-session', account.id)
    else sessionStorage.removeItem('cq-session')
    const paused = { getItem: () => null, setItem: () => {}, removeItem: () => {} }
    useProgress.persist.setOptions({ name: accountKey('cq-progress', account?.id ?? null), storage: paused })
    useDrafts.persist.setOptions({ name: accountKey('cq-drafts', account?.id ?? null), storage: paused })
    useMock.persist.setOptions({ name: accountKey('cq-mock', account?.id ?? null), storage: paused })
    useProgress.setState(initialData())
    useDrafts.setState({ drafts: {} })
    useMock.setState({ active: null, history: [], lastConfig: DEFAULT_MOCK_CONFIG })
    useProgress.persist.setOptions({ storage: createJSONStorage(() => localStorage) })
    useDrafts.persist.setOptions({ storage: createJSONStorage(() => localStorage) })
    useMock.persist.setOptions({ storage: createJSONStorage(() => localStorage) })
    await Promise.all([useProgress.persist.rehydrate(), useDrafts.persist.rehydrate(), useMock.persist.rehydrate()])
    useAccount.setState((s) => ({ account, revision: s.revision + 1 }))
  } finally { setAccountSwitching(false); useAccount.setState({ switching: false }) }
}

export function canClaimGuest() { return readAccounts().length === 0 && stores.some((key) => localStorage.getItem(key) !== null) }

export async function registerAccount(username: string, password: string, claimGuest: boolean) {
  username = username.trim().normalize('NFKC')
  if (!/^[\p{L}\p{N}_-]{2,24}$/u.test(username)) throw new Error('用户名需为 2–24 个汉字、字母、数字、下划线或短横线。')
  if (password.length < 8 || password.length > 128) throw new Error('密码需为 8–128 个字符。')
  const salt = hex(crypto.getRandomValues(new Uint8Array(16)))
  const hash = await passwordHash(password, salt)
  // Re-read after async hashing to avoid stale duplicate checks across tabs.
  const accounts = readAccounts()
  if (accounts.some((a) => a.username.toLocaleLowerCase() === username.toLocaleLowerCase())) throw new Error('本机已有这个用户名，请直接登录。')
  const account: LocalAccount = { id: crypto.randomUUID(), username, salt, hash, iterations: ITERATIONS, createdAt: new Date().toISOString() }
  const claim = claimGuest && accounts.length === 0
  try {
    if (claim) for (const base of stores) {
      const data = localStorage.getItem(base)
      if (data) localStorage.setItem(accountKey(base, account.id), data)
    }
    localStorage.setItem('cq-accounts', JSON.stringify([...accounts, account]))
  } catch {
    for (const base of stores) localStorage.removeItem(accountKey(base, account.id))
    throw new Error('无法保存账号：浏览器存储可能已满或被禁用，请先导出进度。')
  }
  if (claim) for (const base of stores) localStorage.removeItem(base)
  await switchAccount(account)
}

export async function loginAccount(username: string, password: string) {
  const account = readAccounts().find((a) => a.username.toLocaleLowerCase() === username.trim().normalize('NFKC').toLocaleLowerCase())
  if (!account || await passwordHash(password, account.salt, account.iterations) !== account.hash) throw new Error('用户名或密码不正确。账号仅在创建它的浏览器中有效。')
  await switchAccount(account)
}
