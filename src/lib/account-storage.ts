/** Account selection is tab-local; learning data and accounts persist on this device. */
let storageAccountId: string | null | undefined
export function isCloudStorageActive() { return storageAccountId?.startsWith('cloud:') ?? false }
export let accountStorageGeneration = 0
export function setStorageAccountId(id: string | null) { storageAccountId = id; accountStorageGeneration++ }
export let accountSwitchInProgress = false
export function setAccountSwitching(value: boolean) { accountSwitchInProgress = value }

export function currentAccountId(): string | null {
  try {
    const id = sessionStorage.getItem('cq-session')
    const accounts = JSON.parse(localStorage.getItem('cq-accounts') || '[]')
    return id && Array.isArray(accounts) && accounts.some((a) => a.id === id) ? id : null
  } catch { return null }
}

export function accountKey(base: string, id = currentAccountId()) {
  const selected = arguments.length > 1 ? id : storageAccountId ?? id
  return selected ? `${base}::${selected}` : base
}
