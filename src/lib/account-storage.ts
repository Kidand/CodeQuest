/** Account selection is tab-local; learning data and accounts persist on this device. */
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
  return id ? `${base}::${id}` : base
}
