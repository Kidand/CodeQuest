/** Opt-in deployed-site smoke test. Creates one synthetic QA account, clears its sample archive,
 * and revokes its sessions. Never reads or changes existing users' records; secrets stay in memory.
 * CQ_PUBLIC_BASE=https://your-site npm run test:auth:live
 */
import assert from 'node:assert/strict'
import { randomBytes, randomUUID } from 'node:crypto'
import { normalizeCloudSnapshot } from '../shared/cloud'

const base = process.env.CQ_PUBLIC_BASE?.replace(/\/$/, '')
if (!base || new URL(base).protocol !== 'https:') throw new Error('Set CQ_PUBLIC_BASE to the deployed HTTPS site to opt in.')
const username = `qa_${randomBytes(8).toString('hex')}`
const password = randomBytes(32).toString('base64url')
const cookies: string[] = []
async function call(path: string, method = 'GET', body?: unknown, cookie?: string, user?: string) {
  const response = await fetch(base + path, {
    method, redirect: 'manual', signal: AbortSignal.timeout(30_000),
    headers: { ...(body !== undefined ? { 'Content-Type': 'application/json', Origin: base! } : {}), ...(cookie ? { Cookie: cookie } : {}), ...(user ? { 'X-CodeQuest-User': user } : {}) },
    ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
  })
  const data = await response.json() as Record<string, any>
  return { response, data }
}
function sessionCookie(response: Response) {
  const raw = response.headers.getSetCookie().find(value => value.startsWith('__Host-codequest_session='))
  assert.ok(raw && /;\s*HttpOnly/i.test(raw) && /;\s*Secure/i.test(raw) && /;\s*SameSite=Lax/i.test(raw) && /;\s*Path=\//i.test(raw) && !/;\s*Domain=/i.test(raw), 'Secure host-only HttpOnly session cookie is required')
  const cookie = raw.split(';')[0]
  cookies.push(cookie)
  return cookie
}
const clean = normalizeCloudSnapshot({
  app: 'codequest', version: 2,
  state: { xp: 0, levels: {}, streak: { current: 0, best: 0 }, daily: {}, achievements: {}, review: {}, settings: { theme: 'system', freeMode: false, editorFontSize: 14, reduceMotion: false, dailyGoal: 100, sound: false }, createdAt: new Date().toISOString() },
  drafts: {}, mock: { active: null, history: [], lastConfig: { durationMin: 45, algoChallenges: 1, llmChallenges: 1, concepts: 6, scope: 'unlocked', tiers: [] } },
})
let userId: string | undefined
let archiveWritten = false
try {
  const anonymous = await call('/api/cloud/session')
  assert.equal(anonymous.response.status, 200)
  assert.equal(anonymous.data.auth, 'password')
  assert.equal(anonymous.data.user, null)
  assert.equal((await call('/api/cloud/state')).response.status, 401)
  const registered = await call('/api/auth/register', 'POST', { username, password })
  assert.equal(registered.response.status, 201, `Registration status ${registered.response.status}; ${registered.data.error ?? 'unexpected response'}`)
  userId = registered.data.user.id
  const first = sessionCookie(registered.response)
  assert.equal((await call('/api/auth/login', 'POST', { username, password: 'not-the-correct-password' })).response.status, 401)
  const login = await call('/api/auth/login', 'POST', { username: username.toUpperCase(), password })
  assert.equal(login.response.status, 200)
  assert.equal(login.data.user.id, userId)
  const second = sessionCookie(login.response)
  assert.ok(first !== second, 'Each device must receive a different session')
  const initial = await call('/api/cloud/state', 'GET', undefined, first, userId)
  assert.equal(initial.data.revision, 0)
  const sample = structuredClone(clean)
  sample.state.xp = 17
  sample.drafts['qa-cloud-check'] = '# Synthetic cross-device connectivity check'
  const saved = await call('/api/cloud/state', 'PUT', { baseRevision: 0, mutationId: randomUUID(), snapshot: sample }, first, userId)
  assert.equal(saved.response.status, 200)
  archiveWritten = true
  const restored = await call('/api/cloud/state', 'GET', undefined, second, userId)
  assert.equal(restored.response.status, 200)
  assert.equal(restored.data.snapshot.state.xp, 17)
  assert.equal(restored.data.snapshot.drafts['qa-cloud-check'], sample.drafts['qa-cloud-check'])
  const reset = await call('/api/cloud/state', 'PUT', { baseRevision: restored.data.revision, mutationId: randomUUID(), snapshot: clean }, second, userId)
  assert.equal(reset.response.status, 200)
  archiveWritten = false
  assert.equal((await call('/api/auth/logout', 'POST', {}, first, userId)).response.status, 200)
  assert.equal((await call('/api/cloud/state', 'GET', undefined, first, userId)).response.status, 401)
  assert.equal((await call('/api/cloud/state', 'GET', undefined, second, userId)).response.status, 200)
  console.log('Live registration, native password verification, cookie flags, independent-device restore and logout revocation: PASS')
} finally {
  if (archiveWritten && userId && cookies.length) {
    const current = await call('/api/cloud/state', 'GET', undefined, cookies.at(-1), userId).catch(() => null)
    if (current?.response.ok) await call('/api/cloud/state', 'PUT', { baseRevision: current.data.revision, mutationId: randomUUID(), snapshot: clean }, cookies.at(-1), userId).catch(() => {})
  }
  for (const cookie of cookies) await call('/api/auth/logout', 'POST', {}, cookie, userId).catch(() => {})
}
