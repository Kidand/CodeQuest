/** Security and concurrency checks against the real Worker and a disposable D1 database. */
import assert from 'node:assert/strict'
import { randomUUID, randomBytes, createHash, scryptSync } from 'node:crypto'
import { mkdtemp, readFile, readdir, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { build } from 'esbuild'
import { Miniflare, convertV4MiniflareOptions } from 'miniflare'
import { handleRequest, type WorkerEnv } from '../worker/index'

const origin = 'https://codequest.test'
const temp = await mkdtemp(join(tmpdir(), 'codequest-cloud-test-'))
const workerFile = join(temp, 'worker.mjs')
await build({ entryPoints: ['worker/index.ts'], bundle: true, format: 'esm', platform: 'browser', target: 'es2022', external: ['node:crypto'], outfile: workerFile })
const mf = new Miniflare(convertV4MiniflareOptions({ name: 'cloud-test', modules: true, modulesRoot: temp, scriptPath: workerFile, compatibilityDate: '2026-05-15', compatibilityFlags: ['nodejs_compat'], d1Databases: { DB: 'cq-test' } }))
let checks = 0
let workerEnv: WorkerEnv
function checked(label: string) { checks++; console.log(`✓ ${label}`) }
const identities = new Map<string, { cookie: string; id: string; username: string; password: string }>()
let ipCounter = 0
async function request(path: string, options: { user?: string; method?: string; body?: unknown; origin?: string | null; headers?: Record<string, string>; direct?: boolean; base?: string } = {}) {
  const base = options.base ?? origin
  const headers: Record<string, string> = { 'cf-connecting-ip': `192.0.${Math.floor(++ipCounter / 250)}.${ipCounter % 250 + 1}`, ...options.headers }
  if (options.user) headers.cookie = identities.get(options.user)!.cookie
  if (['PUT', 'POST'].includes(options.method ?? 'GET')) {
    headers['content-type'] ??= 'application/json'
    if (options.origin !== null) headers.origin = options.origin ?? base
  }
  const body = options.body === undefined ? undefined : JSON.stringify(options.body)
  if (options.direct && body) headers['content-length'] = String(Buffer.byteLength(body))
  const init = { method: options.method ?? 'GET', headers, body }
  const response = options.direct
    ? await handleRequest(new Request(`${base}${path}`, init), workerEnv)
    : await mf.dispatchFetch(`${base}${path}`, init)
  await response.clone().arrayBuffer()
  return response
}
async function register(alias: string) {
  const username = `Test_${alias.replaceAll('-', '_')}_${randomBytes(3).toString('hex')}`
  const password = randomBytes(24).toString('base64url')
  const response = await request('/api/auth/register', { method: 'POST', body: { username, password } })
  assert.equal(response.status, 201, await response.clone().text())
  const data = await json(response)
  assert.equal(data.auth, 'password')
  const cookie = response.headers.get('set-cookie')!.split(';')[0]
  assert.ok(cookie)
  const identity = { cookie, id: data.user.id as string, username, password }
  identities.set(alias, identity)
  return { ...identity, response }
}
async function json(response: Awaited<ReturnType<typeof request>>) { return await response.json() as Record<string, any> }

// Complete snapshot: progress, unfinished code, review, settings, and mock interview data.
function snapshot(xp = 42) {
  const now = '2026-10-13T08:00:00.000Z'
  return {
    app: 'codequest',
    version: 2,
    state: {
      xp, levels: { 'algo-search-binary': {
        shape: { sections: 3, quiz: 4, guided: ['guide1'], challenges: ['q1'], qa: 2 },
        sectionsRead: ['intro'], checkpointsCorrect: ['intro'], learnDone: false,
        quizBestCorrect: 2, quizBest: 0.5, quizAttempts: 1, guidedDone: ['guide1'],
        challenges: { q1: { passed: true, usedSolution: false, hintsUsed: 0, attempts: 1, passedAt: now, bestTimeSec: 12.5, difficulty: 'easy' } },
        qaRated: ['qa1'], firstCompletedAt: now, lastVisitedAt: now,
      } }, streak: { current: 1, best: 2, lastDate: '2026-10-13' },
      daily: { '2026-10-13': { xp, answered: 2, correct: 1, solved: 0 } },
      achievements: { first: now },
      review: { 'question::algo-search-binary::q1': { kind: 'question', levelId: 'algo-search-binary', itemId: 'q1', source: 'quiz', wrongCount: 1, streak: 0, interval: 1, due: '2026-10-13', lastAt: '2026-10-13', mastered: false } },
      settings: { theme: 'dark', freeMode: false, editorFontSize: 16, reduceMotion: false, dailyGoal: 100, sound: false },
      lastVisited: { levelId: 'algo-search-binary', stage: 'challenge' }, createdAt: now,
    },
    drafts: { 'challenge::algo-search-binary::q1': '# unfinished code\ndef binary_search(nums, target):\n    pass' },
    mock: {
      active: { id: 'active-interview', config: { durationMin: 45, algoChallenges: 1, llmChallenges: 1, concepts: 6, scope: 'unlocked', tiers: [] }, items: [{ kind: 'question', levelId: 'algo-search-binary', itemId: 'q1', track: 'algo', title: 'Binary search' }], results: [{ status: 'pending', timeSec: 1.75 }], startedAt: Date.parse(now), deadline: Date.parse(now) + 45 * 60_000, active: 0, activeSince: Date.parse(now) },
      history: [{ id: 'past-interview', startedAt: now, finishedAt: now, durationMin: 45, usedSec: 45, score: 100, timeUp: false, xpAwarded: 30, items: [{ kind: 'challenge', levelId: 'algo-search-binary', itemId: 'q1', track: 'algo', title: 'Binary search', difficulty: 'easy', status: 'passed', timeSec: 42 }] }],
      lastConfig: { durationMin: 45, algoChallenges: 1, llmChallenges: 1, concepts: 6, scope: 'unlocked', tiers: [] },
    },
  }
}
const put = (user: string, baseRevision: number, value: ReturnType<typeof snapshot>, mutationId = randomUUID()) => request('/api/cloud/state', { user, method: 'PUT', body: { baseRevision, snapshot: value, mutationId } })
// Miniflare 5's loopback proxy reuses a broken upload socket after early 413.
// For capacity probes, invoke the same Worker handler with the real D1 binding;
// all other requests and browser acceptance exercise the actual HTTP Worker.
const putLarge = (user: string, baseRevision: number, value: ReturnType<typeof snapshot>) => request('/api/cloud/state', { user, method: 'PUT', direct: true, body: { baseRevision, snapshot: value, mutationId: randomUUID() } })

try {
  const db = await mf.getD1Database('DB')
  workerEnv = { DB: db as unknown as WorkerEnv['DB'], ASSETS: { fetch: async () => new Response(null, { status: 404 }) } }
  for (const file of (await readdir('drizzle')).filter((f) => f.endsWith('.sql')).sort()) {
    const migration = await readFile(resolve('drizzle', file), 'utf8')
    for (const statement of migration.split('--> statement-breakpoint').filter((s) => s.trim())) await db.prepare(statement).run()
  }
  const session = await request('/api/cloud/session')
  assert.equal(session.status, 200)
  assert.deepEqual(await json(session), { user: null, auth: 'password' })
  assert.match(session.headers.get('cache-control') ?? '', /no-store/)
  checked('匿名会话仅返回站内认证状态，私有响应禁用缓存')

  assert.equal((await request('/api/cloud/state')).status, 401)
  assert.equal((await request('/api/cloud/state', { method: 'PUT', body: { baseRevision: 0, snapshot: snapshot(), mutationId: randomUUID() } })).status, 401)
  const forged = { 'oai-authenticated-user-id': 'forged', 'oai-authenticated-user-email': 'forged@example.test', 'oai-authenticated-user-full-name': 'forged' }
  assert.equal((await request('/api/cloud/state', { headers: forged })).status, 401)
  assert.equal((await json(await request('/api/cloud/session', { headers: forged }))).user, null)
  assert.equal((await request('/api/cloud/state', { headers: { cookie: '__Host-codequest_session=forged; cq_dev_session=forged' } })).status, 401)
  checked('匿名、伪造会话和 ChatGPT 身份请求头均无法认证或读写存档')

  const alice = await register('alice')
  const bob = await register('bob')
  await register('capacity')
  const cookieHeader = alice.response.headers.get('set-cookie')!
  assert.ok(/^__Host-codequest_session=/.test(cookieHeader), 'HTTPS must use the __Host session cookie')
  for (const flag of [/;\s*HttpOnly/i, /;\s*Secure/i, /;\s*SameSite=Lax/i, /;\s*Path=\//i, /;\s*Max-Age=2592000/i]) assert.ok(flag.test(cookieHeader), 'Session cookie must include every required security attribute')
  assert.ok(!/;\s*Domain=/i.test(cookieHeader), 'Session cookie must remain host-only')
  assert.match(alice.id, /^cqp_/)
  assert.equal((await json(await request('/api/cloud/session', { user: 'alice' }))).user.id, alice.id)
  const token = alice.cookie.split('=')[1]
  assert.equal(Buffer.from(token, 'base64url').length, 32)
  const accountRow = await db.prepare('SELECT * FROM cq_password_users WHERE id = ?').bind(alice.id).first()
  const sessionRow = await db.prepare('SELECT * FROM cq_password_sessions WHERE user_id = ?').bind(alice.id).first()
  assert.ok(accountRow && sessionRow)
  assert.equal(JSON.stringify(accountRow).includes(alice.password), false)
  assert.ok(String(accountRow.password_hash).length >= 64)
  assert.ok(String(accountRow.password_salt).length >= 32)
  assert.equal(accountRow.password_scheme, 'scrypt-v1:N16384:r8:p5')
  assert.equal(accountRow.password_hash, scryptSync(alice.password, Buffer.from(String(accountRow.password_salt), 'hex'), 32, { N: 16384, r: 8, p: 5, maxmem: 32 * 1024 * 1024 }).toString('hex'))
  assert.equal((await request('/api/cloud/state', { headers: { cookie: `${alice.cookie}; ${alice.cookie}` } })).status, 401)
  assert.notEqual(accountRow.password_hash, (await db.prepare('SELECT password_hash FROM cq_password_users WHERE id = ?').bind(bob.id).first())!.password_hash)
  assert.equal(sessionRow.token_hash, createHash('sha256').update(token).digest('hex'))
  assert.equal(JSON.stringify(sessionRow).includes(token), false)
  checked('注册签发随机 host-only 安全会话，D1 仅保存加盐密码派生值和会话摘要')

  const fullWidth = alice.username.toUpperCase().replace(/[A-Z0-9]/g, char => String.fromCharCode(char.charCodeAt(0) + 0xfee0))
  assert.equal((await request('/api/auth/register', { method: 'POST', body: { username: `  ${fullWidth}  `, password: alice.password } })).status, 409)
  const loggedIn = await request('/api/auth/login', { method: 'POST', body: { username: `  ${fullWidth}  `, password: alice.password } })
  assert.equal(loggedIn.status, 200)
  assert.equal((await json(loggedIn)).user.id, alice.id)
  assert.ok(loggedIn.headers.get('set-cookie')!.split(';')[0] !== alice.cookie, 'Every login must issue a new random token')
  const wrong = await request('/api/auth/login', { method: 'POST', body: { username: alice.username, password: 'wrong-password' } })
  const missing = await request('/api/auth/login', { method: 'POST', body: { username: 'Missing_User', password: 'wrong-password' } })
  assert.equal(wrong.status, 401)
  assert.deepEqual(await json(wrong), await json(missing))
  checked('用户名 NFKC 与大小写规范化唯一，重复注册被拒绝，错误用户名和密码返回一致错误')

  for (const path of ['/api/auth/register', '/api/auth/login', '/api/auth/logout', '/api/cloud/state']) {
    const method = path.endsWith('/state') ? 'PUT' : 'POST'
    const body = path.endsWith('/state') ? { baseRevision: 0, snapshot: snapshot(), mutationId: randomUUID() } : { username: alice.username, password: alice.password }
    for (const badOrigin of ['https://evil.test', 'null', null]) {
      assert.equal((await request(path, { user: 'alice', method, origin: badOrigin, body })).status, 403, `Unexpected origin accepted: ${path} ${badOrigin}`)
    }
    assert.equal((await request(path, { user: 'alice', method, headers: { 'sec-fetch-site': 'cross-site' }, body })).status, 403)
    assert.equal((await request(path, { user: 'alice', method, headers: { 'content-type': 'text/plain' }, body })).status, 415)
  }
  checked('认证和存档写入均拒绝跨站、opaque/missing Origin、Fetch Metadata 跨站和非 JSON 请求')

  for (const body of [
    { username: 'x', password: alice.password }, { username: 'x'.repeat(25), password: alice.password },
    { username: 'not valid', password: alice.password }, { username: '<script>', password: alice.password },
    { username: 'username', password: 'short' }, { username: 'username', password: 'x'.repeat(129) },
    { username: null, password: alice.password }, { username: 'username', password: null },
    { username: 'username', password: alice.password, admin: true }, ['username', alice.password], null,
  ]) {
    assert.equal((await request('/api/auth/register', { method: 'POST', body })).status, 400)
  }
  assert.equal((await request('/api/auth/register', { method: 'POST', body: { username: 'oversize', password: 'x'.repeat(5000) } })).status, 413)
  assert.equal((await request('/api/auth/login', { method: 'GET' })).status, 405)
  assert.equal((await request('/api/auth/login', { method: 'POST', base: 'http://public.example', body: { username: alice.username, password: alice.password } })).status, 400)
  const local = await request('/api/auth/login', { method: 'POST', base: 'http://127.0.0.1:8787', body: { username: alice.username, password: alice.password } })
  assert.equal(local.status, 200)
  assert.ok(/^cq_dev_session=/.test(local.headers.get('set-cookie')!), 'Local HTTP uses only the development cookie name')
  assert.ok(!/;\s*Secure/i.test(local.headers.get('set-cookie')!), 'Local HTTP cookie must be usable for development')
  // A localhost development cookie is never accepted on the public HTTPS origin.
  assert.equal((await request('/api/cloud/state', { headers: { cookie: local.headers.get('set-cookie')!.split(';')[0] } })).status, 401)
  checked('非法、过短、过长和超大认证输入被拒绝；仅 localhost 允许 HTTP 开发会话')

  const expireToken = loggedIn.headers.get('set-cookie')!.split(';')[0]
  await db.prepare('UPDATE cq_password_sessions SET expires_at = ? WHERE token_hash = ?').bind(Date.now() - 1, createHash('sha256').update(expireToken.split('=')[1]).digest('hex')).run()
  assert.equal((await request('/api/cloud/state', { headers: { cookie: expireToken } })).status, 401)
  assert.equal((await json(await request('/api/cloud/session', { headers: { cookie: expireToken } }))).user, null)
  const mismatchedLogout = await request('/api/auth/logout', { user: 'bob', method: 'POST', headers: { 'x-codequest-user': alice.id }, body: {} })
  assert.equal(mismatchedLogout.status, 409)
  assert.equal((await json(mismatchedLogout)).error, 'identity_changed')
  assert.equal(mismatchedLogout.headers.get('set-cookie'), null)
  assert.equal((await json(await request('/api/cloud/session', { user: 'bob' }))).user.id, bob.id)
  const logoutIdentity = await register('logout')
  const loggedOut = await request('/api/auth/logout', { user: 'logout', method: 'POST', body: {} })
  assert.equal(loggedOut.status, 200)
  assert.ok(/Max-Age=0/i.test(loggedOut.headers.get('set-cookie')!), 'Logout must expire the browser cookie')
  assert.equal((await request('/api/cloud/state', { user: 'logout' })).status, 401)
  assert.equal((await db.prepare('SELECT COUNT(*) AS n FROM cq_password_sessions WHERE user_id = ?').bind(logoutIdentity.id).first())!.n, 0)
  checked('服务端会话过期和退出均立即失效，重放旧 cookie 无法读取存档')

  const raceName = `Race_${randomBytes(5).toString('hex')}`
  const registrations = await Promise.all(Array.from({ length: 2 }, () => request('/api/auth/register', { method: 'POST', body: { username: raceName, password: randomBytes(20).toString('base64url') } })))
  assert.deepEqual(registrations.map(r => r.status).sort(), [201, 409])
  assert.equal((await db.prepare('SELECT COUNT(*) AS n FROM cq_password_users WHERE username_key = ?').bind(raceName.toLowerCase()).first())!.n, 1)
  checked('真实 D1 并发注册由唯一索引保证同名账号只能创建一次')

  const limitedName = `Limit_${randomBytes(4).toString('hex')}`
  const failures = []
  for (let i = 0; i < 11; i++) failures.push(await request('/api/auth/login', { method: 'POST', headers: { 'cf-connecting-ip': '198.51.100.77' }, body: { username: limitedName, password: 'wrong-password' } }))
  assert.equal(failures[0].status, 401)
  assert.equal(failures.at(-1)!.status, 429)
  assert.ok(Number(failures.at(-1)!.headers.get('retry-after')) > 0)
  assert.ok((await db.prepare('SELECT COUNT(*) AS n FROM cq_password_attempts WHERE attempts > 0').first())!.n as number > 0)
  const acrossIps = await request('/api/auth/login', { method: 'POST', headers: { 'cf-connecting-ip': '198.51.100.78' }, body: { username: limitedName, password: 'wrong-password' } })
  assert.equal(acrossIps.status, 429)
  checked('D1 持久化用户名限速跨请求/来源生效，429 提供重试时间')

  const ratePrefix = `Rate_${randomBytes(3).toString('hex')}`
  for (let i = 0; i < 11; i++) {
    const response = await request('/api/auth/register', { method: 'POST', headers: { 'cf-connecting-ip': '198.51.100.80' }, body: { username: `${ratePrefix}_${i}`, password: randomBytes(18).toString('base64url') } })
    assert.equal(response.status, i < 10 ? 201 : 429)
  }
  const loginIpKey = createHash('sha256').update('login:ip:198.51.100.81').digest('hex')
  // Position an existing durable bucket at the last allowed request. The next
  // two real logins check the boundary without wasting 60 password derivations.
  await db.prepare('INSERT INTO cq_password_attempts (attempt_key, attempts, expires_at) VALUES (?, ?, ?)').bind(loginIpKey, 59, Date.now() + 60_000).run()
  const lastAllowed = await request('/api/auth/login', { method: 'POST', headers: { 'cf-connecting-ip': '198.51.100.81' }, body: { username: `${ratePrefix}_10`, password: 'wrong-password' } })
  assert.equal(lastAllowed.status, 401)
  const ipBlocked = await request('/api/auth/login', { method: 'POST', headers: { 'cf-connecting-ip': '198.51.100.81' }, body: { username: `${ratePrefix}_11`, password: 'wrong-password' } })
  assert.equal(ipBlocked.status, 429)
  await db.prepare('UPDATE cq_password_attempts SET expires_at = ? WHERE attempt_key = ?').bind(Date.now() - 1, loginIpKey).run()
  assert.equal((await request('/api/auth/login', { method: 'POST', headers: { 'cf-connecting-ip': '198.51.100.81' }, body: { username: `${ratePrefix}_12`, password: 'wrong-password' } })).status, 401)
  checked('注册与登录的来源 IP 额度独立持久化，跨用户名生效，到期后可重试')

  const empty = await json(await request('/api/cloud/state', { user: 'alice' }))
  assert.equal(empty.revision, 0)
  assert.equal(empty.snapshot, null)
  const firstMutation = randomUUID()
  const first = await put('alice', 0, snapshot(), firstMutation)
  assert.equal(first.status, 200)
  const saved = await json(first)
  assert.equal(saved.revision, 1)
  assert.equal(saved.user.id, alice.id)
  assert.deepEqual(saved.snapshot, snapshot())
  checked('首次写入完整进度、草稿和面试数据，返回修订号')

  const secondUser = await json(await request('/api/cloud/state', { user: 'bob' }))
  assert.equal(secondUser.revision, 0)
  assert.equal(secondUser.snapshot, null)
  const bobState = await put('bob', 0, snapshot(7))
  assert.equal(bobState.status, 200)
  assert.equal((await json(await request('/api/cloud/state', { user: 'alice' }))).snapshot.state.xp, 42)
  checked('不同认证账号的存档完全隔离')

  const changedIdentity = await request('/api/cloud/state', { user: 'bob', method: 'PUT', headers: { 'x-codequest-user': alice.id }, body: { baseRevision: 1, snapshot: snapshot(9999), mutationId: randomUUID() } })
  assert.equal(changedIdentity.status, 409)
  assert.equal((await json(changedIdentity)).error, 'identity_changed')
  assert.equal((await json(await request('/api/cloud/state', { user: 'bob' }))).snapshot.state.xp, 7)
  checked('请求期间认证身份改变时，拒绝把前一账号队列写入后一账号')

  const retry = await put('alice', 0, snapshot(), firstMutation)
  assert.equal(retry.status, 200)
  assert.equal((await json(retry)).revision, 1)
  const differentRetry = await put('alice', 0, snapshot(999), firstMutation)
  assert.equal(differentRetry.status, 409)
  assert.equal((await put('alice', 1, snapshot(), firstMutation)).status, 409)
  checked('网络重试幂等，同一 mutationId 不能被不同内容重用')

  const concurrent = await Promise.all([put('alice', 1, snapshot(50)), put('alice', 1, snapshot(60))])
  assert.deepEqual(concurrent.map((r) => r.status).sort(), [200, 409])
  const afterRace = await json(await request('/api/cloud/state', { user: 'alice' }))
  assert.equal(afterRace.revision, 2)
  assert.ok([50, 60].includes(afterRace.snapshot.state.xp))
  const loser = await json(concurrent.find((r) => r.status === 409)!)
  assert.equal(loser.error, 'conflict')
  assert.equal(loser.revision, 2)
  assert.deepEqual(loser.snapshot, afterRace.snapshot)
  checked('真实 D1 并发 CAS 只接受一份更新，409 返回当前版本供恢复')

  const oldRetry = await put('alice', 0, snapshot(), firstMutation)
  assert.equal(oldRetry.status, 409)
  assert.equal((await json(oldRetry)).revision, 2)
  checked('已有后续更新时，历史请求重试返回冲突而不伪报保存成功')

  const stale = await put('alice', 1, snapshot(1000))
  assert.equal(stale.status, 409)
  assert.deepEqual((await json(await request('/api/cloud/state', { user: 'alice' }))).snapshot, afterRace.snapshot)
  checked('旧设备写入不会覆盖另一设备已同步的成果')

  for (const body of [
    { baseRevision: -1, snapshot: snapshot(), mutationId: randomUUID() },
    { baseRevision: 2.5, snapshot: snapshot(), mutationId: randomUUID() },
    { baseRevision: 2, snapshot: { ...snapshot(), state: { ...snapshot().state, xp: -5 } }, mutationId: randomUUID() },
    { baseRevision: 2, snapshot: { ...snapshot(), drafts: { code: 12 } }, mutationId: randomUUID() },
  ]) {
    assert.equal((await request('/api/cloud/state', { user: 'alice', method: 'PUT', body })).status, 400)
  }
  assert.deepEqual((await json(await request('/api/cloud/state', { user: 'alice' }))).snapshot, afterRace.snapshot)
  checked('负数/非整数修订、非法进度与草稿被拒绝，原存档不变')

  for (const key of ['__proto__', 'constructor', 'prototype']) {
    const poison = snapshot()
    poison.drafts = JSON.parse(`{"${key}": "pollution"}`)
    assert.equal((await put('alice', 2, poison)).status, 400)
  }
  assert.equal(({} as Record<string, unknown>).pollution, undefined)
  checked('原型污染键无法进入云端存档')

  const large = snapshot()
  large.drafts['challenge::algo-search-binary::q1'] = 'x'.repeat(1_600_000)
  assert.equal((await putLarge('alice', 2, large)).status, 413)
  assert.equal((await json(await request('/api/cloud/state', { user: 'alice' }))).revision, 2)
  checked('超过云存档容量返回 413，既有存档不受影响')

  const unicodeLarge = snapshot()
  unicodeLarge.drafts = Object.fromEntries(Array.from({ length: 5 }, (_, i) => [`draft-${i}`, '汉'.repeat(100_000)])) as typeof unicodeLarge.drafts
  assert.equal((await putLarge('alice', 2, unicodeLarge)).status, 413)
  const withinLimit = snapshot()
  withinLimit.drafts = Object.fromEntries(Array.from({ length: 7 }, (_, i) => [`draft-${i}`, 'x'.repeat(190_000)])) as typeof withinLimit.drafts
  const capacity = await putLarge('capacity', 0, withinLimit)
  assert.equal(capacity.status, 200)
  assert.deepEqual((await json(capacity)).snapshot.drafts, withinLimit.drafts)
  checked('容量按 UTF-8 字节检查，合法大存档可以真实写入 D1')

  // Public read-only probes verify the actual deployed Worker rejects forged auth.
  // They never mutate or print any returned user data.
  if (process.env.CQ_PUBLIC_BASE) {
    const publicUrl = new URL('/api/cloud/state', process.env.CQ_PUBLIC_BASE)
    const probes: Record<string, string>[] = [{}, { 'oai-authenticated-user-id': 'local_seedy', 'oai-authenticated-user-email': 'forged@example.test', 'oai-authenticated-user-full-name': 'forged' }]
    for (const headers of probes) {
      const response = await fetch(publicUrl, { headers })
      await response.arrayBuffer()
      assert.equal(response.status, 401, 'Public endpoint must reject anonymous and forged identity requests')
    }
    checked('公开站点拒绝匿名访问和客户端伪造的身份请求')
  }
  console.log(`Cloud Worker: ${checks} groups passed.`)
} finally {
  await mf.dispose()
  await rm(temp, { recursive: true, force: true })
}
