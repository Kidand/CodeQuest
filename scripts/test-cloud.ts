/** Security and concurrency checks against the real Worker and a disposable D1 database. */
import assert from 'node:assert/strict'
import { randomUUID } from 'node:crypto'
import { mkdtemp, readFile, readdir, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { build } from 'esbuild'
import { Miniflare, convertV4MiniflareOptions } from 'miniflare'
import { handleRequest, type WorkerEnv } from '../worker/index'

const origin = 'https://codequest.test'
const temp = await mkdtemp(join(tmpdir(), 'codequest-cloud-test-'))
const workerFile = join(temp, 'worker.mjs')
await build({ entryPoints: ['worker/index.ts'], bundle: true, format: 'esm', platform: 'browser', target: 'es2022', outfile: workerFile })
const mf = new Miniflare(convertV4MiniflareOptions({ name: 'cloud-test', modules: true, modulesRoot: temp, scriptPath: workerFile, compatibilityDate: '2026-05-15', d1Databases: { DB: 'cq-test' } }))
let checks = 0
let workerEnv: WorkerEnv
function checked(label: string) { checks++; console.log(`✓ ${label}`) }
const username = (id: string) => `Test ${id}`
async function request(path: string, options: { user?: string; method?: string; body?: unknown; origin?: string | null; headers?: Record<string, string>; direct?: boolean } = {}) {
  const headers: Record<string, string> = { ...options.headers }
  if (options.user) Object.assign(headers, { 'oai-authenticated-user-id': options.user, 'oai-authenticated-user-email': `${options.user}@example.test`, 'oai-authenticated-user-full-name': username(options.user) })
  if (options.method === 'PUT') {
    headers['content-type'] = 'application/json'
    if (options.origin !== null) headers.origin = options.origin ?? origin
  }
  const body = options.body === undefined ? undefined : JSON.stringify(options.body)
  if (options.direct && body) headers['content-length'] = String(Buffer.byteLength(body))
  const init = { method: options.method ?? 'GET', headers, body }
  const response = options.direct
    ? await handleRequest(new Request(`${origin}${path}`, init), workerEnv)
    : await mf.dispatchFetch(`${origin}${path}`, init)
  await response.clone().arrayBuffer()
  return response
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
  assert.equal((await json(session)).user, null)
  assert.match(session.headers.get('cache-control') ?? '', /no-store/)
  checked('匿名会话只返回未登录状态，私有响应禁用缓存')

  assert.equal((await request('/api/cloud/state')).status, 401)
  const anonymousWrite = await request('/api/cloud/state', { method: 'PUT', body: { baseRevision: 0, snapshot: snapshot(), mutationId: randomUUID() } })
  assert.equal(anonymousWrite.status, 401, await anonymousWrite.text())
  checked('未登录无法读取或写入学习存档')

  for (const badOrigin of ['https://evil.test', 'null', null]) {
    const response = await request('/api/cloud/state', { user: 'alice', method: 'PUT', origin: badOrigin, body: { baseRevision: 0, snapshot: snapshot(), mutationId: randomUUID() } })
    assert.equal(response.status, 403, `Unexpected origin accepted: ${badOrigin}`)
  }
  checked('跨站、opaque Origin 及缺少 Origin 的写入都被拒绝')

  assert.equal((await request('/api/cloud/state', { user: 'alice', method: 'PUT', headers: { 'sec-fetch-site': 'cross-site' }, body: { baseRevision: 0, snapshot: snapshot(), mutationId: randomUUID() } })).status, 403)
  const wrongType = await mf.dispatchFetch(`${origin}/api/cloud/state`, { method: 'PUT', headers: { origin, 'content-type': 'text/plain', 'oai-authenticated-user-id': 'alice', 'oai-authenticated-user-email': 'alice@example.test' }, body: JSON.stringify({ baseRevision: 0, snapshot: snapshot(), mutationId: randomUUID() }) })
  assert.equal(wrongType.status, 415)
  checked('Fetch Metadata 跨站请求与非 JSON 写入均被拒绝')

  const empty = await json(await request('/api/cloud/state', { user: 'alice' }))
  assert.equal(empty.revision, 0)
  assert.equal(empty.snapshot, null)
  const firstMutation = randomUUID()
  const first = await put('alice', 0, snapshot(), firstMutation)
  assert.equal(first.status, 200)
  const saved = await json(first)
  assert.equal(saved.revision, 1)
  assert.equal(saved.user.id, 'alice')
  assert.deepEqual(saved.snapshot, snapshot())
  checked('首次写入完整进度、草稿和面试数据，返回修订号')

  const secondUser = await json(await request('/api/cloud/state', { user: 'bob' }))
  assert.equal(secondUser.revision, 0)
  assert.equal(secondUser.snapshot, null)
  const bob = await put('bob', 0, snapshot(7))
  assert.equal(bob.status, 200)
  assert.equal((await json(await request('/api/cloud/state', { user: 'alice' }))).snapshot.state.xp, 42)
  checked('不同认证账号的存档完全隔离')

  const changedIdentity = await request('/api/cloud/state', { user: 'bob', method: 'PUT', headers: { 'x-codequest-user': 'alice' }, body: { baseRevision: 1, snapshot: snapshot(9999), mutationId: randomUUID() } })
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
  const capacity = await putLarge('capacity-test', 0, withinLimit)
  assert.equal(capacity.status, 200)
  assert.deepEqual((await json(capacity)).snapshot.drafts, withinLimit.drafts)
  checked('容量按 UTF-8 字节检查，合法大存档可以真实写入 D1')

  // Auth headers are injected by Sites only. Public probes assert the gateway strips
  // client-supplied identity headers; they never mutate or print any returned data.
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
