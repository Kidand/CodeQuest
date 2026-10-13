/** Cross-device/cloud durability acceptance against Vite + the real D1-backed Worker. */
import { chromium } from 'playwright-core'
import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import { pbkdf2Sync } from 'node:crypto'

const base = process.env.CQ_TEST_BASE || 'http://127.0.0.1:5173'
if (!['localhost', '127.0.0.1'].includes(new URL(base).hostname)) throw new Error('This test intentionally replaces its disposable local D1 archive. Use a localhost CQ_TEST_BASE.')
const out = 'output/playwright/cloud'
await fs.mkdir(out, { recursive: true })
const browser = await chromium.launch({ channel: 'chrome', args: ['--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows'] })
const errors = []
const contexts = []
async function device(viewport = { width: 1365, height: 1000 }, seed = {}, beforeNavigate) {
  const context = await browser.newContext({ viewport })
  contexts.push(context)
  await context.addInitScript(({ seed }) => {
    window.__cqImport = path => {
      const entries = performance.getEntriesByType('resource').filter(e => new URL(e.name).pathname === path)
      return import(entries.at(-1)?.name || path)
    }
    if (!sessionStorage.getItem('cq-e2e-seeded')) {
      for (const [key, value] of Object.entries(seed)) localStorage.setItem(key, JSON.stringify(value))
      sessionStorage.setItem('cq-e2e-seeded', 'yes')
    }
  }, { seed })
  if (beforeNavigate) await beforeNavigate(context)
  const page = await context.newPage()
  page.on('pageerror', e => errors.push(e.message))
  await page.goto(`${base}/#/account`)
  await page.locator('main').waitFor()
  await ready(page)
  if (!beforeNavigate && !(await state(page)).user) {
    await Promise.all([
      page.waitForEvent('domcontentloaded'),
      page.getByRole('link', { name: '使用 ChatGPT 登录云端', exact: true }).click(),
    ])
    await page.locator('main').waitFor()
    await ready(page)
  }
  return { context, page }
}
const run = (page, fn, value) => page.evaluate(async ({ source, value }) => {
  const cloud = await window.__cqImport('/src/lib/cloud.ts')
  const store = await window.__cqImport('/src/lib/store.ts')
  const mock = await window.__cqImport('/src/lib/mock.ts')
  return (new Function('cloud', 'store', 'mock', 'value', `return (${source})(cloud, store, mock, value)`))(cloud, store, mock, value)
}, { source: fn.toString(), value })
async function wait(page, check, label = 'cloud condition') {
  const deadline = Date.now() + 15_000
  while (Date.now() < deadline) {
    const value = await run(page, (cloud, store, mock) => ({ ...cloud.useCloud.getState(), snapshot: cloud.captureCloudSnapshot() }))
    if (check(value)) return value
    await page.waitForTimeout(100)
  }
  throw new Error(`Timed out: ${label}; state=${JSON.stringify(await state(page))}`)
}
const ready = page => wait(page, s => s.ready && !['checking', 'syncing'].includes(s.status), 'initialized')
const saved = page => wait(page, s => s.status === 'saved' && !s.dirty, 'saved to D1')
const state = page => run(page, cloud => {
  const { user, status, dirty, revision, error, conflict } = cloud.useCloud.getState()
  return { user, status, dirty, revision, error, conflict, snapshot: cloud.captureCloudSnapshot() }
})
const normalized = value => JSON.parse(JSON.stringify(value))
async function clickAndConfirm(page, name) {
  await page.getByRole('button', { name, exact: true }).click()
  await page.getByRole('button', { name: name === '使用云端版本' ? '确认选用云端版本' : '确认选用此设备版本', exact: true }).click()
}

try {
  const a = await device()
  assert.ok((await state(a.page)).user?.id, 'Sites development authentication must inject the local test user')
  const first = await run(a.page, async (cloud, store, mock) => {
    const fixture = cloud.captureCloudSnapshot()
    fixture.state = { ...store.initialData(), xp: 142, settings: { ...store.initialData().settings, theme: 'dark', editorFontSize: 17, dailyGoal: 150 }, lastVisited: { levelId: 'algo-search-binary', stage: 'challenge' } }
    fixture.state.streak = { current: 3, best: 5, lastDate: '2026-10-13' }
    fixture.state.daily = { '2026-10-13': { xp: 142, answered: 7, correct: 5, solved: 1 } }
    fixture.state.achievements = { first_step: new Date().toISOString() }
    fixture.state.review = { 'question::algo-search-binary::binary-q1': { kind: 'question', levelId: 'algo-search-binary', itemId: 'binary-q1', source: 'quiz', wrongCount: 2, streak: 0, interval: 1, due: '2026-10-13', lastAt: '2026-10-13', mastered: false } }
    fixture.drafts = { 'challenge::algo-search-binary::binary-search': '# migrated unfinished draft\ndef search(nums, target):\n    return -1' }
    const now = Date.now()
    const item = { kind: 'question', levelId: 'algo-search-binary', itemId: 'binary-q1', track: 'algo', title: '跨设备面试记录' }
    fixture.mock = {
      active: { id: 'cross-device-active', config: mock.DEFAULT_MOCK_CONFIG, items: [item], results: [{ status: 'pending', timeSec: 7.5 }], startedAt: now, deadline: now + 45 * 60_000, active: 0, activeSince: now },
      history: [{ id: 'cross-device-history', startedAt: new Date(now - 120000).toISOString(), finishedAt: new Date(now - 60000).toISOString(), durationMin: 45, usedSec: 60, score: 100, timeUp: false, xpAwarded: 30, items: [{ ...item, status: 'correct', timeSec: 60 }] }],
      lastConfig: { ...mock.DEFAULT_MOCK_CONFIG, durationMin: 60 },
    }
    await cloud.migrateCloudSnapshot(fixture)
    return fixture
  })
  await saved(a.page)
  const b = await device({ width: 390, height: 844 })
  assert.deepEqual(normalized((await state(b.page)).snapshot), normalized(first))
  console.log('✓ 两个独立设备登录同一云账号，完整恢复 XP、复习、打卡、设置、代码、面试进行中和历史')

  await run(a.page, (cloud, store) => {
    store.useProgress.getState().markSectionRead('algo-search-binary', 'cloud-read')
    store.useDrafts.getState().setDraft('autosaved-draft', '# autosaved from device A')
  })
  await saved(a.page)
  await run(b.page, cloud => cloud.refreshCloud())
  await saved(b.page)
  assert.deepEqual(normalized((await state(b.page)).snapshot), normalized((await state(a.page)).snapshot))
  console.log('✓ 正常学习动作和代码编辑自动同步，另一设备刷新恢复')

  // Simulate an unavailable cloud API, preserving the cached application so a reload
  // can exercise the durable queue even without a service worker.
  await b.page.route('**/api/cloud/**', route => route.abort('internetdisconnected'))
  await run(b.page, (cloud, store) => {
    store.useProgress.setState({ xp: 888 })
    store.useDrafts.getState().setDraft('offline-only', '# device B pending work')
    return cloud.flushCloud()
  })
  await wait(b.page, s => s.dirty && ['offline', 'error'].includes(s.status), 'offline queue persisted')
  await b.page.reload()
  await ready(b.page)
  const offline = await state(b.page)
  assert.equal(offline.snapshot.state.xp, 888)
  assert.equal(offline.snapshot.drafts['offline-only'], '# device B pending work')
  assert.ok(offline.dirty)
  console.log('✓ API 断网后编辑持久化，刷新继续保留未同步进度和代码')

  await run(a.page, (cloud, store) => {
    store.useProgress.setState({ xp: 777 })
    store.useDrafts.getState().setDraft('remote-only', '# device A newest work')
  })
  await saved(a.page)
  await b.page.unroute('**/api/cloud/**')
  await run(b.page, cloud => cloud.refreshCloud())
  await wait(b.page, s => s.status === 'conflict', 'stale device conflict')
  assert.equal((await state(b.page)).snapshot.state.xp, 888)
  assert.equal((await state(a.page)).snapshot.state.xp, 777)
  await b.page.screenshot({ path: `${out}/conflict-mobile.png`, fullPage: true })
  await clickAndConfirm(b.page, '使用云端版本')
  await saved(b.page)
  assert.equal((await state(b.page)).snapshot.state.xp, 777)
  assert.equal((await state(b.page)).snapshot.drafts['remote-only'], '# device A newest work')
  console.log('✓ 离线冲突阻止静默覆盖，手机端可选择云端版本恢复')

  // A second conflict checks the other explicit recovery path.
  await b.page.route('**/api/cloud/**', route => route.abort('internetdisconnected'))
  await run(b.page, (cloud, store) => { store.useProgress.setState({ xp: 999 }); return cloud.flushCloud() })
  await wait(b.page, s => s.dirty, 'second pending change')
  await run(a.page, (cloud, store) => store.useProgress.setState({ xp: 778 }))
  await saved(a.page)
  await b.page.unroute('**/api/cloud/**')
  await run(b.page, cloud => cloud.refreshCloud())
  await wait(b.page, s => s.status === 'conflict', 'second conflict')
  await clickAndConfirm(b.page, '使用此设备版本')
  await saved(b.page)
  await run(a.page, cloud => cloud.refreshCloud())
  await saved(a.page)
  assert.equal((await state(a.page)).snapshot.state.xp, 999)
  console.log('✓ 用户明确选择此设备版本后，其他设备恢复该版本')

  // A 401 must retain the user's pending archive without sending it to guest or
  // to another authenticated account. The real Worker separately tests identity CAS.
  const beforeExpiry = await state(b.page)
  await b.page.route('**/api/cloud/state', route => route.fulfill({ status: 401, contentType: 'application/json', body: JSON.stringify({ error: 'unauthorized' }) }))
  await run(b.page, (cloud, store) => { store.useDrafts.getState().setDraft('expired-only', '# retain after expiration'); return cloud.flushCloud() })
  await wait(b.page, s => s.status === 'expired', 'expired session')
  const expired = await state(b.page)
  assert.ok(expired.dirty)
  assert.equal(expired.snapshot.drafts['expired-only'], '# retain after expiration')
  assert.equal(expired.user?.id, beforeExpiry.user.id)
  await b.page.unroute('**/api/cloud/state')
  await run(b.page, cloud => cloud.refreshCloud())
  await saved(b.page)
  console.log('✓ 登录过期保留原账号未同步队列，重新验证身份后可继续同步')

  const localId = 'legacy-e2e-player'
  const localPassword = 'legacy-e2e-password'
  const salt = 'e2e-salt-only-test'
  const localAccount = { id: localId, username: '旧本机玩家', salt, hash: pbkdf2Sync(localPassword, salt, 600_000, 32, 'sha256').toString('hex'), iterations: 600_000, createdAt: '2026-10-01T00:00:00.000Z' }
  const seed = {
    'cq-accounts': [localAccount],
    'cq-progress': { version: 1, state: { ...first.state, xp: 123 } },
    'cq-drafts': { version: 0, state: { drafts: { guest: '# old guest' } } },
    'cq-mock': { version: 1, state: first.mock },
    [`cq-progress::${localId}`]: { version: 1, state: { ...first.state, xp: 321 } },
    [`cq-drafts::${localId}`]: { version: 0, state: { drafts: { legacy: '# old local account' } } },
    [`cq-mock::${localId}`]: { version: 1, state: first.mock },
  }
  const c = await device({ width: 1365, height: 1000 }, seed)
  for (const [name, xp] of [['游客存档', 123], ['旧本机玩家', 321]]) {
    await c.page.getByRole('heading', { name, exact: true }).locator('..').locator('..').getByRole('button', { name: '迁移此存档', exact: true }).click()
    await c.page.getByRole('button', { name: '确认替换云端存档', exact: true }).click()
    await saved(c.page)
    assert.equal((await state(c.page)).snapshot.state.xp, xp)
    const preserved = await c.page.evaluate(keys => Object.fromEntries(keys.map(key => [key, JSON.parse(localStorage.getItem(key))])), Object.keys(seed))
    assert.deepEqual(preserved, seed)
  }
  console.log('✓ 真实界面分别迁移游客和旧本机账号，原有进度、代码、面试及密码记录全部保留')

  const file = await run(c.page, async () => JSON.parse((await window.__cqImport('/src/lib/backup.ts')).exportBackup()))
  assert.ok(file.exportedAt, 'Fixture must be a genuine v2 export with metadata')
  file.state.xp = 432
  await c.page.locator('input[type=file]').setInputFiles({ name: 'codequest-v2-backup.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(file)) })
  await c.page.getByRole('button', { name: '确认替换云端存档', exact: true }).click()
  await saved(c.page)
  assert.equal((await state(c.page)).snapshot.state.xp, 432)
  assert.equal(JSON.stringify((await state(c.page)).snapshot).includes(localAccount.hash), false)
  console.log('✓ 旧版带 exportedAt 的真实 JSON 备份可上传，账号密码不会进入云端学习存档')

  let anonymousWrites = 0
  const d = await device({ width: 390, height: 844 }, seed, async context => {
    context.on('request', req => { if (req.url().endsWith('/api/cloud/state') && req.method() === 'PUT') anonymousWrites++ })
    await context.route('**/api/cloud/session**', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ user: null, signInUrl: '/signin-with-chatgpt?return_to=%2F%23%2Faccount', signOutUrl: '/signout-with-chatgpt?return_to=%2F%23%2Faccount' }) }))
    await context.route('**/api/cloud/state', route => route.fulfill({ status: 401, contentType: 'application/json', body: '{"error":"unauthenticated"}' }))
  })
  await run(d.page, async (cloud, store, mock, password) => {
    const accounts = await window.__cqImport('/src/lib/accounts.ts')
    await accounts.loginAccount('旧本机玩家', password)
  }, localPassword)
  assert.equal((await state(d.page)).user, null)
  assert.equal((await state(d.page)).snapshot.state.xp, 321)
  assert.equal(anonymousWrites, 0)
  await d.page.goto(`${base}/#/account`)
  await d.page.getByRole('link', { name: '使用 ChatGPT 登录云端', exact: true }).waitFor()
  await d.page.screenshot({ path: `${out}/anonymous-mobile.png`, fullPage: true })
  console.log('✓ 旧本机密码只打开设备存档，不能冒充云端认证身份或产生云端写入')

  await d.page.evaluate(async password => {
    const native = crypto.subtle.deriveBits.bind(crypto.subtle)
    crypto.subtle.deriveBits = async (...args) => {
      const bits = await native(...args)
      await new Promise(resolve => { window.__releaseHash = resolve })
      return bits
    }
    const accounts = await window.__cqImport('/src/lib/accounts.ts')
    window.__oldLogin = accounts.loginAccount('旧本机玩家', password).then(() => ({ ok: true }), error => ({ error: error.message })).finally(() => { crypto.subtle.deriveBits = native })
  }, localPassword)
  await d.page.waitForFunction(() => typeof window.__releaseHash === 'function')
  const hashRaceUser = { id: 'hash-race-cloud', username: 'New cloud account', email: 'new@example.test' }
  const hashRaceSnapshot = { ...structuredClone(first), state: { ...structuredClone(first.state), xp: 5678 } }
  const hashWrites = []
  await d.context.unroute('**/api/cloud/session**')
  await d.context.unroute('**/api/cloud/state')
  await d.context.route('**/api/cloud/session**', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ user: hashRaceUser, signInUrl: '/signin-with-chatgpt', signOutUrl: '/signout-with-chatgpt' }) }))
  await d.context.route('**/api/cloud/state', route => {
    const input = route.request().method() === 'PUT' ? route.request().postDataJSON() : null
    if (input) hashWrites.push(input.snapshot)
    return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ user: hashRaceUser, revision: input ? input.baseRevision + 1 : 1, updatedAt: new Date().toISOString(), snapshot: input?.snapshot ?? hashRaceSnapshot }) })
  })
  await run(d.page, cloud => cloud.refreshCloud())
  assert.equal((await state(d.page)).snapshot.state.xp, 5678)
  await d.page.evaluate(async () => { window.__releaseHash(); await window.__oldLogin })
  assert.equal((await state(d.page)).user.id, hashRaceUser.id)
  assert.equal((await state(d.page)).snapshot.state.xp, 5678)
  assert.equal((await state(d.page)).snapshot.drafts.legacy, undefined)
  await run(d.page, (cloud, store) => store.useDrafts.getState().setDraft('new-cloud-edit', '# after delayed local password'))
  await saved(d.page)
  assert.ok(hashWrites.every(snapshot => snapshot.state.xp === 5678 && snapshot.drafts.legacy === undefined))
  console.log('✓ 本机密码校验晚于云端登录返回时，不会切回旧存档或写错云账号')

  // Controlled auth responses isolate a race that is difficult to reproduce with
  // an interactive provider: the old user's PUT completes after a new login.
  let authId = 'browser-e2e-A'
  const authUser = id => ({ id, username: id, email: `${id}@example.test` })
  const rows = new Map(['browser-e2e-A', 'browser-e2e-B'].map((id, i) => [id, { user: authUser(id), revision: 1, updatedAt: new Date().toISOString(), snapshot: { ...structuredClone(first), state: { ...structuredClone(first.state), xp: 201 + i } } }]))
  const writes = []
  let releaseOld
  const stalled = new Promise(resolve => { releaseOld = resolve })
  let oldRequestSeen = false
  let delayNextGet = false, getRequestSeen = false, releaseGet
  const delayedGet = new Promise(resolve => { releaseGet = resolve })
  const e = await device({ width: 1365, height: 1000 }, {}, async context => {
    await context.route('**/api/cloud/session**', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ user: authUser(authId), signInUrl: '/signin-with-chatgpt', signOutUrl: '/signout-with-chatgpt' }) }))
    await context.route('**/api/cloud/state', async route => {
      const request = route.request()
      const id = authId
      if (request.method() === 'PUT') {
        const input = request.postDataJSON()
        writes.push({ id, expectedId: request.headers()['x-codequest-user'], snapshot: input.snapshot })
        if (id === 'browser-e2e-A') { oldRequestSeen = true; await stalled }
        const next = { user: authUser(id), revision: input.baseRevision + 1, updatedAt: new Date().toISOString(), snapshot: input.snapshot }
        rows.set(id, next)
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(next) })
      } else {
        const row = structuredClone(rows.get(id))
        if (delayNextGet) { delayNextGet = false; getRequestSeen = true; await delayedGet }
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(row) })
      }
    })
  })
  await run(e.page, (cloud, store) => store.useDrafts.getState().setDraft('account-A-private', '# only A'))
  for (let i = 0; i < 50 && !oldRequestSeen; i++) await e.page.waitForTimeout(100)
  assert.ok(oldRequestSeen)
  authId = 'browser-e2e-B'
  await run(e.page, cloud => cloud.refreshCloud())
  assert.equal((await state(e.page)).user.id, authId)
  assert.equal((await state(e.page)).snapshot.state.xp, 202)
  assert.equal((await state(e.page)).snapshot.drafts['account-A-private'], undefined)
  await run(e.page, (cloud, store) => store.useDrafts.getState().setDraft('account-B-private', '# only B'))
  await e.page.waitForTimeout(900)
  releaseOld()
  await saved(e.page)
  assert.equal((await state(e.page)).user.id, authId)
  assert.equal((await state(e.page)).snapshot.drafts['account-A-private'], undefined)
  assert.ok(writes.some(write => write.id === 'browser-e2e-B' && write.expectedId === 'browser-e2e-B' && write.snapshot.drafts['account-B-private'] === '# only B' && !write.snapshot.drafts['account-A-private']))
  console.log('✓ 旧账号请求延迟返回不能污染新账号，新账号编辑仍自动同步到正确身份')

  delayNextGet = true
  await run(e.page, cloud => { void cloud.refreshCloud() })
  for (let i = 0; i < 50 && !getRequestSeen; i++) await e.page.waitForTimeout(100)
  assert.ok(getRequestSeen)
  await run(e.page, (cloud, store) => store.useDrafts.getState().setDraft('newer-than-get', '# saved after GET began'))
  await saved(e.page)
  const newest = await state(e.page)
  releaseGet()
  await run(e.page, cloud => cloud.refreshCloud())
  assert.equal((await state(e.page)).revision, newest.revision)
  assert.deepEqual((await state(e.page)).snapshot, newest.snapshot)
  const cacheRevision = await e.page.evaluate(id => JSON.parse(localStorage.getItem(`cq-cloud-cache::${encodeURIComponent(id)}`)).revision, authId)
  assert.equal(cacheRevision, newest.revision)
  console.log('✓ 延迟返回的旧 GET 不会倒退已保存的进度、修订号或离线缓存')

  const f = await device()
  await f.context.route('**/api/cloud/**', route => route.abort('internetdisconnected'))
  await run(f.page, (cloud, store) => { store.useDrafts.getState().setDraft('parent-start', '# pending before duplicate'); return cloud.flushCloud() })
  const popupEvent = f.page.waitForEvent('popup')
  await f.page.evaluate(() => { window.open(location.href, '_blank') })
  const duplicate = await popupEvent
  duplicate.on('pageerror', error => errors.push(error.message))
  await duplicate.locator('main').waitFor()
  await ready(duplicate)
  const parentTab = await f.page.evaluate(() => sessionStorage.getItem('cq-cloud-tab'))
  const childTab = await duplicate.evaluate(() => sessionStorage.getItem('cq-cloud-tab'))
  assert.notEqual(parentTab, childTab, 'A cloned tab must not reuse the durable journal identity')
  await run(duplicate, (cloud, store) => { store.useDrafts.getState().setDraft('child-only', '# independent child pending work'); return cloud.flushCloud() })
  await run(f.page, (cloud, store) => { store.useDrafts.getState().setDraft('parent-only', '# independent parent pending work'); return cloud.flushCloud() })
  await f.page.reload()
  await ready(f.page)
  assert.equal((await state(f.page)).snapshot.drafts['parent-only'], '# independent parent pending work')
  assert.equal((await state(f.page)).snapshot.drafts['child-only'], undefined)
  assert.ok((await state(f.page)).dirty)
  await duplicate.reload()
  await ready(duplicate)
  assert.equal((await state(duplicate)).snapshot.drafts['child-only'], '# independent child pending work')
  assert.equal((await state(duplicate)).snapshot.drafts['parent-only'], undefined)
  assert.ok((await state(duplicate)).dirty)
  console.log('✓ window.open 复制标签后两份离线日志独立，任一标签刷新均保留自己的未同步代码')

  await a.page.goto(`${base}/#/account`)
  await ready(a.page)
  await a.page.screenshot({ path: `${out}/account-desktop.png`, fullPage: true })
  await b.page.goto(`${base}/#/account`)
  await ready(b.page)
  await b.page.screenshot({ path: `${out}/account-mobile.png`, fullPage: true })
  assert.equal(await b.page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false)
  assert.deepEqual(errors, [])
  console.log('✓ 桌面与手机云账号页无横向溢出，无未捕获异常')
  console.log('All cloud browser acceptance checks passed.')
} catch (error) {
  for (const [index, context] of contexts.entries()) {
    const page = context.pages()[0]
    if (page) {
      await page.screenshot({ path: `${out}/failure-${index}.png`, fullPage: true }).catch(() => {})
      console.log(`Device ${index}:`, await state(page).catch(() => 'unavailable'))
    }
  }
  throw error
} finally { await browser.close() }
