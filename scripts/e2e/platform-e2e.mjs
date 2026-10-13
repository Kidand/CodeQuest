/** Local account isolation, portable backups, mock interview drafts, and runner recovery in real Chrome. */
import { chromium } from 'playwright-core'
import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import { pbkdf2Sync, randomBytes } from 'node:crypto'

const base = process.env.CQ_TEST_BASE || 'http://127.0.0.1:5173'
const browser = await chromium.launch({ channel: 'chrome', args: ['--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows'] })
const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } })
const page = await ctx.newPage()
await page.addInitScript(() => {
  window.__cqImport = (path) => {
    const entries = performance.getEntriesByType('resource').filter((e) => new URL(e.name).pathname === path)
    const url = entries.at(-1)?.name || path
    return import(url)
  }
})
const errors = []
page.on('pageerror', (e) => errors.push(e.message))
await fs.mkdir('.shots/platform', { recursive: true })
const goto = async (path) => {
  await page.goto(`${base}/#${path}`)
  await page.locator('main').first().waitFor()
  if (path === '/account') {
    const details = page.locator('details').filter({ hasText: '打开旧本机账号 / 使用游客存档' })
    if (!await details.evaluate(el => el.open)) await details.locator('summary').click()
  }
}
const read = () => page.evaluate(async () => {
  const { useProgress, useDrafts } = await window.__cqImport('/src/lib/store.ts')
  const { useMock } = await window.__cqImport('/src/lib/mock.ts')
  return { xp: useProgress.getState().xp, drafts: useDrafts.getState().drafts, mock: useMock.getState().active, id: sessionStorage.getItem('cq-session'), achievements: useProgress.getState().achievements }
})
async function loginLegacy(name, password) {
  await goto('/account')
  await page.locator('#local-username').fill(name)
  await page.locator('#local-password').fill(password)
  await page.getByRole('button', { name: '打开旧本机账号', exact: true }).click()
  await page.getByRole('heading', { name: `本机账号：${name}` }).waitFor()
}
try {
  const localPassword = randomBytes(24).toString('base64url')
  const localAccounts = ['测试玩家甲', '测试玩家乙'].map((username, i) => {
    const salt = randomBytes(16).toString('hex')
    return { id: `legacy-platform-${i}`, username, salt, hash: pbkdf2Sync(localPassword, salt, 600_000, 32, 'sha256').toString('hex'), iterations: 600_000, createdAt: new Date().toISOString() }
  })
  await goto('/account')
  await page.evaluate(async () => {
    const { useProgress, useDrafts } = await window.__cqImport('/src/lib/store.ts')
    useProgress.setState({ xp: 42 })
    useDrafts.getState().setDraft('guest-sample', '# retained guest draft')
  })
  await page.evaluate(accounts => {
    localStorage.setItem('cq-accounts', JSON.stringify(accounts))
    for (const key of ['cq-progress', 'cq-drafts', 'cq-mock']) {
      const value = localStorage.getItem(key)
      if (value) localStorage.setItem(`${key}::${accounts[0].id}`, value)
    }
  }, localAccounts)
  await loginLegacy('测试玩家甲', localPassword)
  const first = await read()
  assert.equal(first.xp, 42)
  assert.equal(first.drafts['guest-sample'], '# retained guest draft')
  assert.ok(first.id)
  assert.ok(!await page.evaluate(password => localStorage.getItem('cq-accounts').includes(password), localPassword))
  await page.getByRole('button', { name: '退出本机账号', exact: true }).click()
  await page.getByRole('heading', { name: '正在使用游客存档' }).waitFor()
  assert.equal((await read()).xp, 42)
  assert.equal((await read()).drafts['guest-sample'], '# retained guest draft')
  await loginLegacy('测试玩家乙', localPassword)
  assert.equal((await read()).xp, 0)
  assert.deepEqual((await read()).drafts, {})
  assert.deepEqual((await read()).achievements, {})
  await goto('/account')
  await page.locator('#local-username').fill('测试玩家甲')
  await page.locator('#local-password').fill('wrong-password')
  await page.getByRole('button', { name: '打开旧本机账号', exact: true }).click()
  await page.getByRole('alert').filter({ hasText: '用户名或密码不正确' }).waitFor()
  await page.locator('#local-password').fill(localPassword)
  await page.getByRole('button', { name: '打开旧本机账号', exact: true }).click()
  await page.getByRole('heading', { name: '本机账号：测试玩家甲' }).waitFor()
  assert.equal((await read()).id, first.id)
  assert.equal((await read()).xp, 42)
  await page.reload()
  await page.getByRole('heading', { name: '本机账号：测试玩家甲' }).waitFor()
  assert.equal((await read()).drafts['guest-sample'], '# retained guest draft')
  console.log('✓ 旧 PBKDF2 账号登录、游客原件保留、错误密码、双账号隔离、刷新恢复')

  const backup = await page.evaluate(async () => (await window.__cqImport('/src/lib/backup.ts')).exportBackup())
  const validation = await page.evaluate(async () => {
    const { exportBackup, importBackup } = await window.__cqImport('/src/lib/backup.ts')
    const { useProgress, useDrafts } = await window.__cqImport('/src/lib/store.ts')
    const good = JSON.parse(exportBackup())
    const bad = structuredClone(good)
    bad.state.levels.bad = { sectionsRead: null }
    let rejected = false
    try { importBackup(JSON.stringify(bad)) } catch { rejected = true }
    const before = useProgress.getState().xp
    useProgress.setState({ xp: 0 }); useDrafts.setState({ drafts: {} })
    good.state.awardXp = null
    importBackup(JSON.stringify(good))
    return { rejected, before, xp: useProgress.getState().xp, action: typeof useProgress.getState().awardXp, drafts: useDrafts.getState().drafts }
  })
  assert.ok(validation.rejected)
  assert.equal(validation.xp, 42)
  assert.equal(validation.action, 'function')
  assert.equal(validation.drafts['guest-sample'], '# retained guest draft')
  const quota = await page.evaluate(async () => {
    const { exportBackup, importBackup } = await window.__cqImport('/src/lib/backup.ts')
    const { useDrafts } = await window.__cqImport('/src/lib/store.ts')
    const before = exportBackup()
    const incoming = JSON.parse(before)
    incoming.state.xp = 987
    incoming.drafts['new-draft'] = 'x'.repeat(100)
    const native = Storage.prototype.setItem
    let once = false, rejected = false
    Storage.prototype.setItem = function (key, value) {
      if (!once && key === useDrafts.persist.getOptions().name) { once = true; throw new DOMException('full', 'QuotaExceededError') }
      return native.call(this, key, value)
    }
    try { importBackup(JSON.stringify(incoming)) } catch { rejected = true }
    finally { Storage.prototype.setItem = native }
    const a = JSON.parse(before), b = JSON.parse(exportBackup())
    delete a.exportedAt; delete b.exportedAt
    return { rejected, preserved: JSON.stringify(a) === JSON.stringify(b) }
  })
  assert.ok(quota.rejected && quota.preserved)
  const download = page.waitForEvent('download')
  await page.getByRole('button', { name: '导出进度', exact: true }).click()
  await (await download).saveAs('.shots/platform/backup.json')
  assert.equal(JSON.parse(await fs.readFile('.shots/platform/backup.json', 'utf8')).version, 2)
  await page.locator('input[type=file]').setInputFiles({ name: 'backup.json', mimeType: 'application/json', buffer: Buffer.from(backup) })
  await page.getByRole('dialog').getByRole('button', { name: '导入', exact: true }).click()
  await page.getByRole('status').filter({ hasText: '已从 backup.json 导入进度' }).waitFor()
  console.log('✓ 完整备份下载/导入、错误文件不污染存档、动作字段不可覆盖')

  await page.evaluate(async () => {
    const { useMock, DEFAULT_MOCK_CONFIG } = await window.__cqImport('/src/lib/mock.ts')
    const { loadLevel } = await window.__cqImport('/src/content/catalog.ts')
    const c = await loadLevel('algo-search-binary')
    const items = c.challenges.map((q) => ({ kind: 'challenge', levelId: 'algo-search-binary', itemId: q.id, track: 'algo', title: q.title, difficulty: q.difficulty }))
    const now = Date.now()
    useMock.getState().start({ id: 'e2e-interview', config: DEFAULT_MOCK_CONFIG, items, results: items.map(() => ({ status: 'pending', timeSec: 0 })), startedAt: now, deadline: now + 1800000, active: 0, activeSince: now })
  })
  await goto('/mock')
  const editor = page.locator('[data-editor] .cm-content')
  await editor.waitFor()
  assert.ok((await page.locator('[data-editor]').boundingBox()).height > 150)
  assert.equal(await page.locator('[data-tab="hints"], [data-tab="solution"]').count(), 0)
  await editor.click(); await page.keyboard.press('ControlOrMeta+a'); await page.keyboard.insertText('# interview draft survives switching')
  await page.getByRole('button', { name: '下一题', exact: true }).click()
  await page.getByRole('button', { name: '上一题', exact: true }).click()
  await page.waitForFunction(() => document.querySelector('.cm-content')?.textContent?.includes('interview draft survives'))
  await page.reload()
  await page.waitForFunction(() => document.querySelector('.cm-content')?.textContent?.includes('interview draft survives'))
  await page.screenshot({ path: '.shots/platform/mock-desktop.png' })
  console.log('✓ 模拟面试编辑器高度、隐藏辅助内容、切题与刷新保留草稿')

  const solution = await page.evaluate(async () => (await (await window.__cqImport('/src/content/catalog.ts')).loadLevel('algo-search-binary')).challenges[0].solution)
  await editor.click(); await page.keyboard.press('ControlOrMeta+a'); await page.keyboard.insertText(solution)
  await page.locator('[data-action="submit"]').click()
  await page.locator('[data-passed-banner]').waitFor({ timeout: 60000 })
  assert.equal((await read()).mock.results[0].status, 'passed')
  assert.equal((await read()).xp, 42)
  await page.getByRole('button', { name: '交卷', exact: true }).click()
  await page.getByRole('dialog').getByRole('button', { name: '交卷', exact: true }).click()
  await page.waitForFunction(() => !document.querySelector('[role="timer"]'))
  assert.equal((await read()).mock, null)
  assert.equal((await read()).xp, 92)
  console.log('✓ 面试实战真实通过、隔离普通练习奖励、交卷生成报告与一次性 XP')

  const recovery = await page.evaluate(async () => {
    const { pyRunner } = await window.__cqImport('/src/runner/client.ts')
    const result = await pyRunner.run({ code: 'def identity(x):\n    return x', tests: [{ name: 'finished', code: 'assert identity(7) == 7' }, { name: 'loop', code: 'while True: pass' }] }, 1000)
    const next = await pyRunner.run({ code: 'import numpy as np\nx = np.array([1,2,3])', tests: [{ name: 'numpy recovered', code: 'assert x.sum() == 6' }] })
    return { timeout: result.timedOut, preserved: result.results.map((r) => r.name), recovered: next.ok }
  })
  assert.ok(recovery.timeout)
  assert.deepEqual(recovery.preserved, ['finished'])
  assert.ok(recovery.recovered)
  console.log('✓ 超时保留已完成用例、自动重建 Worker 后 NumPy 继续运行')

  const loading = await page.evaluate(async () => {
    const { pyRunner } = await window.__cqImport('/src/runner/client.ts')
    const NativeWorker = window.Worker
    const nativeTimeout = window.setTimeout
    class StalledWorker { postMessage() {} terminate() {} }
    window.Worker = StalledWorker
    const Ctor = pyRunner.constructor
    const stalled = new Ctor()
    const job = stalled.run({ code: '', tests: [] })
    await new Promise((r) => nativeTimeout(r, 10))
    const busy = stalled.snapshot.busy
    stalled.cancel()
    const cancelled = (await job).cancelled
    await new Promise((r) => nativeTimeout(r, 80))
    stalled.kill()
    window.setTimeout = (fn, ms, ...args) => nativeTimeout(fn, ms === 60000 ? 20 : ms, ...args)
    const limited = new Ctor()
    const failed = await limited.run({ code: '', tests: [] })
    limited.kill()
    window.Worker = NativeWorker
    window.setTimeout = nativeTimeout
    return { busy, cancelled, bounded: /60 秒/.test(failed.infraError ?? '') }
  })
  assert.ok(loading.busy && loading.cancelled && loading.bounded)
  console.log('✓ 加载阶段可取消、挂起资源有加载超时、备份写入失败完整回滚')
  assert.deepEqual(errors, [])
  console.log('全部平台验收通过，页面无未捕获异常。')
} catch (e) {
  console.log('Debug state:', await read())
  console.log('Storage namespaces:', await page.evaluate(() => Object.keys(localStorage).filter((k) => k.startsWith('cq-progress')).map((k) => [k, JSON.parse(localStorage.getItem(k)).state?.xp])))
  await page.screenshot({ path: '.shots/platform/failure.png' }).catch(() => {})
  throw e
} finally { await browser.close() }
