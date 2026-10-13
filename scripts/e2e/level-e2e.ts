#!/usr/bin/env tsx
/**
 * 关卡页端到端回归（真实页面 + 真实 Pyodide，本机 Chrome 无头运行）。
 *
 *   npx tsx scripts/e2e/level-e2e.ts [--base http://localhost:5202] [--level algo-search-binary] [--skip-loop] [--skip-flow] [--headed]
 *
 * 覆盖：
 *  0. 闭环：图解逐节读完 → completeLearn；随堂按题型作答（≥60%）→ ★ → 自动弹出通关结算
 *  1. 实战：每道题 参考解「提交」→ 全部通过（并写入进度 passed）；starter「运行/提交」→ 失败
 *  2. 实战：死循环 → 超时提示；之后再提交参考解仍能正常通过（Worker 已自动重建）
 *  3. 跟练：错误写法判错；未列入 accept 的等价写法经「跑测试」判为等价；填完后完整测试通过并 completeGuided
 *  4. 全程收集 console error / pageerror，有则失败
 * 需要先启动 dev server（默认 5202 端口）。
 */
import { chromium, type Page } from 'playwright-core'
import fs from 'node:fs/promises'
import { loadChapters, loadLevelContent } from '../lib/content'
import type { LevelContent } from '../../src/content/types'

const argv = process.argv.slice(2)
const arg = (k: string, d?: string) => {
  const i = argv.indexOf(k)
  return i >= 0 ? argv[i + 1] : d
}
const BASE = arg('--base', 'http://127.0.0.1:5173')!
const LEVEL = arg('--level', 'algo-search-binary')!
const SKIP_LOOP = argv.includes('--skip-loop')
const SKIP_FLOW = argv.includes('--skip-flow')
const HEADED = argv.includes('--headed')

let failures = 0
const ok = (msg: string) => console.log(`  \x1b[32m✓\x1b[0m ${msg}`)
const bad = (msg: string) => {
  failures++
  console.log(`  \x1b[31m✗ ${msg}\x1b[0m`)
}
const check = (cond: boolean, msg: string, detail = '') => (cond ? ok(msg) : bad(detail ? `${msg} —— ${detail}` : msg))

async function findLevel(id: string): Promise<LevelContent> {
  for (const ch of await loadChapters()) {
    const meta = ch.chapter?.levels.find((l) => l.id === id)
    if (meta) return loadLevelContent(meta)
  }
  throw new Error(`找不到关卡 ${id}`)
}

/** 关掉全局游戏事件浮层（升级弹窗、结算弹窗等），避免遮挡后续点击 */
async function dismissOverlays(page: Page) {
  for (let i = 0; i < 6; i++) {
    const dlg = page.locator('[role="dialog"], [aria-modal="true"]')
    if ((await dlg.count()) === 0 || !(await dlg.first().isVisible().catch(() => false))) return
    await page.keyboard.press('Escape')
    await page.waitForTimeout(350)
    if ((await dlg.count()) > 0 && (await dlg.first().isVisible().catch(() => false))) {
      const btn = dlg.first().locator('button', { hasText: /继续|关闭|知道了|留在本关/ })
      if (await btn.count()) await btn.first().click().catch(() => {})
      await page.waitForTimeout(350)
    }
  }
}

async function setEditor(page: Page, code: string) {
  await dismissOverlays(page)
  const ed = page.locator('[data-editor] .cm-content').first()
  await ed.waitFor({ timeout: 20000 })
  await ed.click()
  await page.keyboard.press('ControlOrMeta+a')
  await page.keyboard.press('Delete')
  await page.keyboard.insertText(code)
  const got = await page.evaluate(() =>
    Array.from(document.querySelectorAll('[data-editor] .cm-line'))
      .map((l) => l.textContent)
      .join('\n'),
  )
  if (got.replace(/\s+$/, '') !== code.replace(/\s+$/, '')) throw new Error('编辑器内容与预期不一致（可能被自动缩进/补全改写）')
}

/** 点击按钮并等待新的运行结果出现 */
async function runAndWait(page: Page, action: 'run' | 'submit', timeout = 60000) {
  await dismissOverlays(page)
  const prevAt = (await page.locator('[data-run-at]').first().getAttribute('data-run-at').catch(() => null)) ?? ''
  await page.locator(`[data-workspace] [data-action="${action}"]`).first().click()
  const clickedAt = Date.now()
  await page.waitForFunction(
    (prev) => {
      const el = document.querySelector('[data-run-at]')
      return !!el && el.getAttribute('data-run-at') !== prev
    },
    prevAt,
    { timeout },
  )
  const completedMs = Date.now() - clickedAt
  const el = page.locator('[data-run-kind]').first()
  return {
    kind: await el.getAttribute('data-run-kind'),
    ok: (await el.getAttribute('data-run-ok')) === 'true',
    timeout: (await page.locator('[data-timeout]').count()) > 0,
    summary: await page.locator('[data-summary]').count() ? (await page.locator('[data-summary]').first().textContent()) ?? '' : '',
    ms: completedMs,
  }
}

async function progress(page: Page) {
  return page.evaluate(() => JSON.parse(localStorage.getItem('cq-progress') || '{}')?.state)
}

const L = await findLevel(LEVEL)
await fs.mkdir('.shots/p2', { recursive: true })
console.log(`关卡 ${LEVEL}：${L.challenges.length} 道实战，${L.guided.length} 个跟练`)

const browser = await chromium.launch({ channel: 'chrome', headless: !HEADED, args: ['--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows'] })
const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } })
const page = await ctx.newPage()
const errors: string[] = []
page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
page.on('pageerror', (e) => errors.push(e.message))
const t0 = Date.now()

try {
  /* ---------------- 闭环：图解 → 随堂 → ★ → 结算 ---------------- */
  if (!SKIP_FLOW) {
    console.log('\n[闭环]')
    await page.goto(`${BASE}/#/level/${LEVEL}/learn`)
    await page.locator('[data-section]').first().waitFor({ timeout: 20000 })
    for (let i = 0; i < L.learn.length + 4; i++) {
      await dismissOverlays(page)
      const btn = page.locator('[data-action="next-section"], [data-action="finish-learn"]')
      if (!(await btn.count())) break
      await btn.first().click()
      await page.waitForTimeout(1100)
    }
    let st = await progress(page)
    check(!!st?.levels?.[LEVEL]?.learnDone, `图解：逐节「下一节」后 completeLearn（已读 ${st?.levels?.[LEVEL]?.sectionsRead?.length}/${L.learn.length}）`)

    await page.goto(`${BASE}/#/level/${LEVEL}/quiz`)
    let correct = 0
    for (let n = 0; n < L.quiz.length; n++) {
      await dismissOverlays(page)
      const qv = page.locator('[data-question]').first()
      await qv.waitFor({ timeout: 10000 })
      const qid = await qv.getAttribute('data-question')
      const q = L.quiz.find((x) => x.id === qid)!
      const tapSel = async (sel: string) => page.locator(`[data-question] ${sel}`).first().click()
      if (q.type === 'single') await tapSel(`[data-option="${q.answer}"]`)
      else if (q.type === 'multi') for (const i of q.answer) await tapSel(`[data-option="${i}"]`)
      else if (q.type === 'judge') await page.locator('[data-question] [role="radio"]').nth(q.answer ? 0 : 1).click()
      else if (q.type === 'numeric') await page.locator('[data-question] input').first().fill(String(q.answer))
      else if (q.type === 'fill') for (const [i, b] of q.blanks.entries()) await page.locator('[data-question] input').nth(i).fill(b.answers[0])
      else if (q.type === 'bug') for (const l of q.bugLines) await tapSel(`[data-line="${l}"]`)
      else if (q.type === 'pick') for (const i of q.answer) await tapSel(`[data-item="${i}"]`)
      else if (q.type === 'match') {
        const Lb = page.locator('[data-question] .grid > div:first-child > div > button')
        const Rb = page.locator('[data-question] .grid > div:nth-child(2) > button')
        for (let i = 0; i < (await Lb.count()); i++) {
          await Lb.nth(i).click()
          await Rb.nth(i).click()
        }
      } else if (q.type === 'parsons') await tapSel('button:has(svg.lucide-plus)')
      await page.locator('[data-question] [data-action="submit"]').click()
      if ((await qv.getAttribute('data-state')) === 'correct') correct++
      await page.locator('[data-question] [data-action="next"]').click()
    }
    await page.locator('[data-quiz-result]').waitFor({ timeout: 10000 })
    st = await progress(page)
    check((st?.levels?.[LEVEL]?.quizBest ?? 0) >= 0.6, `随堂：答对 ${correct}/${L.quiz.length}，completeQuiz 记录最佳 ${Math.round((st?.levels?.[LEVEL]?.quizBest ?? 0) * 100)}%`)
    const summary = await page.locator('[data-summary]').waitFor({ timeout: 8000 }).then(() => true, () => false)
    check(summary, '星级提升后自动弹出通关结算')
    const stars = await page.locator('[data-summary] .text-gold').count()
    check(stars >= 1, `结算显示 ★（点亮 ${stars} 颗）`)
    await page.screenshot({ path: '.shots/p2/e2e-summary.png' })
    await dismissOverlays(page)
  }

  /* ---------------- 实战 ---------------- */
  if (L.challenges.length) {
    console.log('\n[实战]')
    await page.goto(`${BASE}/#/level/${LEVEL}/challenge`)
    await page.locator('[data-py="ready"]').waitFor({ timeout: 90000 })
    ok(`Pyodide 就绪（${((Date.now() - t0) / 1000).toFixed(1)}s）`)

    for (const c of L.challenges) {
      await dismissOverlays(page)
      await page.locator(`[data-challenge-tab="${c.id}"]`).click()
      await page.locator(`[data-workspace="${c.id}"]`).waitFor()

      await setEditor(page, c.starter)
      const rs = await runAndWait(page, 'submit')
      check(!rs.ok, `「${c.title}」starter 提交 → 不通过`, rs.summary)

      await setEditor(page, c.solution)
      const r = await runAndWait(page, 'submit')
      check(r.ok && r.kind === 'submit', `「${c.title}」参考解提交 → 全部通过`, r.summary)
      check((await page.locator('[data-passed-banner]').count()) > 0, `「${c.title}」显示通过庆祝`)
      const st = await progress(page)
      check(!!st?.levels?.[LEVEL]?.challenges?.[c.id]?.passed, `「${c.title}」进度写入 passed`)
    }

    if (!SKIP_LOOP) {
      const c = L.challenges[0]
      await dismissOverlays(page)
      await page.locator(`[data-challenge-tab="${c.id}"]`).click()
      const fn = /def\s+(\w+)\s*\(([^)]*)\)/.exec(c.solution)
      const loop = `def ${fn?.[1] ?? 'f'}(${fn?.[2] ?? '*a'}):\n    while True:\n        pass`
      await setEditor(page, loop)
      const r = await runAndWait(page, 'run', 40000)
      check(r.timeout && r.ms < 20000, `死循环 → 超时提示（点击后 ${(r.ms / 1000).toFixed(1)}s）`)
      await setEditor(page, c.solution)
      const r2 = await runAndWait(page, 'submit', 90000)
      check(r2.ok, '超时后 Worker 自动重建，再次提交参考解通过', r2.summary)
    }
  }

  /* ---------------- 跟练 ---------------- */
  for (const g of L.guided) {
    console.log(`\n[跟练] ${g.title}`)
    await page.goto(`${BASE}/#/level/${LEVEL}/guided`)
    await page.reload()
    const tab = page.locator('button', { hasText: g.title })
    if ((await tab.count()) > 0 && L.guided.length > 1) await tab.first().click()
    for (const [k, s] of g.steps.entries()) {
      const input = page.locator('.cq-blank').first()
      await input.waitFor({ timeout: 10000 })
      // 样板关卡：第 3 空用一个未列入 accept 的等价写法，第 4 空先填一个错误写法
      let answer = s.answer
      if (LEVEL === 'algo-search-binary' && s.blank === 4) {
        await input.fill('nums[mid] <= target')
        await dismissOverlays(page)
      await page.locator('[data-action="check"]').click()
        await page.locator('.cq-blank[data-state="wrong"]').waitFor({ timeout: 60000 })
        ok('错误写法（nums[mid] <= target）被判错')
      }
      if (LEVEL === 'algo-search-binary' && s.blank === 3) answer = '(hi + lo) // 2'
      await page.locator('.cq-blank').first().fill(answer)
      await dismissOverlays(page)
      await page.locator('[data-action="check"]').click()
      await page.waitForFunction((n) => document.querySelectorAll('.cq-blank-done').length >= n || !!document.querySelector('[data-guided-done]'), k + 1, {
        timeout: 60000,
      })
      if (answer !== s.answer) {
        const kind = await page.locator('.cq-blank-done').nth(k).getAttribute('data-kind')
        check(kind === 'equivalent', `等价写法「${answer}」经跑测试判为等价`, `kind=${kind}`)
      }
    }
    await page.locator('[data-guided-done]').waitFor({ timeout: 60000 })
    const st = await progress(page)
    check(!!st?.levels?.[LEVEL]?.guidedDone?.includes(g.id), `「${g.title}」完整测试通过并 completeGuided`)
  }
} catch (e) {
  bad(`异常：${(e as Error).message.split('\n')[0]}`)
  await page.screenshot({ path: '.shots/p2/e2e-failure.png' }).catch(() => {})
}

const errs = errors.filter((e) => !/Download the React DevTools/.test(e))
check(errs.length === 0, 'console 无 error', errs.slice(0, 5).join(' | '))
await browser.close()
console.log(`\n${failures ? `\x1b[31m${failures} 项失败\x1b[0m` : '\x1b[32m全部通过\x1b[0m'}（${((Date.now() - t0) / 1000).toFixed(1)}s）`)
process.exit(failures ? 1 : 0)
