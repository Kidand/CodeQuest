#!/usr/bin/env node
/**
 * 随堂练习走查：逐题按题型做一次作答 → 截图（作答中 / 已提交）→ 下一题 → 结算页。
 *   node scripts/e2e/quiz-walk.mjs [--base http://localhost:5202] [--level algo-search-binary] [--mobile] [--dark] [--out .shots/p2/quiz]
 * 输出每题题型与判定结果，并收集 console error（有则退出码 1）。
 */
import { chromium } from 'playwright-core'
import fs from 'node:fs'

const argv = process.argv.slice(2)
const arg = (k, d) => (argv.includes(k) ? argv[argv.indexOf(k) + 1] : d)
const BASE = arg('--base', 'http://localhost:5202')
const LEVEL = arg('--level', 'algo-search-binary')
const mobile = argv.includes('--mobile')
const dark = argv.includes('--dark')
const OUT = arg('--out', `.shots/p2/quiz${mobile ? '-m' : ''}${dark ? '-dark' : ''}`)
fs.mkdirSync(OUT, { recursive: true })

const browser = await chromium.launch({ channel: 'chrome' })
const ctx = await browser.newContext({
  viewport: mobile ? { width: 390, height: 844 } : { width: 1280, height: 900 },
  deviceScaleFactor: mobile ? 2 : 1,
  isMobile: mobile,
  hasTouch: mobile,
  colorScheme: dark ? 'dark' : 'light',
})
await ctx.addInitScript((d) => localStorage.setItem('cq-theme', d ? 'dark' : 'light'), dark)
const page = await ctx.newPage()
const errors = []
page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
page.on('pageerror', (e) => errors.push(e.message))

await page.goto(`${BASE}/#/level/${LEVEL}/quiz`)
await page.locator('[data-question]').first().waitFor({ timeout: 20000 })

const tap = async (sel, i = 0) => page.locator(sel).nth(i).click({ timeout: 3000 }).catch((e) => console.log('   tap 失败', sel, e.message.split('\n')[0]))

for (let n = 0; n < 40; n++) {
  const qv = page.locator('[data-question]').first()
  if (!(await qv.count())) break
  const type = await qv.getAttribute('data-qtype')
  const id = await qv.getAttribute('data-question')
  switch (type) {
    case 'single':
      await tap('[data-question] [data-option]', 1)
      break
    case 'multi':
      await tap('[data-question] [data-option]', 0)
      await tap('[data-question] [data-option]', 1)
      break
    case 'judge':
      await tap('[data-question] [role="radio"]', 1)
      break
    case 'fill': {
      const inputs = page.locator('[data-question] input')
      for (let i = 0; i < (await inputs.count()); i++) await inputs.nth(i).fill(String(i * 3 + 1))
      break
    }
    case 'numeric':
      await page.locator('[data-question] input').first().fill('2^20/52429')
      break
    case 'order':
      await tap('[data-question] button[aria-label="下移此项"]', 0)
      break
    case 'match': {
      const L = page.locator('[data-question] .grid > div:first-child > div > button')
      const R = page.locator('[data-question] .grid > div:nth-child(2) > button')
      const k = await L.count()
      for (let i = 0; i < k; i++) {
        await L.nth(i).click()
        await R.nth(i).click()
      }
      break
    }
    case 'parsons': {
      // 依次点候选区第一行，直到放入与正确行数相同的行
      for (let i = 0; i < 9; i++) await tap('[data-question] button:has(svg.lucide-plus)', 0)
      break
    }
    case 'bug':
      await tap('[data-question] [data-line="8"]')
      break
    case 'pick':
      await tap('[data-question] [data-item="4"]')
      await tap('[data-question] [data-item="6"]')
      break
  }
  await page.waitForTimeout(250)
  await qv.screenshot({ path: `${OUT}/${String(n + 1).padStart(2, '0')}-${type}-a.png` })
  await page.locator('[data-question] [data-action="submit"]').click()
  await page.waitForTimeout(600)
  const state = await qv.getAttribute('data-state')
  console.log(`#${n + 1} ${type.padEnd(8)} ${id.padEnd(14)} → ${state}`)
  await qv.screenshot({ path: `${OUT}/${String(n + 1).padStart(2, '0')}-${type}-b.png` })
  await page.locator('[data-question] [data-action="next"]').click()
  await page.waitForTimeout(400)
  if (await page.locator('[data-quiz-result]').count()) break
}
await page.locator('[data-quiz-result]').waitFor({ timeout: 5000 }).catch(() => console.log('未到达结算页'))
await page.waitForTimeout(1200)
await page.screenshot({ path: `${OUT}/result.png`, fullPage: false })
const st = await page.evaluate(() => JSON.parse(localStorage.getItem('cq-progress') || '{}')?.state?.levels)
console.log('quiz 进度：', JSON.stringify({ best: st?.[LEVEL]?.quizBest, attempts: st?.[LEVEL]?.quizAttempts }))
console.log(errors.length ? `console error ×${errors.length}:\n${errors.slice(0, 5).join('\n')}` : 'console 无 error')
await browser.close()
process.exit(errors.length ? 1 : 0)
