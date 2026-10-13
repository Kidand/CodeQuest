#!/usr/bin/env node
/**
 * 无头截图（使用本机 Chrome，无需下载浏览器）。供开发/内容作者自检可视化效果。
 *
 *   node scripts/shot.mjs --url "http://localhost:5173/#/dev/level/algo-search-binary" --out .shots/a.png
 *   选项：--mobile  390×844 触屏      --dark  暗色主题       --full  整页
 *         --width 1280 --height 900  --wait 1200(ms)        --selector "figure" 只截该元素(第一个)
 *         --click "button[aria-label=下一步]"（可重复，按顺序点击）
 *         --eval "js 表达式"（截图前在页面执行）
 * 同时打印页面 console 中的 error / warning 以及未捕获异常。
 */
import { chromium } from 'playwright-core'
import fs from 'node:fs'
import path from 'node:path'

const args = process.argv.slice(2)
const opt = { clicks: [], evals: [] }
for (let i = 0; i < args.length; i++) {
  const a = args[i]
  const v = () => args[++i]
  if (a === '--url') opt.url = v()
  else if (a === '--out') opt.out = v()
  else if (a === '--mobile') opt.mobile = true
  else if (a === '--dark') opt.dark = true
  else if (a === '--full') opt.full = true
  else if (a === '--width') opt.width = Number(v())
  else if (a === '--height') opt.height = Number(v())
  else if (a === '--wait') opt.wait = Number(v())
  else if (a === '--selector') opt.selector = v()
  else if (a === '--click') opt.clicks.push(v())
  else if (a === '--eval') opt.evals.push(v())
}
if (!opt.url) {
  console.error('缺少 --url')
  process.exit(1)
}
opt.out ??= `.shots/shot-${Date.now()}.png`
fs.mkdirSync(path.dirname(opt.out), { recursive: true })

// 整体超时保护（偶发 networkidle / 整页截图卡死）
const killer = setTimeout(() => {
  console.error('截图超时（90s），已放弃')
  process.exit(2)
}, 90_000)
killer.unref?.()
const browser = await chromium.launch({ channel: 'chrome', headless: true })
const ctx = await browser.newContext({
  viewport: opt.mobile ? { width: 390, height: 844 } : { width: opt.width ?? 1280, height: opt.height ?? 900 },
  deviceScaleFactor: opt.mobile ? 2 : 1,
  isMobile: !!opt.mobile,
  hasTouch: !!opt.mobile,
  colorScheme: opt.dark ? 'dark' : 'light',
})
const page = await ctx.newPage()
const logs = []
page.on('console', (m) => {
  if (m.type() === 'error' || m.type() === 'warning') logs.push(`[console.${m.type()}] ${m.text()}`)
})
page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`))
await page.addInitScript((dark) => {
  try {
    localStorage.setItem('cq-theme', dark ? 'dark' : 'light')
    const raw = localStorage.getItem('cq-progress')
    const st = raw ? JSON.parse(raw) : { state: {}, version: 1 }
    st.state.settings = { ...(st.state.settings || {}), theme: dark ? 'dark' : 'light' }
    localStorage.setItem('cq-progress', JSON.stringify(st))
  } catch {}
}, !!opt.dark)
await page.goto(opt.url, { waitUntil: 'networkidle' }).catch((e) => logs.push(`[goto] ${e.message}`))
await page.waitForTimeout(opt.wait ?? 800)
for (const sel of opt.clicks) {
  try {
    await page.locator(sel).first().click({ timeout: 3000 })
    await page.waitForTimeout(350)
  } catch (e) {
    logs.push(`[click] ${sel}: ${e.message.split('\n')[0]}`)
  }
}
for (const js of opt.evals) {
  try {
    await page.evaluate(js)
    await page.waitForTimeout(300)
  } catch (e) {
    logs.push(`[eval] ${e.message.split('\n')[0]}`)
  }
}
if (opt.selector) {
  await page.locator(opt.selector).first().screenshot({ path: opt.out }).catch((e) => logs.push(`[selector] ${e.message.split('\n')[0]}`))
} else {
  await page.screenshot({ path: opt.out, fullPage: !!opt.full })
}
await browser.close()
console.log(`截图：${opt.out}`)
if (logs.length) console.log(logs.join('\n'))
