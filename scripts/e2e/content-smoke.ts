#!/usr/bin/env tsx
/**
 * 全站课程渲染冒烟：正式 learn 页，桌面浅色 + 手机暗色。
 * 复用已启动的本地 Vite 服务和项目 Playwright，不安装任何依赖。
 *
 * node --import tsx scripts/e2e/content-smoke.ts --base http://127.0.0.1:5173
 * 可选：--level <id>（逗号分隔） --out output/playwright/content-smoke
 *
 * 验证所有小节与可视化均已渲染；每个逐帧图点击两次下一步；收集
 * pageerror、console.error、错误边界和页面水平溢出，失败保存截图及元素定位。
 * 每种设备使用独立临时浏览器上下文，不读写用户真实学习进度。
 */
import fs from 'node:fs/promises'
import path from 'node:path'
import { chromium, type Page } from 'playwright-core'
import { loadChapters, loadLevelContent } from '../lib/content'

const args = process.argv.slice(2)
const option = (name: string, fallback = '') => {
  const i = args.indexOf(name)
  return i < 0 ? fallback : args[i + 1]
}
const base = option('--base', 'http://127.0.0.1:5173').replace(/\/$/, '')
const output = path.resolve(option('--out', 'output/playwright/content-smoke'))
const selected = new Set(option('--level').split(',').filter(Boolean))
await fs.mkdir(output, { recursive: true })

const chapters = await loadChapters()
const failedChapters = chapters.filter((ch) => !ch.chapter)
if (failedChapters.length) throw new Error(`章节加载失败：${failedChapters.map((ch) => `${ch.dirId}: ${ch.error}`).join('\n')}`)
const levels = chapters.flatMap((ch) => ch.chapter!.levels).filter((m) => !selected.size || selected.has(m.id))
if (!levels.length) throw new Error('没有找到待测关卡')
const contents = new Map(await Promise.all(levels.map(async (meta) => [meta.id, await loadLevelContent(meta)] as const)))

interface CheckResult {
  level: string
  profile: string
  sections: number
  visualizations: number
  errors: string[]
  overflow: Awaited<ReturnType<typeof findOverflow>> | null
  screenshot?: string
}

/** 忽略被内部滚动/裁剪容器约束的代码和公式；只报告实际撑宽页面的元素。 */
async function findOverflow(page: Page) {
  return page.evaluate(() => {
    const viewport = window.innerWidth
    const pageWidth = Math.max(document.documentElement.scrollWidth, document.body.scrollWidth)
    if (pageWidth <= viewport + 2) return { viewport, pageWidth, elements: [] }
    const describe = (el: Element) => {
      const rect = el.getBoundingClientRect()
      return {
        tag: el.tagName.toLowerCase(),
        class: el.getAttribute('class') ?? '',
        section: el.closest('[data-section]')?.getAttribute('data-section') ?? null,
        viz: el.closest('[data-viz]')?.getAttribute('data-viz') ?? null,
        text: (el.textContent ?? '').trim().replace(/\s+/g, ' ').slice(0, 140),
        left: Math.round(rect.left), right: Math.round(rect.right), width: Math.round(rect.width),
        scrollWidth: el.scrollWidth, clientWidth: el.clientWidth,
      }
    }
    const elements = [...document.body.querySelectorAll('*')].filter((el) => {
      if (!el.getClientRects().length || el.tagName === 'SCRIPT' || el.tagName === 'STYLE') return false
      const rect = el.getBoundingClientRect()
      if (rect.width <= 0 || (rect.right <= viewport + 2 && rect.left >= -2)) return false
      // SVG 子路径有自己的几何坐标；外层 SVG 的盒子足以说明页面溢出。
      if (el instanceof SVGElement && el.tagName.toLowerCase() !== 'svg') return false
      let parent = el.parentElement
      while (parent && parent !== document.body) {
        const style = getComputedStyle(parent)
        const box = parent.getBoundingClientRect()
        if (/^(auto|scroll|hidden|clip)$/.test(style.overflowX) && box.left >= -2 && box.right <= viewport + 2) return false
        parent = parent.parentElement
      }
      return true
    }).map(describe)
    return { viewport, pageWidth, elements: elements.slice(-30) }
  })
}

const browser = await chromium.launch({ channel: 'chrome', headless: true })
const results: CheckResult[] = []
const started = Date.now()
try {
  for (const profile of [
    { name: 'desktop-light', width: 1440, height: 1000, dark: false, mobile: false },
    { name: 'mobile-dark', width: 390, height: 844, dark: true, mobile: true },
  ]) {
    const context = await browser.newContext({
      viewport: { width: profile.width, height: profile.height },
      colorScheme: profile.dark ? 'dark' : 'light',
      isMobile: profile.mobile, hasTouch: profile.mobile,
      reducedMotion: 'reduce',
    })
    await context.addInitScript((dark) => {
      localStorage.setItem('cq-theme', dark ? 'dark' : 'light')
      localStorage.setItem('cq-progress', JSON.stringify({ version: 1, state: {
        settings: { theme: dark ? 'dark' : 'light', freeMode: true, reduceMotion: true, sound: false, dailyGoal: 100, editorFontSize: 14 },
      } }))
    }, profile.dark)
    const page = await context.newPage()
    let current: CheckResult | null = null
    page.on('pageerror', (error) => current?.errors.push(`[pageerror] ${error.message}`))
    page.on('console', (message) => {
      if (message.type() === 'error') current?.errors.push(`[console.error] ${message.text()}`)
    })

    for (const meta of levels) {
      const content = contents.get(meta.id)!
      current = { level: meta.id, profile: profile.name, sections: 0, visualizations: 0, errors: [], overflow: null }
      try {
        await page.goto(`${base}/?content-smoke=${encodeURIComponent(meta.id)}#/level/${meta.id}/learn`, { waitUntil: 'domcontentloaded', timeout: 20000 })
        await page.waitForFunction(({ id, title, sections, vizCount }) => {
          const root = document.querySelector(`[data-level="${id}"]`)
          return document.title === `${title} · CodeQuest` && !!root
            && JSON.stringify([...root.querySelectorAll('[data-section]')].map((el) => el.getAttribute('data-section'))) === JSON.stringify(sections)
            && root.querySelectorAll('[data-viz]').length === vizCount
        }, { id: meta.id, title: meta.title, sections: content.learn.map((s) => s.id), vizCount: content.learn.filter((s) => s.viz).length }, { timeout: 15000 })
        await page.evaluate(() => document.fonts.ready)
        await page.waitForTimeout(180)
        await page.locator(`[data-level="${meta.id}"] [data-section]`).first().waitFor({ timeout: 15000 })
        current.sections = await page.locator('[data-section]').count()
        current.visualizations = await page.locator('[data-viz]').count()
        const expectedViz = content.learn.filter((s) => s.viz).length
        if (current.visualizations !== expectedViz) current.errors.push(`可视化数量：期望 ${expectedViz}，实际 ${current.visualizations}`)
        const errorBoundaries = page.getByText(/渲染异常|渲染出错|组件出错|出了点问题|关卡加载失败/)
        if (await errorBoundaries.count()) current.errors.push(`错误边界：${await errorBoundaries.allTextContents()}`)

        for (const viz of await page.locator('[data-viz]').all()) {
          await viz.scrollIntoViewIfNeeded()
          const next = viz.locator('button[aria-label="下一步"]').first()
          for (let step = 0; step < 2 && await next.count() && await next.isEnabled(); step++) {
            await next.click({ timeout: 3000 })
            await page.waitForTimeout(30)
          }
        }
        // MXFP4 的 6 → 6.25 跨过尺度跳变边界，覆盖参数型交互。
        if (meta.id === 'llm-quant-lowprec') {
          const maximum = page.getByLabel('块中最大幅度', { exact: true })
          await maximum.focus()
          await maximum.press('ArrowRight')
          await page.waitForTimeout(80)
          if (await maximum.inputValue() !== '6.25') current.errors.push('MXFP4 幅度滑块未更新到 6.25')
          if (!(await page.locator('[data-viz="mxfp4"]').innerText()).includes('s=2')) current.errors.push('MXFP4 跨界后尺度未更新到 2')
        }
        current.overflow = await findOverflow(page)
        if (current.overflow.pageWidth > current.overflow.viewport + 2) {
          current.errors.push(`页面水平溢出：视口 ${current.overflow.viewport}px，页面 ${current.overflow.pageWidth}px`)
        }
      } catch (error) {
        current.errors.push(`[test] ${error instanceof Error ? error.message : String(error)}`)
      }
      current.errors = [...new Set(current.errors)]
      if (current.errors.length) {
        current.screenshot = path.join(output, `${profile.name}-${meta.id}.png`)
        await page.screenshot({ path: current.screenshot }).catch(() => {})
        console.log(`FAIL ${profile.name} ${meta.id}: ${current.errors.join(' | ')}`)
        if (current.overflow?.elements.length) console.log(JSON.stringify(current.overflow.elements))
      } else {
        console.log(`PASS ${profile.name} ${meta.id}: ${current.sections} 节 / ${current.visualizations} 图`)
      }
      results.push(current)
      current = null
    }
    await context.close()
  }
} finally {
  await browser.close()
}
const report = {
  base, levels: levels.length, pages: results.length,
  passed: results.filter((r) => !r.errors.length).length,
  failed: results.filter((r) => r.errors.length).length,
  durationSeconds: Math.round((Date.now() - started) / 1000),
  results,
}
await fs.writeFile(path.join(output, 'report.json'), JSON.stringify(report, null, 2) + '\n')
console.log(`\n${report.levels} 关 / ${report.pages} 个页面：${report.passed} 通过，${report.failed} 失败，耗时 ${report.durationSeconds}s`)
console.log(`报告：${path.join(output, 'report.json')}`)
process.exitCode = report.failed ? 1 : 0
