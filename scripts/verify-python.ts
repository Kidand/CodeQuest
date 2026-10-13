/**
 * Python 正确性校验：npm run verify:py [-- 章节id ...] [--level 关卡id]
 *  - 每道实战：参考解 solution 必须通过全部测试；starter 不应通过
 *  - 每个跟练：用参考答案拼装后的代码必须通过测试；空白模板不应通过
 * 与浏览器使用同一个 harness 与同一版本的 Pyodide（Python 3.14 + numpy）。
 */
import { loadPyodide } from 'pyodide'
import { HARNESS_PY, assembleTemplate, type RunResult } from '../src/runner/harness'
import { c, loadChapters, loadLevelContent, parseArgs } from './lib/content'

const { chapters: filter, level: onlyLevel } = parseArgs()
const py = await loadPyodide()
await py.loadPackage('numpy', { messageCallback: () => {} })
py.runPython(HARNESS_PY)

function run(code: string, setup: string | undefined, tests: { name: string; code: string; hidden?: boolean }[]): RunResult {
  py.globals.set('__cq_user', code)
  py.globals.set('__cq_setup', setup ?? '')
  py.globals.set('__cq_tests', JSON.stringify(tests))
  return JSON.parse(py.runPython('json.dumps(_cq_run(__cq_user, __cq_setup, __cq_tests), ensure_ascii=False)'))
}

function describeFail(r: RunResult) {
  if (r.compileError) return `代码无法执行：${r.compileError}`
  return r.results
    .filter((t) => !t.passed)
    .map((t) => `「${t.name}」${(t.error ?? '').split('\n').slice(-3).join(' ⏎ ')}`)
    .join('\n      ')
}

let fails = 0
let warns = 0
let checked = 0
const t0 = Date.now()
for (const lc of await loadChapters(filter)) {
  if (!lc.chapter) {
    console.log(c.red(`✗ 章节 ${lc.dirId} 加载失败`), lc.error?.split('\n')[0])
    fails++
    continue
  }
  for (const meta of lc.chapter.levels) {
    if (onlyLevel && meta.id !== onlyLevel) continue
    let L
    try {
      L = await loadLevelContent(meta)
    } catch (e) {
      console.log(c.red(`✗ [${meta.id}] 加载失败`), String(e).split('\n')[0])
      fails++
      continue
    }
    for (const ch of L.challenges) {
      const where = `[${meta.id}] 实战「${ch.title}」`
      checked++
      const s0 = Date.now()
      const r = run(ch.solution, ch.setup, ch.tests)
      const ms = Date.now() - s0
      if (!r.ok) {
        fails++
        console.log(`${c.red('✗')} ${where} 参考解未通过：\n      ${describeFail(r)}`)
      } else {
        console.log(`${c.green('✓')} ${where} ${c.dim(`${ch.tests.length} 个测试 · ${ms}ms`)}`)
        if (ms > 4000) (warns++, console.log(`  ${c.yellow('!')} 测试总耗时 ${ms}ms，浏览器中可能偏慢，请缩小规模`))
      }
      const rs = run(ch.starter, ch.setup, ch.tests)
      if (rs.ok) (warns++, console.log(`  ${c.yellow('!')} ${where} starter 竟然通过了全部测试`))
      else if (rs.compileError && /SyntaxError/.test(rs.compileError))
        (warns++, console.log(`  ${c.yellow('!')} ${where} starter 有语法错误：${rs.compileError}`))
    }
    for (const g of L.guided) {
      const where = `[${meta.id}] 跟练「${g.title}」`
      checked++
      const fills = Object.fromEntries(g.steps.map((s) => [s.blank, s.answer]))
      const code = assembleTemplate(g.template, fills)
      const r = run(code, g.setup, g.tests)
      if (!r.ok) {
        fails++
        console.log(`${c.red('✗')} ${where} 参考拼装未通过：\n      ${describeFail(r)}\n${c.dim(code.split('\n').map((l, i) => `      ${String(i + 1).padStart(3)}| ${l}`).join('\n'))}`)
      } else console.log(`${c.green('✓')} ${where} ${c.dim(`${g.tests.length} 个测试`)}`)
      // 每个 accept 变体单独替换后也应通过
      for (const s of g.steps) {
        for (const alt of s.accept ?? []) {
          const rr = run(assembleTemplate(g.template, { ...fills, [s.blank]: alt }), g.setup, g.tests)
          if (!rr.ok) (warns++, console.log(`  ${c.yellow('!')} ${where} step ${s.blank} 的 accept 变体「${alt}」跑测试不通过`))
        }
      }
      const empty = assembleTemplate(g.template, Object.fromEntries(g.steps.map((s) => [s.blank, 'pass'])))
      const re = run(empty, g.setup, g.tests)
      if (re.ok) (warns++, console.log(`  ${c.yellow('!')} ${where} 空白模板（全填 pass）也通过了测试，测试太弱`))
    }
  }
}
console.log(
  `\n共校验 ${checked} 项，用时 ${((Date.now() - t0) / 1000).toFixed(1)}s：` +
    (fails ? c.red(`${fails} 项失败`) : c.green('全部通过')) +
    (warns ? c.yellow(`，${warns} 个警告`) : ''),
)
process.exit(fails ? 1 : 0)
