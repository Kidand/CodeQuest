/**
 * 内容结构校验：npm run validate [-- 章节id ...] [--level 关卡id]
 *  - Schema 完整性（id 唯一、答案下标合法、题量与题型下限…）
 *  - 反斜杠事故检测（忘记用 md`` / py`` 导致 \f \b \t 被转义）
 *  - KaTeX 公式可解析
 */
import katex from 'katex'
import type { Question, LevelContent, ChapterDef } from '../src/content/types'
import { templateBlanks } from '../src/runner/harness'
import { c, loadChapters, loadLevelContent, parseArgs } from './lib/content'

const { chapters: filter, level: onlyLevel } = parseArgs()
const errors: string[] = []
const warns: string[] = []
const err = (where: string, msg: string) => errors.push(`${c.red('✗')} ${where}: ${msg}`)
const warn = (where: string, msg: string) => warns.push(`${c.yellow('!')} ${where}: ${msg}`)

/** 检查 markdown 字段：控制字符 + KaTeX */
function checkMd(where: string, s: string | undefined) {
  if (s == null) return
  if (typeof s !== 'string') return err(where, '不是字符串')
  if (/[\b\f\v]/.test(s)) err(where, '含控制字符（\\b \\f \\v）——LaTeX 反斜杠被转义了，请用 md`` 标签')
  if (/\t(imes|ext|heta|au|op|riangle|ilde|o\b)/.test(s)) err(where, '含 TAB+字母（疑似 \\times/\\text/\\theta 被转义），请用 md`` 标签')
  // 只在公式内部检测（换行后紧跟 abla/eq/u 等，说明 \nabla \neq \nu 被转义成了换行）
  const mathSegs = [...s.matchAll(/\$\$([\s\S]+?)\$\$|\$([^$\n]+?)\$/g)].map((m) => m[1] ?? m[2])
  if (mathSegs.some((seg) => /\n(abla|eq|ot|ewline|orm)\b/.test(seg))) warn(where, '疑似 \\nabla/\\neq 等被转义成换行，请确认使用了 md``')
  // GFM 表格行中，行内代码 / 公式里未转义的 | 会被当成列分隔符
  for (const line of s.split('\n')) {
    if (!/^\s*\|/.test(line)) continue
    const spans = [...line.matchAll(/`[^`]*`|\$[^$]*\$/g)].map((m) => m[0])
    if (spans.some((sp) => /(?<!\\)\|/.test(sp)))
      err(where, `表格行中的代码/公式含未转义的 |（会被当作列分隔符）：用 \\| 或 \\lvert \\rvert \\mid ⟵ ${line.trim().slice(0, 60)}`)
  }
  // CJK 加粗失效：按 CommonMark 左/右侧规则配对检查 ** 定界符
  const bad = boldFailure(s)
  if (bad) warn(where, `CJK 加粗失效（页面会显示星号，把 ** 移到标点外侧或加空格）⟵ ${bad}`)
  // 去掉代码后提取公式
  const stripped = s.replace(/```[\s\S]*?```/g, '').replace(/`[^`\n]*`/g, '')
  const blocks = [...stripped.matchAll(/\$\$([\s\S]+?)\$\$/g)].map((m) => [m[1], true] as const)
  const rest = stripped.replace(/\$\$[\s\S]+?\$\$/g, '')
  const inlines = [...rest.matchAll(/(?<![\\$])\$(?!\s)([^$\n]+?)(?<!\s)\$(?!\d)/g)].map((m) => [m[1], false] as const)
  for (const [tex, display] of [...blocks, ...inlines]) {
    try {
      katex.renderToString(tex, { displayMode: display, throwOnError: true, strict: false })
    } catch (e) {
      err(where, `KaTeX 解析失败：${String((e as Error).message).split('\n')[0]}  ⟵ ${tex.slice(0, 80)}`)
    }
  }
}


/** 返回第一处不能生效的 ** 片段（按 CommonMark flanking 规则，成对检查），否则 null */
function boldFailure(src: string): string | null {
  const isWs = (ch: string) => ch === '' || /\s/u.test(ch)
  const isPunct = (ch: string) => ch !== '' && /[\p{P}\p{S}]/u.test(ch)
  for (const para of src.replace(/```[\s\S]*?```/g, '\n\n').split(/\n\s*\n/)) {
    const text = para.replace(/`[^`\n]*`/g, (m) => 'x'.repeat(m.length)).replace(/\$[^$\n]*\$/g, (m) => 'x'.repeat(m.length))
    const pos = [...text.matchAll(/(?<!\*)\*\*(?!\*)/g)].map((m) => m.index!)
    for (let k = 0; k + 1 < pos.length; k += 2) {
      const o = pos[k], c = pos[k + 1]
      const oa = text[o - 1] ?? '', ob = text[o + 2] ?? ''
      const ca = text[c - 1] ?? '', cb = text[c + 2] ?? ''
      const leftOk = !isWs(ob) && (!isPunct(ob) || isWs(oa) || isPunct(oa))
      const rightOk = !isWs(ca) && (!isPunct(ca) || isWs(cb) || isPunct(cb))
      if (!leftOk || !rightOk) return text.slice(Math.max(0, o - 2), Math.min(text.length, c + 4)).replace(/\n/g, ' ')
    }
  }
  return null
}

function checkCode(where: string, s: string | undefined) {
  if (s == null) return
  if (/[\b\f\v]/.test(s)) err(where, '代码含控制字符——请用 py`` 标签')
  if (/\t/.test(s)) warn(where, '代码含 TAB，Python 请统一用 4 空格缩进')
}

function checkQuestion(where: string, q: Question) {
  checkMd(`${where}.prompt`, q.prompt)
  checkMd(`${where}.explanation`, q.explanation)
  checkMd(`${where}.hint`, q.hint)
  if (!q.explanation || q.explanation.trim().length < 8) err(where, '缺少解析 explanation（需解释为什么对/错）')
  switch (q.type) {
    case 'single':
      if (q.options.length < 2) err(where, '选项少于 2 个')
      if (!(q.answer >= 0 && q.answer < q.options.length)) err(where, `answer 越界: ${q.answer}`)
      if (new Set(q.options).size !== q.options.length) err(where, '选项重复')
      q.options.forEach((o, i) => checkMd(`${where}.options[${i}]`, o))
      break
    case 'multi':
      if (q.options.length < 3) err(where, '多选选项少于 3 个')
      if (q.answer.length < 2) warn(where, '多选正确项少于 2 个（考虑改为单选）')
      if (q.answer.some((a) => !(a >= 0 && a < q.options.length))) err(where, 'answer 越界')
      if (new Set(q.answer).size !== q.answer.length) err(where, 'answer 重复')
      q.options.forEach((o, i) => checkMd(`${where}.options[${i}]`, o))
      break
    case 'judge':
      if (typeof q.answer !== 'boolean') err(where, 'answer 必须是 boolean')
      break
    case 'fill':
      if (!q.blanks?.length) err(where, 'blanks 为空')
      q.blanks?.forEach((b, i) => {
        if (!b.answers?.length) err(where, `blank[${i}] 没有答案`)
        if (b.numeric && !Number.isFinite(Number(b.answers[0]))) err(where, `blank[${i}] numeric 但答案不是数字`)
      })
      break
    case 'numeric':
      if (!Number.isFinite(q.answer)) err(where, 'answer 不是有限数')
      break
    case 'order':
      if (q.items.length < 3) err(where, '排序项少于 3')
      if (new Set(q.items).size !== q.items.length) err(where, '排序项重复')
      q.items.forEach((o, i) => checkMd(`${where}.items[${i}]`, o))
      break
    case 'match':
      if (q.pairs.length < 3) err(where, '配对少于 3 组')
      if (new Set(q.pairs.map((p) => p[0])).size !== q.pairs.length) err(where, '左列重复')
      if (new Set(q.pairs.map((p) => p[1])).size !== q.pairs.length) err(where, '右列重复')
      q.pairs.forEach((p, i) => (checkMd(`${where}.pairs[${i}][0]`, p[0]), checkMd(`${where}.pairs[${i}][1]`, p[1])))
      break
    case 'parsons':
      if (q.lines.length < 3) err(where, '代码行少于 3')
      if (new Set(q.lines).size !== q.lines.length) err(where, '代码行有完全相同的重复行（会产生多解），请改写')
      q.lines.forEach((l, i) => checkCode(`${where}.lines[${i}]`, l))
      break
    case 'bug': {
      const n = q.code.split('\n').length
      checkCode(`${where}.code`, q.code)
      if (!q.bugLines.length) err(where, 'bugLines 为空')
      if (q.bugLines.some((l) => l < 1 || l > n)) err(where, `bugLines 越界（代码共 ${n} 行）`)
      break
    }
    case 'pick':
      if (q.items.length < 2) err(where, 'items 少于 2')
      if (!q.answer.length) err(where, 'answer 为空')
      if (q.answer.some((a) => !(a >= 0 && a < q.items.length))) err(where, 'answer 越界')
      if (q.cols && q.items.length % q.cols !== 0 && q.rowLabels) err(where, '网格模式 items 数不是 cols 的整数倍')
      if (q.rowLabels && q.cols && q.rowLabels.length !== q.items.length / q.cols) err(where, 'rowLabels 数量与行数不符')
      if (q.colLabels && q.cols && q.colLabels.length !== q.cols) err(where, 'colLabels 数量与列数不符')
      break
    default:
      err(where, `未知题型 ${(q as Question).type}`)
  }
}

function checkLevel(ch: ChapterDef, levelId: string, boss: boolean, L: LevelContent) {
  const w = `[${levelId}]`
  // 图解
  if (!L.learn?.length) err(w, 'learn 为空')
  if (!boss) {
    if (L.learn.length < 3) warn(w, `learn 仅 ${L.learn.length} 节（建议 ≥ 3）`)
    if (!L.learn.some((s) => s.viz)) err(w, '没有任何可交互可视化（learn[].viz）')
  }
  const secIds = new Set<string>()
  L.learn.forEach((s, i) => {
    const ww = `${w}.learn[${s.id ?? i}]`
    if (secIds.has(s.id)) err(ww, 'section id 重复')
    secIds.add(s.id)
    checkMd(`${ww}.body`, s.body)
    checkMd(`${ww}.after`, s.after)
    if (s.viz && typeof s.viz !== 'function' && typeof s.viz !== 'object') err(ww, 'viz 不是组件')
    if (s.checkpoint) checkQuestion(`${ww}.checkpoint`, s.checkpoint)
  })
  // 随堂练习
  const qmin = boss ? 10 : 8
  if (L.quiz.length < qmin) err(w, `quiz 仅 ${L.quiz.length} 题（要求 ≥ ${qmin}）`)
  const types = new Set(L.quiz.map((q) => q.type))
  if (types.size < 5) err(w, `quiz 题型仅 ${types.size} 种（要求 ≥ 5）：${[...types].join(',')}`)
  const qids = new Set<string>()
  L.quiz.forEach((q, i) => {
    if (qids.has(q.id)) err(`${w}.quiz[${i}]`, `题目 id 重复: ${q.id}`)
    qids.add(q.id)
    checkQuestion(`${w}.quiz[${q.id ?? i}]`, q)
  })
  // 跟练
  if (!boss && L.guided.length < 1) err(w, 'guided 至少 1 个')
  L.guided.forEach((g) => {
    const ww = `${w}.guided[${g.id}]`
    checkMd(`${ww}.intro`, g.intro)
    checkMd(`${ww}.outro`, g.outro)
    checkCode(`${ww}.template`, g.template)
    const blanks = templateBlanks(g.template)
    const stepBlanks = g.steps.map((s) => s.blank)
    if (new Set(blanks).size !== blanks.length) err(ww, `模板空位编号重复: ${blanks}`)
    if (JSON.stringify([...blanks].sort((a, b) => a - b)) !== JSON.stringify([...stepBlanks].sort((a, b) => a - b)))
      err(ww, `模板空位 ${blanks} 与 steps ${stepBlanks} 不一致`)
    if (g.steps.length < 2) warn(ww, '跟练步骤少于 2 步')
    g.steps.forEach((s) => {
      checkMd(`${ww}.step${s.blank}.instruction`, s.instruction)
      checkMd(`${ww}.step${s.blank}.explain`, s.explain)
      checkCode(`${ww}.step${s.blank}.answer`, s.answer)
      if (!s.answer?.trim()) err(ww, `step ${s.blank} 缺少 answer`)
    })
    if (g.tests.length < 2) err(ww, '跟练测试少于 2 个')
  })
  // 实战
  if (L.challenges.length < 1) err(w, 'challenges 至少 1 个')
  L.challenges.forEach((ch) => {
    const ww = `${w}.challenge[${ch.id}]`
    checkMd(`${ww}.description`, ch.description)
    checkMd(`${ww}.explanation`, ch.explanation)
    ch.hints.forEach((h, i) => checkMd(`${ww}.hints[${i}]`, h))
    ch.followUps?.forEach((f) => (checkMd(`${ww}.followUp.q`, f.q), checkMd(`${ww}.followUp.a`, f.a)))
    checkCode(`${ww}.starter`, ch.starter)
    checkCode(`${ww}.solution`, ch.solution)
    checkCode(`${ww}.setup`, ch.setup)
    ch.tests.forEach((t) => checkCode(`${ww}.test[${t.name}]`, t.code))
    if (ch.tests.length < 4) err(ww, `测试仅 ${ch.tests.length} 个（要求 ≥ 4）`)
    if (!ch.tests.some((t) => t.hidden)) err(ww, '没有隐藏测试')
    if (!ch.tests.some((t) => !t.hidden)) err(ww, '没有可见测试')
    if (!ch.hints.length) err(ww, '没有提示 hints')
    if (ch.starter.trim() === ch.solution.trim()) err(ww, 'starter 与 solution 相同')
    if (!ch.explanation?.trim()) err(ww, '缺少题解 explanation')
  })
  // 面试追问
  if (!boss && L.interview.length < 3) err(w, `interview 仅 ${L.interview.length} 条（要求 ≥ 3）`)
  L.interview.forEach((q) => (checkMd(`${w}.qa[${q.id}].q`, q.q), checkMd(`${w}.qa[${q.id}].a`, q.a)))
  checkMd(`${w}.cheatsheet`, L.cheatsheet)
  if (!L.cheatsheet) warn(w, '缺少 cheatsheet 速记卡')
  return { quiz: L.quiz.length, challenges: L.challenges.length, guided: L.guided.length, viz: L.learn.filter((s) => s.viz).length }
}

const loaded = await loadChapters(filter)
const allLevelIds = new Map<string, string>()
const stats = { chapters: 0, levels: 0, quiz: 0, challenges: 0, guided: 0, viz: 0 }
for (const lc of loaded) {
  if (lc.error || !lc.chapter) {
    err(`[chapter ${lc.dirId}]`, `加载失败：${lc.error?.split('\n').slice(0, 4).join(' | ')}`)
    continue
  }
  const ch = lc.chapter
  stats.chapters++
  const cw = `[chapter ${ch.id}]`
  if (ch.id !== lc.dirId) err(cw, `id 与目录名不一致（${lc.dirId}）`)
  if (!['algo', 'llm'].includes(ch.track)) err(cw, `track 非法: ${ch.track}`)
  if (!ch.levels?.length) err(cw, '没有关卡')
  checkMd(`${cw}.intro`, ch.intro)
  for (const meta of ch.levels) {
    if (allLevelIds.has(meta.id)) err(cw, `关卡 id 与 ${allLevelIds.get(meta.id)} 冲突: ${meta.id}`)
    allLevelIds.set(meta.id, ch.id)
    if (!meta.id.startsWith(ch.id + '-')) warn(cw, `关卡 id 建议以章节 id 为前缀: ${meta.id}`)
    if (onlyLevel && meta.id !== onlyLevel) continue
    try {
      const L = await loadLevelContent(meta)
      const s = checkLevel(ch, meta.id, meta.kind === 'boss', L)
      stats.levels++
      stats.quiz += s.quiz
      stats.challenges += s.challenges
      stats.guided += s.guided
      stats.viz += s.viz
    } catch (e) {
      err(`[${meta.id}]`, `加载失败：${String((e as Error)?.stack ?? e).split('\n').slice(0, 5).join(' | ')}`)
    }
  }
}

if (warns.length) console.log(warns.join('\n'))
if (errors.length) console.log(errors.join('\n'))
console.log(
  `\n${c.bold('内容统计')}：${stats.chapters} 章 · ${stats.levels} 关 · ${stats.quiz} 道练习 · ${stats.guided} 个跟练 · ${stats.challenges} 道实战 · ${stats.viz} 个可视化`,
)
console.log(errors.length ? c.red(`校验失败：${errors.length} 个错误，${warns.length} 个警告`) : c.green(`校验通过（${warns.length} 个警告）`))
process.exit(errors.length ? 1 : 0)
