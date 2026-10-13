#!/usr/bin/env node
/**
 * 把 Pyodide 运行时 + numpy 复制/下载到 public/pyodide/，实现自托管（国内网络 / 离线可用）。
 * 运行时会优先从站内加载，失败再回退 CDN。
 */
import fs from 'node:fs'
import path from 'node:path'

const root = path.resolve(import.meta.dirname, '..')
const src = path.join(root, 'node_modules/pyodide')
const dst = path.join(root, 'public/pyodide')
const pkg = JSON.parse(fs.readFileSync(path.join(src, 'package.json'), 'utf8'))
const CDN = `https://cdn.jsdelivr.net/pyodide/v${pkg.version}/full/`

fs.mkdirSync(dst, { recursive: true })
const core = ['pyodide.mjs', 'pyodide.js', 'pyodide.asm.mjs', 'pyodide.asm.wasm', 'python_stdlib.zip', 'pyodide-lock.json']
for (const f of core) fs.copyFileSync(path.join(src, f), path.join(dst, f))

const lock = JSON.parse(fs.readFileSync(path.join(src, 'pyodide-lock.json'), 'utf8'))
const want = new Set(['numpy'])
// 递归加入依赖
for (const name of [...want]) for (const d of lock.packages[name]?.depends ?? []) want.add(d)
for (const name of want) {
  const file = lock.packages[name].file_name
  const local = path.join(src, file)
  const out = path.join(dst, file)
  if (fs.existsSync(out)) continue
  if (fs.existsSync(local)) fs.copyFileSync(local, out)
  else {
    process.stdout.write(`下载 ${file} … `)
    const res = await fetch(CDN + file)
    if (!res.ok) throw new Error(`下载失败 ${res.status}`)
    fs.writeFileSync(out, Buffer.from(await res.arrayBuffer()))
    console.log('完成')
  }
}
const size = fs.readdirSync(dst).reduce((s, f) => s + fs.statSync(path.join(dst, f)).size, 0)
console.log(`Pyodide v${pkg.version} 已就绪：public/pyodide/（${(size / 1048576).toFixed(1)} MB）`)
