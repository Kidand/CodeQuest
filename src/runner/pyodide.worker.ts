/// <reference lib="webworker" />
/**
 * Pyodide Web Worker：加载 Python 运行时 + numpy，执行 harness。
 * 加载源按顺序回退：站内自托管(public/pyodide) → jsDelivr → fastly 镜像。
 *
 * 消息协议：
 *   in : { type:'init', base }            base = 站内 pyodide 目录的绝对 URL（由主线程按 document.baseURI 解析）
 *        { type:'run', id, code, setup, tests } / { type:'exec', id, code }
 *   out: { type:'progress', message, pct }  加载进度（pct ∈ [0,1]）
 *        { type:'ready' } / { type:'result', id, result } / { type:'error', id?, message }
 */
import { HARNESS_PY } from './harness'

declare const self: DedicatedWorkerGlobalScope

const VERSION = '314.0.7'

type InMsg =
  | { type: 'init'; base: string }
  | { type: 'run'; id: number; code: string; setup?: string; tests: unknown[] }
  | { type: 'exec'; id: number; code: string }

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let py: any = null
let initPromise: Promise<void> | null = null

const progress = (message: string, pct: number) => self.postMessage({ type: 'progress', message, pct })

async function tryLoad(indexURL: string, label: string) {
  progress(`下载 Python 运行时${label}…`, 0.08)
  const mod = await import(/* @vite-ignore */ `${indexURL}pyodide.mjs`)
  progress(`启动 Python 解释器${label}…`, 0.3)
  const inst = await mod.loadPyodide({ indexURL })
  progress('加载 numpy…', 0.7)
  await inst.loadPackage('numpy', { messageCallback: () => {}, errorCallback: () => {} })
  return inst
}

/** 兼容旧调用：base 既可以是站点根（'./'、'/'），也可以是 pyodide 目录本身 */
function resolveLocal(base: string) {
  const abs = new URL(base, self.location.href).href
  return /pyodide\/?$/.test(abs) ? abs.replace(/\/?$/, '/') : new URL('pyodide/', abs).href
}

function init(base: string) {
  if (initPromise) return initPromise
  initPromise = (async () => {
    const sources = [
      { url: resolveLocal(base), label: '' },
      { url: `https://cdn.jsdelivr.net/pyodide/v${VERSION}/full/`, label: '（站内资源不可用，改用 jsDelivr）' },
      { url: `https://fastly.jsdelivr.net/pyodide/v${VERSION}/full/`, label: '（改用 fastly 镜像）' },
    ]
    let lastErr: unknown
    for (const src of sources) {
      try {
        py = await tryLoad(src.url, src.label)
        break
      } catch (e) {
        lastErr = e
        py = null
      }
    }
    if (!py) throw lastErr ?? new Error('Pyodide 加载失败')
    progress('初始化测试环境…', 0.92)
    py.runPython(HARNESS_PY)
    // 预热常用模块（首次 import numpy 较慢）
    py.runPython('import numpy as np, math, collections, heapq, itertools, functools, bisect, random, inspect')
  })()
  initPromise.catch(() => {
    initPromise = null
  })
  return initPromise
}

self.onmessage = async (ev: MessageEvent<InMsg>) => {
  const msg = ev.data
  try {
    if (msg.type === 'init') {
      await init(msg.base)
      self.postMessage({ type: 'ready' })
      return
    }
    if (!py) throw new Error('运行时尚未就绪')
    if (msg.type === 'run') {
      py.globals.set('__cq_user', msg.code)
      py.globals.set('__cq_setup', msg.setup ?? '')
      py.globals.set('__cq_tests', JSON.stringify(msg.tests))
      py.globals.set('__cq_on_test', (json: string) => self.postMessage({ type: 'test', id: msg.id, result: JSON.parse(json) }))
      try {
        const json = py.runPython('json.dumps(_cq_run(__cq_user, __cq_setup, __cq_tests, __cq_on_test), ensure_ascii=False)')
        self.postMessage({ type: 'result', id: msg.id, result: JSON.parse(json) })
      } finally {
        try { py?.globals.delete('__cq_on_test') } catch { /* Preserve the original fatal error for the client. */ }
      }
    } else if (msg.type === 'exec') {
      py.globals.set('__cq_user', msg.code)
      const json = py.runPython('json.dumps(_cq_exec(__cq_user), ensure_ascii=False)')
      self.postMessage({ type: 'result', id: msg.id, result: JSON.parse(json) })
    }
  } catch (e) {
    const id = 'id' in msg ? msg.id : undefined
    const message = String((e as Error)?.message ?? e)
    const fatal = !!(e as { pyodide_fatal_error?: boolean })?.pyodide_fatal_error || /pyodide_fatal_error|fatal error|Maximum call stack|memory access out of bounds/i.test(message)
    self.postMessage({ type: fatal ? 'fatal' : 'error', id, message })
  }
}
