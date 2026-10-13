/**
 * Python 运行器客户端（单例）。
 *  - preload()：后台预热（进入实战/跟练阶段时调用）
 *  - run()：运行测试；超时自动终止并重建 Worker（防死循环）
 *  - exec()：自由运行，返回 stdout
 *  - cancel()：中止正在运行的任务（同样会重建 Worker）
 *  - usePyStatus()：订阅加载状态（idle/loading/ready/error）、加载进度、是否在运行
 */
import { useSyncExternalStore } from 'react'
import type { RunRequest, RunResult, TestResult } from './harness'

export type PyStatus = 'idle' | 'loading' | 'ready' | 'error'
export interface ExecResult {
  stdout: string
  error: string | null
  errorLine: number | null
  timedOut?: boolean
  /** 用户主动中止 */
  cancelled?: boolean
  infraError?: string
}

export interface PySnapshot {
  status: PyStatus
  message: string
  /** 加载进度 0..1（仅 loading 时有意义） */
  progress: number
  /** 本次加载开始时间（performance.now），用于显示已等待时长 */
  loadStartedAt: number
  /** 是否有任务在运行 */
  busy: boolean
}

type Pending = { resolve: (v: unknown) => void; reject: (e: unknown) => void; onTest?: (result: TestResult) => void }
type Outcome<T> = T | 'timeout' | 'cancelled'

/** 站内自托管 pyodide 目录（绝对 URL）。在主线程解析，保证构建后（相对 base）与开发期都正确 */
function pyodideBase() {
  try {
    return new URL(`${import.meta.env.BASE_URL}pyodide/`, document.baseURI).href
  } catch {
    return `${import.meta.env.BASE_URL}pyodide/`
  }
}

class PyRunner {
  private worker: Worker | null = null
  private ready: Promise<void> | null = null
  private seq = 1
  private pending = new Map<number, Pending>()
  private listeners = new Set<() => void>()
  private current: { abort: () => void } | null = null
  private snap: PySnapshot = { status: 'idle', message: '', progress: 0, loadStartedAt: 0, busy: false }
  /** 串行队列：一次只跑一个任务 */
  private chain: Promise<unknown> = Promise.resolve()
  private loadTimer: ReturnType<typeof setTimeout> | null = null
  private rejectLoad: ((e: Error) => void) | null = null

  get status() {
    return this.snap.status
  }
  get message() {
    return this.snap.message
  }
  get snapshot() {
    return this.snap
  }

  subscribe = (fn: () => void) => {
    this.listeners.add(fn)
    return () => {
      this.listeners.delete(fn)
    }
  }
  getSnapshot = () => this.snap

  private patch(p: Partial<PySnapshot>) {
    this.snap = { ...this.snap, ...p }
    this.listeners.forEach((l) => l())
  }
  private setStatus(status: PyStatus, message = '') {
    this.patch({ status, message })
  }

  private spawn() {
    let w: Worker
    try {
      w = new Worker(new URL('./pyodide.worker.ts', import.meta.url), { type: 'module' })
    } catch (e) {
      this.setStatus('error', String((e as Error)?.message ?? e))
      this.ready = Promise.reject(e)
      this.ready.catch(() => {})
      return
    }
    this.worker = w
    let settle: { resolve: () => void; reject: (e: Error) => void } | null = null
    this.ready = new Promise<void>((resolve, reject) => {
      settle = { resolve, reject }
      this.rejectLoad = reject
    })
    this.ready.catch(() => {})
    w.onmessage = (ev) => {
      if (this.worker !== w) return
      const m = ev.data
      if (m.type === 'fatal') {
        settle?.reject(new Error(m.message))
        settle = null
        this.pending.forEach((p) => p.reject(new Error(m.message)))
        this.restart()
      } else if (m.type === 'test') {
        this.pending.get(m.id)?.onTest?.(m.result)
      } else if (m.type === 'progress') {
        this.patch({ status: 'loading', message: m.message, progress: typeof m.pct === 'number' ? m.pct : this.snap.progress })
      } else if (m.type === 'ready') {
        if (this.loadTimer) clearTimeout(this.loadTimer)
        this.loadTimer = null
        this.rejectLoad = null
        this.patch({ status: 'ready', message: '', progress: 1 })
        settle?.resolve()
        settle = null
      } else if (m.type === 'error' && m.id == null) {
        if (this.loadTimer) clearTimeout(this.loadTimer)
        this.loadTimer = null
        this.rejectLoad = null
        this.setStatus('error', m.message)
        this.ready = null
        settle?.reject(new Error(m.message))
        settle = null
      } else if (m.type === 'result' || m.type === 'error') {
        const p = this.pending.get(m.id)
        if (!p) return
        this.pending.delete(m.id)
        if (m.type === 'result') p.resolve(m.result)
        else p.reject(new Error(m.message))
      }
    }
    w.onerror = (ev) => {
      if (this.worker !== w) return
      ev.preventDefault?.()
      const msg = ev.message || 'Python Worker 异常退出'
      // 加载阶段：标记失败；运行阶段：让当前任务失败并重建
      if (settle) {
        if (this.loadTimer) clearTimeout(this.loadTimer)
        this.loadTimer = null
        this.rejectLoad = null
        this.setStatus('error', msg)
        this.ready = null
        settle.reject(new Error(msg))
        settle = null
        this.worker?.terminate()
        this.worker = null
      } else {
        this.pending.forEach((p) => p.reject(new Error(msg)))
        this.pending.clear()
        this.restart()
      }
    }
    this.patch({ status: 'loading', message: '加载 Python 运行时…', progress: 0.02, loadStartedAt: performance.now() })
    this.loadTimer = setTimeout(() => {
      if (this.worker !== w || !settle) return
      const message = 'Python 资源加载超过 60 秒，请检查网络后重试。'
      settle.reject(new Error(message))
      settle = null
      this.kill()
      this.setStatus('error', message)
    }, 60_000)
    w.postMessage({ type: 'init', base: pyodideBase() })
  }

  /** 预热运行时（幂等）。失败后再次调用会重试 */
  preload(): Promise<void> {
    if (!this.worker || !this.ready) {
      this.worker?.terminate()
      this.worker = null
      this.spawn()
    }
    return this.ready!
  }

  private kill() {
    if (this.loadTimer) clearTimeout(this.loadTimer)
    this.loadTimer = null
    this.rejectLoad?.(new Error('Python 加载已中止'))
    this.rejectLoad = null
    this.worker?.terminate()
    this.worker = null
    this.ready = null
    this.pending.forEach((p) => p.reject(new Error('terminated')))
    this.pending.clear()
  }

  /** 终止并在后台重建（下次运行更快） */
  private restart() {
    this.kill()
    this.patch({ status: 'idle', message: '', progress: 0 })
    setTimeout(() => this.preload().catch(() => {}), 50)
  }

  /** 中止正在运行的任务（排队中的任务不受影响） */
  cancel() {
    this.current?.abort()
  }

  private call<T>(payload: Record<string, unknown>, timeoutMs: number, onTest?: (result: TestResult) => void): Promise<Outcome<T>> {
    const task = async () => {
      const id = this.seq++
      this.patch({ busy: true })
      return new Promise<Outcome<T>>((resolve, reject) => {
        let done = false
        let timer: ReturnType<typeof setTimeout> | undefined
        const finish = (fn: () => void) => {
          if (done) return
          done = true
          clearTimeout(timer)
          this.current = null
          this.patch({ busy: false })
          fn()
        }
        this.current = {
          abort: () =>
            finish(() => {
              this.pending.delete(id)
              this.restart()
              resolve('cancelled')
            }),
        }
        this.preload().then(() => {
          if (done) return
          const w = this.worker
          if (!w) { finish(() => reject(new Error('运行时不可用'))); return }
          timer = setTimeout(() => finish(() => {
            this.pending.delete(id)
            this.restart()
            resolve('timeout')
          }), timeoutMs)
          this.pending.set(id, {
            onTest,
            resolve: (v) => finish(() => resolve(v as T)),
            reject: (e) => finish(() => reject(e)),
          })
          w.postMessage({ ...payload, id })
        }).catch((e) => finish(() => reject(e)))
      })
    }
    const p = this.chain.then(task, task)
    this.chain = p.catch(() => {})
    return p
  }

  async run(req: RunRequest, timeoutMs = 10000): Promise<RunResult & { cancelled?: boolean }> {
    const t0 = performance.now()
    const results: TestResult[] = []
    const empty = { ok: false, compileError: null, errorLine: null, stdout: '', results }
    try {
      const r = await this.call<RunResult>({ type: 'run', code: req.code, setup: req.setup, tests: req.tests }, timeoutMs, (test) => results.push(test))
      if (r === 'timeout') return { ...empty, timedOut: true, durationMs: performance.now() - t0 }
      if (r === 'cancelled') return { ...empty, cancelled: true, durationMs: performance.now() - t0 }
      return { ...r, durationMs: performance.now() - t0 }
    } catch (e) {
      return { ...empty, infraError: String((e as Error)?.message ?? e), durationMs: performance.now() - t0 }
    }
  }

  async exec(code: string, timeoutMs = 10000): Promise<ExecResult> {
    try {
      const r = await this.call<ExecResult>({ type: 'exec', code }, timeoutMs)
      if (r === 'timeout') return { stdout: '', error: null, errorLine: null, timedOut: true }
      if (r === 'cancelled') return { stdout: '', error: null, errorLine: null, cancelled: true }
      return r
    } catch (e) {
      return { stdout: '', error: null, errorLine: null, infraError: String((e as Error)?.message ?? e) }
    }
  }
}

export const pyRunner = new PyRunner()

/** 订阅运行时状态 */
export function usePyStatus(): PySnapshot {
  return useSyncExternalStore(pyRunner.subscribe, pyRunner.getSnapshot)
}

/** 空闲时预热（requestIdleCallback 不可用时退化为 setTimeout） */
export function preloadWhenIdle(delayMs = 3000): () => void {
  let idleId: number | null = null
  const t = setTimeout(() => {
    const ric = (window as unknown as { requestIdleCallback?: (cb: () => void, o?: { timeout: number }) => number }).requestIdleCallback
    if (ric) idleId = ric(() => void pyRunner.preload().catch(() => {}), { timeout: 4000 })
    else pyRunner.preload().catch(() => {})
  }, delayMs)
  return () => {
    clearTimeout(t)
    const cic = (window as unknown as { cancelIdleCallback?: (id: number) => void }).cancelIdleCallback
    if (idleId != null && cic) cic(idleId)
  }
}
