/**
 * Dispatch → expert → combine 的逐帧演示：按专家循环，一次性把属于该专家的 token 批量算完，再加权写回。
 * 洞见：循环的是 E 个专家而不是 T 个 token；同一专家内 token 不重复，所以 y[tok] += … 是安全的。
 */
import { useMemo } from 'react'
import { CodeTrace, Legend, MatrixView, StepControls, VizFrame, useStepper, type VizTone } from '@/viz'
import { softmax, topk } from './moeMath'

const CODE = `def moe_forward(x, W_r, experts, k):
    probs = softmax(x @ W_r)                    # (T, E)
    idx = np.argsort(-probs, axis=-1)[:, :k]    # (T, k) top-k
    w = np.take_along_axis(probs, idx, axis=-1)
    w = w / w.sum(-1, keepdims=True)            # 重归一化
    y = np.zeros_like(x)
    for e in range(E):
        tok, slot = np.nonzero(idx == e)        # 谁选了 e
        if tok.size == 0:
            continue                            # 空闲专家：跳过
        out = experts[e](x[tok])                # 批量计算
        y[tok] += w[tok, slot][:, None] * out   # 加权写回
    return y`

const LOGITS = [
  [2.0, 1.1, 0.2, -0.5],
  [0.3, 1.8, 1.2, -0.2],
  [1.5, -0.3, 1.4, 0.1],
  [0.9, 1.3, -0.4, 0.2],
  [-0.2, 0.6, 1.9, 0.5],
  [1.7, 0.6, 0.3, 0.0],
]
const T = LOGITS.length
const E = 4
const K = 2

interface Frame {
  line: number | number[]
  e?: number
  phase: 'route' | 'gather' | 'compute' | 'done'
  done: number[]
  caption: React.ReactNode
}

export function DispatchViz() {
  const route = useMemo(
    () =>
      LOGITS.map((l) => {
        const p = softmax(l)
        const idx = topk(p, K)
        const s = idx.reduce((a, i) => a + p[i], 0)
        return { p, idx, w: idx.map((i) => p[i] / s) }
      }),
    [],
  )
  const gate: (number | null)[][] = route.map((r) => Array.from({ length: E }, (_, e) => (r.idx.includes(e) ? r.w[r.idx.indexOf(e)] : null)))
  const bucket = Array.from({ length: E }, (_, e) => route.flatMap((r, t) => (r.idx.includes(e) ? [{ t, slot: r.idx.indexOf(e) }] : [])))

  const frames = useMemo<Frame[]>(() => {
    const fr: Frame[] = [
      {
        line: [2, 3, 4, 5],
        phase: 'route',
        done: [],
        caption: <>路由：softmax 后每行取 top-{K}，再在这 {K} 个里重归一化。表中有数字的格子就是被选中的 (token, 专家)，每行恰好 {K} 个、权重和为 1。</>,
      },
    ]
    const done: number[] = []
    for (let e = 0; e < E; e++) {
      const b = bucket[e]
      if (!b.length) {
        fr.push({ line: [8, 9, 10], e, phase: 'gather', done: [...done], caption: <>专家 E{e}：没有任何 token 选它，<b>直接跳过</b>——它这一步既不计算也拿不到梯度（这正是下一关「负载均衡」要解决的问题）。</> })
        continue
      }
      fr.push({
        line: 8,
        e,
        phase: 'gather',
        done: [...done],
        caption: (
          <>
            专家 E{e}：<span className="font-mono">np.nonzero(idx == {e})</span> 得到 tok = [{b.map((x) => x.t).join(', ')}]、slot = [{b.map((x) => x.slot).join(', ')}]
            （slot 表示这是该 token 的第几选择）。
          </>
        ),
      })
      done.push(e)
      fr.push({
        line: [11, 12],
        e,
        phase: 'compute',
        done: [...done],
        caption: (
          <>
            把这 {b.length} 个 token 拼成一个 ({b.length}, d) 的小 batch，<b>一次</b>调用 E{e} 的 FFN，再乘各自的权重
            {b.map((x) => ` x${x.t}:${route[x.t].w[x.slot].toFixed(2)}`).join('，')} 累加回 y。
          </>
        ),
      })
    }
    fr.push({
      line: 13,
      phase: 'done',
      done: [...done],
      caption: (
        <>
          完成：每个 token 恰好累加了 {K} 个专家的加权输出。专家负载 = [{bucket.map((b) => b.length).join(', ')}]，总和 = T·k = {T * K}。
          E3 一个 token 也没拿到，而 E1 拿了 5 个——这就是负载不均衡。
        </>
      ),
    })
    return fr
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  const st = useStepper(frames, { intervalMs: 1400 })
  const f = st.frame

  const hl = (r: number, c: number): VizTone | undefined => {
    if (gate[r][c] == null) return undefined
    if (f.e === c && f.phase !== 'done') return 'accent'
    if (f.done.includes(c)) return 'success'
    return undefined
  }

  return (
    <VizFrame
      title="Dispatch → 专家 FFN → Combine"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '当前专家处理的 token', tone: 'accent', variant: 'soft' },
            { label: '已累加回 y', tone: 'success', variant: 'soft' },
            { label: '· = 未被选中', tone: 'muted', variant: 'soft' },
          ]}
        />
      }
      controls={<StepControls stepper={st} />}
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="min-w-0 space-y-3">
          <div className="text-xs font-medium text-muted">门控权重 w（行 = token，列 = 专家）</div>
          <MatrixView
            data={gate}
            rowLabels={route.map((_, t) => `x${t}`)}
            colLabels={Array.from({ length: E }, (_, e) => `E${e}`)}
            domain={[0, 1]}
            cellSize={50}
            highlight={hl}
          />
          <div className="grid grid-cols-4 gap-1.5">
            {bucket.map((b, e) => (
              <div
                key={e}
                className="rounded-lg border px-1.5 py-1 text-center font-mono text-[11px]"
                style={{
                  borderColor: f.e === e ? 'var(--accent)' : f.done.includes(e) ? 'var(--success)' : 'var(--border)',
                  background: f.e === e ? 'color-mix(in oklab, var(--accent) 12%, var(--surface))' : 'var(--surface)',
                }}
              >
                <div className="font-semibold text-fg">E{e}</div>
                <div className="text-muted">{b.length ? b.map((x) => `x${x.t}`).join(' ') : '空闲'}</div>
              </div>
            ))}
          </div>
        </div>
        <CodeTrace code={CODE} active={f.line} vars={{ e: f.e ?? '—', 'load[e]': f.e != null ? bucket[f.e].length : '—' }} />
      </div>
    </VizFrame>
  )
}
