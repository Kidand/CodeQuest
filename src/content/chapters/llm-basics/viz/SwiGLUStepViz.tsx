/**
 * SwiGLU FFN 前向逐步演示（d = 4，h = 6；W_gate 手工设计，W_up / W_down 由固定种子生成并保留 1 位小数）。
 * 洞见：门控分支 SiLU(x W_gate) 逐元素“调节”内容分支 x W_up——门接近 0 的隐藏单元被关掉，
 * 三个矩阵：两个 d×h 升维，一个 h×d 降维。
 */
import { useMemo } from 'react'
import { ArrayView, CodeTrace, MatrixView, StepControls, VizFrame, useStepper, type VizTone } from '@/viz'
import { rng } from '@/lib/utils'
import { C } from './plot'

const D = 4
const H = 6
const CODE = `def swiglu_ffn(x, W_gate, W_up, W_down):  # x: (d,)
    a = x @ W_gate        # (h,) 门控分支预激活
    s = silu(a)           # (h,) a * sigmoid(a)
    u = x @ W_up          # (h,) 内容分支
    hid = s * u           # (h,) 逐元素门控
    return hid @ W_down   # (d,) 投影回模型维度`

function mat(seed: number, r: number, c: number) {
  const g = rng(seed)
  return Array.from({ length: r }, () => Array.from({ length: c }, () => Math.round((g() * 2 - 1) * 12) / 10))
}
const X = [0.8, -0.5, 1.2, 0.3]
/** 手工设计的门控权重：让 h1、h2、h3 的门几乎关闭，h5 的门为负值（SiLU 负半轴也会放行一点） */
const WG = [
  [1.0, -1.5, 0.3, -1.8, 0.9, -0.2],
  [-0.6, 1.0, 0.4, 0.9, -1.1, 0.6],
  [0.6, -1.8, 0.2, -2.0, 1.0, -0.4],
  [0.2, 0.2, -0.5, -0.6, 0.5, 0.8],
]
const WU = mat(5, D, H)
const WD = mat(8, H, D)
const mv = (v: number[], M: number[][]) => M[0].map((_, j) => v.reduce((s, vi, i) => s + vi * M[i][j], 0))
const silu = (z: number) => z / (1 + Math.exp(-z))

interface Frame {
  line: number
  show: number // 已计算出的向量个数
  mat?: 'g' | 'u' | 'd'
  text: React.ReactNode
}

export function SwiGLUStepViz() {
  const vals = useMemo(() => {
    const a = mv(X, WG)
    const s = a.map(silu)
    const u = mv(X, WU)
    const hid = s.map((v, i) => v * u[i])
    const y = mv(hid, WD)
    return { a, s, u, hid, y }
  }, [])
  const closed = useMemo(() => vals.s.map((v, i) => (Math.abs(v) < 0.1 ? i : -1)).filter((i) => i >= 0), [vals])
  const frames: Frame[] = useMemo(
    () => [
      { line: 1, show: 0, text: <>输入是一个 token 的隐藏向量 <C>x</C>（d = 4）。SwiGLU 有三个权重：<C>W_gate</C>、<C>W_up</C> 形状 (d, h)，<C>W_down</C> 形状 (h, d)。</> },
      { line: 2, show: 1, mat: 'g', text: <>门控分支：<C>a = x @ W_gate</C>，升维到 h = 6。</> },
      { line: 3, show: 2, text: <>过 SiLU：负的预激活被压到 0 附近（但不是严格 0），正的基本原样通过。门值接近 0 的单元：{closed.length ? closed.map((i) => `h${i}`).join('、') : '无'}。</> },
      { line: 4, show: 3, mat: 'u', text: <>内容分支：<C>u = x @ W_up</C>，与门控分支并行计算（实现上常把两个矩阵拼成一个 (d, 2h) 的 matmul）。</> },
      { line: 5, show: 4, text: <>逐元素相乘 <C>hid = s * u</C>：门控决定每个隐藏单元“放行多少”内容。门接近 0 的单元（变暗）不管 u 多大都被关掉；门为负的单元会把内容“反号”后少量放行。</> },
      { line: 6, show: 5, mat: 'd', text: <>降维：<C>hid @ W_down</C> 回到 d = 4，加回残差流。参数共 3·d·h = {3 * D * H} 个。</> },
    ],
    [closed],
  )
  const st = useStepper(frames, { intervalMs: 1400 })
  const f = st.frame
  const rows: { name: string; v: number[]; tone: VizTone }[] = [
    { name: 'a = x·W_gate', v: vals.a, tone: 'info' },
    { name: 's = SiLU(a)', v: vals.s, tone: 'warning' },
    { name: 'u = x·W_up', v: vals.u, tone: 'info' },
    { name: 'hid = s ⊙ u', v: vals.hid, tone: 'success' },
    { name: 'y = hid·W_down', v: vals.y, tone: 'accent' },
  ]
  const M = f.mat === 'g' ? WG : f.mat === 'u' ? WU : f.mat === 'd' ? WD : null

  return (
    <VizFrame
      title="SwiGLU：一条分支当“门”，一条分支当“内容”"
      caption={f.text}
      controls={<StepControls stepper={st} />}
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="min-w-0 space-y-2.5">
          <div>
            <div className="mb-1 font-mono text-[11.5px] text-muted">x（d = 4）</div>
            <ArrayView values={X} cellSize={40} showIndex={false} format={(v) => Number(v).toFixed(1)} highlights={f.show === 0 ? { 0: 'accent', 1: 'accent', 2: 'accent', 3: 'accent' } : {}} />
          </div>
          {rows.slice(0, f.show).map((r, k) => (
            <div key={r.name}>
              <div className="mb-1 font-mono text-[11.5px] text-muted">{r.name}</div>
              <ArrayView
                values={r.v}
                cellSize={44}
                showIndex={false}
                format={(v) => Number(v).toFixed(2)}
                highlights={k === f.show - 1 ? Object.fromEntries(r.v.map((_, i) => [i, r.tone])) : {}}
                dim={(i) => (k === 1 || k === 3) && closed.includes(i)}
              />
            </div>
          ))}
          {M && (
            <div className="pt-1">
              <div className="mb-1 font-mono text-[11.5px] text-muted">
                {f.mat === 'g' ? 'W_gate (4×6)' : f.mat === 'u' ? 'W_up (4×6)' : 'W_down (6×4)'}
              </div>
              <MatrixView data={M} scale="div" cellSize={36} format={(v) => v.toFixed(1)} />
            </div>
          )}
        </div>
        <CodeTrace code={CODE} active={f.line} vars={{ d: D, h: H, '参数': 3 * D * H }} />
      </div>
    </VizFrame>
  )
}
