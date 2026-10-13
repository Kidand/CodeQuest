/**
 * 细粒度专家切分 + 共享专家隔离（DeepSeekMoE）。
 * 设定与论文 §3 的例子一致：N=16 个标准尺寸专家、top-2；每个专家切成 m 份（中间维 /m），激活 m·K 份，
 * 其中 K_s 份隔离为共享专家（总参数与每 token 计算量都不变）。
 * 洞见：参数和 FLOPs 不变，可选的专家组合数从 120 暴涨到 ~4.4e9。
 */
import { useMemo, useState } from 'react'
import { Legend, VizFrame, toneColor, toneSoft } from '@/viz'
import { Segmented } from '@/components/ui'
import { comb, topk, uniforms } from './moeMath'
import { Stat } from './plot'

const N = 16
const K = 2

const SUP: Record<string, string> = { '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' }
const sci = (x: number) => {
  if (x < 1e5) return Math.round(x).toLocaleString('en-US')
  const [mant, ex] = x.toExponential(2).split('e+')
  return `${mant}×10${ex.split('').map((c) => SUP[c]).join('')}`
}

export function FineGrainedViz() {
  const [m, setM] = useState(4)
  const [ks, setKs] = useState(0)
  const total = N * m
  const routed = total - ks
  const act = m * K - ks
  // 确定性的「某个 token 的亲和度」：每个细粒度专家一个分数
  const scores = useMemo(() => uniforms(500 + m, total, 0, 1), [m, total])
  // 共享专家取编号最后的 ks 个（仅用于展示位置）
  const isShared = (i: number) => i >= routed
  const sel = useMemo(() => new Set(topk(scores.map((s, i) => (isShared(i) ? -1 : s)), act)), [scores, act, routed]) // eslint-disable-line react-hooks/exhaustive-deps
  const combos = comb(routed, act)

  return (
    <VizFrame
      title="细粒度切分：参数与计算不变，组合数暴涨"
      caption={
        <>
          把 {N} 个标准专家各切成 <b>{m}</b> 份（中间维变为 1/{m}），共 {total} 个小专家；每 token 激活 {m * K} 份
          {ks > 0 && <>，其中 <b>{ks}</b> 份固定为共享专家、{act} 份由 router 从 {routed} 个里选</>}。
          每 token 计算量始终等于 {K} 个标准 FFN，但可选组合从 C(16,2)=120 变成 <b className="font-mono">C({routed},{act}) ≈ {sci(combos)}</b>。
        </>
      }
      legend={
        <Legend
          items={[
            { label: '路由选中', tone: 'accent' },
            { label: '共享专家（每个 token 必经）', tone: 'gold' },
            { label: '未激活', tone: 'muted', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <div className="flex items-center gap-2 text-[13px] text-muted">
            切分倍数 m
            <Segmented size="sm" value={m} onChange={(v) => { setM(v); setKs(Math.min(ks, v * K - 1)) }} options={[1, 2, 4, 8].map((v) => ({ value: v, label: String(v) }))} />
          </div>
          <div className="flex items-center gap-2 text-[13px] text-muted">
            共享专家 K_s
            <Segmented
              size="sm"
              value={ks}
              onChange={setKs}
              options={[0, 1, 2].filter((v) => v <= m * K - 1).map((v) => ({ value: v, label: String(v) }))}
            />
          </div>
        </>
      }
    >
      <div className="space-y-4">
        <div className="text-xs font-medium text-muted">参数池：每一列是原来的 1 个标准专家，被切成 {m} 个小专家（同样的总参数）</div>
        <div className="mx-auto grid w-full max-w-[560px] gap-1" style={{ gridTemplateColumns: `repeat(${N}, minmax(0, 1fr))` }}>
          {Array.from({ length: N }, (_, col) => (
            <div key={col} className="flex flex-col gap-[3px]">
              {Array.from({ length: m }, (_, r) => {
                const i = col * m + r
                const shared = isShared(i)
                const on = sel.has(i)
                return (
                  <div
                    key={r}
                    className="rounded-[3px] border"
                    style={{
                      height: Math.max(6, 96 / m - 3),
                      background: shared ? toneColor('gold') : on ? toneColor('accent') : toneSoft('muted', 18),
                      borderColor: shared ? toneColor('gold') : on ? toneColor('accent') : 'var(--border)',
                      transition: 'background 200ms',
                    }}
                    title={shared ? '共享专家' : `专家 ${i}：亲和度 ${scores[i].toFixed(2)}`}
                  />
                )
              })}
            </div>
          ))}
        </div>
        <div className="grid grid-cols-2 gap-2 @xl:grid-cols-4">
          <Stat label="单个专家大小" value={`1/${m} 个标准 FFN`} />
          <Stat label="每 token 激活" value={`${act} 路由 + ${ks} 共享`} />
          <Stat label="每 token 计算量" value={`= ${K} 个标准 FFN`} tone="success" />
          <Stat label={`组合数 C(${routed},${act})`} value={sci(combos)} tone="accent" />
        </div>
      </div>
    </VizFrame>
  )
}
