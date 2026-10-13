/**
 * MoE 路由流动图：8 个 token 经 router 选 top-k 个专家；右侧是每个专家的负载。
 * 洞见：总派发量恒为 T·k；k 和 E 一变，负载分布立刻变化——均衡不是自动成立的。
 */
import { useMemo, useState } from 'react'
import { GraphView, Legend, ProbBars, VizFrame, type GraphEdge, type GraphNode, type VizTone } from '@/viz'
import { Slider } from '@/components/ui'
import { softmax, topk, uniforms } from './moeMath'
import { LoadBars, Stat } from './plot'

const TOK = ['The', 'cat', 'sat', 'on', 'the', 'mat', '.', 'It']
const D = 4
const EMAX = 8
// 确定性的 token 表示与 router 权重
const X = (() => {
  const u = uniforms(11, TOK.length * D)
  return TOK.map((_, t) => u.slice(t * D, t * D + D))
})()
const WR = (() => {
  const u = uniforms(23, D * EMAX, -1.2, 1.2)
  return Array.from({ length: D }, (_, i) => u.slice(i * EMAX, i * EMAX + EMAX))
})()
const LOGITS = X.map((x) => Array.from({ length: EMAX }, (_, e) => x.reduce((s, v, i) => s + v * WR[i][e], 0) * 1.6))

const eTone = (e: number) => ((e % 8) + 1) as VizTone

export function MoeFlowViz() {
  const [E, setE] = useState(8)
  const [k, setK] = useState(2)
  const [sel, setSel] = useState<number | null>(1)
  const kk = Math.min(k, E)

  const route = useMemo(
    () =>
      LOGITS.map((row) => {
        const p = softmax(row.slice(0, E))
        const idx = topk(p, kk)
        const s = idx.reduce((a, i) => a + p[i], 0)
        return { p, idx, w: idx.map((i) => p[i] / s) }
      }),
    [E, kk],
  )
  const load = Array.from({ length: E }, (_, e) => route.filter((r) => r.idx.includes(e)).length)
  const ideal = (TOK.length * kk) / E
  const maxLoad = Math.max(...load)

  const W = 460
  const rows = Math.max(TOK.length, E)
  const H = rows * 42 + 16
  const ty = (t: number) => 26 + ((H - 52) * t) / (TOK.length - 1)
  const ey = (e: number) => (E === 1 ? H / 2 : 26 + ((H - 52) * e) / (E - 1))
  const nodes: GraphNode[] = [
    ...TOK.map((t, i) => ({ id: `t${i}`, label: t, x: 60, y: ty(i), tone: (sel === i ? 'gold' : undefined) as VizTone | undefined, r: 19 })),
    ...Array.from({ length: E }, (_, e) => ({ id: `e${e}`, label: `E${e}`, x: 392, y: ey(e), tone: eTone(e), r: 19, dim: load[e] === 0 })),
  ]
  const edges: GraphEdge[] = []
  route.forEach((r, t) =>
    r.idx.forEach((e, j) =>
      edges.push({
        from: `t${t}`,
        to: `e${e}`,
        tone: eTone(e),
        directed: true,
        dim: sel != null && sel !== t,
        label: sel === t ? r.w[j].toFixed(2) : undefined,
      }),
    ),
  )

  const cur = sel != null ? route[sel] : null
  return (
    <VizFrame
      title="token 如何被路由到 top-k 个专家"
      caption={
        <>
          共 {TOK.length} 个 token × k = {kk} → <b>{TOK.length * kk}</b> 次派发，平均每个专家应得 <b className="font-mono">{ideal.toFixed(2)}</b> 个；
          实际最忙的专家拿到 <b className="font-mono">{maxLoad}</b> 个（{(maxLoad / ideal).toFixed(2)}× 均值），
          {load.filter((v) => v === 0).length > 0 ? <> 有 <b>{load.filter((v) => v === 0).length}</b> 个专家一个 token 都没分到。</> : ' 没有空闲专家。'}
          点击左侧 token 看它的路由概率与权重。
        </>
      }
      legend={
        <Legend
          items={[
            { label: '选中的 token', tone: 'gold', variant: 'soft' },
            { label: '边上数字 = 重归一化后的门控权重', tone: 'muted', variant: 'line' },
            { label: '红虚线 = 理想负载 T·k/E', tone: 'danger', variant: 'line' },
          ]}
        />
      }
      controls={
        <>
          <Slider className="w-44" label="专家数 E" value={E} min={2} max={8} onChange={setE} />
          <Slider className="w-44" label="top-k" value={kk} min={1} max={Math.min(4, E)} onChange={setK} />
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-[3fr_2fr]">
        <GraphView nodes={nodes} edges={edges} width={W} height={H} onNodeClick={(id) => String(id).startsWith('t') && setSel(Number(String(id).slice(1)))} />
        <div className="min-w-0 space-y-3">
          <div className="text-xs font-medium text-muted">每个专家处理的 token 数（负载）</div>
          <LoadBars values={load} labels={load.map((_, e) => `E${e}`)} tones={load.map((_, e) => eTone(e))} refValue={ideal} refLabel={`均值 ${ideal.toFixed(1)}`} />
          <div className="grid grid-cols-2 gap-2">
            <Stat label="总派发 T·k" value={TOK.length * kk} />
            <Stat label="最大负载 / 均值" value={(maxLoad / ideal).toFixed(2)} tone={maxLoad / ideal > 1.5 ? 'danger' : 'success'} />
          </div>
          {cur && (
            <div className="space-y-1.5">
              <div className="text-xs font-medium text-muted">
                「{TOK[sel!]}」的 router 概率 softmax(x·W_r)（选中 top-{kk}）
              </div>
              <ProbBars
                items={cur.p.map((p, e) => ({
                  label: `E${e}`,
                  p,
                  tone: cur.idx.includes(e) ? eTone(e) : 'muted',
                  dim: !cur.idx.includes(e),
                  note: cur.idx.includes(e) ? `w=${cur.w[cur.idx.indexOf(e)].toFixed(2)}` : '',
                }))}
                max={1}
              />
            </div>
          )}
        </div>
      </div>
    </VizFrame>
  )
}
