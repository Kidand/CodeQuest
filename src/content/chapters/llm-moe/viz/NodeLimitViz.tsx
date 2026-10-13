/**
 * 节点受限路由（DeepSeek-V2/V3）：先按「每个节点内最高的 2 个分数之和」选出 M 个节点，再在这些节点的专家里取 top-k。
 * V3：256 个路由专家分布在 8 个节点，K_r=8、M=4，每节点取 K_r/M = 2 个最高分求和（HF 实现 topk_group=4、组内 top-2）。
 * 这里缩小为 4 个节点 × 4 个专家、k=4。洞见：少量亲和度损失换来跨节点通信量上限。
 */
import { useState } from 'react'
import { Legend, VizFrame, toneColor, toneSoft } from '@/viz'
import { Slider } from '@/components/ui'
import { topk } from './moeMath'
import { Stat } from './plot'

const NODES = 4
const PER = 4
const K = 4
const S = [
  [0.92, 0.35, 0.3, 0.2],
  [0.88, 0.8, 0.25, 0.15],
  [0.86, 0.4, 0.38, 0.1],
  [0.84, 0.78, 0.3, 0.22],
]

export function NodeLimitViz() {
  const [M, setM] = useState(2)
  const flat = S.flat()
  const nodeScore = S.map((row) => topk(row, 2).reduce((a, i) => a + row[i], 0))
  const keepNodes = new Set(topk(nodeScore, M))
  const masked = flat.map((v, i) => (keepNodes.has(Math.floor(i / PER)) ? v : -Infinity))
  const sel = new Set(topk(masked, K))
  const free = new Set(topk(flat, K))
  const sum = (s: Set<number>) => [...s].reduce((a, i) => a + flat[i], 0)
  const nodesOf = (s: Set<number>) => new Set([...s].map((i) => Math.floor(i / PER))).size

  return (
    <VizFrame
      title="节点受限路由：每个 token 最多发往 M 个节点"
      caption={
        <>
          节点分数 = 节点内最高 2 个亲和度之和：[{nodeScore.map((v) => v.toFixed(2)).join(', ')}]。M = {M} 时保留节点{' '}
          {[...keepNodes].sort().map((n) => `N${n}`).join('、')}，再在其中选 top-{K}。
          不受限的 top-{K}（虚线框）横跨 <b>{nodesOf(free)}</b> 个节点；受限后只跨 <b>{nodesOf(sel)}</b> 个，亲和度之和从 {sum(free).toFixed(2)} 降到{' '}
          <b className="font-mono">{sum(sel).toFixed(2)}</b>。
        </>
      }
      legend={
        <Legend
          items={[
            { label: '受限路由选中', tone: 'accent' },
            { label: '虚线框 = 不受限 top-k', tone: 'muted', variant: 'line' },
            { label: '被保留的节点', tone: 'success', variant: 'soft' },
          ]}
        />
      }
      controls={<Slider className="w-56" label="最多节点数 M" value={M} min={1} max={4} onChange={setM} />}
    >
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-2 @xl:grid-cols-4">
          {S.map((row, n) => (
            <div
              key={n}
              className="rounded-xl border p-2"
              style={{ borderColor: keepNodes.has(n) ? toneColor('success') : 'var(--border)', background: keepNodes.has(n) ? toneSoft('success', 10) : 'var(--surface)' }}
            >
              <div className="mb-1.5 flex items-baseline justify-between text-xs">
                <b className="text-fg">节点 N{n}</b>
                <span className="font-mono text-muted">分 {nodeScore[n].toFixed(2)}</span>
              </div>
              <div className="space-y-1">
                {row.map((v, j) => {
                  const i = n * PER + j
                  const on = sel.has(i)
                  return (
                    <div
                      key={j}
                      className="flex items-center gap-1.5 rounded-md px-1 py-0.5 font-mono text-[11.5px]"
                      style={{ outline: free.has(i) ? `1.5px dashed ${toneColor('muted')}` : undefined, outlineOffset: 1 }}
                    >
                      <span className="w-8 shrink-0 text-muted">E{i}</span>
                      <div className="relative h-3.5 min-w-0 flex-1 overflow-hidden rounded bg-surface-2">
                        <div className="absolute inset-y-0 left-0 rounded" style={{ width: `${v * 100}%`, background: on ? toneColor('accent') : toneSoft('muted', 45) }} />
                      </div>
                      <span className={on ? 'w-8 text-right font-semibold text-fg' : 'w-8 text-right text-muted'}>{v.toFixed(2)}</span>
                    </div>
                  )
                })}
              </div>
            </div>
          ))}
        </div>
        <div className="grid grid-cols-3 gap-2">
          <Stat label="跨节点数（受限 / 不受限）" value={`${nodesOf(sel)} / ${nodesOf(free)}`} tone="accent" />
          <Stat label="选中亲和度之和" value={sum(sel).toFixed(2)} />
          <Stat label="相对最优" value={`${((sum(sel) / sum(free)) * 100).toFixed(1)}%`} tone={sum(sel) / sum(free) > 0.9 ? 'success' : 'danger'} />
        </div>
      </div>
    </VizFrame>
  )
}
