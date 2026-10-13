/**
 * MLA 数据流（DeepSeek-V3 维度）：展开形式 vs 推理吸收形式；点击任一节点看它的作用。
 * 维度来源：DeepSeek-V2/V3 论文（d=7168[V3], n_h=128, d_h=128, d_c=512, d_c'=1536, d_h^R=64）。
 */
import { useState } from 'react'
import { Legend, Tex, VizFrame } from '@/viz'
import { Segmented } from '@/components/ui'
import { cn } from '@/lib/utils'

type Mode = 'expand' | 'absorb'
interface Node {
  id: string
  tex: string
  dim: string
  note: string
  cached?: boolean
  off?: Mode // 在该模式下不计算
  only?: Mode // 只在该模式下出现
}

const COLS: { title: string; nodes: Node[] }[] = [
  {
    title: 'Query 路径',
    nodes: [
      { id: 'cq', tex: String.raw`c^Q_t = h_t W^{DQ}`, dim: '1536', note: 'query 也做低秩压缩（d_c′=1536），目的是减少训练时的激活显存，与 KV cache 无关。V3 在压缩向量后加了 RMSNorm。' },
      { id: 'qc', tex: String.raw`q^C_t = c^Q_t W^{UQ}`, dim: '128 × 128', note: '每个头 128 维的「内容」query（不带位置）。decode 时可以照常算出它，再右乘 W^UK_iᵀ 得到 q̃。' },
      { id: 'qlat', tex: String.raw`\tilde q_{t,i} = q^C_{t,i} {W^{UK}_i}^{\top}`, dim: '128 × 512', note: '吸收 W^UK：把 query 映射到 512 维潜空间，直接与缓存的 c^KV 做点积，永远不需要展开 k^C。可以先算 q^C 再乘 W^UK_iᵀ，也可以预乘 W^UQ_i W^UK_iᵀ。', only: 'absorb' },
      { id: 'qr', tex: String.raw`q^R_t = \mathrm{RoPE}(c^Q_t W^{QR})`, dim: '128 × 64', note: '每个头单独的 64 维位置 query，只负责与 k^R 做带位置的点积。' },
    ],
  },
  {
    title: 'KV 路径',
    nodes: [
      { id: 'ckv', tex: String.raw`c^{KV}_t = h_t W^{DKV}`, dim: '512', cached: true, note: 'K 与 V 联合压缩成一个 512 维潜向量——缓存的主体。V3 在其后加了 RMSNorm。' },
      { id: 'kc', tex: String.raw`k^C_t = c^{KV}_t W^{UK}`, dim: '128 × 128', note: '展开出每个头的内容 key。训练 / prefill 时计算；decode 时被吸收进 query 侧，不再计算。', off: 'absorb' },
      { id: 'vc', tex: String.raw`v^C_t = c^{KV}_t W^{UV}`, dim: '128 × 128', note: '展开出每个头的 value。decode 时被吸收进输出侧（W^UV 与 W^O 结合）。', off: 'absorb' },
    ],
  },
  {
    title: 'RoPE 路径',
    nodes: [
      { id: 'kr', tex: String.raw`k^R_t = \mathrm{RoPE}(h_t W^{KR})`, dim: '64（全头共享）', cached: true, note: '解耦的位置 key：一个 64 维向量被所有 128 个头共享（类似 MQA），必须缓存。RoPE 只作用在这部分上。' },
    ],
  },
]

const ATTN: Record<Mode, { tex: string; note: string }> = {
  expand: {
    tex: String.raw`s_{t,j,i} = \frac{q^C_{t,i}\cdot k^C_{j,i} + q^R_{t,i}\cdot k^R_j}{\sqrt{128+64}}`,
    note: '展开形式：先把缓存的 c^KV 展开成每头的 k^C、v^C（128 头 × 256 维），再做标准 MHA。计算量小，但要读/生成大量 K/V——适合 prefill（计算密集）。',
  },
  absorb: {
    tex: String.raw`s_{t,j,i} = \frac{\tilde q_{t,i}\cdot c^{KV}_j + q^R_{t,i}\cdot k^R_j}{\sqrt{128+64}}`,
    note: '吸收形式：等价于 128 个 query 头共享同一个「KV 头」的 MQA，key = [c^KV; k^R]（576 维），value = c^KV（512 维）。缩放仍是 √192——吸收只改变结合顺序，不改变数值。适合 decode（访存密集）。',
  },
}
const OUT: Record<Mode, { tex: string; note: string }> = {
  expand: { tex: String.raw`u_t = [o_{t,1};\dots;o_{t,128}]\,W^O,\ \ o_{t,i}=\sum_j p\, v^C_{j,i}`, note: '标准合头 + 输出投影。' },
  absorb: {
    tex: String.raw`u_t = \sum_i \Big(\sum_j p_{t,j,i}\, c^{KV}_j\Big) W^{UV}_i W^O_i`,
    note: '先在 512 维潜空间加权求和，再经 W^UV_i 与 W^O_i：W^UV 被「吸收」进 W^O（可预乘成 512×7168 的每头矩阵，或保持两步以免参数膨胀）。',
  },
}

export function MLAFlowViz() {
  const [mode, setMode] = useState<Mode>('expand')
  const [sel, setSel] = useState('ckv')
  const all = [...COLS.flatMap((c) => c.nodes)]
  const cur = all.find((n) => n.id === sel)
  const visible = (n: Node) => !n.only || n.only === mode

  return (
    <VizFrame
      title="MLA：只缓存 512 维潜向量 + 64 维 RoPE key"
      caption={
        <>
          {cur && visible(cur) ? (
            <>
              <b>
                <Tex>{cur.tex}</Tex>
              </b>
              ：{cur.note}
            </>
          ) : (
            '点击任一节点查看说明。'
          )}{' '}
          <span className="text-muted">每 token 每层缓存 512 + 64 = 576 个元素；同头数 MHA 为 2 × 128 × 128 = 32768。</span>
        </>
      }
      legend={
        <Legend
          items={[
            { label: '需要缓存', tone: 'success', variant: 'soft' },
            { label: 'decode 时被吸收、不再计算', tone: 'muted', variant: 'soft' },
            { label: '吸收后新出现的量', tone: 'llm', variant: 'soft' },
          ]}
        />
      }
      controls={
        <Segmented
          size="sm"
          value={mode}
          onChange={setMode}
          options={[
            { value: 'expand', label: '展开形式（训练 / prefill）' },
            { value: 'absorb', label: '吸收形式（decode）' },
          ]}
        />
      }
    >
      <div className="space-y-3">
        <div className="mx-auto w-max rounded-xl border border-border bg-surface px-3 py-1.5 text-center text-[13px]">
          <Tex>{String.raw`h_t \in \mathbb{R}^{7168}`}</Tex>
        </div>
        <div className="grid grid-cols-3 gap-2">
          {COLS.map((col) => (
            <div key={col.title} className="min-w-0 space-y-1.5">
              <div className="text-center text-[11px] font-medium text-muted">{col.title}</div>
              {col.nodes.filter(visible).map((n) => {
                const off = n.off === mode
                return (
                  <button
                    key={n.id}
                    onClick={() => setSel(n.id)}
                    className={cn(
                      'block w-full rounded-xl border px-1.5 py-1.5 text-center transition-colors',
                      n.cached ? 'border-success bg-success-soft' : n.only ? 'border-llm bg-llm-soft' : 'border-border bg-surface',
                      off && 'opacity-35 line-through',
                      sel === n.id && 'ring-2 ring-accent',
                    )}
                  >
                    <div className="overflow-x-auto text-[11.5px] leading-tight no-scrollbar">
                      <Tex>{n.tex}</Tex>
                    </div>
                    <div className="mt-0.5 font-mono text-[10.5px] text-muted">{n.dim}</div>
                  </button>
                )
              })}
            </div>
          ))}
        </div>
        <div className="rounded-xl border border-accent bg-accent-soft px-3 py-2 text-center">
          <div className="overflow-x-auto text-[13px] no-scrollbar">
            <Tex>{ATTN[mode].tex}</Tex>
          </div>
          <div className="mt-1 text-left text-[12px] text-muted">{ATTN[mode].note}</div>
        </div>
        <div className="rounded-xl border border-border bg-surface px-3 py-2 text-center">
          <div className="overflow-x-auto text-[13px] no-scrollbar">
            <Tex>{OUT[mode].tex}</Tex>
          </div>
          <div className="mt-1 text-left text-[12px] text-muted">{OUT[mode].note}</div>
        </div>
      </div>
    </VizFrame>
  )
}
