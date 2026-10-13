/**
 * PPO-RLHF 的四个模型：数据流 + 显存账本（只算模型状态：混合精度 Adam 每个可训练参数 16 字节，冻结模型 bf16 每参数 2 字节）。
 */
import { useState } from 'react'
import { motion } from 'motion/react'
import { GraphView, Legend, VizFrame, toneColor, type GraphEdge, type GraphNode, type VizTone } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { Stat } from './plot'

type Id = 'actor' | 'critic' | 'ref' | 'rm'
const SIZES = [1.5, 7, 13, 34, 70]
const INFO: Record<Id, { name: string; tone: VizTone; train: boolean; role: string }> = {
  actor: { name: 'Actor π_θ', tone: 'accent', train: true, role: 'Actor（策略）：从 SFT 初始化，负责 rollout 生成回复，并用 PPO clipped loss 更新。最终交付的就是它。' },
  critic: { name: 'Critic V_ψ', tone: 'warning', train: true, role: 'Critic（价值模型）：给每个 token 位置估计 V(s_t)，供 GAE 计算优势；常从 RM 初始化（InstructGPT 即如此），与 actor 一起训练。' },
  ref: { name: 'Ref π_ref', tone: 'info', train: false, role: 'Reference：冻结的 SFT 模型，只做前向，给出每个 token 的 log π_ref，用于逐 token KL 惩罚。' },
  rm: { name: 'RM r_φ', tone: 'success', train: false, role: 'Reward Model：冻结，只做前向，对完整回复打一个标量分，加在最后一个 token 的奖励上。' },
}

export function RlhfMemoryViz() {
  const [si, setSi] = useState(1)
  const [same, setSame] = useState<'same' | 'fixed'>('same')
  const [sel, setSel] = useState<Id>('actor')
  const P = SIZES[si]
  const Pc = same === 'same' ? P : 7
  // GB（1e9 字节）：可训练 = bf16 权重 2 + bf16 梯度 2 + fp32 主权重 4 + Adam m 4 + v 4 = 16 B/参数
  const rows: { id: Id; p: number; parts: { label: string; gb: number; tone: VizTone }[] }[] = [
    { id: 'actor', p: P, parts: [{ label: '权重', gb: 2 * P, tone: 'accent' }, { label: '梯度', gb: 2 * P, tone: 'info' }, { label: 'fp32 主权重 + Adam', gb: 12 * P, tone: 'danger' }] },
    { id: 'critic', p: Pc, parts: [{ label: '权重', gb: 2 * Pc, tone: 'accent' }, { label: '梯度', gb: 2 * Pc, tone: 'info' }, { label: 'fp32 主权重 + Adam', gb: 12 * Pc, tone: 'danger' }] },
    { id: 'ref', p: P, parts: [{ label: '权重', gb: 2 * P, tone: 'accent' }] },
    { id: 'rm', p: Pc, parts: [{ label: '权重', gb: 2 * Pc, tone: 'accent' }] },
  ]
  const total = rows.reduce((a, r) => a + r.parts.reduce((b, x) => b + x.gb, 0), 0)
  const max = Math.max(...rows.map((r) => r.parts.reduce((b, x) => b + x.gb, 0)))
  const dpo = 16 * P + 2 * P

  const nodes: GraphNode[] = [
    { id: 'actor', label: 'Actor', x: 60, y: 100, tone: sel === 'actor' ? 'accent' : undefined, r: 28, sub: '训练' },
    { id: 'ref', label: 'Ref', x: 200, y: 30, tone: sel === 'ref' ? 'info' : undefined, r: 24, sub: '冻结' },
    { id: 'rm', label: 'RM', x: 200, y: 100, tone: sel === 'rm' ? 'success' : undefined, r: 24, sub: '冻结' },
    { id: 'critic', label: 'Critic', x: 200, y: 172, tone: sel === 'critic' ? 'warning' : undefined, r: 28, sub: '训练' },
    { id: 'gae', label: 'GAE', x: 330, y: 100, r: 24, sub: '→ 更新' },
  ]
  const edges: GraphEdge[] = [
    { from: 'actor', to: 'ref', directed: true, label: '回复' },
    { from: 'actor', to: 'rm', directed: true },
    { from: 'actor', to: 'critic', directed: true },
    { from: 'ref', to: 'gae', directed: true, label: 'KL' },
    { from: 'rm', to: 'gae', directed: true, label: '分数' },
    { from: 'critic', to: 'gae', directed: true, label: 'V' },
  ]

  return (
    <VizFrame
      title="RLHF 的四个模型与显存"
      caption={<>{INFO[sel].role}</>}
      legend={
        <Legend
          items={[
            { label: 'bf16 权重', tone: 'accent' },
            { label: 'bf16 梯度', tone: 'info' },
            { label: 'fp32 主权重 + Adam m/v', tone: 'danger' },
          ]}
        />
      }
      controls={
        <>
          <Slider className="min-w-[180px] flex-1" label="Actor 参数量" value={si} min={0} max={SIZES.length - 1} onChange={setSi} format={(v) => `${SIZES[v]}B`} />
          <Segmented
            size="sm"
            value={same}
            onChange={setSame}
            options={[
              { value: 'same', label: 'Critic/RM 与 Actor 同规模' },
              { value: 'fixed', label: 'Critic/RM 固定 7B' },
            ]}
          />
        </>
      }
    >
      <div className="grid items-center gap-5 @3xl:grid-cols-[minmax(0,4fr)_minmax(0,5fr)]">
        <GraphView nodes={nodes} edges={edges} width={380} height={215} onNodeClick={(id) => id !== 'gae' && setSel(id as Id)} />
        <div className="flex min-w-0 flex-col gap-2.5">
          {rows.map((r) => {
            const sum = r.parts.reduce((b, x) => b + x.gb, 0)
            return (
              <button key={r.id} onClick={() => setSel(r.id)} className="text-left">
                <div className="mb-0.5 flex items-baseline justify-between text-[12.5px]">
                  <span className="font-medium" style={{ color: toneColor(INFO[r.id].tone) }}>
                    {INFO[r.id].name} · {r.p}B {INFO[r.id].train ? '（训练）' : '（冻结）'}
                  </span>
                  <span className="font-mono text-muted tabular-nums">{sum.toFixed(0)} GB</span>
                </div>
                <div className="flex h-4 overflow-hidden rounded-md bg-surface-2" style={{ outline: sel === r.id ? `2px solid ${toneColor(INFO[r.id].tone)}` : undefined }}>
                  {r.parts.map((x) => (
                    <motion.div key={x.label} initial={false} animate={{ width: `${(x.gb / max) * 100}%` }} style={{ background: toneColor(x.tone) }} />
                  ))}
                </div>
              </button>
            )
          })}
          <div className="mt-1 grid grid-cols-2 gap-2">
            <Stat label="四个模型的模型状态" value={`${total.toFixed(0)} GB`} tone="danger" sub={`≈ ${Math.ceil(total / 80)} 张 80GB（未含激活 / KV）`} />
            <Stat label="对比 DPO（policy + ref）" value={`${dpo.toFixed(0)} GB`} tone="success" sub="ref logp 还可离线预计算" />
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
