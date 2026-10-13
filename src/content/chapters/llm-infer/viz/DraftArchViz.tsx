/**
 * 四种草稿方式的数据流对比：Medusa / EAGLE / EAGLE-3 / P-EAGLE。
 * 记号：已提交到位置 t，目标模型已经给出特征 f(t) 并采样出 x(t+1)（树根）；草稿要猜 x(t+2)、x(t+3)、x(t+4)。
 */
import { useMemo, useState } from 'react'
import { Legend, StepControls, VizFrame, toneColor, useStepper, type VizTone } from '@/viz'
import { Segmented } from '@/components/ui'

type Kind = 'true' | 'pred' | 'shared' | 'tok'
interface Box {
  label: string
  kind: Kind
}
interface Col {
  inputs: Box[]
  module: string
  outputs: Box[]
  pass: number // 第几次草稿前向（同一个数字 = 同一次前向）
}
interface Method {
  cols: Col[]
  passes: number
  notes: string[]
  title: string
}

const METHODS: Record<string, Method> = {
  medusa: {
    title: 'Medusa',
    passes: 1,
    cols: [1, 2, 3].map((k) => ({
      inputs: [{ label: 'h(t) 最后一层', kind: 'true' as Kind }],
      module: `Medusa 头 ${k}`,
      outputs: [{ label: `x̂(t+${k + 1})`, kind: 'tok' as Kind }],
      pass: 1,
    })),
    notes: [
      '3 个头读的是同一个 h(t)，并行给出 x(t+2)、x(t+3)、x(t+4) 的 top-k。头 2 并不知道头 1 猜的是什么——它要「跳着」预测，越远越不准。每个头是带残差的单层 FFN，目标模型可冻结（Medusa-1）或一起微调（Medusa-2）。',
      '各头 top-k 做笛卡尔积组成候选树，用树注意力一次验证。草稿几乎零成本，代价是远端头的命中率低。',
    ],
  },
  eagle: {
    title: 'EAGLE',
    passes: 3,
    cols: [
      { inputs: [{ label: 'f(t) 目标特征', kind: 'true' }, { label: 'e(x(t+1))', kind: 'tok' }], module: '草稿层', outputs: [{ label: 'f̂(t+1)', kind: 'pred' }, { label: 'x̂(t+2)', kind: 'tok' }], pass: 1 },
      { inputs: [{ label: 'f̂(t+1)', kind: 'pred' }, { label: 'e(x̂(t+2))', kind: 'tok' }], module: '草稿层', outputs: [{ label: 'f̂(t+2)', kind: 'pred' }, { label: 'x̂(t+3)', kind: 'tok' }], pass: 2 },
      { inputs: [{ label: 'f̂(t+2)', kind: 'pred' }, { label: 'e(x̂(t+3))', kind: 'tok' }], module: '草稿层', outputs: [{ label: 'f̂(t+3)', kind: 'pred' }, { label: 'x̂(t+4)', kind: 'tok' }], pass: 3 },
    ],
    notes: [
      '第 1 步：输入目标模型倒数第二层（LM head 之前）的特征 f(t)，以及已经采样出的下一个 token 的嵌入 e(x(t+1))。草稿层 = FC（2d→d）+ 1 层 decoder，预测下一个特征 f̂(t+1)，再用目标模型的 LM head 得到 x̂(t+2)。',
      '第 2 步：在「特征层面」自回归——把自己预测的 f̂(t+1) 和刚猜的 x̂(t+2) 喂回去。带上「提前一步的 token」消除了特征的不确定性：同一个 f 可能接不同的 token，告诉草稿真正采到了哪个，它才知道往哪条路走。',
      '第 3 步：继续自回归。训练时 f̂ 用 Smooth L1 回归真实特征，再加 0.1 倍的 token 交叉熵。问题：推理时喂的是有误差的 f̂，训练时喂的是真实 f，误差会累积。',
    ],
  },
  eagle3: {
    title: 'EAGLE-3',
    passes: 3,
    cols: [
      { inputs: [{ label: 'g(t)=FC[低;中;高]', kind: 'true' }, { label: 'e(x(t+1))', kind: 'tok' }], module: '草稿层', outputs: [{ label: 'a(t+1)', kind: 'pred' }, { label: 'x̂(t+2)', kind: 'tok' }], pass: 1 },
      { inputs: [{ label: 'a(t+1) 自己的输出', kind: 'pred' }, { label: 'e(x̂(t+2))', kind: 'tok' }], module: '草稿层', outputs: [{ label: 'a(t+2)', kind: 'pred' }, { label: 'x̂(t+3)', kind: 'tok' }], pass: 2 },
      { inputs: [{ label: 'a(t+2) 自己的输出', kind: 'pred' }, { label: 'e(x̂(t+3))', kind: 'tok' }], module: '草稿层', outputs: [{ label: 'a(t+3)', kind: 'pred' }, { label: 'x̂(t+4)', kind: 'tok' }], pass: 3 },
    ],
    notes: [
      '输入换成目标模型低、中、高三层特征拼接（3d 维）再经 FC 压回 d 维——不只用最后的特征，信息更丰富。',
      '去掉特征回归损失：草稿层的输出 a 不再被要求逼近目标特征，只对最终 token 负责（直接预测 token），表达能力不再被「特征约束」卡住，训练数据越多越好（scaling）。',
      'training-time test：训练时就模拟推理——第 2、3 步喂的是草稿自己的输出 a，而不是真实特征，并相应调整注意力掩码。训练与推理一致，后面位置的接受率不再快速下滑。仍沿用 EAGLE-2 的动态草稿树。',
    ],
  },
  peagle: {
    title: 'P-EAGLE',
    passes: 1,
    cols: [
      { inputs: [{ label: 'f(t) 目标特征', kind: 'true' }, { label: 'e(x(t+1))', kind: 'tok' }], module: '草稿模型（同一次前向）', outputs: [{ label: 'x̂(t+2)', kind: 'tok' }], pass: 1 },
      { inputs: [{ label: 'h_shared（可学习）', kind: 'shared' }, { label: 'e(MASK)（可学习）', kind: 'shared' }], module: '草稿模型（同一次前向）', outputs: [{ label: 'x̂(t+3)', kind: 'tok' }], pass: 1 },
      { inputs: [{ label: 'h_shared（可学习）', kind: 'shared' }, { label: 'e(MASK)（可学习）', kind: 'shared' }], module: '草稿模型（同一次前向）', outputs: [{ label: 'x̂(t+4)', kind: 'tok' }], pass: 1 },
    ],
    notes: [
      '自回归草稿要 K 次串行前向，起草延迟随 K 线性增长。P-EAGLE 一次前向产出全部 K 个草稿：第 1 个位置照常用 f(t) 与 e(x(t+1))；后面位置的输入还不存在，用两个可学习的占位参数——共享的 mask token 嵌入和共享隐状态 h_shared。',
      'vLLM 博客（2026-03）：4 层草稿模型，最多并行预测 10 个 token；在 B200 上相对 vanilla EAGLE-3 最高 1.69×（GPT-OSS-20B，并发 1）；论文报告 1.10–1.36×。vLLM v0.16.0 起通过 parallel_drafting 开启。',
    ],
  },
}

const TONE: Record<Kind, VizTone> = { true: 'success', pred: 'warning', shared: 'llm', tok: 'info' }

function Chip({ b, on }: { b: Box; on: boolean }) {
  const c = toneColor(TONE[b.kind])
  return (
    <div
      className="truncate rounded-md border px-1.5 py-1 text-center font-mono text-[11px] transition-opacity duration-300"
      style={{
        borderColor: c,
        borderStyle: b.kind === 'pred' ? 'dashed' : 'solid',
        background: `color-mix(in oklab, ${c} ${on ? 18 : 6}%, var(--surface))`,
        opacity: on ? 1 : 0.3,
        color: 'var(--fg)',
      }}
    >
      {b.label}
    </div>
  )
}

export function DraftArchViz() {
  const [mk, setMk] = useState<keyof typeof METHODS>('eagle')
  const m = METHODS[mk]
  const frames = useMemo(() => m.notes.map((note, i) => ({ note, upto: m.passes === 1 ? 1 : Math.min(i + 1, m.passes) })), [m])
  const st = useStepper(frames, { intervalMs: 3500 })
  const f = st.frame
  return (
    <VizFrame
      title="四种草稿：数据怎么流"
      caption={
        <>
          <b>{m.title}</b>：{f.note}
        </>
      }
      legend={
        <Legend
          items={[
            { label: '目标模型给的真实特征', tone: 'success', variant: 'soft' },
            { label: '草稿自己预测的（虚线）', tone: 'warning', variant: 'soft' },
            { label: '可学习占位参数', tone: 'llm', variant: 'soft' },
            { label: 'token / 嵌入', tone: 'info', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Segmented
            size="sm"
            value={mk}
            onChange={setMk}
            options={Object.entries(METHODS).map(([k, v]) => ({ value: k as keyof typeof METHODS, label: v.title }))}
          />
        </>
      }
    >
      <div className="grid grid-cols-3 gap-2">
        {m.cols.map((c, i) => {
          const on = c.pass <= f.upto
          return (
            <div key={i} className="min-w-0 space-y-1.5">
              <div className="text-center font-mono text-[10.5px] text-subtle">
                {m.passes === 1 ? '同一次前向' : `第 ${c.pass} 次前向`}
              </div>
              {c.inputs.map((b, j) => (
                <Chip key={j} b={b} on={on} />
              ))}
              <div className="text-center text-[12px] leading-none text-muted" style={{ opacity: on ? 1 : 0.3 }}>
                ↓
              </div>
              <div
                className="rounded-lg border-2 px-1 py-2 text-center text-[11.5px] font-semibold transition-opacity duration-300"
                style={{ borderColor: toneColor('accent'), opacity: on ? 1 : 0.3, color: 'var(--fg)' }}
              >
                {c.module}
              </div>
              <div className="text-center text-[12px] leading-none text-muted" style={{ opacity: on ? 1 : 0.3 }}>
                ↓
              </div>
              {c.outputs.map((b, j) => (
                <Chip key={j} b={b} on={on} />
              ))}
            </div>
          )
        })}
      </div>
      {(mk === 'eagle' || mk === 'eagle3') && (
        <div className="mt-2 text-center text-[11px] text-subtle">右侧一列的输入来自左侧一列的输出：三次草稿前向是串行的</div>
      )}
    </VizFrame>
  )
}
