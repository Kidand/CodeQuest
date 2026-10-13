/**
 * 多轮工具调用轨迹的 loss mask：只有模型自己生成的 token 参与梯度，工具返回（环境观测）必须 mask 掉。
 */
import { useState } from 'react'
import { Legend, TokenRow, VizFrame, type VizTone } from '@/viz'
import { Segmented } from '@/components/ui'
import { Stat } from './plot'

type Role = 'prompt' | 'assistant' | 'tool'
const TRAJ: { text: string; role: Role }[] = [
  { text: '[user] 2023 年诺贝尔物理奖得主有几位？', role: 'prompt' },
  { text: '<think>需要查一下</think>', role: 'assistant' },
  { text: '<search>2023 Nobel physics</search>', role: 'assistant' },
  { text: '<result>Agostini, Krausz, L’Huillier …</result>', role: 'tool' },
  { text: '<think>三位，再确认一下</think>', role: 'assistant' },
  { text: '<search>Nobel 2023 physics laureates number</search>', role: 'assistant' },
  { text: '<result>The prize was awarded jointly to three …</result>', role: 'tool' },
  { text: '<answer>3</answer>', role: 'assistant' },
]
const LEN: Record<number, number> = { 0: 14, 1: 6, 2: 7, 3: 22, 4: 8, 5: 9, 6: 25, 7: 4 }
const TONE: Record<Role, VizTone> = { prompt: 'muted', assistant: 'accent', tool: 'warning' }

export function AgentMaskViz() {
  const [mode, setMode] = useState<'right' | 'wrong'>('right')
  const inLoss = (r: Role) => r === 'assistant' || (mode === 'wrong' && r === 'tool')
  const total = TRAJ.reduce((a, _, i) => a + LEN[i], 0)
  const lossTok = TRAJ.reduce((a, t, i) => a + (inLoss(t.role) ? LEN[i] : 0), 0)
  const toolTok = TRAJ.reduce((a, t, i) => a + (t.role === 'tool' ? LEN[i] : 0), 0)

  return (
    <VizFrame
      title="Agentic RL 的 loss mask：工具返回是环境，不是策略"
      caption={
        mode === 'right' ? (
          <>
            只有 assistant 段（<b>{lossTok}</b> / {total} 个 token）参与 loss；轨迹答对得到的奖励（如 +1）作为优势广播到这些 token 上。
            工具返回的 {toolTok} 个 token 只作为上下文被条件化。
          </>
        ) : (
          <>
            错误做法：把工具返回的 {toolTok} 个 token 也算进 loss。模型被训练去“预测检索结果”——这些 token 不是它的动作，
            梯度既没有因果意义，又会让模型学会<b>自己编造工具输出</b>；Search-R1 把“检索 token 掩码”列为稳定训练的关键。
          </>
        )
      }
      legend={
        <Legend
          items={[
            { label: 'prompt（不计 loss）', tone: 'muted', variant: 'soft' },
            { label: 'assistant（计 loss）', tone: 'accent', variant: 'soft' },
            { label: '工具返回（应 mask）', tone: 'warning', variant: 'soft' },
          ]}
        />
      }
      controls={
        <Segmented
          size="sm"
          value={mode}
          onChange={setMode}
          options={[
            { value: 'right', label: '正确：工具输出 mask' },
            { value: 'wrong', label: '错误：工具输出也算 loss' },
          ]}
        />
      }
    >
      <div className="grid gap-4 @3xl:grid-cols-[minmax(0,3fr)_minmax(0,1fr)]">
        <TokenRow
          tokens={TRAJ.map((t, i) => ({
            text: t.text,
            tone: TONE[t.role],
            dim: !inLoss(t.role),
            dashed: t.role === 'tool',
            sub: `${LEN[i]} tok · mask=${inLoss(t.role) ? 1 : 0}`,
          }))}
        />
        <div className="grid grid-cols-3 gap-2 @3xl:grid-cols-1">
          <Stat label="计入 loss 的 token" value={`${lossTok} / ${total}`} tone="accent" />
          <Stat label="工具返回 token" value={String(toolTok)} tone="warning" />
          <Stat label="轨迹奖励" value="+1（答对）" tone="success" />
        </div>
      </div>
    </VizFrame>
  )
}
