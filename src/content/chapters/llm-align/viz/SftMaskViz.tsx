/**
 * SFT 的 token 级 loss mask：chat template 展开后，哪些 token 的 label 是 -100。
 * 三种掩码策略 × 两种模板（ChatML / Llama 3）× 结束符是否计入 loss。
 */
import { useMemo, useState } from 'react'
import { Legend, TokenRow, VizFrame, type TokenItem, type VizTone } from '@/viz'
import { Segmented, Switch } from '@/components/ui'

type Role = 'system' | 'user' | 'assistant'
type Part = 'bos' | 'header' | 'content' | 'end' | 'sep'
interface Tok {
  text: string
  role: Role | null
  part: Part
  turn: number
}
type Tpl = 'chatml' | 'llama3'
type Mode = 'full' | 'assistant' | 'last'

const CONV: { role: Role; content: string }[] = [
  { role: 'system', content: 'Be brief.' },
  { role: 'user', content: '2+3=?' },
  { role: 'assistant', content: '5' },
  { role: 'user', content: '再×2?' },
  { role: 'assistant', content: '10' },
]

function render(tpl: Tpl): Tok[] {
  const out: Tok[] = []
  if (tpl === 'llama3') out.push({ text: '<|begin_of_text|>', role: null, part: 'bos', turn: -1 })
  CONV.forEach((m, i) => {
    const h = (text: string): Tok => ({ text, role: m.role, part: 'header', turn: i })
    if (tpl === 'chatml') {
      out.push(h('<|im_start|>'), h(m.role), h('\n'))
      out.push({ text: m.content, role: m.role, part: 'content', turn: i })
      out.push({ text: '<|im_end|>', role: m.role, part: 'end', turn: i })
      out.push({ text: '\n', role: m.role, part: 'sep', turn: i })
    } else {
      out.push(h('<|start_header_id|>'), h(m.role), h('<|end_header_id|>'), h('\n\n'))
      out.push({ text: m.content, role: m.role, part: 'content', turn: i })
      out.push({ text: '<|eot_id|>', role: m.role, part: 'end', turn: i })
    }
  })
  return out
}

const LAST_ASSISTANT = CONV.map((m) => m.role).lastIndexOf('assistant')

export function SftMaskViz({ initialMode = 'assistant', initialTpl = 'chatml' }: { initialMode?: Mode; initialTpl?: Tpl }) {
  const [tpl, setTpl] = useState<Tpl>(initialTpl)
  const [mode, setMode] = useState<Mode>(initialMode)
  const [eos, setEos] = useState(true)
  const toks = useMemo(() => render(tpl), [tpl])

  const inLoss = (t: Tok) => {
    if (mode === 'full') return true
    if (t.role !== 'assistant') return false
    if (mode === 'last' && t.turn !== LAST_ASSISTANT) return false
    return t.part === 'content' || (t.part === 'end' && eos)
  }
  const roleTone = (t: Tok): VizTone => (t.role === 'assistant' ? 'success' : t.role === 'user' ? 'info' : t.role === 'system' ? 'gold' : 'muted')
  const items: TokenItem[] = toks.map((t) => {
    const on = inLoss(t)
    return { text: t.text, tone: roleTone(t), dim: !on, dashed: t.part !== 'content', sub: on ? '✓' : '-100' }
  })
  const k = toks.filter(inLoss).length
  const endName = tpl === 'chatml' ? '<|im_end|>' : '<|eot_id|>'

  let caption: React.ReactNode
  if (mode === 'full')
    caption = <>预训练式：system、user 的文字和模板 token 也在学——模型会学着“扮演用户”、复述 system prompt，有效监督信号被大量用户文本稀释。</>
  else if (!eos)
    caption = (
      <>
        <b>{endName}</b> 没进 loss：模型从没被教过“何时停”，推理时会一直生成到 max_tokens，甚至自己续写下一轮 user。结束符必须计入 loss。
      </>
    )
  else if (mode === 'last')
    caption = <>只训最后一轮：第一轮 assistant 只当上下文。若要训练每一轮，就得把对话拆成 N 条样本，公共前缀被重复计算 N 次。</>
  else
    caption = (
      <>
        标准做法：所有 assistant 回复及其 <b>{endName}</b> 计入 loss；模板头（<span className="break-all font-mono text-[12.5px]">{tpl === 'chatml' ? '<|im_start|>assistant⏎' : '<|start_header_id|>assistant<|end_header_id|>⏎⏎'}</span>）在推理时由模板直接给出，不必学。一次前向就训练了每一轮回复。
      </>
    )

  return (
    <VizFrame
      title="Loss mask：只在 assistant 回复上学"
      badge={<span className="shrink-0 font-mono text-xs text-muted">{k}/{toks.length} 计入</span>}
      caption={caption}
      legend={
        <Legend
          items={[
            { label: 'system', tone: 'gold', variant: 'soft' },
            { label: 'user', tone: 'info', variant: 'soft' },
            { label: 'assistant', tone: 'success', variant: 'soft' },
            { label: '虚线框 = 模板 token；变暗 = label 为 -100', tone: 'muted', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <Segmented
            size="sm"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'full', label: '全序列' },
              { value: 'assistant', label: '所有 assistant 轮' },
              { value: 'last', label: '仅最后一轮' },
            ]}
          />
          <Segmented
            size="sm"
            value={tpl}
            onChange={setTpl}
            options={[
              { value: 'chatml', label: 'ChatML（Qwen）' },
              { value: 'llama3', label: 'Llama 3' },
            ]}
          />
          <Switch checked={eos} onChange={setEos} label="结束符计入 loss" />
        </>
      }
    >
      <TokenRow tokens={items} />
    </VizFrame>
  )
}

