/**
 * 后训练方法对照：数据形态、需要的模型、在线 / 离线、损失与显存（模型状态，混合精度 Adam 16 B/参数、冻结 2 B/参数）。
 */
import { useState } from 'react'
import { Tex, VizFrame, toneColor, toneSoft, type VizTone } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { Stat } from './plot'

type M = 'sft' | 'ppo' | 'dpo' | 'simpo' | 'kto' | 'orpo' | 'grpo'
interface Info {
  data: string
  models: { name: string; train: boolean }[]
  online: boolean
  tex: string
  note: string
}
const INFO: Record<M, Info> = {
  sft: { data: '示范数据 (x, y)', models: [{ name: 'Policy', train: true }], online: false, tex: String.raw`-\sum_t\log\pi_\theta(y_t|x,y_{<t})`, note: '只在 assistant 回复上计 loss；教格式与能力，是后续所有方法的 π_ref。' },
  ppo: {
    data: '偏好对训练 RM；RL 阶段只需 prompt',
    models: [
      { name: 'Actor', train: true },
      { name: 'Critic', train: true },
      { name: 'Ref', train: false },
      { name: 'RM', train: false },
    ],
    online: true,
    tex: String.raw`\min(r_tA_t,\ \mathrm{clip}(r_t,1\pm\epsilon)A_t),\ \ r_t^{\mathrm{rew}}=-\beta\,\mathrm{KL}_t+\mathbb 1[t{=}T]\,r_\phi`,
    note: '上限高、在线探索，但四个模型、GAE、超参多，工程最重。',
  },
  dpo: {
    data: '偏好对 (x, y_w, y_l)',
    models: [
      { name: 'Policy', train: true },
      { name: 'Ref', train: false },
    ],
    online: false,
    tex: String.raw`-\log\sigma\Big(\beta\log\tfrac{\pi_\theta(y_w)}{\pi_{\mathrm{ref}}(y_w)}-\beta\log\tfrac{\pi_\theta(y_l)}{\pi_{\mathrm{ref}}(y_l)}\Big)`,
    note: '闭式解 + BT，Z(x) 相消；ref logp 可预计算。离线 → iterative DPO 补救。',
  },
  simpo: { data: '偏好对', models: [{ name: 'Policy', train: true }], online: false, tex: String.raw`-\log\sigma\Big(\tfrac{\beta}{|y_w|}\log\pi_\theta(y_w)-\tfrac{\beta}{|y_l|}\log\pi_\theta(y_l)-\gamma\Big)`, note: '长度归一化 + 目标 margin，无 reference，无 KL 锚点。' },
  kto: {
    data: '单条好/坏标签',
    models: [
      { name: 'Policy', train: true },
      { name: 'Ref', train: false },
    ],
    online: false,
    tex: String.raw`\lambda_y-\lambda_y\,\sigma\big(\pm\beta(r_\theta-z_0)\big)`,
    note: '前景理论，参考点 z₀ 由错位配对估计 KL；适合点赞/点踩数据。',
  },
  orpo: { data: '偏好对（可直接从 base 开始）', models: [{ name: 'Policy', train: true }], online: false, tex: String.raw`\mathcal L_{\mathrm{SFT}}-\lambda\log\sigma\Big(\log\tfrac{\mathrm{odds}_\theta(y_w)}{\mathrm{odds}_\theta(y_l)}\Big)`, note: 'SFT 与对齐一步完成，无 reference。' },
  grpo: {
    data: 'prompt + 可验证奖励（答案 / 测试）',
    models: [
      { name: 'Policy', train: true },
      { name: 'Ref', train: false },
    ],
    online: true,
    tex: String.raw`A_i=\tfrac{r_i-\mathrm{mean}(r)}{\mathrm{std}(r)},\ \ \min(r_tA_i,\ \mathrm{clip}(r_t)A_i)-\beta\,\mathrm{KL}`,
    note: '去掉 critic，用组内均值当基线；推理模型主流（见「推理强化学习」）。',
  },
}

export function PipelineViz() {
  const [m, setM] = useState<M>('dpo')
  const [P, setP] = useState(7)
  const it = INFO[m]
  const mem = it.models.reduce((a, x) => a + (x.train ? 16 : 2) * P, 0)
  const tone: VizTone = it.online ? 'warning' : 'info'

  return (
    <VizFrame
      title="后训练方法对照"
      caption={<>{it.note}</>}
      controls={
        <>
          <Segmented
            size="sm"
            value={m}
            onChange={setM}
            options={[
              { value: 'sft', label: 'SFT' },
              { value: 'ppo', label: 'RM+PPO' },
              { value: 'dpo', label: 'DPO' },
              { value: 'simpo', label: 'SimPO' },
              { value: 'kto', label: 'KTO' },
              { value: 'orpo', label: 'ORPO' },
              { value: 'grpo', label: 'GRPO' },
            ]}
          />
          <Slider className="min-w-[160px] flex-1" label="模型参数量（B）" value={P} min={1} max={70} onChange={setP} />
        </>
      }
    >
      <div className="flex flex-col gap-3">
        <div className="thin-scroll overflow-x-auto rounded-xl border border-border bg-surface px-3 py-3 text-center text-[13px] @md:text-[15px]">
          <Tex>{String.raw`\displaystyle ` + it.tex}</Tex>
        </div>
        <div className="flex flex-wrap gap-2">
          {it.models.map((x) => (
            <span
              key={x.name}
              className="rounded-lg border px-2.5 py-1 text-[13px]"
              style={{ borderColor: toneColor(x.train ? 'accent' : 'muted'), background: toneSoft(x.train ? 'accent' : 'muted', 16) }}
            >
              {x.name} · {x.train ? '训练' : '冻结'}
            </span>
          ))}
          <span className="rounded-lg border px-2.5 py-1 text-[13px]" style={{ borderColor: toneColor(tone), background: toneSoft(tone, 16) }}>
            {it.online ? '在线采样' : '离线数据'}
          </span>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <Stat label="数据形态" value={<span className="text-[13px]">{it.data}</span>} />
          <Stat label={`模型状态显存（${P}B）`} value={`${mem} GB`} tone={mem > 16 * P * 1.5 ? 'danger' : 'success'} sub="未含激活 / KV cache / rollout 引擎" />
        </div>
      </div>
    </VizFrame>
  )
}
