/**
 * Muon 的 update RMS 对齐。满秩的 A×B 正交化矩阵 O = UVᵀ 有 min(A,B) 个奇异值为 1，
 * RMS(O) = √(min/(A·B)) = 1/√max(A,B)：矩阵越大，更新越小，且不同形状的层不一致。
 *  - 不缩放：RMS = 1/√max(A,B)，远小于 AdamW 典型的 0.2–0.4
 *  - Keller Jordan 实现：× √max(1, A/B)（A = 输出维，B = 输入维）→ RMS = 1/√B
 *  - Moonlight / Kimi K2 / GLM-4.5：× 0.2·√max(A,B) → 每层 RMS 都是 0.2；DeepSeek-V4 取 0.18
 * 层形状以 Llama 3-8B（d=4096，GQA 8 个 KV 头，FFN 14336）为例，权重按 PyTorch 的 (out, in) 记。
 */
import { useState } from 'react'
import { ProbBars, VizFrame } from '@/viz'
import { Segmented } from '@/components/ui'
import { C, Stat } from './plot'

const LAYERS: { name: string; out: number; inn: number }[] = [
  { name: 'q_proj', out: 4096, inn: 4096 },
  { name: 'k_proj', out: 1024, inn: 4096 },
  { name: 'o_proj', out: 4096, inn: 4096 },
  { name: 'gate/up', out: 14336, inn: 4096 },
  { name: 'down', out: 4096, inn: 14336 },
]
type Mode = 'none' | 'keller' | 'moon' | 'v4'

function rmsOf(out: number, inn: number, mode: Mode) {
  const base = 1 / Math.sqrt(Math.max(out, inn))
  if (mode === 'none') return base
  if (mode === 'keller') return base * Math.sqrt(Math.max(1, out / inn))
  return base * (mode === 'moon' ? 0.2 : 0.18) * Math.sqrt(Math.max(out, inn))
}

export function UpdateRMSViz() {
  const [mode, setMode] = useState<Mode>('none')
  const vals = LAYERS.map((l) => rmsOf(l.out, l.inn, mode))
  const max = mode === 'none' || mode === 'keller' ? 0.03 : 0.3

  return (
    <VizFrame
      title="Muon 的更新幅度取决于矩阵形状——需要按 √max(A,B) 缩放"
      caption={
        <>
          {mode === 'none' && <>正交化后每层的 update RMS = 1/√max(A,B)：q_proj 约 {vals[0].toFixed(4)}，FFN 约 {vals[3].toFixed(4)}——比 AdamW 典型的 0.2–0.4 小一个数量级以上，而且大矩阵更新更小。直接沿用 AdamW 的学习率完全不行。</>}
          {mode === 'keller' && <>Keller Jordan 的实现乘 <C>√max(1, out/in)</C>，RMS 变成 1/√fan_in：同一输入维的层更新幅度一致，但数值仍很小，需要单独调一个更大的学习率（他的默认 lr = 0.02）。</>}
          {mode === 'moon' && <>Moonlight 乘 <C>0.2·√max(A,B)</C>：每一层的 update RMS 都恰好是 0.2，落在 AdamW 的典型范围内，于是可以<b>直接复用 AdamW 的学习率与权重衰减</b>（Kimi K2、GLM-4.5 同样对齐到 0.2）。</>}
          {mode === 'v4' && <>DeepSeek-V4 把每个更新矩阵的 RMS 缩放到 0.18，论文的说法是为了复用 AdamW 的学习率（峰值 2.7e-4 / 2.0e-4，与 AdamW 部分共用）。</>}
        </>
      }
      controls={
        <Segmented
          size="sm"
          value={mode}
          onChange={setMode}
          options={[
            { value: 'none', label: '不缩放' },
            { value: 'keller', label: '√max(1,out/in)' },
            { value: 'moon', label: 'Moonlight 0.2' },
            { value: 'v4', label: 'V4 0.18' },
          ]}
        />
      }
    >
      <div className="grid items-start gap-5 @3xl:grid-cols-[1fr_220px]">
        <div className="min-w-0">
          <div className="mb-2 text-xs font-medium text-muted">各层正交化更新的 RMS（Llama 3-8B 形状，(out, in)）</div>
          <ProbBars
            max={max}
            format={(v) => v.toFixed(4)}
            items={LAYERS.map((l, i) => ({ label: l.name, p: vals[i], tone: mode === 'moon' || mode === 'v4' ? 'success' : 'accent', note: `${l.out / 1024}K×${l.inn / 1024}K` }))}
          />
          <div className="mt-2 text-[12px] text-muted">
            坐标上限 {max}；AdamW 的 update RMS 经验上约 0.2–0.4（Moonlight 论文）。
          </div>
        </div>
        <div className="grid grid-cols-2 gap-2 @3xl:grid-cols-1">
          <Stat label="满秩正交矩阵 RMS" value="1/√max(A,B)" />
          <Stat label="4096×14336 的缩放因子" value={(0.2 * Math.sqrt(14336)).toFixed(2)} sub="0.2·√14336" tone="success" />
          <Stat label="当前 q_proj RMS" value={vals[0].toFixed(4)} />
          <Stat label="当前 FFN RMS" value={vals[3].toFixed(4)} />
        </div>
      </div>
    </VizFrame>
  )
}
