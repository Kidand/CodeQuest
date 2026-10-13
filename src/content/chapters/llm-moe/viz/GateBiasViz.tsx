/**
 * DeepSeek-V3 门控：bias 只参与「选谁」，不参与「权重多大」。
 * g_i = 2.5 · s_i / Σ_{j∈TopK(s+b)} s_j，s = sigmoid(u·e)。
 * 洞见：拖动热门专家 E0 的 bias，选择集合变化，但被选中专家的权重仍按原始亲和度计算。
 */
import { useState } from 'react'
import { Legend, MatrixView, ProbBars, VizFrame } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { topk } from './moeMath'
import { Stat } from './plot'

const S = [0.83, 0.71, 0.64, 0.6, 0.52, 0.47, 0.35, 0.22]
const B0 = [0, -0.05, 0.03, 0.05, 0.1, 0.12, 0.04, 0.0]
const K = 3
const SCALE = 2.5

export function GateBiasViz() {
  const [b0, setB0] = useState(-0.3)
  const [mode, setMode] = useState<'right' | 'wrong'>('right')
  const b = [b0, ...B0.slice(1)]
  const choice = S.map((s, i) => s + b[i])
  const idx = topk(choice, K)
  const base = mode === 'right' ? S : choice
  const den = idx.reduce((a, i) => a + base[i], 0)
  const w = idx.map((i) => (SCALE * base[i]) / den)
  const plainIdx = topk(S, K)

  return (
    <VizFrame
      title="bias 只决定选谁，不决定权重"
      caption={
        <>
          选择用 s + b：top-{K} = {'{'}
          {idx.map((i) => `E${i}`).join(', ')}
          {'}'}
          {idx.join() !== plainIdx.join() && <>（不加 bias 时是 {'{'}{plainIdx.map((i) => `E${i}`).join(', ')}{'}'}）</>}。
          {mode === 'right' ? (
            <> 权重用<b>原始</b> s 归一化后 ×{SCALE}：bias 不进入前向输出，也就不会产生干扰主损失的梯度。</>
          ) : (
            <> <b className="text-danger">错误实现：</b>权重用了 s + b，每次 bias 更新都会直接改变模型输出——一个不经梯度学习的量混进了前向计算，均衡策略「污染」了模型本身。</>
          )}
        </>
      }
      legend={<Legend items={[{ label: '被选中', tone: 'accent' }, { label: 'E0 = 被挤爆的热门专家', tone: 'warning', variant: 'soft' }]} />}
      controls={
        <>
          <Slider className="min-w-[200px] flex-1" label="热门专家 E0 的 bias b₀" value={b0} min={-0.5} max={0.2} step={0.01} onChange={setB0} format={(v) => v.toFixed(2)} />
          <Segmented
            size="sm"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'right', label: '权重用 s（正确）' },
              { value: 'wrong', label: '权重用 s+b（错误）' },
            ]}
          />
        </>
      }
    >
      <div className="grid items-start gap-5 @3xl:grid-cols-2">
        <div className="min-w-0 space-y-2">
          <div className="text-xs font-medium text-muted">亲和度 s、偏置 b 与选择分数 s+b（描边列 = 选中）</div>
          <MatrixView
            data={[S, b, choice]}
            rowLabels={['s', 'b', 's+b']}
            colLabels={S.map((_, i) => `E${i}`)}
            scale="none"
            cellSize={44}
            highlight={(r, c) => (idx.includes(c) ? 'accent' : r === 1 && c === 0 ? 'warning' : undefined)}
          />
        </div>
        <div className="min-w-0 space-y-3">
          <div className="text-xs font-medium text-muted">最终门控权重 g（和恒为 {SCALE}）</div>
          <ProbBars items={idx.map((i, j) => ({ label: `E${i}`, p: w[j], tone: mode === 'right' ? 'accent' : 'danger' }))} max={SCALE} format={(v) => v.toFixed(3)} />
          <div className="grid grid-cols-2 gap-2">
            <Stat label="E0 是否被选中" value={idx.includes(0) ? '是' : '否'} tone={idx.includes(0) ? 'warning' : 'success'} />
            <Stat label="bias 是否影响输出" value={mode === 'right' ? '否（只改选择）' : '是 ✗'} tone={mode === 'right' ? 'success' : 'danger'} />
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
