/**
 * 激活比例的趋势：按发布时间排列，比较「激活参数 / 总参数」与「路由专家稀疏度 k/N」。
 * 数据均来自官方论文 / model card / config（见关卡正文表格）。
 * 洞见：k/N 一路走低（25% → ~1.6%），但整体激活比例在 3%–5% 附近趋于平台——注意力、嵌入、共享专家是「固定开销」。
 */
import { useState } from 'react'
import { Legend, VizFrame, toneColor } from '@/viz'
import { Segmented } from '@/components/ui'
import { Stat } from './plot'

interface ModelRow {
  name: string
  date: string
  total: number // B
  active: number // B
  k: number
  n: number
  shared: number
}

const MODELS: ModelRow[] = [
  { name: 'Mixtral 8×7B', date: '2023-12', total: 47, active: 13, k: 2, n: 8, shared: 0 },
  { name: 'DeepSeek-V2', date: '2024-05', total: 236, active: 21, k: 6, n: 160, shared: 2 },
  { name: 'DeepSeek-V3', date: '2024-12', total: 671, active: 37, k: 8, n: 256, shared: 1 },
  { name: 'Qwen3-235B-A22B', date: '2025-04', total: 235, active: 22, k: 8, n: 128, shared: 0 },
  { name: 'Kimi K2', date: '2025-07', total: 1000, active: 32, k: 8, n: 384, shared: 1 },
  { name: 'gpt-oss-120b', date: '2025-08', total: 117, active: 5.1, k: 4, n: 128, shared: 0 },
  { name: 'Qwen3.5-397B', date: '2026-02', total: 397, active: 17, k: 10, n: 512, shared: 1 },
  { name: 'DeepSeek-V4-Flash', date: '2026', total: 284, active: 13, k: 6, n: 256, shared: 1 },
  { name: 'DeepSeek-V4-Pro', date: '2026', total: 1600, active: 49, k: 6, n: 384, shared: 1 },
  { name: 'Kimi K3', date: '2026-07', total: 2800, active: 104, k: 16, n: 896, shared: 2 },
]

type Metric = 'act' | 'kn' | 'total'

export function ActivationTrendViz() {
  const [metric, setMetric] = useState<Metric>('act')
  const val = (m: ModelRow) => (metric === 'act' ? (m.active / m.total) * 100 : metric === 'kn' ? (m.k / m.n) * 100 : m.total)
  const max = metric === 'total' ? Math.log10(3000) : Math.max(...MODELS.map(val))
  const width = (m: ModelRow) => (metric === 'total' ? (Math.log10(m.total) - 1) / (max - 1) : val(m) / max) * 100
  const fmtV = (m: ModelRow) => (metric === 'total' ? (m.total >= 1000 ? `${(m.total / 1000).toFixed(m.total % 1000 ? 1 : 0)}T` : `${m.total}B`) : `${val(m).toFixed(metric === 'kn' && val(m) < 10 ? 2 : 1)}%`)
  const recent = MODELS.slice(-4)
  const avgRecent = recent.reduce((a, m) => a + (m.active / m.total) * 100, 0) / recent.length

  return (
    <VizFrame
      title="激活比例越来越低？分两个口径看"
      caption={
        metric === 'act' ? (
          <>
            激活参数 / 总参数：从 Mixtral 的 27.7% 降到 2025 年后的 3%–5%，此后基本走平（最近 4 个模型平均 {avgRecent.toFixed(1)}%）。
            Kimi K3（3.7%）甚至高于 K2（3.2%）——注意力、嵌入、共享专家这些「每个 token 都要算」的部分构成了下限。
          </>
        ) : metric === 'kn' ? (
          <>路由专家稀疏度 k/N 才是一路走低的那条线：8 选 2 的 25% → 256 选 8 的 3.1% → V4-Pro 384 选 6 的 1.56%、K3 896 选 16 的 1.79%（论文称 sparsity 56）。</>
        ) : (
          <>总参数（对数刻度）：两年多从 47B 涨到 2.8T，而每 token 激活参数只从 13B 涨到 104B。</>
        )
      }
      legend={<Legend items={[{ label: metric === 'act' ? '激活参数 / 总参数' : metric === 'kn' ? '激活路由专家数 / 路由专家总数' : '总参数（log）', tone: 'accent' }, { label: '2026 年模型', tone: 'llm' }]} />}
      controls={
        <Segmented
          size="sm"
          value={metric}
          onChange={setMetric}
          options={[
            { value: 'act', label: '激活参数比例' },
            { value: 'kn', label: '路由稀疏度 k/N' },
            { value: 'total', label: '总参数' },
          ]}
        />
      }
    >
      <div className="space-y-3">
        <div className="space-y-1.5">
          {MODELS.map((m) => (
            <div key={m.name} className="flex items-center gap-2 text-[12.5px]">
              <div className="w-[140px] shrink-0 truncate text-right font-mono text-fg @md:w-[150px]" title={m.name}>
                {m.name}
              </div>
              <div className="hidden w-[50px] shrink-0 font-mono text-[11px] text-muted @md:block">{m.date}</div>
              <div className="relative h-5 min-w-0 flex-1 overflow-hidden rounded-md bg-surface-2">
                <div
                  className="absolute inset-y-0 left-0 rounded-md"
                  style={{ width: `${Math.max(1, width(m))}%`, background: toneColor(m.date.startsWith('2026') ? 'llm' : 'accent'), transition: 'width 300ms' }}
                />
              </div>
              <div className="w-[52px] shrink-0 text-right font-mono text-xs tabular-nums text-muted">{fmtV(m)}</div>
            </div>
          ))}
        </div>
        <div className="grid grid-cols-2 gap-2 @xl:grid-cols-4">
          <Stat label="Mixtral 激活比例" value="27.7%" />
          <Stat label="V4-Pro 激活比例" value={`${((49 / 1600) * 100).toFixed(2)}%`} tone="accent" />
          <Stat label="K3 激活比例" value={`${((104 / 2800) * 100).toFixed(2)}%`} tone="accent" />
          <Stat label="K3 稀疏度 N/k" value="896/16 = 56" />
        </div>
      </div>
    </VizFrame>
  )
}
