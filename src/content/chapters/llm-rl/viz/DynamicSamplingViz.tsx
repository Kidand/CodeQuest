/**
 * 动态采样：训练越往后，全对的题越多，batch 里「有梯度的题」越少。
 * 拖动「训练进度」看全对/全错/混合三类题的占比；开启动态采样看要多采多少题才能凑满 batch。
 */
import { useMemo, useState } from 'react'
import { Shuffle } from 'lucide-react'
import { Legend, VizFrame, toneColor, toneSoft } from '@/viz'
import { Button, Segmented, Slider } from '@/components/ui'
import { rng } from '@/lib/utils'
import { Stat } from './plot'

const G = 8
const SHOW = 24
const BATCH = 16

interface Prompt {
  k: number
}

function simulate(seed: number, skill: number, n: number): Prompt[] {
  const r = rng(seed)
  const out: Prompt[] = []
  for (let j = 0; j < n; j++) {
    const d = (r() - 0.5) * 6 // 题目难度 ∈ [-3, 3]
    const p = 1 / (1 + Math.exp(-(skill - d)))
    let k = 0
    for (let g = 0; g < G; g++) if (r() < p) k++
    out.push({ k })
  }
  return out
}

export function DynamicSamplingViz() {
  const [skill, setSkill] = useState(0.5)
  const [seed, setSeed] = useState(3)
  const [dyn, setDyn] = useState(true)
  const stream = useMemo(() => simulate(seed, skill, 400), [seed, skill])
  const shown = stream.slice(0, SHOW)
  const kind = (k: number) => (k === G ? 'all1' : k === 0 ? 'all0' : 'mixed')
  const nMixed = shown.filter((p) => kind(p.k) === 'mixed').length
  const nAll1 = shown.filter((p) => p.k === G).length
  // 动态采样：顺序消费 stream，直到凑满 BATCH 个混合题
  let consumed = 0
  let got = 0
  for (const p of stream) {
    consumed++
    if (kind(p.k) === 'mixed') got++
    if (got === BATCH) break
  }
  const filled = got === BATCH

  return (
    <VizFrame
      title="动态采样：只把「有对有错」的题放进 batch"
      caption={
        dyn ? (
          <>
            开启动态采样：为了凑满 <b>{BATCH}</b> 道有效题，需要顺序采样 <b className="font-mono">{filled ? consumed : `> ${consumed}`}</b> 道题（每题 {G} 个回答），
            rollout 开销约为 <b className="font-mono">{filled ? (consumed / BATCH).toFixed(2) : '∞'}×</b>。代价换来的是：每个 batch 的有效题数恒定、梯度噪声稳定。
          </>
        ) : (
          <>
            不做过滤：前 {SHOW} 道题中只有 <b>{nMixed}</b> 道有对有错；<b>{nAll1}</b> 道全对、{SHOW - nMixed - nAll1} 道全错——它们的 GRPO 优势全为 0，
            batch 实际有效大小只有 {((nMixed / SHOW) * 100).toFixed(0)}%。训练越往后（拖动进度），全对的比例越高。
          </>
        )
      }
      legend={
        <Legend
          items={[
            { label: '混合（保留）', tone: 'accent', variant: 'soft' },
            { label: '全对（过滤）', tone: 'gold', variant: 'soft' },
            { label: '全错（过滤）', tone: 'muted', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <Slider className="min-w-[200px] flex-1" label="训练进度（模型能力）" value={skill} min={-2} max={4} step={0.1} onChange={setSkill} format={(v) => v.toFixed(1)} />
          <Segmented
            size="sm"
            value={dyn ? 'on' : 'off'}
            onChange={(v) => setDyn(v === 'on')}
            options={[
              { value: 'off', label: '不过滤' },
              { value: 'on', label: '动态采样' },
            ]}
          />
          <Button size="sm" variant="ghost" icon={<Shuffle size={14} />} onClick={() => setSeed((s) => s + 1)}>
            换一批题
          </Button>
        </>
      }
    >
      <div className="grid gap-4 @3xl:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <div className="grid grid-cols-6 gap-1.5 @md:grid-cols-8">
          {shown.map((p, j) => {
            const kd = kind(p.k)
            const tone = kd === 'mixed' ? 'accent' : kd === 'all1' ? 'gold' : 'muted'
            return (
              <div
                key={j}
                className="flex h-11 flex-col items-center justify-center rounded-lg border font-mono text-[12px]"
                style={{ borderColor: toneColor(tone), background: toneSoft(tone, kd === 'mixed' ? 22 : 14), opacity: dyn && kd !== 'mixed' ? 0.45 : 1 }}
              >
                <span className="text-[10px] text-muted">q{j + 1}</span>
                <span>
                  {p.k}/{G}
                </span>
              </div>
            )
          })}
        </div>
        <div className="grid grid-cols-3 gap-2 @3xl:grid-cols-1">
          <Stat label="前 24 题有效占比" value={`${((nMixed / SHOW) * 100).toFixed(0)}%`} tone="accent" />
          <Stat label="全对题" value={String(nAll1)} tone="gold" />
          <Stat label="凑满 16 题需采样" value={filled ? `${consumed} 题` : '采不满'} tone={dyn ? 'success' : undefined} />
        </div>
      </div>
    </VizFrame>
  )
}
