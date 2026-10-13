/**
 * GRPO 组内优势：点击切换每个回答的「答案正确 / 格式正确」，实时看 reward → mean/std → advantage。
 * 核心洞见：优势只取决于组内相对好坏；全对或全错 → 优势全 0 → 这道题不产生策略梯度。
 */
import { useMemo, useState } from 'react'
import { Legend, ProbBars, Tex, VizFrame } from '@/viz'
import { Segmented } from '@/components/ui'
import { Stat, f3 } from './plot'

const ANSWERS = ['72', '68', '72', '72', '70', '72', '64', '72']
const INIT_ACC = [true, false, true, false, false, true, false, false]
const INIT_FMT = [true, true, true, false, true, true, true, true]

type RewardMode = 'acc' | 'acc+fmt'
type Norm = 'std' | 'none'

export function GroupAdvantageViz({ initialNorm = 'std' }: { initialNorm?: Norm }) {
  const [acc, setAcc] = useState(INIT_ACC)
  const [fmtOk, setFmtOk] = useState(INIT_FMT)
  const [mode, setMode] = useState<RewardMode>('acc+fmt')
  const [norm, setNorm] = useState<Norm>(initialNorm)
  const G = acc.length

  const { r, mean, std, adv } = useMemo(() => {
    const r = acc.map((a, i) => (a ? 1 : 0) + (mode === 'acc+fmt' ? (fmtOk[i] ? 0.5 : 0) : 0))
    const mean = r.reduce((s, v) => s + v, 0) / G
    const std = Math.sqrt(r.reduce((s, v) => s + (v - mean) ** 2, 0) / (G - 1)) // ddof=1，与 torch.std 一致
    const adv = r.map((v) => (norm === 'std' ? (v - mean) / (std + 1e-6) : v - mean))
    return { r, mean, std, adv }
  }, [acc, fmtOk, mode, norm, G])

  const allSame = std < 1e-9
  const maxAbs = Math.max(1e-9, ...adv.map(Math.abs))
  const toggle = (arr: boolean[], set: (v: boolean[]) => void, i: number) => set(arr.map((v, k) => (k === i ? !v : v)))

  return (
    <VizFrame
      title="组内相对优势：同一道题的 G 个回答互为 baseline"
      caption={
        allSame ? (
          <>
            ⚠️ 全组奖励完全相同（std = 0）：每个回答的优势都是 <b>0</b>，这道题对策略梯度<b>没有任何贡献</b>——这正是 DAPO「动态采样」要过滤掉的组。
          </>
        ) : (
          <>
            组均值 <span className="font-mono">{f3(mean)}</span>，标准差（ddof=1）<span className="font-mono">{f3(std)}</span>。
            高于均值的回答优势为正（其所有 token 的概率被推高），低于均值的为负。注意：同组优势之和恒为 0。
          </>
        )
      }
      legend={
        <Legend
          items={[
            { label: '优势 > 0：鼓励', tone: 'success' },
            { label: '优势 < 0：抑制', tone: 'danger' },
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
              { value: 'acc', label: 'r = 正确性(0/1)' },
              { value: 'acc+fmt', label: 'r = 正确性 + 0.5·格式' },
            ]}
          />
          <Segmented
            size="sm"
            value={norm}
            onChange={setNorm}
            options={[
              { value: 'std', label: '(r−mean)/std（GRPO）' },
              { value: 'none', label: 'r−mean（Dr.GRPO）' },
            ]}
          />
        </>
      }
    >
      <div className="grid gap-5 @3xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="min-w-0">
          <div className="mb-2 text-xs font-medium text-muted">① 题目：「…最终答案是多少？」标准答案 72。点击切换每个回答的判定：</div>
          <div className="grid grid-cols-2 gap-1.5 @md:grid-cols-4">
            {ANSWERS.map((a, i) => (
              <div key={i} className="rounded-xl border border-border bg-surface p-2 text-[12px]">
                <div className="mb-1 flex items-center justify-between font-mono text-muted">
                  <span>o{i + 1}</span>
                  <span>r={r[i]}</span>
                </div>
                <button
                  className="mb-1 w-full rounded-lg border px-1.5 py-1 text-left font-mono"
                  style={{
                    borderColor: acc[i] ? 'var(--success)' : 'var(--danger)',
                    background: acc[i] ? 'color-mix(in oklab, var(--success) 14%, var(--surface))' : 'color-mix(in oklab, var(--danger) 12%, var(--surface))',
                  }}
                  onClick={() => toggle(acc, setAcc, i)}
                >
                  {acc[i] ? '✓' : '✗'} \boxed{'{'}
                  {acc[i] ? '72' : a === '72' ? '71' : a}
                  {'}'}
                </button>
                <button
                  className="w-full rounded-lg border border-border px-1.5 py-0.5 text-left text-[11px] disabled:opacity-40"
                  disabled={mode === 'acc'}
                  onClick={() => toggle(fmtOk, setFmtOk, i)}
                >
                  {fmtOk[i] ? '✓ <think> 格式' : '✗ 缺 </think>'}
                </button>
              </div>
            ))}
          </div>
        </div>
        <div className="min-w-0">
          <div className="mb-2 text-xs font-medium text-muted">② 组内归一化得到优势 Â_i（同一回答的所有 token 共享）</div>
          <ProbBars
            items={adv.map((v, i) => ({ label: `o${i + 1}`, p: v, tone: v > 1e-9 ? 'success' : v < -1e-9 ? 'danger' : 'muted' }))}
            max={maxAbs}
            format={(v) => (v > 0 ? '+' : '') + f3(v)}
          />
          <div className="mt-3 grid grid-cols-3 gap-2">
            <Stat label="mean(r)" value={f3(mean)} />
            <Stat label="std(r)" value={f3(std)} />
            <Stat label="Σ Â_i" value={f3(adv.reduce((s, v) => s + v, 0))} tone="info" />
          </div>
          <div className="mt-3 overflow-x-auto text-[16px]">
            <Tex>{norm === 'std' ? String.raw`\hat A_i=\frac{r_i-\mathrm{mean}(\mathbf r)}{\mathrm{std}(\mathbf r)+10^{-6}}` : String.raw`\tilde A_i=r_i-\mathrm{mean}(\mathbf r)`}</Tex>
          </div>
        </div>
      </div>
    </VizFrame>
  )
}

export const GroupAdvantageNoStdViz = () => <GroupAdvantageViz initialNorm="none" />
