/**
 * 为什么 Switch 的辅助损失是 N·Σ f_i·P_i，而不是只惩罚 P（如 N·Σ P_i²）？
 * 8 个 token、4 个专家：logits = 锐度 × (各 token 的偏好 + λ·[偏向 E0])。
 * 洞见：「平坦但 argmax 全是 E0」时，P 几乎均匀、N·ΣP² ≈ 1，P-only 损失看不出问题；
 *       f·P 的梯度 ∂L/∂P_i = N·f_i 精准指向真正过载的专家。
 */
import { useMemo, useState } from 'react'
import { Legend, VizFrame, toneColor } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { softmax } from './moeMath'
import { Stat } from './plot'

const N = 4
const T = 8
// 每个 token 的「天然偏好」：两两偏向 E0/E1/E2/E3，边际 1.0
const BASE = Array.from({ length: T }, (_, t) => Array.from({ length: N }, (_, e) => (e === Math.floor(t / 2) ? 1 : 0)))
type Preset = 'balanced' | 'flat' | 'collapse' | 'custom'
const PRESETS: Record<Exclude<Preset, 'custom'>, { lam: number; sharp: number }> = {
  balanced: { lam: 0, sharp: 3 },
  flat: { lam: 1.4, sharp: 0.25 },
  collapse: { lam: 1.4, sharp: 3 },
}

export function AuxLossViz() {
  const [preset, setPreset] = useState<Preset>('flat')
  const [lam, setLam] = useState(PRESETS.flat.lam)
  const [sharp, setSharp] = useState(PRESETS.flat.sharp)
  const pick = (p: Preset) => {
    setPreset(p)
    if (p !== 'custom') {
      setLam(PRESETS[p].lam)
      setSharp(PRESETS[p].sharp)
    }
  }
  const { f, P } = useMemo(() => {
    const probs = BASE.map((row) => softmax(row.map((v, e) => sharp * (v + (e === 0 ? lam : 0)))))
    const cnt = new Array(N).fill(0)
    probs.forEach((p) => cnt[p.indexOf(Math.max(...p))]++)
    return { f: cnt.map((v) => v / T), P: Array.from({ length: N }, (_, e) => probs.reduce((a, p) => a + p[e], 0) / T) }
  }, [lam, sharp])
  const LfP = N * f.reduce((a, v, i) => a + v * P[i], 0)
  const LP = N * P.reduce((a, v) => a + v * v, 0)
  const gfP = f.map((v) => N * v) // ∂L_fP/∂P_i
  const gP = P.map((v) => 2 * N * v) // ∂L_P/∂P_i
  const spread = (g: number[]) => Math.max(...g) - Math.min(...g)

  const W = 420
  const H = 180
  const bw = 22
  const sy = (v: number) => H - 26 - v * (H - 46)

  return (
    <VizFrame
      title="为什么要用 f·P，而不是只用 P"
      caption={
        <>
          实际负载 f = [{f.map((v) => v.toFixed(2)).join(', ')}]，平均概率 P = [{P.map((v) => v.toFixed(2)).join(', ')}]。
          {f[0] === 1 && LP < 1.1 ? (
            <> <b>P 几乎均匀</b>，只惩罚 P 的损失 N·ΣP² = {LP.toFixed(3)} ≈ 1（以为很均衡），但所有 token 都去了 E0！f·P 的梯度只落在 E0 上（{gfP[0].toFixed(1)} vs 0），精准地把 E0 的概率往下压。</>
          ) : f[0] === 1 ? (
            <> 完全坍缩：两种损失都很大，但只有 f·P 的梯度集中在真正过载的 E0 上。</>
          ) : (
            <> 负载均衡时 N·Σf·P 取到最小值 1，两种损失一致。</>
          )}
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'f_i：实际分到的 token 比例（argmax 计数，不可导）', tone: 'danger' },
            { label: 'P_i：平均路由概率（可导）', tone: 'accent' },
          ]}
        />
      }
      controls={
        <>
          <Segmented
            size="sm"
            value={preset}
            onChange={pick}
            options={[
              { value: 'balanced', label: '均衡' },
              { value: 'flat', label: '平坦但同一 argmax' },
              { value: 'collapse', label: '坍缩' },
              { value: 'custom', label: '自定义' },
            ]}
          />
          <Slider className="w-40" label="偏向 E0 的程度 λ" value={lam} min={0} max={2} step={0.05} onChange={(v) => { setLam(v); setPreset('custom') }} format={(v) => v.toFixed(2)} />
          <Slider className="w-40" label="router 锐度" value={sharp} min={0.1} max={4} step={0.05} onChange={(v) => { setSharp(v); setPreset('custom') }} format={(v) => v.toFixed(2)} />
        </>
      }
    >
      <div className="grid items-center gap-5 @3xl:grid-cols-2">
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: W * 1.3, display: 'block', margin: '0 auto' }}>
          <line x1={20} x2={W - 10} y1={sy(0.25)} y2={sy(0.25)} stroke={toneColor('muted')} strokeDasharray="4 4" />
          <text x={W - 12} y={14} textAnchor="end" fontSize={11} fill="var(--muted)">
            虚线 = 均匀 0.25
          </text>
          {Array.from({ length: N }, (_, e) => {
            const cx = 60 + e * 95
            return (
              <g key={e}>
                <rect x={cx - bw - 2} y={sy(f[e])} width={bw} height={Math.max(0, H - 26 - sy(f[e]))} rx={3} fill={toneColor('danger')} />
                <rect x={cx + 2} y={sy(P[e])} width={bw} height={Math.max(0, H - 26 - sy(P[e]))} rx={3} fill={toneColor('accent')} />
                <text x={cx - bw / 2 - 2} y={sy(f[e]) - 4} textAnchor="middle" fontSize={11} fill="var(--muted)" className="font-mono">
                  {f[e].toFixed(2)}
                </text>
                <text x={cx + bw / 2 + 2} y={sy(P[e]) - 4} textAnchor="middle" fontSize={11} fill="var(--muted)" className="font-mono">
                  {P[e].toFixed(2)}
                </text>
                <text x={cx} y={H - 8} textAnchor="middle" fontSize={12} fill="var(--fg)" className="font-mono">
                  E{e}
                </text>
              </g>
            )
          })}
        </svg>
        <div className="min-w-0 space-y-2">
          <div className="grid grid-cols-2 gap-2">
            <Stat label="N·Σ f_i·P_i（Switch）" value={LfP.toFixed(3)} tone={LfP > 1.05 ? 'danger' : 'success'} />
            <Stat label="N·Σ P_i²（只用 P）" value={LP.toFixed(3)} tone={LP > 1.05 ? 'danger' : 'success'} />
          </div>
          <div className="overflow-x-auto rounded-xl border border-border bg-surface">
            <table className="w-full text-center font-mono text-[12px]">
              <thead className="text-muted">
                <tr>
                  <th className="px-2 py-1.5 text-left font-normal">∂L/∂P_i</th>
                  {Array.from({ length: N }, (_, e) => (
                    <th key={e} className="px-2 py-1.5 font-normal">E{e}</th>
                  ))}
                  <th className="px-2 py-1.5 font-normal">区分度</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-t border-border">
                  <td className="px-2 py-1.5 text-left text-muted">f·P：N·f_i</td>
                  {gfP.map((g, e) => (
                    <td key={e} className="px-2 py-1.5">{g.toFixed(2)}</td>
                  ))}
                  <td className="px-2 py-1.5 font-semibold" style={{ color: toneColor('success') }}>{spread(gfP).toFixed(2)}</td>
                </tr>
                <tr className="border-t border-border">
                  <td className="px-2 py-1.5 text-left text-muted">P²：2N·P_i</td>
                  {gP.map((g, e) => (
                    <td key={e} className="px-2 py-1.5">{g.toFixed(2)}</td>
                  ))}
                  <td className="px-2 py-1.5 font-semibold" style={{ color: toneColor('danger') }}>{spread(gP).toFixed(2)}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="text-xs text-muted">「区分度」= 各专家梯度的最大差。梯度越集中在过载专家上，越能精准纠偏；梯度对所有专家几乎一样时，只会把每个 token 的分布压平，不改变 argmax。</div>
        </div>
      </div>
    </VizFrame>
  )
}
