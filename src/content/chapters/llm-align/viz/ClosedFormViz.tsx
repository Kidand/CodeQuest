/**
 * KL 约束奖励最大化的闭式解：π*(y) ∝ π_ref(y)·exp(r(y)/β)。
 * 拖动 β：β→0 退化为“只选奖励最高的回答”，β→∞ 退回 π_ref。
 * 点击某个回答，验证“反解奖励”恒等式 r = β·log(π* ÷ π_ref) + β·log Z。
 */
import { useMemo, useState } from 'react'
import { motion } from 'motion/react'
import { Legend, VizFrame, toneColor, toneSoft } from '@/viz'
import { Slider } from '@/components/ui'
import { Stat, f3 } from './plot'

const RESP = [
  { label: 'A 正确且详细', ref: 0.1, r: 2.0 },
  { label: 'B 正确但简短', ref: 0.3, r: 1.5 },
  { label: 'C 啰嗦', ref: 0.25, r: 0.5 },
  { label: 'D 跑题', ref: 0.15, r: -0.5 },
  { label: 'E 拒答', ref: 0.15, r: -1.0 },
  { label: 'F 有害', ref: 0.05, r: -3.0 },
]

export function ClosedFormViz() {
  const [lb, setLb] = useState(0) // log10 β
  const [sel, setSel] = useState(0)
  const beta = 10 ** lb

  const s = useMemo(() => {
    const logw = RESP.map((y) => Math.log(y.ref) + y.r / beta)
    const m = Math.max(...logw)
    const logZ = m + Math.log(logw.reduce((a, v) => a + Math.exp(v - m), 0))
    const pi = logw.map((v) => Math.exp(v - logZ))
    const er = pi.reduce((a, p, i) => a + p * RESP[i].r, 0)
    const kl = pi.reduce((a, p, i) => a + (p > 0 ? p * Math.log(p / RESP[i].ref) : 0), 0)
    return { pi, logZ, er, kl, J: er - beta * kl }
  }, [beta])

  const y = RESP[sel]
  const lhs = beta * Math.log(s.pi[sel] / y.ref)
  const regime = beta < 0.2 ? '几乎把全部概率压到奖励最高的 A 上（接近贪心 argmax，KL 很大）' : beta > 3 ? '几乎不动 π_ref：KL 惩罚太重，奖励信号被“冻结”' : '在“追奖励”与“别离 π_ref 太远”之间折中'

  return (
    <VizFrame
      title="闭式解：β 决定离 π_ref 多远"
      caption={
        <>
          β = <b className="font-mono">{beta.toFixed(2)}</b>：最优策略{regime}。点中的回答 <b>{y.label}</b>：
          <span className="font-mono">
            {' '}
            β·log(π*/π_ref) + β·log Z = {f3(lhs)} + {f3(beta * s.logZ)} = {f3(lhs + beta * s.logZ)}
          </span>
          ，恰好等于它的奖励 r = <b className="font-mono">{y.r.toFixed(1)}</b>。
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'π_ref（参考策略）', tone: 'muted', variant: 'soft' },
            { label: 'π*（闭式最优解）', tone: 'accent' },
            { label: '点击行查看反解奖励', tone: 'warning', variant: 'soft' },
          ]}
        />
      }
      controls={
        <Slider
          className="min-w-[220px] flex-1"
          label="KL 系数 β（对数刻度）"
          value={lb}
          min={-1.3}
          max={1}
          step={0.01}
          onChange={setLb}
          format={(v) => (10 ** v).toFixed(2)}
        />
      }
    >
      <div className="grid gap-5 @3xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <div className="flex flex-col gap-1.5">
          {RESP.map((it, i) => (
            <button
              key={it.label}
              onClick={() => setSel(i)}
              className="flex items-center gap-2 rounded-lg px-1 py-0.5 text-left text-[13px] transition-colors"
              style={{ background: i === sel ? toneSoft('warning', 16) : undefined }}
            >
              <span className="w-[92px] shrink-0 truncate text-fg">{it.label}</span>
              <span className="w-11 shrink-0 text-right font-mono text-xs tabular-nums" style={{ color: toneColor(it.r >= 0 ? 'success' : 'danger') }}>
                r={it.r > 0 ? '+' : ''}
                {it.r.toFixed(1)}
              </span>
              <div className="relative h-5 min-w-0 flex-1 overflow-hidden rounded-md bg-surface-2">
                <div className="absolute inset-y-0 left-0 rounded-md" style={{ width: `${it.ref * 100}%`, background: toneSoft('muted', 65) }} />
                <motion.div
                  className="absolute left-0 top-1 bottom-1 rounded"
                  initial={false}
                  animate={{ width: `${s.pi[i] * 100}%` }}
                  transition={{ type: 'spring', stiffness: 260, damping: 30 }}
                  style={{ background: toneColor('accent') }}
                />
              </div>
              <span className="w-[86px] shrink-0 text-right font-mono text-[11.5px] text-muted tabular-nums">
                {it.ref.toFixed(2)}→<b className="text-fg">{s.pi[i].toFixed(3)}</b>
              </span>
            </button>
          ))}
        </div>
        <div className="grid grid-cols-2 content-start gap-2">
          <Stat label="期望奖励 E_π*[r]" value={f3(s.er)} tone="success" />
          <Stat label="KL(π* ‖ π_ref)" value={f3(s.kl)} tone="warning" />
          <Stat label="目标 J = E[r] − β·KL" value={f3(s.J)} />
          <Stat label="β·log Z(x)" value={f3(beta * s.logZ)} sub="恒等于最优目标值 J" />
        </div>
      </div>
    </VizFrame>
  )
}
