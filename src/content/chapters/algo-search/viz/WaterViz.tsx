/**
 * 接雨水（LC42）与盛最多水的容器（LC11）：同一组柱子，两种对撞指针。
 *  - LC42：lmax < rmax 时第 l 列的水位已确定为 lmax（右边一定有更高的墙兜底）
 *  - LC11：移动短板——以短板为边的容器，宽度只会变小、高度不会超过短板，不可能更大
 */
import { useMemo, useState } from 'react'
import { motion } from 'motion/react'
import { CodeTrace, Legend, StepControls, VizFrame, toneColor, useElementWidth, useStepper } from '@/viz'
import { Segmented } from '@/components/ui'
import { C } from './common'

type Mode = 'trap' | 'container'

const DATA: Record<Mode, number[]> = {
  trap: [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1],
  container: [1, 8, 6, 2, 5, 4, 8, 3, 7],
}

const CODE: Record<Mode, string> = {
  trap: `def trap(height):
    l, r = 0, len(height) - 1
    lmax = rmax = ans = 0
    while l < r:
        lmax = max(lmax, height[l])
        rmax = max(rmax, height[r])
        if lmax < rmax:
            ans += lmax - height[l]   # 第 l 列水位 = lmax
            l += 1
        else:
            ans += rmax - height[r]   # 第 r 列水位 = rmax
            r -= 1
    return ans`,
  container: `def max_area(height):
    l, r = 0, len(height) - 1
    best = 0
    while l < r:
        area = min(height[l], height[r]) * (r - l)
        best = max(best, area)
        if height[l] < height[r]:
            l += 1        # 短板在左，丢掉它
        else:
            r -= 1
    return best`,
}

interface Frame {
  l: number
  r: number
  line: number
  water: number[] // 每列已确定的水量（-1 = 未确定）
  lmax?: number
  rmax?: number
  ans: number
  area?: number
  best?: number
  bestLR?: [number, number]
  settle?: number
  caption: React.ReactNode
}

function buildTrap(h: number[]): Frame[] {
  const fr: Frame[] = []
  const water = h.map(() => -1)
  let l = 0
  let r = h.length - 1
  let lmax = 0
  let rmax = 0
  let ans = 0
  fr.push({ l, r, line: 2, water: [...water], ans, caption: <>两个指针从两端出发，<C>lmax / rmax</C> 记录各自走过的最高柱子。每一步确定<b>一列</b>的水量。</> })
  while (l < r) {
    lmax = Math.max(lmax, h[l])
    rmax = Math.max(rmax, h[r])
    if (lmax < rmax) {
      const w = lmax - h[l]
      water[l] = w
      ans += w
      fr.push({ l, r, lmax, rmax, line: 8, settle: l, water: [...water], ans, caption: <><C>lmax={lmax} &lt; rmax={rmax}</C>：第 {l} 列右边至少有 {rmax} 高的墙，水位只取决于 lmax ⇒ 水量 <C>{lmax} - {h[l]} = {w}</C>，<C>l += 1</C>。</> })
      l++
    } else {
      const w = rmax - h[r]
      water[r] = w
      ans += w
      fr.push({ l, r, lmax, rmax, line: 11, settle: r, water: [...water], ans, caption: <><C>lmax={lmax} ≥ rmax={rmax}</C>：第 {r} 列左边至少有 {lmax} 高的墙，水位只取决于 rmax ⇒ 水量 <C>{rmax} - {h[r]} = {w}</C>，<C>r -= 1</C>。</> })
      r--
    }
  }
  water[l] = 0
  fr.push({ l, r, lmax, rmax, line: 13, water: [...water], ans, caption: <>指针相遇在最高的柱子上（它不存水）。总水量 <C>{ans}</C>，时间 <C>O(n)</C>、额外空间 <C>O(1)</C>。</> })
  return fr
}

function buildContainer(h: number[]): Frame[] {
  const fr: Frame[] = []
  const water = h.map(() => -1)
  let l = 0
  let r = h.length - 1
  let best = 0
  let bestLR: [number, number] = [0, r]
  while (l < r) {
    const area = Math.min(h[l], h[r]) * (r - l)
    if (area > best) {
      best = area
      bestLR = [l, r]
    }
    const short = h[l] < h[r] ? 'l' : 'r'
    fr.push({
      l, r, line: short === 'l' ? 8 : 10, water, ans: 0, area, best, bestLR,
      caption: (
        <>
          面积 <C>min({h[l]}, {h[r]}) × {r - l} = {area}</C>，best = {best}。短板在{short === 'l' ? '左' : '右'}：以它为边的其它容器宽度更小、高度不超过它，<b>不可能更大</b>，丢掉它。
        </>
      ),
    })
    if (short === 'l') l++
    else r--
  }
  fr.push({ l, r, line: 11, water, ans: 0, best, bestLR, caption: <>相遇结束。最大面积 <C>{best}</C>，由下标 <C>({bestLR[0]}, {bestLR[1]})</C> 构成。只看了 n−1 个容器，却没有漏掉最优解。</> })
  return fr
}

function Bars({ h, f, mode }: { h: number[]; f: Frame; mode: Mode }) {
  const [ref, measured] = useElementWidth()
  const width = Math.max(300, measured || 600)
  const n = h.length
  const maxH = Math.max(...h) + 1
  const plotH = 170
  const padT = 10
  const slot = width / n
  const unit = plotH / maxH
  const baseY = padT + plotH
  const y = (v: number) => baseY - v * unit
  const total = baseY + 52
  const ptr = [
    { idx: f.l, label: 'l', tone: 'info' as const },
    { idx: f.r, label: 'r', tone: 'warning' as const },
  ]
  const byIdx = new Map<number, typeof ptr>()
  ptr.forEach((p) => byIdx.set(p.idx, [...(byIdx.get(p.idx) ?? []), p]))

  return (
    <div ref={ref} className="w-full min-w-0">
      <svg width={width} height={total} viewBox={`0 0 ${width} ${total}`} className="block select-none">
        {mode === 'container' && f.area != null && (
          <motion.rect
            initial={false}
            animate={{ x: slot * f.l + slot / 2, width: slot * (f.r - f.l), y: y(Math.min(h[f.l], h[f.r])), height: Math.min(h[f.l], h[f.r]) * unit }}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            fill={toneColor('info')}
            fillOpacity={0.22}
            stroke={toneColor('info')}
            strokeDasharray="4 3"
          />
        )}
        {h.map((v, i) => {
          const w = f.water[i]
          const active = f.settle === i
          return (
            <g key={i}>
              {mode === 'trap' && w > 0 && (
                <rect x={slot * i + 1} width={slot - 2} y={y(v + w)} height={w * unit} fill={toneColor('info')} fillOpacity={active ? 0.75 : 0.45} />
              )}
              <rect
                x={slot * i + slot * 0.12}
                width={slot * 0.76}
                y={y(v)}
                height={Math.max(1, v * unit)}
                rx={3}
                fill={i === f.l || i === f.r ? toneColor(i === f.l ? 'info' : 'warning') : 'var(--surface-3)'}
                fillOpacity={i === f.l || i === f.r ? 0.55 : 1}
                stroke="var(--border-strong)"
              />
              <text x={slot * i + slot / 2} y={baseY + 14} textAnchor="middle" fontSize={11} fill="var(--muted)" className="font-mono">
                {v}
              </text>
              {mode === 'trap' && w >= 0 && (
                <text x={slot * i + slot / 2} y={y(v + Math.max(w, 0)) - 4} textAnchor="middle" fontSize={10.5} fill={toneColor('info')} className="font-mono">
                  {w > 0 ? `+${w}` : ''}
                </text>
              )}
            </g>
          )
        })}
        {mode === 'trap' && f.lmax != null && (
          <line x1={0} x2={slot * (f.l + 1)} y1={y(f.lmax)} y2={y(f.lmax)} stroke={toneColor('info')} strokeWidth={1.5} strokeDasharray="5 4" />
        )}
        {mode === 'trap' && f.rmax != null && (
          <line x1={slot * f.r} x2={width} y1={y(f.rmax)} y2={y(f.rmax)} stroke={toneColor('warning')} strokeWidth={1.5} strokeDasharray="5 4" />
        )}
        <line x1={0} x2={width} y1={baseY} y2={baseY} stroke="var(--border-strong)" />
        {[...byIdx.entries()].map(([idx, g]) => (
          <motion.g key={g.map((p) => p.label).join('|')} initial={false} animate={{ x: slot * idx + slot / 2 }} transition={{ type: 'spring', stiffness: 380, damping: 32 }}>
            <text y={baseY + 30} textAnchor="middle" fontSize={12} fill={toneColor(g[0].tone)}>
              ▲
            </text>
            <text y={baseY + 44} textAnchor="middle" fontSize={11} fontWeight={600} fill={toneColor(g[0].tone)} className="font-mono">
              {g.map((p) => p.label).join(',')}
            </text>
          </motion.g>
        ))}
      </svg>
    </div>
  )
}

export function WaterViz() {
  const [mode, setMode] = useState<Mode>('trap')
  const h = DATA[mode]
  const frames = useMemo(() => (mode === 'trap' ? buildTrap(h) : buildContainer(h)), [mode, h])
  const st = useStepper(frames, { intervalMs: 1000 })
  const f = st.frame

  return (
    <VizFrame
      title={mode === 'trap' ? '接雨水：较矮的一侧水位已确定' : '盛最多水的容器：每次丢掉短板'}
      caption={f.caption}
      legend={
        <Legend
          items={
            mode === 'trap'
              ? [
                  { label: '已确定的水', tone: 'info', variant: 'soft' },
                  { label: 'lmax（左侧虚线）', tone: 'info', variant: 'line' },
                  { label: 'rmax（右侧虚线）', tone: 'warning', variant: 'line' },
                ]
              : [
                  { label: '当前容器', tone: 'info', variant: 'soft' },
                  { label: 'l / r 柱', tone: 'warning', variant: 'soft' },
                ]
          }
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Segmented
            size="sm"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'trap', label: 'LC42 接雨水' },
              { value: 'container', label: 'LC11 盛水容器' },
            ]}
          />
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="min-w-0">
          <div className="mb-1 text-xs text-muted">
            height = <span className="font-mono">[{h.join(', ')}]</span>
          </div>
          <Bars h={h} f={f} mode={mode} />
        </div>
        <CodeTrace
          code={CODE[mode]}
          active={f.line}
          vars={mode === 'trap' ? { l: f.l, r: f.r, lmax: f.lmax ?? 0, rmax: f.rmax ?? 0, ans: f.ans } : { l: f.l, r: f.r, area: f.area ?? '—', best: f.best ?? 0 }}
        />
      </div>
    </VizFrame>
  )
}
