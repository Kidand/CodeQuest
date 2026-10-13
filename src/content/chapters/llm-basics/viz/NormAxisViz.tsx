/**
 * BatchNorm 与 LayerNorm 的区别只在“沿哪个轴求统计量”。
 * 点击任意格子：LN 用同一行（同一个 token 的全部通道），BN 用同一列（batch 内全部 token 的同一通道）。
 * 打开 padding：BN 的统计量被 padding 污染，LN 对真实 token 毫无影响。
 */
import { useMemo, useState } from 'react'
import { Legend, MatrixView, VizFrame } from '@/viz'
import { Segmented, Switch } from '@/components/ui'
import { rng } from '@/lib/utils'
import { C, Stat } from './plot'

const ROWS = ['A·t0', 'A·t1', 'A·t2', 'B·t0', 'B·t1', 'B·t2']
const D = 8
type Mode = 'ln' | 'bn'

function makeData() {
  const r = rng(11)
  const g = () => {
    const u = Math.max(1e-9, r())
    const v = r()
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
  }
  // 每个通道有自己的尺度与偏置，每个 token 有自己的整体幅度（模拟真实激活）
  const chBias = Array.from({ length: D }, () => g() * 0.8)
  return ROWS.map((_, i) => Array.from({ length: D }, (_, c) => Number((chBias[c] + g() * (0.6 + 0.25 * i)).toFixed(2))))
}
const DATA = makeData()

export function NormAxisViz() {
  const [mode, setMode] = useState<Mode>('ln')
  const [pad, setPad] = useState(false)
  const [sel, setSel] = useState<[number, number]>([1, 2])
  const data = useMemo(() => DATA.map((row, i) => (pad && i === 5 ? row.map(() => 0) : row)), [pad])
  const inGroup = (r: number, c: number) => (mode === 'ln' ? r === sel[0] : c === sel[1])
  const group = data.flatMap((row, r) => row.filter((_, c) => inGroup(r, c)))
  const mu = group.reduce((a, b) => a + b, 0) / group.length
  const sd = Math.sqrt(group.reduce((a, b) => a + (b - mu) ** 2, 0) / group.length)
  const v = data[sel[0]][sel[1]]
  const z = (v - mu) / Math.sqrt(sd * sd + 1e-5)
  // 对照：如果没有 padding，这个格子的 BN 输出是多少
  const clean = DATA.map((row) => row[sel[1]])
  const cmu = clean.reduce((a, b) => a + b, 0) / clean.length
  const csd = Math.sqrt(clean.reduce((a, b) => a + (b - cmu) ** 2, 0) / clean.length)
  const zClean = (DATA[sel[0]][sel[1]] - cmu) / Math.sqrt(csd * csd + 1e-5)
  const padSel = pad && sel[0] === 5

  return (
    <VizFrame
      title="LayerNorm 按行（每个 token 自己），BatchNorm 按列（跨 batch）"
      caption={
        <>
          选中 <C>{ROWS[sel[0]]}</C> 的通道 <C>{sel[1]}</C>（值 {v.toFixed(2)}）。{mode === 'ln' ? 'LayerNorm' : 'BatchNorm'} 用高亮的 {group.length} 个数求 μ = {mu.toFixed(2)}、σ = {sd.toFixed(2)}，
          归一化结果 {z.toFixed(2)}。
          {mode === 'bn' && pad && !padSel && ` 没有 padding 时这里应为 ${zClean.toFixed(2)}——同 batch 的 padding 改变了真实 token 的输出。`}
          {mode === 'ln' && pad && ' LayerNorm 只看本行，padding 再多也不影响真实 token。'}
          {' '}点击任意格子换一个位置。
        </>
      }
      legend={<Legend items={[{ label: '参与统计的元素', tone: 'accent', variant: 'soft' }, { label: '选中的元素', tone: 'gold', variant: 'soft' }]} />}
      controls={
        <>
          <Segmented
            size="sm"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'ln', label: 'LayerNorm（行）' },
              { value: 'bn', label: 'BatchNorm（列）' },
            ]}
          />
          <Switch checked={pad} onChange={setPad} label="B·t2 是 padding（全 0）" />
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-[1fr_220px]">
        <div className="min-w-0">
          <div className="mb-2 text-xs font-medium text-muted">激活矩阵：行 = token（2 条序列 × 3 个位置），列 = 隐藏维（通道）</div>
          <MatrixView
            data={data}
            rowLabels={ROWS}
            colLabels={Array.from({ length: D }, (_, c) => `c${c}`)}
            scale="none"
            cellSize={44}
            format={(x) => (Math.abs(x) < 0.05 ? '0.0' : x.toFixed(1))}
            highlight={(r, c) => (r === sel[0] && c === sel[1] ? 'gold' : inGroup(r, c) ? 'accent' : null)}
            onCellClick={(r, c) => setSel([r, c])}
          />
        </div>
        <div className="grid grid-cols-2 gap-2 @3xl:grid-cols-1">
          <Stat label="统计量个数" value={group.length} sub={mode === 'ln' ? '= 隐藏维 d' : '= B × T'} />
          <Stat label="μ / σ" value={`${mu.toFixed(2)} / ${sd.toFixed(2)}`} />
          <Stat label="归一化后" value={z.toFixed(2)} tone="accent" />
          <Stat label="推理时" value={mode === 'ln' ? '与训练相同' : '用滑动平均'} tone={mode === 'ln' ? 'success' : 'warning'} />
        </div>
      </div>
    </VizFrame>
  )
}
