/**
 * MHA → GQA → MQA：8 个 query 头共享 n_kv 个 KV 头；q 头 i 使用 KV 头 i // (n_q / n_kv)。
 * 切到 np.tile 可看到错误的配对（与按 repeat_interleave 训练出的权重布局不符）。
 */
import { useState } from 'react'
import { Legend, VizFrame, catColor, toneColor } from '@/viz'
import { Segmented, Switch } from '@/components/ui'

const NQ = 8

export function GQAGroupingViz() {
  const [nkv, setNkv] = useState(2)
  const [sel, setSel] = useState(5)
  const [tile, setTile] = useState(false)
  const rep = NQ / nkv
  const kvOf = (i: number) => (tile ? i % nkv : Math.floor(i / rep))
  const rightKv = (i: number) => Math.floor(i / rep)
  const qx = (i: number) => 30 + i * 48
  const kvx = (j: number) => 30 + (j + 0.5) * (NQ / nkv) * 48 - 24
  const name = nkv === NQ ? 'MHA' : nkv === 1 ? 'MQA' : `GQA-${nkv}`
  const expanded = Array.from({ length: NQ }, (_, i) => kvOf(i))

  return (
    <VizFrame
      title="GQA：每组 query 头共享一份 K/V"
      caption={
        <>
          <b>{name}</b>：{NQ} 个 Q 头、{nkv} 个 KV 头，每个 KV 头服务 {rep} 个 Q 头；KV cache 与 W_K/W_V 参数都只有 MHA 的 <b>{((nkv / NQ) * 100).toFixed(1)}%</b>。
          选中的 Q 头 {sel} 使用 KV 头 <b>{kvOf(sel)}</b>
          {tile && kvOf(sel) !== rightKv(sel) ? (
            <span className="text-danger">（错误！按 repeat_interleave 的约定应为 {rightKv(sel)}）</span>
          ) : (
            '（= i // n_rep）'
          )}
          。repeat_kv 后的 KV 头序列：<span className="font-mono">[{expanded.join(', ')}]</span>
        </>
      }
      legend={<Legend items={[{ label: '不同颜色 = 不同 KV 组', tone: 1 }, { label: '错误配对', tone: 'danger', variant: 'line' }]} />}
      controls={
        <>
          <Segmented
            size="sm"
            value={nkv}
            onChange={setNkv}
            options={[
              { value: 8, label: 'MHA (8)' },
              { value: 4, label: 'GQA-4' },
              { value: 2, label: 'GQA-2' },
              { value: 1, label: 'MQA (1)' },
            ]}
          />
          <Switch checked={tile} onChange={setTile} label="用 np.tile 展开（错误）" />
        </>
      }
    >
      <svg viewBox="0 0 400 180" width="100%" className="mx-auto block max-w-[560px]">
        {Array.from({ length: NQ }, (_, i) => {
          const j = kvOf(i)
          const wrong = j !== rightKv(i)
          return (
            <line
              key={`l${i}`}
              x1={qx(i)}
              y1={48}
              x2={kvx(j)}
              y2={128}
              stroke={wrong ? toneColor('danger') : catColor(j)}
              strokeWidth={i === sel ? 3 : 1.5}
              strokeDasharray={wrong ? '4 3' : undefined}
              opacity={i === sel ? 1 : 0.55}
            />
          )
        })}
        {Array.from({ length: NQ }, (_, i) => (
          <g key={`q${i}`} onClick={() => setSel(i)} className="cursor-pointer">
            <rect x={qx(i) - 18} y={14} width={36} height={34} rx={8} fill={`color-mix(in oklab, ${catColor(rightKv(i))} 22%, var(--surface))`} stroke={i === sel ? 'var(--fg)' : catColor(rightKv(i))} strokeWidth={i === sel ? 2.5 : 1.5} />
            <text x={qx(i)} y={36} textAnchor="middle" fontSize={13} fill="var(--fg)" fontFamily="var(--font-mono)">
              Q{i}
            </text>
          </g>
        ))}
        {Array.from({ length: nkv }, (_, j) => {
          const w = Math.max(36, (NQ / nkv) * 48 - 12)
          return (
            <g key={`kv${j}`}>
              <rect x={kvx(j) - w / 2} y={128} width={w} height={34} rx={8} fill={`color-mix(in oklab, ${catColor(j)} 30%, var(--surface))`} stroke={catColor(j)} strokeWidth={1.5} />
              <text x={kvx(j)} y={150} textAnchor="middle" fontSize={13} fill="var(--fg)" fontFamily="var(--font-mono)">
                KV{j}
              </text>
            </g>
          )
        })}
        <text x={4} y={10} fontSize={10} fill="var(--subtle)">
          Query 头（点击选择）
        </text>
        <text x={4} y={176} fontSize={10} fill="var(--subtle)">
          KV 头（缓存的就是它们）
        </text>
      </svg>
    </VizFrame>
  )
}
