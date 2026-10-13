import { useState } from 'react'
import { Slider } from '@/components/ui'
import { MatrixView, VizFrame } from '@/viz'

export function MemoryViz() {
  const [billions,setBillions]=useState(7),[checkpoint,setCheckpoint]=useState(50)
  const fields=[['低精度权重',2],['梯度',2],['FP32主权重',4],['Adam两个矩',8]] as const
  return <VizFrame title="训练显存不只模型权重" caption={`本例持久状态 ${16*billions} GB；假设原始激活20 GB，保留${checkpoint}%后共 ${(16*billions+20*checkpoint/100).toFixed(1)} GB（未计通信与碎片）。`} controls={<><Slider label="参数量（十亿）" value={billions} min={1} max={20} onChange={setBillions}/><Slider label="激活保留百分比" value={checkpoint} min={10} max={100} step={10} onChange={setCheckpoint}/></>}><div className="grid gap-3 @md:grid-cols-2">{fields.map(([name,bytes])=><div key={name} className="rounded-xl border border-border bg-surface p-3"><p className="text-sm text-muted">{name}</p><p className="text-xl font-semibold text-accent">{bytes*billions} GB</p><div className="mt-2 h-2 rounded bg-surface-2"><div className="h-2 rounded bg-accent" style={{width:`${bytes/8*100}%`}}/></div></div>)}</div></VizFrame>
}
export function ZeroViz() {
  const [ranks,setRanks]=useState(4),[stage,setStage]=useState(2)
  const bytes=[2/(stage>=3?ranks:1),2/(stage>=2?ranks:1),12/(stage>=1?ranks:1)]
  return <VizFrame title="ZeRO 逐步切分持久状态" caption={`每参数每卡 ${(bytes.reduce((a,b)=>a+b,0)).toFixed(2)} 字节，模型假设低精度权重2B、梯度2B、优化器及主权重12B；峰值还需加临时聚合。`} controls={<><Slider label="数据并行卡数" value={ranks} min={1} max={8} onChange={setRanks}/><Slider label="ZeRO阶段" value={stage} min={0} max={3} onChange={setStage}/></>}><div className="min-w-0"><MatrixView data={[bytes]} colLabels={['权重','梯度','优化器']} format={v=>`${v.toFixed(1)} B`}/></div></VizFrame>
}
export function TensorViz() {
  const [parts,setParts]=useState(2)
  const x=[1,2],w1=[[1,2,3,4,5,6],[2,1,0,-1,-2,-3]],w2=[1,2,1,2,1,2]
  const hidden=w1[0].map((_,j)=>Math.max(0,x[0]*w1[0][j]+x[1]*w1[1][j]))
  const pieces=Array.from({length:parts},(_,r)=>hidden.slice(r*6/parts,(r+1)*6/parts).reduce((s,v,i)=>s+v*w2[r*6/parts+i],0))
  return <VizFrame title="列切后行切，输出求和" caption={`输入 [1,2]，ReLU 后隐藏值 [${hidden.join(', ')}]。每个分片输出是部分和；最终结果 ${pieces.reduce((a,b)=>a+b,0)}。`} controls={<Slider label="分片数（1、2、3）" value={parts} min={1} max={3} onChange={setParts}/>}><div className="min-w-0"><MatrixView data={[pieces]} colLabels={pieces.map((_,i)=>`卡${i}`)} format={v=>v.toFixed(0)}/></div></VizFrame>
}
export function PipelineViz() {
  const [p,setP]=useState(3),[m,setM]=useState(4)
  const slots=m+p-1
  return <VizFrame title="微批摊薄首尾气泡" caption={`GPipe前向填充示意：${p}级、${m}微批，${slots}时隙；理想气泡率 ${(100*(p-1)/slots).toFixed(1)}%。完整训练还包括反向，不含通信和负载不均。`} controls={<><Slider label="流水级数" value={p} min={1} max={4} onChange={setP}/><Slider label="微批数" value={m} min={1} max={5} onChange={setM}/></>}><div className="min-w-0"><MatrixView data={Array.from({length:p},(_,r)=>Array.from({length:slots},(_,t)=>t-r>=0&&t-r<m?t-r+1:null))} rowLabels={Array.from({length:p},(_,i)=>`卡${i}`)} colLabels={Array.from({length:slots},(_,i)=>`${i}`)} format={v=>`F${v-1}`}/></div></VizFrame>
}
export function RingViz() {
  const [round,setRound]=useState(0)
  return <VizFrame title="Query 留在本地，KV 在环上传" caption={`第 ${round+1}/4 轮，卡 r 收到来源 (r−${round}) mod 4 的 KV；每卡用 online softmax 合并，4轮后每块恰好访问一次。`} controls={<Slider label="轮次（从0开始）" value={round} min={0} max={3} onChange={setRound}/>}><div className="grid grid-cols-2 gap-3 @xl:grid-cols-4">{[0,1,2,3].map(r=><div key={r} className="rounded-xl border border-border bg-surface p-3"><p className="font-semibold text-fg">设备 {r}</p><p className="mt-2 text-sm text-muted">固定 Q{r}</p><p className="mt-2 rounded bg-surface-2 p-2 font-mono text-accent">KV{(r-round+4)%4}</p><p className="mt-2 text-xs text-muted">已合并 {round+1} / 4 块</p></div>)}</div></VizFrame>
}
