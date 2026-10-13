import { useState } from 'react'
import { Slider } from '@/components/ui'
import { MatrixView, ProbBars, VizFrame } from '@/viz'

export function WindowViz() {
  const [window, setWindow] = useState(3)
  const [depth, setDepth] = useState(1)
  const reach = 1 + depth * (window - 1)
  return <VizFrame title="多层窗口逐步传递信息" caption={`每层含自己共 ${window} 个位置；${depth} 层理论可达 ${reach} 个位置（图中受序列起点截断）。可达不等于无损记忆。`} controls={<><Slider label="窗口宽度" value={window} min={1} max={5} onChange={setWindow}/><Slider label="层数" value={depth} min={1} max={3} onChange={setDepth}/></>}><div className="min-w-0"><MatrixView data={Array.from({length:8}, (_,i)=>Array.from({length:8},(_,j)=>j<=i && i-j<reach ? 1 : 0))} masked={(i,j)=>j>i||i-j>=reach} rowLabels={Array.from({length:8},(_,i)=>`q${i}`)} colLabels={Array.from({length:8},(_,i)=>`${i}`)} format={()=>'可达'}/></div></VizFrame>
}
export function SinkViz() {
  const [sink, setSink] = useState(0)
  const logits=[1,2,0], m=Math.max(sink,...logits), z=Math.exp(sink-m)+logits.reduce((s,x)=>s+Math.exp(x-m),0)
  return <VizFrame title="空槽吸收概率而不输出值" caption={`真实 value 的权重总和为 ${(1-Math.exp(sink-m)/z).toFixed(3)}；sink 的 value 为零，但它仍参与分母。`} controls={<Slider label="sink logit" value={sink} min={-4} max={6} step={0.5} onChange={setSink}/>}><ProbBars items={[...logits.map((x,i)=>({label:`token ${i}`,p:Math.exp(x-m)/z})),{label:'sink（零值）',p:Math.exp(sink-m)/z,tone:'warning'}]}/></VizFrame>
}
export function SparseViz() {
  const [k,setK]=useState(3)
  const scores=[1,5,2,0,4,3,0,6], ids=scores.map((s,i)=>({s,i})).sort((a,b)=>b.s-a.s||a.i-b.i).slice(0,k).map(x=>x.i)
  return <VizFrame title="选择减少读取，不删除缓存" caption={`indexer 扫描 8 个候选，主注意力读取 ${k} 个。下一次 query 可以选择完全不同的位置。`} controls={<Slider label="top-k" value={k} min={1} max={8} onChange={setK}/>}><div className="min-w-0"><MatrixView data={[scores]} colLabels={scores.map((_,i)=>`${i}`)} highlight={(_,i)=>ids.includes(i)?'success':null}/><p className="mt-4 text-sm text-muted">按索引升序读取：{[...ids].sort((a,b)=>a-b).join('、')}</p></div></VizFrame>
}
export function CompressionViz() {
  const [temperature,setTemperature]=useState(1)
  const x=[1,3,7,9],z=[0,1,2,0].map(v=>v/temperature),m=Math.max(...z),e=z.map(v=>Math.exp(v-m)),p=e.map(v=>v/e.reduce((a,b)=>a+b,0)),out=p.reduce((s,v,i)=>s+v*x[i],0)
  return <VizFrame title="压缩是学习到的加权摘要" caption={`四个数 [1,3,7,9] 压缩为 ${out.toFixed(3)}。温度降低时更偏向 logit 最大的位置，但丢失的信息无法从摘要中完整恢复。`} controls={<Slider label="权重温度" value={temperature} min={0.2} max={4} step={0.2} onChange={setTemperature}/>}><ProbBars items={p.map((v,i)=>({label:`值 ${x[i]}`,p:v}))}/></VizFrame>
}
export function DeltaViz() {
  const [alpha,setAlpha]=useState(0.8),[beta,setBeta]=useState(1)
  const values=[1,1,-1,-1,2,2];let state=0
  const states=values.map(v=>{state=alpha*state+beta*(v-alpha*state);return state})
  return <VizFrame title="先遗忘，再修正预测误差" caption={`一维 k=q=1：S ← αS + β(v−αS)。β=1 时覆盖为当前值；β=0 时仅遗忘。`} controls={<><Slider label="衰减 α" value={alpha} min={0} max={1} step={0.1} onChange={setAlpha}/><Slider label="更新 β" value={beta} min={0} max={1} step={0.1} onChange={setBeta}/></>}><div className="min-w-0"><MatrixView data={[values,states]} rowLabels={['输入','状态']} colLabels={values.map((_,i)=>`${i}`)} scale="div"/></div></VizFrame>
}
