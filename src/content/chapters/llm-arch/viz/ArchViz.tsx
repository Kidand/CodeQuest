import { useState } from 'react'
import { Slider } from '@/components/ui'
import { VizFrame, MatrixView, ProbBars } from '@/viz'
export function RopeViz(){
 const [m,M]=useState(2),[n,N]=useState(5)
 const q=[1,0],k=[.6,.8],rot=(x:number[],p:number)=>[x[0]*Math.cos(p)-x[1]*Math.sin(p),x[0]*Math.sin(p)+x[1]*Math.cos(p)]
 const a=rot(q,m),b=rot(k,n),dot=a[0]*b[0]+a[1]*b[1]
 return <VizFrame title="共同平移保持旋转内积" caption={`q 与 k 的位置分别为 ${m}、${n}，差值 ${n-m}；旋转后内积 ${dot.toFixed(4)}。两个位置一起加一，内积保持。此图只展示频率为 1 的一对维度。`} controls={<><Slider label="q 位置 m" value={m} min={0} max={10} onChange={M}/><Slider label="k 位置 n" value={n} min={0} max={10} onChange={N}/></>}><MatrixView data={[q,k,a,b]} rowLabels={['q','k','R(m)q','R(n)k']} colLabels={['x','y']} scale="div"/></VizFrame>
}
export function FrequencyViz(){
 const [s,S]=useState(4),[d,D]=useState(8)
 const base=10000,f=Array.from({length:d/2},(_,i)=>base**(-2*i/d)),ntk=base*s**(d/(d-2))
 return <VizFrame title="插值与改 base 影响不同频带" caption="PI 将全部频率除以 s；静态 NTK-aware 改变 base，高频首维保持，最低频按 s 缩小。图中数值是角频率，不是周期。" controls={<><Slider label="扩展倍数 s" value={s} min={1} max={16} onChange={S}/><Slider label="旋转维度 d" value={d} min={4} max={16} step={2} onChange={D}/></>}><MatrixView data={[f,f.map(x=>x/s),f.map((_,i)=>ntk**(-2*i/d))]} rowLabels={['原始','PI','NTK']} colLabels={f.map((_,i)=>String(i))} format={x=>x.toExponential(0).replace('+','')}/></VizFrame>
}
export function BlockViz(){
 const [gate,G]=useState(.5),[scale,S]=useState(1)
 const x=[1,-1,.5,-.5],rms=Math.sqrt(x.reduce((s,x)=>s+x*x,0)/4+1e-5),a=x.map((v,i)=>v+scale*(i%2?-.2:.2)),out=a.map((v,i)=>v+gate*Math.tanh(x[i]/rms))
 return <VizFrame title="残差支路保留输入通道" caption="此图用小型确定函数代替注意力和 FFN，突出 x+子层输出 的连接关系；滑块为支路输出缩放，均为 0 时严格回到 x。" controls={<><Slider label="注意力支路缩放" value={scale} min={0} max={2} step={.1} onChange={S}/><Slider label="FFN 支路缩放" value={gate} min={0} max={2} step={.1} onChange={G}/></>}><MatrixView data={[x,a,out]} rowLabels={['输入 x','加 Attention','加 FFN']} scale="div"/></VizFrame>
}
export function ComputeViz(){
 const [power,R]=useState(2),[f,F]=useState(3);const r=2**power
 const d=1024,params=[2*d*d+2*d*d/r,3*d*(d*f),2*d]
 return <VizFrame title="GQA 只缩减 K 与 V 投影" caption={`d=1024，Q/KV 头数比 ${r}，FFN 中间宽度 ${f}d。每层不含 bias 参数 ${(params.reduce((a,b)=>a+b,0)/1e6).toFixed(3)} M。`} controls={<><Slider label="Q/KV 头数比" value={power} min={0} max={3} format={v=>2**v} onChange={R}/><Slider label="FFN 宽度倍数" value={f} min={2} max={4} step={.5} onChange={F}/></>}><ProbBars items={params.map((p,i)=>({label:['Attention','SwiGLU','两次 Norm'][i],p,tone:i===0?'accent':'info'}))} format={x=>(x/1e6).toFixed(3)+'M'}/></VizFrame>
}
export function SinkhornViz(){
 const [steps,S]=useState(3);let a=[[5,1,2],[2,6,1],[1,2,3]].map(r=>[...r])
 for(let k=0;k<steps;k++){a=a.map(r=>{const sum=r.reduce((s,v)=>s+v,0);return r.map(v=>v/sum)});const sums=a[0].map((_,c)=>a.reduce((s,r)=>s+r[c],0));a=a.map(r=>r.map((v,c)=>v/sums[c]))}
 const sums=a.map(r=>r.reduce((s,v)=>s+v,0))
 return <VizFrame title="交替归一化逼近双随机矩阵" caption={`每轮先归一化行、再归一化列。当前行和 ${sums.map(x=>x.toFixed(4)).join(' / ')}；最后一步保证列和为 1，有限轮行和只是近似。`} controls={<Slider label="Sinkhorn 迭代次数" value={steps} min={0} max={12} onChange={S}/>}><MatrixView data={a} rowLabels={['流 0','流 1','流 2']} colLabels={['流 0','流 1','流 2']}/></VizFrame>
}
