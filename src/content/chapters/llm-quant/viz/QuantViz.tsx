import { useState } from 'react'
import { Slider } from '@/components/ui'
import { MatrixView, ProbBars, VizFrame } from '@/viz'
export function LoraViz(){
 const [r,R]=useState(4),[alpha,A]=useState(8)
 const d=1024,params=2*d*r
 return <VizFrame title="低秩更新减少可训练参数" caption={`方阵 W 为 1024×1024；r=${r} 时 A/B 共有 ${params.toLocaleString()} 参数，是原矩阵的 ${(100*params/d**2).toFixed(2)}%；更新缩放 α/r=${(alpha/r).toFixed(2)}。此图展示参数账本，不等价于训练显存比例。`} controls={<><Slider label="rank r" value={r} min={1} max={32} onChange={R}/><Slider label="alpha" value={alpha} min={1} max={32} onChange={A}/></>}><ProbBars items={[{label:'冻结 W',p:d*d,tone:'muted'},{label:'训练 A/B',p:params,tone:'success'}]} format={x=>x.toLocaleString()}/></VizFrame>
}
export function QuantizationViz(){
 const [bits,B]=useState(4),[g,G]=useState(4),[outlier,O]=useState(4)
 const x=[-.7,-.2,.1,.6,-.3,.2,.8,outlier],qmax=2**(bits-1)-1
 const y=x.map((v,i)=>{const start=Math.floor(i/g)*g,block=x.slice(start,start+g),scale=Math.max(...block.map(Math.abs))/qmax||1;return Math.round(v/scale)*scale})
 const mse=x.reduce((sum,v,i)=>sum+(v-y[i])**2,0)/x.length
 return <VizFrame title="离群值扩大整组量化步长" caption={`对称范围 [-${qmax},${qmax}]，组大小 ${g}，均方误差 ${mse.toFixed(5)}。组越小通常误差越低，但 scale 元数据更多。`} controls={<><Slider label="位宽" value={bits} min={4} max={8} step={4} onChange={B}/><Slider label="组大小" value={g} min={1} max={8} onChange={G}/><Slider label="最后一个值" value={outlier} min={1} max={12} onChange={O}/></>}><MatrixView data={[x,y,x.map((v,i)=>y[i]-v)]} rowLabels={['原值','反量化','误差']} scale="div" format={v=>Math.abs(v)<1e-8?'0':String(Number(v.toFixed(2)))}/></VizFrame>
}
export function SmoothViz(){
 const [alpha,A]=useState(.5),x=[1,16,2,1],w=[2,.5,2,1],s=x.map((v,i)=>v**alpha/w[i]**(1-alpha)),xp=x.map((v,i)=>v/s[i]),wp=w.map((v,i)=>v*s[i])
 return <VizFrame title="等价缩放把难度移向权重" caption={`Σ xᵢwᵢ=${x.reduce((sum,v,i)=>sum+v*w[i],0)}；缩放后仍为 ${xp.reduce((sum,v,i)=>sum+v*wp[i],0).toFixed(4)}。alpha 越大越着重压低激活范围，但权重范围可能变大。`} controls={<Slider label="迁移因子 alpha" value={alpha} min={0} max={1} step={.05} onChange={A}/>}><MatrixView data={[x,w,s,xp,wp]} rowLabels={['X','W','scale','X / s','s × W']} scale="seq"/></VizFrame>
}
export function Fp4Viz(){
 const [m,M]=useState(6);const levels=[0,.5,1,1.5,2,3,4,6],scale=2**Math.ceil(Math.log2(m/6)),x=[0,.2,.6,1,1.8,2.5,4,m],y=x.map(v=>levels.reduce((best,n)=>Math.abs(n-v/scale)<Math.abs(best-v/scale)?n:best)*scale)
 return <VizFrame title="FP4 数值间距不是均匀的" caption={`教学 MXFP4：s=2^ceil(log₂(max/6))=${scale}，先除 scale，再映射到 E2M1 的非均匀值。这里展示正数侧，负数对称。`} controls={<Slider label="块中最大幅度" value={m} min={6} max={24} onChange={M}/>}><MatrixView data={[x,y,x.map((v,i)=>y[i]-v)]} rowLabels={['原值','FP4 恢复','误差']} scale="div"/><p className="mt-4 text-sm text-muted">E2M1 非负值：0、0.5、1、1.5、2、3、4、6。scale 取 2 的幂，因此最大值跨越边界会让整块网格跳变。</p></VizFrame>
}
