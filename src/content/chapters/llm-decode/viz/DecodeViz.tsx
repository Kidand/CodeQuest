import { useState } from 'react'
import { Slider, Button } from '@/components/ui'
import { VizFrame, ProbBars, TokenRow, MatrixView } from '@/viz'
export function SamplingViz(){
 const [t,T]=useState(1),[p,P]=useState(.8),[k,K]=useState(5)
 const logits=[2.4,1.8,1,.5,-.2,-1],e=logits.map(x=>Math.exp((x-2.4)/t)),sum=e.slice(0,k).reduce((a,b)=>a+b,0)
 let c=0;const kept=e.map((x,i)=>{const keep=i<k&&c<p;if(i<k)c+=x/sum;return keep}),z=e.reduce((s,x,i)=>s+(kept[i]?x:0),0)
 return <VizFrame title="越过阈值的 token 仍保留" caption="先温度，再 top-k；top-p 按过滤后的概率累计。观察首个跨过 p 的词始终留下。" controls={<><Slider label="温度" value={t} min={.2} max={2} step={.1} onChange={T}/><Slider label="top-p" value={p} min={.05} max={1} step={.05} onChange={P}/><Slider label="top-k" value={k} min={1} max={6} onChange={K}/></>}><ProbBars items={e.map((x,i)=>({label:['我','你','他','我们','猫','EOS'][i],p:kept[i]?x/z:0,dim:!kept[i],tone:kept[i]?'success':'muted'}))}/></VizFrame>
}
export function BpeViz(){
 const [step,S]=useState(0);const rows=[['l','o','w','e','r'],['lo','w','e','r'],['low','e','r'],['low','er']]
 return <VizFrame title="BPE 按训练好的秩合并" caption={['初始字符序列。训练统计频率，编码阶段则读取固定 merge rank。','合并 rank 0: l + o → lo。','合并 rank 1: lo + w → low。','合并 rank 2: e + r → er，得到两个 token。'][step]} controls={<><Button size="sm" onClick={()=>S(Math.max(0,step-1))}>上一步</Button><Button size="sm" onClick={()=>S(Math.min(3,step+1))}>下一步</Button><Button size="sm" variant="secondary" onClick={()=>S(0)}>重置</Button></>}><TokenRow tokens={rows[step].map(text=>({text,tone:'accent'}))}/><p className="mt-4 text-sm text-muted">固定合并表：(l,o) → (lo,w) → (e,r)。将当前位置的相邻对与表中的 rank 比较，而非重新统计当前句子的词频。</p></VizFrame>
}
export function BeamViz(){
 const [width,W]=useState(1),[alpha,A]=useState(0)
 const paths=[{s:'A → EOS',p:.36,n:2},{s:'B → C → EOS',p:.35,n:3},{s:'A → D → EOS',p:.24,n:3}]
 const visible=width===1?[paths[0]]:paths.slice(0,2)
 const ranked=[...visible].sort((a,b)=>Math.log(b.p)/b.n**alpha-Math.log(a.p)/a.n**alpha)
 return <VizFrame title="束宽与长度惩罚改变搜索" caption={`示例初始 P(A)=0.6、P(B)=0.4；宽度 1 丢弃 B。最终得分用 log P / 长度^α，最优为 ${ranked[0].s}。`} controls={<><Slider label="束宽" value={width} min={1} max={2} onChange={W}/><Slider label="长度惩罚 α" value={alpha} min={0} max={2} step={.1} onChange={A}/></>}><div className="space-y-2">{ranked.map((x,i)=><div key={x.s} className="rounded-lg border border-border bg-surface-2 p-3 text-sm"><span className={i===0?'text-success':'text-fg'}>{x.s}</span><span className="float-right font-mono">{(Math.log(x.p)/x.n**alpha).toFixed(3)}</span></div>)}</div><p className="mt-3 text-xs text-muted">B 的后续 C、EOS 条件概率之积为 0.875；A→EOS 为 0.6，A→D→EOS 为 0.4。A→D 分支概率 .24，在宽度 1 或 2 时都被剪掉。</p></VizFrame>
}
export function ConstraintViz(){
 const [state,S]=useState(0);const names=['开始','已读 0','已读 1','完成'],v=['0','1','EOS'],edges=[[1,2,-1],[-1,2,3],[1,-1,3],[-1,-1,-1]]
 return <VizFrame title="先约束合法集合再归一化" caption={`当前状态：${names[state]}。示例语言为相邻数字不能相同的非空二进制串；只有合法前缀才能抵达结束。`} controls={<><Button onClick={()=>S(0)} size="sm" variant="secondary">重置</Button>{v.map((x,i)=><Button key={x} size="sm" disabled={edges[state][i]<0} onClick={()=>S(edges[state][i])}>{x}</Button>)}</>}><MatrixView data={edges.map(row=>row.map(x=>x<0?0:1))} rowLabels={names} colLabels={v} highlight={(r)=>r===state?'accent':undefined}/></VizFrame>
}
