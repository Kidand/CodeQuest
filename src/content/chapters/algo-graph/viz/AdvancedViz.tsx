import { useMemo, useState } from 'react'
import { ArrayView, GraphView, StepControls, VizFrame, useStepper } from '@/viz'
import { Segmented } from '@/components/ui'

export function TreeGainViz() {
 const [negative,setNegative]=useState<'mixed'|'negative'>('mixed')
 const frames=useMemo(()=>{
  const values=negative==='mixed'?[-10,9,20,15,7]:[-8,-3,-6,-4,-2]
  const children=[[1,2],[],[3,4],[],[]]
  const gain=Array(5).fill(null) as (number|null)[]
  let best=-Infinity
  const fs=[{values,gain:[...gain],current:-1,best,note:'后序计算：孩子返回单边贡献，父节点合并两边候选。'}]
  const visit=(i:number):number=>{
   const left=children[i][0]===undefined?0:Math.max(0,visit(children[i][0]))
   const right=children[i][1]===undefined?0:Math.max(0,visit(children[i][1]))
   best=Math.max(best,values[i]+left+right)
   gain[i]=values[i]+Math.max(left,right)
   fs.push({values,gain:[...gain],current:i,best,note:`节点 ${values[i]}：左右非负贡献 ${left}、${right}；经过此点候选 ${values[i]+left+right}，向上只能返回单边 ${gain[i]}。`})
   return gain[i]!
  }
  visit(0); return fs
 },[negative])
 const st=useStepper(frames),f=st.frame
 return <VizFrame title="返回单边，答案可以拐弯" caption={f.note} controls={<><StepControls stepper={st}/><Segmented value={negative} onChange={setNegative} options={[{value:'mixed',label:'混合正负'},{value:'negative',label:'全部负数'}]}/></>}>
  <GraphView width={440} height={240} nodes={f.values.map((v,i)=>({id:i,label:v,x:[180,80,300,230,370][i],y:[35,120,120,205,205][i],tone:i===f.current?'accent':f.gain[i]!==null?'success':'muted',sub:`返回 ${f.gain[i]??'?'}`}))} edges={[[0,1],[0,2],[2,3],[2,4]].map(([from,to])=>({from,to}))}/><p className="text-center text-sm text-fg">当前最大路径和：{Number.isFinite(f.best)?f.best:'尚未访问'}</p>
 </VizFrame>
}

export function DijkstraViz() {
 const edges=[[0,1,4],[0,2,1],[2,1,2],[1,3,1],[2,3,5],[3,4,3]]
 const frames=useMemo(()=>{
  const dist=[0,Infinity,Infinity,Infinity,Infinity],done=new Set<number>()
  const fs=[{dist:[...dist],done:[...done],u:-1,note:'源点 0 的距离为 0；其它点为无穷。每次选最小暂定距离。'}]
  for(let k=0;k<5;k++){
   let u=-1
   for(let i=0;i<5;i++) if(!done.has(i)&&(u<0||dist[i]<dist[u])) u=i
   done.add(u)
   fs.push({dist:[...dist],done:[...done],u,note:`确定节点 ${u} 的最短距离 ${dist[u]}；非负边保证以后不会被更长前缀改小。`})
   for(const [from,to,w] of edges) if(from===u&&dist[to]>dist[u]+w){dist[to]=dist[u]+w; fs.push({dist:[...dist],done:[...done],u,note:`松弛 ${u} → ${to}，候选 ${dist[u]} + ${w} = ${dist[to]}，更新暂定距离。`})}
  }
  return fs
 // eslint-disable-next-line react-hooks/exhaustive-deps
 },[])
 const st=useStepper(frames),f=st.frame
 return <VizFrame title="最小暂定距离先被确定" caption={f.note} controls={<StepControls stepper={st}/>}>
 <GraphView width={440} height={260} nodes={f.dist.map((d,i)=>({id:i,label:i,x:[45,180,180,320,390][i],y:[125,45,215,125,215][i],sub:`d=${Number.isFinite(d)?d:'∞'}`,tone:i===f.u?'accent':f.done.includes(i)?'success':'muted'}))} edges={edges.map(([from,to,w])=>({from,to,label:String(w),directed:true}))}/></VizFrame>
}

export function TrieWalkViz() {
 const [word,setWord]=useState<'cars'|'cat'|'care'>('cars')
 const frames=useMemo(()=>{
  const paths=['','c','ca','car','cars','cat']
  let current=''
  const fs=[{current,note:'根节点代表空前缀。沿字符边走，完整词还需要终止标记。',failed:false}]
  for(const c of word){const next=current+c;if(!paths.includes(next)){fs.push({current,note:`缺少字符 ${c} 的边：单词 ${word} 不存在。`,failed:true});return fs}current=next;fs.push({current,note:`已匹配前缀 ${current}；${['car','cars','cat'].includes(current)?'此处有词尾标记':'此处还不是完整词'}。`,failed:false})}
  return fs
 },[word])
 const st=useStepper(frames),f=st.frame
 const labels=['','c','ca','car','cars','cat']
 return <VizFrame title="路径存在不等于单词存在" caption={f.note} controls={<><StepControls stepper={st}/><Segmented value={word} onChange={setWord} options={[{value:'cars',label:'cars'},{value:'cat',label:'cat'},{value:'care',label:'care'}]}/></>}>
 <GraphView width={420} height={300} nodes={labels.map((s,i)=>({id:s,label:s?s.slice(-1):'根',x:[210,210,210,120,120,300][i],y:[25,85,145,215,280,215][i],sub:['car','cars','cat'].includes(s)?'词尾 ✓':s||'空前缀',tone:s===f.current?(f.failed?'danger':'accent'):'muted'}))} edges={[['','c'],['c','ca'],['ca','car'],['car','cars'],['ca','cat']].map(([from,to])=>({from,to,directed:true}))}/></VizFrame>
}

export function UnionFindViz() {
 const frames=useMemo(()=>{
  const p=[0,1,2,3,4,5],size=Array(6).fill(1)
  const fs=[{p:[...p],note:'六个元素各自是一棵树。只有根节点的 size 有效。'}]
  const find=(x:number):number=>p[x]===x?x:(p[x]=find(p[x]))
  for(const [a,b] of [[0,1],[2,3],[4,5],[0,2],[0,4],[3,5]]){let x=find(a),y=find(b);if(x!==y){if(size[x]<size[y])[x,y]=[y,x];p[y]=x;size[x]+=size[y]};fs.push({p:[...p],note:x===y?`${a} 与 ${b} 已连通；不减少连通分量。find 顺便压缩路径。`:`合并 ${a}、${b}：小树根 ${y} 挂到大树根 ${x}。`})}
  return fs
 },[])
 const st=useStepper(frames),f=st.frame
 return <VizFrame title="只连接根，路径可以压缩" caption={f.note} controls={<StepControls stepper={st}/>}><div className="min-w-0 space-y-3"><p className="text-sm text-muted">位置是元素编号，格子值是 parent。</p><ArrayView values={f.p}/><GraphView width={450} height={180} nodes={f.p.map((p,i)=>({id:i,label:i,x:40+i*72,y:p===i?35:130,tone:p===i?'success':'accent'}))} edges={f.p.flatMap((p,i)=>p===i?[]:[{from:i,to:p,directed:true}])}/></div></VizFrame>
}
