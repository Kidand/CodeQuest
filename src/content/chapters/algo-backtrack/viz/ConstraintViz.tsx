import { useMemo, useState } from 'react'
import { StepControls, VizFrame, useStepper } from '@/viz'
import { Slider } from '@/components/ui'

export function QueensViz() {
 const frames=useMemo(()=>{
  const path:number[]=[],cols=new Set<number>(),diagA=new Set<number>(),diagB=new Set<number>()
  let solutions=0
  const fs=[{path:[...path],row:-1,col:-1,valid:true,solutions,note:'每行放一个皇后；列、主对角线和副对角线不能重复。'}]
  function dfs(row:number){
   if(row===4){solutions++;fs.push({path:[...path],row:-1,col:-1,valid:true,solutions,note:`得到第 ${solutions} 个方案；回溯继续寻找其它方案。`});return}
   for(let col=0;col<4;col++){
    const valid=!cols.has(col)&&!diagA.has(row-col)&&!diagB.has(row+col)
    fs.push({path:[...path],row,col,valid,solutions,note:`尝试 (${row},${col})：${valid?'列和两条对角线都空闲，可以放置':'发生冲突，整棵子树直接剪掉'}。`})
    if(!valid)continue
    path.push(col);cols.add(col);diagA.add(row-col);diagB.add(row+col);dfs(row+1)
    path.pop();cols.delete(col);diagA.delete(row-col);diagB.delete(row+col)
    fs.push({path:[...path],row,col,valid:true,solutions,note:`撤销 (${row},${col})，恢复三组约束，再尝试下一个兄弟分支。`})
   }
  }
  dfs(0);return fs
 },[])
 const st=useStepper(frames),f=st.frame
 return <VizFrame title="冲突发生时立即剪掉整棵子树" caption={f.note} controls={<StepControls stepper={st}/>}>
  <div className="mx-auto grid max-w-xs grid-cols-4 gap-1">{Array.from({length:16},(_,i)=>{const r=Math.floor(i/4),c=i%4,queen=f.path[r]===c,active=f.row===r&&f.col===c;return <div key={i} className={`flex aspect-square items-center justify-center rounded-lg border text-2xl ${active?(f.valid?'border-accent text-accent':'border-danger text-danger'):'border-border text-fg'} ${(r+c)%2?'bg-surface-2':'bg-surface'}`}>{queen?'♛':active?(f.valid?'✓':'×'):''}</div>})}</div>
  <p className="mt-3 text-center text-sm text-muted">已找到 {f.solutions} 个方案 · 当前列序列 [{f.path.join(', ')}]</p>
 </VizFrame>
}

export function PackingViz() {
 const [capacity,setCapacity]=useState(12)
 const frames=useMemo(()=>{
  const lengths=[8,7,6,5,4,3,2],bins:number[][]=[]
  const fs=[{bins:[] as number[][],i:-1,note:'长度已降序排序；每个样本放入第一个仍有空间的箱。'}]
  lengths.forEach((length,i)=>{
   let b=bins.findIndex(bin=>bin.reduce((s,x)=>s+x,0)+length<=capacity)
   if(b<0){b=bins.length;bins.push([])}
   bins[b].push(length)
   fs.push({bins:bins.map(bin=>[...bin]),i,note:`样本长度 ${length} 放入箱 ${b+1}；已用 ${bins.length} 箱。先检查旧箱，全部不够才开新箱。`})
  })
  return fs
 },[capacity])
 const st=useStepper(frames),f=st.frame
 return <VizFrame title="降序优先填入第一个可用箱" caption={f.note} controls={<><StepControls stepper={st}/><Slider label="每箱 token 容量" min={8} max={16} value={capacity} onChange={setCapacity}/></>}>
  <div className="space-y-3">{f.bins.length===0?<p className="text-sm text-muted">等待放入第一个样本</p>:f.bins.map((bin,i)=><div key={i}><p className="mb-1 text-xs text-muted">箱 {i+1} · {bin.reduce((s,x)=>s+x,0)} / {capacity}</p><div className="flex h-10 overflow-hidden rounded-lg border border-border bg-surface-2">{bin.map((x,j)=><div key={j} className="flex shrink-0 items-center justify-center border-r border-border bg-accent-soft font-mono text-sm text-accent" style={{width:`${x/capacity*100}%`}}>{x}</div>)}</div></div>)}<p className="text-sm text-muted">总长度 35 · 体积下界 ⌈35 / {capacity}⌉ = {Math.ceil(35/capacity)}；下界不保证可达到。</p></div>
 </VizFrame>
}
