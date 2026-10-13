import { defineChapter, md } from '@/content/helpers'
export default defineChapter({id:'llm-decode',track:'llm',order:2,title:'分词与解码',subtitle:'BPE · 温度采样 · Beam Search · 结构化约束',icon:'MessageSquare',intro:md`从文字如何变成 token，到下一个 token 如何被选择。把分词秩、概率过滤、搜索和文法状态接成可测试的生成流程。`,levels:[
{id:'llm-decode-bpe',title:'BPE 分词',summary:'训练 pair、merge rank、byte-level 与分词取舍',difficulty:2,estMinutes:40,tier:'foundation',frequency:'high',tags:['BPE','tokenizer'],load:()=>import('./levels/bpe')},
{id:'llm-decode-sampling',title:'采样策略',summary:'温度、top-k、top-p、min-p 与 CDF 边界',difficulty:2,estMinutes:40,tier:'foundation',frequency:'high',tags:['采样','top-p'],load:()=>import('./levels/sampling')},
{id:'llm-decode-beam',title:'Beam Search',summary:'联合概率、长度惩罚、EOS 与 KV 重排',difficulty:3,estMinutes:45,tier:'foundation',frequency:'high',tags:['Beam Search','EOS'],load:()=>import('./levels/beam')},
{id:'llm-decode-constrained',title:'重复惩罚与约束解码',summary:'符号惩罚、n-gram、FSM 与结构合法性',difficulty:3,estMinutes:45,tier:'modern',frequency:'medium',tags:['CTRL','FSM','JSON'],load:()=>import('./levels/constrained')},
{id:'llm-decode-boss',title:'BOSS：解码综合',summary:'实现带状态约束与核采样的生成器',kind:'boss',difficulty:4,estMinutes:50,tier:'foundation',frequency:'high',tags:['综合实战'],load:()=>import('./levels/boss')}
]})
