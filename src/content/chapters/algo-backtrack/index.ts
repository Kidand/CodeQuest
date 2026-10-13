import { defineChapter, md } from '@/content/helpers'
export default defineChapter({
 id:'algo-backtrack',track:'algo',order:6,title:'回溯与贪心',subtitle:'决策树 · 约束剪枝 · 交换论证 · 序列打包',icon:'GitBranch',prerequisites:['algo-ds'],
 intro:md`从决策树写出可撤销的选择，再用约束尽早排除不可能的分支。面对贪心问题，先证明局部决策为何安全；面对序列打包，明确启发式与精确最优的区别。`,
 levels:[{id:'algo-backtrack-template',title:'回溯模板',summary:'子集与排列、排序同层去重、组合总和与状态恢复',difficulty:3,estMinutes:45,tier:'must',tags:['回溯','去重'],problems:['LC 78','LC 47','LC 40'],load:()=>import('./levels/template')},
 {id:'algo-backtrack-pruning',title:'剪枝与约束搜索',summary:'N 皇后、合法括号与棋盘状态恢复',difficulty:4,estMinutes:45,tier:'high',tags:['剪枝','约束搜索'],problems:['LC 22','LC 51','LC 79'],load:()=>import('./levels/pruning')},
 {id:'algo-backtrack-greedy',title:'贪心',summary:'交换论证、合并区间与 SFT 序列打包',difficulty:3,estMinutes:45,tier:'high',tags:['贪心','序列打包'],problems:['LC 45','LC 56','FFD'],llmLink:'SFT 变长样本打包，减少空余 token 槽位',load:()=>import('./levels/greedy')},
 {id:'algo-backtrack-boss',title:'BOSS：搜索与贪心综合',summary:'贪心上界、搜索剪枝与精确最少装箱',kind:'boss',difficulty:5,estMinutes:50,tier:'high',tags:['分支定界','综合挑战'],problems:['精确装箱'],load:()=>import('./levels/boss')}],
})
