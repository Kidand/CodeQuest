import { md, py } from '@/content/helpers'
import type { Question, Challenge, GuidedExercise } from '@/content/types'
export const single = (id: string, difficulty: 1|2|3, prompt: string, options: string[], answer: number, explanation: string): Question => ({id,type:'single',difficulty,prompt,options,answer,explanation})
export const judge = (id: string, difficulty: 1|2|3, prompt: string, answer: boolean, explanation: string): Question => ({id,type:'judge',difficulty,prompt,answer,explanation})
export const numeric = (id: string, difficulty: 1|2|3, prompt: string, answer: number, explanation: string): Question => ({id,type:'numeric',difficulty,prompt,answer,tolerance:0.001,explanation})
export const multi = (id: string, difficulty: 1|2|3, prompt: string, options: string[], answer: number[], explanation: string): Question => ({id,type:'multi',difficulty,prompt,options,answer,explanation})
export const order = (id: string, prompt: string, items: string[], explanation: string): Question => ({id,type:'order',difficulty:2,prompt,items,explanation})
export const fill = (id: string, prompt: string, answers: string[], explanation: string): Question => ({id,type:'fill',difficulty:2,prompt,blanks:[{answers}],explanation})
export const bug = (id: string, code: string, bugLines: number[], explanation: string): Question => ({id,type:'bug',difficulty:3,prompt:md`点出导致结果错误的代码行。输入满足本关约束。`,code,bugLines,explanation})
export function practice(c: Challenge, blanks: {find:string; instruction:string; explain:string}[]): GuidedExercise {
 let template=c.solution
 const steps=blanks.map((b,i)=>{if(!template.includes(b.find))throw Error(b.find);template=template.replace(b.find,`{{${i+1}}}`);return {blank:i+1,instruction:b.instruction,answer:b.find,explain:b.explain}})
 return {id:'guided',title:`逐步实现：${c.title}`,intro:c.description,template:py`${template}`,steps,setup:c.setup,tests:c.tests,outro:md`再到实战阶段从空白函数实现，重点解释每一步维护的语义和边界。`}
}
