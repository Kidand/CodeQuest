import { defineLevel, md, py } from '@/content/helpers'
import type { Challenge } from '@/content/types'
import { ConstraintViz } from '../viz/DecodeViz'
import { single,judge,numeric,multi,order,fill,bug,practice } from '../shared'
export const constrainedChallenge:Challenge={id:'penalties',title:'惩罚、n-gram 与 FSM 合法掩码',difficulty:'hard',description:md`实现 \`process(logits: np.ndarray, history: list[int], theta: float, presence: float, frequency: float, n: int, allowed: np.ndarray) -> np.ndarray\`。V≤100、history≤100，ID 合法，theta≥1，n≥0；allowed 为长度 V 的布尔合法 token 掩码。复制 logits，先对历史中出现过的每个 ID 施加一次 CTRL 惩罚（负数乘 theta，非负除 theta），再减 presence×是否出现 + frequency×出现次数；之后禁止能产生历史已出现 n-gram 的下一 token（n=0 禁用，n=1 禁止已出现 token），最后非法 allowed 设为 -Inf。若无有限候选抛 ValueError；不执行 softmax，不修改输入。例：logits=[2,-2,1]、history=[0,0,1]、theta=2、presence=.5、frequency=.2、n=0、全合法 → [.1,-4.7,1]。`,starter:py`import numpy as np
def process(logits: np.ndarray, history: list[int], theta: float, presence: float, frequency: float, n: int, allowed: np.ndarray):
    raise NotImplementedError`,solution:py`import numpy as np
def process(logits: np.ndarray, history: list[int], theta: float, presence: float, frequency: float, n: int, allowed: np.ndarray):
    z = np.asarray(logits, dtype=np.float64).copy()
    counts = np.bincount(history, minlength=len(z))
    seen = counts > 0
    z[seen] = np.where(z[seen] < 0, z[seen] * theta, z[seen] / theta)
    z -= presence * seen + frequency * counts
    if n > 0 and len(history) >= n-1:
        prefix = tuple(history[-(n-1):]) if n > 1 else ()
        for i in range(len(history)-n+1):
            if tuple(history[i:i+n-1]) == prefix:
                z[history[i+n-1]] = -np.inf
    z[~np.asarray(allowed, dtype=bool)] = -np.inf
    if not np.isfinite(z).any():
        raise ValueError("没有合法候选")
    return z`,solutionAlt:[{label:'PyTorch 版（仅展示）',code:py`import torch
def process(logits, history, theta, presence, frequency, n, allowed):
    z = logits.double().clone()
    counts = torch.bincount(torch.tensor(history,device=z.device),minlength=z.numel()) if history else torch.zeros_like(z)
    seen = counts > 0
    z[seen] = torch.where(z[seen]<0,z[seen]*theta,z[seen]/theta)
    z -= presence*seen + frequency*counts
    if n > 0 and len(history)>=n-1:
        prefix = tuple(history[-(n-1):]) if n>1 else ()
        for i in range(len(history)-n+1):
            if tuple(history[i:i+n-1])==prefix: z[history[i+n-1]]=-torch.inf
    z[~allowed.bool()] = -torch.inf
    if not torch.isfinite(z).any(): raise ValueError("没有合法候选")
    return z`}],explanation:md`CTRL 根据符号处理，使出现过的 logit 总是变小；不能对负数也除 theta，否则会鼓励它。presence 依赖集合，frequency 依赖计数，两者不同。n-gram 只比较当前末尾 n−1 个 token 匹配的历史前缀，不应屏蔽所有历史末尾词。FSM 掩码只表达当前合法集合，需在选择 token 后更新状态。时间 $O(V+Hn)$，空间 $O(V)$；可用前缀哈希降低多步生成成本。`,hints:[md`bincount 同时提供出现集合与频次。`,md`n=1 的前缀是空元组，不要使用 history[-0:]。`,md`所有约束取交集，全空时显式失败，不能悄悄恢复非法 token。`],tests:[{name:'正负 logit 和频次',code:py`out=process(np.array([2.,-2.,1.]),[0,0,1],2,.5,.2,0,np.ones(3,bool))
assert np.allclose(out,[.1,-4.7,1]), f"期望 [.1,-4.7,1]，得到 {out}"`},{name:'bigram',code:py`out=process(np.zeros(4),[0,1,2,0],1,0,0,2,np.ones(4,bool))
assert np.isneginf(out[1]) and np.isfinite(out[[0,2,3]]).all(), f"仅禁止 1，得到 {out}"`},{name:'unigram 与不可变输入',hidden:true,code:py`z=np.array([1.,2.,3.]); old=z.copy()
out=process(z,[0,0],1,0,0,1,np.array([1,1,0],bool))
assert np.isneginf(out[[0,2]]).all() and out[1]==2, f"仅 1 合法，得到 {out}"
assert np.array_equal(z,old), "输入被修改"`},{name:'空历史与死状态',hidden:true,code:py`assert np.array_equal(process(np.array([1.,2.]),[],2,1,1,4,np.ones(2,bool)),[1,2])
try:
    process(np.zeros(2),[0,1],1,0,0,1,np.ones(2,bool))
    assert False, "应抛 ValueError，不能产生 NaN softmax"
except ValueError:
    pass`}],followUps:[{id:'tokens',q:md`为什么 JSON 文法不能简单按字符名单屏蔽 token？`,a:md`一个 token 可解码成多个字符，甚至跨越引号、转义或结构边界；必须把完整 token 字节序列送入当前解析状态，只有能保持合法前缀的 token 才可选。一般嵌套 JSON 需要栈式解析，有限状态机适用于有界或正则子语言。`}]}
const c=constrainedChallenge
export default defineLevel({learn:[{id:'penalty',title:'惩罚是对 logits 的有意偏置',body:md`CTRL repetition penalty 对出现过的 ID 只施加一次：$z<0$ 时乘 $\theta$，否则除 $\theta$，其中 $\theta>1$。这样正负 logit 都下降。presence penalty 减去 $\lambda_p\mathbf1[c_i>0]$，frequency penalty 减去 $\lambda_fc_i$，重复 10 次和重复 1 次对前者相同、对后者不同。惩罚作用范围（prompt 是否算历史）也应明确。它们有助于减少机械重复，却可能惩罚必需的人名、代码关键字或合法 JSON 重复键内容，因此不能替代任务验证。`},{id:'ngram',title:'局部禁止规则需要匹配前缀',body:md`no_repeat_ngram_size=n 禁止的是再次生成相同 n-gram，不是禁止全部出现过的词。例如历史 [A,B,C,A]，n=2，当前前缀 A 已经接过 B，所以只禁止 B；C 仍可选。n=1 退化为已出现 token 集合，n=0 表示关闭。构建「前 n−1 个 ID → 可禁止的后继集合」的哈希表可避免每一步扫描完整历史。这是硬约束，会直接删除概率质量；若与文法约束冲突导致全空，必须上报或采用明确的回退策略。`},{id:'fsm',title:'结构合法是前缀状态的性质',body:md`约束解码维护 FSM 或解析器状态，计算所有允许的 token ID，把其余 logits 设为 -Inf，再做采样。选出 token 后推进状态，EOS 只在接受状态允许。logit bias 是软偏置，有限正数仍无法保证词被选中；文法 mask 才是硬过滤。图中只实现正则语言「不重复相邻数字的非空二进制串」；任意深度 JSON 的匹配括号需要栈，不能假称一个小 FSM 就解决通用文法。语法合法也不保证事实或业务规则正确。参考：[CTRL](https://arxiv.org/abs/1909.05858)、[生成处理器文档](https://huggingface.co/docs/transformers/en/internal/generation_utils)。`,viz:ConstraintViz}],quiz:[single('negative',1,md`历史词 logit=-2、theta=2，CTRL 后为？`,['-1','-4','0','2'],1,md`负 logit 乘 theta 得 -4；除法会把它提升到 -1，反而奖励重复。`),judge('freq',1,md`presence penalty 随同一 token 出现次数线性增加。`,false,md`presence 看是否出现；frequency 才乘出现次数。`),numeric('combined',2,md`logit=4，出现 3 次，theta=2、presence=.5、frequency=.2，按本关顺序结果是多少？`,.9,md`4/2-.5-.2×3=.9。`),multi('mask',2,md`硬约束实现需要考虑？`,['全空候选','EOS 是否处于接受状态','token 可含多个字符','softmax 后把非法概率设零就无需归一化'],[0,1,2],md`若概率后处理删除项，必须重新归一化；logit mask 后 softmax 自动完成归一化。`),order('pipeline',md`结构化采样顺序：`,['读取当前解析状态','计算合法 token 集合并 mask','对合法候选采样','用选中 token 推进解析状态'],md`状态更新依赖实际选中的完整 token，不可预先按最大值更新。`),fill('maskvalue',md`硬 mask 在 logits 中使用 ____(1)____。`,['-inf','-np.inf','float("-inf")','负无穷'],md`其指数为 0，有限偏置无法让非法项概率严格为零。`),single('ngram',2,md`历史 [0,1,2,0]，禁止重复 bigram，应屏蔽哪个 ID？`,['0','1','2','0 和 1'],1,md`当前末尾前缀为 0，历史 0 的后继为 1，因此只屏蔽 1。`),judge('json',3,md`满足 JSON 语法就保证字段值符合业务规则。`,false,md`结构约束不验证真实性、跨字段逻辑或数据库状态，还需 schema 与业务校验。`),bug('bug',py`import numpy as np
def penalty(logits, history, theta):
    z = logits.copy()
    ids = np.unique(history)
    selected = z[ids]
    adjusted = selected / theta
    z[ids] = adjusted
    return z`,[6],md`第 6 行需按符号区分：np.where(selected<0,selected*theta,selected/theta)。unique 是正确的，CTRL 不按频次重复施加。`) ],guided:[practice(c,[{find:'counts > 0',instruction:md`从计数中提取出现集合。`,explain:md`CTRL 与 presence 只对出现 ID 作用一次。`},{find:'np.where(z[seen] < 0, z[seen] * theta, z[seen] / theta)',instruction:md`对正负 logit 分别施加惩罚。`,explain:md`两种符号都必须向较小值移动。`},{find:'presence * seen + frequency * counts',instruction:md`组合 presence 与 frequency 的减项。`,explain:md`一个依赖是否出现，一个依赖次数。`},{find:'tuple(history[-(n-1):]) if n > 1 else ()',instruction:md`得到待匹配 n-gram 前缀，注意 unigram。`,explain:md`history[-0:] 是整个历史，不能表示空前缀。`}])],challenges:[c],interview:[{id:'quality',freq:3,q:md`为什么 repetition penalty 太大容易损害代码生成？`,a:md`代码和结构化文本经常必须重复变量名、括号与关键字。惩罚机制只观察 token 出现情况，不理解语义，可能降低这些必需片段概率。应按任务选择弱惩罚或关闭，结合编译、单测或结构约束控制输出；不要用重复率单一指标优化代码质量。`},{id:'state',freq:3,q:md`文法约束的主要工程难点是什么？`,a:md`需要将子词 token 的字节序列与解析状态对接，正确处理多字符 token、转义、UTF-8、可结束状态和全空候选。词表大时逐个模拟解析昂贵，可缓存状态到掩码或利用 token trie 共享前缀，但缓存必须包含会影响合法性的完整状态。`},{id:'order',freq:2,q:md`惩罚与温度的顺序能交换吗？`,a:md`不是总能。纯乘除型 CTRL 与正数温度缩放在不改变符号时可交换，但固定加性 presence、frequency 或 logit bias 不可直接交换，否则等价惩罚强度也要随温度缩放。多个硬过滤的集合若独立可取交集，但 top-p 依赖归一化质量，通常不可随意换序。`}],cheatsheet:md`CTRL：负乘、正除；presence：减 $\lambda_p1[c>0]$；frequency：减 $\lambda_fc$。n-gram 看当前后缀匹配的历史前缀。FSM/文法 mask → 采样 → 状态转移；全空显式报错。`})
