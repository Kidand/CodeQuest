import { defineLevel, md, py } from '@/content/helpers'
import type { Challenge } from '@/content/types'
import { SamplingViz } from '../viz/DecodeViz'
import { single,judge,numeric,multi,order,fill,bug,practice } from '../shared'
const c: Challenge={id:'filter',title:'过滤概率并按给定随机数采样',difficulty:'medium',description:md`实现 \`sample(logits: np.ndarray, temperature: float, k: int, p: float, u: float) -> tuple[np.ndarray, int]\`。输入一维有限 logits，$1\le V\le1000$、$T>0$、$1\le k\le V$、$0<p\le1$、$0\le u<1$。先除温度，按分数降序保留恰好 k 个（并列按 token ID 小优先）；在 top-k 内重新归一化，再保留累计概率首次达到或超过 p 的最短前缀；返回原词表顺序的最终概率及逆 CDF 采样 ID。例：log([.5,.3,.2])，T=1,k=3,p=.7,u=.7 → ([.625,.375,0],1)。不可原地修改输入。`,starter:py`import numpy as np
def sample(logits: np.ndarray, temperature: float, k: int, p: float, u: float):
    raise NotImplementedError`,solution:py`import numpy as np
def sample(logits: np.ndarray, temperature: float, k: int, p: float, u: float):
    z = np.asarray(logits, dtype=np.float64) / temperature
    ids = np.argsort(-z, kind="stable")[:k]
    e = np.exp(z[ids] - z[ids].max())
    probs = e / e.sum()
    count = min(k, int(np.searchsorted(np.cumsum(probs), p, side="left")) + 1)
    ids, probs = ids[:count], probs[:count]
    out = np.zeros_like(z)
    out[ids] = probs / probs.sum()
    cdf = np.cumsum(out)
    last = int(np.flatnonzero(out > 0)[-1])
    cdf[last:] = 1.0
    token = int(np.searchsorted(cdf, u, side="right"))
    return out, token`,solutionAlt:[{label:'PyTorch 版（仅展示）',code:py`import torch
def sample(logits, temperature, k, p, u):
    z = logits.double() / temperature
    ids = torch.argsort(z, descending=True, stable=True)[:k]
    probs = torch.softmax(z[ids], 0)
    n = min(k, int(torch.searchsorted(probs.cumsum(0), p)) + 1)
    out = torch.zeros_like(z)
    out[ids[:n]] = probs[:n] / probs[:n].sum()
    cdf = out.cumsum(0)
    last = int(torch.where(out > 0)[0][-1])
    cdf[last:] = 1
    return out, int(torch.searchsorted(cdf, u, right=True))`}],explanation:md`稳定排序把并列决策固定下来。top-p 比较的是累计质量，因此跨过阈值的那个 token 必须保留；最终还要重新归一化。采样使用严格大于 u 的第一个 CDF 值，跳过零概率区间。排序时间 $O(V\log V)$，辅助空间 $O(V)$；生产中 top-k 可用选择算法。`,hints:[md`先用稳定 argsort 选 k 个，再在它们上做 softmax。`,md`保留 searchsorted(cdf,p,left)+1 个；采样则使用 right。`],setup:py`import numpy as np
def _check(z,t,k,p,u,exp,token):
    old=z.copy()
    out,got=sample(z,t,k,p,u)
    assert out.shape==z.shape, f"期望形状 {z.shape}，得到 {out.shape}"
    assert np.allclose(out,exp), f"期望 {exp}，得到 {out}"
    assert got==token, f"期望 ID {token}，得到 {got}"
    assert np.array_equal(z,old), "不得修改输入"`,tests:[{name:'核采样示例',code:py`_check(np.log([.5,.3,.2]),1,3,.7,.7,[.625,.375,0],1)`},{name:'边界 u 与零概率',code:py`_check(np.array([0.,9.,8.]),1,1,1,0,[0,1,0],1)`},{name:'并列与精确累计阈值',hidden:true,code:py`_check(np.zeros(4),1,4,.5,.5,[.5,.5,0,0],1)`},{name:'大 logit 与温度',hidden:true,code:py`for t in [.1,1,10]:
    z=np.array([10000.,9999.,9998.]); e=np.exp((z-z.max())/t); e/=e.sum()
    _check(z,t,3,1,.999999,e,2 if t>=1 else 1)`},{name:'CDF 尾部舍入',hidden:true,code:py`rng=np.random.default_rng(88)
for _ in range(30):
    z=rng.normal(size=11);z[-1]=-1000
    out,token=sample(z,1,4,.97,np.nextafter(1.,0))
    assert out[token]>0, f"不应采中尾部零概率项 {token}，分布 {out}"`}],followUps:[{id:'minp',q:md`min-p 和 top-p 有什么区别？`,a:md`min-p 保留 $p_i\ge m\max_jp_j$ 的 token，是相对最高概率的逐项阈值；top-p 则控制累计质量，依赖排序。两者通常都至少保留最大值，组合顺序要写进接口契约。`}]}
export default defineLevel({learn:[{id:'temperature',title:'温度改变概率比，不改变排序',body:md`自回归模型输出的是未归一化 logits。温度采样采用 $p_i=\exp(z_i/T)/\sum_j\exp(z_j/T)$，任意两词的概率比为 $\exp((z_i-z_j)/T)$。T 越小，分布越尖；T 越大越平。有限正温度不会改变 argmax，但会改变概率质量，进而改变 top-p 的候选数。实现应先减最大值防溢出。T=0 必须单独定义为 greedy 分支，不能真的除以零。即使 greedy 也需固定并列处理，才能在单元测试中复现。`,viz:SamplingViz},{id:'filters',title:'top-k、top-p、min-p 分别约束什么',body:md`top-k 固定候选数量；top-p（nucleus）选择降序概率累计首次达到 p 的最小前缀；min-p 则将低于最大概率某个比例的项删除。对 [.5,.3,.2]，top-p=.7 应保留前两个，随后归一化成 [.625,.375,0]。错误写法直接删除累计值大于 p 的位置，会连跨阈值的一项一起删掉，p 很小时甚至全空。本关严格使用「温度 → top-k → 在剩余项中 top-p」，实际框架可能还有其它处理器，不能把不同顺序的结果当作实现错误。`},{id:'quality',title:'解码目标和工程可复现性',body:md`greedy 适合需要确定输出的评测与抽取，但逐步最优不等于序列全局最优。随机采样在对话与开放生成中提供多样性，而截断尾部降低不可靠候选的影响；参数过强又会造成重复或固定措辞。给定随机种子只控制随机流，batch 顺序、浮点精度与内核仍可能影响结果。教学测试显式输入 u，将概率计算与随机数来源分开，既能测 CDF 边界，也能避免偶然失败。参考：[Nucleus 原论文](https://arxiv.org/abs/1904.09751)、[Transformers 生成处理器](https://huggingface.co/docs/transformers/en/internal/generation_utils)。`}],quiz:[single('greedy',1,md`T>0 改变时一定不变的是？`,['argmax 排序（无数值并列变化）','top-p 集合大小','熵','所有概率'],0,md`除以正数保持排序，但 softmax 的比例和熵改变，累计阈值集合也可能改变。`),judge('zero',1,md`可直接计算 logits/0 来实现 greedy。`,false,md`会产生 Inf/NaN；greedy 应直接 argmax，并指定 tie-break。`),numeric('mass',2,md`原概率 [.5,.3,.2]，top-p=.7，过滤后第二项概率是多少？`,.375,md`保留 .5+.3=.8 的前缀，第二项 .3/.8=.375。`),multi('stable',2,md`哪些步骤有助于稳定与复现？`,['softmax 前减最大值','固定并列顺序','将 -Inf 改成 0 来 mask','显式传入采样随机数'],[0,1,3],md`0 logit 仍可能有很高概率，mask 应用 -Inf；其它选项分别处理溢出、排序不唯一与随机测试。`),order('pipeline',md`按本关契约排列：`,['温度缩放','选择 top-k','在剩余候选中计算 top-p','归一化并逆 CDF 采样'],md`top-k 后重新算累计质量，因此和先 top-p 不一定等价。`),fill('cdf',md`逆 CDF 应找第一个 cdf[i] ____(1)____ u，填比较符。`,['>'],md`严格大于可以跳过零概率区间，等于边界时进入下一区间。`),single('minp',2,md`概率 [.6,.25,.1,.05]，min-p=.2 保留哪些？`,['前 1 个','前 2 个','前 3 个','全部'],1,md`阈值 .2×.6=.12，所以 .6 和 .25 合格；不是保留概率超过 .2 的累计核。`),judge('order',3,md`先温度后 top-p 与先 top-p 后温度一般等价。`,false,md`温度改变累计质量，过滤集合可不同；仅固定 k 且无并列时，正温度不改变 top-k 索引。`),bug('bug',py`import numpy as np
def nucleus(z, p):
    ids = np.argsort(-z)
    e = np.exp(z[ids] - z.max())
    prob = e / e.sum()
    remove = np.cumsum(prob) > p
    prob[remove] = 0
    return ids, prob / prob.sum()`,[6],md`第 6 行删除了首次跨阈值的一项。应将删除掩码右移一位且保留第一项，或用 left searchsorted+1。`) ],guided:[practice(c,[{find:'np.argsort(-z, kind="stable")[:k]',instruction:md`固定并列优先级，并截取 top-k。`,explain:md`稳定排序让相同 logit 保持原 token ID 顺序。`},{find:'np.exp(z[ids] - z[ids].max())',instruction:md`写出数值稳定的指数。`,explain:md`减最大值不改变 softmax 比例。`},{find:'np.searchsorted(np.cumsum(probs), p, side="left")',instruction:md`找到首次达到 top-p 阈值的位置。`,explain:md`left 让恰好等于阈值时无需多留一项。`},{find:'np.searchsorted(cdf, u, side="right")',instruction:md`把 u 映射为 token ID。`,explain:md`right 对应严格大于 u。`}])],challenges:[c],interview:[{id:'why',freq:3,q:md`为什么不能只降低温度来代替 top-p？`,a:md`温度只是连续改变分布尖锐程度，任何有限 logit 的概率仍大于零；top-p 明确删除尾部并归一化，改变支持集。低温可能减少坏词概率，也可能把最高的错误答案进一步放大。应结合任务质量、重复率与多样性评估参数，不存在所有模型通用的最佳设置。`},{id:'sequence',freq:3,q:md`greedy 找到最可能句子吗？`,a:md`不能。它只保证当前步的条件概率最大，较小的首词概率可能接上一条高概率后缀而成为更好的完整序列。Beam Search 保留多个前缀缓解局部决策，但仍有剪枝近似；而开放式生成的最佳人类评价又不等于模型最大似然。`},{id:'tie',freq:2,q:md`top-k 是否总保留恰好 k 个？`,a:md`取决于接口。有的实现通过第 k 大值作阈值，遇到并列会保留超过 k 个；本关用稳定排序后切片，保证恰好 k 个。面试先明确边界契约，生产复现则记录处理器顺序、版本与 tie-break。`}],cheatsheet:md`$p=\operatorname{softmax}(z/T)$；top-k 控制数量，top-p 控制累计质量，min-p 控制相对峰值阈值。

核集合保留**首个跨阈值项**。阈值查找用 left；CDF 采样用 right。mask 用 -Inf，过滤后重归一化；T=0 单独 greedy。`})
