import { defineLevel, md, py } from '@/content/helpers'
import type { Challenge } from '@/content/types'
import { BeamViz } from '../viz/DecodeViz'
import { single,judge,numeric,multi,order,fill,bug,practice } from '../shared'
const c:Challenge={id:'beam',title:'带 EOS 与长度惩罚的 Beam Search',difficulty:'hard',description:md`实现 \`beam_search(logp: np.ndarray, start: int, eos: int, width: int, max_steps: int, alpha: float) -> tuple[list[int],float]\`。logp 为 V×V 的转移 log 概率，允许 -Inf，V≤8；每行至少一个有限项，width≥1、max_steps≥0、alpha≥0。初始路径只含 start，原始分数为 0。每轮扩展所有未完成路径的有限转移；已以 eos 结束的路径原样保留，和扩展路径一起按原始分数降序选 width 条。并列按整条 ID 元组字典序升序。所有束均完成则停止；最后以 logP/max(1,生成长度)^alpha 选最佳（长度含 EOS、不含 start），同分仍按路径排序。返回路径及归一化分数。该明确的教学契约与框架的 completed hypotheses 池实现不同。`,starter:py`import numpy as np
def beam_search(logp: np.ndarray, start: int, eos: int, width: int, max_steps: int, alpha: float):
    raise NotImplementedError`,solution:py`import numpy as np
def beam_search(logp: np.ndarray, start: int, eos: int, width: int, max_steps: int, alpha: float):
    beams = [([start], 0.0)]
    for _ in range(max_steps):
        candidates = []
        for path, score in beams:
            if path[-1] == eos:
                candidates.append((path, score))
                continue
            for token, lp in enumerate(logp[path[-1]]):
                if np.isfinite(lp):
                    candidates.append((path + [token], score + float(lp)))
        beams = sorted(candidates, key=lambda x: (-x[1], tuple(x[0])))[:width]
        if all(path[-1] == eos for path, _ in beams):
            break
    ranked = [(path, score / max(1, len(path)-1)**alpha) for path, score in beams]
    return min(ranked, key=lambda x: (-x[1], tuple(x[0])))`,solutionAlt:[{label:'PyTorch 版（仅展示）',code:py`import torch
def beam_search(logp, start, eos, width, max_steps, alpha):
    beams = [([start], 0.0)]
    for _ in range(max_steps):
        candidates = []
        for path, score in beams:
            if path[-1] == eos:
                candidates.append((path, score))
            else:
                for token in torch.where(torch.isfinite(logp[path[-1]]))[0].tolist():
                    candidates.append((path+[token], score+logp[path[-1],token].item()))
        beams = sorted(candidates, key=lambda x:(-x[1],tuple(x[0])))[:width]
        if all(x[0][-1] == eos for x in beams): break
    return min(((x,s/max(1,len(x)-1)**alpha) for x,s in beams), key=lambda x:(-x[1],tuple(x[0])))`}],explanation:md`搜索阶段累加 log 概率，避免长序列概率乘积下溢。EOS 不扩展也不再次计分，否则同一已完成路径会被虚构后缀改变。每轮排序约 $O(KV\log(KV))$，此清晰实现还拷贝长度 O(T) 的路径；工程实现用 parent 索引回溯与 KV cache 重排。长度惩罚只在最终排序使用，不能假定宽度有限时获得全局归一化最优。`,hints:[md`将 EOS 路径作为原候选保留。`,md`搜索保留的原始分数和最终归一化分数要分开。`],setup:py`import numpy as np
_P=np.log(np.array([[.05,.6,.35],[.05,.05,.9],[.1,.1,.8]]))`,tests:[{name:'普通搜索',code:py`path,s=beam_search(_P,0,2,2,4,0)
assert path==[0,1,2], f"期望 [0,1,2]，得到 {path}"
assert np.isclose(s,np.log(.54)), f"期望 log(.54)，得到 {s}"`},{name:'零步',code:py`assert beam_search(_P,0,2,3,0,1)==([0],0.0), "零步返回起始路径"`},{name:'EOS 已结束与并列',hidden:true,code:py`assert beam_search(_P,2,2,3,5,2)==([2],0.0), "已结束不能追加 EOS"
a=np.log(np.full((3,3),1/3)); path,s=beam_search(a,0,2,1,2,0)
assert path==[0,0,0], f"并列取字典序小，得到 {path}"`},{name:'穷举宽束检查',hidden:true,code:py`rng=np.random.default_rng(11)
for _ in range(8):
    a=rng.random((3,3)); a/=a.sum(1,keepdims=True); lp=np.log(a)
    active=[([0],0.)]
    for step in range(3):
        active=[(path+[j],s+lp[path[-1],j]) for path,s in active if path[-1]!=2 for j in range(3)]+[(path,s) for path,s in active if path[-1]==2]
    exp=min(((path,s/(len(path)-1)) for path,s in active),key=lambda x:(-x[1],tuple(x[0])))
    got=beam_search(lp,0,2,100,3,1)
    assert got[0]==exp[0] and np.isclose(got[1],exp[1]), f"期望 {exp}，得到 {got}"`}],followUps:[{id:'cache',q:md`真实模型如何同步 KV cache？`,a:md`每个新束记录父束索引，按该索引 gather 或重排 KV cache，再追加新 token。多个候选可共享父前缀，采用页表引用计数与写时复制能减少复制；仅重排 token 而不重排 KV 会产生语义错误。`}]}
export default defineLevel({learn:[{id:'search',title:'保留多条前缀，延迟局部决策',body:md`greedy 每步只保留一条前缀。Beam Search 保留 K 条，将每条前缀扩展到 V 个候选，再从 K×V 个候选中筛出 K 条。序列分数是条件 log 概率之和，不能把本步分数单独排序；否则会让概率很低的父路径凭一个高概率子词翻盘。束宽是计算预算，不是全局最优保证：某条被早早删掉的前缀可能拥有最好的后缀。图中 B 前缀是否被保留取决于束宽，长度惩罚又会改变完整路径的排序。`,viz:BeamViz},{id:'length',title:'EOS、长度惩罚与停止条件',body:md`原始 log 概率每扩展一步通常都会下降，所以直接比较不同长度的路径会偏爱短句。常见得分形式为 $s=\log P(y)/\lvert y\rvert^\alpha$，也有 GNMT 的 $(5+\lvert y\rvert)^\alpha/6^\alpha$ 等变体；面试一定先说明约定。EOS 是生成事件，要计入概率；EOS 后不继续扩展。若只要某一个束完成就结束，可能错过另一个束的更优完整结果。引入长度归一化后，原始分数的单调性也不能直接用作归一化得分的停止上界。`},{id:'dialog',title:'最大似然不等于最好对话',body:md`Beam Search 常用于机器翻译、受约束转写等输出相对集中的任务；开放对话更需要多样性与自然措辞，单纯追求高似然可能偏向常见、泛泛、重复的文本。增加 K 可能把搜索目标优化得更彻底，却不改善人的评价。工程代价还包括 K 倍左右的活跃状态、KV cache 和候选重排，实际成本取决于前缀共享。参考：[GNMT 长度与覆盖惩罚](https://arxiv.org/abs/1609.08144)、[生成退化研究](https://arxiv.org/abs/1904.09751)。`}],quiz:[single('score',1,md`扩展路径的正确分数是？`,['本步概率','父 logP 加本步 logP','父 logP 乘本步 logP','概率均值'],1,md`概率相乘在 log 域变为相加。只用本步分数忽略前缀质量。`),judge('optimal',1,md`只要 K>1 就能保证找到全局最优序列。`,false,md`有限 K 仍会剪掉前缀，之后无法恢复。`),numeric('count',2,md`活跃束 K=4，词表 V=10，无 EOS、无 mask，本轮筛选前有多少候选？`,40,md`每条束扩展 10 个，共 4×10=40。`),multi('eos',2,md`EOS 的正确处理包括？`,['计入生成概率','后续不再扩展','任何一个束 EOS 就无条件退出','明确长度是否含 EOS'],[0,1,3],md`一个完成候选不代表其它路径已无改进机会，停止需依算法界限判断。`),order('flow',md`每轮搜索顺序：`,['从父束取条件 log 概率','加上父路径累计分数','在所有候选间全局选束','重排路径与 KV cache'],md`全局选择后才能确定新的父索引；仅对各父束独立取 K 条不够。`),fill('penalty',md`logP=-6、长度=3、alpha=1，归一化得分为 ____(1)____。`,['-2'],md`-6/3=-2。分母增加使负分数靠近零。`),single('tie',2,md`本关得分相同如何决定？`,['随机','短路径一定优先','整条 ID 元组字典序','最后 token 较大优先'],2,md`必须按已声明的确定规则处理并列；长度不是额外规则。`),judge('lengthbound',3,md`有长度归一化时，可直接用当前原始 logP 作为未来归一化分数的上界。`,false,md`分母会增长，负分数可能变得更接近零，原始分数单调下降不足以提供该上界。`),bug('bug',py`def expand(path, score, row, eos):
    candidates = []
    if path[-1] == eos:
        candidates.append((path, score))
    for token, lp in enumerate(row):
        new_path = path + [token]
        new_score = score + lp
        candidates.append((new_path, new_score))
    return candidates`,[4],md`第 4 行保留 EOS 后还应 return candidates；当前代码继续执行循环，给已结束序列添加虚构 token。`) ],guided:[practice(c,[{find:'candidates.append((path, score))',instruction:md`保留已完成路径，不改变分数。`,explain:md`EOS 只计一次，后续不扩展。`},{find:'score + float(lp)',instruction:md`更新联合 log 概率。`,explain:md`log 域中联合概率用加法。`},{find:'(-x[1], tuple(x[0]))',instruction:md`实现原始分数降序、路径字典序升序。`,explain:md`负分数用于升序排序；整个元组处理并列。`},{find:'score / max(1, len(path)-1)**alpha',instruction:md`按生成长度归一化最终分数。`,explain:md`起始 token 不计长度，零步时分母至少 1。`}])],challenges:[c],interview:[{id:'width',freq:3,q:md`为什么束越宽质量不一定越好？`,a:md`更宽主要降低搜索误差，但训练目标的概率偏差、长度偏差与人类偏好差距仍存在。对开放生成，它可能更彻底找到常见模板而降低多样性。应分别诊断模型误差、搜索误差与评价指标，并在同样延迟预算下比较采样方案。`},{id:'end',freq:3,q:md`什么时候可以安全提前停止？`,a:md`需要已完成候选达到要求，并证明未完成路径的最佳可能最终分数不超过当前候选。原始 logP 下每步 log 概率非正可以构造上界，但长度惩罚、覆盖奖励会改变界。生产框架还有 heuristic early stopping，必须区分保证最优的界与节省算力的启发式。`},{id:'memory',freq:2,q:md`Beam Search 显存为什么增加？`,a:md`不同束在分叉后需要各自 KV 状态，额外保存的路径只是小头。前缀可共享，分叉时通过 parent ID 重排页表并在修改时复制，避免整段 KV 拷贝。新增 token 的条件分布一定要对应正确父束，否则分数和模型上下文不一致。`}],cheatsheet:md`扩展 K×V 候选；累计 logP；全局选 K；重排 KV。EOS 计分一次。长度惩罚约定、tie-break、停止上界都要说清。有限束宽是近似搜索。`})
