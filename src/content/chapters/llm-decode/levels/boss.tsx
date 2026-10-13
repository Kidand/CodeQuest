import { defineLevel, md, py } from '@/content/helpers'
import { ConstraintViz } from '../viz/DecodeViz'
import { single,judge,numeric,multi,order,fill,bug } from '../shared'
export default defineLevel({learn:[{id:'review',title:'把分布、合法集合与随机性分开',body:md`本关把固定转移模型、FSM、温度、top-p 与 EOS 接在一起。模型回答「哪个 token 更可能」，FSM 回答「哪个 token 仍是合法前缀」，采样器把最终分布映射为一个 token，状态机再据此推进。最常见集成错误是先在全词表取核再做文法 mask，可能把全部合法项提前删掉；另一个错误是忽略一个 token 对应多个字符的语义。本题直接给 ID 级 FSM，排除 tokenizer 接口复杂度，以便把注意力放在顺序、CDF 边界与停止条件上。`,viz:ConstraintViz}],quiz:[single('rank',1,md`BPE 新文本合并优先级由谁决定？`,['新文本频次','训练 merge rank','字符串最长','随机选择'],1,md`训练合并秩固定，编码不重新训练。`),judge('beam',1,md`Beam Search 的父路径概率可忽略，只保留本步最大值。`,false,md`联合概率包含完整前缀，log 域需要累加。`),numeric('nucleus',2,md`合法集合中的概率为 [.4,.35,.25]，top-p=.6 保留几项？`,2,md`前两项累计 .75，首次超过 .6；需保留跨阈值项。`),multi('eos',2,md`正确 EOS 规则有？`,['EOS 本身计入输出','EOS 后不继续采样','EOS 总是合法，无需 FSM 判断','达到步数上限可返回未完成前缀并说明状态'],[0,1,3],md`只有接受状态才能允许 EOS；最大长度截断不保证序列完整。`),order('pipeline',md`按本 BOSS 契约排列：`,['取转移 logits 并做合法 mask','温度缩放与 softmax','在合法概率上 top-p 并归一化','采样、推进 FSM、检查 EOS'],md`合法集合必须先确定，否则核集合可能只剩非法项。`),fill('cdf',md`u 恰好落在 CDF 边界时，searchsorted 的 side 应是 ____(1)____。`,['right'],md`使用严格大于，跳过零质量区间。`),single('sign',2,md`CTRL 遇到历史词 logit=-3、theta=2 应变成？`,['-1.5','-6','3','6'],1,md`负数乘系数才能降低它。`),judge('syntax',3,md`仅 token ID 层面禁止相同 token，就等价于禁止重复自然语言短语。`,false,md`相同字符串可能有不同切分，大小写和同义表达也可能不同；ID 规则只提供离散层约束。`),numeric('bytes',2,md`256 基础字节 token 加 1000 次各产生一个新 token 的合并，忽略特殊 token，词表多少项？`,1256,md`256+1000=1256；实际模型还需计特殊 token，合并产物可能受词表规则约束。`),bug('empty',py`import numpy as np
def masked_softmax(z, allowed):
    out = z.copy()
    out[~allowed] = -np.inf
    maximum = out.max()
    weights = np.exp(out-maximum)
    denominator = weights.sum()
    return weights / denominator`,[5],md`第 5 行在全 mask 情况下得到 -Inf，之后 -Inf-(-Inf) 为 NaN。应先检查至少一个有限候选，否则抛出明确异常。`) ],guided:[],challenges:[{id:'fsm-nucleus',title:'FSM 下的核采样生成器',difficulty:'hard',description:md`实现 \`generate(logits: np.ndarray, transitions: np.ndarray, start_token: int, start_state: int, eos: int, temperature: float, top_p: float, us: list[float]) -> tuple[list[int],int]\`。logits 是 V×V 有限转移 logits；transitions 是 S×V 整数表，-1 表示非法，否则是下一状态。先取当前 token 对应的 logits，按当前状态 mask，再温度与 top-p（稳定降序，首次达到阈值），按词表 ID 顺序逆 CDF 采样，更新状态；选中 eos 后立即停止。返回不含起始 token 的生成 ID 列表与最终状态。无合法候选抛 ValueError；us 空直接返回空列表和起始状态。V,S≤8，T>0，0<p≤1，每个 u∈[0,1)。`,starter:py`import numpy as np
def generate(logits, transitions, start_token, start_state, eos, temperature, top_p, us):
    raise NotImplementedError`,solution:py`import numpy as np
def generate(logits, transitions, start_token, start_state, eos, temperature, top_p, us):
    token, state, result = start_token, start_state, []
    for u in us:
        ids = np.flatnonzero(transitions[state] >= 0)
        if not len(ids): raise ValueError("没有合法候选")
        z = np.asarray(logits[token, ids], dtype=float) / temperature
        order = np.argsort(-z, kind="stable")
        ids, z = ids[order], z[order]
        p = np.exp(z-z.max()); p /= p.sum()
        count = min(len(ids), int(np.searchsorted(np.cumsum(p), top_p)) + 1)
        out = np.zeros(logits.shape[1])
        out[ids[:count]] = p[:count] / p[:count].sum()
        cdf = np.cumsum(out)
        last = int(np.flatnonzero(out > 0)[-1]); cdf[last:] = 1
        token = int(np.searchsorted(cdf, u, side="right"))
        result.append(token)
        state = int(transitions[state, token])
        if token == eos: break
    return result, state`,solutionAlt:[{label:'PyTorch 版（仅展示）',code:py`import torch
def generate(logits, transitions, start_token, start_state, eos, temperature, top_p, us):
    token, state, result = start_token, start_state, []
    for u in us:
        ids = torch.where(transitions[state]>=0)[0]
        if ids.numel()==0: raise ValueError("没有合法候选")
        z=logits[token,ids].double()/temperature
        order=torch.argsort(z,descending=True,stable=True)
        ids,z=ids[order],z[order]
        p=z.softmax(0)
        n=min(ids.numel(),int(torch.searchsorted(p.cumsum(0),top_p))+1)
        out=torch.zeros(logits.shape[1],dtype=z.dtype,device=z.device)
        out[ids[:n]]=p[:n]/p[:n].sum()
        cdf=out.cumsum(0)
        last=int(torch.where(out>0)[0][-1]);cdf[last:]=1
        token=int(torch.searchsorted(cdf,u,right=True))
        result.append(token);state=int(transitions[state,token])
        if token==eos:break
    return result,state`}],explanation:md`先在合法集合内求分布，确保最高非法 logit 无法挤掉合法项。稳定排序用于核，散回词表顺序用于逆 CDF；两个顺序不能混淆。每步状态推进一次，EOS 后立即停止消费随机数。总时间 $O(TV\log V)$、临时空间 $O(V)$，生成列表 O(T)。`,hints:[md`先 flatnonzero 得到合法 ID。`,md`概率排序用于选核，采样前 scatter 回词表 ID 顺序。`,md`状态更新后再检查 EOS，才能返回正确最终状态。`],setup:py`import numpy as np
_L=np.array([[9.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
_F=np.array([[-1,1,-1],[1,-1,2],[-1,-1,-1]])`,tests:[{name:'mask 先于核',code:py`got=generate(_L,_F,0,0,2,1,.1,[0,.99,.1])
assert got==([1,0,0],1), f"期望 ([1,0,0],1)，得到 {got}"`},{name:'EOS 停止并更新',code:py`got=generate(_L,_F,0,0,2,1,1,[0,.9,0])
assert got==([1,2],2), f"期望 ([1,2],2)，得到 {got}"`},{name:'空随机流与死状态',hidden:true,code:py`assert generate(_L,_F,0,2,2,1,1,[])==([],2)
try:
    generate(_L,_F,0,2,2,1,1,[.1])
    assert False, "死状态应报错"
except ValueError: pass`},{name:'并列和 CDF 边界',hidden:true,code:py`f=np.array([[0,0,0]])
got=generate(np.zeros((3,3)),f,0,0,2,1,2/3,[.5,.999])
assert got==([1,1],0), f"前两项均为 .5，期望 [1,1]，得到 {got}"`}],followUps:[{id:'production',q:md`如果要支持真实 tokenizer，哪里需要扩展？`,a:md`将 transitions[state,token] 替换为解析器对整个 token 字节串的状态推进；token trie 可共享公共字节前缀。还要处理拒绝字符串、最大长度截断和流式 UTF-8，不能只在输出完成后检查 JSON。`}]}],interview:[],cheatsheet:md`BPE 先明确词边界和 rank；采样明确顺序、并列、CDF；beam 明确分数、长度和 EOS；约束明确合法前缀与死状态。生成管线的每个阶段都要有独立可测的契约。`})
