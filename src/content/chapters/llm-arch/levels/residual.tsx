import { defineLevel, md, py } from '@/content/helpers'
import type { Challenge } from '@/content/types'
import { SinkhornViz } from '../viz/ArchViz'
import { single,judge,numeric,multi,order,fill,bug,practice } from '../shared'
const c:Challenge={id:'residual',title:'Sinkhorn 投影与深度注意力',difficulty:'hard',description:md`实现 \`sinkhorn(logits: np.ndarray, steps: int) -> np.ndarray\`：输入有限 N×N logits（N≤8），在 log 域交替做行 logsumexp 归一化、列 logsumexp 归一化，共 steps≥1 轮，最后 exp。实现 \`depth_attention(values: np.ndarray, query: np.ndarray, eps: float=1e-6) -> tuple[np.ndarray,np.ndarray]\`：values 为 [L,T,D]、query 为 [D]，L,T,D≤16；每个 value 沿 D 做无可学习缩放 RMSNorm 得到 key，score=sum(key×query)，沿 L 做 softmax，使用未归一化 values 加权，返回 [T,D] 输出和 [L,T] 权重。这是两个核心算子，不是完整 mHC/AttnRes 网络。`,starter:py`import numpy as np
def sinkhorn(logits: np.ndarray, steps: int) -> np.ndarray:
    raise NotImplementedError
def depth_attention(values: np.ndarray, query: np.ndarray, eps: float=1e-6):
    raise NotImplementedError`,solution:py`import numpy as np
def _logsumexp(x, axis):
    maximum = x.max(axis=axis, keepdims=True)
    return maximum + np.log(np.exp(x-maximum).sum(axis=axis, keepdims=True))
def sinkhorn(logits: np.ndarray, steps: int) -> np.ndarray:
    z = np.asarray(logits,dtype=np.float64).copy()
    for _ in range(steps):
        z -= _logsumexp(z, axis=1)
        z -= _logsumexp(z, axis=0)
    return np.exp(z)
def depth_attention(values: np.ndarray, query: np.ndarray, eps: float=1e-6):
    keys = values / np.sqrt(np.mean(values*values,axis=-1,keepdims=True)+eps)
    scores = np.sum(keys*query,axis=-1)
    weights = np.exp(scores-scores.max(axis=0,keepdims=True))
    weights /= weights.sum(axis=0,keepdims=True)
    return np.sum(weights[...,None]*values,axis=0), weights`,solutionAlt:[{label:'PyTorch 版（仅展示）',code:py`import torch
def sinkhorn(logits,steps):
    z=logits.double()
    for _ in range(steps):
        z=z-torch.logsumexp(z,dim=1,keepdim=True)
        z=z-torch.logsumexp(z,dim=0,keepdim=True)
    return z.exp()
def depth_attention(values,query,eps=1e-6):
    keys=values*torch.rsqrt(values.square().mean(-1,keepdim=True)+eps)
    weights=(keys*query).sum(-1).softmax(dim=0)
    return (weights[...,None]*values).sum(0),weights`}],explanation:md`logsumexp 避免 exp(巨大 logits) 溢出。每轮最后一次列归一化使列和接近精确 1，行和要通过迭代逼近，不能声称一轮就双随机。深度注意力的归一化轴是层 L，同一 token 在不同深度间选择信息，不能沿 token T 混合。Sinkhorn 时间 O(steps×N²)，深度聚合 O(LTD)。`,hints:[md`在 log 域中除以行和等价于减行 logsumexp。`,md`深度是 axis=0；keys 用于打分，values 原尺度用于聚合。`],tests:[{name:'均匀与单流',code:py`a=sinkhorn(np.zeros((3,3)),2);assert a.shape==(3,3) and np.allclose(a,1/3), f"应全为1/3，得到 {a}"
assert np.allclose(sinkhorn(np.array([[9999.]]),1),[[1]]), "单流应为1"`},{name:'均匀深度权重',code:py`v=np.arange(24.).reshape(3,2,4);out,p=depth_attention(v,np.zeros(4))
assert out.shape==(2,4) and p.shape==(3,2), f"形状错误 {out.shape},{p.shape}"
assert np.allclose(out,v.mean(0)) and np.allclose(p,1/3), f"应均值与1/3，得到 {out},{p}"`},{name:'稳定性与双随机收敛',hidden:true,code:py`z=np.array([[1.,2.,0.],[3.,1.,2.],[1.,0.,3.]])
a=sinkhorn(z,100);b=sinkhorn(z+10000,100)
assert np.isfinite(b).all() and np.allclose(a,b), "加常数不应改变结果或溢出"
assert np.allclose(a.sum(0),1,atol=1e-7) and np.allclose(a.sum(1),1,atol=1e-7), f"行列和 {a.sum(0)},{a.sum(1)}"`},{name:'内容选择与零向量',hidden:true,code:py`v=np.array([[[1.,0.]],[[0.,1.]]]);q=np.array([1.,0.]);out,p=depth_attention(v,q)
s=1/np.sqrt(.5+1e-6);exp=np.exp([s,0]);exp/=exp.sum()
assert np.allclose(p[:,0],exp) and np.allclose(out[0],exp), f"期望 {exp}，得到 {out},{p}"
out,p=depth_attention(np.zeros((2,3,4)),np.ones(4));assert np.isfinite(out).all() and np.allclose(p,.5), "零向量需 eps"`}],followUps:[{id:'guarantee',q:md`双随机矩阵是否意味着完整网络范数一定不增长？`,a:md`不是。它约束残差流之间的混合，完整网络还包含非线性子层和输出注入。即使残差混合受控，也不能推出所有激活或梯度都不增长；有限 Sinkhorn 迭代还存在行列和误差。`}]}
export default defineLevel({learn:[{id:'hc',title:'从一条残差流到多流连接',body:md`传统残差为 x+F(x)，跨层累加有稳定的恒等路径，但混合权重固定。Hyper-Connections 扩展为多条残差流，通过可学习 pre、post、res 映射决定输入子层、接收子层输出和跨流混合。它增加表达灵活性，也让跨层连接的乘积可能放大或衰减信号，并增加内存读写。这里要分清「流」与注意力头：残差流贯穿网络深度，不是把 QKV 多切几个头。参考：[Hyper-Connections](https://arxiv.org/abs/2409.19606)。`},{id:'mhc',title:'mHC 约束残差混合矩阵',body:md`mHC 将残差混合矩阵约束为非负、行和与列和均为 1 的双随机矩阵。行和使每个输出流成为输入流的凸组合；列和控制各输入流的总体分配。正矩阵上交替归一化行与列可逼近该集合，称 Sinkhorn-Knopp。有限轮通常只是近似，最后一步精确满足的轴不能代表另一轴也已经精确满足。这一约束不是把矩阵设成单位阵，而是保留受控的可学习混合；完整网络的非线性支路仍可改变信号尺度。参考：[mHC 原论文](https://arxiv.org/abs/2512.24880)。`,viz:SinkhornViz},{id:'attnres',title:'Attention Residuals 沿深度选择信息',body:md`Attention Residuals 用内容相关的 softmax 权重聚合此前层输出，替代固定权重累加。查询与经过归一化的历史表征打分，归一化轴是深度而非时间；value 保留原表征用于加权。Full 形式要保存较多历史输出，Block AttnRes 用块级表征减少存储和通信。它与 mHC 的思路不同：一个在历史深度上做选择，一个约束多残差流间的连接。不能把任一方案视作无需训练即可装到已有模型上的推理开关。参考：[Attention Residuals](https://arxiv.org/abs/2603.15031)。`}],quiz:[single('axis',1,md`Attention Residuals 的主要选择轴是？`,['词表','网络深度','batch','注意力头内维度'],1,md`同一 token 在此前不同层输出间聚合，不是标准的时间自注意力。`),judge('identity',1,md`双随机矩阵只能是单位阵。`,false,md`例如全为 1/N 的矩阵也双随机，会平均混合所有流。`),numeric('uniform',2,md`4×4 均匀双随机矩阵每个元素是多少？`,.25,md`每行 4 个相同元素、和为1，所以每项1/4。`),multi('doubly',2,md`双随机要求哪些？`,['元素非负','每行和为1','每列和为1','每个特征通道方差为1'],[0,1,2],md`方差归一化是另一种约束，不属于双随机定义。`),order('sinkhorn',md`log 域 Sinkhorn 一轮：`,['计算行 logsumexp','每行减去对应 logsumexp','计算列 logsumexp','每列减去对应 logsumexp'],md`第二次计算必须使用已做行归一化的结果。`),fill('axis0',md`values=[L,T,D]，沿深度 softmax 的 NumPy axis 是 ____(1)____。`,['0'],md`L 是第0轴，不能沿 T 归一化。`),single('block',2,md`Block AttnRes 主要缓解 Full 形式的什么代价？`,['词表 OOV','保存与通信全部层历史表征','所有矩阵乘','数据标签噪声'],1,md`块级聚合减少深度历史状态数量，但并不消除模型全部计算。`),judge('finite',3,md`任意正矩阵做一次行归一化、一次列归一化后两轴和必然都精确为1。`,false,md`列归一化会再次改变行和，需要迭代收敛；有限轮仍可能有误差。`),bug('wrongaxis',py`import numpy as np
def depth_weights(values, query):
    rms = np.sqrt((values*values).mean(-1,keepdims=True)+1e-6)
    keys = values/rms
    score = (keys*query).sum(-1)
    e = np.exp(score-score.max(axis=1,keepdims=True))
    p = e/e.sum(axis=1,keepdims=True)
    return p`,[6,7],md`scores 形状 [L,T]，应沿 axis=0 做稳定 softmax；当前在不同 token 间竞争，与深度聚合目标不同。`) ],guided:[practice(c,[{find:'z -= _logsumexp(z, axis=1)',instruction:md`在 log 域归一化每行。`,explain:md`对 N×N 矩阵，行内求和沿 axis=1。`},{find:'z -= _logsumexp(z, axis=0)',instruction:md`接着归一化每列。`,explain:md`列内求和沿 axis=0。`},{find:'weights.sum(axis=0,keepdims=True)',instruction:md`让每个 token 的深度权重和为1。`,explain:md`深度 L 在第0轴。`}])],challenges:[c],interview:[{id:'stability',freq:3,q:md`mHC 的双随机约束提供什么直觉？`,a:md`非负且行和为1让每个输出流成为已有流的凸组合，避免残差混合层任意扩大系数；列和为1让输入流总分配受控。它保留灵活混合而非固定恒等。但整体网络包含额外子层输出，不能据此承诺全网络任意深度完全稳定，训练与实现精度仍需验证。`},{id:'difference',freq:2,q:md`mHC 与 AttnRes 是同一技术吗？`,a:md`不是。mHC 以多条残差流及其受约束混合矩阵为核心；AttnRes 在此前深度表征中通过内容相关权重进行聚合。二者都改变残差信息流，但状态形状、存储成本和归一化约束不同。比较时应该画出各自的输入输出维度，而不是只说都加了注意力。`},{id:'numerics',freq:2,q:md`Sinkhorn 为什么适合在 log 域实现？`,a:md`对 logits 直接取 exp 容易溢出，而 logsumexp 用减最大值稳定计算。归一化在 log 域成为减法，可保留大幅度输入的有效信息。有限迭代数是性能与误差的取舍，应监控两轴残差；输入有结构性零或支持不满足条件时还需要更谨慎的收敛分析。`}],cheatsheet:md`HC：多流连接；mHC：非负、行列和1的残差混合；Sinkhorn 交替行列归一化，有限轮近似。AttnRes：softmax 沿深度，Block 形式减少历史状态。`})
