import { defineLevel, md, py } from '@/content/helpers'
import type { Challenge } from '@/content/types'
import { RopeViz } from '../viz/ArchViz'
import { single,judge,numeric,multi,order,fill,bug,practice } from '../shared'
const c:Challenge={id:'rope',title:'两种布局的 partial RoPE',difficulty:'medium',description:md`实现 \`apply_rope(x: np.ndarray, positions: np.ndarray, rotary_dim: int, base: float=10000., layout: str="interleaved") -> np.ndarray\`。x 形状 [T,H,D]，positions 为 [T] 非负位置（可不连续），0≤rotary_dim≤D 且为偶数；T,H,D≤32，base>1。只旋转前 rotary_dim 维，其余保持原值。角频率为 base^(-2i/rotary_dim)。interleaved 将 (0,1)、(2,3) 配对；split 将前一半与后一半对应维度配对。返回同形状 float64，不修改输入。位置 0 必须为恒等；rotary_dim=0 直接复制。`,starter:py`import numpy as np
def apply_rope(x: np.ndarray, positions: np.ndarray, rotary_dim: int, base: float=10000., layout: str="interleaved") -> np.ndarray:
    raise NotImplementedError`,solution:py`import numpy as np
def apply_rope(x: np.ndarray, positions: np.ndarray, rotary_dim: int, base: float=10000., layout: str="interleaved") -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    if rotary_dim == 0: return out
    inv = base ** (-np.arange(0, rotary_dim, 2, dtype=float) / rotary_dim)
    angle = positions[:, None, None] * inv[None, None, :]
    cos, sin = np.cos(angle), np.sin(angle)
    if layout == "interleaved":
        a, b = x[..., :rotary_dim:2], x[..., 1:rotary_dim:2]
        out[..., :rotary_dim:2] = a*cos-b*sin
        out[..., 1:rotary_dim:2] = a*sin+b*cos
    else:
        half = rotary_dim // 2
        a, b = x[..., :half], x[..., half:rotary_dim]
        out[..., :half] = a*cos-b*sin
        out[..., half:rotary_dim] = a*sin+b*cos
    return out`,solutionAlt:[{label:'PyTorch 版（仅展示）',code:py`import torch
def apply_rope(x, positions, rotary_dim, base=10000., layout="interleaved"):
    out=x.double().clone()
    if rotary_dim==0:return out
    inv=base**(-torch.arange(0,rotary_dim,2,device=x.device,dtype=torch.float64)/rotary_dim)
    angle=positions[:,None,None]*inv[None,None,:]
    c,s=angle.cos(),angle.sin()
    if layout=="interleaved":
        a,b=x[...,:rotary_dim:2],x[...,1:rotary_dim:2]
        out[...,:rotary_dim:2]=a*c-b*s
        out[...,1:rotary_dim:2]=a*s+b*c
    else:
        h=rotary_dim//2;a,b=x[...,:h],x[...,h:rotary_dim]
        out[...,:h]=a*c-b*s;out[...,h:rotary_dim]=a*s+b*c
    return out`}],explanation:md`旋转矩阵为 [[cos,-sin],[sin,cos]]。positions 加两个广播轴，与 [T,H,d/2] 的配对值相乘；频率除数必须是 rotary_dim，不是完整 D。先从原输入取 a,b，写到副本，避免先覆盖 a 再计算 b。时间 $O(THD)$、输出空间 $O(THD)$。`,hints:[md`先写一对二维向量的旋转，再广播到 T、H。`,md`partial RoPE 的频率基于旋转子空间的维度。`],tests:[{name:'二维旋转与零位置',code:py`x=np.array([[[1.,0.]],[[1.,0.]]]);out=apply_rope(x,np.array([0,1]),2)
exp=np.array([[[1.,0.]],[[np.cos(1),np.sin(1)]]])
assert out.shape==x.shape and np.allclose(out,exp), f"期望 {exp}，得到 {out}"`},{name:'partial 尾部',code:py`x=np.arange(12.).reshape(2,1,6);out=apply_rope(x,np.array([2,8]),4)
assert np.array_equal(out[...,4:],x[...,4:]), f"未旋转尾部被改变：{out}"`},{name:'布局置换等价与零维',hidden:true,code:py`rng=np.random.default_rng(17);x=rng.normal(size=(3,2,6));p=np.array([0,7,9]);perm=[0,2,1,3,4,5]
a=apply_rope(x,p,4);b=apply_rope(x[...,perm],p,4,layout='split')
assert np.allclose(a[...,perm],b), f"布局不等价，误差 {np.max(abs(a[...,perm]-b))}"
assert np.array_equal(apply_rope(x,p,0),x), "零旋转应恒等"`},{name:'范数和相对位置',hidden:true,code:py`rng=np.random.default_rng(18);q=rng.normal(size=(1,2,8));k=rng.normal(size=(1,2,8))
def dot(m,n):
    return (apply_rope(q,np.array([m]),8)*apply_rope(k,np.array([n]),8)).sum(-1)
a,b=dot(2,5),dot(12,15)
assert np.allclose(a,b), f"共同平移应不变，得到 {a} 与 {b}"
y=apply_rope(q,np.array([11]),8)
assert np.allclose((y*y).sum(-1),(q*q).sum(-1)), "旋转应保持范数"`}],followUps:[{id:'cache',q:md`KV cache 中缓存旋转前还是旋转后的 K？`,a:md`常见实现缓存已经按绝对位置旋转的 K；新 Q 采用自己的位置。若前缀复用时位置语义或 RoPE 缩放配置不同，不能直接复用这些 K。其它实现可以缓存未旋转 K，但每次读出需正确旋转并付出计算成本。`}]}
export default defineLevel({learn:[{id:'derive',title:'用二维旋转把位置写入内积',body:md`RoPE 将特征两两配对，第 i 对按角度 $m\omega_i$ 旋转，其中 $\omega_i=\theta^{-2i/d_r}$。旋转保范数，因此不会仅因位置增大就放大向量。关键恒等式是 $(R_mq)^\top(R_nk)=q^\top R_{n-m}k$，由 $R_m^\top R_n=R_{n-m}$ 得到；位置作用落在相对差 n−m 上。这里「只依赖相对位置」指固定内容 q,k 的旋转内积，不意味着完整网络状态与绝对位置无关。`,viz:RopeViz},{id:'layout',title:'交错与 split 是基底排列差异',body:md`交错布局将 (x0,x1)、(x2,x3) 配对；split/rotate_half 把向量分两半，对应位置配成对。两种实现只要权重和输入维度做一致置换就等价，但直接把一个已训练模型的实现替换为另一种会改变结果。partial RoPE 只旋转前 d_r 维，剩余维保持原值，频率分母使用 d_r。广播错误也很常见：[T,H,D] 输入的位置角应为 [T,1,d_r/2]，否则可能把时间当作头轴。`},{id:'limits',title:'位置泛化不是无限长度保证',body:md`不同频带的旋转周期不同，高频区分局部位置，低频承载更长距离变化。base 越大，低频越低；但这不自动让模型掌握训练外的长距离任务。更长上下文还改变注意力归一化、检索难度、KV 显存和模型见到的位置组合。LLaMA 等 RoPE 模型的实现细节应以各自配置和代码为准，不能把某个 base 当成统一标准。参考：[RoFormer 原论文](https://arxiv.org/abs/2104.09864)。`}],quiz:[single('relative',1,md`RoPE 的相对位置性质来自？`,['旋转矩阵相乘的角度差','softmax 消掉所有位置','对 V 求均值','把位置加到 embedding'],0,md`R_m 转置乘 R_n 等于 R_(n-m)，不是由 softmax 导致。`),judge('norm',1,md`精确算术下二维旋转保持欧氏范数。`,true,md`旋转矩阵正交；有限精度会有舍入误差。`),numeric('freq',2,md`base=10000、rotary_dim=4，第 i=1 对的角频率是多少？`,.01,md`10000^(-2/4)=.01；周期是 2π/.01，频率和周期不要混淆。`),multi('partial',2,md`partial RoPE 正确实现应？`,['只旋转指定偶数维数','未旋转尾部保持不变','频率分母永远取完整 hidden size','位置轴与头轴正确广播'],[0,1,3],md`频率分母是实际旋转维数，不能用模型 hidden size 代替。`),order('steps',md`组装旋转前向：`,['生成每对维度的角频率','位置乘角频率得到角度','按布局取出配对 a 与 b','计算 a cos−b sin 和 a sin+b cos'],md`布局与角频率必须配套，最后写到副本防止原地污染。`),fill('identity',md`位置 m=0 时旋转矩阵为 ____(1)____ 矩阵。`,['单位','identity','I'],md`cos0=1、sin0=0，因此 x 完全保留。`),single('split',2,md`D=4 的 split 布局中，维度 0 与哪一维配对？`,['1','2','3','自身'],1,md`前半 [0,1] 与后半 [2,3] 对应配对。`),judge('unlimited',3,md`内积的相对位置性质足以保证任意上下文长度下模型效果不下降。`,false,md`训练分布、频率范围和注意力统计都可能改变；代数性质不等于任务泛化保证。`),bug('overwrite',py`import numpy as np
def rotate(x, c, s):
    y = x.copy()
    a = y[..., 0::2]
    b = y[..., 1::2]
    y[..., 0::2] = a*c-b*s
    y[..., 1::2] = a*s+b*c
    return y`,[4,5],md`a、b 是 y 的视图，第 6 行写入后 a 已改变，导致第 7 行错误。应从未修改的 x 读取或 copy 两个切片。`) ],guided:[practice(c,[{find:'base ** (-np.arange(0, rotary_dim, 2, dtype=float) / rotary_dim)',instruction:md`构造 d_r/2 个角频率。`,explain:md`指数是 -2i/d_r。`},{find:'positions[:, None, None] * inv[None, None, :]',instruction:md`对时间和头轴广播角度。`,explain:md`得到 [T,1,d_r/2]，共享给各头。`},{find:'a*cos-b*sin',instruction:md`计算旋转后的第一坐标。`,explain:md`右手平面旋转第一行是 cos,-sin。`}])],challenges:[c],interview:[{id:'relative',freq:3,q:md`现场推导 RoPE 的相对位置性质。`,a:md`固定内容向量 q,k，分别旋转 R_m 和 R_n。内积写成 qᵀR_mᵀR_nk，利用旋转群的角度相加性质与转置等于逆，得到 qᵀR_(n−m)k。多对维度是分块对角矩阵，逐块成立。网络本身仍通过因果上下文形成内容，因此不能把这条恒等式过度解释为端到端绝对位置不变。`},{id:'v',freq:2,q:md`为什么常见 RoPE 只加在 Q、K？`,a:md`设计目标是让注意力匹配分数显式包含相对位置，Q、K 就足够。V 承载聚合内容，常见实现不旋转 V，使输出保留统一内容坐标系；这是架构约定，不代表旋转 V 在数学上不允许，但会改变模型和输出解释。`},{id:'layout',freq:3,q:md`两个 RoPE 实现公式都对，结果为什么不同？`,a:md`先检查维度配对方式、频率重复方式、rotary_dim、position offset、base 和缩放配置，再看是否在 Q、K 同时应用。交错与 split 虽能通过置换等价，但预训练权重绑定具体布局。KV cache 的历史位置偏移错一位也会造成逐步生成与全量前向不一致。`}],cheatsheet:md`$\omega_i=\theta^{-2i/d_r}$；角度为 $m\omega_i$。$(R_mq)^\top(R_nk)=q^\top R_{n-m}k$。配对布局必须与权重一致；partial 尾部不变；禁止原地视图相互污染。`})
