import { defineLevel, md, py } from '@/content/helpers'
import type { Challenge } from '@/content/types'
import { LoraViz } from '../viz/QuantViz'
import { single,judge,numeric,multi,order,fill,bug,practice } from '../shared'
const c:Challenge={id:'lora',title:'LoRA 前向、合并与反向梯度',difficulty:'medium',description:md`实现 \`lora(x: np.ndarray, w: np.ndarray, a: np.ndarray, b: np.ndarray, alpha: float, grad: np.ndarray) -> tuple\`。x=[N,I]，w=[O,I] 为冻结权重，a=[R,I]，b=[O,R]，上游梯度 grad=[N,O]，维度 1…32，alpha≥0。返回 (y,merged,da,db)：y=xWᵀ+(alpha/R)(xAᵀ)Bᵀ；merged=W+(alpha/R)BA；da、db 为标量目标 sum(y×grad) 对 A、B 的梯度。不计算 W 梯度，不修改输入；float64。示例 B=0 时 y=xWᵀ，dA=0，但 dB 通常非零。`,starter:py`import numpy as np
def lora(x: np.ndarray, w: np.ndarray, a: np.ndarray, b: np.ndarray, alpha: float, grad: np.ndarray):
    raise NotImplementedError`,solution:py`import numpy as np
def lora(x: np.ndarray, w: np.ndarray, a: np.ndarray, b: np.ndarray, alpha: float, grad: np.ndarray):
    scale = alpha / a.shape[0]
    hidden = x @ a.T
    y = x @ w.T + scale * (hidden @ b.T)
    merged = w + scale * (b @ a)
    db = scale * grad.T @ hidden
    da = scale * (grad @ b).T @ x
    return y, merged, da, db`,solutionAlt:[{label:'PyTorch 版（仅展示）',code:py`import torch
def lora(x,w,a,b,alpha,grad):
    scale=alpha/a.shape[0]
    y=torch.nn.functional.linear(x,w)+scale*torch.nn.functional.linear(torch.nn.functional.linear(x,a),b)
    da,db=torch.autograd.grad(y,(a,b),grad_outputs=grad)
    merged=w+scale*(b@a)
    return y,merged,da,db`}],explanation:md`LoRA 把大矩阵更新限制到 rank≤R 的 BA 子空间，前向先乘 A 再乘 B，避免每次显式构造完整 ΔW。链式法则先 dB=scale·gradᵀ·hidden，再 dA=scale·(gradB)ᵀx。合并可验证 y≈x@merged.T；训练中重复 merge 会造成双计。时间含基座 O(NIO) 与低秩 O(NR(I+O))，显式 merged 额外 O(IO)。`,hints:[md`固定权重布局 [out,in]，按矩阵维度推出转置。`,md`dB 用上游 grad 与低秩 hidden 的外积累计。`,md`B=0 初始不意味着所有 LoRA 梯度都为零。`],setup:py`import numpy as np
_rng=np.random.default_rng(45)
_x=_rng.normal(size=(3,5));_w=_rng.normal(size=(4,5));_a=_rng.normal(size=(2,5));_b=_rng.normal(size=(4,2));_g=_rng.normal(size=(3,4))`,tests:[{name:'前向与合并',code:py`y,m,da,db=lora(_x,_w,_a,_b,4,_g)
assert y.shape==(3,4) and m.shape==(4,5) and da.shape==_a.shape and db.shape==_b.shape, "输出 shape 错误"
exp=_x@(_w+2*_b@_a).T
assert np.allclose(y,exp) and np.allclose(y,_x@m.T), f"合并误差 {np.max(abs(y-exp))}"`},{name:'零 B 初始化',code:py`y,m,da,db=lora(_x,_w,_a,np.zeros_like(_b),4,_g)
assert np.allclose(y,_x@_w.T) and np.allclose(da,0), "零 B 应保持基座且 dA=0"
assert np.linalg.norm(db)>0, "dB 通常非零"`},{name:'有限差分梯度',hidden:true,code:py`y,m,da,db=lora(_x,_w,_a,_b,3,_g)
def loss(a,b):return np.sum((_x@_w.T+1.5*(_x@a.T)@b.T)*_g)
for param,grad,is_a in [(_a,da,True),(_b,db,False)]:
    for idx in np.ndindex(param.shape):
        plus=param.copy();minus=param.copy();plus[idx]+=1e-6;minus[idx]-=1e-6
        val=(loss(plus,_b)-loss(minus,_b))/(2e-6) if is_a else (loss(_a,plus)-loss(_a,minus))/(2e-6)
        assert np.isclose(val,grad[idx],atol=1e-6), f"{idx} 期望 {val}，得到 {grad[idx]}"`},{name:'alpha=0 与输入不变',hidden:true,code:py`old=[v.copy() for v in [_x,_w,_a,_b]]
y,m,da,db=lora(_x,_w,_a,_b,0,_g)
assert np.allclose(y,_x@_w.T) and np.allclose(da,0) and np.allclose(db,0), "alpha0 应关闭适配器"
assert all(np.array_equal(a,b) for a,b in zip(old,[_x,_w,_a,_b])), "输入被修改"`}],followUps:[{id:'bothzero',q:md`为什么 A 与 B 不能都初始化为 0？`,a:md`双线性更新 BA 的 dA 包含 B、dB 包含 A，两者为零时两个梯度均为零，适配器不能从该状态学习。常见初始化 A 随机、B 为零，既保持原前向，又让 B 先获得梯度。`}]}
export default defineLevel({learn:[{id:'rank',title:'在低秩子空间学习权重更新',body:md`对 W∈R^(O×I)，LoRA 学习 $\Delta W=(\alpha/r)BA$，A 为 r×I，B 为 O×r，只训练 r(I+O) 个参数。常见初始化 A 随机、B=0，让初始前向与基座一致；但 B 可以先获得非零梯度，之后 A 也开始更新。训练参数减少不等于激活全部消失：梯度仍需穿过冻结层把学习信号传给适配器，优化器状态节省与激活显存节省必须分开估算。`,viz:LoraViz},{id:'merge',title:'合并是线性等价，不是重新训练',body:md`推理时可以把 BA 加入浮点 W，从两条线性路径变成单个 merged 权重，代数上 y=x(W+ΔW)ᵀ。启用 dropout 的训练路径、量化基座重新量化后的权重，都不能直接声称和合并前逐 bit 一致。多适配器服务常保留基座与 LoRA 分离，以便动态切换；反复合并、忘记关闭低秩分支会把更新计算两次。秩 r 提高增强表达能力，也增加训练参数和执行成本。参考：[LoRA](https://arxiv.org/abs/2106.09685)。`},{id:'family',title:'QLoRA 与 DoRA 改变不同环节',body:md`QLoRA 冻结量化的基座权重，计算时反量化到较高精度，更新低秩适配器；NF4 面向近似正态权重分布设计非均匀码本，双重量化进一步压缩量化尺度，paged optimizer 管理内存峰值。它不意味着全部梯度和激活都以 4-bit 存储。DoRA 将权重大小与方向解耦，用低秩更新方向并学习幅度，改变适配器参数化。二者分别关注存储路径和更新表达方式，不能把 NF4 当成硬件原生 FP4。参考：[QLoRA](https://arxiv.org/abs/2305.14314)、[DoRA](https://arxiv.org/abs/2402.09353)。`}],quiz:[single('shape',1,md`W=[O,I]，LoRA ΔW 应为？`,['AB','BA','A+B','BᵀAᵀ'],1,md`[O,R]×[R,I]=[O,I]，与基座形状一致。`),judge('zero',1,md`A 随机、B=0 时，初始 LoRA 前向为零但 B 的梯度可以非零。`,true,md`dB 依赖 A 与上游梯度；dA 初始因 B=0 为零。`),numeric('count',2,md`I=O=1024、r=8，A/B 可训练参数总量是多少？`,16384,md`8×(1024+1024)=16384，不是 1024²。`),multi('qlora',2,md`QLoRA 包含的概念是？`,['冻结量化基座','NF4','量化尺度的双重量化','所有激活强制 INT4'],[0,1,2],md`存储低位宽不等于计算、激活和梯度全部低位宽。`),order('forward',md`不显式构造 ΔW 的 LoRA 前向：`,['x 乘 Aᵀ 得低秩表示','低秩表示乘 Bᵀ','乘 alpha/r','加到基座输出 xWᵀ'],md`先降维再升维是低秩执行的关键；先构造 BA 会增加大矩阵工作区。`),fill('scale',md`alpha=16、r=8，缩放系数是 ____(1)____。`,['2'],md`16/8=2。不同变体可有其它缩放，必须说明使用哪种。`),single('dora',2,md`DoRA 主要解耦的是？`,['训练与测试','权重幅度与方向','Q 与 K','token 与字符'],1,md`它改变权重更新参数化，不是普通数据集拆分。`),judge('merge',3,md`量化基座上先合并再重新量化，通常与保留高精度 LoRA 支路逐 bit 等价。`,false,md`重新量化引入额外舍入，量化不是线性操作。`),bug('grad',py`def gradients(x, a, b, grad, alpha):
    rank = a.shape[0]
    scale = alpha/rank
    hidden = x @ a.T
    db = scale * grad.T @ hidden
    da = scale * (grad @ b).T @ x
    db = db / rank
    return da, db`,[7],md`scale 已含 1/r，第7行再次除 rank 使 dB 缩小。两个梯度都只乘一次 alpha/r。`) ],guided:[practice(c,[{find:'alpha / a.shape[0]',instruction:md`从 A 的形状计算更新倍率。`,explain:md`A 第0轴是秩 r。`},{find:'w + scale * (b @ a)',instruction:md`形成等价浮点合并权重。`,explain:md`BA 与 W 同形状；避免重复加入。`},{find:'scale * grad.T @ hidden',instruction:md`推导 B 的梯度。`,explain:md`输出线性层梯度为上游转置乘输入。`},{find:'scale * (grad @ b).T @ x',instruction:md`把梯度继续传到 A。`,explain:md`先经过 B，再对 A 求权重梯度。`}])],challenges:[c],interview:[{id:'memory',freq:3,q:md`LoRA 节省了什么显存，没节省什么？`,a:md`它显著减少需要保存的可训练参数梯度和优化器状态，基座权重仍需存在；QLoRA 才进一步压缩基座存储。激活通常仍需为反向传播保存或重计算，长序列下可能成为主项。估算时应列出基座、适配器、优化器、梯度与激活，不能按可训练参数比例直接缩放全部显存。`},{id:'init',freq:3,q:md`B=0 会不会使 LoRA 学不起来？`,a:md`不会，只要 A 非零且输入、上游梯度有效，dB=scale·gradᵀ·xAᵀ 可以非零。初始 dA 包含 B 所以为零，B 更新后 A 开始变化。如果两者同时为零，两条梯度链都被截断，这才是退化初始化。`},{id:'rank',freq:2,q:md`如何选择 rank 与作用模块？`,a:md`rank 控制低秩更新的容量，目标模块决定在哪些变换上增加自由度。更大 rank 或覆盖 attention/FFN 更多矩阵通常增加预算，但收益依数据量与任务而定，应在相同训练资源下做验证。不要把一个模型的最佳 rank 当通用结论，并同时记录 alpha 与 dropout 等配置。`}],cheatsheet:md`ΔW=(alpha/r)BA；参数 r(I+O)。A 随机、B=0：初始输出不变、dB 可非零。merge 要关闭支路；QLoRA 压缩基座，NF4≠FP4；DoRA 解耦幅度与方向。`})
