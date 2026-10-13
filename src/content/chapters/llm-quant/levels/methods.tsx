import { defineLevel, md, py } from '@/content/helpers'
import type { Challenge } from '@/content/types'
import { SmoothViz } from '../viz/QuantViz'
import { single,judge,numeric,multi,order,fill,bug,practice } from '../shared'
const c:Challenge={id:'smooth',title:'SmoothQuant 校准与等价变换',difficulty:'medium',description:md`实现 \`smoothquant(calibration: np.ndarray, x: np.ndarray, w: np.ndarray, alpha: float, eps: float=1e-8) -> tuple\`。calibration=[C,I] 是独立校准激活、x=[N,I] 是当前输入、w=[I,O]，维度1…64，均有限，0≤alpha≤1。按输入通道计算 amax=max(abs(calibration),axis=0)、wmax=max(abs(w),axis=1)，分别 clamp 到 eps，s=amax^alpha/wmax^(1−alpha)。返回 (x/s, s[:,None]×w, s)，保持浮点恒等 (x/s)@(sW)=x@W；不在函数内量化，不修改输入。例单通道校准幅度16、权重幅度1、alpha=.5 → s=4。`,starter:py`import numpy as np
def smoothquant(calibration: np.ndarray, x: np.ndarray, w: np.ndarray, alpha: float, eps: float=1e-8):
    raise NotImplementedError`,solution:py`import numpy as np
def smoothquant(calibration: np.ndarray, x: np.ndarray, w: np.ndarray, alpha: float, eps: float=1e-8):
    amax = np.maximum(np.max(np.abs(calibration),axis=0),eps)
    wmax = np.maximum(np.max(np.abs(w),axis=1),eps)
    scale = amax**alpha / wmax**(1-alpha)
    x_new = x / scale[None,:]
    w_new = w * scale[:,None]
    return x_new, w_new, scale`,solutionAlt:[{label:'PyTorch 版（仅展示）',code:py`import torch
def smoothquant(calibration,x,w,alpha,eps=1e-8):
    amax=calibration.abs().amax(dim=0).clamp_min(eps)
    wmax=w.abs().amax(dim=1).clamp_min(eps)
    scale=amax.pow(alpha)/wmax.pow(1-alpha)
    return x/scale[None,:],w*scale[:,None],scale`}],explanation:md`对角矩阵 S 插在 X 与 W 之间：XS⁻¹SW=XW。尺度按共享的输入通道 I 计算，因此 X 除列、W 乘行。alpha 越大越把动态范围压力从激活迁向权重。eps 防止全零通道的 0/0，但不是量化后的精度保证；量化后还会有舍入。时间 O(CI+NI+IO)，空间 O(NI+IO)。`,hints:[md`根据 x@w 的收缩轴定位共享通道。`,md`校准统计来自 calibration，而不是运行时 x。`,md`检查乘积恒等，但也检查指定 scale，否则全返回原值会逃过恒等测试。`],tests:[{name:'单通道尺度',code:py`xn,wn,s=smoothquant(np.array([[16.]]),np.array([[8.]]),np.array([[1.,.5]]),.5)
assert np.allclose(s,[4]) and np.allclose(xn,[[2]]) and np.allclose(wn,[[4,2]]), f"期望 s4，得到 {s},{xn},{wn}"`},{name:'不对称形状等价',code:py`rng=np.random.default_rng(47);cal=rng.normal(size=(7,3));x=rng.normal(size=(2,3));w=rng.normal(size=(3,4))
a,b,s=smoothquant(cal,x,w,.6)
assert a.shape==x.shape and b.shape==w.shape and s.shape==(3,), "形状错误"
assert np.allclose(a@b,x@w), f"最大误差 {np.max(abs(a@b-x@w))}"`},{name:'全零通道有限',hidden:true,code:py`a,b,s=smoothquant(np.zeros((3,2)),np.zeros((1,2)),np.zeros((2,4)),.5)
assert np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(s).all(), "零通道不能产生NaN"
assert np.allclose(a@b,0), "零矩阵结果应为零"`},{name:'alpha 边界与独立校准',hidden:true,code:py`cal=np.array([[2.,8.],[4.,1.]]);x=np.array([[100.,100.]]);w=np.array([[2.,1.,0.],[1.,4.,2.]])
old=cal.copy()
a,b,s=smoothquant(cal,x,w,1);assert np.allclose(s,[4,8]), f"应按校准而非x统计，得到{s}"
a,b,s=smoothquant(cal,x,w,0);assert np.allclose(s,[.5,.25]), f"alpha0应为1/wmax，得到{s}"
assert np.array_equal(cal,old), "修改了校准数据"`}],followUps:[{id:'fold',q:md`部署时怎样避免每次额外除 X 的开销？`,a:md`若前驱归一化与计算图允许，可把输入通道逆尺度融合进前驱 norm 的缩放参数，把正尺度融合到线性权重。存在共享输入的多条分支时必须一致处理各消费者，不能只改一条边就宣称整个网络等价。`}]}
export default defineLevel({learn:[{id:'outliers',title:'激活离群值让低比特网格失衡',body:md`LLM 中某些输入通道可能出现幅度较大的激活，per-tensor INT8 的步长被这些值主导，普通通道失去分辨率。LLM.int8 将离群通道的乘法分离到高精度路径，其余用低精度执行；SmoothQuant 则用等价缩放把激活范围迁移到权重侧，更适合 W8A8 路径。两者都针对离群值，却选择不同的成本分配，不能仅凭「8-bit」认为存储、计算和内核相同。参考：[LLM.int8](https://arxiv.org/abs/2208.07339)。`},{id:'smooth',title:'等价变换不等于量化无误差',body:md`对 X=[N,I]、W=[I,O]，任选正对角 S 都有 XW=(XS⁻¹)(SW)。SmoothQuant 根据校准激活幅度 a_i 和权重幅度 w_i 选择 $s_i=a_i^\alpha/w_i^{1-\alpha}$。alpha 越大越压缩激活、放大权重；alpha=0 则更偏向平衡权重。这个浮点变换代数等价，但之后把两个因子量化会产生误差，所以需要校准与任务验证。测试既要检查输出恒等，也要检查尺度公式，否则直接返回原数组也能蒙混过关。参考：[SmoothQuant](https://arxiv.org/abs/2211.10438)。`,viz:SmoothViz},{id:'ptq',title:'GPTQ、AWQ 与 KV 量化各自优化什么',body:md`GPTQ 利用校准激活构造近似二阶信息，逐列量化权重时将误差补偿到尚未量化的列，以降低层输出重建误差；并非逐元素最近舍入。AWQ 用激活信息识别重要权重通道，通过尺度搜索保护它们，主要面向低比特权重量化。KV 量化处理随上下文增长的缓存，K 与 V 的误差路径不同：K 影响 attention logits，V 影响加权值；粒度、最近 token 保留高精度与反量化开销需一起评估。参考：[GPTQ](https://arxiv.org/abs/2210.17323)、[AWQ](https://arxiv.org/abs/2306.00978)。`}],quiz:[single('identity',1,md`SmoothQuant 等价变换是？`,['X/S 与 S×W','X×S 与 S×W','X/S 与 W/S','只除 X'],0,md`在收缩输入通道上，S⁻¹S 抵消。两边都同方向缩放无法保持乘积。`),judge('exact',1,md`浮点等价变换意味着后续量化也无误差。`,false,md`舍入和裁剪不具有同样的线性等价性。`),numeric('scale',2,md`amax=16、wmax=4、alpha=.5，尺度是多少？`,2,md`s=sqrt(16)/sqrt(4)=2。`),multi('methods',2,md`正确的方法描述是？`,['GPTQ 用近似二阶信息做逐列误差补偿','AWQ 使用激活信息选择尺度','LLM.int8 分离离群通道','SmoothQuant 从不需要校准统计'],[0,1,2],md`SmoothQuant 的尺度需要代表性的激活统计，不是单靠位宽就能确定。`),order('calibration',md`SmoothQuant 部署准备顺序：`,['收集代表性校准激活','沿输入通道计算激活和权重幅度','计算尺度并实施等价变换','量化并验证层输出及任务质量'],md`先浮点验证等价，再评估量化误差，有助定位问题来源。`),fill('axis',md`W=[I,O]，每输入通道求权重 maxabs，应沿 axis=____(1)____。`,['1'],md`每行是一个输入通道，沿输出 O 轴求最大值。`),single('kv',2,md`K 量化误差首先直接影响什么？`,['attention logits','tokenizer merge rank','优化器动量','embedding词表大小'],0,md`K 进入 QKᵀ；V 则直接影响概率加权后的内容。`),judge('compensation',3,md`GPTQ 的误差补偿可以改变已经固定的量化列而不重新考虑量化约束。`,false,md`逐列方法将本列冻结后把补偿施加到尚未量化的列，避免破坏已经确定的离散值。`),bug('axisbug',py`import numpy as np
def smooth(x, w, alpha):
    amax = np.abs(x).max(axis=0)
    wmax = np.abs(w).max(axis=0)
    scale = amax**alpha / wmax**(1-alpha)
    xnew = x/scale
    wnew = w*scale[:,None]
    return xnew,wnew`,[4],md`W=[I,O] 应沿 axis=1 得 [I]，当前得到 [O]；当 I=O 时形状不报错但语义仍错。还需处理零通道 eps。`) ],guided:[practice(c,[{find:'np.max(np.abs(calibration),axis=0)',instruction:md`统计每个输入通道的校准幅度。`,explain:md`沿校准样本轴约简，保留 I。`},{find:'np.max(np.abs(w),axis=1)',instruction:md`统计相同输入通道的权重幅度。`,explain:md`W 每行对应一个输入通道。`},{find:'amax**alpha / wmax**(1-alpha)',instruction:md`构造迁移尺度。`,explain:md`alpha 平衡激活与权重两侧的动态范围。`},{find:'w * scale[:,None]',instruction:md`对权重实施抵消缩放。`,explain:md`X 除列尺度，W 乘相同输入通道的行尺度。`}])],challenges:[c],interview:[{id:'gptq',freq:3,q:md`GPTQ 为什么比独立 round 更准确？`,a:md`独立舍入只关心某个权重离哪个格点近，忽略输入激活对输出误差的放大。GPTQ 用校准数据估计近似二阶几何，把量化某列产生的误差分配到剩余自由权重上，以降低层重建误差。它依赖代表性校准、阻尼与稳定求解，也增加离线量化成本。`},{id:'alpha',freq:3,q:md`SmoothQuant alpha 越大越好吗？`,a:md`不是。更大的 alpha 会更强地压低激活范围，同时把幅度负担迁向权重，可能让权重量化更难。最佳折中依模型、层、数据和量化粒度而变，需要在校准集上搜索并在独立任务上验证。极端 alpha 可用于理解公式，不能当作默认最优。`},{id:'kv',freq:2,q:md`为什么权重量化后仍可能显存不足？`,a:md`长上下文或大 batch 下 KV cache 随 token 数线性增长，而权重大小固定；压缩权重不能减少这些缓存。可考虑 KV 量化、GQA/MLA、窗口策略或更小并发，但每种方式影响精度、架构或可访问历史。估算时先列出权重与 KV 各自字节数。`}],cheatsheet:md`LLM.int8：离群通道高精度。SmoothQuant：X/S × SW，s=amax^α/wmax^(1−α)。GPTQ：二阶近似+逐列补偿。AWQ：激活感知尺度。KV 量化另算缓存与误差路径。`})
