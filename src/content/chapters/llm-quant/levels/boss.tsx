import { defineLevel, md, py } from '@/content/helpers'
import { QuantizationViz } from '../viz/QuantViz'
import { single,judge,numeric,multi,order,fill,bug } from '../shared'
export default defineLevel({learn:[{id:'pipeline',title:'把逻辑量化落实为可读取的存储格式',body:md`低比特微调同时涉及三种精度：基座的存储格式、矩阵乘的计算精度和适配器的更新精度。本关将浮点基座逐组量化成窄范围 INT4，再把两个 4-bit 编码打包为一个 uint8；前向正确读取组尺度与 nibble，恢复基座矩阵，并叠加高精度 LoRA。它展示量化存储与适配器计算的接口，不声称模拟真实 Tensor Core 内核或完整 QLoRA。实现应特别检查奇数输入维度的填充、尾组尺度和符号编码，这些边界往往比公式更容易导致上线错误。`,viz:QuantizationViz}],quiz:[single('sign',1,md`窄范围 INT4 的有符号值 −7…7，用 q+8 编码后范围是？`,['0…14','1…15','−8…7','8…22'],1,md`逐项加8，最小为1、最大为15；编码0在本题未使用。`),judge('precision',1,md`基座4-bit存储意味着 LoRA 梯度也必须4-bit。`,false,md`存储、计算与更新精度可以不同，QLoRA 正是冻结量化基座并训练较高精度适配器。`),numeric('pack',2,md`一行 5 个 INT4 编码按两个每字节打包，需要几个字节？`,3,md`ceil(5/2)=3，最后一个字节只使用低4位。`),multi('metadata',2,md`正确反解打包权重至少需要知道？`,['组尺度','原始输入维数','nibble 顺序与符号编码约定','训练使用的学习率'],[0,1,2],md`学习率与解码无关；布局、长度和尺度缺一会产生错误。`),order('pipeline',md`本 BOSS 前向过程：`,['按输出行和输入分组确定尺度','量化并将两个编码打包到 uint8','按原始长度解码并应用组尺度','计算基座输出并加 LoRA 支路'],md`量化误差发生在离散化阶段，打包本身应无损保存整数码。`),fill('lora',md`I=5、O=3、r=2，A/B 参数数量为 ____(1)____。`,['16'],md`r(I+O)=2×8=16。`),single('smoothing',2,md`W 使用 [I,O] 布局时，SmoothQuant 给 X 的第 i 列除 s_i，应对 W 做什么？`,['第 i 行乘 s_i','第 i 列除 s_i','整体乘平均尺度','不变'],0,md`缩放必须在矩阵乘收缩轴上抵消。`),judge('merge',3,md`Q(W+BA) 与 Q(W)+BA 对任意 W、A、B 恒等。`,false,md`量化舍入是非线性；合并再量化可能丢掉低秩支路的小更新。`),numeric('storage',2,md`1024 个权重，INT4 打包、组大小64，每组一个 FP16 尺度，忽略其它元数据，共多少字节？`,544,md`1024/2=512，加16组×2字节=32，共544。`),bug('nibble',py`def unpack_one(packed, index):
    byte = int(packed[index//2])
    shift = 4*(index%2)
    code = byte >> shift
    signed = code - 8
    value = float(signed)
    assert isinstance(value, float)
    return value`,[4],md`右移后还需 &15。偶数 index 的低 nibble 否则包含高 nibble，解码值会远超范围。`) ],guided:[],challenges:[{id:'packed-lora',title:'可验证的 INT4 打包与 LoRA 前向',difficulty:'hard',description:md`实现 \`packed_lora(x: np.ndarray, w: np.ndarray, a: np.ndarray, b: np.ndarray, alpha: float, group_size: int) -> tuple[np.ndarray,np.ndarray,np.ndarray]\`。x=[N,I]、w=[O,I]、a=[R,I]、b=[O,R]，维度1…32，group_size≥1，alpha≥0。每输出行沿 I 分组，以 maxabs/7 为尺度（全零组1），用 np.rint 与 clip(−7,7) 得 q。编码 q+8；偶数列放低 nibble，奇数列放高 nibble，最后不足两列的高 nibble 保持0。返回 packed [O,ceil(I/2)] uint8、scales [O,ceil(I/group)] float64、y [N,O]。y 必须从 packed 解码得到 W_hat，再算 xW_hatᵀ+(alpha/R)(xAᵀ)Bᵀ。不得用原始 W 计算基座输出，不得修改输入。`,starter:py`import numpy as np
def packed_lora(x: np.ndarray, w: np.ndarray, a: np.ndarray, b: np.ndarray, alpha: float, group_size: int):
    raise NotImplementedError`,solution:py`import numpy as np
def packed_lora(x: np.ndarray, w: np.ndarray, a: np.ndarray, b: np.ndarray, alpha: float, group_size: int):
    out_dim,in_dim=w.shape
    groups=(in_dim+group_size-1)//group_size
    packed=np.zeros((out_dim,(in_dim+1)//2),dtype=np.uint8)
    scales=np.ones((out_dim,groups),dtype=np.float64)
    for row in range(out_dim):
        for group in range(groups):
            start=group*group_size;end=min(in_dim,start+group_size)
            maximum=np.max(np.abs(w[row,start:end]))
            scale=maximum/7 if maximum else 1.0
            scales[row,group]=scale
            q=np.clip(np.rint(w[row,start:end]/scale),-7,7).astype(np.int16)
            for offset,value in enumerate(q):
                col=start+offset
                packed[row,col//2] |= np.uint8((int(value)+8) << (4*(col%2)))
    restored=np.empty_like(w,dtype=np.float64)
    for col in range(in_dim):
        code=(packed[:,col//2].astype(np.int16) >> (4*(col%2))) & 15
        restored[:,col]=(code-8)*scales[:,col//group_size]
    y=x@restored.T+(alpha/a.shape[0])*(x@a.T)@b.T
    return packed,scales,y`,solutionAlt:[{label:'PyTorch 版（仅展示）',code:py`import torch
def packed_lora(x,w,a,b,alpha,group_size):
    o,i=w.shape;g=(i+group_size-1)//group_size
    packed=torch.zeros((o,(i+1)//2),dtype=torch.uint8,device=w.device)
    scales=torch.ones((o,g),dtype=w.dtype,device=w.device)
    for row in range(o):
        for group in range(g):
            start=group*group_size;end=min(i,start+group_size)
            maximum=w[row,start:end].abs().max()
            scale=float(maximum)/7 if maximum else 1.
            scales[row,group]=scale
            q=(w[row,start:end]/scale).round().clamp(-7,7).to(torch.int16)
            for offset,value in enumerate(q):
                col=start+offset
                packed[row,col//2] |= (int(value)+8) << (4*(col%2))
    restored=torch.empty_like(w)
    for col in range(i):
        code=(packed[:,col//2].to(torch.int16)>>(4*(col%2)))&15
        restored[:,col]=(code-8)*scales[:,col//group_size]
    return packed,scales,x@restored.T+(alpha/a.shape[0])*(x@a.T)@b.T`}],explanation:md`整数编码必须与有符号值分开：q 是 −7…7，nibble code 是 q+8。打包初值0，逐列按位或写入，奇数宽度高位自然保留0。解码先转 int16，右移并 &15，再减8，避免 uint8 减法下溢。LoRA 保留独立高精度支路，不能直接把它混入已经量化的基座又继续加一次。此教学实现先反量化全矩阵；实际内核可在 tile 内融合反量化，减少工作区。`,hints:[md`每列的字节位置为 col//2，位偏移为 4×(col%2)。`,md`码值读取一定要 &15，再在有符号或浮点类型中减8。`,md`用 basis 输入 x=I 可直接观察恢复后的权重。`],setup:py`import numpy as np
def _ref_restore(w,group):
    out=w.copy()
    for row in range(w.shape[0]):
        for start in range(0,w.shape[1],group):
            values=w[row,start:start+group];scale=max(abs(values))/7 if np.any(values) else 1.
            out[row,start:start+group]=np.rint(values/scale).clip(-7,7)*scale
    return out`,tests:[{name:'明确的字节布局',code:py`x=np.eye(3);w=np.array([[-7.,0.,7.]]);a=np.zeros((1,3));b=np.zeros((1,1))
p,s,y=packed_lora(x,w,a,b,1,3)
assert p.dtype==np.uint8 and p.shape==(1,2) and p.tolist()==[[129,15]], f"期望字节 [129,15]，得到 {p}"
assert np.allclose(s,1) and np.allclose(y,w.T), f"应精确恢复，得到 {s},{y}"`},{name:'零组与 LoRA 支路',code:py`w=np.zeros((2,3));x=np.ones((1,3));a=np.ones((1,3));b=np.array([[1.],[2.]])
p,s,y=packed_lora(x,w,a,b,2,2)
assert p.tolist()==[[136,8],[136,8]] and np.allclose(s,1), f"零编码错误 {p},{s}"
assert np.allclose(y,[[6,12]]), f"LoRA输出应[6,12]，得到{y}"`},{name:'尾组与不得使用原始基座',hidden:true,code:py`w=np.array([[.3,.8,7.,-1.2,2.]]);x=np.eye(5);a=np.zeros((2,5));b=np.zeros((1,2))
p,s,y=packed_lora(x,w,a,b,0,3);exp=_ref_restore(w,3).T
assert np.allclose(y,exp) and not np.allclose(y,w.T), f"必须使用量化恢复值，期望{exp}，得到{y}"
assert (p[0,-1]>>4)==0, "奇数尾部高nibble应为0"`},{name:'随机矩形与不变性',hidden:true,code:py`rng=np.random.default_rng(48)
for i in [1,4,7]:
    x=rng.normal(size=(2,i));w=rng.normal(size=(3,i));a=rng.normal(size=(2,i));b=rng.normal(size=(3,2));old=w.copy()
    p,s,y=packed_lora(x,w,a,b,3,3)
    exp=x@_ref_restore(w,3).T+1.5*(x@a.T)@b.T
    assert y.shape==(2,3) and np.allclose(y,exp), f"I={i}，最大误差{np.max(abs(y-exp))}"
    assert np.array_equal(w,old), "修改了权重"`}],followUps:[{id:'speed',q:md`既然最终又反量化成浮点，为什么真实低比特推理还能快？`,a:md`真实内核可以在加载压缩权重到寄存器或片上 tile 时反量化，降低显存带宽与存储需求，不必先展开全矩阵到 HBM。是否加速取决于 batch、算术强度、量化格式和目标硬件，教学 NumPy 实现用于验证语义，不代表低比特性能。`}]}],interview:[],cheatsheet:md`码值 q 与编码 q+8 分开。偶数低 nibble、奇数高 nibble；右移后 &15；先转有符号再减零点。组尺度、原始形状、编码布局都属于模型资产。量化基座+高精度 LoRA ≠ 合并后再次量化。`})
