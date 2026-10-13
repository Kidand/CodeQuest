import { defineLevel, md, py } from '@/content/helpers'
import type { Challenge } from '@/content/types'
import { QuantizationViz } from '../viz/QuantViz'
import { single,judge,numeric,multi,order,fill,bug,practice } from '../shared'
export const quantPython=py`import numpy as np
def quantize_groups(x: np.ndarray, bits: int, group_size: int, symmetric: bool=True):
    rows,cols=x.shape;groups=(cols+group_size-1)//group_size
    q=np.empty(x.shape,dtype=np.int16)
    scales=np.ones((rows,groups),dtype=float)
    zeros=np.zeros((rows,groups),dtype=np.int16)
    restored=np.empty(x.shape,dtype=float)
    for row in range(rows):
        for group in range(groups):
            start=group*group_size;end=min(cols,start+group_size)
            values=x[row,start:end]
            if symmetric:
                hi=2**(bits-1)-1;lo=-hi
                maximum=np.max(np.abs(values))
                scale=maximum/hi if maximum else 1.0
                zero=0
            else:
                lo,hi=0,2**bits-1
                minimum=min(0.,float(values.min()));maximum=max(0.,float(values.max()))
                scale=(maximum-minimum)/hi if maximum>minimum else 1.0
                zero=int(np.clip(np.rint(-minimum/scale),lo,hi))
            values_q=np.clip(np.rint(values/scale)+zero,lo,hi).astype(np.int16)
            q[row,start:end]=values_q
            scales[row,group]=scale;zeros[row,group]=zero
            restored[row,start:end]=(values_q.astype(float)-zero)*scale
    return q,scales,zeros,restored`
export const quantTorch=py`import torch
def quantize_groups(x,bits,group_size,symmetric=True):
    rows,cols=x.shape;groups=(cols+group_size-1)//group_size
    q=torch.empty_like(x,dtype=torch.int16);scales=torch.ones((rows,groups),dtype=x.dtype,device=x.device)
    zeros=torch.zeros((rows,groups),dtype=torch.int16,device=x.device);restored=torch.empty_like(x)
    for row in range(rows):
        for group in range(groups):
            start=group*group_size;end=min(cols,start+group_size);v=x[row,start:end]
            if symmetric:
                hi=2**(bits-1)-1;lo=-hi;m=v.abs().max();scale=float(m)/hi if m else 1.;zero=0
            else:
                lo,hi=0,2**bits-1;mn=min(0.,float(v.min()));mx=max(0.,float(v.max()))
                scale=(mx-mn)/hi if mx>mn else 1.;zero=round(-mn/scale)
            vq=(torch.round(v/scale)+zero).clamp(lo,hi)
            q[row,start:end]=vq;scales[row,group]=scale;zeros[row,group]=zero
            restored[row,start:end]=(vq-zero)*scale
    return q,scales,zeros,restored`
const c:Challenge={id:'groups',title:'分组 INT4/INT8 量化与反量化',difficulty:'medium',description:md`实现 \`quantize_groups(x: np.ndarray, bits: int, group_size: int, symmetric: bool=True) -> tuple\`，x=[R,C] 为有限 float64，1≤R,C≤64、bits∈{4,8}、group_size≥1。沿每一行每 group_size 列一组，末组可不足。对称使用窄范围 [−Q,Q]、Q=2^(bits−1)−1，s=maxabs/Q、z=0；非对称使用 [0,2^bits−1]，先把范围扩展包含 0，再 s=(max−min)/Q，z=round(−min/s) 并裁剪。全零组 s=1,z=0。q=clip(round(x/s)+z)，round 使用 ties-to-even 的 np.rint。返回 q（[R,C] int16，尚未打包）、scales 与 zeros（[R,ceil(C/group)]）、反量化 (q−z)s。不可修改输入。`,starter:py`import numpy as np
def quantize_groups(x: np.ndarray, bits: int, group_size: int, symmetric: bool=True):
    raise NotImplementedError`,solution:quantPython,solutionAlt:[{label:'PyTorch 版（仅展示）',code:quantTorch}],explanation:md`每组独立尺度能缩小离群值影响范围，但引入更多元数据。非对称先包含零，使 zero point 可表示真实零；它不保证端点也完全精确，因为 zero point 本身要取整。题目输出 int16 只是便于验证逻辑，不声称内存已压到 4bit，真实存储还要 nibble 打包。时间 O(RC)、输出及辅助空间 O(RC)。`,hints:[md`先按行切组，特别处理最后一组和全零组。`,md`对称窄范围不使用 −8 或 −128；按题目约定计算。`,md`反量化前转换成浮点，避免整数减法溢出。`],tests:[{name:'对称网格',code:py`q,s,z,y=quantize_groups(np.array([[-7.,0.,3.,7.]]),4,4)
assert q.shape==(1,4) and s.shape==(1,1), f"形状错误 {q.shape},{s.shape}"
assert np.array_equal(q,[[-7,0,3,7]]) and np.allclose(y,[[-7,0,3,7]]) and s[0,0]==1, f"期望精确网格，得到 {q},{s},{y}"`},{name:'非对称正范围',code:py`q,s,z,y=quantize_groups(np.array([[0.,5.,10.,15.]]),4,4,False)
assert np.array_equal(q,[[0,5,10,15]]) and z[0,0]==0 and np.allclose(y,[[0,5,10,15]]), f"错误 {q},{s},{z}"`},{name:'尾组 全零 并列舍入',hidden:true,code:py`x=np.array([[.5,1.5,7.,0.,0.]])
q,s,z,y=quantize_groups(x,4,3)
assert q.tolist()==[[0,2,7,0,0]] and s.shape==(1,2) and s[0,1]==1, f"ties-to-even 或零尾组错误 {q},{s}"
assert np.array_equal(x,[[.5,1.5,7,0,0]]), "修改输入"`},{name:'随机误差界与范围',hidden:true,code:py`rng=np.random.default_rng(46)
for bits in [4,8]:
    for sym in [True,False]:
        x=rng.normal(size=(4,17));q,s,z,y=quantize_groups(x,bits,5,sym)
        hi=2**(bits-1)-1 if sym else 2**bits-1;lo=-hi if sym else 0
        assert np.issubdtype(q.dtype,np.integer) and q.min()>=lo and q.max()<=hi, f"整数范围错误 {q}"
        for j in range(4):
            sl=slice(j*5,min(17,(j+1)*5));err=np.abs(y[:,sl]-x[:,sl]);bound=s[:,j,None]*.500001+1e-10
            assert np.all(err<=bound), f"最大误差 {err.max()} 超过步长一半"`}],followUps:[{id:'group',q:md`减小 group_size 为什么不一定加速推理？`,a:md`更多 scale/zero 元数据和更频繁反量化可能增加读写与内核工作，分组形状也要匹配硬件 tile。小组常改善精度，但吞吐应实测；低 bit 只有配合压缩布局和适用内核才转化为速度优势。`}]}
export default defineLevel({learn:[{id:'affine',title:'量化把连续值投到离散网格',body:md`仿射量化写成 $q=\operatorname{clip}(\operatorname{round}(x/s)+z,q_{min},q_{max})$，反量化 $\hat x=s(q-z)$。s 控制步长，z 是真实零在整数格点的位置。对称量化令 z=0，适合围绕零的权重；非对称范围可更有效覆盖偏斜数据，但多一个 zero point 的处理。本关对称 INT4 明确使用 [−7,7] 窄范围，留下一个编码不用；其它实现可能用 [−8,7]，不能混着算误差。`},{id:'granularity',title:'量化粒度决定离群值影响范围',body:md`per-tensor 共用一个尺度，元数据最少；per-channel 每个输出通道一套尺度；per-group 再细分通道内权重，让离群值只影响一小组。只看最大值确定尺度会让一个异常大值扩大整组网格，小值大量舍入为零。更细分组能缓解此问题，却增大尺度存储和内核复杂度。量化轴必须结合 W 的布局说明：对 [out,in] 权重，沿每一行分组通常表示在输入维度切块。`,viz:QuantizationViz},{id:'error',title:'权重误差不是任务误差的全部',body:md`无裁剪时最邻近均匀量化的单元素误差不超过 s/2；裁剪区间外的值会有额外饱和误差。线性层输出误差为 $X(\hat W-W)$，会被输入激活方向放大，因此只优化权重 MSE 不保证输出最接近。校准集应覆盖部署激活分布，评估除困惑度外还要包含任务质量、异常输入和实际吞吐。4-bit 数值写进 int16 数组仍占 16-bit 存储；逻辑量化、打包格式和低精度矩阵内核是三个不同阶段。`}],quiz:[single('zero',1,md`仿射反量化正确公式是？`,['sq+z','s(q−z)','(q+s)z','q/s−z'],1,md`zero point 在整数网格，先减 z 再乘尺度 s。`),judge('int4',1,md`np.int16 数组中所有值都小于16，就已经实现每参数4-bit存储。`,false,md`dtype 仍为16-bit，需要打包两个4-bit编码到1字节才能压缩。`),numeric('scale',2,md`窄范围对称 INT4，maxabs=14，步长 s 为多少？`,2,md`Q=7，s=14/7=2。`),multi('metadata',2,md`减小 group_size 可能造成？`,['更多尺度元数据','较小离群值影响范围','保证任意硬件更快','反量化工作增加'],[0,1,3],md`吞吐取决于内核和布局，不能由位宽或分组大小直接保证。`),order('quantize',md`仿射量化步骤：`,['统计范围并选择 scale/zero','除以 scale 后舍入','加 zero point 并裁剪到整数范围','存储整数值与尺度元数据'],md`反量化还需要元数据，不能只保存整数 q。`),fill('half',md`无饱和的最邻近均匀量化误差界为 s/____(1)____。`,['2'],md`每个网格单元距最近端点不超过半个步长。`),single('axis',2,md`W=[out,in]，每个输出通道一个尺度，scale 应沿哪一维广播？`,['[out,1]','[1,out]','[in,1]','[in,out]'],0,md`每行输出通道共用一个尺度，要在列方向广播。`),judge('mse',3,md`两个量化权重的权重 MSE 相同，则所有输入上的输出误差必相同。`,false,md`输出误差依赖 X 与误差矩阵的方向关系，不仅依赖总体平方和。`),bug('dequant',py`import numpy as np
def dequant(q, scales, zeros):
    q = q.astype(float)
    assert scales.ndim == 2
    assert zeros.shape == scales.shape
    offset = zeros * scales
    out = q * scales + offset
    return out`,[7],md`应减 offset，即 scales*(q−zeros)。当前将零点偏移加反。`) ],guided:[practice(c,[{find:'(cols+group_size-1)//group_size',instruction:md`计算含尾组的组数。`,explain:md`向上取整，不可丢弃最后不足整组的列。`},{find:'maximum/hi if maximum else 1.0',instruction:md`求对称尺度并处理全零组。`,explain:md`全零组使用尺度1避免除零，恢复仍全零。`},{find:'np.clip(np.rint(values/scale)+zero,lo,hi)',instruction:md`量化并裁剪到整数范围。`,explain:md`rint 使用 ties-to-even；zero point 在整数域添加。`},{find:'(values_q.astype(float)-zero)*scale',instruction:md`完成反量化。`,explain:md`先转浮点、减零点再乘尺度。`}])],challenges:[c],interview:[{id:'asym',freq:3,q:md`什么时候非对称量化更合适？`,a:md`当数值范围明显偏向一侧时，非对称格点能更充分利用有限整数码。例如非负激活不必把一半编码浪费在负半轴。但 zero point 带来额外运算，硬件支持也不同。对围绕零的权重，对称格式常更简单，最终需要结合精度与执行成本选择。`},{id:'outlier',freq:3,q:md`一个离群值为什么会破坏整组精度？`,a:md`若尺度由最大绝对值决定，离群值变大时 s 也变大，所有普通小值落入更宽的网格区间，很多值被舍入为零。减小组、裁剪离群值、做等价重参数化或分离高精度通道是不同应对方式，每种都在误差、元数据和计算成本间取舍。`},{id:'validation',freq:2,q:md`量化上线前如何验证？`,a:md`先用精确网格、全零、尾组和舍入边界验证编码反解，再用独立浮点基线比较层输出和完整任务表现。需要统计不同长度、语言和异常输入的误差，检查校准集代表性。最后在目标硬件实测显存与吞吐，避免把理论压缩率当端到端加速比。`}],cheatsheet:md`q=clip(round(x/s)+z)，x̂=s(q−z)。对称 narrow INT4：Q=7。无裁剪误差≤s/2。粒度越细尺度越多；输出误差=XΔW；逻辑码值不等于压缩存储。`})
