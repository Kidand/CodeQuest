import { defineLevel, md, py } from '@/content/helpers'
import type { Challenge } from '@/content/types'
import { FrequencyViz } from '../viz/ArchViz'
import { single,judge,numeric,multi,order,fill,bug,practice } from '../shared'
const c:Challenge={id:'freq',title:'PI、NTK 与按频带缩放',difficulty:'medium',description:md`实现 \`scaled_freq(dim: int, base: float, scale: float, length: int, method: str, low: float=1., high: float=32.) -> tuple[np.ndarray,float]\`。dim 为大于 2 的偶数，base>1、scale≥1、length>0、0<low<high。原频率 w_i=base^(-2i/dim)。PI 返回 w/scale；ntk 返回 (base×scale^(dim/(dim−2)))^(-2i/dim)；yarn 按论文中的周期数 ramp 形式：r=length×w/(2π)，g=clip((r−low)/(high−low),0,1)，返回 w×(g+(1−g)/scale)。第二项为 Q、K 各自幅度倍率，yarn 是 1+.1log(scale)，其它为 1。本题明确用周期数线性 ramp，工程库也可能采用维度索引 ramp，数值不要求与后者逐项一致。`,starter:py`import numpy as np
def scaled_freq(dim, base, scale, length, method, low=1., high=32.):
    raise NotImplementedError`,solution:py`import numpy as np
def scaled_freq(dim, base, scale, length, method, low=1., high=32.):
    exponent = -np.arange(0, dim, 2, dtype=float) / dim
    w = base ** exponent
    if method == "pi": return w / scale, 1.0
    if method == "ntk":
        new_base = base * scale ** (dim / (dim-2))
        return new_base ** exponent, 1.0
    cycles = length * w / (2*np.pi)
    ramp = np.clip((cycles-low)/(high-low), 0, 1)
    return w*(ramp+(1-ramp)/scale), 1+0.1*np.log(scale)`,solutionAlt:[{label:'PyTorch 版（仅展示）',code:py`import torch
import math
def scaled_freq(dim,base,scale,length,method,low=1.,high=32.):
    e=-torch.arange(0,dim,2,dtype=torch.float64)/dim
    w=base**e
    if method=="pi":return w/scale,1.
    if method=="ntk":return (base*scale**(dim/(dim-2)))**e,1.
    g=((length*w/(2*math.pi)-low)/(high-low)).clamp(0,1)
    return w*(g+(1-g)/scale),1+.1*math.log(scale)`}],explanation:md`PI 缩放所有角频率。NTK 的指数 dim/(dim−2) 保证最低频项正好缩小 scale 倍，而最高频 w0=1 不变。YaRN 的 ramp 从低周期频带的插值平滑过渡到高周期频带的不缩放；Q、K 同时乘幅度 a，点积会乘 a²，不能只乘一次。计算和空间均为 O(dim)。`,hints:[md`先生成统一的原始角频率向量。`,md`周期数=训练长度/周期，周期=2π/w。`,md`返回的幅度作用于 Q、K 两者，不是直接的 logits 倍率。`],tests:[{name:'PI',code:py`w,a=scaled_freq(4,10000,4,2048,'pi')
assert w.shape==(2,) and np.allclose(w,[.25,.0025]) and a==1, f"得到 {w},{a}"`},{name:'NTK 两端',code:py`w,a=scaled_freq(8,10000,8,2048,'ntk');old=10000.**(-np.arange(0,8,2)/8)
assert np.isclose(w[0],1) and np.isclose(w[-1],old[-1]/8), f"两端频率错误 {w}"`},{name:'scale=1 恒等',hidden:true,code:py`for method in ['pi','ntk','yarn']:
    w,a=scaled_freq(16,10000,1,4096,method)
    assert np.allclose(w,10000.**(-np.arange(0,16,2)/16)) and a==1, f"scale=1 应恒等，得到 {w},{a}"`},{name:'ramp 中间区间',hidden:true,code:py`w,a=scaled_freq(4,100,4,20,'yarn',1,5)
old=np.array([1.,.1]);r=20*old/(2*np.pi);g=np.clip((r-1)/4,0,1)
exp=old*(g+(1-g)/4)
assert np.allclose(w,exp) and np.isclose(a,1+.1*np.log(4)), f"期望 {exp}，得到 {w}"`}],followUps:[{id:'dynamic',q:md`动态更改 base 后，旧 KV cache 能原样使用吗？`,a:md`如果 K 缓存了旧频率下的旋转值，新 Q 使用新频率，两者位置基底不一致。需要保持本次序列统一配置、重算旧 K 或采用设计上保证一致的策略；不能只更新新 token 的 cos/sin 表。`}]}
export default defineLevel({learn:[{id:'pi',title:'压缩位置避免直接跑到训练区间外',body:md`直接把位置 m 从训练长度 L 扩展到 sL，会使某些 RoPE 频带进入未见过的角度组合。Position Interpolation 用 m/s，等价于将所有角频率除以 s，把新位置范围压回原区间。代价是相邻位置也更靠近，局部位置信号分辨率变低。静态 NTK-aware 改 base 为 $\theta'=\theta s^{d/(d-2)}$，使高频首项保持、最低频缩小 s 倍，折中局部区分和长距离覆盖。这里 d 是旋转维度，必须大于 2。`,viz:FrequencyViz},{id:'yarn',title:'按频带选择插值，并补偿注意力尺度',body:md`YaRN 根据训练窗口内完成的周期数区分频带：周期很多的高频维保持原频率，周期不足的低频维做插值，中间平滑过渡。它还调整注意力的温度；原论文对 LLaMA 系列给出 Q、K 各乘 $a=1+0.1\ln s$ 的经验形式，故 logits 乘 a²。这个系数是经验设置，不能说成对全部架构的定理。实战明确实现论文周期数 ramp，实际库的边界取整和索引 ramp 可能不同，复现时需对照具体版本。参考：[PI](https://arxiv.org/abs/2306.15595)、[YaRN](https://arxiv.org/abs/2309.00071)。`},{id:'abf',title:'更大 base 也需要数据和评测支撑',body:md`ABF（Adjusted Base Frequency）通过增大 RoPE base 改变低频分布，通常与长序列继续训练结合；它不是「只改配置就保证有效上下文翻倍」的捷径。位置机制之外，长文本数据质量、短长样本混合、训练长度调度与真实检索任务都影响结果。困惑度下降只说明局部预测改善，不能单独证明跨全文的信息检索、冲突证据处理或多跳推理可靠。还应测不同位置的目标、上下文中部表现、最大 KV 预算与延迟。参考：[Effective Long-Context Scaling](https://arxiv.org/abs/2309.16039)。`}],quiz:[single('pi',1,md`PI 的 m→m/s 等价于？`,['频率乘 s','频率除 s','只改最高频','只改 V'],1,md`角度 m×w/s 等于 (m/s)×w。`),judge('zero',1,md`仅增大最大位置配置即可保证长文本任务效果。`,false,md`实现能接受更长输入与模型能利用更长上下文是两个条件。`),numeric('scale',2,md`训练 4096 token，PI 扩至 32768 token，频率应除以几？`,8,md`32768/4096=8。`),multi('yarn',2,md`YaRN 的关键因素包含？`,['按频带插值','注意力尺度补偿','必定删除所有高频','扩展倍数'],[0,1,3],md`高频通常保持以保留局部距离，不是删除。`),order('frequency',md`按本题 YaRN 形式排序：`,['计算原角频率','计算训练长度内周期数','裁剪得到 ramp','混合原频率与插值频率'],md`ramp 由周期数决定，最后是两种频率的凸组合。`),fill('square',md`Q、K 各乘 a，QK 点积乘 ____(1)____。`,['a^2','a**2','a*a','a²'],md`双线性内积两侧各贡献一个 a。`),single('ntk',2,md`静态 NTK-aware 中 i=0 的角频率为？`,['1','1/s','s','base'],0,md`任何正 base 的 0 次幂都是 1，高频首项保持。`),judge('cache',3,md`动态更换 RoPE 频率后，旧的已旋转 K 与新 Q 一定一致。`,false,md`旧 K 与新 Q 不同基底，需重算或统一频率策略。`),bug('cycles',py`import numpy as np
def yarn_freq(w, length, scale, low, high):
    wavelength = 2*np.pi/w
    cycles = wavelength/length
    ramp = np.clip((cycles-low)/(high-low),0,1)
    original = w*ramp
    interpolated = w/scale*(1-ramp)
    return original+interpolated`,[4],md`周期数应 length/wavelength。倒置会反转高低频的插值策略。`) ],guided:[practice(c,[{find:'base * scale ** (dim / (dim-2))',instruction:md`写出静态 NTK 改 base 公式。`,explain:md`保证最低频按目标倍数缩小。`},{find:'length * w / (2*np.pi)',instruction:md`计算训练窗口内周期数。`,explain:md`周期为 2π/w，因此周期数=Lw/(2π)。`},{find:'np.clip((cycles-low)/(high-low), 0, 1)',instruction:md`实现周期数线性 ramp。`,explain:md`区间外分别固定为 0 与 1。`}])],challenges:[c],interview:[{id:'abf',freq:3,q:md`PI 与增大 base 的主要差别是什么？`,a:md`PI 统一缩放所有频率，包括用于局部区分的高频；增大 base 时首个频率不变，后续频率按维度变化，因此频带作用不同。两者都改变位置表征分布，需要用训练或验证判断副作用。回答时先给频率公式，再讨论长距离覆盖与短距离分辨率的取舍。`},{id:'eval',freq:3,q:md`如何证明模型真正支持长上下文？`,a:md`需要在多个长度与目标位置上测检索、理解和跨段组合任务，并与短上下文基线比较。单一 passkey 或低困惑度容易只覆盖某一种能力，还应考察干扰文本、多个目标和冲突证据。部署还要报告上下文预算下的 TTFT、TPOT 和显存，区分能运行与有质量地使用。`},{id:'precision',freq:2,q:md`很大位置编号为什么容易数值出错？`,a:md`位置乘角频率的误差会进入 sin/cos 相位，低精度表示大整数或相近角度时可能丢失差异。通常位置和频率计算使用较高精度，再把旋转结果转换回激活 dtype。缓存 cos/sin 表也要绑定 base、缩放参数与位置偏移，避免复用不兼容表。`}],cheatsheet:md`PI：w/s；静态 NTK：base×s^(d/(d−2))；YaRN：低频插值、高频保持、中间 ramp，加注意力幅度校正。base 改变 ≠ 长上下文能力保证。`})
