import { useState } from 'react'
import { defineLevel, md, py } from '@/content/helpers'
import type { Challenge } from '@/content/types'
import { Slider } from '@/components/ui'
import { MatrixView, VizFrame } from '@/viz'
import { single, judge, numeric, multi, order, fill, bug, practice } from '../shared'

const FP4 = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
function nearestEven(value: number) {
  return FP4.reduce((best, candidate, i) => {
    const delta = Math.abs(candidate - value), old = Math.abs(FP4[best] - value)
    return delta < old || (delta === old && i % 2 === 0) ? i : best
  }, 0)
}
function LowPrecisionViz() {
  const [maximum, setMaximum] = useState(6)
  const scale = 2 ** Math.ceil(Math.log2(maximum / 6))
  const values = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, maximum]
  const quantized = values.map((x) => FP4[nearestEven(x / scale)] * scale)
  return <VizFrame title="尺度跳变会改变整块网格"
    caption={`教学保守尺度 s=${scale}。最大值从6越过6时，s从1跳到2，所有值都重新量化；并列距离选编码最低位为0的值。图仅显示一个32元素块中的8个非零值，其余填0。`}
    controls={<Slider label="块中最大幅度" value={maximum} min={6} max={12} step={0.25} onChange={setMaximum} />}>
    <div className="min-w-0"><MatrixView data={[values, quantized, values.map((x, i) => quantized[i] - x)]} rowLabels={['原值', '恢复值', '误差']} scale="div" /></div>
    <p className="mt-4 text-sm leading-6 text-muted">E2M1 非负码本：0、0.5、1、1.5、2、3、4、6。s=1 时，0.75→1，1.25→1，3.5→4；不是统一向下取整。</p>
  </VizFrame>
}

export const mxfp4Python = py`
import numpy as np

def mxfp4_quantize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    levels = np.array([0., .5, 1., 1.5, 2., 3., 4., 6.])
    n = x.size
    codes = np.empty(n, dtype=np.uint8)
    scale_codes = np.empty((n + 31) // 32, dtype=np.uint8)
    restored = np.empty(n, dtype=np.float64)
    for group, start in enumerate(range(0, n, 32)):
        block = x[start:start + 32]
        maximum = float(np.max(np.abs(block)))
        exponent = 0
        if maximum > 0:
            # 先求最大值二进制指数，再比较，避免 log2 在幂边界舍入失真
            exponent = int(np.frexp(maximum)[1]) - 3
            if maximum > 6 * np.ldexp(1.0, exponent):
                exponent += 1
        scale = np.ldexp(1.0, exponent)
        distance = np.abs(np.abs(block[:, None]) / scale - levels)
        nearest = np.argmin(distance, axis=1)
        tied_even = (distance == distance.min(axis=1, keepdims=True)) & (np.arange(8) % 2 == 0)
        index = np.where(tied_even.any(axis=1), tied_even.argmax(axis=1), nearest)
        sign = np.signbit(block).astype(np.uint8)
        codes[start:start + len(block)] = (index | (sign << 3)).astype(np.uint8)
        scale_codes[group] = exponent + 127
        restored[start:start + len(block)] = np.copysign(levels[index] * scale, block)
    return codes, scale_codes, restored
`
const mxfp4Torch = py`
import torch

def mxfp4_quantize(x):
    # 数值模拟，非低位Tensor Core kernel；输入使用float64方便对拍
    levels = x.new_tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    codes = torch.empty(x.numel(), dtype=torch.uint8, device=x.device)
    scale_codes = torch.empty((x.numel()+31)//32, dtype=torch.uint8, device=x.device)
    restored = torch.empty_like(x)
    for group, start in enumerate(range(0, x.numel(), 32)):
        block = x[start:start+32]
        maximum = block.abs().max()
        exponent = 0
        if maximum > 0:
            exponent = int(torch.frexp(maximum)[1]) - 3
            if maximum > 6 * 2.0**exponent:
                exponent += 1
        scale = 2.0**exponent
        distance = (block.abs()[:, None]/scale - levels).abs()
        nearest = distance.argmin(dim=1)
        even = (distance == distance.min(dim=1, keepdim=True).values) & (torch.arange(8, device=x.device)%2 == 0)
        index = torch.where(even.any(dim=1), even.long().argmax(dim=1), nearest)
        sign = torch.signbit(block).to(torch.uint8)
        codes[start:start+len(block)] = (index | (sign << 3)).to(torch.uint8)
        scale_codes[group] = exponent + 127
        restored[start:start+len(block)] = torch.copysign(levels[index]*scale, block)
    return codes, scale_codes, restored
`

const challenge: Challenge = {
  id: 'mxfp4',
  title: 'MXFP4：32 元素分块、E8M0 尺度与偶数舍入',
  difficulty: 'hard',
  timeLimitMin: 35,
  description: md`
    实现 \`mxfp4_quantize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]\`。
    x 为一维 float64，长度 $0\le N\le4096$；非零幅度在 $[2^{-100},2^{100}]$，不含 Inf/NaN。不修改输入。

    每连续 **32 个元素** 为一组，末组不足32时视作以正零补齐，但输出不含 padding。
    采用本题明确规定的**保守尺度策略**：全零组 s=1；其余选最小的二次幂 $s=2^e$，使 $6s\ge\max\lvert x\rvert$。
    这是一种可用策略，**不是 OCP 规定的唯一尺度算法**。

    E2M1 非负码本按3位幅度编码0…7排列为 [0,0.5,1,1.5,2,3,4,6]。
    对每个 $\lvert x\rvert/s$ 选最近值；距离并列时取**幅度编码最低位为0**的值（roundTiesToEven）。
    第3位是符号位，保留输入符号，包括负零与负数下溢后的负零。

    返回：
    - \`codes\`：(N,) uint8，每元素存一个0…15的E2M1编码，**尚未进行两个码合并为一字节的打包**；
    - \`scale_codes\`：(ceil(N/32),) uint8，存 E8M0 编码 e+127；
    - \`restored\`：(N,) float64，按符号、幅度码本和scale恢复。

    例如 x=[0.75,1.25,−3.5,6]：s=1，码值=[2,2,14,7]，scale码=[127]，恢复=[1,1,−4,6]。
    要求 O(N) 时间；除输出外 O(32) 工作空间。无需实现硬件指令或训练反向。
  `,
  starter: py`
    import numpy as np
    def mxfp4_quantize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # TODO: 分块、选择二次幂尺度、ties-to-even编码、反量化
        raise NotImplementedError
  `,
  solution: mxfp4Python,
  solutionAlt: [{ label: 'PyTorch 版（仅展示）', code: mxfp4Torch }],
  explanation: md`
    幅度编码的最低位就是1位尾数位，因此并列时选择偶数编码；这不同于对原始实数做 \`np.rint\`，也不同于总取较小幅度。
    符号位独立编码，\`signbit\` 可以正确保留 −0；\`x<0\` 不行。

    设 \`frexp(amax)\` 返回 $a\cdot2^p$，$a\in[0.5,1)$。从 e=p−3 开始，若 $6\cdot2^e<amax$ 就增加1。
    这样直接比较可覆盖恰好跨越6·2ᵉ的边界，不依赖浮点 \`log2\` 在大指数处辨别极微小差别。
    全零组另选e=0，不对0取log。

    E2M1相邻幅度的最大间隔是2；本题尺度保证不发生范围外饱和，故绝对恢复误差至多s。
    小值可能舍入为零，相对误差却能达到100%，所以不能从绝对误差界推出统一的小相对误差。
    每组与固定8项码本比较，总时间O(N)，工作空间32×8，除输出外不随N增长。
  `,
  hints: [
    md`先固定32元素的分组边界。最后一组补零不改变amax，但不能与上一组共用尺度。`,
    md`由frexp获得二进制指数，再比较6·2ᵉ是否覆盖amax；全零组单独处理。`,
    md`最近码距离相同时，看幅度编码的奇偶，不看数值本身是不是整数。`,
    md`E8M0存e+127；真实scale用2ᵉ恢复。码0不是scale=0。`,
  ],
  setup: py`
    import numpy as np

    def _assert_result(got, x):
        assert isinstance(got, tuple) and len(got) == 3, "期望(codes, scale_codes, restored)三元组"
        codes, scales, restored = got
        assert codes.shape == x.shape, f"码形状期望{x.shape}，得到{codes.shape}"
        assert scales.shape == ((x.size+31)//32,), f"尺度形状期望{((x.size+31)//32,)}，得到{scales.shape}"
        assert restored.shape == x.shape, f"恢复形状期望{x.shape}，得到{restored.shape}"
        assert codes.dtype == np.uint8 and scales.dtype == np.uint8, f"编码应为uint8，得到{codes.dtype}/{scales.dtype}"
        assert restored.dtype == np.float64, f"恢复应为float64，得到{restored.dtype}"
        assert np.all(codes < 16), f"E2M1应为4位码，得到{codes}"
        assert np.all(scales < 255), f"有限输入不应返回NaN尺度码，得到{scales}"
        return codes, scales, restored

    def _reference(x):
        # 独立标量参考：用倍增/减半找最小覆盖尺度，不使用frexp
        levels = [0., .5, 1., 1.5, 2., 3., 4., 6.]
        codes, scales, restored = [], [], []
        for start in range(0, len(x), 32):
            values = x[start:start+32]
            amax = max(abs(float(v)) for v in values)
            scale, exponent = 1., 0
            if amax:
                while 6*scale < amax:
                    scale *= 2; exponent += 1
                while 3*scale >= amax:
                    scale /= 2; exponent -= 1
            scales.append(exponent+127)
            for v in values:
                idx = min(range(8), key=lambda i: (abs(abs(v)/scale-levels[i]), i%2))
                sign = int(np.signbit(v))
                codes.append(idx+8*sign)
                restored.append(np.copysign(levels[idx]*scale, v))
        return np.array(codes,dtype=np.uint8), np.array(scales,dtype=np.uint8), np.array(restored,dtype=np.float64)
  `,
  tests: [
    {
      name: '示例：码本与符号位',
      code: py`
        x=np.array([.75,1.25,-3.5,6.])
        q,s,y=_assert_result(mxfp4_quantize(x),x)
        assert q.tolist()==[2,2,14,7], f"期望码[2,2,14,7]，得到{q}"
        assert s.tolist()==[127], f"期望s编码127，得到{s}"
        assert np.array_equal(y,[1.,1.,-4.,6.]), f"期望恢复[1,1,-4,6]，得到{y}"
      `,
    },
    {
      name: '32 元素边界与尾组独立尺度',
      code: py`
        x=np.r_[np.full(32,6.),12.,-12.]
        q,s,y=_assert_result(mxfp4_quantize(x),x)
        assert s.tolist()==[127,128], f"前32项scale=1、尾组scale=2，得到{s}"
        assert q[-2:].tolist()==[7,15], f"尾组期望码[7,15]，得到{q[-2:]}"
        assert np.array_equal(y,x), f"精确网格应原样恢复，最大误差{np.max(np.abs(y-x))}"
      `,
    },
    {
      name: '全部中点的 ties-to-even 与符号', hidden: true,
      code: py`
        positive=np.array([.25,.75,1.25,1.75,2.5,3.5,5.,6.])
        x=np.r_[positive,-positive]
        q,s,y=_assert_result(mxfp4_quantize(x),x)
        expected=np.array([0,2,2,4,4,6,6,7],dtype=np.uint8)
        assert np.array_equal(q[:8],expected), f"中点期望{expected}，得到{q[:8]}"
        assert np.array_equal(q[8:],expected+8), f"负数符号位错误：{q[8:]}"
        assert np.signbit(y[8]), "负数舍入到0后应保留负零"
      `,
    },
    {
      name: '空输入、全零与负零', hidden: true,
      code: py`
        for n in [0,1,32,33]:
            x=np.zeros(n,dtype=np.float64)
            if n:x[-1]=-0.0
            q,s,y=_assert_result(mxfp4_quantize(x),x)
            assert np.all(s==127), f"零组scale=1编码127，得到{s}"
            assert np.all(y==0), f"零组恢复应全零，得到{y}"
            if n:assert q[-1]==8 and np.signbit(y[-1]), f"负零应为码8，得到{q[-1]}"
      `,
    },
    {
      name: '二次幂尺度的精确边界与极小/极大幅度', hidden: true,
      code: py`
        for exponent in [-90,0,90]:
            edge=np.ldexp(6.,exponent)
            for maximum,expected_e in [(np.nextafter(edge,0.),exponent),(edge,exponent),(np.nextafter(edge,np.inf),exponent+1)]:
                x=np.array([maximum,-maximum,maximum/8])
                q,s,y=_assert_result(mxfp4_quantize(x),x)
                assert int(s[0])==expected_e+127, f"amax={maximum}期望尺度指数{expected_e}，得到{int(s[0])-127}"
                bound=np.ldexp(1.,expected_e)
                assert np.max(np.abs(y-x))<=bound, f"误差超过{bound}，得到{np.max(np.abs(y-x))}"
      `,
    },
    {
      name: '固定种子对拍、解码一致性与误差界', hidden: true,
      code: py`
        rng=np.random.default_rng(204)
        levels=np.array([0.,.5,1.,1.5,2.,3.,4.,6.])
        for n in [17,31,32,65,257]:
            x=rng.normal(size=n)*rng.choice([.125,1.,16.],size=n);original=x.copy()
            q,s,y=_assert_result(mxfp4_quantize(x),x);eq,es,ey=_reference(x)
            assert np.array_equal(q,eq) and np.array_equal(s,es), f"编码对拍失败，期望{eq}/{es}，得到{q}/{s}"
            assert np.array_equal(y,ey), f"反量化与参考不同，最大误差{np.max(np.abs(y-ey))}"
            scale=np.repeat(np.exp2(s.astype(int)-127),32)[:n]
            decoded=np.where(q&8,-levels[q&7],levels[q&7])*scale
            assert np.array_equal(y,decoded), "restored与输出码值/尺度无法互相解码"
            assert np.all(np.abs(y-x)<=scale+1e-12), f"绝对误差超过尺度，最大{np.max(np.abs(y-x))}"
            assert np.array_equal(x,original), "函数不应修改输入"
      `,
    },
  ],
  followUps: [
    {
      id: 'scale-choice', q: md`本题保守尺度一定最小化块内 MSE 吗？`,
      a: md`不一定。它保证最大幅度落在可表示范围，避免范围外饱和；但scale翻倍会让全组网格更粗。若只有一个离群值，容忍它被裁剪可能改善其余31个值的总误差。OCP没有规定唯一scale优化算法，比较实现时应同时固定scale选择、舍入、饱和和块边界，而不能只说都叫MXFP4。`,
    },
    {
      id: 'transpose', q: md`按行量化后直接转置码值，等价于对浮点转置矩阵按行重新量化吗？`,
      a: md`一般不等价。转置会把原本属于不同块、不同scale的元素放入新的同一块。只转置码值没有产生新方向需要的共享尺度。训练中的前向与反向若要求不同归约方向，应从高精度输入分别生成对应布局，或按硬件明确支持的布局使用现有scale。`,
    },
  ],
}

export default defineLevel({
  learn: [
    {
      id: 'bits', title: '指数保范围，尾数保局部精度',
      body: md`
        正规二进制浮点数可写为 $(-1)^s2^{E-b}(1+M2^{-m})$。指数决定数值覆盖多少数量级；尾数决定同一数量级内的格点密度。
        因而「位数相同」不代表同一行为：BF16与FP16都用16位，却分配给范围与精度不同的预算。

        | 格式 | 符号 / 指数 / 尾数位 | 1附近的步长 | 最大有限值（近似） |
        |---|---|---|---|
        | FP16 | 1 / 5 / 10 | $2^{-10}$ | 65504 |
        | BF16 | 1 / 8 / 7 | $2^{-7}$ | $3.39\times10^{38}$ |
        | FP8 E4M3（OCP） | 1 / 4 / 3 | $2^{-3}$ | 448 |
        | FP8 E5M2（OCP） | 1 / 5 / 2 | $2^{-2}$ | 57344 |

        BF16范围接近FP32，因此通常较少依赖loss scaling，但同数量级内比FP16更粗。尾数只计显式位，正规数还有隐含的首位1；次正规数没有该隐含1。
        FP8必须声明具体编码变体；本关使用OCP/TE对应的E4M3、E5M2，不能把FNUZ等变体的范围直接套入。
      `,
      checkpoint: single('cp-bits', 1, md`BF16相较FP16的主要取舍是？`, ['指数范围更宽，但尾数更短', '所有数值都更精确', '参数存储字节数减半', '一定无需更高精度累加'], 0, md`BF16以7位尾数换取8位指数；两者都是16位，更高精度归约仍有价值。`),
    },
    {
      id: 'fp8', title: 'FP8 训练是一套尺度与累加策略',
      body: md`
        [Transformer Engine FP8 文档](https://docs.nvidia.com/deeplearning/transformer-engine/examples/fp8_primer.html) 中的E4M3有更细尾数，常用于前向；E5M2范围更宽，常用于梯度。
        这是常见配方，不是硬性定律。量化写成 $q=Q(x/s)$、恢复 $\hat x=sq$；如果库把scale定义成量化乘数，公式中的s就对应它的倒数，面试要先说明约定。

        scale可以由当前amax计算，也可以由历史amax估计。后者节省某些依赖，却可能遇到新出现的离群值；前者需要计算当前统计并及时应用。
        小块尺度限制一个离群值影响的范围，代价是更多元数据、转换和广播。
        低精度输入也不表示乘积总和必须低精度累加：长归约容易积累舍入误差，通常需要更高精度累加与敏感算子的保护，优化器也不应机械降到FP8。
      `,
    },
    {
      id: 'v3', title: '真实案例：DeepSeek-V3 的细粒度 FP8',
      body: md`
        [DeepSeek-V3 报告 §3.3](https://arxiv.org/html/2412.19437v2#S3.SS3) 给出激活按1×128、权重按128×128分组的策略，并结合更高精度累加。
        报告使用E4M3处理其FP8张量，说明细粒度scale可以缓解范围不足，并不一定要固定「前向E4M3、反向E5M2」。

        这里1×128表示每token的一段通道共享scale，128×128表示矩阵二维块；两者不能交换。
        例如把多个token激活合成一个大块求amax，一个token的异常通道便可能拉大其它token的步长。
        分组方向还必须匹配GEMM归约轴：若K维跨多个scale块，部分乘加需要带上各块尺度再合并，不能只在整个矩阵乘结束后乘一个张量尺度。
        这些结构解释了为什么保存为FP8与真正高效、稳定的FP8训练仍有距离。
      `,
    },
    {
      id: 'mxfp4', title: 'MXFP4：低位浮点值加共享指数',
      body: md`
        [OCP MX 1.0 规范](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) 定义MXFP4为32个E2M1元素共享一个8位E8M0尺度。
        E2M1非负幅度是0、0.5、1、1.5、2、3、4、6；正负零有不同符号码，格式没有独立Inf/NaN编码。
        E8M0没有符号和尾数，有限编码c表示 $2^{c-127}$，c=255保留给NaN。特别注意c=0不是零尺度。

        一个完整块的逻辑容量是32×4+8=136位，即平均4.25位/元素，不能忽略尺度元数据。
        规范要求支持roundTiesToEven，但没有规定唯一的详细scale算法。本关采用最小二次幂覆盖amax的保守策略，随后按真正的E2M1码本舍入。
        由于码本非均匀，不能用INT4的线性网格或统一round代替。图中调节amax，观察尺度越过阈值时全块格点的跳变。
      `,
      viz: LowPrecisionViz,
      checkpoint: numeric('cp-scale', 2, md`E8M0编码129代表的尺度是多少？`, 4, md`2^(129−127)=4；编码值129不是实际乘数129。`),
    },
    {
      id: 'nvfp4', title: 'NVFP4 与 QAT：格式之外还有训练配方',
      body: md`
        [NVIDIA TE 的 NVFP4 文档](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html) 用E2M1值、每16元素的E4M3局部尺度与一个FP32张量尺度表示 $x\approx q\,s_{block}\,s_{global}$。
        E4M3尺度可以表达非二次幂，但范围有限，张量尺度帮助覆盖整体范围。与32元素MXFP4相比，常规1D布局元数据更多：平均4.5位/元素，再加32位张量尺度。
        TE训练配方还支持权重16×16的二维共享尺度，不能把1D布局当所有变体的唯一规则。

        QAT在训练中显式模拟量化误差，让参数适应部署网格；常用straight-through估计传递舍入操作的代理梯度，不等于舍入函数真的处处可导。
        随机舍入、离群值平滑、较高精度主权重等是不同的稳定性手段。FP4 QAT既不保证任意模型无损，也不意味着所有算子、梯度和优化器状态都存4位。
        评估需要报告量化轴、尺度策略、保护层、任务质量和目标硬件实测时延。
      `,
    },
  ],
  quiz: [
    single('range', 1, md`FP16与BF16都用16位，为什么BF16更不容易因范围不足溢出？`, ['指数位更多', '尾数位更多', '每元素有额外FP32尺度', 'BF16没有负数'], 0, md`BF16有8位指数、7位尾数，FP16有5位指数、10位尾数；BF16不是靠额外张量尺度得到范围。`),
    single('fp8', 1, md`按本关OCP E4M3与E5M2约定，哪项正确？`, ['E4M3尾数更细，E5M2范围更宽', 'E4M3最大值57344', 'E5M2没有Inf', 'E4M3与BF16动态范围相同'], 0, md`E4M3最大有限值448，E5M2为57344且有Inf编码；格式名必须与编码变体一起理解。`),
    judge('optimizer', 1, md`使用FP8 GEMM意味着Adam一二阶矩和所有归约也必须用FP8。`, false, md`乘法输入、累加精度、优化器状态是不同层面的选择。对敏感状态保留高精度是混合精度的组成部分。`),
    numeric('storage', 2, md`1024个元素采用完整MXFP4块，忽略对齐/额外头部，每32元素共享1字节尺度。合计多少字节？`, 544, md`元素1024×4/8=512字节，尺度1024/32=32字节，总544；不是512。`),
    multi('mxfacts', 2, md`关于MXFP4，哪些正确？`, ['元素码本间距不均匀', '全零块可用元素零码配有限scale', 'E8M0码0表示scale=0', '32元素共享E8M0尺度'], [0,1,3], md`E8M0有限编码0表示2^-127，格式没有零尺度；全零通过元素零码表达。`),
    order('encode', md`按本题定义完成块量化的顺序是？`, ['统计32元素块的amax', '选择覆盖范围的二次幂尺度', '归一化幅度并在E2M1码本中偶数舍入', '加入符号位并保存E8M0指数码', '按码本与尺度恢复数值'], md`应先确定该块尺度，再量化每个元素；必须同时保存码值与尺度才能恢复。`),
    fill('e8m0', md`有限E8M0编码c的尺度是2^(c−____(1)____)。`, ['127'], md`指数偏置为127；255保留为NaN，不能作为有限指数128。`),
    numeric('tie', 2, md`scale=1时，正数3.5等距于E2M1幅度3（码5）与4（码6）。ties-to-even的恢复值是多少？`, 4, md`选择最低编码位为0的码6，对应4；不是总向下，也不是按实数奇偶判断。`),
    multi('nv', 2, md`常规1D NVFP4与MXFP4比较，哪些正确？`, ['NVFP4局部尺度用E4M3', 'NVFP4另有FP32张量尺度', '两者局部块都固定32元素', 'NVFP4元素同样使用E2M1'], [0,1,3], md`常规NVFP4局部块为16元素，MXFP4为32；NVFP4训练还可能用不同的二维权重分组变体。`),
    judge('v3-rule', 2, md`所有FP8训练都必须前向E4M3、反向E5M2，DeepSeek-V3也不能例外。`, false, md`这是一种常见配方。DeepSeek-V3通过更细分组与累加策略使用E4M3处理其FP8张量；选择应结合尺度粒度和数值验证。`),
    bug('power', py`
      import numpy as np
      def choose_scale(values):
          amax = float(np.abs(values).max())
          if amax == 0:
              return 1.0
          exponent = np.floor(np.log2(amax / 6))
          scale = 2.0 ** exponent
          return scale
    `, [6], md`本题要求最小覆盖amax的二次幂，数学上应ceil而非floor。amax=7时floor给scale=1，范围只到6；ceil给2。边界稳健实现可用frexp加直接比较。`),
    numeric('ulp', 3, md`正规数区间[1,2)内，FP16尾数10位、BF16尾数7位。BF16的相邻格点间距是FP16的多少倍？`, 8, md`2^-7 / 2^-10 = 2^3 = 8。动态范围更宽不等于该区间的局部误差更小。`),
  ],
  guided: [practice(challenge, [
    { find: '(n + 31) // 32', instruction: md`计算包括不足32元素尾组在内的scale个数。`, explain: md`向上取整；N=0产生0组，末组不与上一组混合。` },
    { find: 'int(np.frexp(maximum)[1]) - 3', instruction: md`从amax的二进制指数构造一个候选scale指数。`, explain: md`E2M1最大值6介于4与8之间，从p−3开始，再按6·2ᵉ是否覆盖amax决定加1。` },
    { find: '(np.arange(8) % 2 == 0)', instruction: md`在等距候选中找尾数最低位为0的幅度编码。`, explain: md`编码序号0…7对应E2M1的3位幅度字段，最低位奇偶决定ties-to-even。` },
    { find: 'exponent + 127', instruction: md`把真实二进制指数编码为E8M0字节。`, explain: md`E8M0使用127偏置；本题幅度约束保证指数在合法有限范围内。` },
  ])],
  challenges: [challenge],
  interview: [
    {
      id: 'bf16', freq: 3, q: md`BF16比FP16更稳定，能不能说它精度也更高？`,
      a: md`不能笼统比较。BF16的指数位更多，能覆盖更大的数量级，溢出/下溢风险常较低；FP16尾数更多，在相同正规数数量级内有更密的格点，例如[1,2)的步长小8倍。训练稳定性还受归约、主权重、scale与敏感算子的实现影响。面试中应分别回答动态范围与局部相对精度，再结合激活和梯度分布讨论选择，而不是用一个“精度高低”概括所有行为。`,
    },
    {
      id: 'not-int4', freq: 3, q: md`FP4不也是16种编码吗，为什么不能用INT4量化代码直接替代？`,
      a: md`编码数相同但数值网格不同。对称INT4通常在固定scale下均匀间隔；E2M1的间隔随幅度变大，有次正规数和正负零。把x/s做整数round再clip会落到错误的值集，也不遵守浮点中点的偶数舍入规则。此外MXFP4/NVFP4的scale类型、块大小与布局不同，必须一起实现。可以用查表模拟数值语义，但实际4位存储与矩阵加速还需要打包和硬件内核，NumPy数组不是这些优化本身。`,
    },
    {
      id: 'qat', freq: 2, q: md`FP4 QAT 为什么还经常保留高精度参数或累加？`,
      a: md`QAT让模型在前向中体验离散网格误差，不意味着每一次参数更新都只能落在4位格点。若更新小于网格步长，直接改4位权重可能完全丢失更新；高精度主权重可以累积这些变化，再重新量化。舍入的真实导数几乎处处为零，训练常用代理梯度而不是把它当作光滑算子。还应验证scale统计、异常值、保护层与任务质量；训练能收敛也不自动保证部署内核获得同样数值行为与吞吐。`,
    },
    {
      id: 'metadata', freq: 2, q: md`为什么“4位模型显存是BF16的四分之一”通常只是粗略下界？`,
      a: md`4位只描述元素字段，还要存scale、可能的张量尺度、对齐padding及保留高精度的层。MXFP4完整块本身就平均4.25位，常规1D NVFP4平均4.5位并另有张量尺度。服务时还包括KVcache、临时workspace和并发请求；训练还包括梯度与优化器状态。比较时要先明确统计的是权重文件、GPU常驻权重还是进程峰值，再按真实布局逐项计数，并在目标硬件验证。`,
    },
  ],
  cheatsheet: md`
    | 格式 | 元素 | 共享scale |
    |---|---|---|
    | FP16 / BF16 | 1/5/10 与1/8/7 | 视训练配方而定 |
    | FP8 | E4M3范围±448；E5M2范围±57344 | 可按tensor或block |
    | MXFP4 | 32个E2M1 | 1个E8M0，完整块平均4.25bit |
    | NVFP4（常规1D） | 16个E2M1 | E4M3块尺度×FP32张量尺度 |

    E2M1幅度码0…7：**0、0.5、1、1.5、2、3、4、6**。
    E8M0：s=2^(code−127)，255=NaN，没有零尺度。
    本题保守scale：最小2ᵉ覆盖amax/6；全零s=1；中点选偶数幅度码。
    DeepSeek-V3：激活1×128、权重128×128分组；精度、分组轴与累加须一起解释。
    QAT≠所有状态4位；逻辑编码≠打包存储≠硬件加速。
  `,
})
