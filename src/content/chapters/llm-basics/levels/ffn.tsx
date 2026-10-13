/**
 * llm-basics-ffn · 激活函数与 FFN
 * ReLU → GELU（含 tanh 近似）→ SiLU；GLU 族与 SwiGLU；隐藏维 8/3·d 的由来与对齐；FFN 参数量。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { ActivationViz } from '../viz/ActivationViz'
import { SwiGLUStepViz } from '../viz/SwiGLUStepViz'
import { FFNParamViz } from '../viz/FFNParamViz'

export default defineLevel({
  /* ================================================================ */
  /* 图解                                                              */
  /* ================================================================ */
  learn: [
    {
      id: 'role',
      title: 'FFN：每个 token 各自过一遍的两层 MLP',
      body: md`
        Transformer block 由两部分组成：注意力负责**在 token 之间搬运信息**，FFN 负责**对每个 token 单独做非线性变换**（逐位置，position-wise）：

        $$
        \mathrm{FFN}(x) = \phi(x W_1 + b_1)\, W_2 + b_2,\qquad W_1 \in \mathbb{R}^{d \times d_{ff}},\ W_2 \in \mathbb{R}^{d_{ff} \times d}
        $$

        原始 Transformer 用 ReLU、$d_{ff} = 4d$（base 模型 $d=512, d_{ff}=2048$）。现代 LLM 普遍去掉 bias。

        **为什么它重要**：
        - **参数大户**：标准 MHA 约 $4d^2$（Q、K、V、O 四个投影），FFN 约 $2 \cdot d \cdot 4d = 8d^2$——FFN 占每层参数的约 **2/3**；FLOPs 同理。
        - **知识存储**：一种流行的解读把 $W_1$ 的列看作“键”、$W_2$ 的行看作“值”，FFN 是一个巨大的键值记忆（Geva 等，2021）。
        - **MoE 替换的就是它**：混合专家把一个大 FFN 拆成很多个小 FFN（专家），每个 token 只激活其中几个。DeepSeek-V3 的稠密层 FFN 中间维是 18432（$d = 7168$），而每个路由专家的中间维只有 2048。

        > [!NOTE] 逐位置 = 可以把 (B, T) 合并
        > FFN 对每个 token 独立计算，所以 \`(B, T, d)\` 可以直接 reshape 成 \`(B·T, d)\` 做一次大矩阵乘法——这也是 FFN 的 GPU 利用率通常比注意力高的原因。
      `,
      checkpoint: {
        id: 'cp-role',
        type: 'numeric',
        prompt: md`一个标准 Transformer 层（MHA + ReLU FFN，$d_{ff} = 4d$，忽略 bias 与 norm），$d = 1024$。FFN 的参数量占该层的百分之多少？（保留一位小数）`,
        answer: 66.7,
        tolerance: 0.002,
        unit: '%',
        explanation: md`注意力 $4d^2$，FFN $8d^2$，占比 $8/12 \approx 66.7\%$，与 $d$ 无关。`,
      },
    },
    {
      id: 'act',
      title: '激活函数：ReLU → GELU → SiLU',
      body: md`
        | 名称 | 公式 | 代表模型 |
        |---|---|---|
        | ReLU | $\max(0, x)$ | 原始 Transformer、T5 |
        | GELU | $x\,\Phi(x) = \frac{x}{2}\big(1 + \mathrm{erf}(x/\sqrt2)\big)$ | BERT、GPT-2/3 |
        | GELU（tanh 近似） | $\frac{x}{2}\Big(1 + \tanh\big(\sqrt{2/\pi}\,(x + 0.044715\,x^3)\big)\Big)$ | GPT-2 的原始实现 |
        | SiLU / Swish | $x\,\sigma(x)$（Swish 一般为 $x\,\sigma(\beta x)$） | LLaMA、Qwen、DeepSeek（作为 SwiGLU 的门） |

        - **GELU 的直觉**：把输入乘以“它比标准正态随机变量大的概率” $\Phi(x)$——大的正数几乎原样通过，大的负数几乎被关掉，中间平滑过渡。
        - **共同点**：二者都是光滑的“软 ReLU”，在 0 附近导数连续；负半轴有一小段**负值**（非单调），GELU 最小约 $-0.17$（$x \approx -0.75$），SiLU 最小约 $-0.28$（$x \approx -1.28$）。负输入仍有非零梯度，缓解了 ReLU 的“死神经元”问题。
        - **tanh 近似**：早年 erf 在部分硬件上慢，GPT-2 用了 tanh 近似；两者在 $[-6, 6]$ 上的最大绝对差约 $4.7\times10^{-4}$。PyTorch 中是 \`F.gelu(x, approximate="tanh")\`。加载预训练权重时要用与训练一致的那一种。
      `,
      viz: ActivationViz,
      checkpoint: {
        id: 'cp-act',
        type: 'single',
        prompt: md`关于 SiLU（$x\sigma(x)$），下列说法正确的是：`,
        options: [
          '它是单调递增函数',
          '当 $x \\to -\\infty$ 时它趋于 0，但在 $x \\approx -1.28$ 处取得约 $-0.28$ 的最小值',
          '它在 $x = 0$ 处不可导',
          '它的输出总是非负的',
        ],
        answer: 1,
        explanation: md`SiLU 在负半轴先下降再回升到 0，最小值约 $-0.278$，所以不单调（A 错）、有负输出（D 错）；它处处光滑可导（C 错），$\mathrm{SiLU}'(0) = 0.5$。`,
      },
    },
    {
      id: 'glu',
      title: 'GLU 族与 SwiGLU：多一个“门”',
      body: md`
        **GLU**（Dauphin 等，2017）把一条线性分支当作“门”，逐元素调节另一条分支。Shazeer（2020）把它用到 Transformer 的 FFN 里，并比较了 ReGLU、GEGLU、SwiGLU 等变体：

        $$
        \mathrm{FFN}_{\text{SwiGLU}}(x) = \big(\mathrm{SiLU}(x W_{\text{gate}}) \odot x W_{\text{up}}\big)\, W_{\text{down}}
        $$

        - 三个矩阵：$W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d\times h}$，$W_{\text{down}} \in \mathbb{R}^{h\times d}$（HF 里叫 \`gate_proj\`、\`up_proj\`、\`down_proj\`）。
        - 把 SiLU 换成 GELU 就是 **GeGLU**（Gemma 系列），换成 ReLU 就是 ReGLU。
        - 输出是两条分支的**乘积**，是输入的二次型交互，表达力比“线性 + 逐元素激活”更强。Shazeer 论文中 GEGLU、SwiGLU 的困惑度最好，原文对原因的“解释”是一句玩笑：归功于 divine benevolence（神的眷顾）——也就是说，这是**经验结论**，没有公认的理论解释。

        实现上通常把 $W_{\text{gate}}$ 和 $W_{\text{up}}$ 拼成一个 $(d, 2h)$ 的矩阵，一次 matmul 后再切开，减少 kernel 启动并提高 GPU 利用率。
      `,
      viz: SwiGLUStepViz,
      checkpoint: {
        id: 'cp-glu',
        type: 'judge',
        prompt: md`SwiGLU 中，SiLU 作用在 $x W_{\text{up}}$ 分支上，$x W_{\text{gate}}$ 分支保持线性。`,
        answer: false,
        explanation: md`激活作用在**门控分支** $xW_{\text{gate}}$ 上：$\mathrm{SiLU}(xW_{\text{gate}}) \odot xW_{\text{up}}$。名字上的 gate / up 只是约定，但 HF LLaMA 的实现就是 \`down_proj(act_fn(gate_proj(x)) * up_proj(x))\`，加载权重时搞反会得到完全错误的输出。`,
      },
    },
    {
      id: 'dim',
      title: '8/3·d 是怎么来的',
      body: md`
        标准 FFN 两个矩阵，参数 $2 \cdot d \cdot 4d = 8d^2$。SwiGLU 有三个矩阵，若隐藏维仍取 $4d$，参数会变成 $12d^2$，与基线比较就不公平了。Shazeer 的做法是让参数量（以及 FLOPs）保持不变：

        $$
        3 \cdot d \cdot h = 8 d^2 \quad\Rightarrow\quad h = \frac{8}{3}d \approx 2.67d
        $$

        $\frac{8}{3}d$ 一般不是整数，也不是硬件友好的尺寸。LLaMA 的代码是：

        \`\`\`python
        hidden_dim = 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)                      # 8/3·d，向下取整
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)     # 可选：再放宽
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)   # 向上取整到倍数
        \`\`\`

        LLaMA-7B：$d = 4096$，$\lfloor 8 \cdot 4096/3 \rfloor = 10922$，向上取整到 256 的倍数得 **11008**。
        为什么要对齐：GPU 的矩阵乘以 tile 为单位（如 64/128），维度是大 2 的幂的倍数时 Tensor Core 利用率最高；张量并行切分时也要求能被 TP 数整除。
      `,
      viz: FFNParamViz,
      checkpoint: {
        id: 'cp-dim',
        type: 'fill',
        prompt: md`按 LLaMA 的公式（无 multiplier，\`multiple_of = 256\`），$d = 2048$ 的模型 SwiGLU 隐藏维 $h$ = ____(1)____；$d = 5120$ 时 $h$ = ____(2)____。`,
        blanks: [
          { label: '(1)', answers: ['5632'], numeric: true, tolerance: 0 },
          { label: '(2)', answers: ['13824'], numeric: true, tolerance: 0 },
        ],
        explanation: md`$\lfloor 8\cdot2048/3\rfloor = 5461$，向上取整到 256 的倍数 $= 22 \times 256 = 5632$；$\lfloor 8\cdot5120/3\rfloor = 13653$，$\lceil 13653/256\rceil = 54$，$54 \times 256 = 13824$（正是 LLaMA-13B 的配置）。`,
      },
    },
  ],

  /* ================================================================ */
  /* 随堂练习                                                           */
  /* ================================================================ */
  quiz: [
    {
      id: 'q-formula',
      type: 'single',
      difficulty: 1,
      prompt: md`LLaMA 中 SwiGLU FFN 的正确计算是：`,
      options: [
        '`down(silu(gate(x)) * up(x))`',
        '`down(silu(gate(x) * up(x)))`',
        '`down(silu(up(x))) * gate(x)`',
        '`silu(down(gate(x) + up(x)))`',
      ],
      answer: 0,
      explanation: md`激活只作用在门控分支，再与内容分支逐元素相乘，最后降维。B 把激活放在乘积之后；C 在降维后才乘门（形状都对不上）；D 是加法且激活位置错误。`,
    },
    {
      id: 'q-gelu-neg',
      type: 'judge',
      difficulty: 1,
      prompt: md`GELU 在某些负输入上的输出是负数。`,
      answer: true,
      explanation: md`$\mathrm{GELU}(x) = x\Phi(x)$，$x<0$ 时 $x$ 为负、$\Phi(x)$ 为正，乘积为负。最小值约 $-0.17$，出现在 $x \approx -0.75$。`,
    },
    {
      id: 'q-params',
      type: 'numeric',
      difficulty: 2,
      prompt: md`LLaMA-7B 每层 SwiGLU FFN（$d = 4096$，$h = 11008$，无 bias）有多少个参数？（输入精确整数，或用表达式如 \`3*4096*11008\`）`,
      answer: 135266304,
      tolerance: 0,
      explanation: md`$3 \times 4096 \times 11008 = 135{,}266{,}304 \approx 135.3\text{M}$。32 层共约 4.33B，占 7B 模型参数的约 2/3（注意力每层 $4d^2 \approx 67.1\text{M}$）。`,
    },
    {
      id: 'q-equal',
      type: 'fill',
      difficulty: 2,
      prompt: md`Shazeer（2020）的实验中基线 FFN 为 $d = 768$、$d_{ff} = 3072$。为了让 GLU 变体的参数量与基线相同，GLU 变体的 $d_{ff}$ 取 ____(1)____。`,
      blanks: [{ label: '(1)', answers: ['2048'], numeric: true, tolerance: 0 }],
      explanation: md`$2 \cdot 768 \cdot 3072 = 3 \cdot 768 \cdot d_{ff} \Rightarrow d_{ff} = 2 \cdot 3072/3 = 2048$，正是 $\frac{2}{3}\times 4d$。`,
    },
    {
      id: 'q-act-multi',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于激活函数，正确的说法有（多选）：`,
      options: [
        'GELU 与 SiLU 都是光滑函数，在 0 处可导',
        'tanh 近似的 GELU 与精确 GELU 的最大绝对误差小于 $10^{-3}$',
        'Swish$_\\beta(x) = x\\sigma(\\beta x)$ 在 $\\beta \\to \\infty$ 时趋近 ReLU',
        'ReLU 在负半轴的梯度为 0，可能导致神经元“死亡”',
        'SiLU 的输出范围是 $[0, +\\infty)$',
      ],
      answer: [0, 1, 2, 3],
      explanation: md`B：最大差约 $4.7\times10^{-4}$。C：$\beta\to\infty$ 时 $\sigma(\beta x)$ 变成阶跃函数。E 错：SiLU 最小约 $-0.278$，有负输出。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面的 SwiGLU 实现（\`x: (T, d)\`，\`w_gate, w_up: (d, h)\`，\`w_down: (h, d)\`）有 **2 处** bug：`,
      code: py`
        def swiglu_ffn(x, w_gate, w_up, w_down):
            gate = x @ w_gate
            up = x @ w_up
            h = silu(gate) + up
            return h @ w_down

        def silu(z):
            return z / (1 + np.exp(z))
      `,
      bugLines: [4, 8],
      explanation: md`
        - 第 4 行门控应是**逐元素乘** \`silu(gate) * up\`，不是加法。
        - 第 8 行 sigmoid 写反了：$\sigma(z) = 1/(1 + e^{-z})$，所以应为 \`z / (1 + np.exp(-z))\`。现在的写法等于 $z\,\sigma(-z)$。
        另外 \`np.exp(-z)\` 在 $z$ 很负时会溢出成 inf（结果 \`z / inf = -0.0\` 虽然数值上可接受，但会触发溢出警告），稳妥写法见跟练。
      `,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 2,
      prompt: md`把一个 token 通过 Pre-Norm block 中 SwiGLU FFN 子层的步骤排好序：`,
      items: [
        '$h = \\mathrm{RMSNorm}(x)$',
        '并行计算 $g = hW_{\\text{gate}}$ 与 $u = hW_{\\text{up}}$',
        '$z = \\mathrm{SiLU}(g) \\odot u$',
        '$o = z W_{\\text{down}}$',
        '$x \\leftarrow x + o$',
      ],
      explanation: md`先归一化、再两条分支升维、门控相乘、降维，最后加回残差流。$g$ 与 $u$ 互不依赖，实现上通常拼成一次 matmul。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 1,
      prompt: md`把 FFN 变体与采用它的代表模型配对：`,
      pairs: [
        ['ReLU，$d_{ff} = 4d$', '原始 Transformer'],
        ['GELU，$d_{ff} = 4d$', 'GPT-2 / BERT'],
        ['SwiGLU，$h \\approx \\frac{8}{3}d$', 'LLaMA'],
        ['GeGLU', 'Gemma'],
      ],
      explanation: md`原始 Transformer 用 ReLU；BERT 与 GPT-2 用 GELU；LLaMA 把 SwiGLU 带进了开源 LLM 的标准配方（PaLM 也用 SwiGLU）；Gemma 用 GELU 作门的 GeGLU。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出 LLaMA 风格的 SwiGLU 隐藏维计算函数。有 2 行干扰项。`,
      lines: [
        'def ffn_hidden_dim(dim, multiple_of=256, ffn_dim_multiplier=None):',
        '    hidden = int(2 * (4 * dim) / 3)',
        '    if ffn_dim_multiplier is not None:',
        '        hidden = int(ffn_dim_multiplier * hidden)',
        '    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)',
      ],
      distractors: ['    return multiple_of * (hidden // multiple_of)', '    hidden = int(4 * dim * 3 / 2)'],
      explanation: md`先取 $\frac{2}{3}\cdot 4d$，可选乘 multiplier，最后**向上**取整到 \`multiple_of\` 的倍数。干扰项 1 是向下取整（LLaMA-7B 会得到 10752 而不是 11008）；干扰项 2 把 2/3 写成了 3/2。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`SwiGLU 的 8 个隐藏单元，门控分支的预激活 $a$ 如下。哪些单元的门值 $|\mathrm{SiLU}(a)| < 0.05$（几乎被关闭）？`,
      items: ['a = −6', 'a = −3', 'a = −1', 'a = 0', 'a = 0.5', 'a = 2', 'a = −0.02', 'a = 4'],
      answer: [0, 3, 6],
      explanation: md`$\mathrm{SiLU}(-6) \approx -0.015$，$\mathrm{SiLU}(0) = 0$，$\mathrm{SiLU}(-0.02) \approx -0.0099$，都几乎关闭。$\mathrm{SiLU}(-3) \approx -0.142$、$\mathrm{SiLU}(-1) \approx -0.269$ 的绝对值并不小——SiLU 的负半轴不是 0，这是它与 ReLU 门的关键区别。`,
    },
    {
      id: 'q-why-83',
      type: 'single',
      difficulty: 3,
      prompt: md`SwiGLU 把隐藏维从 $4d$ 改成约 $\frac{8}{3}d$，最主要的目的是：`,
      options: [
        '让隐藏维成为 2 的幂',
        '三个矩阵 $3dh = 8d^2$，让参数量与 FLOPs 和标准 4d FFN 持平，便于公平比较与替换',
        '减少激活函数的计算量',
        '让门控分支和内容分支的维度不同',
      ],
      answer: 1,
      explanation: md`$\frac{8}{3}d$ 恰好使 $3 \cdot d \cdot h = 2 \cdot d \cdot 4d$。A 恰恰相反——$\frac{8}{3}d$ 不是 2 的幂，所以才需要再对齐到 \`multiple_of\`。C：激活的计算量相对 matmul 可以忽略。D：两条分支维度必须相同才能逐元素相乘。`,
    },
    {
      id: 'q-llama3',
      type: 'numeric',
      difficulty: 3,
      prompt: md`Llama 3-8B：$d = 4096$，\`ffn_dim_multiplier = 1.3\`，\`multiple_of = 1024\`。按 LLaMA 代码算出的 FFN 隐藏维是多少？`,
      answer: 14336,
      tolerance: 0,
      explanation: md`$\lfloor 2\cdot16384/3 \rfloor = 10922$ → $\lfloor 1.3 \times 10922 \rfloor = 14198$ → 向上取整到 1024 的倍数：$14 \times 1024 = 14336$，即 $3.5d$。比“等参数”的 $\frac{8}{3}d$ 宽了约 31%。`,
    },
  ],

  /* ================================================================ */
  /* 跟练                                                              */
  /* ================================================================ */
  guided: [
    {
      id: 'g-swiglu',
      title: '数值稳定的 sigmoid / SiLU 与 SwiGLU 前向',
      intro: md`
        实现 \`sigmoid\`、\`silu\` 与 \`swiglu_ffn(x, w_gate, w_up, w_down)\`（\`x\` 形状 \`(..., d)\`）。
        要求 \`sigmoid\` 在 $|z|$ 很大（如 $\pm 1000$）时**不产生任何溢出警告**——测试会在 \`np.errstate(over="raise")\` 下运行。
      `,
      template: py`
        import numpy as np

        def sigmoid(z):
            z = np.asarray(z, dtype=np.float64)
            e = {{1}}
            return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))

        def silu(z):
            return {{2}}

        def swiglu_ffn(x, w_gate, w_up, w_down):
            h = {{3}}
            return {{4}}
      `,
      steps: [
        {
          blank: 1,
          instruction: md`计算一个**永远不会溢出**的指数：对 $|z|$ 取负再求 exp，结果在 $(0, 1]$。`,
          answer: 'np.exp(-np.abs(z))',
          hint: '两个分支共用同一个 e。',
          explain: md`
            $z \ge 0$ 时 $\sigma(z) = 1/(1 + e^{-z})$；$z < 0$ 时 $\sigma(z) = e^{z}/(1 + e^{z})$。两种情况下指数的参数都是 $-|z| \le 0$，不会溢出。
            注意 \`np.where\` 会**同时计算两个分支**，所以不能写成 \`np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))\`——没被选中的分支照样会溢出并报警。
          `,
        },
        {
          blank: 2,
          instruction: md`SiLU：$z\,\sigma(z)$。`,
          answer: 'z * sigmoid(z)',
          explain: md`在 $z = -1000$ 时 \`sigmoid\` 返回约 $e^{-1000} \approx 0$（下溢为 0，不会报溢出），乘积为 0。`,
        },
        {
          blank: 3,
          instruction: md`门控相乘：激活作用在门控分支 \`x @ w_gate\`，再与内容分支 \`x @ w_up\` 逐元素相乘。`,
          answer: 'silu(x @ w_gate) * (x @ w_up)',
          accept: ['(x @ w_up) * silu(x @ w_gate)', 'silu(np.matmul(x, w_gate)) * np.matmul(x, w_up)'],
          explain: md`\`@\` 对 \`(..., d) @ (d, h)\` 自动广播前导维，得到 \`(..., h)\`。`,
        },
        {
          blank: 4,
          instruction: md`降维回 $d$。`,
          answer: 'h @ w_down',
          accept: ['np.matmul(h, w_down)', 'np.dot(h, w_down)'],
          explain: md`\`w_down\` 形状 \`(h, d)\`，不需要转置。PyTorch 的 \`nn.Linear\` 存的是 \`(out, in)\` 的权重并计算 $xW^\top$，从 PyTorch 导出权重到 NumPy 时要注意转置。`,
        },
      ],
      setup: py`
        def _ref_ffn(x, wg, wu, wd):
            a = x @ wg
            return (a / (1 + np.exp(-a)) * (x @ wu)) @ wd
      `,
      tests: [
        {
          name: 'sigmoid / silu 数值',
          code: py`
            assert np.allclose(sigmoid(np.array([0.0, 2.0, -2.0])), [0.5, 0.88079708, 0.11920292]), f"sigmoid 错误：{sigmoid(np.array([0.0, 2.0, -2.0]))}"
            assert np.allclose(silu(np.array([-1.0, 0.0, 1.0])), [-0.26894142, 0.0, 0.73105858]), f"silu 错误：{silu(np.array([-1.0, 0.0, 1.0]))}"
          `,
        },
        {
          name: '极端输入不溢出',
          code: py`
            with np.errstate(over="raise"):
                s = sigmoid(np.array([-1000.0, 1000.0, -50.0]))
                y = silu(np.array([-1000.0, 1000.0]))
            assert np.allclose(s, [0.0, 1.0, 1.9287498e-22]), f"得到 {s}"
            assert np.allclose(y, [0.0, 1000.0]), f"得到 {y}"
          `,
        },
        {
          name: 'SwiGLU 前向（三维输入）',
          code: py`
            rng = np.random.default_rng(0)
            x = rng.normal(size=(2, 3, 8))
            wg, wu, wd = rng.normal(size=(8, 20)), rng.normal(size=(8, 20)), rng.normal(size=(20, 8))
            out = swiglu_ffn(x, wg, wu, wd)
            assert out.shape == (2, 3, 8), f"形状应为 (2, 3, 8)，得到 {out.shape}"
            assert np.allclose(out, _ref_ffn(x, wg, wu, wd)), f"最大误差 {np.abs(out - _ref_ffn(x, wg, wu, wd)).max():.2e}"
          `,
        },
      ],
      outro: md`
        这三个函数就是 LLaMA \`MLP.forward\` 的全部内容。实战题会再加上 GELU 的两种写法，并把激活函数做成可切换的参数（SwiGLU / GeGLU / ReGLU）。
      `,
    },
  ],

  /* ================================================================ */
  /* 实战                                                              */
  /* ================================================================ */
  challenges: [
    {
      id: 'c-glu-ffn',
      title: 'GLU 族 FFN：SwiGLU / GeGLU / ReGLU',
      difficulty: 'medium',
      source: '大模型岗手撕 · FFN',
      timeLimitMin: 20,
      description: md`
        实现：

        \`\`\`python
        def gelu(x: np.ndarray, approximate: str = "none") -> np.ndarray: ...
        def glu_ffn(x, w_gate, w_up, w_down, act: str = "silu") -> np.ndarray: ...
        \`\`\`

        - \`gelu\`：\`approximate="none"\` 为精确形式 $x\Phi(x) = \frac{x}{2}(1 + \mathrm{erf}(x/\sqrt2))$；\`"tanh"\` 为 $\frac{x}{2}\big(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3))\big)$。（语义同 \`torch.nn.functional.gelu\`）
        - \`glu_ffn\`：$\big(\phi(xW_{\text{gate}}) \odot xW_{\text{up}}\big)W_{\text{down}}$，\`act\` 取 \`"silu"\`（SwiGLU）、\`"gelu"\`（GeGLU，精确 GELU）、\`"gelu_tanh"\`、\`"relu"\`（ReGLU）。
        - \`x\` 形状 \`(..., d)\`；\`w_gate, w_up\`：\`(d, h)\`；\`w_down\`：\`(h, d)\`。
        - 数值稳定：输入可能有 $\pm 1000$ 量级的值，不能产生溢出警告（测试在 \`np.errstate(over="raise")\` 下运行）。

        **注意**：NumPy 没有 \`erf\`，可以用标准库 \`math.erf\` 配合 \`np.vectorize\`。
      `,
      starter: py`
        import math
        import numpy as np

        def gelu(x, approximate="none"):
            # TODO: 精确版用 erf；tanh 近似版用 0.044715 那个公式
            raise NotImplementedError

        def glu_ffn(x, w_gate, w_up, w_down, act="silu"):
            # TODO: 门控分支过激活，与内容分支逐元素相乘，再降维
            raise NotImplementedError
      `,
      solution: py`
        import math
        import numpy as np

        _erf = np.vectorize(math.erf, otypes=[np.float64])

        def gelu(x, approximate="none"):
            x = np.asarray(x, dtype=np.float64)
            if approximate == "tanh":
                c = math.sqrt(2.0 / math.pi)
                return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))
            return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))

        def _sigmoid(z):
            e = np.exp(-np.abs(z))                        # 指数参数 ≤ 0，永不溢出
            return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))

        def glu_ffn(x, w_gate, w_up, w_down, act="silu"):
            x = np.asarray(x, dtype=np.float64)
            g = x @ w_gate
            if act == "silu":
                a = g * _sigmoid(g)
            elif act == "gelu":
                a = gelu(g)
            elif act == "gelu_tanh":
                a = gelu(g, approximate="tanh")
            elif act == "relu":
                a = np.maximum(g, 0.0)
            else:
                raise ValueError(f"未知激活 {act}")
            return (a * (x @ w_up)) @ w_down
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示：HF LLaMA 风格 MLP）',
          code: py`
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            class GLUFFN(nn.Module):
                def __init__(self, d: int, h: int, act: str = "silu"):
                    super().__init__()
                    self.gate_proj = nn.Linear(d, h, bias=False)
                    self.up_proj = nn.Linear(d, h, bias=False)
                    self.down_proj = nn.Linear(h, d, bias=False)
                    self.act = {"silu": F.silu, "gelu": F.gelu,
                                "gelu_tanh": lambda t: F.gelu(t, approximate="tanh"), "relu": F.relu}[act]

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
          `,
        },
      ],
      explanation: md`
        **要点**
        - 精确 GELU 需要 erf：NumPy 没有，可用 \`np.vectorize(math.erf)\`（较慢但正确）；PyTorch / SciPy 有向量化的 \`erf\`。
        - SiLU 的 sigmoid 要用 $e^{-|z|}$ 的写法；\`np.where\` 两个分支都会被计算，不能把会溢出的表达式放进去。
        - tanh 近似里的 $x^3$ 在 $|x| = 1000$ 时约 $10^9$，不会溢出，\`np.tanh\` 对大输入会饱和到 ±1，是安全的。

        **复杂度**：$O(N d h)$，三次矩阵乘法主导；激活是 $O(Nh)$。
      `,
      complexity: { time: 'O(N·d·h)', space: 'O(N·h)' },
      hints: [
        md`精确 GELU：\`0.5 * x * (1 + erf(x / sqrt(2)))\`，其中 \`erf = np.vectorize(math.erf)\`。`,
        md`稳定 sigmoid：\`e = np.exp(-np.abs(z))\`，再 \`np.where(z >= 0, 1/(1+e), e/(1+e))\`。`,
        md`\`glu_ffn\`：\`(act(x @ w_gate) * (x @ w_up)) @ w_down\`。`,
      ],
      setup: py`
        _ref_erf = np.vectorize(math.erf, otypes=[np.float64])

        def _ref_act(g, act):
            if act == "silu":
                return g * (0.5 * (1 + np.tanh(g / 2)))       # sigmoid(g) = (1 + tanh(g/2)) / 2
            if act == "gelu":
                return 0.5 * g * (1 + _ref_erf(g / math.sqrt(2)))
            if act == "gelu_tanh":
                return 0.5 * g * (1 + np.tanh(math.sqrt(2 / math.pi) * (g + 0.044715 * g ** 3)))
            return np.maximum(g, 0)

        def _ref_glu(x, wg, wu, wd, act):
            g = x @ wg
            return (_ref_act(g, act) * (x @ wu)) @ wd
      `,
      tests: [
        {
          name: 'GELU 数值',
          code: py`
            x = np.array([-3.0, -0.75, 0.0, 1.0, 2.5])
            exact = gelu(x)
            approx = gelu(x, approximate="tanh")
            exp = np.array([-0.00404969, -0.16997051, 0.0, 0.84134475, 2.48447584])
            assert np.allclose(exact, exp, atol=1e-7), f"精确 GELU 期望 {exp}，得到 {exact}"
            assert np.abs(exact - approx).max() < 5e-4, f"tanh 近似与精确值相差过大：{np.abs(exact - approx).max():.2e}"
            assert not np.allclose(exact, approx, atol=1e-6), "两种 GELU 不应完全相同——是不是忽略了 approximate 参数？"
          `,
        },
        {
          name: 'SwiGLU 形状与数值',
          code: py`
            rng = np.random.default_rng(0)
            x = rng.normal(size=(3, 8))
            wg, wu, wd = rng.normal(size=(8, 12)), rng.normal(size=(8, 12)), rng.normal(size=(12, 8))
            out = glu_ffn(x, wg, wu, wd)
            assert out.shape == (3, 8), f"形状应为 (3, 8)，得到 {out.shape}"
            exp = _ref_glu(x, wg, wu, wd, "silu")
            assert np.allclose(out, exp), f"最大误差 {np.abs(out - exp).max():.2e}"
          `,
        },
        {
          name: '四种激活',
          code: py`
            rng = np.random.default_rng(1)
            x = rng.normal(size=(2, 5, 6))
            wg, wu, wd = rng.normal(size=(6, 10)), rng.normal(size=(6, 10)), rng.normal(size=(10, 6))
            for act in ("silu", "gelu", "gelu_tanh", "relu"):
                out = glu_ffn(x, wg, wu, wd, act=act)
                exp = _ref_glu(x, wg, wu, wd, act)
                assert out.shape == (2, 5, 6), f"act={act} 形状错误：{out.shape}"
                assert np.allclose(out, exp, atol=1e-9), f"act={act} 最大误差 {np.abs(out - exp).max():.2e}"
          `,
        },
        {
          name: '极端输入不溢出',
          hidden: true,
          code: py`
            x = np.array([[1000.0, -1000.0, 0.5]])
            w = np.eye(3)
            with np.errstate(over="raise"):
                for act in ("silu", "gelu", "gelu_tanh", "relu"):
                    out = glu_ffn(x, w, w, w, act=act)
                    assert np.all(np.isfinite(out)), f"act={act} 出现 inf/NaN：{out}"
                    exp = _ref_glu(x, w, w, w, act)
                    assert np.allclose(out, exp), f"act={act} 期望 {exp}，得到 {out}"
          `,
        },
        {
          name: 'GELU 大规模对拍',
          hidden: true,
          code: py`
            rng = np.random.default_rng(2)
            x = rng.normal(size=(64, 16)) * 4
            assert np.allclose(gelu(x), 0.5 * x * (1 + _ref_erf(x / math.sqrt(2))), atol=1e-12), "精确 GELU 不一致"
            t = 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
            assert np.allclose(gelu(x, approximate="tanh"), t, atol=1e-12), "tanh 近似 GELU 不一致"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-fuse',
          q: md`推理时 SwiGLU FFN 有哪些常见优化？`,
          a: md`
            (1) 把 \`gate_proj\` 与 \`up_proj\` 合并成一个 $(d, 2h)$ 的矩阵，一次 GEMM；(2) 把 SiLU 与逐元素乘融合进 GEMM 的 epilogue，或单独写一个 fused \`silu_and_mul\` kernel（vLLM 就有），避免把 $(N, 2h)$ 的中间结果写回显存；(3) 张量并行时按列切 \`gate/up\`、按行切 \`down\`，每层只需一次 all-reduce。
          `,
        },
      ],
    },
    {
      id: 'c-ffn-dim',
      title: 'FFN 隐藏维与参数量计算',
      difficulty: 'easy',
      source: '大模型岗手撕 · 参数量估算',
      timeLimitMin: 10,
      description: md`
        实现两个函数：

        \`\`\`python
        def ffn_hidden_dim(dim: int, multiple_of: int = 256, ffn_dim_multiplier: float | None = None) -> int: ...
        def ffn_params(dim: int, hidden: int, gated: bool = True, bias: bool = False) -> int: ...
        \`\`\`

        - \`ffn_hidden_dim\`：严格按 LLaMA 代码——初值 $4d$；取 \`int(2 * hidden / 3)\`；若 multiplier 非 None，再取 \`int(multiplier * hidden)\`；最后向上取整到 \`multiple_of\` 的倍数。
        - \`ffn_params\`：\`gated=True\` 为 SwiGLU 式三个矩阵（gate、up：$d\times h$，down：$h\times d$），\`False\` 为标准两矩阵 FFN。\`bias=True\` 时每个线性层都带 bias（升维层 bias 长度 $h$，降维层长度 $d$）。

        **示例**：\`ffn_hidden_dim(4096)\` → \`11008\`；\`ffn_params(4096, 11008)\` → \`135266304\`。
      `,
      starter: py`
        def ffn_hidden_dim(dim, multiple_of=256, ffn_dim_multiplier=None):
            # TODO
            return 4 * dim

        def ffn_params(dim, hidden, gated=True, bias=False):
            # TODO
            return 0
      `,
      solution: py`
        def ffn_hidden_dim(dim, multiple_of=256, ffn_dim_multiplier=None):
            hidden = 4 * dim
            hidden = int(2 * hidden / 3)                          # 8/3·d 向下取整
            if ffn_dim_multiplier is not None:
                hidden = int(ffn_dim_multiplier * hidden)
            return multiple_of * ((hidden + multiple_of - 1) // multiple_of)   # 向上取整到倍数

        def ffn_params(dim, hidden, gated=True, bias=False):
            n_up = 2 if gated else 1                              # 升维矩阵个数
            params = n_up * dim * hidden + hidden * dim
            if bias:
                params += n_up * hidden + dim
            return params
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示：用模型实例核对）',
          code: py`
            import torch.nn as nn

            def count_params(module: nn.Module) -> int:
                return sum(p.numel() for p in module.parameters())

            # 例：HF LLaMA 的 MLP
            # from transformers.models.llama.modeling_llama import LlamaMLP
            # count_params(LlamaMLP(config))  == 3 * hidden_size * intermediate_size
          `,
        },
      ],
      explanation: md`
        **要点**：三个取整步骤的顺序与方向都不能错——先向下取整 \`int(2h/3)\`，multiplier 后再 \`int\`，最后**向上**取整到倍数。

        **核对**：LLaMA-7B → 11008；Llama 3-8B（1.3, 1024）→ 14336；Llama 2-70B（$d$=8192, 1.3, 4096）→ 28672。

        **参数量**：SwiGLU $3dh$，标准 FFN $2dh$；带 bias 时再加 $2h + d$ 或 $h + d$。
      `,
      complexity: { time: 'O(1)', space: 'O(1)' },
      hints: [
        md`向上取整到倍数：\`m * ((x + m - 1) // m)\`。`,
        md`\`ffn_params\`：门控时升维矩阵有 2 个，否则 1 个；降维矩阵始终 1 个。`,
      ],
      tests: [
        {
          name: 'LLaMA-7B',
          code: py`
            assert ffn_hidden_dim(4096) == 11008, f"期望 11008，得到 {ffn_hidden_dim(4096)}"
            assert ffn_params(4096, 11008) == 135266304, f"期望 135266304，得到 {ffn_params(4096, 11008)}"
          `,
        },
        {
          name: 'Llama 3-8B',
          code: py`
            got = ffn_hidden_dim(4096, multiple_of=1024, ffn_dim_multiplier=1.3)
            assert got == 14336, f"期望 14336，得到 {got}"
          `,
        },
        {
          name: '标准 FFN 与 bias',
          code: py`
            assert ffn_params(768, 3072, gated=False) == 2 * 768 * 3072, f"得到 {ffn_params(768, 3072, gated=False)}"
            assert ffn_params(768, 3072, gated=False, bias=True) == 2 * 768 * 3072 + 3072 + 768, "bias 计数错误"
            assert ffn_params(8, 16, gated=True, bias=True) == 3 * 8 * 16 + 2 * 16 + 8, "门控 + bias 计数错误"
          `,
        },
        {
          name: '更多真实配置',
          hidden: true,
          code: py`
            cases = [((8192, 4096, 1.3), 28672), ((5120, 256, None), 13824), ((2048, 256, None), 5632), ((6656, 256, None), 17920)]
            for (d, mo, mult), exp in cases:
                got = ffn_hidden_dim(d, multiple_of=mo, ffn_dim_multiplier=mult)
                assert got == exp, f"d={d}, multiple_of={mo}, mult={mult}：期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '取整细节',
          hidden: true,
          code: py`
            # d=96：int(2*384/3)=256，恰好是 256 的倍数，不应再进位
            assert ffn_hidden_dim(96) == 256, f"恰为倍数时不应进位，得到 {ffn_hidden_dim(96)}"
            # d=100：int(800/3)=266 → 向上取整到 64 的倍数 = 320
            assert ffn_hidden_dim(100, multiple_of=64) == 320, f"得到 {ffn_hidden_dim(100, multiple_of=64)}"
            # multiplier 之后也要 int()：d=100, mult=1.5 → int(1.5*266)=399 → 向上取整到 32 的倍数 = 416
            assert ffn_hidden_dim(100, multiple_of=32, ffn_dim_multiplier=1.5) == 416, f"得到 {ffn_hidden_dim(100, multiple_of=32, ffn_dim_multiplier=1.5)}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-why-not-4d',
          q: md`如果 SwiGLU 直接用 $h = 4d$ 会怎样？Llama 3 为什么又乘了 1.3？`,
          a: md`
            $h = 4d$ 时 FFN 参数变成 $12d^2$，比标准 FFN 多 50%，与基线比较时混入了“参数更多”的因素。Llama 3 乘 1.3 是有意让 FFN 更宽（约 $3.5d$）：在总参数预算下把更多参数放到 FFN，同时用 GQA 缩小注意力的 KV 投影——这是架构取舍，不是公式本身的要求。
          `,
        },
      ],
    },
  ],

  /* ================================================================ */
  /* 面试追问                                                           */
  /* ================================================================ */
  interview: [
    {
      id: 'qa-swiglu',
      freq: 3,
      q: md`SwiGLU 是什么？为什么大模型都用它？隐藏维为什么是 8/3·d？`,
      a: md`
        **结论**：SwiGLU FFN $= (\mathrm{SiLU}(xW_g) \odot xW_u)W_d$，用一条分支做门、逐元素调节另一条分支；为保持与 $4d$ 标准 FFN 相同的参数量与 FLOPs，隐藏维取 $\frac{8}{3}d$ 并对齐到硬件友好的倍数。

        **原因**：Shazeer（2020）在相同参数量下比较多种 GLU 变体，GEGLU / SwiGLU 的困惑度与下游效果都优于 ReLU / GELU FFN；乘性门控引入二次交互，表达力更强，且 SiLU 光滑、负半轴有梯度。论文本身没有给出理论解释，这是经验结论。

        **延伸**：$3dh = 8d^2 \Rightarrow h = \frac{8}{3}d$；LLaMA-7B 为 11008（$\lfloor 10922.67 \rfloor$ 向上对齐到 256 的倍数）；Llama 3 再乘 1.3 得 14336。工程上把 gate、up 合并成一次 GEMM，并融合 \`silu_and_mul\`。
      `,
    },
    {
      id: 'qa-gelu',
      freq: 2,
      q: md`GELU 和 ReLU 的区别？tanh 近似是怎么回事？`,
      a: md`
        **结论**：$\mathrm{GELU}(x) = x\Phi(x)$，是光滑、非单调的“软 ReLU”：大正数近似恒等，大负数近似 0，中间平滑过渡，负半轴有小的负值与非零梯度。

        **原因**：ReLU 在 0 处不可导、负半轴梯度为 0（可能“死神经元”）；GELU 的平滑性在深层 Transformer 中训练更稳定、效果更好（BERT、GPT 系列采用）。

        **tanh 近似**：$\frac{x}{2}(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$，来自对 $\Phi$ 的近似，早期 erf 在部分硬件上较慢；与精确值最大差约 $4.7\times10^{-4}$。GPT-2 用的是 tanh 近似，PyTorch 对应 \`approximate="tanh"\`。加载预训练权重时激活要与训练时一致，否则会有细微但系统性的偏差。
      `,
    },
    {
      id: 'qa-ffn-share',
      freq: 2,
      q: md`一个 Transformer 层里 FFN 和注意力各占多少参数和计算？对 MoE 有什么启发？`,
      a: md`
        **结论**：标准 MHA 约 $4d^2$，FFN（$4d$）约 $8d^2$，FFN 占约 2/3；SwiGLU（$\frac{8}{3}d$）同样是 $8d^2$。FLOPs 方面，每 token 的线性层 FLOPs 约为 $2\times$参数量，FFN 同样占大头；注意力分数部分（$QK^\top$、$PV$）随序列长度线性增长，长上下文时才变得显著。

        **对 MoE 的启发**：既然参数与计算主要在 FFN，把 FFN 换成多个专家、每个 token 只激活少数几个，就能在几乎不增加每 token 计算的情况下大幅增加总参数。DeepSeek-V3 每个 MoE 层有 256 个路由专家 + 1 个共享专家，每个专家的中间维只有 2048（$d = 7168$），每 token 激活 8 个路由专家。
      `,
    },
  ],

  cheatsheet: md`
    **SwiGLU FFN**
    \`\`\`python
    def swiglu_ffn(x, w_gate, w_up, w_down):
        g = x @ w_gate
        return (g * sigmoid(g) * (x @ w_up)) @ w_down

    def sigmoid(z):                      # 不溢出
        e = np.exp(-np.abs(z))
        return np.where(z >= 0, 1 / (1 + e), e / (1 + e))
    \`\`\`

    | | 公式 / 数字 |
    |---|---|
    | GELU | $x\Phi(x)$；tanh 近似常数 0.044715，误差 ≤ 4.7e-4 |
    | SiLU | $x\sigma(x)$，最小 ≈ −0.278 |
    | 隐藏维 | $3dh = 8d^2 \Rightarrow h = \frac83 d$，再向上对齐 |
    | LLaMA-7B | $d$=4096 → 11008；Llama 3-8B（×1.3, 1024）→ 14336 |
    | 参数 | 注意力 $4d^2$，FFN $8d^2$（占 2/3） |

    - 激活在 **gate** 分支；\`np.where\` 两个分支都会算
    - LLaMA 取整：\`int(2*4d/3)\` →（×mult 再 int）→ 向上取整到 multiple_of
    - nn.Linear 权重是 (out, in)，导出到 NumPy 要转置
  `,
})
