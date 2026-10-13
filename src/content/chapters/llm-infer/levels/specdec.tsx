/**
 * L10-4 投机解码原理：接受规则 min(1, p/q)、残差分布 norm(max(0, p−q))、无损性证明、
 * 期望接受长度 (1−α^{γ+1})/(1−α) 与加速比。
 * 一手资料：Leviathan, Kalman, Matias. Fast Inference from Transformers via Speculative Decoding. ICML 2023（arXiv 2211.17192）
 *          Chen et al.（DeepMind）. Accelerating LLM Decoding with Speculative Sampling. arXiv 2302.01318
 */
import { defineLevel, md, py } from '@/content/helpers'
import { SpecDecodeStepViz } from '../viz/SpecDecodeStepViz'
import { AcceptRuleViz, ProofViz } from '../viz/SpecMathViz'
import { SpeedupViz } from '../viz/SpeedupViz'

export default defineLevel({
  /* ================================================================ */
  /* 图解                                                              */
  /* ================================================================ */
  learn: [
    {
      id: 'why',
      title: '为什么能加速：多验证几个 token 几乎不花时间',
      body: md`
        自回归解码每生成 1 个 token 都要跑一次完整前向。batch 不大时，这次前向的耗时主要花在**把全部权重从 HBM 读一遍**上
        （上一关 Roofline 的结论：decode 的算术强度约等于 batch 大小，远低于 GPU 的拐点）。

        关键观察：如果一次前向不是算 1 个位置，而是**并行算 $\gamma+1$ 个位置**，权重仍然只读一遍，FLOPs 变成 $\gamma+1$ 倍，
        但只要还在访存瓶颈区，耗时几乎不变。Chen 等人在论文里把原因拆成三项：小 batch 下线性层受带宽限制；注意力要读的 KV cache 大小不随 $\gamma$ 变化；
        张量并行的 all-reduce 只传少量 token 的激活，受延迟而非带宽限制。

        于是有了投机解码：

        1. 用一个**便宜的草稿模型** $M_q$ 自回归地猜 $\gamma$ 个 token；
        2. 用**目标模型** $M_p$ 做 **1 次**前向，并行算出这 $\gamma+1$ 个位置的分布；
        3. 用一种修正的拒绝采样从左到右决定接受几个，保证最终输出**与只用目标模型采样的分布完全相同**。

        | | 普通解码 | 投机解码（一轮） |
        |---|---|---|
        | 目标模型前向 | 每个 token 1 次 | 1 次 |
        | 草稿模型前向 | 0 | $\gamma$ 次 |
        | 产出 token | 1 | 1 ~ $\gamma+1$ 个 |
        | 输出分布 | $p$ | **仍是 $p$**（无损） |
      `,
      after: md`
        > [!NOTE] 两篇同期论文
        > - Leviathan、Kalman、Matias（Google），*Fast Inference from Transformers via Speculative Decoding*，ICML 2023（arXiv 2211.17192）：T5-XXL 上相对 T5X 实现加速 2–3×，输出完全相同。
        > - Chen 等（DeepMind），*Accelerating Large Language Model Decoding with Speculative Sampling*（arXiv 2302.01318）：Chinchilla 70B 配一个 4B、8 层的草稿模型（每 token 1.8 ms vs 14.1 ms），$K=4$，在分布式部署下加速 2–2.5×。
        >
        > 两篇的核心算法相同；Chen 等人称之为 speculative sampling。
      `,
      checkpoint: {
        id: 'cp-why',
        type: 'single',
        prompt: md`小 batch 下，目标模型一次前向「验证 $\gamma+1$ 个 token」与「生成 1 个 token」耗时接近，根本原因是：`,
        options: [
          '验证时不需要计算 softmax',
          'decode 受访存带宽限制，多算几个位置不增加权重读取量，增加的 FLOPs 被闲置的算力吸收',
          '验证阶段可以跳过注意力层',
          '草稿 token 的 KV cache 已经由草稿模型算好，目标模型直接复用',
        ],
        answer: 1,
        explanation: md`decode 时 GPU 大部分时间在等权重从 HBM 读进来；同一份权重服务 $\gamma+1$ 个位置，读取量不变，额外的 FLOPs 落在闲置的算力上。A、C 都是错的：验证需要完整前向；D 错在两个模型的 KV cache 维度、层数都不同，不能互用——目标模型必须自己为草稿 token 计算 KV。`,
      },
    },
    {
      id: 'flow',
      title: '一轮投机解码：起草 → 验证 → 接受 / 修正',
      body: md`
        设已提交的前缀为 $x_{<t}$，一轮的完整流程：

        1. **起草**：草稿模型依次采样 $\tilde x_1 \sim q_1,\ \tilde x_2 \sim q_2, \dots, \tilde x_\gamma \sim q_\gamma$，其中 $q_i = q(\cdot \mid x_{<t}, \tilde x_{<i})$；
        2. **验证**：目标模型输入 $x_{<t}$ 和全部草稿，一次前向得到 $p_1, \dots, p_{\gamma+1}$，其中 $p_i = p(\cdot \mid x_{<t}, \tilde x_{<i})$；
        3. **逐个检查**：$i = 1..\gamma$，以概率 $\min\!\left(1, \frac{p_i(\tilde x_i)}{q_i(\tilde x_i)}\right)$ 接受 $\tilde x_i$；
           第一次被拒绝时，从**残差分布** $\mathrm{norm}(\max(0, p_i - q_i))$ 采一个修正 token，**本轮结束**；
        4. **bonus**：若 $\gamma$ 个草稿全被接受，目标模型那次前向已经顺带算出 $p_{\gamma+1}$，再免费采一个 token。

        所以每轮至少产出 1 个 token（第一个就被拒时，修正 token 本身也是一个合法样本），最多 $\gamma+1$ 个。
        下面用一个 8 个词的 toy 模型完整播放 3 轮；调 $\gamma$ 和草稿质量，观察「每次目标前向产出的 token 数」怎么变。
      `,
      viz: SpecDecodeStepViz,
      after: md`
        > [!WARN] 被拒位置之后的草稿为什么全部作废
        > $\tilde x_{i+1}$ 是以 $\tilde x_i$ 为条件起草的，$\tilde x_i$ 被换掉之后，后面草稿的条件就错了，目标模型为它们算出的 $p_{i+1}, \dots$ 也一并失效。
        > 工程上要把目标模型（和草稿模型）为这些位置写入的 KV cache **回滚**：在分页 KV 里只需把序列长度改回去、释放多出来的块。

        > [!TIP] 修正 / bonus token 的 KV 还没算
        > 本轮最后那个 token 是从分布里采出来的，目标模型还没把它当输入跑过。下一轮的验证前向会把它和新草稿一起输入，所以不会多花一次前向。
      `,
      checkpoint: {
        id: 'cp-flow',
        type: 'numeric',
        prompt: md`$\gamma = 4$，草稿 $\tilde x_1, \tilde x_2$ 被接受，$\tilde x_3$ 被拒绝。本轮一共产出多少个 token？`,
        answer: 3,
        tolerance: 0,
        unit: '个',
        explanation: md`2 个被接受的草稿 + 1 个从残差分布采出的修正 token = 3 个。$\tilde x_4$ 以被拒的 $\tilde x_3$ 为条件，直接作废；也没有 bonus token（bonus 只在全部接受时才有）。`,
      },
    },
    {
      id: 'accept',
      title: '接受规则：min(1, p/q) 与单步接受率 β',
      body: md`
        对某个位置，设目标分布为 $p$、草稿分布为 $q$，草稿采到 $x \sim q$。接受概率：

        $$
        a(x) = \min\!\left(1, \frac{p(x)}{q(x)}\right)
        $$

        - $q(x) \le p(x)$：草稿**没有高估** $x$，提出就一定接受；
        - $q(x) > p(x)$：草稿**高估**了 $x$，只按比例 $p/q$ 保留，把多出来的质量砍掉。

        「提出 $x$ 且被接受」的概率是 $q(x) \cdot a(x) = \min(p(x), q(x))$。对 $x$ 求和得到**单步接受率**：

        $$
        \beta = \sum_x \min(p(x), q(x)) = 1 - \tfrac{1}{2}\sum_x |p(x) - q(x)| = 1 - \mathrm{TV}(p, q)
        $$

        $\beta$ 只取决于两个分布的**重叠面积**。Leviathan 等人把它在上下文上的期望记作 $\alpha = \mathbb{E}[\beta]$，作为衡量草稿质量的唯一指标。
        点击下面任意一行，看这个 token 的接受概率和被接受的质量。
      `,
      viz: AcceptRuleViz,
      after: md`
        > [!WARN] 两篇论文的记号正好相反
        > Leviathan 等：$p$ = 目标模型，$q$ = 草稿，接受条件 $\min(1, p/q)$。Chen 等的 Algorithm 2：$q$ = **目标**模型，$p$ = **草稿**，接受条件写作 $r < \min(1, q/p)$、残差写作 $(q - p)_+$。
        > 两者完全等价。面试时先声明记号：「$p$ 是 target，$q$ 是 draft」。
      `,
      checkpoint: {
        id: 'cp-accept',
        type: 'single',
        prompt: md`某位置 $p(x) = 0.2$，$q(x) = 0.5$，草稿恰好提出了 $x$。它被接受的概率是？`,
        options: ['0.2', '0.4', '0.5', '1.0'],
        answer: 1,
        explanation: md`$\min(1, 0.2/0.5) = 0.4$。「提出且被接受」的联合概率才是 $\min(p,q) = 0.2$——别把条件概率和联合概率混为一谈。`,
      },
    },
    {
      id: 'proof',
      title: '正确性证明：输出分布与目标模型完全一致',
      body: md`
        **单个位置**。记 $\beta = \sum_x \min(p(x), q(x))$，残差分布 $r(x) = \dfrac{\max(0, p(x) - q(x))}{\sum_{x'} \max(0, p(x') - q(x'))}$。

        先算两个常数：

        $$
        P(\text{拒绝}) = \sum_x q(x)\big(1 - a(x)\big) = \sum_x \big(q(x) - \min(p(x), q(x))\big) = 1 - \beta
        $$

        $$
        \sum_x \max(0, p(x) - q(x)) = \sum_x \big(p(x) - \min(p(x), q(x))\big) = 1 - \beta
        $$

        输出 $x'$ 有两条路径：草稿提出 $x'$ 并被接受；或者被拒绝后从 $r$ 中采到 $x'$。于是

        $$
        P(\text{输出}=x') = \underbrace{q(x')\,a(x')}_{\min(p(x'),\,q(x'))} + \underbrace{(1-\beta)\cdot \frac{\max(0,\,p(x')-q(x'))}{1-\beta}}_{\text{拒绝后重采样}} = \min(p,q) + \max(0, p-q) = p(x')
        $$

        最后一步用了恒等式 $\min(a,b) + \max(0, a-b) = a$。$\beta = 1$（$p = q$）时永远不会拒绝，直接成立。

        **多个位置**用归纳法：在前 $i-1$ 个草稿都被接受的条件下，$\tilde x_i \sim q(\cdot \mid x_{<t}, \tilde x_{<i})$，
        而目标模型算出的 $p_i$ 恰好是 $p(\cdot \mid x_{<t}, \tilde x_{<i})$，由单步引理，第 $i$ 个输出服从目标模型的条件分布；
        被拒绝时本轮结束；全部接受时 bonus 直接采自 $p_{\gamma+1}$。逐位置的条件分布都对，整条序列的联合分布就与目标模型自回归采样相同。
      `,
      viz: ProofViz,
      after: md`
        > [!INTERVIEW] 面试官想听到的三个细节
        > 1. 残差的归一化常数**恰好**等于拒绝概率 $1-\beta$，两者相消——这是证明能成立的关键，也是为什么「拒绝后从 $p$ 重采样」是错的（切换上图的「错误」选项看偏差）；
        > 2. 每个位置要用**独立**的均匀随机数；
        > 3. 温度、top-k、top-p 要**先**作用到 $p$ 和 $q$ 上（Chen 等论文中的做法），保证的是「与带这些采样参数的目标模型同分布」。
        >    贪心解码时 $p$、$q$ 退化为 one-hot：草稿等于目标模型的 argmax 才接受，被拒就换成 argmax，输出与目标模型贪心解码逐 token 相同（浮点误差除外）。
      `,
      checkpoint: {
        id: 'cp-proof',
        type: 'judge',
        prompt: md`为了简单，被拒绝后直接从目标分布 $p$ 重新采样一个 token，输出分布仍然等于 $p$。`,
        answer: false,
        explanation: md`错误。此时 $P(x) = \min(p,q) + (1-\beta)\,p(x)$。接受部分 $\min(p,q)$ 已经偏向「草稿也喜欢」的 token，再按 $p$ 补会重复计入它们，低估 $p > q$ 的 token。只有残差 $\max(0, p-q)$ 才恰好补齐缺口。`,
      },
    },
    {
      id: 'expect',
      title: '期望接受长度与加速比',
      body: md`
        假设各位置的接受事件独立、接受率都为 $\alpha$（Leviathan 等人的 i.i.d. 假设）。一轮至少产出 $k+1$ 个 token，当且仅当前 $k$ 个草稿都被接受，概率为 $\alpha^k$（$k = 0..\gamma$）。用「尾概率求和」：

        $$
        \mathbb{E}[\text{每轮产出}] = \sum_{k=0}^{\gamma} \alpha^k = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}
        $$

        设草稿一次前向的耗时是目标模型的 $c$ 倍，并假设验证 $\gamma+1$ 个 token 与生成 1 个 token 耗时相同，则一轮耗时 $(\gamma c + 1)\,T$，加速比

        $$
        \text{speedup}(\gamma) = \frac{1 - \alpha^{\gamma+1}}{(1 - \alpha)(\gamma c + 1)}
        $$

        - $\gamma \to \infty$ 时期望产出趋于 $\frac{1}{1-\alpha}$：$\alpha = 0.8$ 时每轮最多期望 5 个 token；
        - $\gamma = 1$ 时加速比为 $\frac{1+\alpha}{1+c}$，所以 **$\alpha > c$ 是能加速的充要条件**（论文推论 3.9）；
        - 最优 $\gamma$ 没有闭式解，在整数上枚举即可。

        | 草稿 → 目标（Leviathan 等表 3） | $\alpha$（贪心，t=0） | $\alpha$（采样，t=1） |
        |---|---|---|
        | T5-small → T5-XXL（英德翻译） | 0.75 | 0.62 |
        | T5-large → T5-XXL（英德翻译） | 0.82 | 0.71 |
        | bigram → T5-XXL（英德翻译） | 0.20 | 0.19 |
        | LaMDA 8B → LaMDA 137B | 0.75 | 0.74 |
      `,
      viz: SpeedupViz,
      after: md`
        > [!WARN] 公式是理想化的上界
        > 它忽略了两件事：① batch 变大后，验证 $\gamma+1$ 个 token 的耗时不再「和 1 个一样」——系统逐渐转向算力瓶颈，被拒绝的草稿就是白算的 FLOPs；
        > ② 真实的接受率依赖上下文（代码、复述类文本高，开放式闲聊低），而且同一轮中越靠后的位置条件越「偏」。EAGLE-2 的动态草稿树正是利用了「接受率依赖上下文」这一点（下一关）。
      `,
      checkpoint: {
        id: 'cp-expect',
        type: 'numeric',
        prompt: md`$\alpha = 0.9$、$\gamma = 3$ 时，每轮期望产出多少个 token？（保留 3 位小数）`,
        answer: 3.439,
        tolerance: 0.002,
        explanation: md`$\frac{1 - 0.9^4}{1 - 0.9} = \frac{1 - 0.6561}{0.1} = 3.439$，也就是 $1 + 0.9 + 0.81 + 0.729$。`,
      },
    },
    {
      id: 'practice',
      title: '工程落地：batch、回滚、草稿从哪来',
      body: md`
        **1. batch 越大，收益越小。** 投机解码是「用闲置算力换访存次数」。batch 大到接近 Roofline 拐点时算力不再闲置：
        验证要算 $B(\gamma+1)$ 个 token，被拒的那部分纯属浪费，加速比下降甚至小于 1。所以线上通常只在低并发 / 低延迟场景开大 $\gamma$，
        有的推理框架提供按负载自动降低 $\gamma$ 或关闭投机的选项。（反例也有：EAGLE-3 报告在 SGLang 中 batch = 64 时吞吐仍提升 1.38×，取决于草稿多便宜、接受率多高。）

        **2. KV cache 回滚。** 目标模型在验证时为全部 $\gamma$ 个草稿写了 KV；被拒位置之后的要丢掉。草稿模型同理。

        **3. 草稿必须和目标模型共用词表**，否则 $p/q$ 无从比较（跨词表需要额外的 token 对齐，代价很大）。

        **4. 草稿从哪来：**

        | 草稿来源 | 代表 | 特点 |
        |---|---|---|
        | 同族小模型 | Chen 等的 4B 草稿、Llama 系 1B/8B 草稿 | 最简单；要额外部署一个模型 |
        | 检索 / n-gram | prompt lookup：从上下文里找相同的 n-gram 直接抄 | 零训练；摘要、改代码、RAG 类任务接受率高 |
        | 目标模型上加头 | Medusa、EAGLE 系列 | 复用目标模型的隐状态，草稿极便宜，接受率高 |
        | 预训练自带的 MTP 模块 | DeepSeek-V3 | 第 2 个 token 接受率 85–90% |

        > [!NOTE] 无损 vs 有损
        > 上面的拒绝采样是**无损**的。Medusa 论文提出的 typical acceptance 用熵相关的阈值决定是否接受，**不保证**输出分布与目标模型一致，换来更长的接受长度。面试时注意区分。
      `,
      checkpoint: {
        id: 'cp-practice',
        type: 'multi',
        prompt: md`关于投机解码的工程实践，正确的有：（多选）`,
        options: [
          '草稿模型和目标模型必须共用同一个词表（或做额外的 token 对齐）',
          '被拒绝位置之后，目标模型为草稿写入的 KV cache 需要回滚',
          'batch 越大，投机解码的加速比一定越高',
          '贪心解码时，投机解码的输出与目标模型贪心解码逐 token 相同（不计浮点误差）',
        ],
        answer: [0, 1, 3],
        explanation: md`C 反了：batch 越大越接近算力瓶颈，验证的额外 FLOPs 不再免费，被拒草稿的计算是浪费，加速比下降。其余三项都是落地时必须处理的细节。`,
      },
    },
  ],

  /* ================================================================ */
  /* 随堂练习                                                          */
  /* ================================================================ */
  quiz: [
    {
      id: 'q-ratio',
      type: 'single',
      difficulty: 1,
      prompt: md`投机采样中（$p$ 为目标模型、$q$ 为草稿模型），草稿 token $x$ 的接受概率是：`,
      options: [md`$\min(1, q(x)/p(x))$`, md`$\min(1, p(x)/q(x))$`, md`$p(x)$`, md`$\min(p(x), q(x))$`],
      answer: 1,
      explanation: md`接受概率是 $\min(1, p/q)$：草稿高估（$q > p$）时按比例保留。A 是把比值写反（Chen 等论文里 $q$ 是目标模型，记号不同，别照抄）；D 是「提出且被接受」的联合概率，不是条件接受概率。`,
    },
    {
      id: 'q-judge-batch',
      type: 'judge',
      difficulty: 1,
      prompt: md`投机解码在高并发（大 batch）的在线服务中加速比通常比单请求时更高。`,
      answer: false,
      explanation: md`相反。投机解码利用的是小 batch 下闲置的算力；batch 增大后系统趋向算力瓶颈，验证 $\gamma+1$ 个 token 的 FLOPs 不再免费，被拒草稿的计算全部浪费，加速比下降，甚至可能 < 1。`,
    },
    {
      id: 'q-fill',
      type: 'fill',
      difficulty: 2,
      prompt: md`
        某位置 $p = [0.5,\ 0.3,\ 0.2]$，$q = [0.2,\ 0.6,\ 0.2]$。
        单步接受率 $\beta$ = ____(1)____；若草稿提出 token 1，它被接受的概率 = ____(2)____；若被拒，残差分布中 token 0 的概率 = ____(3)____。
      `,
      blanks: [
        { label: '(1)', answers: ['0.7'], numeric: true, tolerance: 0.001 },
        { label: '(2)', answers: ['0.5'], numeric: true, tolerance: 0.001 },
        { label: '(3)', answers: ['1'], numeric: true, tolerance: 0.001 },
      ],
      explanation: md`(1) $\beta = \min(0.5,0.2)+\min(0.3,0.6)+\min(0.2,0.2) = 0.2+0.3+0.2 = 0.7$。(2) $\min(1, 0.3/0.6) = 0.5$。(3) $\max(0, p-q) = [0.3, 0, 0]$，归一化后 token 0 的概率为 1——只有 $p > q$ 的 token 才会出现在残差里。`,
    },
    {
      id: 'q-expect',
      type: 'numeric',
      difficulty: 2,
      prompt: md`$\alpha = 0.8$、$\gamma = 4$，按 i.i.d. 假设，每轮期望产出多少个 token？（保留 4 位小数）`,
      answer: 3.3616,
      tolerance: 0.001,
      explanation: md`$\frac{1-0.8^5}{1-0.8} = \frac{1-0.32768}{0.2} = 3.3616$，即 $1 + 0.8 + 0.64 + 0.512 + 0.4096$。`,
    },
    {
      id: 'q-speedup',
      type: 'numeric',
      difficulty: 2,
      prompt: md`接上题（$\alpha = 0.8$，$\gamma = 4$），草稿模型单次前向耗时是目标模型的 $c = 0.05$ 倍，且验证 5 个 token 与生成 1 个 token 耗时相同。理论加速比是多少？（保留 2 位小数）`,
      answer: 2.8,
      tolerance: 0.005,
      unit: '×',
      explanation: md`一轮耗时 $(\gamma c + 1)T = 1.2T$，产出 3.3616 个 token；加速比 $= 3.3616 / 1.2 \approx 2.80$。常见错误是忘记扣掉草稿的 $\gamma c$ 成本，直接答 3.36。`,
    },
    {
      id: 'q-reject-prob',
      type: 'numeric',
      difficulty: 3,
      prompt: md`$p = [0.4,\ 0.3,\ 0.2,\ 0.1]$，$q = [0.1,\ 0.5,\ 0.1,\ 0.3]$。求「草稿提出 token 1 **并且**被拒绝」的概率。`,
      answer: 0.2,
      tolerance: 0.001,
      explanation: md`$q(1) \cdot (1 - \min(1, p(1)/q(1))) = 0.5 \times (1 - 0.6) = 0.2$。验算：总拒绝率 $1-\beta = 1 - (0.1+0.3+0.1+0.1) = 0.4$，其中 token 1 贡献 0.2、token 3 贡献 $0.3 - 0.1 = 0.2$；残差 $\max(0,p-q) = [0.3, 0, 0.1, 0]$ 之和也是 0.4，与拒绝率相等，证明里的归一化常数正是这样相消的。`,
    },
    {
      id: 'q-multi-lossless',
      type: 'multi',
      difficulty: 2,
      prompt: md`以下哪些做法能**严格保证**输出分布与目标模型（在相同采样参数下）一致？（多选）`,
      options: [
        md`按 $\min(1,p/q)$ 接受，拒绝后从 $\mathrm{norm}(\max(0,p-q))$ 重采样`,
        '贪心解码时：草稿等于目标模型 argmax 才接受，否则换成 argmax',
        'Medusa 的 typical acceptance（熵相关阈值）',
        md`按 $\min(1,p/q)$ 接受，拒绝后从 $p$ 重采样`,
        'EAGLE 草稿 + 标准的投机采样验证',
      ],
      answer: [0, 1, 4],
      explanation: md`A 是标准算法；B 是 A 在 one-hot 分布下的特例；E 只是换了草稿来源，验证规则不变，所以也无损。C 是 Medusa 论文明确说明「不追求与原分布严格一致」的有损方案；D 会重复计入草稿偏爱的 token（见第 4 节）。`,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 1,
      prompt: md`把一轮投机解码的步骤按正确顺序排列：`,
      items: [
        md`草稿模型自回归采样 $\gamma$ 个 token，并记录每步的 $q_i$`,
        md`目标模型一次前向，得到 $p_1, \dots, p_{\gamma+1}$`,
        md`$i$ 从 1 到 $\gamma$：以 $\min(1, p_i/q_i)$ 的概率接受 $\tilde x_i$`,
        md`第一次拒绝：从 $\mathrm{norm}(\max(0, p_i - q_i))$ 采修正 token 并结束本轮；若全部接受则从 $p_{\gamma+1}$ 采 bonus`,
        '回滚被拒位置之后的 KV cache，把新 token 追加到前缀，开始下一轮',
      ],
      explanation: md`起草 → 一次验证 → 从左到右检查 → 修正或 bonus → 回滚 KV 并进入下一轮。注意检查必须**从左到右**且在第一次拒绝时停止：后面的草稿以被拒 token 为条件。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把符号与含义配对：`,
      pairs: [
        [md`$\beta = \sum_x \min(p,q)$`, '某个位置的单步接受率'],
        [md`$c$`, '草稿与目标模型单次前向的耗时比'],
        [md`$\frac{1-\alpha^{\gamma+1}}{1-\alpha}$`, '每轮期望产出的 token 数'],
        [md`$\mathrm{norm}(\max(0, p-q))$`, '被拒绝后的重采样分布'],
        [md`$1 - \beta$`, '总拒绝概率，也等于 p、q 的总变差距离'],
      ],
      explanation: md`$\beta$ 是重叠面积；$1-\beta$ 既是拒绝率，也是 TV 距离，还是残差分布的归一化常数——三者相等是正确性证明的核心。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`
        6 个 token 上 $p = [0.30, 0.25, 0.20, 0.10, 0.10, 0.05]$，$q = [0.10, 0.35, 0.20, 0.25, 0.05, 0.05]$。
        被拒绝后，残差分布 $\mathrm{norm}(\max(0, p-q))$ 中概率**大于 0** 的有哪些 token？
      `,
      items: ['t0', 't1', 't2', 't3', 't4', 't5'],
      answer: [0, 4],
      explanation: md`只有 $p > q$ 的 token 有残差：t0（0.30 > 0.10）、t4（0.10 > 0.05）。t2、t5 的 $p = q$，残差为 0；t1、t3 被草稿高估，残差为 0。残差 = [0.20, 0, 0, 0, 0.05, 0]，归一化后为 [0.8, 0, 0, 0, 0.2, 0]。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 3,
      prompt: md`下面的验证函数有 **2 处** bug（$p$ 为目标模型、$q$ 为草稿模型）。点出有问题的行：`,
      code: py`
        import numpy as np

        def verify(p, q, drafts, rng):
            """p: (γ+1, V) 目标分布；q: (γ, V) 草稿分布；drafts: γ 个草稿 token"""
            out = []
            for i, x in enumerate(drafts):
                u = rng.random()
                if u < min(1.0, q[i, x] / p[i, x]):
                    out.append(x)
                else:
                    r = np.maximum(p[i] - q[i], 0.0)
                    r /= r.sum()
                    out.append(int(rng.choice(len(r), p=r)))
            if len(out) == len(drafts):
                out.append(int(rng.choice(p.shape[1], p=p[-1])))
            return out
      `,
      bugLines: [8, 13],
      explanation: md`
        第 8 行比值写反了，应为 \`p[i, x] / q[i, x]\`。第 13 行采到修正 token 后必须 **立即结束**（\`return out\` 或 \`break\` 并跳过 bonus）：
        现在循环会继续检查以被拒 token 为条件的后续草稿，而且 \`len(out) == len(drafts)\` 恒成立，又会错误地再追加一个 bonus。修正：

        \`\`\`python
                    out.append(int(rng.choice(len(r), p=r)))
                    return out
        \`\`\`
      `,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出投机采样的验证循环（\`sample(w, u)\` 按未归一化权重 \`w\` 做逆 CDF 采样）。有 2 行干扰项：`,
      lines: [
        'def verify(p, q, drafts, us, u_last):',
        '    out = []',
        '    for i, x in enumerate(drafts):',
        '        if us[i] < p[i][x] / q[i][x]:',
        '            out.append(x)',
        '            continue',
        '        out.append(sample(np.maximum(p[i] - q[i], 0), u_last))',
        '        return out',
        '    out.append(sample(p[len(drafts)], u_last))',
        '    return out',
      ],
      distractors: ['        out.append(sample(p[i], u_last))', '    out.append(sample(q[len(drafts) - 1], u_last))'],
      explanation: md`拒绝后从残差 \`max(p - q, 0)\` 采样并立即返回；全部接受后从目标模型的第 $\gamma+1$ 个分布 \`p[len(drafts)]\` 采 bonus。干扰项一个是「从 $p$ 重采样」（有偏），一个是从草稿分布采 bonus（bonus 必须来自目标模型）。`,
    },
    {
      id: 'q-alpha-c',
      type: 'single',
      difficulty: 3,
      prompt: md`Chen 等人的草稿模型每 token 1.8 ms、Chinchilla 70B 每 token 14.1 ms，即 $c \approx 0.13$。在理想化模型下，平均接受率 $\alpha$ 至少要满足什么条件，投机解码才**可能**比普通解码快？`,
      options: [md`$\alpha > 0.13$`, md`$\alpha > 0.5$`, md`$\alpha > 0.87$`, md`任何 $\alpha > 0$ 都能加速`],
      answer: 0,
      explanation: md`$\gamma = 1$ 时加速比为 $\frac{1+\alpha}{1+c}$，大于 1 当且仅当 $\alpha > c$；Leviathan 等人的推论 3.9 证明了 $\alpha > c$ 时一定存在使加速比 > 1 的 $\gamma$，而 $\alpha \le c$ 时任何 $\gamma$ 都不划算。C 是把条件误写成 $\alpha > 1 - c$。`,
    },
  ],

  /* ================================================================ */
  /* 跟练                                                              */
  /* ================================================================ */
  guided: [
    {
      id: 'g-verify-one',
      title: '单个位置的接受 / 拒绝与残差采样',
      intro: md`
        实现 \`verify_one(p, q, x, u, u2)\`：草稿在这个位置提出了 token \`x\`（\`p\`、\`q\` 为长度 $V$ 的 NumPy 概率向量）。
        - 若 \`u < min(1, p[x] / q[x])\`，接受，返回 \`(True, x)\`；
        - 否则从残差分布 $\mathrm{norm}(\max(0, p-q))$ 中用逆 CDF 采样（第一个前缀和 \`> u2\` 的下标），返回 \`(False, y)\`。
      `,
      template: py`
        import numpy as np

        def verify_one(p, q, x, u, u2):
            accept_prob = {{1}}
            if u < accept_prob:
                return True, x
            residual = {{2}}
            residual = {{3}}
            cdf = np.cumsum(residual)
            y = {{4}}
            return False, int(y)
      `,
      steps: [
        {
          blank: 1,
          instruction: md`写出接受概率 $\min(1, p(x)/q(x))$。`,
          answer: 'min(1.0, p[x] / q[x])',
          accept: ['min(1, p[x] / q[x])', 'min(1.0, p[x]/q[x])', 'min(p[x] / q[x], 1.0)'],
          hint: '比值是「目标 / 草稿」。',
          explain: md`$q(x) > 0$ 是有保证的：草稿就是从 $q$ 里采出 $x$ 的。$p(x) \ge q(x)$ 时比值 ≥ 1，必然接受。`,
        },
        {
          blank: 2,
          instruction: md`被拒绝：计算未归一化的残差 $\max(0, p - q)$（逐元素）。`,
          answer: 'np.maximum(p - q, 0.0)',
          accept: ['np.maximum(p - q, 0)', 'np.clip(p - q, 0, None)', 'np.maximum(0, p - q)'],
          hint: '用 np.maximum 做逐元素的 max，不要用 Python 内置 max。',
          explain: md`残差只保留「目标比草稿更想要」的 token。它的总和恰好等于拒绝概率 $1-\beta$。`,
        },
        {
          blank: 3,
          instruction: md`归一化。`,
          answer: 'residual / residual.sum()',
          accept: ['residual / np.sum(residual)'],
          explain: md`拒绝发生时一定有 $p(x) < q(x)$，于是 $\sum \max(0, p-q) = \sum \max(0, q-p) \ge q(x)-p(x) > 0$，不会除以 0。`,
        },
        {
          blank: 4,
          instruction: md`逆 CDF 采样：返回第一个满足 \`cdf[i] > u2\` 的下标。`,
          answer: 'np.searchsorted(cdf, u2, side="right")',
          accept: ["np.searchsorted(cdf, u2, side='right')", 'int((cdf <= u2).sum())', 'np.argmax(cdf > u2)'],
          hint: 'side="right" 的 searchsorted 返回第一个严格大于 u2 的位置。',
          explain: md`用「严格大于」可以保证概率为 0 的 token 永远采不到（它的前缀和与前一个相同）。`,
        },
      ],
      setup: py`
        import numpy as np
      `,
      tests: [
        {
          name: '必然接受（p ≥ q）',
          code: py`
            p = np.array([0.5, 0.3, 0.2]); q = np.array([0.2, 0.6, 0.2])
            got = verify_one(p, q, 0, 0.99, 0.5)
            assert got == (True, 0), f"p[0] > q[0] 应必然接受，得到 {got}"
          `,
        },
        {
          name: '按比例接受 / 拒绝',
          code: py`
            p = np.array([0.5, 0.3, 0.2]); q = np.array([0.2, 0.6, 0.2])
            got = verify_one(p, q, 1, 0.49, 0.5)
            assert got == (True, 1), f"接受概率 0.5，u=0.49 应接受，得到 {got}"
            got = verify_one(p, q, 1, 0.51, 0.7)
            assert got == (False, 0), f"u=0.51 应拒绝并从残差 [1,0,0] 采到 0，得到 {got}"
          `,
        },
        {
          name: '残差采样的边界',
          code: py`
            p = np.array([0.375, 0.25, 0.375, 0.0]); q = np.array([0.125, 0.5, 0.125, 0.25])
            # 残差 [0.25, 0, 0.25, 0] → 归一化 [0.5, 0, 0.5, 0]，cdf = [0.5, 0.5, 1, 1]
            got = verify_one(p, q, 1, 0.9, 0.49)
            assert got == (False, 0), f"期望 (False, 0)，得到 {got}"
            got = verify_one(p, q, 1, 0.9, 0.5)
            assert got == (False, 2), f"cdf[0] == u2 == 0.5 时应取第一个严格大于的下标 2，得到 {got}"
            got = verify_one(p, q, 3, 0.0, 0.7)
            assert got == (False, 2), f"p[3] = 0 时接受概率为 0，任何 u 都应拒绝；期望 (False, 2)，得到 {got}"
          `,
        },
        {
          name: '输出分布等于 p（8000 次模拟）',
          code: py`
            rng = np.random.default_rng(0)
            p = np.array([0.35, 0.25, 0.15, 0.12, 0.08, 0.05])
            q = np.array([0.10, 0.10, 0.40, 0.05, 0.30, 0.05])
            n = 8000
            xs = np.searchsorted(np.cumsum(q), rng.random(n) * np.cumsum(q)[-1], side="right")
            us, u2s = rng.random(n), rng.random(n)
            cnt = np.zeros(6)
            for k in range(n):
                cnt[verify_one(p, q, int(xs[k]), us[k], u2s[k])[1]] += 1
            freq = cnt / n
            assert np.abs(freq - p).max() < 0.03, f"经验分布 {np.round(freq, 3)} 偏离 p = {p}"
          `,
        },
      ],
      outro: md`
        这 4 行就是投机采样的数学核心。真实实现会把它向量化到 $\gamma$ 个位置、整个 batch：
        先对所有位置并行算出「是否接受」，再用 \`cumprod\` / \`argmin\` 找到每条序列第一个被拒的位置。实战题里你会写出完整的多位置版本。
      `,
    },
  ],

  /* ================================================================ */
  /* 实战                                                              */
  /* ================================================================ */
  challenges: [
    {
      id: 'c-verify',
      title: '实现投机采样的验证步骤',
      difficulty: 'hard',
      source: '大模型岗手撕 · 投机解码',
      timeLimitMin: 25,
      description: md`
        给定一轮投机解码的全部信息，实现验证步骤，返回本轮最终输出的 token 列表（被接受的草稿 + 1 个修正 / bonus token）。

        \`\`\`python
        def speculative_verify(p: np.ndarray, q: np.ndarray, draft: list[int],
                               u_accept: np.ndarray, u_final: float) -> list[int]: ...
        \`\`\`

        **输入**
        - \`p\`：形状 \`(γ+1, V)\`，目标模型在 $\gamma+1$ 个位置的概率分布（第 $i$ 行以前 $i$ 个草稿为条件）；
        - \`q\`：形状 \`(γ, V)\`，草稿模型在 $\gamma$ 个位置的分布；\`draft[i]\` 是从 \`q[i]\` 采出的 token，保证 \`q[i, draft[i]] > 0\`；
        - \`u_accept\`：长度 $\gamma$ 的均匀随机数；\`u_final\`：一个均匀随机数，$u \in [0, 1)$。$\gamma$ 可以为 0。

        **规则**（随机数用法必须严格一致，隐藏测试会逐 token 对拍）
        1. $i = 0, 1, \dots$：令 \`x = draft[i]\`，若 \`u_accept[i] < p[i, x] / q[i, x]\` 则接受，继续下一个；
        2. 第一次不满足时：残差 \`w = max(p[i] - q[i], 0)\`，用 \`u_final\` 从 \`w\` 中采样一个 token，返回 \`draft[:i] + [该 token]\`；
        3. 全部接受：用 \`u_final\` 从 \`p[γ]\` 采样 bonus token，返回 \`draft + [bonus]\`。

        **采样约定**（不需要先归一化）：对非负权重 \`w\`，令 \`c = cumsum(w)\`，返回第一个满足 \`c[j] > u * c[-1]\` 的下标 \`j\`。

        **示例**：\`p = [[.5,.3,.2],[.1,.6,.3],[.2,.2,.6]]\`，\`q = [[.2,.6,.2],[.1,.8,.1]]\`，\`draft = [0, 1]\`
        - \`u_accept = [0.9, 0.7]\`、\`u_final = 0.5\` → \`[0, 1, 2]\`（$0.5/0.2 \ge 1$ 必接受；$0.6/0.8 = 0.75 > 0.7$ 接受；bonus 从 \`p[2]\` 采到 2）
        - \`u_accept = [0.9, 0.8]\`、\`u_final = 0.5\` → \`[0, 2]\`（第 2 个被拒，残差 \`[0, 0, 0.2]\`）

        **隐藏测试**会用固定种子做几千次模拟，检验输出 token 的分布等于目标分布、平均接受数等于理论值。
      `,
      starter: py`
        import numpy as np

        def speculative_verify(p, q, draft, u_accept, u_final):
            # 1) 从左到右：u < p/q 则接受
            # 2) 第一次拒绝：从 max(p - q, 0) 采样修正 token 并返回
            # 3) 全部接受：从 p[γ] 采 bonus token
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def _sample(w, u):
            """逆 CDF：第一个 cumsum > u * total 的下标（w 非负、无需归一化）"""
            c = np.cumsum(w)
            return int(np.searchsorted(c, u * c[-1], side="right"))

        def speculative_verify(p, q, draft, u_accept, u_final):
            p = np.asarray(p, dtype=np.float64)
            q = np.asarray(q, dtype=np.float64)
            gamma = len(draft)
            out = []
            for i in range(gamma):
                x = int(draft[i])
                # 接受概率 min(1, p/q)；u < 1，所以写成 u < p/q 即可
                if u_accept[i] < p[i, x] / q[i, x]:
                    out.append(x)
                    continue
                # 第一次拒绝：从残差分布修正，后面的草稿以被拒 token 为条件，全部作废
                residual = np.maximum(p[i] - q[i], 0.0)
                out.append(_sample(residual, u_final))
                return out
            # 全部接受：目标模型的第 γ+1 个分布免费送一个 bonus token
            out.append(_sample(p[gamma], u_final))
            return out
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示，batch 向量化）',
          code: py`
            import torch

            def speculative_verify_batched(p, q, draft, u_accept, u_final):
                """p: (B, γ+1, V)  q: (B, γ, V)  draft: (B, γ)  u_accept: (B, γ)  u_final: (B,)
                返回 n_acc: (B,) 每条序列接受的草稿数，next_tok: (B,) 修正/bonus token"""
                B, G, V = q.shape
                px = p[:, :G].gather(-1, draft.unsqueeze(-1)).squeeze(-1)      # (B, γ)
                qx = q.gather(-1, draft.unsqueeze(-1)).squeeze(-1)
                accept = u_accept < px / qx                                      # (B, γ)
                # 第一个 False 的位置 = 接受数；cumprod 把第一次拒绝之后全部置 0
                n_acc = accept.int().cumprod(dim=-1).sum(dim=-1)                 # (B,)
                idx = n_acc.clamp(max=G - 1) if G > 0 else n_acc
                rows = torch.arange(B)
                residual = (p[rows, idx] - q[rows, idx]).clamp_min(0) if G > 0 else p[:, 0]
                w = torch.where((n_acc == G).unsqueeze(-1), p[rows, n_acc], residual)  # 全接受时用 p[γ]
                c = w.cumsum(-1)
                next_tok = torch.searchsorted(c, (u_final * c[:, -1]).unsqueeze(-1), right=True).squeeze(-1)
                return n_acc, next_tok
          `,
        },
      ],
      explanation: md`
        **思路**：严格按算法从左到右扫描。接受条件 \`u < p/q\` 与 \`u < min(1, p/q)\` 等价（$u < 1$）。第一次拒绝就从残差采样并**立即返回**；全部接受才有 bonus。

        **为什么不需要归一化残差**：逆 CDF 用 \`u * c[-1]\` 做阈值，等价于先归一化再与 \`u\` 比较，还顺带避免了归一化后前缀和不精确等于 1 的浮点问题。

        **常见错误**
        - 比值写成 \`q/p\`（照抄了 Chen 等论文却没注意记号相反）；
        - 拒绝后从 \`p[i]\` 重采样——单步分布会偏向草稿喜欢的 token，隐藏的分布检验会失败；
        - 拒绝后没有 \`return\`，继续检查后面的草稿或多加一个 bonus；
        - bonus 用了 \`p[γ-1]\` 或 \`q\` 的分布。

        **复杂度**：$O(\gamma V)$。真实系统在 GPU 上对整个 batch 向量化（见 PyTorch 版）：用 \`cumprod\` 找到每条序列第一个被拒的位置。
      `,
      complexity: { time: 'O(γ·V)', space: 'O(V)' },
      hints: [
        md`写一个辅助函数 \`_sample(w, u)\`：\`c = np.cumsum(w)\`，返回 \`np.searchsorted(c, u * c[-1], side="right")\`。`,
        md`循环 $i = 0..\gamma-1$：接受就 \`append\` 并 \`continue\`；拒绝就从 \`np.maximum(p[i] - q[i], 0)\` 采样、追加并 **return**。`,
        md`循环正常结束说明全部接受，从 \`p[len(draft)]\` 采 bonus。别忘了 $\gamma = 0$ 时直接从 \`p[0]\` 采样。`,
      ],
      setup: py`
        import numpy as np

        def _ref_sample(w, u):
            c = np.cumsum(w)
            return int(np.searchsorted(c, u * c[-1], side="right"))

        def _ref_verify(p, q, draft, u_accept, u_final):
            out = []
            for i, x in enumerate(draft):
                x = int(x)
                if u_accept[i] < p[i, x] / q[i, x]:
                    out.append(x)
                    continue
                out.append(_ref_sample(np.maximum(p[i] - q[i], 0.0), u_final))
                return out
            out.append(_ref_sample(p[len(draft)], u_final))
            return out

        def _as_list(got):
            assert isinstance(got, (list, tuple)), f"应返回 list，得到 {type(got).__name__}"
            return [int(t) for t in got]

        _P = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6]])
        _Q = np.array([[0.2, 0.6, 0.2], [0.1, 0.8, 0.1]])
      `,
      tests: [
        {
          name: '示例：全部接受 + bonus',
          code: py`
            got = _as_list(speculative_verify(_P, _Q, [0, 1], np.array([0.9, 0.7]), 0.5))
            assert got == [0, 1, 2], f"期望 [0, 1, 2]，得到 {got}"
          `,
        },
        {
          name: '示例：第 2 个被拒',
          code: py`
            got = _as_list(speculative_verify(_P, _Q, [0, 1], np.array([0.9, 0.8]), 0.5))
            assert got == [0, 2], f"期望 [0, 2]（残差 [0,0,0.2]），得到 {got}"
          `,
        },
        {
          name: '第 1 个就被拒',
          code: py`
            got = _as_list(speculative_verify(_P, _Q, [1, 1], np.array([0.6, 0.1]), 0.99))
            assert got == [0], f"0.3/0.6=0.5 ≤ 0.6 应拒绝，残差 [0.3,0,0] → [0]，得到 {got}"
          `,
        },
        {
          name: 'γ = 0：直接从 p[0] 采样',
          code: py`
            got = _as_list(speculative_verify(np.array([[0.1, 0.0, 0.9]]), np.zeros((0, 3)), [], np.zeros(0), 0.1))
            assert got == [2], f"cumsum=[0.1,0.1,1.0]，u=0.1 → 第一个 >0.1 的是 2，得到 {got}"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rng = np.random.default_rng(2026)
            for _ in range(300):
                V = int(rng.integers(2, 9)); g = int(rng.integers(0, 6))
                p = rng.dirichlet(np.ones(V) * 0.6, size=g + 1)
                q = rng.dirichlet(np.ones(V) * 0.6, size=g)
                draft = [int(np.searchsorted(np.cumsum(q[i]), rng.random() * q[i].sum(), side="right")) for i in range(g)]
                draft = [min(d, V - 1) for d in draft]
                draft = [d if q[i, d] > 0 else int(np.argmax(q[i])) for i, d in enumerate(draft)]
                ua = rng.random(g); uf = float(rng.random())
                got = _as_list(speculative_verify(p, q, draft, ua, uf))
                exp = _ref_verify(p, q, draft, ua, uf)
                assert got == exp, f"γ={g}, draft={draft}, u_accept={np.round(ua, 3)}, u_final={uf:.3f}: 期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '分布检验：输出服从目标分布',
          hidden: true,
          code: py`
            rng = np.random.default_rng(7)
            P = np.array([[0.35, 0.25, 0.15, 0.15, 0.10],
                          [0.10, 0.40, 0.20, 0.20, 0.10],
                          [0.30, 0.10, 0.10, 0.20, 0.30],
                          [0.20, 0.20, 0.20, 0.20, 0.20]])
            Q = np.array([[0.10, 0.10, 0.50, 0.20, 0.10],
                          [0.30, 0.30, 0.10, 0.20, 0.10],
                          [0.25, 0.25, 0.25, 0.15, 0.10]])
            n, g = 4000, 3
            cq = np.cumsum(Q, axis=1)
            U = rng.random((n, g))
            D = np.stack([(U[:, i, None] * cq[i, -1] >= cq[i][None, :]).sum(1) for i in range(g)], axis=1)
            UA = rng.random((n, g)); UF = rng.random(n)
            first = np.zeros(5); second = np.zeros(5); n2 = 0; acc_total = 0
            for k in range(n):
                out = _as_list(speculative_verify(P, Q, [int(d) for d in D[k]], UA[k], float(UF[k])))
                assert 1 <= len(out) <= g + 1, f"输出长度应在 [1, {g + 1}]，得到 {len(out)}"
                first[out[0]] += 1
                if len(out) >= 2:
                    second[out[1]] += 1; n2 += 1
                acc_total += len(out) - 1
            f1 = first / n
            assert np.abs(f1 - P[0]).max() < 0.035, f"第 1 个输出的经验分布 {np.round(f1, 3)} 应接近 p[0] = {P[0]}"
            f2 = second / max(n2, 1)
            assert np.abs(f2 - P[1]).max() < 0.045, f"第 2 个输出的经验分布 {np.round(f2, 3)} 应接近 p[1] = {P[1]}"
            b = np.minimum(P[:g], Q).sum(1)
            exp_acc = b[0] + b[0] * b[1] + b[0] * b[1] * b[2]
            got_acc = acc_total / n
            assert abs(got_acc - exp_acc) < 0.08, f"平均接受数 {got_acc:.3f}，理论值 {exp_acc:.3f}"
          `,
        },
        {
          name: '零概率 token 永不输出',
          hidden: true,
          code: py`
            rng = np.random.default_rng(11)
            P = np.array([[0.0, 0.5, 0.5, 0.0], [0.25, 0.25, 0.0, 0.5]])
            Q = np.array([[0.4, 0.2, 0.2, 0.2]])
            for _ in range(500):
                d = int(np.searchsorted(np.cumsum(Q[0]), rng.random(), side="right"))
                out = _as_list(speculative_verify(P, Q, [d], rng.random(1), float(rng.random())))
                assert out[-1] in ((1, 2) if len(out) == 1 else (0, 1, 3)), f"采到了零概率 token：{out}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-batch',
          q: md`batch 里每条序列接受的草稿数不同，下一轮怎么对齐？`,
          a: md`不对齐。现代框架把 batch 展平成一维 token 序列（不 padding），每条序列维护自己的长度和块表；下一轮各自追加不同数量的 token。被拒位置的 KV 通过把序列长度改回去来「回滚」，分页 KV 下只需释放多出来的整块。若用静态形状的 CUDA Graph，则按最大 $\gamma+1$ 预留并用掩码处理。`,
        },
        {
          id: 'f-temp',
          q: md`采样时用了 temperature = 0.7 和 top-p = 0.9，投机采样怎么做？`,
          a: md`对目标与草稿的 logits **分别**施加相同的 temperature / top-k / top-p 处理，得到 $p'$ 和 $q'$，再对 $p'$、$q'$ 做标准验证。保证的是「输出等于带这些采样参数的目标模型」。草稿可以用与目标不同的温度（甚至更尖锐），只影响接受率，不影响正确性——前提是验证时用的 $q$ 就是草稿真实采样所用的分布。`,
        },
      ],
    },
    {
      id: 'c-plan',
      title: '由分布估计接受率并选择最优 γ',
      difficulty: 'medium',
      source: 'Leviathan et al. 2023 · 定理 3.8',
      timeLimitMin: 15,
      description: md`
        离线评估草稿模型：在 $N$ 个上下文上分别拿到目标分布 \`P[n]\` 和草稿分布 \`Q[n]\`（形状都是 \`(N, V)\`），草稿与目标单次前向耗时比为 \`c\`。实现

        \`\`\`python
        def spec_decode_plan(P: np.ndarray, Q: np.ndarray, c: float, max_gamma: int = 10) -> dict: ...
        \`\`\`

        返回字典 \`{"alpha", "best_gamma", "speedup", "expected_tokens"}\`：
        1. \`alpha\` = 各上下文单步接受率 $\beta_n = \sum_x \min(P_{n,x}, Q_{n,x})$ 的平均；
        2. 对 $\gamma = 1..$\`max_gamma\` 计算 $E(\gamma) = \frac{1-\alpha^{\gamma+1}}{1-\alpha}$ 与 $\text{speedup}(\gamma) = \frac{E(\gamma)}{\gamma c + 1}$；
           当 $1 - \alpha < 10^{-9}$ 时按 $\alpha = 1$ 处理：$E(\gamma) = \gamma + 1$；
        3. 选加速比最大的 $\gamma$（并列取较小者）；若所有 $\gamma$ 的加速比都 $\le 1$，返回 \`best_gamma = 0\`、\`speedup = 1.0\`、\`expected_tokens = 1.0\`（不开投机解码）。

        **示例**：\`P = [[.5,.5],[.9,.1]]\`，\`Q = [[.3,.7],[.6,.4]]\`，\`c = 0.1\` → \`alpha = 0.75\`，\`best_gamma = 5\`，\`speedup ≈ 2.1921\`，\`expected_tokens ≈ 3.2881\`。
      `,
      starter: py`
        import numpy as np

        def spec_decode_plan(P, Q, c, max_gamma=10):
            # alpha = mean(sum(min(P, Q), axis=-1))，再枚举 γ
            return {"alpha": 0.0, "best_gamma": 0, "speedup": 1.0, "expected_tokens": 1.0}
      `,
      solution: py`
        import numpy as np

        def spec_decode_plan(P, Q, c, max_gamma=10):
            P = np.asarray(P, dtype=np.float64)
            Q = np.asarray(Q, dtype=np.float64)
            alpha = float(np.minimum(P, Q).sum(axis=-1).mean())   # 每个上下文的重叠面积，再取平均

            def expected(g):
                if 1.0 - alpha < 1e-9:               # α→1 时公式 0/0，直接用极限 γ+1
                    return float(g + 1)
                return (1.0 - alpha ** (g + 1)) / (1.0 - alpha)

            best_g, best_s = 0, 1.0                  # γ=0 即不开投机解码
            for g in range(1, max_gamma + 1):
                s = expected(g) / (g * c + 1.0)
                if s > best_s + 1e-12:               # 严格更优才替换 → 并列取较小 γ
                    best_g, best_s = g, s
            return {"alpha": alpha, "best_gamma": best_g, "speedup": best_s,
                    "expected_tokens": expected(best_g) if best_g > 0 else 1.0}
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示）',
          code: py`
            import torch

            def spec_decode_plan(P: torch.Tensor, Q: torch.Tensor, c: float, max_gamma: int = 10):
                alpha = torch.minimum(P, Q).sum(-1).mean().item()
                g = torch.arange(1, max_gamma + 1, dtype=torch.float64)
                E = g + 1 if 1 - alpha < 1e-9 else (1 - alpha ** (g + 1)) / (1 - alpha)
                s = E / (g * c + 1)
                k = int(torch.argmax(s))              # argmax 返回第一个最大值 → 并列取较小 γ
                if s[k] <= 1:
                    return {"alpha": alpha, "best_gamma": 0, "speedup": 1.0, "expected_tokens": 1.0}
                return {"alpha": alpha, "best_gamma": k + 1, "speedup": s[k].item(), "expected_tokens": E[k].item()}
          `,
        },
      ],
      explanation: md`
        **接受率**：单个上下文的接受率是两分布的重叠面积 $\sum_x \min(p, q) = 1 - \mathrm{TV}(p, q)$；Leviathan 等人取它在上下文上的期望作为 $\alpha$。

        **最优 γ**：加速比 $\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}$ 关于 $\gamma$ 先增后减，但没有闭式最优解，直接枚举。$\alpha \le c$ 时 $\gamma=1$ 的加速比 $\frac{1+\alpha}{1+c} \le 1$，更大的 $\gamma$ 只会更差，所以不开投机解码。

        **数值细节**：$\alpha \to 1$ 时公式是 0/0，$1-\alpha^{\gamma+1}$ 与 $1-\alpha$ 都会被浮点舍入严重放大误差，必须单独处理。

        **局限**：真实接受率随上下文变化、同一轮里越往后越低；验证成本也随 batch 增长。工程上更可靠的做法是在目标负载上直接扫 $\gamma$ 测吞吐与延迟。
      `,
      complexity: { time: 'O(N·V + max_gamma)', space: 'O(N·V)' },
      hints: [
        md`\`alpha = np.minimum(P, Q).sum(axis=-1).mean()\`。`,
        md`写一个 \`expected(g)\`，特判 \`1 - alpha < 1e-9\` 时返回 \`g + 1\`。`,
        md`初始化 \`best_g, best_s = 0, 1.0\`，只有严格更大时才更新，自然得到「并列取较小」和「都不划算时返回 0」。`,
      ],
      setup: py`
        import numpy as np, math

        def _ref_plan(P, Q, c, max_gamma=10):
            P = np.asarray(P, float); Q = np.asarray(Q, float)
            a = float(np.minimum(P, Q).sum(-1).mean())
            E = lambda g: float(g + 1) if 1 - a < 1e-9 else (1 - a ** (g + 1)) / (1 - a)
            bg, bs = 0, 1.0
            for g in range(1, max_gamma + 1):
                s = E(g) / (g * c + 1)
                if s > bs + 1e-12:
                    bg, bs = g, s
            return {"alpha": a, "best_gamma": bg, "speedup": bs, "expected_tokens": E(bg) if bg else 1.0}

        def _check(got, exp, tag):
            assert isinstance(got, dict), f"{tag}: 应返回 dict，得到 {type(got).__name__}"
            for k in ("alpha", "best_gamma", "speedup", "expected_tokens"):
                assert k in got, f"{tag}: 缺少键 {k}"
            assert got["best_gamma"] == exp["best_gamma"], f"{tag}: best_gamma 期望 {exp['best_gamma']}，得到 {got['best_gamma']}"
            for k in ("alpha", "speedup", "expected_tokens"):
                assert math.isclose(got[k], exp[k], rel_tol=1e-6, abs_tol=1e-9), f"{tag}: {k} 期望 {exp[k]:.6f}，得到 {got[k]}"
      `,
      tests: [
        {
          name: '示例',
          code: py`
            P = np.array([[0.5, 0.5], [0.9, 0.1]]); Q = np.array([[0.3, 0.7], [0.6, 0.4]])
            got = spec_decode_plan(P, Q, 0.1)
            assert math.isclose(got["alpha"], 0.75), f"alpha 期望 0.75，得到 {got['alpha']}"
            assert got["best_gamma"] == 5, f"best_gamma 期望 5，得到 {got['best_gamma']}"
            assert math.isclose(got["speedup"], 3.2880859375 / 1.5, rel_tol=1e-6), f"speedup 期望 2.19206，得到 {got['speedup']}"
          `,
        },
        {
          name: 'α ≤ c：不开投机解码',
          code: py`
            got = spec_decode_plan(np.array([[1.0, 0.0]]), np.array([[0.1, 0.9]]), 0.2)
            _check(got, {"alpha": 0.1, "best_gamma": 0, "speedup": 1.0, "expected_tokens": 1.0}, "α=0.1, c=0.2")
          `,
        },
        {
          name: 'P == Q：α = 1 的极限',
          code: py`
            P = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]])
            got = spec_decode_plan(P, P.copy(), 0.05, max_gamma=8)
            _check(got, {"alpha": 1.0, "best_gamma": 8, "speedup": 9 / 1.4, "expected_tokens": 9.0}, "P == Q")
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rng = np.random.default_rng(42)
            for t in range(200):
                N, V = int(rng.integers(1, 6)), int(rng.integers(2, 12))
                P = rng.dirichlet(np.ones(V) * rng.uniform(0.2, 3), size=N)
                lam = rng.uniform(0, 1)
                Q = (1 - lam) * P + lam * rng.dirichlet(np.ones(V), size=N)
                c = float(rng.uniform(0, 0.8)); mg = int(rng.integers(1, 15))
                _check(spec_decode_plan(P, Q, c, mg), _ref_plan(P, Q, c, mg), f"case {t}")
          `,
        },
        {
          name: '加速比恒为 1 时不开投机解码',
          hidden: true,
          code: py`
            P = np.array([[0.5, 0.5]]); Q = np.array([[0.5, 0.5]])
            got = spec_decode_plan(P, Q, 1.0, max_gamma=6)
            # α=1、c=1：speedup(γ) = (γ+1)/(γ+1) = 1，全部不大于 1 → 不开
            _check(got, {"alpha": 1.0, "best_gamma": 0, "speedup": 1.0, "expected_tokens": 1.0}, "α=1, c=1")
          `,
        },
      ],
      followUps: [
        {
          id: 'f-position',
          q: md`实测发现第 1 个草稿位置接受率 0.85，第 3 个只有 0.6，i.i.d. 公式还准吗？`,
          a: md`会高估。应按位置分别估计 $\alpha_1, \alpha_2, \dots$，期望产出为 $1 + \alpha_1 + \alpha_1\alpha_2 + \dots$。接受率随位置下降很常见：越靠后，草稿的条件里累积的「猜测」越多（MTP 递归起草、Medusa 的远端头都有这个现象）。这也是 EAGLE-2 用动态树、按累计置信度决定往哪里展开的动机。`,
        },
      ],
    },
  ],

  /* ================================================================ */
  /* 面试追问                                                           */
  /* ================================================================ */
  interview: [
    {
      id: 'qa-lossless',
      freq: 3,
      q: md`投机解码为什么是无损的？请给出证明。`,
      a: md`
        **结论**：按 $\min(1, p/q)$ 接受、被拒后从 $\mathrm{norm}(\max(0, p-q))$ 重采样，每个位置的输出恰好服从目标分布 $p$。

        **证明**：输出 $x$ 的概率 = 草稿提出并接受 $q(x)\min(1, p/q) = \min(p, q)$ + 拒绝概率 × 残差概率。拒绝概率 $\sum_x (q - \min(p,q)) = 1-\beta$，
        残差的未归一化总量 $\sum_x (p - \min(p,q))$ 也是 $1-\beta$，相乘后约掉，剩下 $\max(0, p-q)$。$\min(p,q) + \max(0,p-q) = p$。
        多位置时，每个 $p_i$ 都以已接受的前缀为条件，归纳即可；拒绝后本轮结束，全部接受时 bonus 直接采自 $p_{\gamma+1}$。

        **延伸**：贪心解码是 one-hot 特例；temperature / top-p 要先作用到 $p$、$q$ 上；「拒绝后从 $p$ 重采样」是常见的错误答案。
      `,
    },
    {
      id: 'qa-speedup',
      freq: 3,
      q: md`投机解码的加速比由什么决定？γ 怎么选？`,
      a: md`
        **结论**：由接受率 $\alpha$、草稿成本比 $c$ 和草稿长度 $\gamma$ 决定：$\text{speedup} = \frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}$。

        **原因**：每轮期望产出 $\sum_{k=0}^{\gamma}\alpha^k$ 个 token，代价是 1 次目标前向 + $\gamma$ 次草稿前向。$\gamma$ 增大时第 $\gamma$ 个草稿被用上的概率 $\alpha^\gamma$ 指数下降，成本 $\gamma c$ 线性上升，所以存在最优 $\gamma$，要枚举；$\alpha > c$ 才可能加速。

        **延伸**：公式假设验证成本与 1 个 token 相同、接受率与位置无关，都只在小 batch 下近似成立。线上通常按负载动态调整 $\gamma$；EAGLE-2 进一步按置信度动态决定草稿树的形状。
      `,
    },
    {
      id: 'qa-batch',
      freq: 3,
      q: md`为什么投机解码在大 batch 下收益变小？线上服务该怎么用？`,
      a: md`
        **结论**：投机解码是用闲置算力换更少的串行前向；大 batch 下算力不再闲置。

        **原因**：decode 的算术强度约等于 batch 大小（权重部分）。batch 小时，验证 $\gamma+1$ 个 token 的额外 FLOPs 被闲置的 Tensor Core 吸收；batch 接近 Roofline 拐点后，每轮要算 $B(\gamma+1)$ 个 token，被拒草稿的计算是纯浪费，单步耗时随 $\gamma$ 上升，加速比下降，甚至 < 1。

        **延伸**：① 低并发 / 延迟敏感场景开大 $\gamma$，高并发时减小或关闭；② 用更便宜、接受率更高的草稿（EAGLE、MTP 草稿的成本远低于独立小模型）；③ 在 MoE 模型上，验证多个 token 还可能激活更多专家，额外增加访存。
      `,
    },
    {
      id: 'qa-draft',
      freq: 2,
      q: md`草稿模型有哪些选择？各自的优缺点？`,
      a: md`
        **结论**：独立小模型、n-gram / 检索、目标模型上加的草稿头（Medusa / EAGLE）、预训练自带的 MTP 模块。

        **对比**：独立小模型最通用，但要额外部署、占显存，且必须同词表；prompt lookup / n-gram 零成本，适合输出大量复用输入的任务（摘要、改代码、RAG）；Medusa / EAGLE 复用目标模型的隐状态，草稿只有一两层，接受率高（EAGLE-3 报告最高约 6.5× 加速）；DeepSeek-V3 的 MTP 模块随预训练一起得到，第 2 个 token 接受率 85–90%，解码 TPS 提升到 1.8×。

        **延伸**：接受率依赖任务分布，评估草稿要在真实流量上测 $\alpha$ 与端到端吞吐，而不只看离线困惑度。
      `,
    },
  ],

  cheatsheet: md`
    **记号**：$p$ = 目标（target），$q$ = 草稿（draft）。⚠️ Chen 等论文中记号相反。

    | 量 | 公式 |
    |---|---|
    | 接受概率 | $\min(1, p(x)/q(x))$ |
    | 拒绝后重采样 | $\mathrm{norm}(\max(0, p - q))$ |
    | 单步接受率 | $\beta = \sum_x \min(p,q) = 1 - \mathrm{TV}(p,q)$，$\alpha = \mathbb{E}[\beta]$ |
    | 每轮期望产出 | $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$（$\gamma\to\infty$ 时 $\frac{1}{1-\alpha}$） |
    | 加速比 | $\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}$，需 $\alpha > c$ |

    \`\`\`python
    for i, x in enumerate(draft):
        if u[i] < p[i][x] / q[i][x]:          # 接受
            continue
        return draft[:i] + [sample(np.maximum(p[i] - q[i], 0))]
    return draft + [sample(p[len(draft)])]   # bonus
    \`\`\`

    - 证明关键：拒绝概率 $1-\beta$ = 残差归一化常数，二者相消
    - 每轮产出 1 ~ $\gamma+1$ 个；被拒位置后的草稿作废、KV 回滚
    - 贪心：草稿 == argmax 才接受；temperature / top-p 先作用到 $p$、$q$
    - 大 batch 收益下降（算力不再闲置）
    - 数字：Leviathan T5-XXL 2–3×；Chen Chinchilla 70B + 4B 草稿 2–2.5×；T5-small→XXL $\alpha$ = 0.75（t=0）
  `,
})
