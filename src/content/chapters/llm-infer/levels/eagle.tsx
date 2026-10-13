/**
 * L10-5 Medusa、EAGLE-3 与 MTP：草稿从哪来、树注意力、动态草稿树、多层特征融合与 training-time test、
 * DeepSeek-V3 MTP 作草稿、P-EAGLE 并行草稿。
 * 一手资料：Medusa（arXiv 2401.10774）；EAGLE（2401.15077，ICML 2024）；EAGLE-2（2406.16858，EMNLP 2024）；
 *          EAGLE-3（2503.01840）；DeepSeek-V3 技术报告（2412.19437）；P-EAGLE（arXiv 2602.01469；vLLM 博客 2026-03-13）
 */
import { defineLevel, md, py } from '@/content/helpers'
import { TreeMaskViz } from '../viz/TreeMaskViz'
import { DraftArchViz } from '../viz/DraftArchViz'
import { DynamicTreeViz } from '../viz/DynamicTreeViz'

export default defineLevel({
  learn: [
    {
      id: 'medusa',
      title: 'Medusa：在目标模型上长出多个预测头',
      body: md`
        上一关的投机解码需要一个独立的小草稿模型：要额外部署、占显存、必须同词表，而且小模型和大模型「想法」差得远，接受率有限。
        **Medusa**（Cai 等，arXiv 2401.10774）换了个思路：直接在目标模型最后一层隐状态 $h_t$ 上加 $K$ 个额外的**解码头**。

        - 原有 LM head 预测 $x_{t+1}$；第 $k$ 个 Medusa 头预测 $x_{t+k+1}$；
        - 每个头是**一层带残差的 FFN**，参数很少，草稿几乎零成本，也不需要另一个模型的 KV cache；
        - 每个头取 top-$s_k$ 个候选，各头候选做**笛卡尔积**组成一棵候选树（例如 top-2 × top-2 × top-2，共 $2 + 4 + 8 = 14$ 个节点），用**树注意力**一次前向全部验证；
        - **Medusa-1**：冻结主干只训头，报告加速 2.2× 以上；**Medusa-2**：主干与头联合微调（需要特殊训练配方保住主干能力），加速 2.3–2.8×；
        - 没有训练数据时可以用模型自己生成数据做**自蒸馏**。

        **弱点**：所有头都只看同一个 $h_t$，第 2 个头并不知道第 1 个头猜了什么，只能「跳着」预测远处的 token，越远命中率越低。
      `,
      after: md`
        > [!WARN] Medusa 的 typical acceptance 是有损的
        > Medusa 论文除了标准的拒绝采样外，还提出 **typical acceptance**：候选 token 的概率高于一个与熵相关的阈值就接受。它换来更长的接受长度，
        > 但论文明确说明不追求与原模型分布严格一致。面试中说「投机解码无损」时要排除这种变体。
      `,
      checkpoint: {
        id: 'cp-medusa',
        type: 'numeric',
        prompt: md`3 个 Medusa 头分别取 top-3、top-2、top-2，做笛卡尔积组成候选树。树上一共有多少个节点（不含根）？`,
        answer: 21,
        tolerance: 0,
        unit: '个',
        explanation: md`第 1 层 3 个，第 2 层 $3 \times 2 = 6$ 个，第 3 层 $3 \times 2 \times 2 = 12$ 个，共 21 个；完整路径（叶子）12 条。实际部署会剪掉低概率的分支，只保留经验上命中率高的子树。`,
      },
    },
    {
      id: 'treeattn',
      title: '树注意力：一次前向验证一整棵树',
      body: md`
        候选不再是一条链，而是一棵树。把树上 $n$ 个节点按 BFS 顺序展平，和前缀一起送进目标模型，只需改两样东西：

        1. **注意力掩码**：节点 $i$ 能看到全部已提交前缀，以及树中自己的**祖先链和自己**；看不到兄弟和其它分支。形状为 $n \times (P + n)$。
        2. **位置编号**：节点 $i$ 的 position id = 前缀长度 $P$ + 深度（根的孩子深度为 0）。同一深度的不同分支共享相同的位置，RoPE 才会把它们当作「同一个位置的不同候选」。

        前向结束后，每个节点都拿到了「接在它后面」的目标分布。**贪心验证**：从根开始，看目标模型在当前节点的 argmax 是否等于某个孩子的 token，相等就走下去，直到走不动；走到的路径全部接受，再加一个 bonus（最后接受节点处的 argmax）。
        采样场景则要用基于树的多候选拒绝采样（每个节点的多个孩子依次尝试，残差分布逐次更新），细节较多，这里不展开。

        验证完成后，被接受路径上的 KV 要「搬」成连续的序列（或修改块表），其余分支的 KV 丢弃。
      `,
      viz: TreeMaskViz,
      after: md`
        > [!TIP] 为什么不把每条路径各跑一次
        > 14 个节点的树有 8 条完整路径，逐条验证要 8 次前向（或一个 batch = 8、有大量重复前缀的前向）。树注意力把公共前缀合并，只算 14 个位置，一次前向搞定。
        > 代价是验证的 token 数从 $\gamma$ 变成树的节点数——batch 大时这部分 FLOPs 不再免费，所以树的大小要随负载调整。
      `,
      checkpoint: {
        id: 'cp-tree',
        type: 'single',
        prompt: md`前缀长度 100，草稿树中节点 7 的祖先链为 节点 1 → 节点 4 → 节点 7（节点 1 是根的孩子）。节点 7 的 position id 是：`,
        options: ['107', '102', '103', '100'],
        answer: 1,
        explanation: md`深度从 0 计：节点 1 深度 0、节点 4 深度 1、节点 7 深度 2，position id = 100 + 2 = 102。107 是错把展平下标当作位置；用错位置编号会让 RoPE 把它当成更远的 token，验证出的分布就不对了。`,
      },
    },
    {
      id: 'eagle',
      title: 'EAGLE：在特征层面自回归',
      body: md`
        **EAGLE**（Li 等，ICML 2024，arXiv 2401.15077）的两个观察：

        1. **在特征层面自回归比在 token 层面容易**：草稿去预测目标模型倒数第二层（LM head 之前）的特征 $f$，再复用目标模型的 LM head 得到 token；
        2. **特征有不确定性**：同一个 $f_t$ 之后可能采到不同的 token，下一步的特征取决于采到了哪个。所以输入里要加上**提前一步的 token**：用 $(f_{1:t},\ x_{2:t+1})$ 预测 $f_{t+1}$。

        草稿模块很小：嵌入层与 LM head 复用目标模型，中间是一个 **FC（把拼接的 $2d$ 维压回 $d$ 维）+ 1 层 decoder**。训练损失 $L = L_{reg} + w_{cls} L_{cls}$：特征用 Smooth L1 回归，token 用交叉熵，$w_{cls} = 0.1$。
        训练数据是固定的 ShareGPT 约 6.8 万段对话，70B 模型在 4 张 A100（40G）上训 1–2 天。论文报告 LLaMA2-Chat 70B 上延迟加速 2.7–3.5×、吞吐翻倍；
        相对 Medusa 快 1.5–1.6×（MT-bench）。EAGLE-1 使用一棵**静态**的草稿树（按经验设计、所有输入共用）。

        切换下图的四种方法，对比数据是怎么流动的：
      `,
      viz: DraftArchViz,
      after: md`
        > [!NOTE] EAGLE 和 Medusa 的本质区别
        > Medusa 的各个头**并行、互不依赖**；EAGLE 的草稿**串行、自回归**，每一步都知道前一步猜的 token，所以远处的预测准得多。代价是起草 $\gamma$ 个 token 要 $\gamma$ 次草稿前向（但每次只有 1 层，非常便宜）——P-EAGLE 后来又试图把这部分并行化。
      `,
      checkpoint: {
        id: 'cp-eagle',
        type: 'single',
        prompt: md`EAGLE 的草稿输入里为什么要加入「提前一步的 token」$x_{t+1}$ 的嵌入？`,
        options: [
          '为了让草稿模型能复用目标模型的 KV cache',
          '同一个特征之后可能采到不同的 token，告诉草稿真正采到了哪个，才能确定下一个特征该往哪走',
          '为了替代 LM head',
          '因为特征维度太小，需要嵌入补足',
        ],
        answer: 1,
        explanation: md`这是 EAGLE 论文标题里的 feature uncertainty：特征 → token 是一个采样过程，只给特征序列，草稿无法知道采样走了哪个分支。加上已经确定的下一个 token，不确定性就消除了。`,
      },
    },
    {
      id: 'eagle2',
      title: 'EAGLE-2：按置信度长出动态树',
      body: md`
        EAGLE-1 的静态树对所有输入用同一个形状。**EAGLE-2**（arXiv 2406.16858，EMNLP 2024）发现：

        - 草稿 token 的接受率**依赖上下文**，不只依赖它在树中的位置；
        - EAGLE 的草稿模型**校准得很好**：它给出的置信度（概率）与真实接受率误差很小。

        于是用置信度来决定树往哪里长。定义节点的**全局接受分数** = 从根到它的路径上各置信度之积（近似整条路径都被接受的概率）：

        1. **扩展**：每一层只选全局分数最高的 top-$k$ 个节点继续展开（每次草稿前向处理这 $k$ 个节点）；
        2. **重排**：在所有生成的节点中选全局分数最高的 $m$ 个作为送去验证的草稿。分数沿路径单调不增，所以选出的节点天然构成连通子树。

        若置信度≈接受率，「选分数最高的 $m$ 个」就是在最大化期望接受长度（期望接受数 = 被选节点分数之和）。EAGLE-2 **不需要额外训练**（直接用 EAGLE 的草稿模型），论文报告 3.05–4.26× 加速，比 EAGLE-1 快 20%–40%。
      `,
      viz: DynamicTreeViz,
      checkpoint: {
        id: 'cp-eagle2',
        type: 'numeric',
        prompt: md`某草稿节点的路径置信度依次为 0.9、0.8、0.5（从根的孩子到它自己）。它的全局接受分数是多少？`,
        answer: 0.36,
        tolerance: 0.001,
        explanation: md`$0.9 \times 0.8 \times 0.5 = 0.36$。它被接受当且仅当整条路径都被接受，所以分数是路径上条件接受率的乘积，而不是它自己的置信度 0.5。`,
      },
    },
    {
      id: 'eagle3',
      title: 'EAGLE-3：去掉特征约束，多层融合，训练时模拟推理',
      body: md`
        EAGLE-3（Li 等，arXiv 2503.01840）的出发点：EAGLE 的接受率随训练数据增加**几乎不涨**。原因是「预测下一个特征」是一个额外约束——我们真正关心的是 token，
        强迫草稿输出逼近目标特征，限制了草稿模型的表达能力。三个改动：

        | 改动 | 做法 | 作用 |
        |---|---|---|
        | 去掉特征回归损失 | 草稿直接预测 token，中间输出 $a$ 不再要求 $\approx f$ | 去掉约束后接受长度随数据增加持续提升（scaling） |
        | 多层特征融合 | 取目标模型**低、中、高**三层特征拼成 $3d$ 维，经 FC 压回 $d$ 维 | 只用最后一层特征信息不够丰富，尤其当它不再被要求预测特征时 |
        | training-time test（TTT） | 训练时就模拟多步起草：第 2、3… 步把草稿自己的输出 $a$ 喂回去，并相应调整注意力掩码 | 训练与推理的输入分布一致，后面位置的接受率不再快速下滑 |

        EAGLE-3 沿用 EAGLE-2 的动态草稿树（论文把树深从 6 加到 8、节点数不变）。论文报告：最高 **6.5×** 加速，比 EAGLE-2 快约 1.4×（batch = 1 的延迟）；
        Vicuna-13B 在 MT-bench（温度 0）上平均接受长度从 EAGLE-2 的 4.83 提到 6.65；LLaMA-Instruct 3.1 8B 上从 4.05 提到 6.13，其中去掉特征约束贡献到 5.37、再加多层融合到 6.13；
        在 SGLang 中 batch = 64 时吞吐仍提升 1.38×。今天 vLLM、SGLang 都把 EAGLE-3 作为主力的投机解码方案之一。
      `,
      after: md`
        > [!INTERVIEW] 一句话讲清 EAGLE → EAGLE-3
        > EAGLE：特征级自回归 + 提前一步的 token；EAGLE-2：用校准好的置信度动态长树；EAGLE-3：去掉特征回归、融合多层特征、训练时用自己的输出模拟推理，让草稿能随数据 scaling。
      `,
      checkpoint: {
        id: 'cp-eagle3',
        type: 'multi',
        prompt: md`EAGLE-3 相对 EAGLE 的改动包括：（多选）`,
        options: ['去掉特征预测（回归）损失，直接预测 token', '融合目标模型低、中、高三层特征', '训练时把草稿自己的输出作为后续步骤的输入（training-time test）', '把草稿模型加深到与目标模型相同层数', '放弃树结构，改用单链'],
        answer: [0, 1, 2],
        explanation: md`前三项是论文的核心改动。D 错：草稿仍然很小；E 错：EAGLE-3 沿用 EAGLE-2 的动态树。`,
      },
    },
    {
      id: 'mtp',
      title: 'MTP 作草稿与 P-EAGLE 并行草稿',
      body: md`
        **DeepSeek-V3 的 MTP**（multi-token prediction，技术报告 arXiv 2412.19437）：预训练时就在主模型后面串联 $D$ 个 MTP 模块，第 $k$ 个模块把上一深度的表示与 $x_{i+k}$ 的嵌入各自 RMSNorm 后拼接、投影，过一个 Transformer block，用**共享**的输出头预测 $x_{i+k+1}$。
        与 Gloeckle 等人「多个独立输出头并行预测」不同，它**顺序**预测、在每个深度都保留完整的因果链——结构上和 EAGLE 的草稿头非常像，只不过是和主模型一起预训练出来的。

        - V3 设 $D = 1$；MTP 损失权重前 10T token 为 0.3，之后 4.8T token 为 0.1；
        - 推理时 MTP 模块可以直接丢掉，也可以**作为投机解码的草稿**：报告第 2 个 token 的接受率在 **85%–90%**，解码速度（TPS）提升到 **1.8×**；
        - 只训练了 1 个深度，若递归复用同一个模块起草第 2、3 个草稿，接受率会明显下降。

        **P-EAGLE**（arXiv 2602.01469；vLLM 博客 2026-03-13，Amazon 与 NVIDIA 团队）：自回归起草 $K$ 个 token 要 $K$ 次串行前向，起草延迟随 $K$ 线性增长。P-EAGLE 改为**一次前向并行**产出全部 $K$ 个草稿：
        第 1 个位置照常输入目标特征与新 token 的嵌入；第 2…$K$ 个位置的输入还不存在，用两个**可学习的共享参数**（mask token 嵌入、共享隐状态）占位。
        博客中的草稿模型是 4 层、最多并行预测 10 个 token；训练时 $N \times K$ 个位置的注意力矩阵太大，论文用注意力掩码预计算与序列切分做梯度累积。
        结果：论文报告相对自回归 EAGLE-3 加速 1.10–1.36×；博客在 B200、GPT-OSS-20B 上报告最高 1.69×（并发 1）。vLLM 从 v0.16.0 起用 \`parallel_drafting\` 开启。
      `,
      after: md`
        > [!NEW] 2026 年的格局
        > EAGLE-3 已是开源推理引擎里的常用基线；自带 MTP 的模型（DeepSeek 系列等）可以直接拿 MTP 模块做草稿；P-EAGLE 这类并行草稿进一步压缩起草延迟。
        > 选型时比较的是「每轮期望接受长度 / 每轮总耗时（起草 + 验证）」，并且要在目标并发下实测——并发越高，验证的额外 FLOPs 越不免费。
      `,
      checkpoint: {
        id: 'cp-mtp',
        type: 'judge',
        prompt: md`DeepSeek-V3 的 MTP 模块与 Gloeckle 等人的方案一样，用多个相互独立的输出头并行预测未来多个 token。`,
        answer: false,
        explanation: md`错误。DeepSeek-V3 的 MTP 是**顺序**的：第 $k$ 个模块以第 $k-1$ 个深度的表示和 $x_{i+k}$ 的嵌入为输入，保留完整因果链，并共享嵌入与输出头。`,
      },
    },
  ],

  quiz: [
    {
      id: 'q-head',
      type: 'single',
      difficulty: 1,
      prompt: md`Medusa 的第 $k$ 个额外解码头（$k$ 从 1 开始），基于位置 $t$ 的最后一层隐状态预测的是：`,
      options: [md`$x_{t+1}$`, md`$x_{t+k}$`, md`$x_{t+k+1}$`, md`$x_{t-k}$`],
      answer: 2,
      explanation: md`原 LM head 预测 $x_{t+1}$，第 $k$ 个 Medusa 头预测 $x_{t+k+1}$。B 是常见的错位：$x_{t+1}$ 已由原 LM head 负责。`,
    },
    {
      id: 'q-typical',
      type: 'judge',
      difficulty: 1,
      prompt: md`Medusa 的 typical acceptance 与标准投机采样一样，保证输出分布与原模型完全一致。`,
      answer: false,
      explanation: md`typical acceptance 用熵相关的阈值判断「这个候选够合理就接受」，Medusa 论文明确说不追求与原分布严格一致；它是用一点分布偏差换更长的接受长度。`,
    },
    {
      id: 'q-eagle3-what',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于 EAGLE 系列，正确的有：（多选）`,
      options: [
        'EAGLE 的草稿预测的是目标模型 LM head 之前的特征，再用目标模型的 LM head 得到 token',
        'EAGLE-2 需要重新训练一个专门的置信度模型',
        'EAGLE-2 的扩展与重排都依据「路径上置信度之积」',
        'EAGLE-3 去掉了特征回归损失',
        'EAGLE-3 只使用目标模型最后一层的特征',
      ],
      answer: [0, 2, 3],
      explanation: md`B 错：EAGLE-2 直接用 EAGLE 草稿模型的置信度，**不需要额外训练**；E 错：EAGLE-3 融合低、中、高三层特征。`,
    },
    {
      id: 'q-medusa-nodes',
      type: 'numeric',
      difficulty: 2,
      prompt: md`4 个 Medusa 头都取 top-2，做完整笛卡尔积。树上共有多少个节点（不含根）？`,
      answer: 30,
      tolerance: 0,
      unit: '个',
      explanation: md`$2 + 4 + 8 + 16 = 30$。节点数随深度指数增长，这正是 Medusa 实际使用时要剪枝、EAGLE-2 要按分数动态选择的原因。`,
    },
    {
      id: 'q-expect-tree',
      type: 'numeric',
      difficulty: 3,
      prompt: md`EAGLE-2 选出 4 个草稿节点，全局分数分别为 0.9、0.72、0.5、0.36，且置信度校准良好。贪心验证下，每轮期望产出多少个 token（含 bonus）？`,
      answer: 3.48,
      tolerance: 0.001,
      unit: '个',
      explanation: md`每个节点被接受的概率就是它的全局分数，期望接受数 = $0.9 + 0.72 + 0.5 + 0.36 = 2.48$；无论接受几个，最后都会多产出 1 个（bonus 或修正 token），合计 3.48。`,
    },
    {
      id: 'q-fill-pos',
      type: 'fill',
      difficulty: 2,
      prompt: md`
        前缀长度 $P = 10$，草稿树 \`parents = [-1, -1, 0, 0, 2]\`（-1 表示根的孩子）。
        节点 4 的 position id = ____(1)____；树内掩码第 4 行中值为 True 的个数 = ____(2)____；完整掩码（含前缀列）的形状为 $5 \times$ ____(3)____。
      `,
      blanks: [
        { label: '(1)', answers: ['12'], numeric: true, tolerance: 0 },
        { label: '(2)', answers: ['3'], numeric: true, tolerance: 0 },
        { label: '(3)', answers: ['15'], numeric: true, tolerance: 0 },
      ],
      explanation: md`节点 4 的祖先链为 0 → 2 → 4，深度 2，position id = 10 + 2 = 12；树内可见 {0, 2, 4} 共 3 个；完整掩码 $n \times (P + n) = 5 \times 15$。`,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 2,
      prompt: md`把 EAGLE-2 一轮投机解码的步骤排序：`,
      items: [
        '草稿模型从根开始，每层对全局分数最高的 top-k 个节点各取 top-k 个孩子',
        '在全部生成的节点中按全局分数重排，取前 m 个',
        '按这 m 个节点构造树注意力掩码与位置编号',
        '目标模型一次前向，得到每个节点之后的分布',
        '从根沿接受的分支走到底，追加 bonus token，丢弃其余分支的 KV',
      ],
      explanation: md`扩展 → 重排 → 构造掩码 → 一次验证 → 选最长接受路径。扩展阶段的草稿前向次数等于树深，重排决定送去验证的 token 数。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把方法与其关键特征配对：`,
      pairs: [
        ['Medusa', '多个独立 FFN 头 + 笛卡尔积候选树'],
        ['EAGLE', '特征级自回归，输入加提前一步的 token'],
        ['EAGLE-2', '用草稿置信度动态构建草稿树，无需额外训练'],
        ['EAGLE-3', '去掉特征回归，多层特征融合，training-time test'],
        ['DeepSeek-V3 MTP', '随预训练一起训练的顺序预测模块，第 2 个 token 接受率 85–90%'],
        ['P-EAGLE', '一次前向并行产出全部草稿，后续位置用可学习占位输入'],
      ],
      explanation: md`演进主线：独立头（Medusa）→ 特征自回归（EAGLE）→ 动态树（EAGLE-2）→ 去约束 + 多层融合 + TTT（EAGLE-3）→ 并行起草（P-EAGLE）；MTP 是预训练副产品，结构与 EAGLE 草稿头相似。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 3,
      prompt: md`下面构造树注意力输入的代码有 **2 处** bug（\`parents[i] < i\`，-1 表示根的孩子）。点出有问题的行：`,
      code: py`
        import numpy as np

        def tree_inputs(parents, prefix_len):
            n = len(parents)
            tree = np.zeros((n, n), dtype=bool)
            depth = np.zeros(n, dtype=int)
            for i in range(n):
                p = parents[i]
                if p >= 0:
                    tree[i] = tree[p]
                    depth[i] = depth[p] + 1
                tree[i, p] = True
            full = np.concatenate([np.ones((n, prefix_len), dtype=bool), tree], axis=1)
            position_ids = prefix_len + np.arange(n)
            return full, position_ids
      `,
      bugLines: [12, 14],
      explanation: md`
        - 第 12 行应为 \`tree[i, i] = True\`：节点要能看到自己。写成 \`tree[i, p]\` 时，父节点本来已经从 \`tree[p]\` 继承到了，自己却被漏掉；而 \`p = -1\` 时更会错误地把最后一列置 True。
        - 第 14 行位置编号应为 \`prefix_len + depth\`：同一深度的兄弟节点必须共享位置，不能按展平下标递增。
      `,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`草稿树 \`parents = [-1, -1, 0, 0, 1]\`（节点 0、1 是根的孩子；2、3 是 0 的孩子；4 是 1 的孩子）。在 $5 \times 5$ 的树内掩码中，点出所有为 True 的格子（行 = query 节点，列 = key 节点）：`,
      items: Array.from({ length: 25 }, (_, i) => `${Math.floor(i / 5)}→${i % 5}`),
      cols: 5,
      rowLabels: ['0', '1', '2', '3', '4'],
      colLabels: ['0', '1', '2', '3', '4'],
      answer: [0, 6, 10, 12, 15, 18, 21, 24],
      explanation: md`每行为「祖先链 + 自己」：行 0 {0}，行 1 {1}，行 2 {0, 2}，行 3 {0, 3}，行 4 {1, 4}。兄弟之间（2 与 3、0 与 1）互不可见。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出贪心树验证：\`children[c]\` 是节点 c 的孩子列表（\`children[-1]\` 为根的孩子），\`argmax_at(c)\` 是目标模型在节点 c 之后的 argmax（\`c = -1\` 表示根）。有 2 行干扰项：`,
      lines: [
        'def greedy_verify(tokens, children, argmax_at):',
        '    path, cur = [], -1',
        '    while True:',
        '        want = argmax_at(cur)',
        '        nxt = [c for c in children[cur] if tokens[c] == want]',
        '        if not nxt:',
        '            return path, want',
        '        cur = nxt[0]',
        '        path.append(cur)',
      ],
      distractors: ['        want = tokens[cur]', '            return path, tokens[cur]'],
      explanation: md`当前节点处目标模型的 argmax 就是「下一个应该是什么」；有孩子匹配就走下去，否则停止，此时 \`want\` 正是 bonus / 修正 token。干扰项把草稿 token 当成了目标模型的预测。`,
    },
    {
      id: 'q-mtp',
      type: 'single',
      difficulty: 2,
      prompt: md`DeepSeek-V3 技术报告中，把 MTP 模块用于投机解码的结果是：`,
      options: ['第 2 个 token 接受率 85%–90%，解码 TPS 提升到 1.8×', '第 2 个 token 接受率 50% 左右，几乎没有加速', '接受率 99%，加速 4×', '报告没有给出 MTP 用于推理的结果'],
      answer: 0,
      explanation: md`报告原文给出了 85%–90% 的第 2 个 token 接受率与 1.8 倍 TPS。注意 V3 只训练了 $D = 1$ 个 MTP 模块，所以天然只起草 1 个 token。`,
    },
  ],

  guided: [
    {
      id: 'g-tree-inputs',
      title: '由草稿树构造掩码与位置编号',
      intro: md`
        输入 \`parents\`（长度 $n$，保证 \`parents[i] < i\`，-1 表示挂在根上）和前缀长度 $P$，返回
        - \`full\`：形状 \`(n, P + n)\` 的布尔掩码，前 $P$ 列全为 True，树内部分为「祖先链 + 自己」；
        - \`position_ids\`：形状 \`(n,)\`，等于 $P$ + 深度（根的孩子深度为 0）。

        技巧：按拓扑序处理，节点 $i$ 能看到的 = 父节点能看到的 ∪ {自己}，每行 $O(n)$ 复制即可。
      `,
      template: py`
        import numpy as np

        def tree_inputs(parents, prefix_len):
            n = len(parents)
            tree = np.zeros((n, n), dtype=bool)
            depth = np.zeros(n, dtype=int)
            for i in range(n):
                p = parents[i]
                if p >= 0:
                    tree[i] = {{1}}
                    depth[i] = {{2}}
                tree[i, i] = True
            full = {{3}}
            position_ids = {{4}}
            return full, position_ids
      `,
      steps: [
        {
          blank: 1,
          instruction: md`节点 $i$ 继承父节点那一行（父节点能看到的祖先链）。`,
          answer: 'tree[p]',
          accept: ['tree[p].copy()', 'tree[p, :]'],
          explain: md`NumPy 的整行赋值会复制数值，后续修改 \`tree[i]\` 不会影响 \`tree[p]\`。因为 \`p < i\`，\`tree[p]\` 已经算好。`,
        },
        {
          blank: 2,
          instruction: md`深度。`,
          answer: 'depth[p] + 1',
          explain: md`根的孩子保持默认的 0。`,
        },
        {
          blank: 3,
          instruction: md`在左边拼上 $P$ 列全 True（前缀对所有草稿节点可见）。`,
          answer: 'np.concatenate([np.ones((n, prefix_len), dtype=bool), tree], axis=1)',
          accept: ['np.hstack([np.ones((n, prefix_len), dtype=bool), tree])', 'np.concatenate((np.ones((n, prefix_len), dtype=bool), tree), axis=1)'],
          explain: md`实际框架里前缀部分通常走 paged KV，不会真的物化这 $P$ 列；树内部那块小掩码作为额外的 mask 传给 kernel。`,
        },
        {
          blank: 4,
          instruction: md`位置编号。`,
          answer: 'prefix_len + depth',
          accept: ['depth + prefix_len'],
          explain: md`同一深度的兄弟共享位置：它们是「同一个位置的不同候选」。`,
        },
      ],
      setup: py`
        import numpy as np

        def _ref_tree(parents, P):
            n = len(parents)
            full = np.zeros((n, P + n), dtype=bool); full[:, :P] = True
            pos = np.zeros(n, dtype=int)
            for i in range(n):
                j, d = i, -1
                while j != -1:
                    full[i, P + j] = True; j = parents[j]; d += 1
                pos[i] = P + d
            return full, pos
      `,
      tests: [
        {
          name: '小例子',
          code: py`
            full, pos = tree_inputs([-1, -1, 0, 0, 2], 10)
            assert full.shape == (5, 15), f"形状应为 (5, 15)，得到 {full.shape}"
            assert pos.tolist() == [10, 10, 11, 11, 12], f"position_ids 期望 [10,10,11,11,12]，得到 {pos.tolist()}"
            assert full[4, 10:].tolist() == [True, False, True, False, True], f"节点 4 应看到 0、2、4，得到 {full[4, 10:].tolist()}"
          `,
        },
        {
          name: '单链退化为因果掩码',
          code: py`
            full, pos = tree_inputs([-1, 0, 1, 2], 3)
            assert np.array_equal(full[:, 3:], np.tril(np.ones((4, 4), dtype=bool))), "单链时树内掩码应为下三角"
            assert pos.tolist() == [3, 4, 5, 6]
          `,
        },
        {
          name: '随机树对拍',
          code: py`
            rng = np.random.default_rng(0)
            for _ in range(100):
                n = int(rng.integers(1, 20)); P = int(rng.integers(0, 6))
                parents = [-1] + [int(rng.integers(-1, i)) for i in range(1, n)]
                full, pos = tree_inputs(parents, P)
                ef, ep = _ref_tree(parents, P)
                assert np.array_equal(full, ef), f"parents={parents}: 掩码不一致"
                assert np.array_equal(pos, ep), f"parents={parents}: 位置期望 {ep.tolist()}，得到 {np.asarray(pos).tolist()}"
          `,
        },
      ],
      outro: md`
        有了掩码和位置编号，树注意力就是普通的带掩码注意力（实战题 1）。实际 kernel 会把「前缀部分」与「树内部分」分开处理：前缀走 paged KV 的标准路径，树内掩码很小，可以做成位图传进去。
      `,
    },
  ],

  challenges: [
    {
      id: 'c-tree-attn',
      title: '树注意力：一次算出所有草稿节点的注意力',
      difficulty: 'medium',
      source: 'Medusa / EAGLE · 树注意力',
      timeLimitMin: 20,
      description: md`
        已提交前缀的 K/V 为 \`k_cache, v_cache\`（形状 \`(P, d)\`），草稿树有 $n$ 个节点，其 query/key/value 为 \`q, k, v\`（形状 \`(n, d)\`），
        \`parents[i] < i\`，-1 表示挂在根上。实现

        \`\`\`python
        def tree_attention(q, k, v, parents, k_cache, v_cache) -> np.ndarray: ...   # 返回 (n, d)
        \`\`\`

        节点 $i$ 的输出 = 它的 query 对「全部前缀 + 自己的祖先链 + 自己」做缩放点积注意力（$\mathrm{softmax}(qK^\top/\sqrt d)V$）。
        **等价性**：结果必须与「对每条根到节点的路径单独做一次普通注意力」完全一致（隐藏测试逐节点对拍）。

        **示例**：\`parents = [-1, -1, 0]\` 时，节点 2 可见前缀 + 节点 0 + 节点 2，不可见节点 1。

        **约束**：$n \le 300$，$P \le 64$，$d \le 64$。$P = 0$ 时也要正确。
      `,
      starter: py`
        import numpy as np

        def tree_attention(q, k, v, parents, k_cache, v_cache):
            # 1) 构造 (n, P+n) 的可见性掩码  2) 拼接 K、V  3) 带掩码的 softmax 注意力
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def tree_attention(q, k, v, parents, k_cache, v_cache):
            n, d = q.shape
            P = k_cache.shape[0]
            # 树内可见性：节点 i 看到的 = 父节点看到的 ∪ {i}（parents[i] < i，按顺序递推）
            tree = np.zeros((n, n), dtype=bool)
            for i, p in enumerate(parents):
                if p >= 0:
                    tree[i] = tree[p]
                tree[i, i] = True
            mask = np.concatenate([np.ones((n, P), dtype=bool), tree], axis=1)   # 前缀对所有节点可见
            K = np.concatenate([k_cache, k], axis=0)
            V = np.concatenate([v_cache, v], axis=0)
            s = q @ K.T / np.sqrt(d)
            s = np.where(mask, s, -np.inf)          # 每行至少能看到自己，不会全是 -inf
            s = s - s.max(axis=1, keepdims=True)
            w = np.exp(s)
            w /= w.sum(axis=1, keepdims=True)
            return w @ V
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示）',
          code: py`
            import torch
            import torch.nn.functional as F

            def tree_attention(q, k, v, parents, k_cache, v_cache):
                n, P = q.shape[0], k_cache.shape[0]
                tree = torch.zeros(n, n, dtype=torch.bool, device=q.device)
                for i, p in enumerate(parents):
                    if p >= 0:
                        tree[i] = tree[p]
                    tree[i, i] = True
                mask = torch.cat([torch.ones(n, P, dtype=torch.bool, device=q.device), tree], dim=1)
                K, V = torch.cat([k_cache, k]), torch.cat([v_cache, v])
                # SDPA 的布尔掩码：True 表示可以 attend
                return F.scaled_dot_product_attention(q[None], K[None], V[None], attn_mask=mask[None]).squeeze(0)
          `,
        },
      ],
      explanation: md`
        **思路**：树注意力 = 普通注意力 + 特殊掩码。掩码按拓扑序递推：\`tree[i] = tree[parents[i]]\`，再把对角线置 True，总复杂度 $O(n^2)$。

        **为什么与逐路径计算等价**：softmax 只在可见的位置上归一化；节点 $i$ 可见的集合恰好是「前缀 + 根到 $i$ 的路径」，与把这条路径当作普通序列做因果注意力时最后一个位置看到的集合相同。
        真实模型里还要配合位置编号（RoPE 用 $P$ + 深度），这里没有位置编码所以不涉及。

        **复杂度**：时间 $O(n(P + n)d)$，空间 $O(n(P + n))$。

        **常见错误**：忘了前缀列；把掩码做成普通下三角（兄弟之间互相可见）；用 0 而不是 $-\infty$ 屏蔽（softmax 后被屏蔽位置仍有权重）。
      `,
      complexity: { time: 'O(n·(P+n)·d)', space: 'O(n·(P+n))' },
      hints: [
        md`先按拓扑序构造 \`tree\`：\`tree[i] = tree[p]\`（若 \`p >= 0\`），\`tree[i, i] = True\`。`,
        md`\`mask = np.concatenate([np.ones((n, P), bool), tree], axis=1)\`，\`K = np.concatenate([k_cache, k])\`，V 同理。`,
        md`\`s = np.where(mask, q @ K.T / sqrt(d), -inf)\`，再做数值稳定的 softmax。`,
      ],
      setup: py`
        import numpy as np

        def _ref_path_attn(q, k, v, parents, kc, vc):
            n, d = q.shape
            out = np.zeros_like(q)
            for i in range(n):
                path = []
                j = i
                while j != -1:
                    path.append(j); j = parents[j]
                path = path[::-1]
                K = np.concatenate([kc, k[path]]); V = np.concatenate([vc, v[path]])
                s = K @ q[i] / np.sqrt(d); s -= s.max()
                w = np.exp(s); w /= w.sum()
                out[i] = w @ V
            return out

        def _rand(seed, n, P, d, branch=0.5):
            rng = np.random.default_rng(seed)
            parents = [-1]
            for i in range(1, n):
                parents.append(int(rng.integers(-1, i)) if rng.random() < branch else i - 1)
            q, k, v = (rng.standard_normal((n, d)) for _ in range(3))
            kc, vc = rng.standard_normal((P, d)), rng.standard_normal((P, d))
            return q, k, v, parents, kc, vc
      `,
      tests: [
        {
          name: '示例：兄弟互不可见',
          code: py`
            q, k, v, _, kc, vc = _rand(0, 3, 4, 8)
            parents = [-1, -1, 0]
            out = tree_attention(q, k, v, parents, kc, vc)
            assert out.shape == (3, 8), f"形状应为 (3, 8)，得到 {out.shape}"
            exp = _ref_path_attn(q, k, v, parents, kc, vc)
            assert np.allclose(out, exp, atol=1e-10), f"最大误差 {np.abs(out - exp).max():.2e}"
          `,
        },
        {
          name: '单链 = 普通因果注意力',
          code: py`
            q, k, v, _, kc, vc = _rand(1, 6, 5, 16)
            parents = [-1, 0, 1, 2, 3, 4]
            out = tree_attention(q, k, v, parents, kc, vc)
            assert np.allclose(out, _ref_path_attn(q, k, v, parents, kc, vc), atol=1e-10)
          `,
        },
        {
          name: '空前缀 P = 0',
          code: py`
            q, k, v, parents, kc, vc = _rand(2, 7, 0, 8)
            out = tree_attention(q, k, v, parents, kc, vc)
            assert np.allclose(out, _ref_path_attn(q, k, v, parents, kc, vc), atol=1e-10), "P=0 时结果不对"
          `,
        },
        {
          name: '随机树逐节点对拍',
          hidden: true,
          code: py`
            for seed in range(30):
                rng = np.random.default_rng(100 + seed)
                n, P, d = int(rng.integers(1, 40)), int(rng.integers(0, 20)), int(rng.choice([4, 8, 32]))
                q, k, v, parents, kc, vc = _rand(seed, n, P, d, branch=float(rng.random()))
                out = tree_attention(q, k, v, parents, kc, vc)
                exp = _ref_path_attn(q, k, v, parents, kc, vc)
                assert np.allclose(out, exp, atol=1e-9), f"seed={seed}, n={n}, P={P}: 最大误差 {np.abs(out - exp).max():.2e}"
          `,
        },
        {
          name: '规模：300 个节点',
          hidden: true,
          code: py`
            q, k, v, parents, kc, vc = _rand(7, 300, 64, 64, branch=0.3)
            out = tree_attention(q, k, v, parents, kc, vc)
            assert out.shape == (300, 64), f"形状应为 (300, 64)，得到 {out.shape}"
            ref = _ref_path_attn(q, k, v, parents, kc, vc)
            assert np.allclose(out, ref, atol=1e-9), f"最大误差 {np.abs(out - ref).max():.2e}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-kv',
          q: md`验证完成后，被接受路径的 KV 怎么处理？`,
          a: md`目标模型在这次前向里为全部 $n$ 个节点写了 KV（按展平顺序）。接受路径上的节点在展平序列里不一定连续，需要把它们的 KV 拷到前缀之后的连续位置（或修改分页 KV 的块表 / slot 映射），其余节点的 KV 丢弃。EAGLE / Medusa 的实现里都有这一步「KV 重排」，它是树验证相对链式验证的额外开销之一。`,
        },
      ],
    },
    {
      id: 'c-greedy-verify',
      title: '贪心树验证：找最长被接受路径',
      difficulty: 'medium',
      source: 'Medusa / EAGLE · 贪心验证',
      timeLimitMin: 15,
      description: md`
        目标模型对草稿树做完一次前向后，给出：
        - \`root_argmax\`：在根（最后一个已提交 token）之后的 argmax；
        - \`node_argmax[i]\`：在节点 $i$ 之后的 argmax。

        草稿树：\`tokens[i]\` 为节点 $i$ 的草稿 token，\`parents[i]\` 为父节点（-1 为根），\`parents[i] < i\`。实现

        \`\`\`python
        def greedy_tree_verify(tokens, parents, root_argmax, node_argmax) -> tuple[list[int], int]: ...
        \`\`\`

        从根出发：若当前节点有孩子的 token 等于「当前位置的 argmax」，就走到这个孩子（多个孩子 token 相同时取**下标最小**的）并继续；否则停止。
        返回 \`(接受的节点下标路径, 下一个 token)\`，其中「下一个 token」是停止处的 argmax（bonus / 修正 token）。

        **示例**：\`tokens = [5, 7, 8, 9, 7]\`，\`parents = [-1, -1, 0, 0, 1]\`，\`root_argmax = 5\`，\`node_argmax = [9, 1, 2, 3, 4]\`
        → 根处 argmax 5 匹配节点 0；节点 0 处 argmax 9 匹配节点 3；节点 3 处 argmax 3，节点 3 没有孩子 → 返回 \`([0, 3], 3)\`。

        **要求**：$O(n)$ 预处理孩子列表，不要每一步都扫描全部节点（隐藏测试 $n = 20000$）。
      `,
      starter: py`
        def greedy_tree_verify(tokens, parents, root_argmax, node_argmax):
            # 1) 建 children 列表  2) 从根开始沿匹配的孩子走
            return [], root_argmax
      `,
      solution: py`
        def greedy_tree_verify(tokens, parents, root_argmax, node_argmax):
            n = len(tokens)
            children = [[] for _ in range(n + 1)]       # children[n] 代表根
            for i, p in enumerate(parents):              # 按下标顺序加入 → 每个列表天然有序
                children[p if p >= 0 else n].append(i)
            path, cur, want = [], n, root_argmax
            while True:
                nxt = next((c for c in children[cur] if tokens[c] == want), None)
                if nxt is None:
                    return path, want                   # want 即 bonus / 修正 token
                path.append(nxt)
                cur, want = nxt, node_argmax[nxt]
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示：Medusa / EAGLE 实现中按路径矩阵向量化验证）',
          code: py`
            import torch

            def greedy_tree_verify_paths(tokens, root_argmax, node_argmax, retrieve):
                """retrieve: (num_paths, depth)，每条根→叶路径上的节点下标，不足处补 -1（即实现中的 retrieve_indices）
                tokens / node_argmax: (n,) LongTensor"""
                valid = retrieve >= 0
                idx = retrieve.clamp(min=0)
                cand = torch.where(valid, tokens[idx], torch.full_like(idx, -1))          # 路径上的草稿 token
                prev = torch.cat([torch.full_like(idx[:, :1], root_argmax), node_argmax[idx[:, :-1]]], dim=1)
                match = (cand == prev) & valid                                            # 每一步是否命中目标 argmax
                acc_len = match.int().cumprod(dim=1).sum(dim=1)                           # 从头开始连续命中的长度
                best = int(acc_len.argmax())
                L = int(acc_len[best])
                path = retrieve[best, :L].tolist()
                nxt = root_argmax if L == 0 else int(node_argmax[retrieve[best, L - 1]])
                return path, nxt
          `,
        },
      ],
      explanation: md`
        **思路**：一次遍历建立孩子列表（按下标升序，天然满足「取下标最小」），然后从根贪心下行：当前位置目标模型的 argmax 就是唯一正确的下一个 token，
        有孩子匹配就前进。停止时的 argmax 不需要额外前向——它已经由这次验证前向算好了，这就是 bonus（全部匹配到叶子）或修正 token（中途不匹配）。

        **复杂度**：建表 $O(n)$；下行每一步只扫描当前节点的孩子，总共 $O(n)$。

        **常见错误**：每步遍历全部节点找孩子（$O(n \cdot \text{depth})$ 甚至更差）；返回的下一个 token 用了草稿 token；忘了根的孩子 \`parents = -1\`。

        **采样场景**：贪心只需比较 argmax；按温度采样时要用多候选的拒绝采样（依次尝试同一父节点的多个孩子、每次失败后更新残差分布），才能保持无损。
      `,
      complexity: { time: 'O(n)', space: 'O(n)' },
      hints: [
        md`\`children = [[] for _ in range(n + 1)]\`，用下标 \`n\` 代表根，\`for i, p in enumerate(parents): children[p if p >= 0 else n].append(i)\`。`,
        md`\`cur = n\`，\`want = root_argmax\`；循环中在 \`children[cur]\` 里找 \`tokens[c] == want\` 的第一个孩子。`,
        md`找到就 \`path.append(c)\`、\`want = node_argmax[c]\`；找不到就返回 \`(path, want)\`。`,
      ],
      setup: py`
        import random, time

        def _ref_verify(tokens, parents, ra, na):
            path, cur, want = [], -1, ra
            while True:
                nxt = None
                for c in range(len(tokens)):
                    if parents[c] == cur and tokens[c] == want:
                        nxt = c; break
                if nxt is None:
                    return path, want
                path.append(nxt); cur, want = nxt, na[nxt]

        def _norm(r):
            return [int(x) for x in r[0]], int(r[1])

        class _CountList(list):
            """计数访问次数的 list：超出预算立即失败（避免 O(n^2) 写法拖到超时）"""
            def __init__(self, data, budget):
                super().__init__(data); self.reads = 0; self.budget = budget
            def _hit(self, k):
                self.reads += k
                if self.reads > self.budget:
                    raise AssertionError(f"访问 parents 已超过 {self.budget} 次（O(n) 预算）——是不是每一步都扫描了全部节点？")
            def __getitem__(self, i):
                self._hit(len(range(*i.indices(len(self)))) if isinstance(i, slice) else 1)
                return super().__getitem__(i)
            def __iter__(self):
                self._hit(len(self))
                return super().__iter__()
      `,
      tests: [
        { name: '示例', code: py`assert _norm(greedy_tree_verify([5, 7, 8, 9, 7], [-1, -1, 0, 0, 1], 5, [9, 1, 2, 3, 4])) == ([0, 3], 3)` },
        {
          name: '第一层就不匹配',
          code: py`
            got = _norm(greedy_tree_verify([5, 7], [-1, -1], 6, [1, 2]))
            assert got == ([], 6), f"根处 argmax 6 不匹配任何孩子，应返回 ([], 6)，得到 {got}"
          `,
        },
        {
          name: '走到叶子：bonus 来自叶子处的 argmax',
          code: py`
            got = _norm(greedy_tree_verify([1, 2, 3], [-1, 0, 1], 1, [2, 3, 42]))
            assert got == ([0, 1, 2], 42), f"期望 ([0, 1, 2], 42)，得到 {got}"
          `,
        },
        {
          name: '同 token 的兄弟取下标最小',
          code: py`
            got = _norm(greedy_tree_verify([4, 4, 6, 7], [-1, -1, 1, 0], 4, [7, 6, 0, 9]))
            assert got == ([0, 3], 9), f"两个孩子都是 4，应走节点 0，期望 ([0, 3], 9)，得到 {got}"
          `,
        },
        {
          name: '随机树对拍',
          hidden: true,
          code: py`
            rnd = random.Random(1)
            for _ in range(300):
                n = rnd.randint(1, 30); V = rnd.randint(2, 5)
                parents = [-1] + [rnd.randint(-1, i - 1) for i in range(1, n)]
                tokens = [rnd.randrange(V) for _ in range(n)]
                na = [rnd.randrange(V) for _ in range(n)]
                ra = rnd.randrange(V)
                exp = _ref_verify(tokens, parents, ra, na)
                got = _norm(greedy_tree_verify(tokens, parents, ra, na))
                assert got == exp, f"tokens={tokens}, parents={parents}, root={ra}, node_argmax={na}: 期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '规模：20000 个节点的深链',
          hidden: true,
          code: py`
            n = 20000
            parents = _CountList([-1] + list(range(n - 1)), budget=4 * n + 100)
            tokens = [i % 7 for i in range(n)]
            na = [(i + 1) % 7 for i in range(n)]
            t0 = time.perf_counter()
            got = _norm(greedy_tree_verify(tokens, parents, 0, na))
            dt = time.perf_counter() - t0
            assert len(got[0]) == n and got[1] == n % 7, f"应接受全部 {n} 个节点，实际接受 {len(got[0])} 个"
            assert dt < 0.5, f"耗时 {dt:.2f}s，超出线性复杂度的预期"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-sample',
          q: md`如果按温度采样，树验证怎么保持无损？`,
          a: md`对当前节点，把它的孩子当作多个独立草稿依次尝试：第 1 个孩子按 $\min(1, p/q)$ 接受；被拒后把目标分布更新为残差 $\mathrm{norm}(\max(0, p - q))$，再用它去检验第 2 个孩子……全部被拒则从最终残差采样。这就是多候选的投机采样（SpecInfer 等工作中讨论过），要求各孩子是按草稿分布独立（或无放回并相应修正）采样得到的。`,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-medusa-eagle',
      freq: 3,
      q: md`Medusa 和 EAGLE 的区别是什么？为什么 EAGLE 接受率更高？`,
      a: md`
        **结论**：Medusa 是多个**独立并行**的头直接从同一个隐状态预测远处 token；EAGLE 是一个小草稿层在**特征层面自回归**、每步都知道前一步猜了什么。

        **原因**：Medusa 的第 $k$ 个头要在不知道中间 token 的情况下猜 $x_{t+k+1}$，越远越难；EAGLE 把上一步预测的特征和 token 喂回去，相当于一个复用目标模型 LM head 的微型语言模型，而且加入提前一步的 token 消除了特征的采样不确定性。论文报告 EAGLE 在 MT-bench 上比 Medusa 快 1.5–1.6×。

        **延伸**：EAGLE-2 用置信度动态长树（无需再训练），EAGLE-3 去掉特征回归、融合多层特征并用 training-time test 训练，最高 6.5× 加速；DeepSeek-V3 的 MTP 模块结构上与 EAGLE 草稿头相似，可直接作草稿。
      `,
    },
    {
      id: 'qa-tree',
      freq: 2,
      q: md`树注意力怎么实现？需要注意什么？`,
      a: md`
        **结论**：把草稿树按 BFS 展平成一个序列，用特殊掩码 + 特殊位置编号做一次前向。

        **细节**：① 掩码 $n \times (P+n)$：前缀全可见，树内只可见祖先链与自己；② position id = 前缀长度 + 深度，兄弟共享位置；③ 前向后每个节点都得到「它之后」的目标分布，贪心验证沿匹配的孩子走到底；④ 接受路径的 KV 要重排到连续位置（或改块表），其余丢弃；⑤ 树的节点数就是验证的 token 数，大 batch 下要缩小树。

        **延伸**：采样模式下需要多候选的拒绝采样才能无损；Medusa 的 typical acceptance 是有损的捷径。
      `,
    },
    {
      id: 'qa-eagle3',
      freq: 2,
      q: md`EAGLE-3 相对 EAGLE / EAGLE-2 做了哪些改进？为什么有效？`,
      a: md`
        **结论**：去掉特征预测损失、改为直接预测 token；用低、中、高三层特征融合作为输入；训练时用 training-time test 模拟多步起草。

        **原因**：EAGLE 的接受率随训练数据增加几乎不涨，作者认为「预测下一特征」这个额外约束限制了草稿的表达能力；去掉后，只用最后一层特征信息不够，于是融合多层；而草稿在推理时第 2 步起吃的是自己的输出，训练时也必须这样喂，才能避免训练 / 推理分布不一致。消融：LLaMA-3.1-8B 上接受长度 4.05（EAGLE-2）→ 5.37（去约束）→ 6.13（多层融合）。

        **延伸**：最高 6.5× 加速，比 EAGLE-2 快约 1.4×；SGLang 中 batch = 64 吞吐提升 1.38×。仍使用 EAGLE-2 的动态树。
      `,
    },
    {
      id: 'qa-mtp',
      freq: 2,
      q: md`DeepSeek-V3 的 MTP 怎么用于推理加速？和 Medusa 的多头有什么不同？`,
      a: md`
        **结论**：MTP 模块可以直接作为投机解码的草稿：报告第 2 个 token 接受率 85%–90%，TPS 提升到 1.8×。

        **区别**：Medusa（以及 Gloeckle 等的多头 MTP）是多个独立头并行预测；DeepSeek-V3 的 MTP 是**顺序**模块，第 $k$ 个模块输入上一深度的表示与 $x_{i+k}$ 的嵌入，保留完整因果链，共享嵌入与输出头，和主模型一起预训练（损失权重 0.3 → 0.1）。

        **延伸**：V3 只有 $D = 1$ 个模块，递归复用来起草更多 token 时接受率明显下降；若想要更长的草稿，可以训练更多 MTP 深度或另训 EAGLE-3 草稿。
      `,
    },
    {
      id: 'qa-peagle',
      freq: 1,
      q: md`P-EAGLE 解决了什么问题？怎么做到并行起草？`,
      a: md`
        **结论**：解决自回归起草的串行延迟——起草 $K$ 个 token 需要 $K$ 次草稿前向。

        **做法**：一次前向产出全部 $K$ 个草稿。第 1 个位置用真实的目标特征与新 token 嵌入；后续位置用可学习的共享 mask 嵌入与共享隐状态占位。训练时 $N \times K$ 个位置的注意力矩阵过大，用掩码预计算与序列切分做梯度累积。

        **效果**：论文报告比自回归 EAGLE-3 快 1.10–1.36×；vLLM 博客（B200、GPT-OSS-20B）报告最高 1.69×（并发 1）。vLLM v0.16.0 起通过 \`parallel_drafting\` 支持。
      `,
    },
  ],

  cheatsheet: md`
    | 方法 | 草稿怎么来 | 关键数字 |
    |---|---|---|
    | Medusa | $K$ 个带残差的 FFN 头，头 $k$ 预测 $x_{t+k+1}$；top-k 笛卡尔积树 | Medusa-1 >2.2×，Medusa-2 2.3–2.8× |
    | EAGLE | FC + 1 层 decoder，输入 (特征, 提前一步 token)，特征级自回归；$L_{reg} + 0.1 L_{cls}$ | LLaMA2-Chat 70B 2.7–3.5× |
    | EAGLE-2 | 置信度≈接受率；按路径置信度积扩展 top-k、重排 top-m；无需训练 | 3.05–4.26×，比 EAGLE-1 快 20–40% |
    | EAGLE-3 | 去特征回归、低中高三层融合、training-time test | 最高 6.5×，接受长度 4.83→6.65（Vicuna-13B） |
    | DeepSeek-V3 MTP | 顺序 MTP 模块（$D=1$），共享嵌入与输出头 | 第 2 token 接受率 85–90%，1.8× TPS |
    | P-EAGLE | 一次前向并行起草，可学习 mask 嵌入 + 共享隐状态 | 比 EAGLE-3 快 1.10–1.36×（论文） |

    **树注意力**：掩码 $n \times (P+n)$ = 前缀全 1 + 祖先链；position id = $P$ + 深度
    \`\`\`python
    for i, p in enumerate(parents):
        if p >= 0: tree[i] = tree[p]
        tree[i, i] = True
    \`\`\`
    **贪心验证**：从根沿「token == 当前位置 argmax」的孩子走到底，停止处的 argmax 即 bonus
    - Medusa 的 typical acceptance 有损；标准拒绝采样无损
    - 树节点数 = 验证 token 数，大 batch 时要缩小
  `,
})
