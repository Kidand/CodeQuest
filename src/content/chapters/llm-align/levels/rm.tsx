/**
 * 奖励模型：Bradley-Terry 推导与损失、margin、K 路排序、ORM vs PRM、生成式 RM 与 LLM-as-a-Judge 偏差、reward hacking。
 * 事实来源：InstructGPT（arXiv 2203.02155 §3.5）、Llama 2（2307.09288 §3.2.2、Table 27/28）、Llama 3（2407.21783 §4.1.2）、
 * Let's Verify Step by Step（2305.20050）、Math-Shepherd（2312.08935）、DeepSeek-R1（2501.12948 §4.2）、
 * Judging LLM-as-a-Judge（2306.05685）、Generative Verifiers（2408.15240）、DeepSeek-GRM（2504.02495）、
 * Scaling Laws for Reward Model Overoptimization（2210.10760）、Tülu 3（2411.15124）。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { BtViz } from '../viz/BtViz'
import { RankPairsViz } from '../viz/RankPairsViz'
import { PrmViz } from '../viz/PrmViz'
import { JudgeBiasViz } from '../viz/JudgeBiasViz'

export default defineLevel({
  learn: [
    {
      id: 'bt',
      title: 'Bradley-Terry：从“谁更好”到一个标量分数',
      body: md`
        人类给不出可靠的绝对分，但比较两个回答很容易。**Bradley-Terry** 模型假设每个回答有潜在分数 $r$，偏好概率只取决于分差：

        $$
        P(y_w\succ y_l\mid x)=\frac{e^{r(x,y_w)}}{e^{r(x,y_w)}+e^{r(x,y_l)}}=\sigma\big(r(x,y_w)-r(x,y_l)\big)
        $$

        对偏好数据做最大似然，就得到 RM 的训练损失（InstructGPT Eq.1、Llama 2 Eq.1）：

        $$
        \mathcal L_{\mathrm{RM}}=-\,\mathbb E_{(x,y_w,y_l)}\big[\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\big],\qquad
        \frac{\partial\ell}{\partial r_w}=-\sigma(r_l-r_w)=-\frac{\partial\ell}{\partial r_l}
        $$

        **结构**：拿 SFT（或预训练）模型，去掉 unembedding 层，换成输出标量的线性头，取**最后一个 token（EOS）处**的隐状态打分。InstructGPT 用 6B RM；Llama 2 分别训练有用性与安全性两个 RM，从 chat 模型 checkpoint 初始化；Llama 3 从预训练 checkpoint 初始化。

        **margin**：Llama 2 让分差至少超过一个由标注强度决定的 $m(r)$：$-\log\sigma(r_w-r_l-m(r))$。拖动下图验证平移不变性，并比较两种 margin 方案。
      `,
      viz: BtViz,
      after: md`
        > [!TIP] Llama 2 的 margin 数值（Table 27）
        > | | 显著更好 | 更好 | 略好 | 几乎一样 / 不确定 |
        > |---|---|---|---|---|
        > | Margin Small | 1 | 2/3 | 1/3 | 0 |
        > | Margin Large | 3 | 2 | 1 | 0 |
        >
        > 消融（Table 28）显示 margin 在可分性强的样本上提升准确率，large margin 会损害相似样本上的表现；**Llama 3 去掉了 margin**，理由是数据规模变大后收益递减。

        **平移不变性的后果**：RM 分数只有同一 prompt 内的差值有意义，绝对值可以整体漂移。所以 InstructGPT 在 RL 前加一个偏置使人类示范的平均分为 0；Llama 2 对分数做 whiten；跨 prompt 比较 RM 分数需要格外小心。
      `,
      checkpoint: {
        id: 'cp-bt',
        type: 'numeric',
        prompt: md`Llama 2 的 Margin Small 方案下，一对“显著更好”的样本，RM 给出 $r_w=2$、$r_l=1$。这对样本的 loss 是多少？（自然对数，保留 3 位小数）`,
        answer: 0.693,
        tolerance: 0.005,
        explanation: md`“显著更好”对应 $m=1$，$-\log\sigma(2-1-1)=-\log\sigma(0)=\ln2\approx0.693$。没有 margin 时是 $-\log\sigma(1)\approx0.313$——margin 要求它把分差拉得更大。`,
      },
    },
    {
      id: 'ranking',
      title: 'K 路排序与 RM 训练细节',
      body: md`
        InstructGPT 让标注员一次对 **K = 4 到 9** 个回答排序，得到 $\binom K2$ 个比较。完整损失（Eq.1）：

        $$
        \mathcal L(\theta)=-\frac{1}{\binom K2}\,\mathbb E_{(x,y_w,y_l)\sim D}\big[\log\sigma\big(r_\theta(x,y_w)-r_\theta(x,y_l)\big)\big]
        $$

        关键细节：**同一 prompt 的全部比较放进同一个 batch 元素**。原因有二：
        1. 效率：K 个回答各前向一次，就能算出全部 $\binom K2$ 个比较；
        2. 防过拟合：同一 prompt 的比较高度相关，打散成独立样本时，每个回答会在一个 epoch 里被用 $K-1$ 次，论文观察到 RM 一个 epoch 后就过拟合。

        后来的做法类似：Llama 3 的偏好样本有 2–3 个回答（edited > chosen > rejected），把 prompt 和多个回答拼成**一行**训练（回答顺序随机打乱），报告称提高效率且不损失准确率。
      `,
      viz: RankPairsViz,
      after: md`
        > [!NOTE] 还有哪些工程细节
        > - RM 通常只训 **1 个 epoch**（InstructGPT、Llama 2 都提到多训会过拟合）；
        > - 分数头的初始化影响训练稳定性，例如 Tülu 3 的 RM 只取 EOS 处的 logit，线性头按 $\mathcal N(0,1/\sqrt{d_{\mathrm{model}}+1})$ 初始化；
        > - RM 的准确率通常只有 60%–80%（人类标注者之间的一致率也不高），所以“RM 准确率 65%”并不一定是 bug；
        > - 评测上常用 RewardBench 一类基准，但 RM 基准分数高不等于下游 RL 效果好（Tülu 3 也提到这一点）。
      `,
      checkpoint: {
        id: 'cp-ranking',
        type: 'fill',
        prompt: md`InstructGPT 中某个 prompt 有 K = 9 个排好序的回答。它产生 ____(1)____ 个成对比较；若把这些比较打散成独立样本，每个回答在一个 epoch 内会被使用 ____(2)____ 次。`,
        blanks: [
          { label: '(1)', answers: ['36'], numeric: true, tolerance: 0 },
          { label: '(2)', answers: ['8'], numeric: true, tolerance: 0 },
        ],
        explanation: md`$\binom92=36$；每个回答与其余 8 个各比较一次。把同一 prompt 的 36 个比较放在一个 batch 元素里，只需 9 次前向。`,
      },
    },
    {
      id: 'prm',
      title: 'ORM vs PRM：给结果打分，还是给每一步打分',
      body: md`
        对数学推理这类多步任务：
        - **ORM**（outcome RM）：对整条解答打一个分，监督来自最终答案对不对；
        - **PRM**（process RM）：对**每个推理步骤**打“这一步正确”的概率。

        OpenAI 的 *Let's Verify Step by Step*（Lightman et al. 2023）发布了 **PRM800K**（80 万个人工步骤级标签）。在 MATH 测试子集上做 best-of-1860 重排：PRM 解对 **78.2%**，ORM 72.4%，多数投票 69.6%；解答得分取**各步正确概率的连乘**。

        人工步骤标注太贵，**Math-Shepherd**（Wang et al. 2023）用自动标注：从某一步出发让“补全模型”继续解 N 次，**硬估计**——只要有一次得到正确答案就标为好步骤；**软估计**——得到正确答案的比例作为标签。
      `,
      viz: PrmViz,
      after: md`
        > [!WARN] DeepSeek-R1 为什么没用 PRM
        > R1 论文（§4.2 Unsuccessful Attempts）列了三点：① 一般推理中很难明确定义“一步”；② 判断中间步骤是否正确本身很难，自动标注效果不佳、人工标注难以扩展；③ 一旦引入基于模型的 PRM，大规模 RL 中**不可避免地出现 reward hacking**，重训 RM 又增加成本和流水线复杂度。
        > 结论是 PRM 适合给 top-N 重排或引导搜索，但在大规模 RL 中收益不抵开销——R1 用的是规则奖励（答案正确性 + 格式）。详见「推理强化学习」章节。
      `,
      checkpoint: {
        id: 'cp-prm',
        type: 'numeric',
        prompt: md`按 Let's Verify 的做法用连乘聚合：一条 3 步解答的步骤正确概率为 0.9、0.8、0.5，它的 PRM 分数是多少？`,
        answer: 0.36,
        tolerance: 0.001,
        explanation: md`$0.9\times0.8\times0.5=0.36$。连乘的含义是“所有步骤都正确”的概率（假设独立），它对任何一步出错都很敏感，也会让步数多的解答系统性得分偏低。`,
      },
    },
    {
      id: 'genrm',
      title: '生成式 RM 与 LLM-as-a-Judge 的偏差',
      body: md`
        判别式 RM 只输出一个标量，不会“想一想再打分”。**生成式 RM** 让 LLM 用文字给出评价：

        - **LLM-as-a-Judge**（Zheng et al. 2023，MT-Bench / Chatbot Arena）：直接提示强模型比较或打分。GPT-4 与人类偏好的一致率超过 80%，与人类之间的一致率相当；
        - **GenRM**（Zhang et al. 2024，*Generative Verifiers*）：把验证当成下一个 token 预测，可以先写思维链再判断，用多次采样多数投票提升准确率；
        - **DeepSeek-GRM**（Liu et al. 2025）：pointwise 生成式 RM，用在线 RL（SPCT）训练模型自己生成评价准则与点评，推理时**并行采样多次 + 投票**，再用 meta RM 引导投票，实现推理时扩展。

        代价：生成式评判贵（每次打分要生成一段文字），而且继承了 LLM 的系统性偏差。Zheng et al. 列出的四类问题：**位置偏差**（偏爱某个位置的回答）、**冗长偏差**（偏爱更长的回答）、**自我增强偏差**（偏爱自己生成的回答）、**有限的推理能力**（数学 / 推理题判错）。
      `,
      viz: JudgeBiasViz,
      after: md`
        > [!INTERVIEW] 怎么缓解 judge 偏差
        > - 位置：交换顺序各评一次，不一致记为平局或取平均；
        > - 长度：评分细则里明确“不因长度加分”，或像 AlpacaEval 2 那样做**长度控制**（LC win rate，用回归去掉长度的影响）；
        > - 自我增强：用与被评模型不同家族的 judge，或多个 judge 集成；
        > - 推理：给参考答案（reference-guided），或让 judge 先独立解题再评判。
        >
        > 在 RLHF / iterative DPO 中，这些偏差会被策略**主动利用**——模型学会写更长、更“像好答案”的回复，这就是 reward hacking。
      `,
      checkpoint: {
        id: 'cp-genrm',
        type: 'multi',
        prompt: md`Zheng et al. 2023（*Judging LLM-as-a-Judge*）指出的 LLM 裁判问题包括：`,
        options: ['位置偏差', '冗长偏差', '自我增强偏差', '有限的推理能力', '无法输出数值分数'],
        answer: [0, 1, 2, 3],
        explanation: md`论文摘要列出 position、verbosity、self-enhancement bias 与 limited reasoning ability，并提出了部分缓解办法。E 不成立：LLM 可以按要求给 1–10 分（MT-Bench 就是单答打分）。`,
      },
    },
    {
      id: 'hacking',
      title: 'Reward hacking 与过度优化',
      body: md`
        RM 只是人类偏好的**代理**。策略越用力优化代理，越可能找到代理与真实目标不一致的地方（Goodhart 定律）。

        Gao, Schulman & Hilton 2022（*Scaling Laws for Reward Model Overoptimization*）用一个固定的“金标准 RM”代替人类，在它的标注上训练代理 RM，再对代理做 best-of-n 或 RL。以 $d=\sqrt{\mathrm{KL}(\pi\|\pi_{\mathrm{init}})}$ 衡量优化强度，金标准分数遵循：

        $$
        R_{\mathrm{bon}}(d)=d(\alpha_{\mathrm{bon}}-\beta_{\mathrm{bon}}d),\qquad R_{\mathrm{RL}}(d)=d(\alpha_{\mathrm{RL}}-\beta_{\mathrm{RL}}\log d)
        $$

        即**先升后降**：代理分数一路上涨，真实分数在某个 KL 之后开始下降；系数随 RM 参数量平滑变化（RM 越大越抗过度优化）。

        常见症状：回复越来越长、出现套话 / 讨好、格式异常、KL 快速增长；对策：KL 惩罚、早停（按金标准或人工评估选 checkpoint）、RM 集成、定期用新策略的样本重训 RM（Llama 2 的迭代做法）、把容易被钻空子的维度拆成独立 RM（Llama 2 的 helpfulness / safety）。
      `,
      after: md`
        > [!NEW] 可验证奖励
        > 数学、代码等任务转向**规则奖励**（答案匹配、单元测试），从根上绕开 RM 的可被利用性，但规则本身也会被钻空子（如格式投机、改测试）。详见「推理强化学习」的训练动力学一关。
      `,
      checkpoint: {
        id: 'cp-hacking',
        type: 'judge',
        prompt: md`按 Gao et al. 2022 的结论，只要代理 RM 的分数在 RL 训练中持续上升，真实（金标准）奖励也一定持续上升。`,
        answer: false,
        explanation: md`真实奖励关于 $d=\sqrt{\mathrm{KL}}$ 先升后降，而代理分数可以继续上升——这正是过度优化。实践中要用独立的评估（人工或更强的 RM）监控并早停。`,
      },
    },
  ],

  quiz: [
    {
      id: 'q-arch',
      type: 'single',
      difficulty: 1,
      prompt: md`InstructGPT 的奖励模型是怎么构造的？`,
      options: [
        '从 SFT 模型出发，去掉最后的 unembedding 层，输出一个标量分数',
        '从头训练一个 BERT 分类器',
        '用 SFT 模型生成“好”或“坏”两个字，取其概率',
        '直接用 SFT 模型对回复的 log-prob 作为奖励',
      ],
      answer: 0,
      explanation: md`论文 §3.5：“Starting from the SFT model with the final unembedding layer removed”，训练成输入 prompt + 回复、输出标量奖励的模型，规模 6B。C 是生成式 RM 的做法；D 没有用到偏好数据。`,
    },
    {
      id: 'q-cross',
      type: 'judge',
      difficulty: 1,
      prompt: md`用 BT 损失训练的 RM，不同 prompt 下回答的分数可以直接比较大小，绝对值有明确含义。`,
      answer: false,
      explanation: md`BT 损失只依赖同一 prompt 内的分差，任意（甚至依赖于 prompt 的）平移都不改变损失。所以绝对分数没有标定，跨 prompt 比较不可靠；这也是 RL 前要做归一化 / whiten 的原因。`,
    },
    {
      id: 'q-bt-num',
      type: 'numeric',
      difficulty: 2,
      prompt: md`一个 batch 有两对偏好：分差 $r_w-r_l$ 分别为 $0$ 和 $\ln3$。不加 margin，平均 BT loss 是多少？（自然对数，保留 3 位小数）`,
      answer: 0.491,
      tolerance: 0.005,
      explanation: md`第一对 $-\log\sigma(0)=\ln2\approx0.693$；第二对 $\sigma(\ln3)=3/4$，$-\log0.75\approx0.288$；平均 $(0.693+0.288)/2\approx0.491$。`,
    },
    {
      id: 'q-grad',
      type: 'fill',
      difficulty: 2,
      prompt: md`BT 损失 $\ell=-\log\sigma(r_w-r_l)$ 中，$\partial\ell/\partial r_w=$ ____(1)____ 。当 $r_w-r_l=\ln3$ 时其数值为 ____(2)____ 。`,
      blanks: [
        { label: '(1)', answers: ['-σ(r_l-r_w)', '-σ(rl-rw)', '-(1-σ(r_w-r_l))', 'σ(r_w-r_l)-1', '-sigmoid(r_l-r_w)', '-σ(-(r_w-r_l))'] },
        { label: '(2)', answers: ['-0.25'], numeric: true, tolerance: 0.001 },
      ],
      explanation: md`$\frac{d}{dz}[-\log\sigma(z)]=-(1-\sigma(z))=-\sigma(-z)$。$z=\ln3$ 时 $\sigma(-z)=1/4$，梯度 $-0.25$；$\partial\ell/\partial r_l=+0.25$。排得越对，梯度越小。`,
    },
    {
      id: 'q-prm-multi',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于 ORM 与 PRM，下列说法正确的有：`,
      options: [
        'PRM 给每个推理步骤打分，能定位第一个错误步骤',
        'Let’s Verify Step by Step 中，best-of-1860 时 PRM 解对 78.2%，高于 ORM 的 72.4%',
        'Math-Shepherd 通过从中间步骤多次补全并检查最终答案来自动生成步骤标签',
        'DeepSeek-R1 在大规模 RL 中使用神经网络 PRM 作为主要奖励',
        'ORM 的训练标签必须由人工逐步标注',
      ],
      answer: [0, 1, 2],
      explanation: md`D 错：R1 用的是规则奖励，并在论文中解释了放弃 PRM 的三个原因。E 错：ORM 只需要最终答案对错，通常可以自动判定；需要逐步标注的是 PRM。`,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 2,
      prompt: md`把 InstructGPT 式 RM 训练与使用的步骤排序：`,
      items: [
        '对同一 prompt 采样 K 个回答，标注员排序',
        '同一 prompt 的全部 C(K,2) 个比较放进同一个 batch 元素',
        '用 BT 损失训练标量头（通常 1 个 epoch）',
        '加偏置让人类示范的平均分为 0（归一化）',
        '作为 PPO 的奖励（并用来初始化 value function）',
      ],
      explanation: md`采集排序 → 组 batch → 训练 → 归一化 → 用于 RL。InstructGPT 的 value function 由 RM 初始化。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把工作与其奖励建模方式配对：`,
      pairs: [
        ['Let’s Verify（PRM800K）', '人工步骤级标注的 PRM'],
        ['Math-Shepherd', '从中间步骤补全 N 次，自动生成步骤标签'],
        ['GenRM（Generative Verifiers）', '把验证当作下一个 token 预测，可用 CoT + 投票'],
        ['DeepSeek-GRM', 'SPCT 训练的 pointwise 生成式 RM，推理时采样投票扩展'],
      ],
      explanation: md`前两者是过程监督（人工 / 自动），后两者是生成式验证器 / 奖励模型。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`
        RM 前向：batch 是 **右侧 padding**，\`attention_mask\` 为 0/1，要取每条序列最后一个有效 token 的隐状态打分。下面代码有 **2 处** bug，点出来。
      `,
      code: py`
        def rm_forward(hidden, attention_mask, w, b):
            # hidden: (B, T, d)，attention_mask: (B, T)
            B, T, d = hidden.shape
            last = hidden[:, -1, :]                     # 取最后一个 token
            scores = last @ w + b                       # (B,)
            return scores

        def rm_loss(s_chosen, s_rejected):
            return np.logaddexp(0.0, s_chosen - s_rejected).mean()
      `,
      bugLines: [4, 9],
      explanation: md`
        第 4 行：右 padding 时位置 $T-1$ 多半是 pad，应取 \`idx = attention_mask.sum(1) - 1\`，\`hidden[np.arange(B), idx]\`（更通用的写法见实战题，可同时处理左右 padding）。
        第 9 行：$-\log\sigma(z)=$ \`logaddexp(0, -z)\`，应为 \`np.logaddexp(0.0, -(s_chosen - s_rejected))\`；现在在惩罚 chosen 得分高。
      `,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出 InstructGPT 式的单个 prompt 的 K 路排序损失（\`scores\` 已按人类排名从好到差排列）。有 2 行干扰项。`,
      lines: [
        'def k_way_loss(scores):',
        '    K = len(scores)',
        '    total, n = 0.0, 0',
        '    for i in range(K):',
        '        for j in range(i + 1, K):',
        '            total += np.logaddexp(0.0, -(scores[i] - scores[j]))',
        '            n += 1',
        '    return total / n',
      ],
      distractors: ['        for j in range(K):', '            total += np.logaddexp(0.0, scores[i] - scores[j])'],
      explanation: md`只对“i 排在 j 前面”的 $\binom K2$ 对求和并取平均。干扰项一个会把 $i=j$ 与反向对也算进去，一个把符号写反。`,
    },
    {
      id: 'q-pick-hack',
      type: 'pick',
      difficulty: 3,
      prompt: md`PPO-RLHF 训练中观察到以下现象，哪些是 reward hacking / 过度优化的典型信号？`,
      items: [
        '代理 RM 分数持续上升，人工评估胜率开始下降',
        '平均回复长度持续增长',
        'KL(π‖π_ref) 快速增大',
        '回复中出现大量重复的客套与讨好语句',
        '验证集上 RM 的偏好准确率保持稳定',
        '学习率 warmup 阶段 loss 波动',
      ],
      answer: [0, 1, 2, 3],
      explanation: md`前四项都是策略在钻代理奖励空子的表现（长度与讨好是 RM / judge 的常见偏差）。E 说明 RM 在**旧分布**上依然准确，恰恰不能说明它在策略的新分布上可靠；F 是正常的优化现象。`,
    },
    {
      id: 'q-overopt',
      type: 'single',
      difficulty: 3,
      prompt: md`Gao et al. 2022 用 $d=\sqrt{\mathrm{KL}}$ 刻画优化强度。对 RL，金标准 RM 分数的函数形式是：`,
      options: [
        md`$R_{\mathrm{RL}}(d)=d(\alpha-\beta\log d)$：先升后降`,
        md`$R_{\mathrm{RL}}(d)=\alpha d$：单调上升`,
        md`$R_{\mathrm{RL}}(d)=\alpha(1-e^{-\beta d})$：饱和但不下降`,
        md`$R_{\mathrm{RL}}(d)=d(\alpha-\beta d)$：与 best-of-n 完全相同`,
      ],
      answer: 0,
      explanation: md`论文给出 $R_{\mathrm{bon}}(d)=d(\alpha_{\mathrm{bon}}-\beta_{\mathrm{bon}}d)$ 与 $R_{\mathrm{RL}}(d)=d(\alpha_{\mathrm{RL}}-\beta_{\mathrm{RL}}\log d)$，两者都会在足够大的 $d$ 之后下降，但形式不同（D 错）。B、C 都否认了过度优化。`,
    },
    {
      id: 'q-margin',
      type: 'judge',
      difficulty: 2,
      prompt: md`Llama 3 沿用了 Llama 2 RM 损失中的 margin 项。`,
      answer: false,
      explanation: md`Llama 3 §4.1.2：训练目标与 Llama 2 相同，**但去掉了 margin 项**，因为数据规模扩大后收益递减。`,
    },
  ],

  guided: [
    {
      id: 'g-bt',
      title: '一步步写出 BT loss 与梯度',
      intro: md`
        实现 \`bt_loss_and_grad(r_c, r_r, margin)\`：输入 chosen / rejected 的 RM 分数（形状 \`(B,)\`）与 margin（\`(B,)\`，可全 0），
        返回 batch 平均 loss 以及它对 \`r_c\`、\`r_r\` 的梯度。要求数值稳定（分差可能上千）。
      `,
      template: py`
        import numpy as np

        def bt_loss_and_grad(r_c, r_r, margin):
            B = len(r_c)
            z = {{1}}
            loss = {{2}}
            w = {{3}}
            g_c = {{4}}
            g_r = {{5}}
            return float(loss), g_c, g_r
      `,
      steps: [
        {
          blank: 1,
          instruction: md`logit：分差减去 margin。`,
          answer: 'r_c - r_r - margin',
          explain: md`$z=r_w-r_l-m$；Llama 2 的 margin 只在 σ 里平移。`,
        },
        {
          blank: 2,
          instruction: md`稳定的 $-\log\sigma(z)$ 的 batch 平均。`,
          answer: 'np.logaddexp(0.0, -z).mean()',
          accept: ['np.mean(np.logaddexp(0.0, -z))', 'np.logaddexp(0, -z).mean()'],
          explain: md`$-\log\sigma(z)=\log(1+e^{-z})$；\`-np.log(1/(1+np.exp(-z)))\` 在 $z=-1000$ 时溢出。`,
        },
        {
          blank: 3,
          instruction: md`稳定地计算 $\sigma(-z)$（纠错权重）。`,
          answer: 'np.exp(-np.logaddexp(0.0, z))',
          accept: ['np.exp(-np.logaddexp(0, z))'],
          explain: md`$\sigma(-z)=\frac{1}{1+e^{z}}=\exp(-\log(1+e^{z}))$，任何 $z$ 都不溢出。`,
        },
        {
          blank: 4,
          instruction: md`对 r_c 的梯度（别忘了 batch 平均的 1/B）。`,
          answer: '-w / B',
          explain: md`$\partial\ell/\partial z=-\sigma(-z)$，$\partial z/\partial r_c=1$。`,
        },
        {
          blank: 5,
          instruction: md`对 r_r 的梯度。`,
          answer: 'w / B',
          explain: md`$\partial z/\partial r_r=-1$，符号相反、大小相同。这正是 RM 训练“把 chosen 往上推、把 rejected 往下压”的方式。`,
        },
      ],
      tests: [
        {
          name: '手算示例',
          code: py`
            loss, gc, gr = bt_loss_and_grad(np.array([1.0, 2.0]), np.array([1.0, 2.0 - np.log(3)]), np.zeros(2))
            exp = (np.log(2) - np.log(0.75)) / 2
            assert abs(loss - exp) < 1e-9, f"期望 {exp:.6f}，得到 {loss}"
            assert np.allclose(gc, [-0.25, -0.125]) and np.allclose(gr, [0.25, 0.125]), f"梯度错：{gc}, {gr}"
          `,
        },
        {
          name: '数值梯度检查 + 大分差稳定',
          code: py`
            rng = np.random.default_rng(0)
            rc, rr, m = rng.normal(size=5), rng.normal(size=5), rng.uniform(0, 1, size=5)
            loss, gc, gr = bt_loss_and_grad(rc, rr, m)
            eps = 1e-6
            for k in range(5):
                e = np.zeros(5); e[k] = eps
                num = (bt_loss_and_grad(rc + e, rr, m)[0] - bt_loss_and_grad(rc - e, rr, m)[0]) / (2 * eps)
                assert abs(num - gc[k]) < 1e-6, f"g_c[{k}]：数值 {num:.6f}，解析 {gc[k]:.6f}"
            big = bt_loss_and_grad(np.array([1000.0, -1000.0]), np.array([0.0, 0.0]), np.zeros(2))
            assert np.isfinite(big[0]) and abs(big[0] - 500.0) < 1e-6, f"大分差：期望 500，得到 {big[0]}"
            assert np.all(np.isfinite(big[1])), "梯度出现 inf/nan"
          `,
        },
      ],
      outro: md`
        PyTorch 一行：\`loss = -F.logsigmoid(r_c - r_r - margin).mean()\`，梯度交给 autograd。
        手写一遍的意义在于记住 $\sigma(-z)$ 这个权重——DPO、SimPO、KTO 的梯度里都有它的影子。
      `,
    },
  ],

  challenges: [
    {
      id: 'c-rm-head',
      title: 'RM 前向：取最后一个有效 token 打分（兼容左右 padding）',
      difficulty: 'medium',
      source: '大模型岗手撕 · 奖励模型',
      timeLimitMin: 20,
      description: md`
        \`\`\`python
        def rm_scores(hidden: np.ndarray, attention_mask: np.ndarray, w: np.ndarray, b: float) -> np.ndarray: ...
        def rm_pair_loss(hidden, attention_mask, w, b) -> tuple[float, float]: ...
        \`\`\`

        - \`hidden\`：\`(N, T, d)\` 最后一层隐状态；\`attention_mask\`：\`(N, T)\` 0/1，**可能是左 padding、也可能是右 padding**（同一 batch 内一致，但你不知道是哪种）；
        - \`rm_scores\`：取每行**最后一个 mask 为 1 的位置**的隐状态，返回 \`hidden[i, last_i] @ w + b\`，形状 \`(N,)\`；
        - \`rm_pair_loss\`：batch 前一半是 chosen、后一半是 rejected（第 $i$ 行与第 $N/2+i$ 行是一对），返回 \`(loss, acc)\`：平均 BT loss 与 chosen 得分**严格高于** rejected 的比例。

        要求：不要对 batch 维写 Python 循环；loss 数值稳定。
      `,
      starter: py`
        import numpy as np

        def rm_scores(hidden, attention_mask, w, b):
            # TODO: 找每行最后一个有效位置（注意：左 padding 时 mask.sum() - 1 是错的）
            raise NotImplementedError

        def rm_pair_loss(hidden, attention_mask, w, b):
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def rm_scores(hidden, attention_mask, w, b):
            N, T, _ = hidden.shape
            m = np.asarray(attention_mask) > 0
            # 最后一个 1 的位置 = T-1 - (翻转后第一个 1 的位置)；左右 padding 都适用
            last = T - 1 - np.argmax(m[:, ::-1], axis=1)
            h = hidden[np.arange(N), last]               # (N, d)
            return h @ w + b

        def rm_pair_loss(hidden, attention_mask, w, b):
            s = rm_scores(hidden, attention_mask, w, b)
            half = len(s) // 2
            z = s[:half] - s[half:]
            loss = float(np.logaddexp(0.0, -z).mean())   # -log σ(z)
            acc = float((z > 0).mean())
            return loss, acc
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示，HF 序列分类头的做法）',
          code: py`
            import torch
            import torch.nn.functional as F

            def rm_scores(hidden, attention_mask, score_head):
                # 与 transformers 的 *ForSequenceClassification 类似：找最后一个非 pad 的位置
                T = attention_mask.shape[1]
                idx = torch.arange(T, device=hidden.device)
                last = (idx * attention_mask).argmax(-1)          # 左右 padding 都适用
                h = hidden[torch.arange(hidden.shape[0]), last]
                return score_head(h).squeeze(-1)                  # nn.Linear(d, 1)

            def rm_pair_loss(scores):
                half = scores.shape[0] // 2
                z = scores[:half] - scores[half:]
                return -F.logsigmoid(z).mean(), (z > 0).float().mean()
          `,
        },
      ],
      explanation: md`
        **陷阱**：\`attention_mask.sum(1) - 1\` 只对右 padding 成立；左 padding（生成时常用）的最后一个有效 token 永远在 $T-1$。通用写法：
        - 翻转后 \`argmax\` 找第一个 1：\`T - 1 - argmax(mask[:, ::-1])\`；或
        - \`argmax(arange(T) * mask)\`（位置编号乘 mask 后取最大，HF 新版序列分类头即类似思路）。

        **为什么取最后一个 token**：因果模型中只有最后一个位置看过完整的 prompt + 回复；也有实现在回复末尾显式加 EOS 并取 EOS 位置（Tülu 3 即取 EOS 处的 logit）。

        **复杂度**：$O(N\cdot d)$（不含 backbone）。
      `,
      complexity: { time: 'O(N·d)', space: 'O(N·d)' },
      hints: [
        md`对每行找最后一个 1：\`T - 1 - np.argmax(mask[:, ::-1], axis=1)\`。`,
        md`用花式索引 \`hidden[np.arange(N), last]\` 取出 \`(N, d)\`。`,
        md`\`z = s[:half] - s[half:]\`，loss 用 \`np.logaddexp(0, -z)\`，acc 用 \`(z > 0).mean()\`。`,
      ],
      setup: py`
        import numpy as np

        def _ref_scores(hidden, mask, w, b):
            out = []
            for i in range(hidden.shape[0]):
                idx = max(t for t in range(mask.shape[1]) if mask[i, t] > 0)
                out.append(hidden[i, idx] @ w + b)
            return np.array(out)

        def _make_rm(seed, N=6, T=7, d=5, side="right"):
            rng = np.random.default_rng(seed)
            lens = rng.integers(1, T + 1, size=N)
            ar = np.arange(T)[None, :]
            mask = (ar < lens[:, None]) if side == "right" else (ar >= T - lens[:, None])
            hidden = rng.normal(size=(N, T, d))
            return hidden, mask.astype(np.int64), rng.normal(size=d), float(rng.normal())
      `,
      tests: [
        {
          name: '右 padding',
          code: py`
            h, m, w, b = _make_rm(0, side="right")
            got = rm_scores(h, m, w, b)
            assert got.shape == (6,), f"形状应为 (6,)，得到 {got.shape}"
            assert np.allclose(got, _ref_scores(h, m, w, b)), "右 padding 下取错了位置"
          `,
        },
        {
          name: '左 padding',
          code: py`
            h, m, w, b = _make_rm(1, side="left")
            assert np.allclose(rm_scores(h, m, w, b), _ref_scores(h, m, w, b)), "左 padding 下取错了位置——mask.sum()-1 只适用于右 padding"
          `,
        },
        {
          name: 'pair loss 与准确率',
          code: py`
            h, m, w, b = _make_rm(2, N=8)
            s = _ref_scores(h, m, w, b); z = s[:4] - s[4:]
            loss, acc = rm_pair_loss(h, m, w, b)
            assert abs(loss - np.logaddexp(0, -z).mean()) < 1e-9, f"loss 期望 {np.logaddexp(0, -z).mean():.6f}，得到 {loss}"
            assert abs(acc - (z > 0).mean()) < 1e-12, f"acc 期望 {(z > 0).mean()}，得到 {acc}"
          `,
        },
        {
          name: '随机对拍（两种 padding、无 padding）',
          hidden: true,
          code: py`
            for s in range(10, 20):
                for side in ("left", "right"):
                    h, m, w, b = _make_rm(s, N=10, T=9, d=4, side=side)
                    if s == 10:
                        m[:] = 1
                    assert np.allclose(rm_scores(h, m, w, b), _ref_scores(h, m, w, b)), f"seed={s} side={side}"
          `,
        },
        {
          name: '数值稳定：分差上千',
          hidden: true,
          code: py`
            h, m, w, b = _make_rm(30, N=4, T=5, d=3)
            loss, acc = rm_pair_loss(h * 2000.0, m, w, b)
            s = _ref_scores(h * 2000.0, m, w, b); z = s[:2] - s[2:]
            assert np.isfinite(loss) and abs(loss - np.logaddexp(0, -z).mean()) < 1e-6 * max(1, abs(loss)), f"loss = {loss}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-rm-init',
          q: md`RM 从 SFT 模型初始化还是从预训练模型初始化？各有什么考虑？`,
          a: md`
            InstructGPT 从 SFT 模型初始化；Llama 2 从 chat 模型 checkpoint 初始化，理由是 RM 与策略“知道的东西一致”，避免信息不对称导致偏好幻觉；Llama 3 从预训练 checkpoint 初始化。经验上，RM 与策略同源、规模不小于策略时更不容易被 hack；InstructGPT 则为了省算力和稳定性只用 6B RM。
          `,
        },
      ],
    },
    {
      id: 'c-rank-loss',
      title: '多 prompt 的 K 路排序损失（InstructGPT 式）',
      difficulty: 'medium',
      source: '大模型岗手撕 · 奖励模型',
      timeLimitMin: 20,
      description: md`
        一个 batch 中混有多个 prompt 的回答：

        \`\`\`python
        def ranking_loss(scores: np.ndarray, prompt_ids: np.ndarray, ranks: np.ndarray) -> tuple[float, float]: ...
        \`\`\`

        - \`scores\`：\`(N,)\` RM 分数；\`prompt_ids\`：\`(N,)\` 所属 prompt（**不保证连续、不保证有序**）；\`ranks\`：\`(N,)\` 人类排名，**越小越好，可能有并列**；
        - 对每个 prompt，取所有“排名严格更好”的有序对 $(i, j)$（$\mathrm{rank}_i<\mathrm{rank}_j$），该 prompt 的 loss 为这些对的 $-\log\sigma(s_i-s_j)$ 的**平均**；
        - 返回 \`(loss, acc)\`：\`loss\` 是**各 prompt loss 的平均**（每个 prompt 等权，对应 InstructGPT 的 $1/\binom K2$ 归一化）；\`acc\` 是所有有效对中 $s_i>s_j$ 的比例；
        - 没有任何有效对的 prompt（只有 1 个回答或全部并列）不参与平均；若整个 batch 都没有有效对，返回 \`(0.0, 0.0)\`。
      `,
      starter: py`
        import numpy as np

        def ranking_loss(scores, prompt_ids, ranks):
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def ranking_loss(scores, prompt_ids, ranks):
            scores, prompt_ids, ranks = map(np.asarray, (scores, prompt_ids, ranks))
            per_prompt, correct, total = [], 0, 0
            for p in np.unique(prompt_ids):
                idx = np.flatnonzero(prompt_ids == p)
                s, r = scores[idx], ranks[idx]
                better = r[:, None] < r[None, :]                  # (k, k)：i 严格优于 j
                if not better.any():
                    continue
                z = (s[:, None] - s[None, :])[better]
                per_prompt.append(np.logaddexp(0.0, -z).mean())   # 该 prompt 内对所有有效对取平均
                correct += int((z > 0).sum())
                total += z.size
            if not per_prompt:
                return 0.0, 0.0
            return float(np.mean(per_prompt)), correct / total
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示，按 prompt 分组的向量化写法）',
          code: py`
            import torch
            import torch.nn.functional as F

            def ranking_loss(scores, prompt_ids, ranks):
                same = prompt_ids[:, None] == prompt_ids[None, :]
                better = same & (ranks[:, None] < ranks[None, :])          # (N, N)
                z = scores[:, None] - scores[None, :]
                pair_loss = -F.logsigmoid(z) * better
                n_pairs = better.sum(1).float()                            # 每行（每个回答）参与的有效对数
                # 按 prompt 求和：用 index_add 把每行的贡献加到所属 prompt
                uniq, inv = torch.unique(prompt_ids, return_inverse=True)
                num = torch.zeros(len(uniq)).index_add_(0, inv, pair_loss.sum(1))
                cnt = torch.zeros(len(uniq)).index_add_(0, inv, n_pairs)
                valid = cnt > 0
                loss = (num[valid] / cnt[valid]).mean()
                acc = ((z > 0) & better).sum() / better.sum().clamp(min=1)
                return loss, acc
          `,
        },
      ],
      explanation: md`
        **要点**：
        1. 分组：prompt_ids 乱序，用 \`np.unique\` + 掩码，或排序后切片；
        2. 有序对：\`better = r[:, None] < r[None, :]\` 自动排除并列与自身；
        3. 归一化口径：先在 prompt 内对 $\binom K2$ 个对平均、再在 prompt 间平均——每个 prompt 等权；若直接对所有对平均，回答多的 prompt 会主导（K = 9 有 36 对，K = 4 只有 6 对）；
        4. acc 按“对”统计，是 RM 评测中最常报的指标。

        **复杂度**：$O(\sum_p K_p^2)$。
      `,
      complexity: { time: 'O(Σ K_p²)', space: 'O(max K_p²)' },
      hints: [
        md`对每个 \`p in np.unique(prompt_ids)\` 取出 \`s, r\`。`,
        md`\`better = r[:, None] < r[None, :]\`；\`z = (s[:, None] - s[None, :])[better]\`。`,
        md`prompt 内 \`logaddexp(0, -z).mean()\`，再对 prompt 取平均；acc 按对累计。`,
      ],
      setup: py`
        import numpy as np

        def _ref_rank(scores, pids, ranks):
            groups = {}
            for s, p, r in zip(scores, pids, ranks):
                groups.setdefault(p, []).append((s, r))
            per, c, t = [], 0, 0
            for items in groups.values():
                ls = []
                for si, ri in items:
                    for sj, rj in items:
                        if ri < rj:
                            ls.append(float(np.logaddexp(0.0, -(si - sj))))
                            c += si > sj; t += 1
                if ls:
                    per.append(np.mean(ls))
            return (float(np.mean(per)), c / t) if per else (0.0, 0.0)

        def _make_rank(seed, P=4):
            rng = np.random.default_rng(seed)
            s, pid, rk = [], [], []
            for p in rng.permutation(100)[:P]:
                K = int(rng.integers(1, 10))
                s += list(rng.normal(size=K)); pid += [int(p)] * K; rk += list(rng.integers(0, max(1, K - 1), size=K))
            perm = rng.permutation(len(s))
            return np.array(s)[perm], np.array(pid)[perm], np.array(rk)[perm]
      `,
      tests: [
        {
          name: '单个 prompt、K = 3',
          code: py`
            loss, acc = ranking_loss(np.array([2.0, 1.0, 0.0]), np.array([7, 7, 7]), np.array([0, 1, 2]))
            exp = np.mean([np.logaddexp(0, -1.0), np.logaddexp(0, -2.0), np.logaddexp(0, -1.0)])
            assert abs(loss - exp) < 1e-9 and acc == 1.0, f"期望 ({exp:.6f}, 1.0)，得到 ({loss}, {acc})"
          `,
        },
        {
          name: '每个 prompt 等权（不是每个对等权）',
          code: py`
            # prompt 0：K=4（6 对，全排对、loss 小）；prompt 1：K=2（1 对，排反、loss 大）
            s = np.array([3.0, 2.0, 1.0, 0.0, -1.0, 1.0]); pid = np.array([0, 0, 0, 0, 1, 1]); rk = np.array([0, 1, 2, 3, 0, 1])
            loss, acc = ranking_loss(s, pid, rk)
            exp, eacc = _ref_rank(s, pid, rk)
            assert abs(loss - exp) < 1e-9, f"期望 {exp:.6f}（两个 prompt 等权），得到 {loss:.6f}"
            assert abs(acc - 6 / 7) < 1e-12, f"acc 期望 6/7，得到 {acc}"
          `,
        },
        {
          name: '并列与单回答的 prompt 被跳过',
          code: py`
            s = np.array([1.0, 0.5, 2.0, 0.3]); pid = np.array([1, 1, 2, 3]); rk = np.array([0, 0, 0, 5])
            assert ranking_loss(s, pid, rk) == (0.0, 0.0), "没有有效对时应返回 (0.0, 0.0)"
          `,
        },
        {
          name: '随机对拍（乱序 prompt_ids、并列）',
          hidden: true,
          code: py`
            for seed in range(30):
                s, pid, rk = _make_rank(seed)
                got, exp = ranking_loss(s, pid, rk), _ref_rank(s, pid, rk)
                assert abs(got[0] - exp[0]) < 1e-9 and abs(got[1] - exp[1]) < 1e-12, f"seed={seed}：期望 {exp}，得到 {got}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-pl',
          q: md`除了拆成成对比较，K 个回答的完整排序还能怎么建模？`,
          a: md`
            用 **Plackett-Luce** 模型：排序概率 $\prod_{k=1}^{K}\frac{e^{r(y_{\tau(k)})}}{\sum_{j\ge k}e^{r(y_{\tau(j)})}}$，即依次从剩余回答中“选出最好的”。K = 2 时退化为 BT。DPO 论文附录也给出了 Plackett-Luce 下的 DPO 形式。成对拆分更简单、对并列和部分排序更友好，是工程上的主流。
          `,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-bt',
      freq: 3,
      q: md`奖励模型怎么训练？请推导 Bradley-Terry 损失。`,
      a: md`
        **结论**：$\mathcal L=-\mathbb E\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))$。

        **推导**：BT 假设 $P(y_w\succ y_l)=\frac{e^{r_w}}{e^{r_w}+e^{r_l}}=\sigma(r_w-r_l)$，对偏好数据做最大似然即得。梯度 $\partial\ell/\partial r_w=-\sigma(r_l-r_w)$：排错的对权重大。结构上用 SFT/预训练模型去掉 LM head，换成标量头，取最后一个 token（EOS）打分。

        **延伸**：K 路排序拆成 $\binom K2$ 对并在 prompt 内平均（InstructGPT，K = 4–9，同 prompt 放同一 batch 元素防过拟合）；Llama 2 加 margin $m(r)$，Llama 3 去掉；分数平移不变，RL 前要归一化；RM 一般只训 1 个 epoch。
      `,
    },
    {
      id: 'qa-prm',
      freq: 2,
      q: md`PRM 和 ORM 有什么区别？为什么 DeepSeek-R1 没有用 PRM？`,
      a: md`
        **结论**：ORM 对整条回答打分，PRM 对每个推理步骤打分；PRM 提供更密的信号和错误定位，但标注与使用成本高、易被 hack。

        **原因**：Let's Verify 的 best-of-1860 实验中 PRM（78.2%）优于 ORM（72.4%）与多数投票（69.6%），说明过程监督对重排有效；但 PRM800K 需要 80 万个人工步骤标签，Math-Shepherd 用补全 rollout 自动标注又很贵且有噪声。R1 列出的三个原因：步骤难以定义、步骤正确性难以判断、模型化的 PRM 在大规模 RL 中必然被 reward hack 且重训开销大。

        **延伸**：PRM 仍适合 best-of-N 重排、树搜索引导；RL 主线转向规则可验证奖励。
      `,
    },
    {
      id: 'qa-judge',
      freq: 2,
      q: md`用 LLM-as-a-Judge 做奖励或评测有哪些偏差？怎么缓解？`,
      a: md`
        **结论**：位置偏差、冗长偏差、自我增强偏差、推理能力有限（Zheng et al. 2023）。

        **缓解**：交换位置各评一次、不一致判平局；评分细则明确长度无关，或做长度控制（AlpacaEval 2 LC）；用不同模型家族的 judge 或多 judge 集成；给参考答案或让 judge 先自己解题；生成式 RM 可以先写 CoT 再判断并多次采样投票（GenRM、DeepSeek-GRM）。

        **延伸**：作为训练奖励时这些偏差会被策略放大（变长、讨好），需要配合 KL 约束、长度惩罚与人工抽检。GPT-4 judge 与人类的一致率约 80%，与人类之间的一致率相当——它是可用的代理，但仍是代理。
      `,
    },
    {
      id: 'qa-hacking',
      freq: 2,
      q: md`什么是 reward hacking？RLHF 里怎么防？`,
      a: md`
        **结论**：策略利用代理奖励的缺陷获得高分，而真实质量不升反降（Goodhart 定律）。

        **证据**：Gao et al. 2022 发现金标准 RM 分数关于 $d=\sqrt{\mathrm{KL}}$ 先升后降（RL：$d(\alpha-\beta\log d)$），代理分数则持续上升；RM 越大越抗过度优化。

        **对策**：KL 惩罚（逐 token）；按独立评估早停；RM 集成、更大的 RM；用新策略的样本迭代更新 RM（Llama 2）；拆分 helpfulness / safety RM；长度惩罚与格式检查；对可验证任务直接用规则奖励。
      `,
    },
  ],

  cheatsheet: md`
    $$P(y_w\succ y_l)=\sigma(r_w-r_l),\quad \mathcal L=-\log\sigma(r_w-r_l-m),\quad \partial\ell/\partial r_w=-\sigma(m+r_l-r_w)$$

    \`\`\`python
    last = T - 1 - np.argmax(mask[:, ::-1], axis=1)   # 左右 padding 通用
    s = hidden[np.arange(N), last] @ w + b
    z = s[:B] - s[B:]
    loss = np.logaddexp(0, -z).mean(); acc = (z > 0).mean()
    \`\`\`

    - InstructGPT：6B RM，K = 4–9，$\binom K2$ 对放同一 batch 元素，1/C(K,2) 归一化，RL 前加偏置使示范均分为 0
    - Llama 2：helpfulness + safety 两个 RM；margin Small 1 / 2/3 / 1/3 / 0，Large 3 / 2 / 1 / 0；Llama 3 去掉 margin
    - PRM800K：80 万步骤标签；best-of-1860：PRM 78.2% · ORM 72.4% · 多数投票 69.6%；连乘聚合
    - Math-Shepherd：补全 N 次，硬估计（任一正确）/ 软估计（正确比例）
    - LLM judge 偏差：位置 · 冗长 · 自我增强 · 推理有限；GPT-4 与人类一致率 > 80%
    - 过度优化：$R_{\mathrm{bon}}=d(\alpha-\beta d)$，$R_{\mathrm{RL}}=d(\alpha-\beta\log d)$，$d=\sqrt{\mathrm{KL}}$
  `,
})
