/**
 * llm-rl-opd：On-policy 蒸馏（Thinking Machines 2025-10、DeepSeek-V4 合并专家）与 Agentic RL
 */
import { defineLevel, md, py } from '@/content/helpers'
import { KLModeViz } from '../viz/KLModeViz'
import { OPDTokenViz } from '../viz/OPDTokenViz'
import { AgentMaskViz } from '../viz/AgentMaskViz'
import { PY_REF_ADV } from '../shared'

const SETUP =
  PY_REF_ADV +
  '\n\n' +
  py`
  def _lsm(x):
      x = np.asarray(x, dtype=np.float64)
      m = x.max(axis=-1, keepdims=True)
      return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))

  def _ref_opd(student_logits, teacher_logits, tokens, mask):
      ls, lt = _lsm(student_logits), _lsm(teacher_logits)
      m = np.asarray(mask, dtype=np.float64)
      full = (np.exp(ls) * (ls - lt)).sum(axis=-1) * m
      tok = np.asarray(tokens)[..., None]
      sampled = (np.take_along_axis(ls, tok, -1) - np.take_along_axis(lt, tok, -1))[..., 0] * m
      return {"full_kl": full, "sampled_kl": sampled, "adv": -sampled, "mean_full_kl": float(full.sum() / m.sum())}

  def _opd_batch(seed, B=3, T=6, V=11, scale=2.0, offset=0.0):
      rng = np.random.default_rng(seed)
      s = rng.normal(0, scale, size=(B, T, V)) + offset
      t = s + rng.normal(0, 1.0, size=(B, T, V))
      lens = rng.integers(1, T + 1, size=B)
      mask = (np.arange(T)[None, :] < lens[:, None]).astype(np.float64)
      tokens = rng.integers(0, V, size=(B, T))
      return s, t, tokens, mask

  def _ref_agentic_loss(logp_new, logp_old, role_ids, rewards, group_ids, clip_eps=0.2):
      lm = (np.asarray(role_ids) == 2).astype(np.float64)
      A = _ref_group_adv(rewards, group_ids)[:, None]
      ratio = np.exp(np.where(lm > 0, logp_new - logp_old, 0.0))
      tok = -np.minimum(ratio * A, np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A)
      return float((tok * lm).sum() / lm.sum()), lm > 0

  def _agent_batch(seed, n_groups=2, G=4, T=24):
      rng = np.random.default_rng(seed)
      B = n_groups * G
      roles = np.zeros((B, T), dtype=np.int64)
      for i in range(B):
          pos = 3
          roles[i, :3] = 1                                  # prompt
          turns = int(rng.integers(1, 4))
          for k in range(turns):
              a = int(rng.integers(2, 4)); roles[i, pos:pos + a] = 2; pos += a       # 模型：思考 + 工具调用
              tl = int(rng.integers(2, 5)); roles[i, pos:pos + tl] = 3; pos += tl     # 工具返回
          a = int(rng.integers(1, 3)); roles[i, pos:pos + a] = 2                     # 最终回答
      logp_old = -rng.exponential(1.0, size=(B, T)) - 0.05
      logp_new = logp_old + rng.normal(0, 0.3, size=(B, T))
      other = roles != 2
      logp_new[other] = rng.uniform(-4, 4, size=other.sum())
      logp_old[other] = rng.uniform(-4, 4, size=other.sum())
      rewards = rng.integers(0, 2, size=B).astype(np.float64)
      rewards[:G] = [1, 0, 0, 1][:G]
      return logp_new, logp_old, roles, rewards, np.repeat(np.arange(n_groups), G)
`

export default defineLevel({
  learn: [
    {
      id: 'three-ways',
      title: '三种后训练信号：off-policy 蒸馏、RL、on-policy 蒸馏',
      body: md`
        Thinking Machines 的博客「On-Policy Distillation」（Kevin Lu 等，2025-10-27）用两个维度把后训练方法摆在一张表里：
        **样本从哪来**（教师 / 外部数据 vs 学生自己）与**信号有多密**（逐 token vs 整条序列一个数）。

        | | 稀疏奖励（序列级） | 稠密奖励（逐 token） |
        |---|---|---|
        | **off-policy**（样本来自教师 / 数据集） | — | SFT / 传统蒸馏：在教师的轨迹上模仿 |
        | **on-policy**（样本来自学生） | RL：自己采样，只知道最后对不对 | **on-policy 蒸馏**：自己采样，教师给每个 token 打分 |

        - **off-policy 蒸馏**的问题：学生只在教师走过的状态上学习，推理时一旦走偏进入教师没走过的前缀，误差会**累积**（compounding error / exposure bias）。
        - **RL** 是 on-policy 的，但每条几千 token 的回答只换来 1 个比特左右的信息，样本效率低。
        - **on-policy 蒸馏**兼得：在学生自己的轨迹上学习（无分布偏移），每个 token 都有教师的反馈（稠密）。

        学术源头：GKD（Agarwal 等，ICLR 2024，学生自生成序列上的广义 JSD 蒸馏）与 MiniLLM（Gu 等，ICLR 2024，用反向 KL 蒸馏 LLM）。
        Qwen3 技术报告也用它训练小模型，博客转引的数字（Qwen3-8B）：off-policy 蒸馏 AIME24 55.0% → 再做 RL 67.6%（17,920 GPU 小时）；
        改做 on-policy 蒸馏 74.4%，只用 1,800 GPU 小时。
      `,
      checkpoint: {
        id: 'cp-three',
        type: 'single',
        prompt: md`on-policy 蒸馏与 off-policy 蒸馏（在教师生成的数据上 SFT）的根本区别是：`,
        options: ['使用的教师模型更大', '训练样本由学生自己采样，教师只对学生的 token 打分', '只能用于数学任务', '不需要教师模型的 logits'],
        answer: 1,
        explanation: md`on-policy = 学生采样；教师只做一次前向给出对数概率。这样学生在自己真正会走到的状态上学习，避免 exposure bias。D 说反了：恰恰需要教师在学生轨迹上的对数概率。`,
      },
    },
    {
      id: 'kl-direction',
      title: '为什么是反向 KL：mode-seeking',
      body: md`
        蒸馏要缩小学生 $q=\pi_\theta$ 与教师 $p=\pi_{T}$ 的距离，方向有两种：

        $$
        \underbrace{\mathrm{KL}(p\Vert q)=\mathbb E_{y\sim p}\Big[\log\frac{p}{q}\Big]}_{\text{正向：SFT / 传统 KD}}\qquad
        \underbrace{\mathrm{KL}(q\Vert p)=\mathbb E_{y\sim q}\Big[\log\frac{q}{p}\Big]}_{\text{反向：on-policy 蒸馏}}
        $$

        - **正向 KL** 在教师有概率的地方惩罚学生没概率（$q\to0$ 时爆炸）→ 学生被迫“盖住”教师的所有模式，容量不够时会把概率放在模式之间的谷底，生成教师从不会说的话；
        - **反向 KL** 在学生有概率而教师没概率的地方重罚 → 学生选一个模式做好（mode-seeking），样本质量高；而且期望在**学生分布**下取，天然就是 on-policy 的。

        Thinking Machines 还强调反向 KL “不可 hack”：低 KL 总是对应教师认可的行为，不像奖励模型可能被钻空子。
      `,
      viz: KLModeViz,
      checkpoint: {
        id: 'cp-dir',
        type: 'judge',
        prompt: md`学生容量不足时，最小化反向 KL(q‖p) 的学生倾向于把概率质量摊开、覆盖教师的所有模式。`,
        answer: false,
        explanation: md`那是**正向** KL 的行为（mode-covering）。反向 KL 对“学生有、教师无”的区域重罚，倾向锁定一个模式（mode-seeking）。`,
      },
    },
    {
      id: 'opd-algo',
      title: 'On-policy 蒸馏 = 把 RL 的奖励换成逐 token 的负反向 KL',
      body: md`
        Thinking Machines 的实现只是对 RL 代码的“一行改动”：

        1. 学生采样轨迹（和 RL 一样），记录学生的 logprob；
        2. 教师对同一条轨迹做一次前向（\`compute_logprobs\`），得到每个采样 token 的 $\log\pi_T(y_t)$；
        3. 每个 token 的**优势** $=-\big(\log\pi_\theta(y_t)-\log\pi_T(y_t)\big)$，即逐 token 反向 KL 的单样本估计取负；**折扣因子取 0**（只优化下一个 token，他们发现更大的折扣在实践中没有帮助）；
        4. 用 RL 的 importance-sampling loss 训练。

        如果原来的 RL 代码带 KL 正则，等价于“把正则的参考模型换成教师、去掉环境奖励”。

        **博客中的结果**（学生 Qwen3-8B-Base）：
        - 从 400K 条 SFT 数据的 checkpoint（AIME24 约 60%）出发，on-policy 蒸馏约 **150 步**到 **70%**（约 77K 个 prompt、每个 4 个样本）；
          与外推“SFT 到 200 万条才能到 70%”相比，算力便宜约 **9–30 倍**；
        - **个性化 / 持续学习**：用内部文档做中训练后，指令遵循（IF-eval）从 85% 掉到 45%（纯文档）/ 79%（70% 文档 + 30% 对话）；
          再用原 Qwen3-8B 做教师 on-policy 蒸馏，IF-eval 回到 83%，内部知识问答保持 41%——“几乎恢复指令遵循而不丢知识”。

        下图展示一条学生采样序列上的逐 token 信号。点击 token 对比两边的分布。
      `,
      viz: OPDTokenViz,
      checkpoint: {
        id: 'cp-adv',
        type: 'numeric',
        prompt: md`学生采样到某 token，$\log\pi_\theta=-0.5$，$\log\pi_T=-2.0$。按 Thinking Machines 的做法，这个 token 的优势是多少？`,
        answer: -1.5,
        tolerance: 0.001,
        explanation: md`优势 $=-(\log\pi_\theta-\log\pi_T)=-(-0.5+2.0)=-1.5$：学生比教师更自信地说了这个词，教师不认同 → 压低。`,
      },
    },
    {
      id: 'v4',
      title: 'DeepSeek-V4：先训领域专家，再用 OPD 合并',
      body: md`
        DeepSeek-V4 的后训练是两段式，并且**用 on-policy 蒸馏完全替代了 V3.2 的混合 RL 阶段**：

        1. **专家训练**：对每个目标领域（数学、代码、agent、指令遵循等）各训一个专家：先 SFT，再用**带领域奖励模型的 GRPO** 做 RL（超参与此前工作保持接近）。
           为了支持不同的**推理强度档位**，还在不同的 RL 配置（不同的长度惩罚与上下文窗口）下训练了不同的专家。
        2. **多教师 OPD 合并**：统一模型作为学生，在自己生成的轨迹上最小化对各专家的加权反向 KL

        $$
        \mathcal L_{OPD}(\theta)=\sum_{i=1}^{N}w_i\cdot\mathrm D_{KL}\big(\pi_\theta\,\Vert\,\pi_{E_i}\big)
        $$

        $w_i$ 按专家的重要性设定；学生在数学上下文里向数学专家看齐、在代码上下文里向代码专家看齐。这一阶段用了**十个以上**的教师。
        论文认为这种 logits 级对齐规避了权重合并或混合 RL 常见的性能退化。

        **一个关键的实现选择**：以往工作把全词表 KL 简化成**采样 token 上的估计**，并把 $\mathrm{sg}[\log\frac{\pi_{E}}{\pi_\theta}]$ 当作逐 token 优势复用 RL 框架（即上一节的做法）。
        V4 指出这样梯度方差大、训练不稳，改为**全词表 logit 蒸馏**。为了让它在 10 万以上的词表、十几个万亿参数级教师上可行：

        - 教师权重放在分布式存储，前向时按需以 ZeRO 式分片加载；
        - 不物化教师 logits，只缓存教师**最后一层 hidden state**，训练时再过对应的预测头重建 logits；
        - 按教师下标排序样本，保证每个 mini-batch 每个教师的预测头只加载一次、显存里最多一个；
        - 用专门的 TileLang kernel 计算精确的 KL。
      `,
      checkpoint: {
        id: 'cp-v4',
        type: 'multi',
        prompt: md`关于 DeepSeek-V4 的后训练，正确的是：（多选）`,
        options: [
          '各领域专家先 SFT，再用 GRPO 做 RL',
          '用多教师 on-policy 蒸馏（反向 KL）把专家合并成一个模型',
          'OPD 使用采样 token 上的 KL 估计，以节省显存',
          '用 OPD 替代了 V3.2 的混合 RL 阶段',
        ],
        answer: [0, 1, 3],
        explanation: md`C 错：V4 明确改用**全词表** logit 蒸馏，认为采样估计方差大、不稳定，并通过缓存教师 hidden state 等工程手段解决显存问题。`,
      },
    },
    {
      id: 'agentic',
      title: 'Agentic RL：多轮工具调用、观测 mask 与轨迹级奖励',
      body: md`
        Agent 任务的一条轨迹是**模型 token 与环境 token 交错**的：思考 → 工具调用 → 工具返回 → 再思考 → … → 最终答案。RL 训练要处理三件事：

        1. **loss mask**：只有模型生成的 token 是“动作”，工具返回是环境给的**观测**，必须从 loss 中 mask 掉。
           Search-R1（2025-03）把“检索 token 掩码”列为稳定 RL 训练的关键之一；不 mask 会让模型去拟合检索结果，学会自己编造工具输出。
        2. **奖励与信用分配**：常用**轨迹级结果奖励**（任务是否完成 / 答案是否正确），优势广播到该轨迹所有模型 token 上；
           想对不同轮次区别对待时，可用逐 token 的优势（如 GSPO-token 变体就是为此设计的）。
        3. **系统问题**：rollout 时要与沙箱 / 搜索引擎交互，长度不一、耗时长尾严重；多轮场景下训推不一致与 off-policy 更突出（见 GSPO 一关）。
      `,
      viz: AgentMaskViz,
      checkpoint: {
        id: 'cp-agent',
        type: 'pick',
        prompt: md`下列轨迹片段中，哪些应该**计入 loss**？`,
        items: ['用户问题', '<think> 计划 </think>', '<search> 查询 </search>', '<result> 检索结果 </result>', '<answer> 3 </answer>'],
        answer: [1, 2, 4],
        explanation: md`模型自己生成的思考、工具调用和最终答案计入；用户问题是 prompt，检索结果是环境观测，都不计入。`,
      },
    },
  ],

  quiz: [
    {
      id: 'q-who',
      type: 'single',
      difficulty: 1,
      prompt: md`on-policy 蒸馏中，教师模型需要做什么？`,
      options: ['生成完整的示范回答', '对学生采样的轨迹做一次前向，给出每个 token 的对数概率', '训练一个奖励模型', '与学生交替生成 token'],
      answer: 1,
      explanation: md`教师只需 \`compute_logprobs\`（一次前向，可高度并行）；生成由学生完成。A 是 off-policy 蒸馏的做法。`,
    },
    {
      id: 'q-dir',
      type: 'judge',
      difficulty: 1,
      prompt: md`反向 KL 是 mode-covering 的，所以 on-policy 蒸馏的学生会覆盖教师的所有回答模式。`,
      answer: false,
      explanation: md`反向 KL 是 **mode-seeking**；正向 KL 才是 mode-covering。`,
    },
    {
      id: 'q-v4',
      type: 'multi',
      difficulty: 2,
      prompt: md`DeepSeek-V4 为了让全词表 OPD 在十几个大教师上可行，采用了哪些工程手段？（多选）`,
      options: [
        '只缓存教师最后一层 hidden state，训练时用预测头重建 logits',
        '按教师下标排序样本，每个 mini-batch 每个教师头只加载一次',
        '把教师 logits 全部物化到磁盘',
        '用 TileLang kernel 计算精确的 KL',
      ],
      answer: [0, 1, 3],
      explanation: md`C 正是论文说“即使落盘也不可行”的做法（词表 > 10 万、教师众多），所以改为缓存 hidden state 再重建。`,
    },
    {
      id: 'q-adv',
      type: 'numeric',
      difficulty: 2,
      prompt: md`学生采样到某 token：$\pi_\theta=0.2$，$\pi_T=0.6$。按逐 token 负反向 KL 作为优势，值是多少？（自然对数，保留 3 位小数）`,
      answer: 1.099,
      tolerance: 0.005,
      explanation: md`$-(\ln0.2-\ln0.6)=\ln3=1.099>0$：教师比学生更认可这个词，推高它。`,
    },
    {
      id: 'q-fullkl',
      type: 'numeric',
      difficulty: 3,
      prompt: md`词表只有 2 个词：学生 $q=[0.8, 0.2]$，教师 $p=[0.5, 0.5]$。该位置的全词表反向 KL $\mathrm{KL}(q\Vert p)$ 是多少？（保留 3 位小数）`,
      answer: 0.193,
      tolerance: 0.01,
      explanation: md`$0.8\ln\frac{0.8}{0.5}+0.2\ln\frac{0.2}{0.5}=0.8\times0.470-0.2\times0.916=0.376-0.183=0.193$。采样估计在采到词 1 时是 $0.470$、采到词 2 时是 $-0.916$，期望相同但方差大。`,
    },
    {
      id: 'q-tm',
      type: 'fill',
      difficulty: 2,
      prompt: md`Thinking Machines 的实现中，折扣因子取 ____(1)____；从 AIME24 约 60% 的 SFT checkpoint 出发，约 ____(2)____ 步达到 70%。`,
      blanks: [
        { label: '(1)', answers: ['0'], numeric: true, tolerance: 0 },
        { label: '(2)', answers: ['150'], numeric: true, tolerance: 0.1 },
      ],
      explanation: md`折扣因子为 0：每个 token 只优化自己的反向 KL；约 150 步（约 77K 个 prompt、每个 4 个样本）到 70%。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把方法与它的“样本来源 / 信号密度”配对：`,
      pairs: [
        ['SFT / off-policy 蒸馏', '教师样本 · 逐 token'],
        ['RLVR', '学生样本 · 每条一个奖励'],
        ['on-policy 蒸馏', '学生样本 · 逐 token'],
        ['DeepSeek-V4 OPD', '学生样本 · 全词表多教师 KL'],
      ],
      explanation: md`on-policy 蒸馏是“RL 的采样 + SFT 的稠密信号”；V4 在此基础上用全词表 KL 并支持多教师。`,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 2,
      prompt: md`按 DeepSeek-V4 后训练的先后顺序排列：`,
      items: ['各领域专家：领域数据 SFT', '各领域专家：带领域奖励的 GRPO（含不同推理强度配置）', '统一学生模型在自身轨迹上采样', '对十余个专家做加权全词表反向 KL 蒸馏'],
      explanation: md`先分别培养专家，再用 OPD 合并：学生采样 → 教师们给出全词表分布 → 最小化加权反向 KL。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 3,
      prompt: md`下面的多轮工具调用 RL loss 有 **2 处** bug（\`role_ids\`：1=prompt，2=模型生成，3=工具返回），点出来：`,
      code: py`
        def agentic_loss(logp_new, logp_old, role_ids, adv, eps=0.2):
            # logp_*, role_ids: (B, T)；adv: (B,) 轨迹级优势
            loss_mask = (role_ids >= 2).astype(float)
            A = adv[:, None]
            ratio = np.exp(logp_new - logp_old)
            surr = np.minimum(ratio * A, np.clip(ratio, 1 - eps, 1 + eps) * A)
            return -(surr * loss_mask).sum() / role_ids.size
      `,
      bugLines: [3, 7],
      explanation: md`第 3 行把工具返回（3）也算进了 loss，应为 \`role_ids == 2\`。第 7 行分母用了全部位置数（含 prompt、工具、padding），应为 \`loss_mask.sum()\`。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出 Thinking Machines 式 on-policy 蒸馏的 loss（有 2 行干扰项）：`,
      lines: [
        'def opd_loss(lp_new, lp_old, lp_teacher, mask):',
        '    reverse_kl = lp_old - lp_teacher',
        '    adv = -reverse_kl',
        '    ratio = np.exp(lp_new - lp_old)',
        '    return -(ratio * adv * mask).sum() / mask.sum()',
      ],
      distractors: ['    reverse_kl = lp_teacher - lp_old', '    adv = reverse_kl'],
      explanation: md`反向 KL 的单样本估计是 $\log\pi_\theta-\log\pi_T$（学生减教师），优势取它的负值；loss 是 RL 的 importance-sampling 形式。`,
    },
    {
      id: 'q-why',
      type: 'single',
      difficulty: 3,
      prompt: md`Thinking Machines 的持续学习实验中，为什么“用模型自己的样本做 SFT”仍会让 IF-eval 退化，而 on-policy 蒸馏不会？`,
      options: [
        '自采样 SFT 的 KL 期望不为 0',
        '有限 batch 的噪声让自训练逐渐变成 off-policy；而蒸馏的教师固定，提供了不漂移的目标',
        'on-policy 蒸馏使用了更大的学习率',
        '自采样 SFT 没有用 LoRA',
      ],
      answer: 1,
      explanation: md`博客的解释：自采样 SFT 的 KL 期望为 0，但任何实际的非零学习率下，有限 batch 的噪声会累积，训练逐渐偏离；on-policy 蒸馏以固定教师为锚，不会漂移。A 说反了。`,
    },
  ],

  guided: [
    {
      id: 'g-opd',
      title: '把 RL 的 loss 改成 on-policy 蒸馏',
      intro: md`
        输入均为 \`(B, T)\`：学生当前的 \`lp_new\`、采样时的 \`lp_old\`、教师在同一轨迹上的 \`lp_teacher\`、\`mask\`。
        按 Thinking Machines 的做法：逐 token 反向 KL 估计 → 取负作为优势（折扣 0）→ importance-sampling loss（token 级平均）。
      `,
      template: py`
        import numpy as np

        def opd_loss(lp_new, lp_old, lp_teacher, mask):
            reverse_kl = {{1}}            # 采样 token 上的反向 KL 估计（用采样时的学生概率）
            adv = {{2}}                   # 逐 token 优势（折扣因子 0）
            ratio = {{3}}                 # importance-sampling 比值
            return {{4}}                  # token 级平均，取负
      `,
      steps: [
        { blank: 1, instruction: md`$\log\pi_\theta(y_t)-\log\pi_T(y_t)$，学生概率取采样时的 \`lp_old\`（它是常数，不求梯度）。`, answer: 'lp_old - lp_teacher', explain: md`优势必须是常数；用 \`lp_new\` 会让梯度穿过“奖励”本身。` },
        { blank: 2, instruction: md`优势 = 负的反向 KL。`, answer: '-reverse_kl', accept: ['lp_teacher - lp_old'], explain: md`学生比教师更自信的词被压低，反之推高。` },
        { blank: 3, instruction: md`新旧学生策略的比值。`, answer: 'np.exp(lp_new - lp_old)', explain: md`第一个 mini-batch 时它等于 1，梯度就是 $\mathrm{adv}\cdot\nabla\log\pi_\theta$。` },
        {
          blank: 4,
          instruction: md`$-\frac{1}{N}\sum \text{ratio}\cdot\text{adv}$，只在有效 token 上。`,
          answer: '-(ratio * adv * mask).sum() / mask.sum()',
          accept: ['-np.sum(ratio * adv * mask) / np.sum(mask)'],
          explain: md`这就是“把 RL 奖励换成教师逐 token 打分”的全部改动。`,
        },
      ],
      setup: SETUP,
      tests: [
        {
          name: '学生与教师一致时 loss = 0',
          code: py`
            rng = np.random.default_rng(0)
            lp = -rng.exponential(1.0, size=(2, 5)); m = np.ones((2, 5))
            got = float(opd_loss(lp, lp, lp, m))
            assert abs(got) < 1e-12, f"教师 = 学生时优势全 0，loss 应为 0，得到 {got}"
          `,
        },
        {
          name: '对拍',
          code: py`
            rng = np.random.default_rng(1)
            lo = -rng.exponential(1.0, size=(3, 6)); ln = lo + rng.normal(0, 0.2, size=lo.shape); lt = lo + rng.normal(0, 1, size=lo.shape)
            m = (np.arange(6)[None, :] < np.array([[6], [3], [1]])).astype(float)
            got = float(opd_loss(ln, lo, lt, m))
            exp = float(-(np.exp(ln - lo) * (lt - lo) * m).sum() / m.sum())
            assert abs(got - exp) < 1e-12, f"期望 {exp}，得到 {got}"
          `,
        },
      ],
      outro: md`
        DeepSeek-V4 认为这种采样估计方差大，改用全词表 KL：需要教师在每个位置的完整分布——实战题第一题就实现两种信号并比较。
      `,
    },
  ],

  challenges: [
    {
      id: 'c-opd-kl',
      title: '逐 token 反向 KL：全词表 vs 采样估计',
      difficulty: 'medium',
      source: 'Thinking Machines OPD · DeepSeek-V4',
      timeLimitMin: 18,
      description: md`
        \`\`\`python
        def opd_token_stats(student_logits, teacher_logits, tokens, mask) -> dict: ...
        \`\`\`

        - \`student_logits / teacher_logits\`：\`(B, T, V)\`（可能数值很大）；\`tokens\`：\`(B, T)\` 学生采样的 token id；\`mask\`：\`(B, T)\`。
        - 返回字典（padding 位置均为 0）：
          - \`"full_kl"\`：\`(B, T)\`，$\sum_v q(v)\big(\log q(v)-\log p(v)\big)$，$q$ 学生、$p$ 教师；
          - \`"sampled_kl"\`：\`(B, T)\`，$\log q(y_t)-\log p(y_t)$；
          - \`"adv"\`：\`(B, T)\`，$-$\`sampled_kl\`；
          - \`"mean_full_kl"\`：有效 token 上 \`full_kl\` 的平均（float）。
      `,
      starter: py`
        import numpy as np

        def opd_token_stats(student_logits, teacher_logits, tokens, mask):
            # TODO: 稳定的 log_softmax → 全词表 KL、采样 token 上的估计 → 掩码
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def _log_softmax(x):
            m = x.max(axis=-1, keepdims=True)
            return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))

        def opd_token_stats(student_logits, teacher_logits, tokens, mask):
            ls = _log_softmax(np.asarray(student_logits, dtype=np.float64))
            lt = _log_softmax(np.asarray(teacher_logits, dtype=np.float64))
            m = np.asarray(mask, dtype=np.float64)
            full = (np.exp(ls) * (ls - lt)).sum(axis=-1) * m              # 全词表反向 KL
            idx = np.asarray(tokens)[..., None]
            sampled = (np.take_along_axis(ls, idx, -1) - np.take_along_axis(lt, idx, -1))[..., 0] * m
            return {
                "full_kl": full,
                "sampled_kl": sampled,
                "adv": -sampled,
                "mean_full_kl": float(full.sum() / m.sum()),
            }
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示）',
          code: py`
            import torch
            import torch.nn.functional as F

            def opd_token_stats_torch(student_logits, teacher_logits, tokens, mask):
                ls = F.log_softmax(student_logits.float(), -1)
                lt = F.log_softmax(teacher_logits.float(), -1)
                full = (ls.exp() * (ls - lt)).sum(-1) * mask          # 也可写 F.kl_div(lt, ls, log_target=True, reduction="none").sum(-1)
                sampled = (ls.gather(-1, tokens[..., None]) - lt.gather(-1, tokens[..., None])).squeeze(-1) * mask
                return {"full_kl": full, "sampled_kl": sampled, "adv": -sampled.detach(),
                        "mean_full_kl": full.sum() / mask.sum()}
          `,
        },
      ],
      explanation: md`
        - 采样估计是全词表 KL 的**无偏单样本估计**：$\mathbb E_{y\sim q}[\log q(y)-\log p(y)]=\mathrm{KL}(q\Vert p)$（测试会验证这一点）。
        - 全词表版本需要教师在每个位置的完整分布（显存 / 通信是 $O(BTV)$），这正是 DeepSeek-V4 用“缓存 hidden state + 重建 logits”解决的问题。
        - 注意 \`F.kl_div(input, target)\` 计算的是 $\mathrm{KL}(\text{target}\Vert\text{input})$，参数顺序极易写反。
      `,
      hints: [md`先写稳定的 \`_log_softmax\`（减最大值）。`, md`全词表：\`(exp(ls) * (ls - lt)).sum(-1)\`；采样：\`np.take_along_axis(ls, tokens[..., None], -1)\`。`],
      setup: SETUP,
      tests: [
        {
          name: '字段与形状',
          code: py`
            s, t, tok, m = _opd_batch(0)
            out = opd_token_stats(s, t, tok, m)
            for k in ("full_kl", "sampled_kl", "adv"):
                assert np.shape(out[k]) == m.shape, f"{k} 形状应为 {m.shape}，得到 {np.shape(out[k])}"
                assert np.all(np.asarray(out[k])[m == 0] == 0), f"{k} 的 padding 位置应为 0"
            assert np.all(np.asarray(out["full_kl"])[m > 0] >= -1e-12), "KL 不应为负"
          `,
        },
        {
          name: '采样估计的期望 = 全词表 KL',
          code: py`
            s, t, tok, m = _opd_batch(1, B=1, T=1, V=6)
            m = np.ones((1, 1))
            full = opd_token_stats(s, t, tok, m)["full_kl"][0, 0]
            q = np.exp(_lsm(s))[0, 0]
            ev = sum(q[v] * opd_token_stats(s, t, np.array([[v]]), m)["sampled_kl"][0, 0] for v in range(6))
            assert abs(ev - full) < 1e-10, f"Σ_v q(v)·sampled_kl(v) = {ev}，应等于 full_kl = {full}"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            for seed in range(20):
                s, t, tok, m = _opd_batch(10 + seed)
                out = opd_token_stats(s, t, tok, m)
                exp = _ref_opd(s, t, tok, m)
                for k in ("full_kl", "sampled_kl", "adv"):
                    assert np.allclose(out[k], exp[k], atol=1e-10), f"seed={seed}: {k} 最大误差 {np.abs(np.asarray(out[k]) - exp[k]).max():.3e}"
                assert abs(out["mean_full_kl"] - exp["mean_full_kl"]) < 1e-10
          `,
        },
        {
          name: '大 logits 数值稳定',
          hidden: true,
          code: py`
            s, t, tok, m = _opd_batch(99, offset=800.0, scale=5.0)
            out = opd_token_stats(s, t, tok, m)
            exp = _ref_opd(s, t, tok, m)
            assert np.all(np.isfinite(out["full_kl"])), "出现 inf / nan：log_softmax 要先减最大值"
            assert np.allclose(out["full_kl"], exp["full_kl"], atol=1e-9)
          `,
        },
      ],
    },
    {
      id: 'c-agentic',
      title: '多轮工具调用 RL 的 loss（观测 mask + 轨迹级优势）',
      difficulty: 'medium',
      source: 'Agentic RL · Search-R1 式 loss mask',
      timeLimitMin: 15,
      description: md`
        \`\`\`python
        def agentic_loss(logp_new, logp_old, role_ids, rewards, group_ids, clip_eps=0.2) -> tuple[float, np.ndarray]: ...
        \`\`\`

        - \`role_ids\`：\`(B, T)\` 整数，0=padding、1=prompt、2=**模型生成**、3=**工具返回**；只有 2 计入 loss。非 2 位置的 logp 是任意值。
        - \`rewards / group_ids\`：轨迹级奖励与分组（同一问题的多条轨迹为一组），优势用组内 $(r-\text{mean})/(\text{std}_{ddof=1}+10^{-6})$，广播到该轨迹所有模型 token。
        - PPO-clip，token 级聚合（除以模型 token 总数）。返回 \`(loss, loss_mask)\`，\`loss_mask\` 为 \`(B, T)\` 布尔数组。
      `,
      starter: py`
        import numpy as np

        def agentic_loss(logp_new, logp_old, role_ids, rewards, group_ids, clip_eps=0.2):
            # TODO: loss_mask = 模型生成的 token；组内优势广播；PPO-clip；token 级平均
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def agentic_loss(logp_new, logp_old, role_ids, rewards, group_ids, clip_eps=0.2):
            loss_mask = np.asarray(role_ids) == 2                 # 只训练模型自己的动作
            lm = loss_mask.astype(np.float64)
            r = np.asarray(rewards, dtype=np.float64)
            _, inv = np.unique(group_ids, return_inverse=True)
            inv = inv.reshape(-1)
            cnt = np.bincount(inv).astype(np.float64)
            dev = r - (np.bincount(inv, weights=r) / cnt)[inv]
            std = np.sqrt(np.bincount(inv, weights=dev ** 2) / (cnt - 1))
            A = (dev / (std[inv] + 1e-6))[:, None]                # 轨迹级优势广播到每个 token
            ratio = np.exp(np.where(loss_mask, logp_new - logp_old, 0.0))
            tok = -np.minimum(ratio * A, np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A)
            return float((tok * lm).sum() / lm.sum()), loss_mask
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示）',
          code: py`
            import torch

            def agentic_loss_torch(logp_new, logp_old, role_ids, adv, clip_eps=0.2):
                lm = (role_ids == 2).float()                       # 工具返回、prompt、padding 全部为 0
                A = adv.unsqueeze(-1).detach()
                ratio = torch.exp((logp_new - logp_old.detach()) * lm)
                surr = torch.min(ratio * A, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * A)
                return -(surr * lm).sum() / lm.sum()
          `,
        },
      ],
      explanation: md`
        与单轮 GRPO 的唯一区别是 loss mask：它不再是“回答区间”，而是**按角色**挑出模型生成的片段。实际框架里通常在拼接多轮对话时同步生成这个 mask。
        工具返回 token 的 logp 根本不该被用到——测试会修改它们并检查 loss 不变。
      `,
      hints: [md`\`loss_mask = role_ids == 2\`。`, md`其余与 GRPO 相同：组内优势 → ratio → PPO-clip → \`(tok * mask).sum() / mask.sum()\`。`],
      setup: SETUP,
      tests: [
        {
          name: 'loss_mask 只含模型 token',
          code: py`
            ln, lo, roles, r, gid = _agent_batch(0)
            loss, lm = agentic_loss(ln, lo, roles, r, gid)
            assert np.asarray(lm).dtype == bool and np.asarray(lm).tolist() == (roles == 2).tolist(), "loss_mask 应恰为 role_ids == 2"
          `,
        },
        {
          name: '工具返回 token 的 logp 不影响 loss',
          code: py`
            ln, lo, roles, r, gid = _agent_batch(1)
            a, _ = agentic_loss(ln, lo, roles, r, gid)
            ln2 = ln.copy(); ln2[roles == 3] += 5.0
            b, _ = agentic_loss(ln2, lo, roles, r, gid)
            assert abs(a - b) < 1e-12, f"修改工具返回 token 的 logp 后 loss 从 {a} 变为 {b}"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            for seed in range(25):
                ln, lo, roles, r, gid = _agent_batch(50 + seed, n_groups=3, G=4, T=28)
                loss, _ = agentic_loss(ln, lo, roles, r, gid)
                exp, _ = _ref_agentic_loss(ln, lo, roles, r, gid)
                assert abs(loss - exp) < 1e-10, f"seed={seed}: 期望 {exp:.10f}，得到 {loss:.10f}"
          `,
        },
        {
          name: '乱序分组',
          hidden: true,
          code: py`
            ln, lo, roles, r, gid = _agent_batch(7, n_groups=3, G=3, T=24)
            perm = np.random.default_rng(0).permutation(len(r))
            ids = np.array([30, 10, 20])[gid]
            args = (ln[perm], lo[perm], roles[perm], r[perm], ids[perm])
            loss, _ = agentic_loss(*args)
            exp, _ = _ref_agentic_loss(*args)
            assert abs(loss - exp) < 1e-10, f"期望 {exp:.10f}，得到 {loss:.10f}"
          `,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-opd',
      freq: 3,
      q: md`什么是 on-policy 蒸馏？为什么比 SFT 蒸馏好、比 RL 便宜？`,
      a: md`
        **结论**：学生自己采样，教师对学生轨迹上的每个 token 给出对数概率，以逐 token 的负反向 KL 作为优势（或直接最小化全词表反向 KL）。

        **原因**：比 off-policy 蒸馏好——在学生真正会走到的状态上学习，没有 exposure bias；比 RL 便宜——每个 token 都有稠密信号，而 RL 一条回答只有一个标量。教师只需前向，可高度并行。

        **延伸**：Thinking Machines 报告从 60% 到 70%（AIME24）约 150 步，算力比继续 SFT 便宜 9–30 倍；Qwen3-8B 上 74.4% vs RL 的 67.6%，GPU 小时 1,800 vs 17,920。源头是 GKD、MiniLLM。
      `,
    },
    {
      id: 'qa-v4',
      freq: 3,
      q: md`DeepSeek-V4 如何把多个领域专家合并成一个模型？为什么不用权重合并或混合 RL？`,
      a: md`
        **结论**：先为每个领域（数学、代码、agent、指令遵循等）SFT + GRPO 训练专家，再让统一模型在自己的轨迹上最小化对十余个专家的加权反向 KL（多教师 OPD），替代了 V3.2 的混合 RL 阶段。

        **原因**：logits 级对齐让学生在不同上下文里向对应专家看齐；论文认为它规避了权重合并与混合 RL 常见的性能退化。

        **延伸**：V4 用**全词表** KL 而不是采样估计（后者方差大）；工程上缓存教师最后一层 hidden state 重建 logits、按教师排序样本、TileLang 算 KL；推理强度档位由不同长度惩罚 / 上下文的专家提供。
      `,
    },
    {
      id: 'qa-agentic',
      freq: 2,
      q: md`多轮工具调用的 RL 和单轮推理 RL 有什么不同？`,
      a: md`
        **结论**：轨迹由模型 token 与环境 token 交错组成；loss 只算模型 token，奖励通常是轨迹级的。

        **原因**：工具返回是观测而不是动作，计入 loss 会让模型拟合并编造工具输出（Search-R1 的检索 token 掩码）；结果奖励稀疏，信用分配更难。

        **延伸**：可用逐 token 优势（GSPO-token）、过程奖励或中间状态奖励细化信用；rollout 需要与沙箱交互、耗时长尾严重；训推不一致与 off-policy 问题更突出。
      `,
    },
    {
      id: 'qa-kldir',
      freq: 2,
      q: md`蒸馏用正向 KL 还是反向 KL？`,
      a: md`
        **结论**：传统 KD / SFT 对应正向 KL（在教师分布上取期望）；on-policy 蒸馏用反向 KL（在学生分布上取期望）。

        **原因**：正向 KL 是 mode-covering，学生容量不足时会把概率放在教师模式之间；反向 KL 是 mode-seeking，学生专注于做好一个模式，生成质量更高，而且期望在学生分布下，天然对应 on-policy 采样。

        **延伸**：GKD 用广义 JSD 在两者之间插值；反向 KL 的风险是多样性下降。
      `,
    },
  ],

  cheatsheet: md`
    | 方法 | 样本 | 信号 |
    |---|---|---|
    | SFT / 传统蒸馏 | 教师 | 逐 token（正向 KL） |
    | RL | 学生 | 每条一个奖励 |
    | on-policy 蒸馏 | 学生 | 逐 token（反向 KL） |

    \`\`\`python
    adv = -(lp_student_old - lp_teacher)            # 折扣 0
    loss = -(np.exp(lp_new - lp_old) * adv * mask).sum() / mask.sum()
    full_kl = (np.exp(ls) * (ls - lt)).sum(-1)       # V4：全词表
    loss_mask = role_ids == 2                        # agent：工具返回不计 loss
    \`\`\`
    - TM（2025-10）：60% → 70% AIME24 约 150 步；比 SFT 便宜 9–30×；Qwen3-8B：OPD 74.4%（1.8K GPU·h）vs RL 67.6%（17.9K）
    - DeepSeek-V4：专家 SFT + GRPO → 多教师 OPD $\sum_i w_i\mathrm{KL}(\pi_\theta\Vert\pi_{E_i})$，10+ 教师，替代混合 RL
    - 反向 KL：mode-seeking；正向 KL：mode-covering
  `,
})
