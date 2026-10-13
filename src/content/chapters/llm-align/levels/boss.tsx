/**
 * BOSS：后训练与偏好对齐综合。复盘 + 混合题 + 两道综合手撕（PPO 一步的数值部分、统一的偏好损失族）。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { PipelineViz } from '../viz/PipelineViz'
import { PY_MAKE_PAIRS, PY_REF_LOGPS } from '../shared'

export default defineLevel({
  learn: [
    {
      id: 'recap',
      title: '复盘：一张图串起后训练',
      body: md`
        | 阶段 | 关键公式 | 必会细节 |
        |---|---|---|
        | SFT | $-\sum_{t\in\text{回复}}\log\pi_\theta(y_t\mid\cdot)$ | labels = -100、结束符要训、shift、packing（position 重置 + 块对角 + 边界 -100）、token vs 样本平均 |
        | RM | $-\log\sigma(r_w-r_l-m)$ | 最后有效 token 打分、K 路排序 $\binom K2$、平移不变、PRM vs ORM、judge 偏差 |
        | PPO | $\min(rA,\mathrm{clip}(r)A)$，$\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}$ | 四个模型、逐 token KL + 末 token RM、GAE 倒序、padding 置 0、value clip、clipfrac |
        | DPO | $-\log\sigma(\beta\Delta)$ | 闭式解 → 反解奖励 → BT → Z 相消；隐式奖励；第一步 ln2；序列 logp 只累加回复 |
        | 变体 | IPO / SimPO / KTO / ORPO | 有限目标 margin、长度归一化 + γ、前景理论 + z₀、odds ratio + SFT、iterative DPO |

        选型的三个问题：**数据是什么形态**（示范 / 成对 / 单条 / 可验证）？**算力能放下几个模型**？**需不需要在线探索**？用下面的对照器复习一遍。
      `,
      viz: PipelineViz,
      after: md`
        > [!INTERVIEW] 常见的压轴组合题
        > “从一个 base 模型出发，只有 10 万条偏好数据和 8 张 80GB 卡，你怎么做后训练？”——好的回答会先 SFT（用 chosen 或另外的示范数据），再 DPO（β ≈ 0.1，ref 预计算或 LoRA 共享底座），监控第一步 ln2、rewards/margins、chosen logp，必要时加 NLL 正则；若有可验证任务，再上 GRPO；若数据只有点赞/点踩，换 KTO。
      `,
    },
  ],

  quiz: [
    {
      id: 'b-order',
      type: 'order',
      difficulty: 1,
      prompt: md`把 InstructGPT 的三阶段流程按顺序排列：`,
      items: ['收集示范数据，做 SFT', '对 SFT 模型的多个输出排序，训练奖励模型', '以 RM 为奖励、SFT 为 reference，用 PPO 优化策略'],
      explanation: md`SFT → RM → PPO。DPO 把后两步合成一个分类损失；GRPO 系在可验证任务上用规则奖励替代 RM、去掉 critic。`,
    },
    {
      id: 'b-match',
      type: 'match',
      difficulty: 1,
      prompt: md`把方法与它最适合的数据形态配对：`,
      pairs: [
        ['SFT', '高质量示范 (prompt, response)'],
        ['DPO / IPO / SimPO', '同 prompt 的成对偏好'],
        ['KTO', '单条的好 / 坏标签'],
        ['GRPO（RLVR）', 'prompt + 可程序验证的答案'],
      ],
      explanation: md`数据形态决定方法：成对 → DPO 系；单条 → KTO；可验证 → RLVR；示范 → SFT。`,
    },
    {
      id: 'b-sft-judge',
      type: 'judge',
      difficulty: 1,
      prompt: md`SFT 把 prompt 的 label 设为 -100 后，这些 token 不再参与前向计算，因此能节省大量显存。`,
      answer: false,
      explanation: md`被 mask 的 token 仍要前向（它们是上下文），显存与计算几乎不变；mask 只影响哪些位置贡献 loss。真正省算力的是 packing / padding-free。`,
    },
    {
      id: 'b-gae',
      type: 'numeric',
      difficulty: 2,
      prompt: md`三个 token 的回复，奖励 $r=[-0.1,\,-0.2,\,2.0]$，$V(s_0)=1.2$，$\gamma=\lambda=1$。求 $\hat A_0$。`,
      answer: 0.5,
      tolerance: 0.001,
      explanation: md`γ = λ = 1 时 $\hat A_0=\sum_t r_t-V(s_0)=1.7-1.2=0.5$（中间的 V 望远镜相消，与 $V(s_1)$、$V(s_2)$ 无关）。`,
    },
    {
      id: 'b-multi',
      type: 'multi',
      difficulty: 2,
      prompt: md`下列说法正确的有：`,
      options: [
        'DPO 的隐式奖励是 β·log(π_θ/π_ref)，同一 prompt 下与真实奖励相差常数 β·log Z(x)',
        'PPO 中 A>0 且 ratio>1+ε 的 token 梯度为 0',
        'RM 的 BT 损失对两个分数同时加一个常数不变',
        'SimPO 与 DPO 一样需要 reference model',
        'packing 时只需重置 position_ids 就能避免跨样本注意力',
      ],
      answer: [0, 1, 2],
      explanation: md`D 错：SimPO 无 reference。E 错：position_ids 只影响位置编码，注意力隔离需要块对角掩码或 varlen kernel。`,
    },
    {
      id: 'b-fill',
      type: 'fill',
      difficulty: 2,
      prompt: md`DPO 训练第一步（policy = reference）的 loss 应等于 ____(1)____（保留 3 位小数）；IPO（τ = 0.25）第一步的 loss 等于 ____(2)____。`,
      blanks: [
        { label: '(1)', answers: ['0.693'], numeric: true, tolerance: 0.005 },
        { label: '(2)', answers: ['4'], numeric: true, tolerance: 0.001 },
      ],
      explanation: md`$h=0$：DPO 为 $-\log\sigma(0)=\ln2$；IPO 为 $(0-\frac{1}{2\times0.25})^2=2^2=4$。`,
    },
    {
      id: 'b-debug',
      type: 'single',
      difficulty: 3,
      prompt: md`你接手的 DPO 训练第一步 loss = 0.52（β = 0.1），而不是 0.693。最可能的原因是：`,
      options: [
        'policy 与 reference 实际不一致（加载了不同 checkpoint、dropout 没关、或两边的 padding / 截断不同）',
        'β 设得太小',
        '学习率太大',
        '偏好数据中有 10% 的噪声标签',
      ],
      answer: 0,
      explanation: md`第一步 policy 尚未更新，若两者一致则每对 Δ = 0、loss 恰为 ln2，与 β、学习率、数据无关。loss 偏离说明 Δ ≠ 0：reference 不是同一个模型、有随机性（dropout）、或两次前向的输入不一致。`,
    },
    {
      id: 'b-bug',
      type: 'bug',
      difficulty: 3,
      prompt: md`
        下面的 SimPO 实现有 **2 处** bug（\`tok_c\` / \`tok_r\` 是已 gather 的逐 token logp，\`m_c\` / \`m_r\` 是回复 mask；\`ref_*\` 是 reference 的逐 token logp）。点出有问题的行。
      `,
      code: py`
        def simpo_loss(tok_c, m_c, tok_r, m_r, ref_c, ref_r, beta=2.0, gamma=1.0):
            # tok_*: (B, T) 逐 token logp；m_*: (B, T) 回复 mask
            n_c = m_c.sum(-1)
            n_r = m_r.sum(-1)
            s_c = (tok_c * m_c).sum(-1)
            s_r = (tok_r * m_r).sum(-1)
            s_c, s_r = s_c - (ref_c * m_c).sum(-1), s_r - (ref_r * m_r).sum(-1)
            z = beta * (s_c - s_r) - gamma
            losses = np.logaddexp(0.0, -z)
            return losses.mean()
      `,
      bugLines: [7, 8],
      explanation: md`
        第 7 行：SimPO 不使用 reference——减去 ref 的 logp 是 DPO / length-normalized DPO 的做法。
        第 8 行：SimPO 的奖励是**长度归一化**的 $\frac{\beta}{|y|}\log\pi$，应为 \`beta * (s_c / n_c - s_r / n_r) - gamma\`（第 3、4 行算出的长度没被用上就是线索）。
      `,
    },
    {
      id: 'b-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`下列量中，哪些需要 **reference model** 的前向（或预先算好的 reference logp）？`,
      items: ['DPO 的隐式奖励', 'SimPO 的奖励', 'PPO 的逐 token KL 惩罚', 'KTO 的参考点 z₀', 'ORPO 的 odds ratio', 'SFT 的交叉熵'],
      answer: [0, 2, 3],
      explanation: md`DPO 奖励 $\beta\log\frac{\pi_\theta}{\pi_{\mathrm{ref}}}$、PPO 的 KL、KTO 的 $z_0$（以及 $r_\theta$）都要 reference；SimPO、ORPO、SFT 都只用 $\pi_\theta$。`,
    },
    {
      id: 'b-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出只在有效 token 上统计的优势白化（masked whiten）。有 2 行干扰项。`,
      lines: [
        'def masked_whiten(adv, mask, eps=1e-8):',
        '    n = mask.sum()',
        '    mean = (adv * mask).sum() / n',
        '    var = (((adv - mean) * mask) ** 2).sum() / n',
        '    return (adv - mean) / np.sqrt(var + eps) * mask',
      ],
      distractors: ['    mean = adv.mean()', '    var = adv.var()'],
      explanation: md`均值和方差都只在 mask 内统计；干扰项把 padding（通常是 0）也算进去，会把均值拉向 0、方差算错。注意 returns 要用白化**前**的优势计算。`,
    },
    {
      id: 'b-kto',
      type: 'numeric',
      difficulty: 3,
      prompt: md`KTO 训练数据中 desirable : undesirable = 1 : 4。取 λ_U = 1，希望 $\frac{\lambda_Dn_D}{\lambda_Un_U}=1.2$（在论文建议的 [1, 1.5] 内），λ_D 应取多少？`,
      answer: 4.8,
      tolerance: 0.001,
      explanation: md`$\lambda_D\cdot1/(1\cdot4)=1.2\Rightarrow\lambda_D=4.8$。`,
    },
    {
      id: 'b-online',
      type: 'single',
      difficulty: 2,
      prompt: md`下列方法中，训练过程本身包含**用当前策略在线采样**的是：`,
      options: ['PPO-RLHF 与 GRPO', 'DPO 与 SimPO', 'SFT 与 ORPO', 'KTO 与 IPO'],
      answer: 0,
      explanation: md`PPO、GRPO 每轮都用当前策略 rollout；DPO 系、KTO、ORPO、SFT 都在固定数据上训练（iterative DPO 是在外层循环中重新采样）。`,
    },
  ],

  guided: [],

  challenges: [
    {
      id: 'c-ppo-step',
      title: '综合：一次 PPO 更新的全部数值计算',
      difficulty: 'hard',
      source: '大模型岗手撕 · PPO 综合',
      timeLimitMin: 40,
      description: md`
        把 PPO-RLHF 一次 mini-batch 更新的数值部分串起来（不含反向传播）：

        \`\`\`python
        def ppo_step(logp, old_logp, ref_logp, values, old_values, rm_score, mask,
                     beta=0.05, gamma=1.0, lam=0.95, clip_eps=0.2, clip_vf=0.2,
                     vf_coef=0.1, whiten=True) -> dict: ...
        \`\`\`

        输入均为 \`(B, T)\`（\`rm_score\` 为 \`(B,)\`），\`mask\` 右侧 padding、每行至少 1 个有效 token，padding 位置是任意有限值。

        1. **奖励**：用 rollout 时的 \`old_logp\` 计算 $r_t=-\beta(\mathrm{old\_logp}-\mathrm{ref\_logp})$，最后一个有效 token 再加 \`rm_score\`；
        2. **GAE**：用 rollout 时的 \`old_values\`（padding 处视为 0）倒序递推，得到 \`advantages\`；\`returns = advantages + old_values\`（**白化之前**）；
        3. **白化**（\`whiten=True\` 时）：在有效 token 上求均值 $\mu$ 与总体标准差 $s$（ddof = 0），\`adv = (adv - μ) / (s + 1e-8)\`；
        4. **PPO loss**：ratio = exp(logp − old_logp)；pg_loss、vf_loss（以 \`old_values\` 为裁剪中心、\`returns\` 为目标）、clipfrac、approx_kl 的定义与「PPO 与 RLHF」关卡实战题完全相同（全部为有效 token 平均）；\`loss = pg_loss + vf_coef * vf_loss\`。

        返回 dict：\`loss\`、\`pg_loss\`、\`vf_loss\`、\`clipfrac\`、\`approx_kl\`（float），以及 \`advantages\`（白化后）、\`returns\`（均为 \`(B, T)\`，padding 处为 0）。
      `,
      starter: py`
        import numpy as np

        def ppo_step(logp, old_logp, ref_logp, values, old_values, rm_score, mask,
                     beta=0.05, gamma=1.0, lam=0.95, clip_eps=0.2, clip_vf=0.2,
                     vf_coef=0.1, whiten=True):
            # 1) 奖励塑形  2) GAE（old_values）  3) 白化  4) PPO loss
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def ppo_step(logp, old_logp, ref_logp, values, old_values, rm_score, mask,
                     beta=0.05, gamma=1.0, lam=0.95, clip_eps=0.2, clip_vf=0.2,
                     vf_coef=0.1, whiten=True):
            m = np.asarray(mask, dtype=np.float64)
            on = m > 0
            B, T = m.shape
            n = m.sum()

            # 1) 奖励：rollout 时的 logp 与 ref 的逐 token KL，末 token 加 RM 分数
            rewards = np.where(on, -beta * (old_logp - ref_logp), 0.0)
            last = m.sum(1).astype(int) - 1
            rewards[np.arange(B), last] += rm_score

            # 2) GAE（rollout 时的 critic），padding 价值为 0
            v_old = np.where(on, old_values, 0.0)
            adv = np.zeros((B, T))
            last_gae = np.zeros(B)
            for t in reversed(range(T)):
                next_v = v_old[:, t + 1] if t + 1 < T else np.zeros(B)
                delta = rewards[:, t] + gamma * next_v - v_old[:, t]
                last_gae = delta + gamma * lam * last_gae
                adv[:, t] = last_gae
            adv = np.where(on, adv, 0.0)
            returns = np.where(on, adv + v_old, 0.0)        # 白化之前

            # 3) masked whiten
            if whiten:
                mu = adv[on].mean()
                sd = adv[on].std()
                adv = np.where(on, (adv - mu) / (sd + 1e-8), 0.0)

            # 4) PPO loss
            mmean = lambda x: float(np.where(on, x, 0.0).sum() / n)
            log_ratio = np.where(on, logp - old_logp, 0.0)
            ratio = np.exp(log_ratio)
            pg1 = -adv * ratio
            pg2 = -adv * np.clip(ratio, 1 - clip_eps, 1 + clip_eps)
            pg_loss = mmean(np.maximum(pg1, pg2))
            v_clip = old_values + np.clip(values - old_values, -clip_vf, clip_vf)
            vf_loss = 0.5 * mmean(np.maximum((values - returns) ** 2, (v_clip - returns) ** 2))
            return {
                "loss": pg_loss + vf_coef * vf_loss,
                "pg_loss": pg_loss,
                "vf_loss": vf_loss,
                "clipfrac": mmean((pg2 > pg1).astype(np.float64)),
                "approx_kl": mmean((ratio - 1.0) - log_ratio),
                "advantages": adv,
                "returns": returns,
            }
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示：真实训练循环中的位置）',
          code: py`
            import torch

            @torch.no_grad()
            def make_experience(batch, actor, ref, critic, rm, beta, gamma, lam):
                seq, mask = actor.generate(batch)                       # rollout
                old_logp = actor.logprobs(seq) ; ref_logp = ref.logprobs(seq)
                old_values = critic(seq) ; score = rm(seq)
                rewards = -beta * (old_logp - ref_logp) * mask
                rewards[torch.arange(len(seq)), mask.sum(1) - 1] += score
                adv, ret = gae(rewards, old_values * mask, mask, gamma, lam)
                adv = masked_whiten(adv, mask)
                return dict(seq=seq, mask=mask, old_logp=old_logp, old_values=old_values, adv=adv, ret=ret)

            def ppo_update(exp, actor, critic, clip_eps, clip_vf, vf_coef):
                logp, values = actor.logprobs(exp["seq"]), critic(exp["seq"])
                loss, stats = ppo_loss(logp, exp["old_logp"], exp["adv"], values,
                                       exp["old_values"], exp["ret"], exp["mask"], clip_eps, clip_vf, vf_coef)
                loss.backward()
                return stats
          `,
        },
      ],
      explanation: md`
        **要点回顾**：
        1. KL 奖励用 **rollout 时**的 logp（\`old_logp\`），在 make_experience 阶段一次算好，之后若干个 PPO epoch 中保持不变；
        2. GAE 用 rollout 时的 \`old_values\`，padding 处置 0，否则垃圾值漏进最后一个 token；
        3. returns 用白化前的优势；白化只在有效 token 上统计；
        4. value 裁剪以 \`old_values\` 为中心；clipfrac 统计“裁剪项被选中”；masked mean 用 \`where\` 防 inf·0。

        这道题把「PPO 与 RLHF」关卡的两道实战串成了 OpenRLHF / TRL 中 make_experience + ppo_update 的数值骨架。
      `,
      complexity: { time: 'O(B·T)', space: 'O(B·T)' },
      hints: [
        md`先把「RLHF 奖励塑形 + 批量 GAE」那题的代码搬过来，注意这里奖励用 \`old_logp\`、GAE 用 \`old_values\`。`,
        md`白化：\`mu = adv[on].mean()\`，\`sd = adv[on].std()\`（ddof = 0），只改有效位置。`,
        md`PPO loss 部分与「手写 PPO loss」完全相同；advantages 用白化后的，returns 用白化前的。`,
      ],
      setup: py`
        import numpy as np

        def _ref_ppo_step(logp, old_logp, ref_logp, values, old_values, rm, mask, beta, gamma, lam, ce, cv, vc, whiten):
            B, T = mask.shape
            adv = np.zeros((B, T)); ret = np.zeros((B, T))
            for b in range(B):
                L = int(mask[b].sum())
                r = [-beta * (old_logp[b, t] - ref_logp[b, t]) for t in range(L)]
                r[-1] += rm[b]
                last = 0.0
                for t in reversed(range(L)):
                    nv = old_values[b, t + 1] if t + 1 < L else 0.0
                    last = r[t] + gamma * nv - old_values[b, t] + gamma * lam * last
                    adv[b, t] = last
                    ret[b, t] = last + old_values[b, t]
            vals = [adv[b, t] for b in range(B) for t in range(int(mask[b].sum()))]
            if whiten:
                mu, sd = np.mean(vals), np.std(vals)
                for b in range(B):
                    for t in range(int(mask[b].sum())):
                        adv[b, t] = (adv[b, t] - mu) / (sd + 1e-8)
            acc = {"pg": [], "vf": [], "cf": [], "kl": []}
            for b in range(B):
                for t in range(int(mask[b].sum())):
                    lr = logp[b, t] - old_logp[b, t]; r_ = np.exp(lr); A = adv[b, t]
                    p1, p2 = -A * r_, -A * min(max(r_, 1 - ce), 1 + ce)
                    vcl = old_values[b, t] + min(max(values[b, t] - old_values[b, t], -cv), cv)
                    acc["pg"].append(max(p1, p2)); acc["cf"].append(float(p2 > p1)); acc["kl"].append(r_ - 1 - lr)
                    acc["vf"].append(max((values[b, t] - ret[b, t]) ** 2, (vcl - ret[b, t]) ** 2))
            pg, vf = np.mean(acc["pg"]), 0.5 * np.mean(acc["vf"])
            return {"loss": pg + vc * vf, "pg_loss": pg, "vf_loss": vf, "clipfrac": np.mean(acc["cf"]),
                    "approx_kl": np.mean(acc["kl"]), "advantages": adv, "returns": ret}

        def _make_step(seed, B=4, T=8):
            rng = np.random.default_rng(seed)
            lens = rng.integers(1, T + 1, size=B); lens[0] = T
            mask = (np.arange(T)[None, :] < lens[:, None]).astype(np.int64)
            old = -rng.exponential(1.0, size=(B, T)); new = old + rng.normal(0, 0.3, size=(B, T))
            ref = old + rng.normal(0, 0.3, size=(B, T))
            ov = rng.normal(0.5, 0.5, size=(B, T)); v = ov + rng.normal(0, 0.3, size=(B, T))
            rm = rng.normal(0, 1.5, size=B)
            pad = mask == 0
            for x in (old, new, ref, ov, v):
                x[pad] = rng.uniform(-4, 4, size=pad.sum())
            return new, old, ref, v, ov, rm, mask

        def _cmp_step(got, exp, tag=""):
            assert isinstance(got, dict), "应返回 dict"
            for k in ("loss", "pg_loss", "vf_loss", "clipfrac", "approx_kl"):
                assert k in got, f"缺少键 {k}"
                assert abs(float(got[k]) - exp[k]) < 1e-8, f"{tag}{k}：期望 {exp[k]:.8f}，得到 {float(got[k]):.8f}"
            for k in ("advantages", "returns"):
                g = np.asarray(got[k])
                assert g.shape == exp[k].shape, f"{tag}{k} 形状应为 {exp[k].shape}，得到 {g.shape}"
                assert np.allclose(g, exp[k], atol=1e-8), f"{tag}{k} 最大误差 {np.abs(g - exp[k]).max():.3e}"
      `,
      tests: [
        {
          name: '不白化时与分步参考一致',
          code: py`
            args = _make_step(0)
            _cmp_step(ppo_step(*args, whiten=False), _ref_ppo_step(*args, 0.05, 1.0, 0.95, 0.2, 0.2, 0.1, False))
          `,
        },
        {
          name: '白化后的优势：有效位置均值 0、标准差 1',
          code: py`
            args = _make_step(1)
            out = ppo_step(*args)
            mask = args[-1] > 0
            a = np.asarray(out["advantages"])
            assert abs(a[mask].mean()) < 1e-9 and abs(a[mask].std() - 1) < 1e-6, f"均值 {a[mask].mean():.2e}，标准差 {a[mask].std():.6f}"
            assert np.allclose(a[~mask], 0), "padding 处的优势应为 0"
          `,
        },
        {
          name: 'returns 用白化前的优势',
          code: py`
            args = _make_step(2)
            a = ppo_step(*args, whiten=True)["returns"]; b = ppo_step(*args, whiten=False)["returns"]
            assert np.allclose(a, b), "returns 不应受白化影响"
          `,
        },
        {
          name: '完整对拍（白化、不同超参）',
          hidden: true,
          code: py`
            for s, (beta, g, l, ce, cv, vc) in enumerate([(0.05, 1.0, 0.95, 0.2, 0.2, 0.1), (0.1, 0.99, 0.9, 0.1, 0.5, 1.0), (0.0, 1.0, 1.0, 0.3, 10.0, 0.5)]):
                args = _make_step(10 + s, B=5, T=9)
                got = ppo_step(*args, beta=beta, gamma=g, lam=l, clip_eps=ce, clip_vf=cv, vf_coef=vc, whiten=True)
                _cmp_step(got, _ref_ppo_step(*args, beta, g, l, ce, cv, vc, True), f"seed={10+s}: ")
          `,
        },
        {
          name: 'padding 处的垃圾值不影响结果',
          hidden: true,
          code: py`
            args = list(_make_step(20))
            out1 = ppo_step(*args)
            pad = args[-1] == 0
            for i in (0, 1, 2, 3, 4):
                args[i] = args[i].copy(); args[i][pad] += 3.0
            out2 = ppo_step(*args)
            for k in ("loss", "pg_loss", "vf_loss", "clipfrac", "approx_kl"):
                assert abs(out1[k] - out2[k]) < 1e-10, f"{k} 受到了 padding 垃圾值的影响"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-multi-epoch',
          q: md`同一批 experience 训练多个 PPO epoch 时，哪些量变、哪些不变？`,
          a: md`
            不变：old_logp、ref_logp、rewards、old_values、advantages、returns（都在 make_experience 时固定）。变：logp 与 values（每个 mini-batch 用当前参数重新前向），因此 ratio、clipfrac、approx_kl 会随 epoch 增大。若 approx_kl 超阈值就早停本批剩余更新。
          `,
        },
      ],
    },
    {
      id: 'c-pref-family',
      title: '综合：一个函数实现 DPO / cDPO / IPO / length-normalized DPO / SimPO',
      difficulty: 'hard',
      source: '大模型岗手撕 · 偏好优化综合',
      timeLimitMin: 40,
      description: md`
        参照 TRL 的 \`loss_type\` 设计，实现统一的偏好损失与训练指标：

        \`\`\`python
        def preference_loss(policy_logits, ref_logits, labels, loss_type="dpo",
                            beta=0.1, gamma=0.0, label_smoothing=0.0) -> dict: ...
        \`\`\`

        数据约定同「DPO」关卡：\`(2B, T, V)\` logits，前 $B$ 行 chosen、后 $B$ 行 rejected；\`labels\` 未移位，-100 不计入。记 $S$ 为回复 token 的 logp **之和**，$\bar S=S/|y|$ 为**平均**（$|y|$ 为计入的 token 数）。\`ref_logits\` 在 \`simpo\` 时可能为 \`None\`。

        | loss_type | 每对的 logit / 损失 | 隐式奖励（chosen 与 rejected 各自） |
        |---|---|---|
        | \`dpo\` | $z=\beta[(S_w-S_w^{\mathrm{ref}})-(S_l-S_l^{\mathrm{ref}})]$，$\ell=-(1-\epsilon)\log\sigma(z)-\epsilon\log\sigma(-z)$ | $\beta(S-S^{\mathrm{ref}})$ |
        | \`ipo\` | $h=(S_w-S_w^{\mathrm{ref}})-(S_l-S_l^{\mathrm{ref}})$，$\ell=(h-\frac{1}{2\beta})^2$（β 即 τ） | $\beta(S-S^{\mathrm{ref}})$ |
        | \`ln_dpo\` | $z=\beta[(\bar S_w-\bar S_w^{\mathrm{ref}})-(\bar S_l-\bar S_l^{\mathrm{ref}})]$，$\ell=-\log\sigma(z)$ | $\beta(\bar S-\bar S^{\mathrm{ref}})$ |
        | \`simpo\` | $z=\beta(\bar S_w-\bar S_l)-\gamma$，$\ell=-\log\sigma(z)$ | $\beta\bar S$ |

        （\`label_smoothing\` ε 只作用于 \`dpo\`，即 conservative DPO。）返回 dict（均为 Python float）：
        \`loss\`（各对平均）、\`rewards/chosen\`、\`rewards/rejected\`（各自的 batch 平均）、\`rewards/margins\`（chosen − rejected 的平均）、\`rewards/accuracies\`（chosen 奖励**严格大于** rejected 的比例）。

        要求数值稳定（logits 可达 $\pm10^4$）。
      `,
      starter: py`
        import numpy as np

        def preference_loss(policy_logits, ref_logits, labels, loss_type="dpo",
                            beta=0.1, gamma=0.0, label_smoothing=0.0):
            # 1) 序列 logp 之和与计入 token 数（policy / ref）
            # 2) 按 loss_type 计算奖励与每对的损失
            # 3) 汇总指标
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def _seq_stats(logits, labels):
            lg, lab = logits[:, :-1, :], labels[:, 1:]
            mask = lab != -100
            m = lg.max(axis=-1, keepdims=True)
            lp = lg - m - np.log(np.exp(lg - m).sum(axis=-1, keepdims=True))
            tok = np.take_along_axis(lp, np.where(mask, lab, 0)[..., None], axis=-1)[..., 0]
            return np.where(mask, tok, 0.0).sum(-1), mask.sum(-1)        # (2B,), (2B,)

        def _nls(x):                                                     # -log σ(x)，稳定
            return np.logaddexp(0.0, -x)

        def preference_loss(policy_logits, ref_logits, labels, loss_type="dpo",
                            beta=0.1, gamma=0.0, label_smoothing=0.0):
            B = labels.shape[0] // 2
            S, n = _seq_stats(policy_logits, labels)
            if loss_type == "simpo":
                r = beta * S / n                                         # 无 reference
                z = r[:B] - r[B:] - gamma
                losses = _nls(z)
            else:
                R, _ = _seq_stats(ref_logits, labels)
                ratio = S - R
                if loss_type == "ln_dpo":
                    ratio = ratio / n
                r = beta * ratio
                h = ratio[:B] - ratio[B:]
                if loss_type == "ipo":
                    losses = (h - 1.0 / (2.0 * beta)) ** 2
                elif loss_type in ("dpo", "ln_dpo"):
                    z = beta * h
                    eps = label_smoothing if loss_type == "dpo" else 0.0
                    losses = (1 - eps) * _nls(z) + eps * _nls(-z)
                else:
                    raise ValueError(f"unknown loss_type {loss_type}")
            rc, rr = r[:B], r[B:]
            return {
                "loss": float(losses.mean()),
                "rewards/chosen": float(rc.mean()),
                "rewards/rejected": float(rr.mean()),
                "rewards/margins": float((rc - rr).mean()),
                "rewards/accuracies": float((rc > rr).mean()),
            }
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示，核心分支）',
          code: py`
            import torch.nn.functional as F

            def pref_losses(pi_c, pi_r, ref_c, ref_r, n_c, n_r, loss_type, beta, gamma=0.0, eps=0.0):
                if loss_type == "simpo":
                    z = beta * (pi_c / n_c - pi_r / n_r) - gamma
                    return -F.logsigmoid(z)
                lr_c, lr_r = pi_c - ref_c, pi_r - ref_r
                if loss_type == "ln_dpo":
                    lr_c, lr_r = lr_c / n_c, lr_r / n_r
                h = lr_c - lr_r
                if loss_type == "ipo":
                    return (h - 1 / (2 * beta)) ** 2
                return -(1 - eps) * F.logsigmoid(beta * h) - eps * F.logsigmoid(-beta * h)
          `,
        },
      ],
      explanation: md`
        **结构**：所有方法共享“序列 logp（和 / 平均）→ 两两作差 → 一个标量函数”的骨架，区别只在三处：是否减 reference、是否除以长度、外层函数（logistic / 平方 / 带 margin / 带 label smoothing）。

        **易错点**：
        - SimPO 不能读 \`ref_logits\`（可能是 None）；
        - length-normalized 的长度是**各自**的计入 token 数，不是 batch 最大长度；
        - IPO 的目标是 $\frac{1}{2\beta}$，平方损失没有 σ；
        - cDPO：$-(1-\epsilon)\log\sigma(z)-\epsilon\log\sigma(-z)$，ε = 0 退化为 DPO；
        - accuracies 用严格大于，第一步（policy = ref）应为 0。
      `,
      complexity: { time: 'O(N·T·V)', space: 'O(N·T·V)' },
      hints: [
        md`写一个 \`_seq_stats(logits, labels)\` 同时返回 logp 之和 S 与计入 token 数 n。`,
        md`\`simpo\` 分支先处理（不碰 ref）；其余分支 \`ratio = S - R\`，\`ln_dpo\` 再除以 n。`,
        md`IPO：\`(h - 1/(2*beta))**2\`；DPO/cDPO：\`(1-ε)·softplus(-z) + ε·softplus(z)\`，softplus 用 \`logaddexp(0, ·)\`。`,
      ],
      setup: PY_REF_LOGPS + '\n' + PY_MAKE_PAIRS + '\n' + py`
        def _ref_pref(pol, ref, labels, lt, beta, gamma, eps):
            B = labels.shape[0] // 2
            tok, mask = _ref_token_logps(pol, labels)
            S, n = tok.sum(-1), mask.sum(-1)
            if lt == "simpo":
                r = beta * S / n
                z = r[:B] - r[B:] - gamma
                L = -_ref_log_sigmoid(z)
            else:
                tr, _ = _ref_token_logps(ref, labels)
                ratio = S - tr.sum(-1)
                if lt == "ln_dpo":
                    ratio = ratio / n
                r = beta * ratio
                h = ratio[:B] - ratio[B:]
                if lt == "ipo":
                    L = (h - 1 / (2 * beta)) ** 2
                else:
                    e = eps if lt == "dpo" else 0.0
                    L = -(1 - e) * _ref_log_sigmoid(beta * h) - e * _ref_log_sigmoid(-beta * h)
            rc, rr = r[:B], r[B:]
            return {"loss": L.mean(), "rewards/chosen": rc.mean(), "rewards/rejected": rr.mean(),
                    "rewards/margins": (rc - rr).mean(), "rewards/accuracies": (rc > rr).mean()}

        def _cmp_pref(got, exp, tag=""):
            assert isinstance(got, dict), "应返回 dict"
            for k, e in exp.items():
                assert k in got, f"缺少键 {k}"
                g = float(got[k])
                assert np.isfinite(g), f"{tag}{k} 不是有限数：{g}"
                assert abs(g - e) <= 1e-7 * max(1.0, abs(e)), f"{tag}{k}：期望 {e:.8f}，得到 {g:.8f}"
      `,
      tests: [
        {
          name: 'policy == ref：DPO loss = ln2，accuracies = 0',
          code: py`
            pol, _, labels = _make_pairs(1)
            out = preference_loss(pol, pol.copy(), labels, "dpo", beta=0.1)
            assert abs(out["loss"] - np.log(2)) < 1e-9, f"期望 ln2，得到 {out['loss']}"
            assert out["rewards/accuracies"] == 0.0 and abs(out["rewards/margins"]) < 1e-12, f"得到 {out}"
          `,
        },
        {
          name: 'dpo 与 cDPO（label smoothing）',
          code: py`
            pol, ref, labels = _make_pairs(2)
            for eps in (0.0, 0.1):
                _cmp_pref(preference_loss(pol, ref, labels, "dpo", beta=0.1, label_smoothing=eps),
                          _ref_pref(pol, ref, labels, "dpo", 0.1, 0.0, eps), f"dpo ε={eps}: ")
          `,
        },
        {
          name: 'ipo / ln_dpo',
          code: py`
            pol, ref, labels = _make_pairs(3)
            _cmp_pref(preference_loss(pol, ref, labels, "ipo", beta=0.1), _ref_pref(pol, ref, labels, "ipo", 0.1, 0.0, 0.0), "ipo: ")
            _cmp_pref(preference_loss(pol, ref, labels, "ln_dpo", beta=5.0), _ref_pref(pol, ref, labels, "ln_dpo", 5.0, 0.0, 0.0), "ln_dpo: ")
          `,
        },
        {
          name: 'simpo（ref_logits = None）',
          code: py`
            pol, _, labels = _make_pairs(4)
            _cmp_pref(preference_loss(pol, None, labels, "simpo", beta=2.5, gamma=1.4), _ref_pref(pol, None, labels, "simpo", 2.5, 1.4, 0.0), "simpo: ")
          `,
        },
        {
          name: '随机对拍（所有类型、不同超参）',
          hidden: true,
          code: py`
            cases = [("dpo", 0.5, 0.0, 0.2), ("ipo", 0.5, 0.0, 0.0), ("ipo", 0.05, 0.0, 0.0), ("ln_dpo", 2.0, 0.0, 0.0), ("simpo", 2.0, 1.0, 0.0), ("simpo", 10.0, 3.0, 0.0)]
            for s, (lt, beta, gamma, eps) in enumerate(cases):
                pol, ref, labels = _make_pairs(40 + s, B=4, T=10, V=8)
                _cmp_pref(preference_loss(pol, None if lt == "simpo" else ref, labels, lt, beta=beta, gamma=gamma, label_smoothing=eps),
                          _ref_pref(pol, ref, labels, lt, beta, gamma, eps), f"{lt} β={beta}: ")
          `,
        },
        {
          name: '数值稳定：logits ±1e4',
          hidden: true,
          code: py`
            pol, _, labels = _make_pairs(50, B=3, T=8, V=6, scale=3000.0)
            ref = np.random.default_rng(51).normal(0, 3000.0, size=pol.shape)
            for lt in ("dpo", "ln_dpo", "simpo"):
                _cmp_pref(preference_loss(pol, None if lt == "simpo" else ref, labels, lt, beta=0.5, gamma=0.5, label_smoothing=0.1 if lt == "dpo" else 0.0),
                          _ref_pref(pol, ref, labels, lt, 0.5, 0.5, 0.1 if lt == "dpo" else 0.0), f"大数值 {lt}: ")
          `,
        },
      ],
      followUps: [
        {
          id: 'f-which',
          q: md`如果让你在新项目里从这几个 loss 中选一个作为默认，你怎么选、怎么验证？`,
          a: md`
            默认从 DPO（β = 0.1）起步：最成熟、超参少、有 reference 约束。验证：第一步 loss = ln2；监控 rewards/margins 上升、accuracies 上升、chosen logp 是否大幅下降、输出长度是否膨胀。若出现长度膨胀 → 试 length-normalized DPO 或 SimPO；若对确定性偏好过拟合（margin 飙升、生成退化）→ IPO 或加大 β / label smoothing；数据有噪声 → cDPO / Robust DPO。最终以独立评估（人工、长度控制的 win rate、下游任务）选型，而不是训练 loss。
          `,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-design',
      freq: 3,
      q: md`给你一个 7B base 模型、一批通用对话偏好数据和一批带标准答案的数学题，设计后训练方案。`,
      a: md`
        **结论**：SFT 冷启动 → DPO 系做通用偏好对齐 → GRPO 系在数学上做 RLVR，必要时迭代。

        **步骤**：① SFT：chat template + 只训 assistant 回复 + packing，用偏好数据的 chosen 或另外的示范数据，1–3 个 epoch；② DPO：β ≈ 0.1，reference = SFT 模型（logp 预计算），监控 ln2 起点、margins、chosen logp 与长度，必要时加 NLL 正则或改 length-normalized；③ 数学：在线采样多个解答，用答案匹配给 0/1 奖励，GRPO / DAPO 训练（组内基线、无 critic、KL 可小或去掉）；④ 若 DPO 数据与模型分布差距大，做 2–3 轮 iterative DPO。

        **延伸**：显存估算（7B：DPO 训练态约 112 GB + ref 14 GB；GRPO 还有 rollout 引擎）；评估用长度控制的 win rate 与数学准确率；警惕 reward hacking 与通用能力回退（alignment tax）。
      `,
    },
    {
      id: 'qa-compare',
      freq: 3,
      q: md`PPO、DPO、GRPO 三者的核心区别是什么？`,
      a: md`
        **结论**：PPO 是带 critic 的在线 RL；DPO 是离线的分类损失（隐式奖励）；GRPO 是去掉 critic、用组内相对奖励作基线的在线 RL。

        **对比**：模型数——PPO 四个、DPO 两个（ref 可预计算）、GRPO 两个（policy + 可选 ref）；数据——PPO / GRPO 需要奖励信号（RM 或验证器）+ 在线采样，DPO 需要成对偏好；信用分配——PPO 用 GAE 逐 token，GRPO 对整条回答一个优势、广播到每个 token，DPO 没有显式的 token 级信用分配；稳定性与成本——DPO 最简单便宜，PPO 最重，GRPO 介于两者之间、在可验证任务上效果最好。

        **延伸**：DPO 可以看作“reward model + RL 的闭式解捷径”；GRPO 的细节（std 归一化、长度偏差、clip-higher、动态采样）见「推理强化学习」。
      `,
    },
  ],

  cheatsheet: md`
    **全章一页纸**

    - SFT：labels -100（含 prompt / 模板头 / pad），结束符训练；\`logits[:, :-1]\` ↔ \`labels[:, 1:]\`；packing 三件事；token 平均 + 全局归一化
    - RM：$-\log\sigma(r_w-r_l-m)$；最后有效 token；K 路 $\binom K2$ 同 batch；平移不变；PRM 连乘；judge 四类偏差
    - PPO：$r_t=-\beta\mathrm{KL}_t+\mathbb 1[t=T]r_\phi$；$\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}$；$\max(-rA,-\mathrm{clip}(r)A)$；value clip 取 max；白化只在有效 token
    - DPO：$\pi^*\propto\pi_{\mathrm{ref}}e^{r/\beta}$ → $r=\beta\log\frac{\pi^*}{\pi_{\mathrm{ref}}}+\beta\log Z$ → Z 相消；第一步 ln2；梯度权重 $\sigma(-\beta\Delta)$
    - 变体：IPO $(h-\frac1{2\tau})^2$；SimPO $\frac{\beta}{|y|}\log\pi-\gamma$ 无 ref；KTO $\lambda_y-v$、$z_0$ 错位配对；ORPO SFT + odds ratio
    - 数字：InstructGPT β = 0.02、γ_ptx = 27.8、ε = 0.2、K = 4–9；Llama 2 β = 0.01 / 0.005；Llama 3 DPO β = 0.1 + 0.2·NLL；Tülu 3 ln-DPO β = 5；SimPO β 2–2.5 / γ 0.5–1.5
  `,
})
