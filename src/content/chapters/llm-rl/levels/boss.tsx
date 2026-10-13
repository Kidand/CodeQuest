/**
 * llm-rl-boss：推理强化学习综合关
 */
import { defineLevel, md, py } from '@/content/helpers'
import { PY_MAKE_BATCH, PY_REF_ADV } from '../shared'

const PY_REF_BOSS = py`
  import re
  from fractions import Fraction

  def _ref_policy_loss(algo, logp_new, logp_old, rewards, group_ids, mask, max_len=None):
      m = np.asarray(mask, dtype=np.float64)
      d = np.where(m > 0, logp_new - logp_old, 0.0)
      if algo == "gspo":
          A = _ref_group_adv(rewards, group_ids, "group")
          s = np.exp(d.sum(1) / m.sum(1))
          return float(np.mean(-np.minimum(s * A, np.clip(s, 1 - 3e-4, 1 + 4e-4) * A)))
      A = _ref_group_adv(rewards, group_ids, "none" if algo == "dr_grpo" else "group")[:, None]
      ratio = np.exp(d)
      if algo == "cispo":
          w = np.minimum(ratio, 1.28)
          lp = np.where(m > 0, logp_new, 0.0)
          return float(-(w * A * lp * m).sum() / m.sum())
      hi = 1.28 if algo == "dapo" else 1.2
      tok = -np.minimum(ratio * A, np.clip(ratio, 0.8, hi) * A) * m
      if algo == "grpo":
          return float(np.mean(tok.sum(1) / m.sum(1)))
      if algo == "dapo":
          return float(tok.sum() / m.sum())
      if algo == "dr_grpo":
          L = m.shape[1] if max_len is None else max_len
          return float(np.mean(tok.sum(1)) / L)
      raise ValueError(algo)

  def _ref_last_boxed(s):
      i = s.rfind("\\boxed{")
      if i < 0:
          return None
      j, depth = i + 7, 1
      while j < len(s) and depth:
          depth += {"{": 1, "}": -1}.get(s[j], 0)
          j += 1
      return s[i + 7:j - 1] if depth == 0 else None

  def _ref_norm(a):
      a = a.replace(" ", "").replace("$", "")
      a = re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", a)
      try:
          return Fraction(a)
      except (ValueError, ZeroDivisionError):
          return a

  def _ref_rlvr(responses, answers, lengths, L_max=20480, L_cache=4096):
      correct = []
      for s, ans in zip(responses, answers):
          ok = False
          b = _ref_last_boxed(s)
          t0, t1 = s.find("<think>"), s.find("</think>")
          fmt = t0 >= 0 and t1 > t0 and b is not None and s.rfind("\\boxed{") > t1
          if fmt:
              ok = _ref_norm(b) == _ref_norm(ans)
          correct.append(ok)
      correct = np.array(correct, dtype=bool)
      L = np.asarray(lengths, dtype=np.float64)
      soft = L_max - L_cache
      r_len = np.where(L <= soft, 0.0, np.where(L <= L_max, (soft - L) / L_cache, -1.0))
      return np.where(correct, 1.0, -1.0) + r_len, correct
`

const SETUP = PY_MAKE_BATCH + '\n\n' + PY_REF_ADV + '\n\n' + PY_REF_BOSS

export default defineLevel({
  learn: [
    {
      id: 'recap',
      title: '复盘：一张表看懂 RLVR 算法谱系',
      body: md`
        所有算法都可以拆成同样的五个旋钮：**优势怎么算、比值是什么、怎么裁剪、loss 怎么聚合、要不要 KL**。

        | 算法 | 优势 | 比值 | 裁剪 | 聚合 | KL |
        |---|---|---|---|---|---|
        | PPO-RLHF | GAE + critic | token $r_{t}$ | $[1\pm0.2]$ | （实现中多为行平均） | 逐 token 进奖励 |
        | GRPO | 组内 $(r-\bar r)/\sigma$ | token | $[1\pm\varepsilon]$ | 行内平均 → 行间平均 | k3 进 loss |
        | Dr.GRPO | 组内 $r-\bar r$ | token | $[1\pm\varepsilon]$ | 行内求和 / 常数 | 可选 |
        | DAPO | 组内 $(r-\bar r)/\sigma$ + 动态采样 | token | $[1-0.2,\ 1+0.28]$ | token-mean | 无 |
        | GSPO | 组内 $(r-\bar r)/\sigma$ | 序列几何平均 $s_i$ | $[1-3\text{e-}4,\ 1+4\text{e-}4]$ | 回答平均 | 略去 |
        | CISPO | 组内 $(r-\bar r)/\sigma$ | $\mathrm{sg}(\mathrm{clip}(r))$ 作权重 | 只截权重不丢梯度 | token-mean | 无 |
        | REINFORCE++ | 全局 batch 归一化 | token | PPO 式 | — | k1 进奖励 / k2 进 loss |
        | On-policy 蒸馏 | $-(\log\pi_\theta-\log\pi_T)$ 逐 token | IS | — | token | 本身就是反向 KL |

        **工程层的补丁**：训推不一致用 TIS / MIS（$\min(\pi_{train}/\pi_{infer},C)$），MoE 用 Routing Replay / Keep Routing，
        top-p 采样用 Keep Sampling Mask，LM head 用 FP32 或整体 FP16；超长样本用软惩罚或过滤；agent 轨迹 mask 掉工具返回。

        **怎么选**：dense 小模型 + 数学 → DAPO / Dr.GRPO 的组合足够；MoE 或长序列不稳 → GSPO（或 Keep Routing）；
        反思 token 学不出来 → CISPO / Clip-Higher；已有强领域专家要合并 → on-policy 蒸馏。
      `,
    },
  ],

  quiz: [
    {
      id: 'b-gspo',
      type: 'single',
      difficulty: 1,
      prompt: md`以下哪个算法在**序列级**计算重要性比并做裁剪？`,
      options: ['GRPO', 'DAPO', 'GSPO', 'CISPO'],
      answer: 2,
      explanation: md`GSPO 用 $s_i=(\pi_\theta(y_i)/\pi_{old}(y_i))^{1/|y_i|}$ 并在序列上裁剪。其余三者都是 token 级比值；CISPO 裁剪的是 token 级 IS 权重。`,
    },
    {
      id: 'b-kl',
      type: 'multi',
      difficulty: 2,
      prompt: md`按各自原论文的设定，哪些算法**不使用** KL 项？（多选）`,
      options: ['DAPO', 'CISPO', 'DeepSeekMath 的 GRPO', 'DeepSeek-R1-Zero 的 GRPO', 'DeepSeek-V3.2 的 GRPO'],
      answer: [0, 1],
      explanation: md`DAPO、CISPO 明确去掉 KL。DeepSeekMath β=0.04；R1-Zero β=0.001；DeepSeek-V3.2 保留 KL 并做了 IS 修正（只是说部分领域弱 KL 或不用更好）。`,
    },
    {
      id: 'b-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把算法与它最核心的改动配对：`,
      pairs: [
        ['GRPO', '用组内奖励标准化代替 critic'],
        ['Dr.GRPO', '去掉 std 与 1/|o_i| 两个归一化'],
        ['DAPO', 'Clip-Higher + 动态采样 + token 级 loss + 超长整形'],
        ['GSPO', '序列级重要性比与裁剪'],
        ['CISPO', '裁剪 IS 权重并 stop-gradient'],
      ],
      explanation: md`五个名字对应五句话，是 RL 方向面试的高频开场。`,
    },
    {
      id: 'b-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`DAPO（$\varepsilon_{low}=0.2,\ \varepsilon_{high}=0.28$）下，哪些 token 的梯度被 clip 截断为 0？`,
      items: ['A>0, r=1.25', 'A>0, r=1.30', 'A<0, r=0.75', 'A<0, r=0.85', 'A>0, r=0.70'],
      answer: [1, 2],
      explanation: md`A>0 时上界 1.28：只有 1.30 越界；A<0 时下界 0.8：只有 0.75 越界。A>0、r=0.7 在不利方向，min 选未裁剪项，仍有梯度。`,
    },
    {
      id: 'b-overlong',
      type: 'numeric',
      difficulty: 2,
      prompt: md`DAPO 设置（$L_{max}=20480,\ L_{cache}=4096$，奖励 ±1）。一个**答对**、长度 19456 的回答，总奖励是多少？`,
      answer: 0.25,
      tolerance: 0.001,
      explanation: md`$R_{length}=(16384-19456)/4096=-0.75$，总奖励 $1-0.75=0.25$。`,
    },
    {
      id: 'b-passk',
      type: 'numeric',
      difficulty: 2,
      prompt: md`$n=4$ 个样本中 1 个正确，pass@2 的无偏估计是多少？`,
      answer: 0.5,
      tolerance: 0.001,
      explanation: md`$1-\binom32/\binom42=1-3/6=0.5$；连乘：$1-(1-\frac24)=0.5$。`,
    },
    {
      id: 'b-order',
      type: 'order',
      difficulty: 2,
      prompt: md`按提出时间从早到晚排列：`,
      items: ['PPO', 'GRPO（DeepSeekMath）', 'DeepSeek-R1', 'DAPO', 'CISPO（MiniMax-M1）', 'GSPO', 'Thinking Machines 的 on-policy 蒸馏博客'],
      explanation: md`2017 → 2024-02 → 2025-01 → 2025-03 → 2025-06 → 2025-07 → 2025-10。`,
    },
    {
      id: 'b-opd',
      type: 'judge',
      difficulty: 1,
      prompt: md`on-policy 蒸馏中，教师需要为每个 prompt 生成完整回答，学生再模仿这些回答。`,
      answer: false,
      explanation: md`那是 off-policy 蒸馏。on-policy 蒸馏由学生采样，教师只对学生轨迹做一次前向，给出逐 token 的对数概率（或全词表分布）。`,
    },
    {
      id: 'b-bug',
      type: 'bug',
      difficulty: 3,
      prompt: md`下面的 GRPO loss（含 KL）有 **2 处** bug，点出来：`,
      code: py`
        def grpo_loss(lp_new, lp_old, lp_ref, adv, mask, eps=0.2, beta=0.04):
            A = adv[:, None]
            ratio = np.exp(lp_new - lp_old)
            surr = np.minimum(ratio * A, np.clip(ratio, 1 - eps, 1 + eps) * A)
            d = lp_new - lp_ref
            kl = np.exp(d) - d - 1
            tok = -surr - beta * kl
            per_seq = (tok * mask).sum(axis=1) / mask.sum(axis=1)
            return per_seq.mean()
      `,
      bugLines: [5, 7],
      explanation: md`第 5 行方向反了：k3 的 $x=\pi_{ref}/\pi_\theta$，应为 \`d = lp_ref - lp_new\`（写反后虽然仍非负，但估计的不再是 KL(π_θ‖π_ref)）。第 7 行 KL 应**加**在 loss 上：\`tok = -surr + beta * kl\`，减号会鼓励远离参考模型。`,
    },
    {
      id: 'b-fill',
      type: 'fill',
      difficulty: 2,
      prompt: md`GSPO 论文实验中序列级裁剪的右侧范围是 ____(1)____；DeepSeek-R1-Zero 每题采样 ____(2)____ 个输出。`,
      blanks: [
        { label: '(1)', answers: ['4e-4'], numeric: true, tolerance: 0.001 },
        { label: '(2)', answers: ['16'], numeric: true, tolerance: 0 },
      ],
      explanation: md`GSPO：左 3e-4、右 4e-4；R1-Zero：G=16，β=0.001，学习率 3e-6。`,
    },
    {
      id: 'b-bias',
      type: 'single',
      difficulty: 3,
      prompt: md`一个团队把 GRPO 换成 DAPO 的 token-mean 聚合后，发现“错误回答越来越长”的现象明显缓解。最合理的解释是：`,
      options: [
        'token-mean 去掉了组内 std 归一化',
        'token-mean 让长回答中每个 token 的惩罚不再被 1/|o_i| 稀释',
        'token-mean 自动过滤了截断样本',
        'token-mean 等价于给 KL 加权',
      ],
      answer: 1,
      explanation: md`GRPO 的 $1/|o_i|$ 让错误长回答每个 token 罚得更轻（Dr.GRPO 的长度偏差）；token-mean 让 batch 内 token 等权。A 是另一个偏差（DAPO 仍除 std）；C 是超长过滤；D 无关。`,
    },
    {
      id: 'b-parsons',
      type: 'parsons',
      difficulty: 3,
      prompt: md`拼出“取最后一个 \`\boxed{...}\` 的内容（支持嵌套花括号）”的函数（有 2 行干扰项）：`,
      lines: [
        'def last_boxed(s):',
        '    i = s.rfind("\\\\boxed{")',
        '    if i < 0:',
        '        return None',
        '    j, depth = i + 7, 1',
        '    while j < len(s) and depth:',
        '        depth += {"{": 1, "}": -1}.get(s[j], 0)',
        '        j += 1',
        '    return s[i + 7:j - 1] if depth == 0 else None',
      ],
      distractors: ['    i = s.find("\\\\boxed{")', '    return s[i + 7:j] if depth == 0 else None'],
      explanation: md`取**最后一个** boxed（\`rfind\`）是防 hacking 的关键：只认最终答案，输出多个候选不会因为“其中一个对”而得分。括号计数处理 \`\frac{1}{2}\` 这类嵌套；循环结束时 \`j\` 已越过右括号，所以切片到 \`j - 1\`。`,
    },
  ],

  guided: [],

  challenges: [
    {
      id: 'c-unified',
      title: '一个函数实现五种 RLVR loss',
      difficulty: 'hard',
      source: 'BOSS · GRPO / Dr.GRPO / DAPO / GSPO / CISPO',
      timeLimitMin: 35,
      description: md`
        \`\`\`python
        def policy_loss(algo, logp_new, logp_old, rewards, group_ids, mask, max_len=None) -> float: ...
        \`\`\`

        输入约定同前几关：\`(B, T)\` 的 logp 与 mask（padding 是垃圾值），\`(B,)\` 的 rewards / group_ids（可乱序，每组 ≥ 2）。
        优势 std 一律 ddof=1、分母加 1e-6。本题**都不加 KL**。各算法的设定：

        | algo | 优势 | 目标（取负为 loss） | 聚合 |
        |---|---|---|---|
        | \`"grpo"\` | $(r-\bar r)/\sigma$ | token 级 $\min(r\hat A,\mathrm{clip}(r,0.8,1.2)\hat A)$ | 行内平均 → 行间平均 |
        | \`"dr_grpo"\` | $r-\bar r$ | 同上 | 行内求和 → 行间平均 → 除以 \`max_len\`（None 时用 T） |
        | \`"dapo"\` | $(r-\bar r)/\sigma$ | token 级，clip 到 $[0.8, 1.28]$ | 全部有效 token 平均 |
        | \`"gspo"\` | $(r-\bar r)/\sigma$ | 序列级 $s_i$，clip 到 $[1-3\text{e-}4, 1+4\text{e-}4]$ | 对 B 行平均 |
        | \`"cispo"\` | $(r-\bar r)/\sigma$ | $\mathrm{sg}(\min(r,1.28))\cdot\hat A\cdot\log\pi_{new}$ | 全部有效 token 平均 |

        未知 algo 抛 \`ValueError\`。返回 Python float。
      `,
      starter: py`
        import numpy as np

        def policy_loss(algo, logp_new, logp_old, rewards, group_ids, mask, max_len=None):
            # 建议拆成：group_adv(...)、agg(...) 两个辅助函数
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def _group_adv(r, group_ids, use_std):
            _, inv = np.unique(group_ids, return_inverse=True)
            inv = inv.reshape(-1)
            cnt = np.bincount(inv).astype(np.float64)
            dev = r - (np.bincount(inv, weights=r) / cnt)[inv]
            if not use_std:
                return dev
            std = np.sqrt(np.bincount(inv, weights=dev ** 2) / (cnt - 1))
            return dev / (std[inv] + 1e-6)

        def policy_loss(algo, logp_new, logp_old, rewards, group_ids, mask, max_len=None):
            if algo not in ("grpo", "dr_grpo", "dapo", "gspo", "cispo"):
                raise ValueError(f"unknown algo: {algo}")
            m = np.asarray(mask, dtype=np.float64)
            r = np.asarray(rewards, dtype=np.float64)
            d = np.where(m > 0, logp_new - logp_old, 0.0)          # padding 清零
            A = _group_adv(r, group_ids, use_std=(algo != "dr_grpo"))

            if algo == "gspo":                                      # 序列级
                s = np.exp(d.sum(1) / m.sum(1))
                return float(np.mean(-np.minimum(s * A, np.clip(s, 1 - 3e-4, 1 + 4e-4) * A)))

            A = A[:, None]
            ratio = np.exp(d)
            if algo == "cispo":                                     # 截断权重，保留梯度
                w = np.minimum(ratio, 1.28)
                lp = np.where(m > 0, logp_new, 0.0)
                return float(-(w * A * lp * m).sum() / m.sum())

            hi = 1.28 if algo == "dapo" else 1.2
            tok = -np.minimum(ratio * A, np.clip(ratio, 0.8, hi) * A) * m
            if algo == "grpo":
                return float(np.mean(tok.sum(1) / m.sum(1)))
            if algo == "dapo":
                return float(tok.sum() / m.sum())
            L = m.shape[1] if max_len is None else max_len           # dr_grpo
            return float(np.mean(tok.sum(1)) / L)
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示：连续分组的统一实现骨架）',
          code: py`
            import torch

            def policy_loss_torch(algo, logp_new, logp_old, rewards, mask, G, max_len=None):
                mask = mask.float()
                r = rewards.view(-1, G)
                A = r - r.mean(-1, keepdim=True)
                if algo != "dr_grpo":
                    A = A / (r.std(-1, keepdim=True) + 1e-6)
                A = A.reshape(-1).detach()
                d = (logp_new - logp_old.detach()) * mask
                if algo == "gspo":
                    s = torch.exp(d.sum(-1) / mask.sum(-1))
                    return -torch.min(s * A, s.clamp(1 - 3e-4, 1 + 4e-4) * A).mean()
                ratio, A = d.exp(), A[:, None]
                if algo == "cispo":
                    w = ratio.clamp(max=1.28).detach()
                    return -(w * A * logp_new * mask).sum() / mask.sum()
                hi = 1.28 if algo == "dapo" else 1.2
                tok = -torch.min(ratio * A, ratio.clamp(0.8, hi) * A) * mask
                if algo == "grpo":
                    return (tok.sum(-1) / mask.sum(-1)).mean()
                if algo == "dapo":
                    return tok.sum() / mask.sum()
                return tok.sum(-1).mean() / (max_len or mask.shape[-1])
          `,
        },
      ],
      explanation: md`
        把差异拆成正交的旋钮：**优势**（是否除 std）、**比值**（token / 序列 / 截断权重）、**聚合**（行平均 / token 平均 / 常数）。
        面试时先写公共部分（分组统计、padding 清零），再用分支处理差异，比写五个独立函数更能体现对算法谱系的理解。

        **易错点**：GSPO 的比值要先在 log 空间按有效长度平均；CISPO 的目标里是 \`logp_new\` 本身（不是 ratio），权重视为常数；Dr.GRPO 不除 std 且除以常数。
      `,
      hints: [
        md`公共部分：\`d = np.where(mask > 0, logp_new - logp_old, 0)\`；分组优势用 \`np.unique\` + \`np.bincount\`。`,
        md`GSPO 单独处理：\`s = exp(d.sum(1) / mask.sum(1))\`，loss 对 B 行平均。`,
        md`token 级算法：先算 \`tok\`，再按 algo 选聚合方式。`,
      ],
      setup: SETUP,
      tests: [
        {
          name: '未知 algo 抛错',
          code: py`
            ln, lo, lr, r, gid, m = _make_batch(0)
            try:
                policy_loss("ppo", ln, lo, r, gid, m)
            except ValueError:
                pass
            else:
                raise AssertionError("未知 algo 应抛出 ValueError")
          `,
        },
        {
          name: 'on-policy 时 GRPO / GSPO 的 loss 为 0',
          code: py`
            ln, lo, lr, r, gid, m = _make_batch(1)
            for algo in ("grpo", "gspo"):
                got = policy_loss(algo, lo, lo, r, gid, m)
                assert abs(got) < 1e-9, f"{algo}: on-policy 时 loss 应为 0，得到 {got}"
          `,
        },
        {
          name: '五种算法逐一对拍（seed 2）',
          code: py`
            ln, lo, lr, r, gid, m = _make_batch(2, n_groups=3, G=4, T=8)
            ln_small = lo + 0.003 * (ln - lo)
            for algo in ("grpo", "dr_grpo", "dapo", "gspo", "cispo"):
                x = ln_small if algo == "gspo" else ln
                got = policy_loss(algo, x, lo, r, gid, m)
                exp = _ref_policy_loss(algo, x, lo, r, gid, m)
                assert abs(got - exp) < 1e-10, f"{algo}: 期望 {exp:.10f}，得到 {got:.10f}"
          `,
        },
        {
          name: '随机对拍（乱序分组、max_len）',
          hidden: true,
          code: py`
            rng = np.random.default_rng(5)
            for seed in range(15):
                ln, lo, lr, r, gid, m = _make_batch(600 + seed, n_groups=3, G=4, T=9, reward_kind="real")
                perm = rng.permutation(len(r))
                ids = rng.choice(1000, size=3, replace=False)[gid]
                args = [ln[perm], lo[perm], r[perm], ids[perm], m[perm]]
                for algo in ("grpo", "dr_grpo", "dapo", "gspo", "cispo"):
                    a = list(args)
                    if algo == "gspo":
                        a[0] = a[1] + 0.003 * (a[0] - a[1])
                    ml = 64 if algo == "dr_grpo" and seed % 2 else None
                    got = policy_loss(algo, *a, max_len=ml)
                    exp = _ref_policy_loss(algo, *a, max_len=ml)
                    assert abs(got - exp) < 1e-10, f"seed={seed}, {algo}: 期望 {exp:.10f}，得到 {got:.10f}"
          `,
        },
        {
          name: 'Dr.GRPO 对奖励尺度线性、GRPO 不变',
          hidden: true,
          code: py`
            ln, lo, lr, r, gid, m = _make_batch(8, reward_kind="real")
            a, b = policy_loss("dr_grpo", ln, lo, r, gid, m), policy_loss("dr_grpo", ln, lo, 3 * r, gid, m)
            assert abs(b - 3 * a) < 1e-9, f"Dr.GRPO 不除 std：奖励 ×3 时 loss 应 ×3（{a} → {b}）"
            a, b = policy_loss("grpo", ln, lo, r, gid, m), policy_loss("grpo", ln, lo, 3 * r, gid, m)
            assert abs(a - b) < 1e-6, f"GRPO 除以 std：奖励缩放不应改变 loss（{a} → {b}）"
          `,
        },
      ],
    },
    {
      id: 'c-rlvr',
      title: '可验证奖励：答案抽取 + 格式检查 + 超长惩罚',
      difficulty: 'hard',
      source: 'BOSS · RLVR 奖励函数',
      timeLimitMin: 25,
      description: md`
        实现一个数学 RLVR 的奖励函数：

        \`\`\`python
        def rlvr_rewards(responses: list[str], answers: list[str], lengths, L_max=20480, L_cache=4096) -> tuple[np.ndarray, np.ndarray]: ...
        \`\`\`

        **判定为正确**需同时满足：
        1. **格式**：含 \`<think>\`，其后有 \`</think>\`，且**最后一个** \`\boxed{\` 出现在第一个 \`</think>\` 之后；
        2. **答案**：取**最后一个** \`\boxed{...}\` 的内容（花括号可嵌套，如 \`\boxed{\frac{1}{2}}\`；括号不配平视为无答案），与标准答案**等价**：
           两边都先去掉空格和 \`$\`，把 \`\frac{a}{b}\` / \`\dfrac{a}{b}\`（a、b 内无花括号）改写为 \`a/b\`；若能被 \`fractions.Fraction\` 解析则按数值比较，否则按字符串比较。

        **奖励** = （正确 $+1$，否则 $-1$）+ DAPO 软超长惩罚 $R_{length}(|y|)$（\`lengths\` 给出每条回答的 token 数）。

        返回 \`(rewards, correct)\`：\`(B,)\` float64 与 \`(B,)\` bool。
      `,
      starter: py`
        import re
        import numpy as np
        from fractions import Fraction

        def rlvr_rewards(responses, answers, lengths, L_max=20480, L_cache=4096):
            # TODO 1: last_boxed —— rfind + 括号计数
            # TODO 2: 格式检查（<think> ... </think> 在最后一个 boxed 之前）
            # TODO 3: 答案规范化与等价判断（Fraction）
            # TODO 4: ±1 + 软超长惩罚
            raise NotImplementedError
      `,
      solution: py`
        import re
        import numpy as np
        from fractions import Fraction

        def _last_boxed(s):
            i = s.rfind("\\boxed{")
            if i < 0:
                return None, -1
            j, depth = i + 7, 1
            while j < len(s) and depth:
                depth += {"{": 1, "}": -1}.get(s[j], 0)
                j += 1
            return (s[i + 7:j - 1] if depth == 0 else None), i

        def _normalize(a):
            a = a.replace(" ", "").replace("$", "")
            a = re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", a)
            try:
                return Fraction(a)                     # "0.5"、"1/2"、"3" 都能解析
            except (ValueError, ZeroDivisionError):
                return a

        def rlvr_rewards(responses, answers, lengths, L_max=20480, L_cache=4096):
            correct = np.zeros(len(responses), dtype=bool)
            for k, (s, ans) in enumerate(zip(responses, answers)):
                box, pos = _last_boxed(s)
                t0, t1 = s.find("<think>"), s.find("</think>")
                if box is None or t0 < 0 or t1 <= t0 or pos <= t1:
                    continue                                   # 格式不合格 → 判错
                correct[k] = _normalize(box) == _normalize(ans)
            L = np.asarray(lengths, dtype=np.float64)
            soft = L_max - L_cache
            r_len = np.where(L <= soft, 0.0, np.where(L <= L_max, (soft - L) / L_cache, -1.0))
            return np.where(correct, 1.0, -1.0) + r_len, correct
      `,
      solutionAlt: [
        {
          label: '工程中的做法（仅展示）',
          code: py`
            # 生产环境通常直接用成熟的判分库，例如 HuggingFace 的 Math-Verify（Dr.GRPO 论文所用）：
            #   from math_verify import parse, verify
            #   ok = verify(parse(gold), parse(pred))
            # 它处理 LaTeX 解析、集合 / 区间 / 符号表达式等价等；面试手写时说清"只取最后答案 + 规范化 + 数值比较"即可。
            # 奖励张量与 PyTorch 无关，通常在 CPU 上的 reward worker 里批量计算后再转成 tensor：
            import torch
            def to_tensor(rewards, correct):
                return torch.as_tensor(rewards, dtype=torch.float32), torch.as_tensor(correct)
          `,
        },
      ],
      explanation: md`
        **防 hacking 的三个细节**：只认**最后一个** boxed（输出多个候选不会因其中之一正确而得分）；格式要求答案在思考结束之后；括号不配平视为无答案。
        数值比较用 \`Fraction\` 而不是 float，避免 \`0.1 + 0.2\` 式的误差，并统一 \`1/2\`、\`0.5\`、\`\frac{1}{2}\`。

        奖励构成与 DAPO 一致：$\pm1$ 加软超长惩罚。真实系统还会处理单位、集合、区间、符号表达式等价，通常交给 Math-Verify 一类的库。
      `,
      hints: [
        md`\`i = s.rfind("\\boxed{")\`，从 \`i + 7\` 开始做括号计数，\`depth\` 归零时停。`,
        md`\`re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", a)\` 把分数改写成 \`a/b\`，再 \`Fraction(a)\`。`,
        md`格式：\`t0 = s.find("<think>")\`、\`t1 = s.find("</think>")\`，要求 \`t0 >= 0\`、\`t1 > t0\`、最后一个 boxed 的位置 \`> t1\`。`,
      ],
      setup: SETUP,
      tests: [
        {
          name: '基本判定',
          code: py`
            resp = [
                "<think>2+2</think> 所以 \\boxed{4}",
                "<think>...</think> \\boxed{5}",
                "\\boxed{4}",
                "<think>先猜 \\boxed{4}</think> 最终 \\boxed{3}",
            ]
            rw, ok = rlvr_rewards(resp, ["4", "4", "4", "4"], [100, 100, 100, 100])
            assert ok.tolist() == [True, False, False, False], f"correct 期望 [True, False, False, False]，得到 {ok.tolist()}"
            assert np.allclose(rw, [1, -1, -1, -1]), f"rewards 得到 {rw}"
          `,
        },
        {
          name: '分数、小数与嵌套括号',
          code: py`
            resp = [
                "<think>x</think>\\boxed{\\frac{1}{2}}",
                "<think>x</think>\\boxed{0.5}",
                "<think>x</think>\\boxed{ 1/2 }",
                "<think>x</think>\\boxed{\\dfrac{3}{6}}",
                "<think>x</think>\\boxed{\\frac{1}{2}",
            ]
            _, ok = rlvr_rewards(resp, ["1/2"] * 5, [10] * 5)
            assert ok.tolist() == [True, True, True, True, False], f"得到 {ok.tolist()}（最后一个括号不配平应判错）"
          `,
        },
        {
          name: '超长惩罚',
          code: py`
            resp = ["<think>x</think>\\boxed{7}"] * 3
            rw, _ = rlvr_rewards(resp, ["7", "7", "8"], [16384, 18432, 30000])
            assert np.allclose(rw, [1.0, 0.5, -2.0]), f"期望 [1.0, 0.5, -2.0]，得到 {rw}"
          `,
        },
        {
          name: '随机对拍（hacking 样本）',
          hidden: true,
          code: py`
            import random
            rnd = random.Random(0)
            pieces = ["\\boxed{3}", "\\boxed{\\frac{6}{2}}", "\\boxed{4}", "\\boxed{x+1}", "\\boxed{{3}}", "\\boxed{3.0}", "\\boxed{}"]
            resp, ans, lens = [], [], []
            for _ in range(300):
                parts = []
                if rnd.random() < 0.8: parts.append("<think>")
                parts.append(rnd.choice(["推理", rnd.choice(pieces)]))
                if rnd.random() < 0.8: parts.append("</think>")
                for _ in range(rnd.randint(0, 2)):
                    parts.append(rnd.choice(pieces))
                resp.append(" ".join(parts))
                ans.append(rnd.choice(["3", "x+1", "4"]))
                lens.append(rnd.randint(1000, 22000))
            rw, ok = rlvr_rewards(resp, ans, lens)
            erw, eok = _ref_rlvr(resp, ans, lens)
            bad = [i for i in range(300) if ok[i] != eok[i]]
            assert not bad, f"判定不一致，例如：{resp[bad[0]]!r}（答案 {ans[bad[0]]!r}，期望 {eok[bad[0]]}）"
            assert np.allclose(rw, erw), "奖励数值不一致"
          `,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-design',
      freq: 3,
      q: md`如果让你从零搭一套数学推理 RL 训练，你会怎么选算法和关键配置？`,
      a: md`
        **结论**：GRPO 系 + 规则奖励起步，叠加 DAPO 的四招与 Dr.GRPO 的去偏；MoE 或不稳定时换 GSPO / 加 Keep Routing；有强专家时用 on-policy 蒸馏合并。

        **原因**：规则验证器（只取最后答案、Math-Verify）避免 reward hacking；动态采样保证有效 batch；Clip-Higher 防熵坍缩；token 级或常数归一化去掉长度偏差；软超长惩罚处理截断；训推不一致用 TIS / FP32 LM head 修正。

        **延伸**：监控 reward、正确 / 错误回答长度、熵与高熵 token 占比、clip_frac、训推 logprob 差异、全对 / 全错组占比；评估同时报 pass@1 与大 k 的 pass@k。
      `,
    },
    {
      id: 'qa-debug',
      freq: 2,
      q: md`RL 训练中 reward 不涨甚至崩溃，你会按什么顺序排查？`,
      a: md`
        **结论**：先查数据与奖励，再查数值一致性，最后查算法超参。

        **原因**：① 奖励：抽查高 / 低奖励样本，确认验证器正确、没有被 hack；看全对 / 全错比例（题太难或太易都没信号）。② 一致性：训推 logprob 差异（vLLM vs 训练框架）、MoE 路由是否一致、top-p 掩码是否保留。③ 算法：熵是否坍缩、clip_frac 是否过高、长度是否暴涨或频繁截断、学习率与 off-policy 步数。

        **延伸**：MiniMax-M1 的经验——LM head 精度、AdamW 的 β2 / eps（他们用 β2=0.95、eps=1e-15）、重复检测提前截断，都可能是“reward 不涨”的根因。
      `,
    },
  ],

  cheatsheet: md`
    **五个旋钮**：优势（组内 / 全局 / 是否除 std）· 比值（token / 序列 / 截断权重）· 裁剪（对称 / 非对称 / 只截权重）· 聚合（行平均 / token / 常数）· KL（loss / reward / 无）

    | 算法 | 一句话 |
    |---|---|
    | GRPO | 组内标准化优势 + token clip + k3 KL |
    | Dr.GRPO | 去 std、去 1/\|o\|，除常数 |
    | DAPO | 0.2/0.28 · 动态采样 · token-mean · 软超长 · 无 KL |
    | GSPO | 序列几何平均比值，3e-4/4e-4 |
    | CISPO | sg(clip(r))·A·log π，保留全部梯度 |
    | TIS | min(π_train/π_infer, C) |
    | OPD | 学生采样 + 教师逐 token 反向 KL |

    - 奖励：只取最后 \`\boxed\`、格式门槛、Fraction 比较、超长软惩罚
  `,
})
