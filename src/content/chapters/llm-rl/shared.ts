/**
 * 本章公共 Python 片段（测试 setup 里复用的参考实现与造数函数）。
 * 约定：所有参考实现以 _ref_ / _ 开头；std 一律用 ddof=1（与 torch.std 默认一致），分母加 1e-6（与 verl 一致）。
 */
import { py } from '@/content/helpers'

/** 造一批 RL 训练数据：logp 为负数，padding 位置填垃圾值（必须被 mask 掉） */
export const PY_MAKE_BATCH = py`
  import numpy as np

  def _make_batch(seed, n_groups=3, G=4, T=7, min_len=2, garbage=True, reward_kind="binary"):
      rng = np.random.default_rng(seed)
      B = n_groups * G
      lens = rng.integers(min_len, T + 1, size=B)
      mask = (np.arange(T)[None, :] < lens[:, None]).astype(np.float64)
      logp_old = -rng.exponential(1.0, size=(B, T)) - 0.02
      logp_new = logp_old + rng.normal(0, 0.25, size=(B, T))
      logp_ref = logp_old + rng.normal(0, 0.15, size=(B, T))
      if garbage:
          pad = mask == 0
          logp_old[pad] = rng.uniform(-3, 3, size=pad.sum())
          logp_new[pad] = rng.uniform(-3, 3, size=pad.sum())
          logp_ref[pad] = rng.uniform(-3, 3, size=pad.sum())
      if reward_kind == "binary":
          rewards = rng.integers(0, 2, size=B).astype(np.float64)
      else:
          rewards = np.round(rng.uniform(-1, 1, size=B), 2)
      rewards[:G] = 1.0                       # 第 0 组全对：优势应全为 0
      if n_groups > 1:
          rewards[G:2 * G] = [1.0] + [0.0] * (G - 1)   # 第 1 组只有一个对
      group_ids = np.repeat(np.arange(n_groups), G)
      return logp_new, logp_old, logp_ref, rewards, group_ids, mask
`

/** 组内优势（支持乱序 group_ids）。scale: 'group' 组内 std / 'none' 只减均值 / 'batch' 组内减均值后除以全 batch 的 std */
export const PY_REF_ADV = py`
  import numpy as np

  def _ref_group_adv(rewards, group_ids, scale="group", eps=1e-6):
      r = np.asarray(rewards, dtype=np.float64)
      _, inv = np.unique(np.asarray(group_ids), return_inverse=True)
      inv = inv.reshape(-1)
      cnt = np.bincount(inv).astype(np.float64)
      mean = np.bincount(inv, weights=r) / cnt
      dev = r - mean[inv]
      if scale == "none":
          return dev
      if scale == "batch":
          return dev / (dev.std(ddof=1) + eps)
      std = np.sqrt(np.bincount(inv, weights=dev ** 2) / (cnt - 1))
      return dev / (std[inv] + eps)
`

/** GRPO（DeepSeekMath 原式）：token 级 clip + k3 KL 直接进 loss + 先回答内平均再回答间平均 */
export const PY_REF_GRPO =
  PY_REF_ADV +
  '\n\n' +
  py`
  def _ref_grpo_parts(logp_new, logp_old, logp_ref, rewards, group_ids, mask, clip_eps=0.2, beta=0.04, adv_eps=1e-6):
      m = np.asarray(mask, dtype=np.float64)
      adv = _ref_group_adv(rewards, group_ids, "group", adv_eps)
      A = adv[:, None]
      ratio = np.exp(np.where(m > 0, logp_new - logp_old, 0.0))
      surr1 = ratio * A
      surr2 = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A
      d = np.where(m > 0, logp_ref - logp_new, 0.0)
      kl = np.exp(d) - d - 1
      return m, adv, A, ratio, surr1, surr2, d, kl

  def _ref_grpo_loss(logp_new, logp_old, logp_ref, rewards, group_ids, mask, clip_eps=0.2, beta=0.04, adv_eps=1e-6):
      m, adv, A, ratio, surr1, surr2, d, kl = _ref_grpo_parts(logp_new, logp_old, logp_ref, rewards, group_ids, mask, clip_eps, beta, adv_eps)
      tok = (-np.minimum(surr1, surr2) + beta * kl) * m
      lens = m.sum(axis=1)
      loss = float(np.mean(tok.sum(axis=1) / np.maximum(lens, 1.0)))
      n = max(m.sum(), 1.0)
      stats = {
          "adv": adv,
          "kl": float((kl * m).sum() / n),
          "clip_frac": float(((surr2 < surr1) * m).sum() / n),
      }
      return loss, stats

  def _ref_grpo_grad(logp_new, logp_old, logp_ref, rewards, group_ids, mask, clip_eps=0.2, beta=0.04, adv_eps=1e-6):
      m, adv, A, ratio, surr1, surr2, d, kl = _ref_grpo_parts(logp_new, logp_old, logp_ref, rewards, group_ids, mask, clip_eps, beta, adv_eps)
      g_pg = np.where(surr1 <= surr2, -A * ratio, 0.0)
      g_kl = beta * (1.0 - np.exp(d))
      B = m.shape[0]
      w = m / (np.maximum(m.sum(axis=1), 1.0)[:, None] * B)
      return (g_pg + g_kl) * w
`
