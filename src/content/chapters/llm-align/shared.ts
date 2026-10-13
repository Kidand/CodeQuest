/**
 * 本章公共 Python 片段（测试 setup 复用）。约定：辅助函数一律以 _ 或 _ref_ 开头。
 */
import { py } from '@/content/helpers'

/** 数值稳定的 log_softmax / 序列 log-prob（labels 未移位，-100 不计入） */
export const PY_REF_LOGPS = py`
  import numpy as np

  def _ref_log_softmax(x, axis=-1):
      m = x.max(axis=axis, keepdims=True)
      return x - m - np.log(np.exp(x - m).sum(axis=axis, keepdims=True))

  def _ref_token_logps(logits, labels, ignore_index=-100):
      """返回 (逐 token logp (N, T-1), mask (N, T-1))，已做 shift"""
      lg = logits[:, :-1, :]
      lab = labels[:, 1:]
      mask = lab != ignore_index
      safe = np.where(mask, lab, 0)
      lp = _ref_log_softmax(lg)
      tok = np.take_along_axis(lp, safe[..., None], axis=-1)[..., 0]
      return np.where(mask, tok, 0.0), mask

  def _ref_sequence_logps(logits, labels, ignore_index=-100):
      tok, mask = _ref_token_logps(logits, labels, ignore_index)
      return tok.sum(axis=-1)

  def _ref_log_sigmoid(x):
      return -np.logaddexp(0.0, -x)
`

/**
 * 造一批偏好数据：前 B 行 chosen、后 B 行 rejected（TRL concatenated_forward 的排布）。
 * prompt 与 padding 的 label 为 -100；被 mask 的位置 logits 填大幅度垃圾值。
 */
export const PY_MAKE_PAIRS = py`
  import numpy as np

  def _make_pairs(seed, B=3, T=9, V=7, scale=2.0, garbage=True):
      rng = np.random.default_rng(seed)
      N = 2 * B
      labels = np.full((N, T), -100, dtype=np.int64)
      for b in range(B):
          p = int(rng.integers(1, 4))                  # prompt 长度（chosen / rejected 共享）
          for row in (b, B + b):
              L = int(rng.integers(1, T - p + 1))      # 回复长度
              labels[row, p:p + L] = rng.integers(0, V, size=L)
      pol = rng.normal(0, scale, size=(N, T, V))
      ref = pol + rng.normal(0, 0.5, size=(N, T, V))
      if garbage:
          # 被 mask 位置的“预测”不应影响结果：labels[:, t+1] == -100 时 logits[:, t] 无关
          bad = np.zeros((N, T), dtype=bool)
          bad[:, :-1] = labels[:, 1:] == -100
          bad[:, -1] = True
          pol[bad] = rng.uniform(-40, 40, size=(bad.sum(), V))
          ref[bad] = rng.uniform(-40, 40, size=(bad.sum(), V))
      return pol, ref, labels
`
