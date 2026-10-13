/**
 * algo-backtrack 章节公共 Python 片段：复用 algo-ds 的计数数值包装 _Num（超预算立即失败），
 * 并补充结果规范化 / 对比辅助函数。
 */
import { py } from '@/content/helpers'
import { COUNTED_NUM } from '../algo-ds/shared'

export { COUNTED_NUM }

export const RESULT_HELPERS = py`
  def _norm_sets(res, inner_sort=True):
      """把 list[list] 规范化为可比较的有序元组列表（组合题内部排序；排列题保留顺序）"""
      out = []
      for r in res:
          t = tuple(_unwrap(list(r)))
          out.append(tuple(sorted(t)) if inner_sort else t)
      return sorted(out)

  def _assert_same(got, exp, inner_sort, what):
      g = _norm_sets(got, inner_sort)
      e = _norm_sets(exp, inner_sort)
      assert len(g) == len(set(g)), f"{what}：结果中有重复项，例如 {[x for x in g if g.count(x) > 1][:2]}"
      if g != e:
          miss = sorted(set(e) - set(g))[:3]
          extra = sorted(set(g) - set(e))[:3]
          raise AssertionError(f"{what}：期望 {len(e)} 个结果，得到 {len(g)} 个；缺少 {miss}，多出 {extra}")
`
