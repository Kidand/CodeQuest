/**
 * Python 测试 harness —— 浏览器 Worker 与 Node 校验脚本（scripts/verify-python.ts）共用，
 * 保证“参考解在校验脚本里通过” ⇔ “同样的代码在浏览器里通过”。
 *
 * 执行模型：
 *   1. exec(用户代码) → 命名空间 ns（文件名 solution.py，报错行号可映射回编辑器）
 *   2. exec(setup) 到 ns（隐藏辅助/参考实现，约定命名 _ref_xxx）
 *   3. 每个测试在 dict(ns) 的浅拷贝中 exec，AssertionError → 失败（展示 assert 消息）
 */

export interface TestResult {
  name: string
  hidden: boolean
  passed: boolean
  error: string | null
  /** 用户代码中出错的行（1 起始），若可定位 */
  errorLine?: number | null
  stdout: string
  ms: number
}

export interface RunResult {
  /** 全部测试通过 */
  ok: boolean
  /** 用户代码本身无法执行（语法错误 / 顶层异常） */
  compileError: string | null
  errorLine: number | null
  /** 用户代码顶层 print 输出 */
  stdout: string
  results: TestResult[]
  /** 以下由 JS 侧填充 */
  timedOut?: boolean
  /** 基础设施错误（Pyodide 加载失败等） */
  infraError?: string
  durationMs?: number
}

export interface RunRequest {
  code: string
  setup?: string
  tests: { name: string; code: string; hidden?: boolean }[]
}

export const HARNESS_PY = String.raw`
import sys, io, json, time, traceback, linecache

def _cq_register(fname, src):
    linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)

def _cq_fmt_exc(e):
    tb = traceback.extract_tb(e.__traceback__)
    line = None
    parts = []
    for fr in tb:
        if fr.filename == "solution.py":
            line = fr.lineno
            where = "你的代码"
        elif fr.filename == "<test>":
            where = "测试用例"
        elif fr.filename == "<setup>":
            where = "辅助代码"
        else:
            continue
        src = (fr.line or "").strip()
        parts.append("  " + where + " 第 " + str(fr.lineno) + " 行" + ((": " + src) if src else ""))
    msg = type(e).__name__ + ": " + str(e)
    text = ("Traceback:\n" + "\n".join(parts) + "\n" + msg) if parts else msg
    return text, line

def _cq_assert_msg(e, test_src):
    msg = str(e)
    if msg:
        return msg
    tb = traceback.extract_tb(e.__traceback__)
    for fr in reversed(tb):
        if fr.filename == "<test>":
            lines = test_src.splitlines()
            if 0 < fr.lineno <= len(lines):
                return "断言失败: " + lines[fr.lineno - 1].strip()
    return "断言失败"

def _cq_run(user_code, setup_code, tests_json, on_test=None):
    tests = json.loads(tests_json)
    ns = {"__name__": "__cq__", "__builtins__": __builtins__}
    out = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    res = {"ok": False, "compileError": None, "errorLine": None, "stdout": "", "results": []}
    try:
        sys.stdout = out
        sys.stderr = out
        _cq_register("solution.py", user_code)
        try:
            exec(compile(user_code, "solution.py", "exec"), ns)
        except SyntaxError as e:
            res["compileError"] = "SyntaxError: " + str(e.msg) + "（第 " + str(e.lineno) + " 行）"
            res["errorLine"] = e.lineno
            res["stdout"] = out.getvalue()[:8000]
            return res
        except BaseException as e:
            txt, line = _cq_fmt_exc(e)
            res["compileError"] = txt
            res["errorLine"] = line
            res["stdout"] = out.getvalue()[:8000]
            return res
        res["stdout"] = out.getvalue()[:8000]
        if setup_code:
            _cq_register("<setup>", setup_code)
            exec(compile(setup_code, "<setup>", "exec"), ns)
        all_ok = True
        for t in tests:
            buf = io.StringIO()
            sys.stdout = buf
            sys.stderr = buf
            src = t.get("code", "")
            _cq_register("<test>", src)
            item = {"name": t.get("name", ""), "hidden": bool(t.get("hidden")), "passed": False,
                    "error": None, "errorLine": None, "stdout": "", "ms": 0}
            t0 = time.perf_counter()
            try:
                exec(compile(src, "<test>", "exec"), dict(ns))
                item["passed"] = True
            except AssertionError as e:
                item["error"] = _cq_assert_msg(e, src)
            except BaseException as e:
                txt, line = _cq_fmt_exc(e)
                item["error"] = txt
                item["errorLine"] = line
                if line and not res["errorLine"]:
                    res["errorLine"] = line
            item["ms"] = round((time.perf_counter() - t0) * 1000, 2)
            item["stdout"] = buf.getvalue()[:4000]
            if not item["passed"]:
                all_ok = False
            res["results"].append(item)
            if on_test is not None:
                on_test(json.dumps(item, ensure_ascii=False))
        res["ok"] = all_ok and len(tests) > 0
        return res
    finally:
        sys.stdout, sys.stderr = old_out, old_err

def _cq_exec(user_code):
    """自由运行（不跑测试）：返回 stdout 与错误"""
    out = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    ns = {"__name__": "__main__"}
    err, line = None, None
    try:
        sys.stdout = out
        sys.stderr = out
        _cq_register("solution.py", user_code)
        try:
            exec(compile(user_code, "solution.py", "exec"), ns)
        except SyntaxError as e:
            err = "SyntaxError: " + str(e.msg) + "（第 " + str(e.lineno) + " 行）"
            line = e.lineno
        except BaseException as e:
            err, line = _cq_fmt_exc(e)
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return {"stdout": out.getvalue()[:20000], "error": err, "errorLine": line}
`

/** 组装跟练代码：把 {{n}} 替换为答案；多行答案按占位符所在行的缩进对齐 */
export function assembleTemplate(template: string, fills: Record<number, string>): string {
  return template.replace(/^([ \t]*)(.*?)\{\{(\d+)\}\}/gm, (_m, indent: string, before: string, n: string) => {
    const fill = fills[Number(n)] ?? ''
    const lines = fill.replace(/\r\n/g, '\n').split('\n')
    const contIndent = indent + ' '.repeat(before.length)
    const body = lines.map((l, i) => (i === 0 ? l : (before.trim() === '' ? indent : contIndent) + l)).join('\n')
    return indent + before + body
  }).replace(/\{\{(\d+)\}\}/g, (_m, n: string) => fills[Number(n)] ?? '') // 同一行的第二个及以后的空位
}

/** 模板中出现的空位编号（按出现顺序） */
export function templateBlanks(template: string): number[] {
  return [...template.matchAll(/\{\{(\d+)\}\}/g)].map((m) => Number(m[1]))
}
