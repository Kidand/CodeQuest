/**
 * 本章可视化使用的硬件与模型规格（仅本章使用）。
 *
 * GPU：NVIDIA 官网规格页（2026-10 查阅）。官网 Tensor Core 数字标注为「with sparsity」，稠密值为其一半：
 *   - A100 80GB SXM：BF16 312 TFLOPS（稠密）、HBM2e 2,039 GB/s、80 GB        nvidia.com/en-us/data-center/a100
 *   - H100 SXM：BF16 1,979 TFLOPS（稀疏）→ 989 稠密、HBM3 3.35 TB/s、80 GB     nvidia.com/en-us/data-center/h100
 *   - H200 SXM：BF16 1,979 TFLOPS（稀疏）→ 989 稠密、HBM3e 4.8 TB/s、141 GB    nvidia.com/en-us/data-center/h200
 *   - B200（DGX B200 每卡）：HGX B200 8 卡 FP16/BF16 36 PFLOPS（稀疏）→ 每卡 2.25 PFLOPS 稠密；
 *     DGX B200 总 HBM3e 带宽 64 TB/s、总显存 1,440 GB → 每卡 8 TB/s、180 GB      nvidia.com/en-us/data-center/dgx-b200 , /hgx
 *
 * 模型：Llama-3.1 官方 config（hidden / layers / heads / kv heads），参数量为按 config 计算的近似值。
 */
export interface Gpu {
  id: string
  name: string
  tflops: number // BF16 稠密
  tbs: number // HBM 带宽 TB/s
  gb: number // 显存 GB
}

export const GPUS: Gpu[] = [
  { id: 'a100', name: 'A100 80GB SXM', tflops: 312, tbs: 2.039, gb: 80 },
  { id: 'h100', name: 'H100 SXM', tflops: 989, tbs: 3.35, gb: 80 },
  { id: 'h200', name: 'H200 SXM', tflops: 989, tbs: 4.8, gb: 141 },
  { id: 'b200', name: 'B200', tflops: 2250, tbs: 8, gb: 180 },
]

export interface Model {
  id: string
  name: string
  params: number // 参数量
  layers: number
  heads: number
  kvHeads: number
  headDim: number
  hidden: number
}

export const MODELS: Model[] = [
  { id: '8b', name: 'Llama-3.1-8B', params: 8.03e9, layers: 32, heads: 32, kvHeads: 8, headDim: 128, hidden: 4096 },
  { id: '70b', name: 'Llama-3.1-70B', params: 70.6e9, layers: 80, heads: 64, kvHeads: 8, headDim: 128, hidden: 8192 },
  { id: '405b', name: 'Llama-3.1-405B', params: 405.9e9, layers: 126, heads: 128, kvHeads: 8, headDim: 128, hidden: 16384 },
]

/** 每个 token 的 KV 字节数：2(K,V) · L · n_kv · d_h · bytes */
export const kvPerToken = (m: Model, kvBytes = 2) => 2 * m.layers * m.kvHeads * m.headDim * kvBytes

/**
 * decode 一步（batch 个序列、每个序列上下文长度 ctx）的 FLOPs 与 HBM 字节（单卡等效，忽略激活读写与通信）：
 *   FLOPs = 2·N·B（线性层）+ 4·L·n_h·d_h·S·B（注意力 QKᵀ 与 AV）
 *   Bytes = N·w（读一遍权重）+ B·S·kv（读全部 KV cache）
 */
export function decodeCost(m: Model, batch: number, ctx: number, wBytes = 2, kvBytes = 2) {
  const linF = 2 * m.params * batch
  const attF = 4 * m.layers * m.heads * m.headDim * ctx * batch
  const wB = m.params * wBytes
  const kvB = batch * ctx * kvPerToken(m, kvBytes)
  return { flops: linF + attF, bytes: wB + kvB, linF, attF, wB, kvB }
}

/** prefill 一个长度为 P 的请求：线性层 2·N·P + 因果注意力 2·L·n_h·d_h·P²；字节 = 权重 + 写出 KV */
export function prefillCost(m: Model, P: number, wBytes = 2, kvBytes = 2) {
  const flops = 2 * m.params * P + 2 * m.layers * m.heads * m.headDim * P * P
  const bytes = m.params * wBytes + P * kvPerToken(m, kvBytes)
  return { flops, bytes }
}

/** 放得下权重所需的最小张量并行度（2 的幂），给 KV 留出空间：权重 ≤ 72% 显存 */
export function minTP(m: Model, g: Gpu, wBytes = 2) {
  let k = 1
  while (m.params * wBytes > 0.72 * g.gb * 1e9 * k && k < 64) k *= 2
  return k
}

export const fmtNum = (x: number, d = 1) => {
  if (!Number.isFinite(x)) return '∞'
  const a = Math.abs(x)
  if (a >= 1e12) return `${(x / 1e12).toFixed(d)}T`
  if (a >= 1e9) return `${(x / 1e9).toFixed(d)}G`
  if (a >= 1e6) return `${(x / 1e6).toFixed(d)}M`
  if (a >= 1e3) return `${(x / 1e3).toFixed(d)}K`
  return x.toFixed(d)
}

export const fmtMs = (s: number) => (s >= 1 ? `${s.toFixed(2)} s` : s >= 1e-3 ? `${(s * 1e3).toFixed(2)} ms` : `${(s * 1e6).toFixed(1)} µs`)
