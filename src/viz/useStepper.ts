import { useCallback, useEffect, useRef, useState } from 'react'

export interface Stepper<T> {
  /** 当前步下标 */
  i: number
  /** 当前帧 */
  frame: T
  total: number
  atStart: boolean
  atEnd: boolean
  playing: boolean
  speed: number
  setI: (i: number) => void
  next: () => void
  prev: () => void
  reset: () => void
  toggle: () => void
  setPlaying: (p: boolean) => void
  setSpeed: (s: number) => void
}

/**
 * 逐帧播放器：把算法过程预先展开成帧数组（每帧是一个完整的可渲染状态），
 * 用 stepper 控制前进/后退/自动播放。frames 变化（例如参数改变）时自动回到第 0 帧。
 *
 *   const frames = useMemo(() => buildFrames(arr, target), [arr, target])
 *   const st = useStepper(frames, { intervalMs: 900 })
 *   ... render st.frame ...
 *   <StepControls stepper={st} />
 */
export function useStepper<T>(frames: T[], opts: { intervalMs?: number; autoPlay?: boolean; loop?: boolean } = {}): Stepper<T> {
  const { intervalMs = 1000, autoPlay = false, loop = false } = opts
  const [i, setIRaw] = useState(0)
  const [playing, setPlaying] = useState(autoPlay)
  const [speed, setSpeed] = useState(1)
  const total = frames.length
  const framesRef = useRef(frames)

  useEffect(() => {
    if (framesRef.current !== frames) {
      framesRef.current = frames
      setIRaw(0)
    }
  }, [frames])

  const setI = useCallback((n: number) => setIRaw(Math.max(0, Math.min(total - 1, n))), [total])
  const next = useCallback(() => setIRaw((x) => Math.min(total - 1, x + 1)), [total])
  const prev = useCallback(() => setIRaw((x) => Math.max(0, x - 1)), [])
  const reset = useCallback(() => {
    setIRaw(0)
    setPlaying(false)
  }, [])

  useEffect(() => {
    if (!playing) return
    const t = setTimeout(() => {
      setIRaw((x) => {
        if (x + 1 < total) return x + 1
        if (loop) return 0
        setPlaying(false)
        return x
      })
    }, intervalMs / speed)
    return () => clearTimeout(t)
  }, [playing, i, total, intervalMs, speed, loop])

  const toggle = useCallback(() => {
    setPlaying((p) => {
      if (!p) setIRaw((x) => (x >= total - 1 ? 0 : x)) // 到末尾再按播放则重播
      return !p
    })
  }, [total])

  const idx = Math.min(i, Math.max(0, total - 1))
  return {
    i: idx,
    frame: frames[idx],
    total,
    atStart: idx === 0,
    atEnd: idx >= total - 1,
    playing,
    speed,
    setI,
    next,
    prev,
    reset,
    toggle,
    setPlaying,
    setSpeed,
  }
}
