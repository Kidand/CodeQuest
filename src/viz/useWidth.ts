import { useEffect, useRef, useState } from 'react'

/** 监听元素宽度（ResizeObserver）。自绘 SVG 需要按容器宽度布局时使用。 */
export function useElementWidth<T extends HTMLElement = HTMLDivElement>() {
  const ref = useRef<T>(null)
  const [width, setWidth] = useState(0)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setWidth(Math.round(e.contentRect.width)))
    ro.observe(el)
    setWidth(el.clientWidth)
    return () => ro.disconnect()
  }, [])
  return [ref, width] as const
}

/** 由可用宽度计算格子尺寸：不超过 max，不小于 min */
export function fitCell(avail: number, count: number, gap: number, max: number, min = 24) {
  if (!avail || count <= 0) return max
  return Math.max(min, Math.min(max, Math.floor((avail - (count - 1) * gap) / count)))
}
