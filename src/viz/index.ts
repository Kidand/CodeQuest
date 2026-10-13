/**
 * 可视化原语库。内容作者优先组合这些组件，再按需手写 SVG。
 * 规范见 docs/AUTHORING.md「可视化」一节。
 */
export { VizFrame } from './VizFrame'
export { useStepper, type Stepper } from './useStepper'
export { StepControls } from './StepControls'
export { CodeTrace } from './CodeTrace'
export { ArrayView, type ArrayPointer, type ArrayRange } from './ArrayView'
export { MatrixView } from './MatrixView'
export { ProbBars, type ProbItem } from './ProbBars'
export { TokenRow, type TokenItem } from './TokenRow'
export { TreeView, type TreeNode } from './TreeView'
export { GraphView, type GraphNode, type GraphEdge } from './GraphView'
export { Legend } from './Legend'
export { Tex } from './Tex'
export { toneColor, toneSoft, catColor, heatColor, divergingColor, textOnHeat, type VizTone } from './tones'
export { useElementWidth, fitCell } from './useWidth'
