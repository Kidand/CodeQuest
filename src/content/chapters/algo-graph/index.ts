import { defineChapter, md } from '@/content/helpers'

export default defineChapter({
  id: 'algo-graph',
  track: 'algo',
  order: 4,
  title: '树与图',
  subtitle: '二叉树遍历 · 后序框架 · BFS/DFS · 拓扑排序 · 并查集 · 最短路 · Trie',
  icon: 'Network',
  prerequisites: ['algo-search'],
  intro: md`
    树与图是手撕题的第二大来源：层序遍历、岛屿数量几乎每场都会遇到，最近公共祖先、最大路径和、课程表是高频追问。
    本章从**二叉树的三种遍历与统一的显式栈写法**出发，过渡到**后序“向上返回信息”框架**，再到图上的 **BFS / DFS / 拓扑排序**、
    **并查集与 Dijkstra**，最后是 **Trie**。

    每一关都配了大模型岗的真实场景：autograd 按反向拓扑序传梯度、预训练数据用并查集合并近重复文档、WordPiece 的最长匹配分词与
    SGLang 的 RadixAttention 前缀缓存。
  `,
  levels: [
    {
      id: 'algo-graph-tree',
      title: '二叉树遍历',
      summary: '递归三序的本质、统一的显式栈写法（标记法）、层序 BFS 按层切分与锯齿形变体',
      difficulty: 2,
      estMinutes: 45,
      tier: 'must',
      tags: ['二叉树', '遍历', '显式栈', 'BFS'],
      problems: ['LC 94 中序遍历', 'LC 144 前序遍历', 'LC 145 后序遍历', 'LC 102 层序遍历', 'LC 103 锯齿形层序', 'LC 104 最大深度', 'LC 226 翻转二叉树'],
      load: () => import('./levels/tree'),
    },
    { id: 'algo-graph-tree-adv', title: '二叉树进阶', summary: '后序信息汇总、最大路径和与结构序列化', difficulty: 4, estMinutes: 45, tier: 'high', tags: ['二叉树进阶'], problems: ['LC 236', 'LC 124', 'LC 297'], load: () => import('./levels/tree-adv') },
    { id: 'algo-graph-search', title: 'BFS/DFS 与拓扑排序', summary: '洪水填充、多源扩散与计算图的反向拓扑传播', difficulty: 3, estMinutes: 45, tier: 'must', tags: ['BFS', 'DFS', '拓扑排序'], problems: ['LC 200', 'LC 994', 'LC 210', 'mini-autograd'], llmLink: '计算图按反向拓扑顺序传播梯度', load: () => import('./levels/search') },
    { id: 'algo-graph-uf', title: '并查集与最短路', summary: '连通闭包、路径压缩与非负边上的堆优化最短路', difficulty: 3, estMinutes: 45, tier: 'high', tags: ['并查集'], problems: ['LC 547', 'LC 684', 'LC 743'], llmLink: '相似文档候选边合并成连通分组', load: () => import('./levels/uf') },
    { id: 'algo-graph-trie', title: 'Trie 与前缀匹配', summary: '词尾标记、共享前缀与 WordPiece 最长匹配', difficulty: 3, estMinutes: 45, tier: 'high', tags: ['Trie', '前缀匹配'], problems: ['LC 208', 'LC 211', 'WordPiece'], llmLink: 'WordPiece 最长匹配与前缀缓存索引', load: () => import('./levels/trie') },
    { id:'algo-graph-boss', title:'BOSS：树与图综合', summary:'跨结构综合实战与边界审计', kind:'boss', difficulty:5, estMinutes:50, tier:'high', tags:['综合挑战'], problems:['综合设计'], load:()=>import('./levels/boss') },
  ],
})
