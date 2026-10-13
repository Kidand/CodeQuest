import { defineChapter, md } from '@/content/helpers'

export default defineChapter({
  id: 'algo-ds',
  track: 'algo',
  order: 3,
  title: '链表、栈与哈希',
  subtitle: '链表指针 · LRU / LFU · 单调栈 · 前缀和 + 哈希',
  icon: 'Layers',
  intro: md`
    这一章是手撕题里“数据结构设计”的主场：**LRU 缓存几乎是大模型岗手撕的第一名**，反转链表 / K 个一组翻转紧随其后。
    我们先练链表指针的基本功（哑节点、三指针、快慢指针），再用“哈希表 + 双向链表”拼出 O(1) 的 LRU / LFU，
    最后是单调栈 / 单调队列和前缀和 + 哈希计数。

    和大模型系统的联系也最直接：vLLM 的 KV 块空闲队列就是一条双向链表 + LRU 淘汰；
    变长序列打包后的 \`cu_seqlens\` 就是前缀和，token 定位就是在前缀和上二分。
  `,
  levels: [
    {
      id: 'algo-ds-linkedlist',
      title: '链表基本功',
      summary: '先存后改的三指针反转、哑节点与头插法、K 个一组翻转、快慢指针与环入口证明',
      difficulty: 3,
      estMinutes: 45,
      tier: 'must',
      tags: ['链表', '哑节点', '快慢指针'],
      problems: ['LC 206 反转链表', 'LC 92 反转链表 II', 'LC 25 K 个一组翻转链表', 'LC 21 合并两个有序链表', 'LC 142 环形链表 II', 'LC 19 删除倒数第 N 个节点'],
      load: () => import('./levels/linkedlist'),
    },
    {
      id: 'algo-ds-lru',
      title: 'LRU 与 LFU 缓存',
      summary: '哈希表 + 双向链表的 O(1) 设计、哨兵与两个原语、LFU 频次桶与 min_freq，以及 vLLM 前缀缓存的块淘汰',
      difficulty: 3,
      estMinutes: 50,
      tier: 'must',
      tags: ['LRU', 'LFU', '哈希表', '双向链表', '设计题'],
      problems: ['LC 146 LRU 缓存', 'LC 460 LFU 缓存'],
      llmLink: 'KV / 前缀缓存的块淘汰：块哈希 → 物理块 + 引用计数 + 双向链表空闲队列（LRU）',
      load: () => import('./levels/lru'),
    },
    { id: 'algo-ds-monostack', title: '单调栈与单调队列', summary: '弹栈确定边界，单调队列维护滑动窗口最值', difficulty: 3, estMinutes: 45, tier: 'high', tags: ['单调栈'], problems: ['LC 739', 'LC 84', 'LC 239'], load: () => import('./levels/monostack') },
    { id: 'algo-ds-hash', title: '哈希与前缀和', summary: '区间计数、最长连续序列与变长 token 定位', difficulty: 3, estMinutes: 45, tier: 'high', tags: ['哈希'], problems: ['LC 560', 'LC 128', 'cu_seqlens'], llmLink: '变长序列累积长度 cu_seqlens 与全局 token 定位', load: () => import('./levels/hash') },
    { id:'algo-ds-boss', title:'BOSS：数据结构综合', summary:'跨结构综合实战与边界审计', kind:'boss', difficulty:5, estMinutes:50, tier:'high', tags:['综合挑战'], problems:['综合设计'], load:()=>import('./levels/boss') },
  ],
})
