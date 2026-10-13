import { defineLevel, md, py } from '@/content/helpers'
import { TrieWalkViz } from '../viz/AdvancedViz'
export default defineLevel({
 learn:[
  {id:'trie',title:'共享前缀，把字符串匹配变成路径',viz:TrieWalkViz,body:md`Trie 的边标记字符，根到某节点的路径代表一个前缀。插入字符串时沿字符走，缺少边就创建节点；最后设置词尾标记。查询完整词既要求路径存在，也要求词尾标记为真。查询前缀则只要求路径存在，因此插入 apple 后 search(app) 为假而 starts_with(app) 为真。

  单次操作时间与字符串长度 L 成正比，而不与已插入词数直接成正比。字典存孩子适合稀疏字符集；固定大小数组访问快但浪费空槽。重复插入若只表示词集合，应保持幂等；若要统计次数，需要额外维护词频与前缀计数，并明确删除时怎样更新。`},
  {id:'wildcard',title:'通配符与多词搜索：共享搜索状态',body:md`支持点号通配符的词典中，普通字符只沿一条边，点号需要尝试当前节点的所有孩子，因此最坏仍会有指数分支。Trie 省去重复前缀比较，并不消除所有搜索复杂度。单词搜索 II 则把棋盘 DFS 和词典 Trie 同步推进；当前前缀在 Trie 中不存在时立即剪枝。

  棋盘状态要记录当前路径使用过的格子，递归结束恢复，不能用全局永久 visited，否则一个起点走过的格子会让其它起点失去合法答案。找到词后可清除词尾标记实现输出去重；删除已经没有孩子且不是词尾的枝条，还可以减少后续重复探索。`},
  {id:'wordpiece',title:'WordPiece：贪心最长匹配与整词失败',body:md`本关实现已完成预分词的单个词的 WordPiece 编码：从左到右，每一步优先选词表中最长匹配片段；非首片段查询时加续接标记 ##。如果某一步找不到匹配，整个词输出 [UNK]，不能保留此前成功片段。这里不包含规范化、标点切分、训练词表或最大词长限制。

  贪心最长匹配是 WordPiece 的编码规则，并不保证寻找任意可行切分。例如词表包含 ab、a、##bc，输入 abc 会先选 ab，随后失败；不能回溯改成 a、##bc。用 Trie 扫描字符并记录最近词尾可减少重复切片，首片段和续接片段可分别建 Trie。[官方模型接口](https://huggingface.co/docs/tokenizers/en/api/models) 描述了 unknown token 与 continuation prefix；[论文](https://arxiv.org/abs/2012.15524) 讨论最长匹配与加速。`},
 ],
 quiz:[
  {id:'q1',type:'single',difficulty:1,prompt:md`只插入 apple，查询 app 的完整单词存在性应该返回什么？`,options:['False，因为 app 节点没有词尾标记','True，因为路径存在','取决于哈希顺序'],answer:0,explanation:md`路径表示前缀，终止标记才表示完整词；哈希遍历顺序与这个查询无关。`},
  {id:'q2',type:'judge',difficulty:1,prompt:md`Trie 查询单词的时间通常是 O(L)，L 是查询词长度。`,answer:true,explanation:md`每个字符走一条边；这里按字典访问平均 O(1) 分析。空间仍与所有不同前缀数量有关。`},
  {id:'q3',type:'numeric',difficulty:2,prompt:md`插入 car、cat、cars，不含根节点，Trie 一共需要几个字符节点？`,answer:5,tolerance:0,explanation:md`共享 c、a 两个节点，再有 r、t、s，总计 5；不能把三个单词长度直接相加。`},
  {id:'q4',type:'fill',difficulty:2,prompt:md`词表含 play、##ing，输入 playing 的第二个 token 是 ____。`,blanks:[{answers:['##ing']}],explanation:md`非首片段带 ## 续接标记；原字符串中的字符仍是 ing，标记只是词表命名规则。`},
  {id:'q5',type:'multi',difficulty:2,prompt:md`棋盘多词搜索中哪些做法合理？`,options:['沿 Trie 中不存在的前缀立即剪枝','递归结束恢复当前格子的访问标记','全局永久标记每个访问过的格子','清除已经输出的词尾标记防止重复输出'],answer:[0,1,3],explanation:md`格子禁用范围是当前路径；全局永久 visited 会误删不同起点和不同路径的合法搜索。`},
  {id:'q6',type:'order',difficulty:2,prompt:md`用 Trie 匹配一个完整字符串的流程：`,items:['从根节点开始','逐字符查找对应孩子，缺边立即失败','全部字符消费后检查词尾标记'],explanation:md`不能在中间遇到词尾就提前返回真，因为剩余字符也必须匹配；只走完路径也不足以确认完整词。`},
  {id:'q7',type:'single',difficulty:3,prompt:md`词表 {ab,a,##bc}，按本关 WordPiece 规则编码 abc 的结果是？`,options:['[UNK]','a, ##bc','ab, [UNK]'],answer:0,explanation:md`先选最长的 ab，剩余 c 无 ##c，整词失败。规则不回溯，且不能保留部分成功 token。`},
  {id:'q8',type:'judge',difficulty:3,prompt:md`允许任意数量的点号通配符后，Trie 查询仍保证 O(L)。`,answer:false,explanation:md`每个通配符可能分叉到全部孩子，最坏需要遍历大量节点；普通单路径查询的界不能直接套用。`},
 ],
 guided:[{id:'g-trie',title:'字典节点 Trie 的插入与查询',intro:md`实现 contains(words,query)，把 words 插入 Trie 后检查完整词；支持空串。词尾使用不会与字符键混淆的 None。`,template:py`
  def contains(words, query):
      root = {}
      for word in words:
          node = root
          for ch in word:
              node = node.setdefault({{1}}, {})
          node[None] = {{2}}
      node = root
      for ch in query:
          if ch not in node:
              return False
          node = node[ch]
      return {{3}}
 `,steps:[{blank:1,instruction:md`每个字符对应一条边。`,answer:'ch'},{blank:2,instruction:md`走完整个词后设置终止标记。`,answer:'True'},{blank:3,instruction:md`前缀走完后仍需检查词尾。`,answer:'None in node'}],tests:[{name:'完整词与前缀',code:py`assert contains(['apple'],'apple'); assert not contains(['apple'],'app')`},{name:'空词与重复',code:py`assert contains(['','car','car'],''); assert not contains([],'')`}]}],
 challenges:[
 {id:'c-trie',title:'实现 Trie 词集合',difficulty:'medium',source:'LC 208',description:md`实现 Trie()，提供 insert(word: str)、search(word: str) -> bool、starts_with(prefix: str) -> bool。大小写敏感，允许空串，重复插入不改变集合。空前缀总存在；空词仅在插入空串后存在。每次操作平均 O(L) 时间。`,starter:py`
  class Trie:
      def __init__(self):
          raise NotImplementedError
      def insert(self, word: str):
          raise NotImplementedError
      def search(self, word: str) -> bool:
          raise NotImplementedError
      def starts_with(self, prefix: str) -> bool:
          raise NotImplementedError
 `,solution:py`
  class Trie:
      def __init__(self):
          self.root = {}
      def insert(self, word: str):
          node = self.root
          for ch in word:
              node = node.setdefault(ch, {})
          node[None] = True
      def _walk(self, text):
          node = self.root
          for ch in text:
              if ch not in node:
                  return None
              node = node[ch]
          return node
      def search(self, word: str) -> bool:
          node = self._walk(word)
          return node is not None and None in node
      def starts_with(self, prefix: str) -> bool:
          return self._walk(prefix) is not None
 `,explanation:md`用嵌套字典表示稀疏孩子，None 是词尾标记，不会与单个字符串字符冲突。共享 _walk 消除重复遍历逻辑；注意空字典也是合法节点，不能用 bool(node) 判断路径存在。每次平均 O(L)，总空间与不同前缀数成正比。`,hints:[md`完整词和前缀查询共用路径检查，末尾条件不同。`,md`用专门的标记保存词尾，包含根节点对应的空词。`],setup:py`import random`,tests:[{name:'典型查询',code:py`t=Trie(); t.insert('apple'); assert t.search('apple') is True; assert t.search('app') is False; assert t.starts_with('app') is True`},{name:'空串',code:py`t=Trie(); assert t.starts_with('') is True; assert t.search('') is False; t.insert(''); assert t.search('') is True`},{name:'大小写与重复',hidden:true,code:py`t=Trie(); t.insert('A'); t.insert('A'); assert t.search('A'); assert not t.search('a'); assert not t.starts_with('AB')`},{name:'随机集合对拍',hidden:true,code:py`
  rnd=random.Random(208); t=Trie(); words=set()
  for _ in range(250):
      w=''.join(rnd.choice('abc') for _ in range(rnd.randrange(7)))
      if rnd.random()<.5: t.insert(w); words.add(w)
      got=t.search(w); exp=w in words; assert got==exp, f"词 {w}：期望 {exp}，得到 {got}"
      exp=(w=='') or any(s.startswith(w) for s in words); got=t.starts_with(w)
      assert got==exp, f"前缀 {w}：期望 {exp}，得到 {got}"
 `}]},
 {id:'c-wordpiece',title:'手写 WordPiece 最长匹配',difficulty:'hard',source:'大模型分词',description:md`实现 wordpiece(word: str, vocab: set[str]) -> list[str]。输入是已经预分词的小写英文字母词，词表 token 仅含小写字母和可选的开头 ## 标记；首片段不加标记，后续片段加 ## 查词表。每步选择最长匹配，某步失败则整个词返回 ['[UNK]']，不回溯。空串返回 []。n≤100；无需规范化、预分词或词表训练。例：playing,{play,##ing} → [play,##ing]。可以用两棵 Trie 达到 O(n²) 字符访问。`,starter:py`
  def wordpiece(word: str, vocab: set[str]) -> list[str]:
      raise NotImplementedError
 `,solution:py`
  def wordpiece(word: str, vocab: set[str]) -> list[str]:
      roots = [{}, {}]
      for token in vocab:
          continuation = token.startswith('##')
          piece = token[2:] if continuation else token
          if not piece:
              continue
          node = roots[int(continuation)]
          for ch in piece:
              node = node.setdefault(ch, {})
          node[None] = token
      result, start = [], 0
      while start < len(word):
          node = roots[int(start > 0)]
          end, best_end, best_token = start, start, None
          while end < len(word) and word[end] in node:
              node = node[word[end]]
              end += 1
              if None in node:
                  best_end, best_token = end, node[None]
          if best_token is None:
              return ['[UNK]']
          result.append(best_token)
          start = best_end
      return result
 `,explanation:md`分别建立首片段与续接片段 Trie，扫描时记住最后经过的词尾。词尾以后路径失败时退回最长合法终点，并从新位置继续。失败就丢弃之前的片段。建表 O(S)，单词匹配最坏 O(n²)；若重复处理许多词，应该复用 Trie，避免每次重建。词表约定 ## 仅作续接标记。`,hints:[md`首片段和续接片段属于两个不同命名空间。`,md`沿 Trie 走时记录最近词尾，而不是在第一个词尾停止。`,md`局部失败意味着整个词未知。`],setup:py`
  import random
  def _ref_wp(word,vocab):
      out=[]; start=0
      while start<len(word):
          for end in range(len(word),start,-1):
              token=('##' if start else '')+word[start:end]
              if token in vocab: out.append(token); start=end; break
          else: return ['[UNK]']
      return out
 `,tests:[{name:'续接',code:py`got=wordpiece('playing',{'play','##ing'}); assert got==['play','##ing'], f"得到 {got}"`},{name:'空与未知',code:py`assert wordpiece('',set())==[]; assert wordpiece('xyz',{'x'})==['[UNK]']`},{name:'贪心不回溯',hidden:true,code:py`got=wordpiece('abc',{'ab','a','##bc'}); assert got==['[UNK]'], f"期望整词未知，得到 {got}"`},{name:'随机朴素匹配对拍',hidden:true,code:py`
  rnd=random.Random(212)
  for _ in range(120):
      word=''.join(rnd.choice('abc') for _ in range(rnd.randrange(10))); vocab=set()
      for _ in range(30): vocab.add(('##' if rnd.random()<.5 else '')+''.join(rnd.choice('abc') for _ in range(rnd.randrange(1,5))))
      exp=_ref_wp(word,vocab); got=wordpiece(word,vocab); assert got==exp, f"词 {word}：期望 {exp}，得到 {got}"
 `}]},
 ],
 interview:[
 {id:'i-radix',freq:2,q:md`Trie 与压缩前缀树有什么区别？`,a:md`普通 Trie 一条边代表一个字符或 token，只有一个孩子的长链仍需多个节点。压缩前缀树将这些单分支路径合并到一条边中，减少节点和指针开销；查询时要比较边上的整段标签，插入时可能需要分裂边。用于前缀缓存时，树维护的是可共享的前缀关系，还要另外处理块存储、引用计数和淘汰策略。`},
 {id:'i-hash',freq:2,q:md`有哈希集合为什么还需要 Trie？`,a:md`只做完整词查询时哈希集合往往更简洁。Trie 的优势在前缀相关操作：检查前缀、枚举补全、沿字符串寻找最长词尾、约束下一字符候选等。共享前缀也能减少重复存储，但 Python 每个字典节点有额外开销，实际内存未必更小。选择时应比较具体查询类型、字符集和内存布局。`},
 {id:'i-token',freq:3,q:md`WordPiece 编码失败为什么不能回溯？`,a:md`最长匹配优先本身就是规定的编码规则；回溯寻找其它可行分解会得到不同 token 序列，改变输入语义。词表训练与编码是两个步骤，不能将“存在一种切分”当成编码成功标准。实现完整 tokenizer 时还需遵守规范化、预分词、特殊 token 与最大词长等规则，本关刻意只练单词级匹配内核。`},
 ],cheatsheet:md`Trie：字符路径 + 词尾标记；前缀存在≠词存在。插入/查询平均 O(L)。通配符可能分叉；棋盘 visited 必须回溯恢复。WordPiece：首片段原样、续接 ##、最长匹配、失败整词 [UNK]、不回溯。多词复用 Trie；压缩前缀树合并单分支路径。`,
})
