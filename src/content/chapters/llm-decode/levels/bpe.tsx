import { defineLevel, md, py } from '@/content/helpers'
import type { Challenge } from '@/content/types'
import { BpeViz } from '../viz/DecodeViz'
import { single,judge,numeric,multi,order,fill,bug,practice } from '../shared'
const c:Challenge={id:'bpe',title:'训练并编码一个字符级 BPE',difficulty:'medium',description:md`实现 \`train_bpe(words: list[str], steps: int) -> list[tuple[str,str]]\` 与 \`encode_bpe(text: str, merges: list[tuple[str,str]]) -> np.ndarray\`。每个列表元素视为独立单词，不跨词合并；重复单词计入频次。训练每轮统计所有相邻 pair（含重叠出现），选次数最多者，并列取 Python 元组字典序最小者，然后各词从左到右进行不重叠替换；无 pair 就停止。编码固定使用 merge rank，每轮选择当前最低 rank 的 pair 并替换。输出字符串 dtype 的一维 NumPy 数组。约束总字符数 ≤ 1000，steps ≤ 30。例：['aaa'] 训练 1 步得到 [('a','a')]，编码 'aaa' 得 ['aa','a']。这是字符教学模型，不包含正式 byte-level 分词器的预切分、规范化和特殊 token。`,starter:py`import numpy as np
def train_bpe(words: list[str], steps: int):
    raise NotImplementedError
def encode_bpe(text: str, merges: list[tuple[str,str]]) -> np.ndarray:
    raise NotImplementedError`,solution:py`import numpy as np
from collections import Counter
def _merge(tokens, pair):
    out, i = [], 0
    while i < len(tokens):
        if i+1 < len(tokens) and (tokens[i], tokens[i+1]) == pair:
            out.append(tokens[i] + tokens[i+1])
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out

def train_bpe(words: list[str], steps: int):
    corpus = [list(w) for w in words]
    merges = []
    for _ in range(steps):
        counts = Counter(pair for word in corpus for pair in zip(word, word[1:]))
        if not counts:
            break
        pair = min(counts, key=lambda x: (-counts[x], x))
        merges.append(pair)
        corpus = [_merge(word, pair) for word in corpus]
    return merges

def encode_bpe(text: str, merges: list[tuple[str,str]]) -> np.ndarray:
    tokens = list(text)
    rank = {pair: i for i, pair in enumerate(merges)}
    while len(tokens) > 1:
        pairs = [p for p in zip(tokens, tokens[1:]) if p in rank]
        if not pairs:
            break
        tokens = _merge(tokens, min(pairs, key=rank.get))
    return np.asarray(tokens, dtype=str)`,solutionAlt:[{label:'PyTorch 版（仅展示）',code:py`# BPE 是离散字符串算法，训练和 rank 合并沿用上面的 Python 实现。
# 接入 PyTorch 时在最终词表查 ID，不对字符串使用浮点 Tensor。
import torch
def encode_tensor(text, merges, vocab):
    tokens = list(text)
    rank = {pair: i for i, pair in enumerate(merges)}
    while len(tokens) > 1:
        pairs = [p for p in zip(tokens, tokens[1:]) if p in rank]
        if not pairs: break
        a, b = min(pairs, key=rank.get)
        result, i = [], 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i:i+2] == [a, b]:
                result.append(a+b)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        tokens = result
    return torch.tensor([vocab[token] for token in tokens], dtype=torch.long)`}],explanation:md`训练的统计量来自整个当前语料，编码的优先级来自训练好的合并表。统计重叠 pair 不等于实际替换重叠片段：aaa 有 2 次 aa，但一次扫描只能合并出 aa,a。每轮重算统计的朴素训练为 $O(MN)$（忽略字符串拼接长度），编码最坏 $O(N^2)$，适用于题目规模；工业实现用邻接表与堆增量维护候选。`,hints:[md`先写从左到右、不重叠的 pair 替换辅助函数。`,md`Counter 统计所有词内相邻对；(-count,pair) 确定并列顺序。`],tests:[{name:'重叠 pair',code:py`assert train_bpe(['aaa'],1)==[('a','a')], "aaa 应选择 aa"
a=encode_bpe('aaa',[('a','a')]); assert a.tolist()==['aa','a'], f"期望 ['aa','a']，得到 {a}"`},{name:'rank 优先',code:py`a=encode_bpe('abc',[('b','c'),('a','b'),('a','bc')]); assert a.tolist()==['abc'], f"期望 ['abc']，得到 {a}"`},{name:'空和单字',hidden:true,code:py`assert train_bpe(['','x'],30)==[], "无 pair 应终止"
a=encode_bpe('',[]); assert a.shape==(0,), f"期望 (0,)，得到 {a.shape}"
assert train_bpe(['ab'],0)==[], "零轮不可合并"`},{name:'频次、并列和不可跨词',hidden:true,code:py`assert train_bpe(['ab','ac','ac'],1)==[('a','c')], "重复词要计频次"
assert train_bpe(['ab','ac'],1)==[('a','b')], "并列按字典序"
assert train_bpe(['a','b'],3)==[], "不可跨词合并"
merges=train_bpe(['lowest','lower','lowest'],8)
for text in ['lowest','lower','new','中文','aaaa']:
    got=encode_bpe(text,merges)
    assert ''.join(got)==text, f"拼接应还原 {text}，得到 {got}"`}],followUps:[{id:'byte',q:md`byte-level BPE 为什么能覆盖任意文本？`,a:md`基础字母表包含全部 256 个字节，UTF-8 文本可以先退回字节，再按 merge rank 合并。因此不因未知 Unicode 字符丢失信息，但非法字节切片不一定能独立解码成字符，展示与流式解码必须处理 UTF-8 边界。`}]}
export default defineLevel({learn:[{id:'train',title:'从压缩相邻对到可复用词表',body:md`词级词表会遇到未登录词，字符级又会把序列拉长。BPE 从基础符号开始，反复合并频率最高的相邻 pair，让常见片段获得独立 token、罕见词仍可拆开表示。训练保存的是有序 merge 表，合并后要更新邻接关系。例如 aaa 中 aa 统计出现 2 次，但替换必须从左到右不重叠，得到 aa,a；若同时替换两个重叠 pair 会重复消费字符。实际 tokenizer 还包含规范化、预切分和特殊 token 规则，不能只保存词表就认为重建了完整 tokenizer。`,viz:BpeViz},{id:'encode',title:'编码使用 rank，而非重新训练',body:md`训练完成后，每个 pair 有固定 rank。对新文本，每轮只合并当前存在且 rank 最小的相邻对，直到无合法 pair。不要在当前输入里重新找最高频 pair；这样同一个词会随句子上下文得到不受训练规则支持的 token。byte-level BPE 用字节作基础，保留任意 UTF-8 文本的可表达性。可逆性来自字节覆盖与解码规则，不代表每个 token 都对应一个完整 Unicode 字符；汉字、emoji 和组合字符尤其需要测试。`},{id:'tradeoff',title:'BPE、WordPiece 与 Unigram 的取舍',body:md`BPE 基于合并秩；常见 WordPiece 编码在词表上做最长匹配；Unigram 给候选子词分配概率，通过动态规划寻找高概率切分，还能采样多种切分。SentencePiece 是支持 BPE 与 Unigram 等方法的工具，不是与它们平级的单一算法。增大词表通常减少 token 数，却增加 embedding 和输出投影的参数与计算；多语言覆盖不均还会影响相同文本的 token 成本。结论应依据数据分布、压缩率、未知字符处理与任务效果，而不只比较词表规模。参考：[BPE 原论文](https://arxiv.org/abs/1508.07909)、[SentencePiece](https://github.com/google/sentencepiece)。`}],quiz:[single('encode',1,md`BPE 编码时决定合并优先级的是？`,['当前句子 pair 频次','训练好的 merge rank','合并后字符串长度','token ID 数值大小'],1,md`rank 固定训练所得的操作顺序，token ID 未必等于 rank。`),judge('sp',1,md`SentencePiece 只能训练 Unigram 模型。`,false,md`它也支持 BPE；算法和工具框架不能混为一谈。`),numeric('pairs',2,md`单词 aaaa 中 pair (a,a) 在统计阶段出现几次？`,3,md`相邻起点 0、1、2，共 3 次，替换却只能得到两个 aa。`),multi('bytes',2,md`关于 byte-level BPE，正确的是？`,['可从 256 字节基础表表示 UTF-8 文本','每个 token 必定是完整汉字','预切分规则影响合并范围','解码时可能需要拼接多个 token 的字节'],[0,2,3],md`字节覆盖不等于字符边界对齐；预切分防止某些跨边界合并。`),order('train',md`一轮 BPE 训练的顺序：`,['按语料频次统计邻接 pair','选出最高频 pair 并处理并列','从左到右做不重叠替换','记录 merge rank 并重新统计'],md`替换改变邻接关系，不能永久使用第一轮频次。`),fill('dp',md`Unigram 最大概率切分可在 log 域用 ____(1)____ 算法求最优路径。`,['Viterbi','维特比'],md`Viterbi 在位置节点上比较候选子词边，累加 log 概率并回溯。`),single('vocab',2,md`词表从 32k 增至 64k，d=1024，tied embedding 增加多少参数？`,['32×1024','32000×1024','64000×1024×2','无需增加'],1,md`按 k=1000 计，增加 32000 行×1024；tied 时输出权重不重复计数。`),judge('longest',3,md`任意 BPE merge 表都可直接用词表最长匹配替代，结果必定一致。`,false,md`BPE 的历史合并顺序约束切分。最长匹配忽略 rank，存在重叠候选时可能走另一条路径。`),bug('bug',py`def merge(tokens, pair):
    result = []
    i = 0
    while i < len(tokens):
        if tuple(tokens[i:i+2]) == pair:
            result.append(''.join(pair))
            i += 1
        else:
            result.append(tokens[i])
            i += 1
    return result`,[7],md`合并消费两个符号，应 i+=2。否则第二个符号被重复使用，拼接后无法还原原文。`)],guided:[practice(c,[{find:'i += 2',instruction:md`合并后应消费多少符号？`,explain:md`一次合并消费两个相邻符号，禁止重叠复用。`},{find:'min(counts, key=lambda x: (-counts[x], x))',instruction:md`选择频次最大、并列字典序最小的 pair。`,explain:md`负频次把最大化转换为最小化。`},{find:'min(pairs, key=rank.get)',instruction:md`编码阶段选择下一次合并。`,explain:md`必须以固定 rank 决定优先级。`}])],challenges:[c],interview:[{id:'size',freq:3,q:md`词表越大一定越好吗？`,a:md`不一定。词表大可减少序列长度并提高压缩率，但 embedding 与输出层更大，稀有 token 得到的训练次数更少。词表小会拉长上下文并提高注意力与 KV 成本。应在相同语料上比较不同语言的 token/字节比、下游效果和端到端推理成本。`},{id:'space',freq:2,q:md`为什么空格处理属于模型的一部分？`,a:md`模型学习的是 tokenizer 输出的离散序列，空格是否作为前缀、是否规范化连续空白、是否允许跨词合并都会改变 token ID。训练和推理必须使用一致 tokenizer。对代码或结构化数据，随意去空格更可能改变语义。`},{id:'unicode',freq:2,q:md`如何测试一个 tokenizer 的正确性？`,a:md`先检查特殊 token 策略，再做编码解码往返测试，包括空串、空白、中文、emoji、组合字符和重复字符。对规范化 tokenizer，往返对象应是规范化后的文本而非原字节。还应检查 merge tie-break 与参考实现一致，避免用只有 ASCII 的测试掩盖字节边界错误。`}],cheatsheet:md`训练：计 pair → 最大频次 → 非重叠替换 → 更新。编码：**最低 merge rank** 优先。byte-level 的基本单位为字节；SentencePiece 是框架。词表增量参数：tied 为 $\Delta Vd$，untied 为 $2\Delta Vd$。`})
