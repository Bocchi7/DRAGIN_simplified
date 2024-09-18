from typing import List, Tuple
from dataclasses import dataclass
import logging
import string

import spacy
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers.generation.utils import GenerateDecoderOnlyOutput # .generate(...)的输出。用来代码提示
from transformers.generation.utils import GenerationMixin # 可以用GenerationMixin.generate查代码
from transformers import PreTrainedModel, LlamaTokenizer # 用来代码提示

from retriever import BM25

DEBUG = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

@dataclass
class Block:
    text: str = None
    tokens: List[str] = None # 选择存tokens而不是ids是因为合并词时处理"▁"方便。
    range_: List[Tuple[int, int]] = None # 合并后的一个单位记作word。这里记下各word的区间（左闭右开）
    @property
    def len_tokens(self):
        return len(self.tokens)
    @property
    def len_words(self):
        return len(self.range_)

def merge_blocks(blocks: List[Block]) -> Block:
    text = "".join([block.text for block in blocks])
    tokens = sum([block.tokens for block in blocks], [])
    range_ = []
    st = 0
    for block in blocks:
        if block.range_:
            for l, r in block.range_:
                range_.append((st+l, st+r))
            st = range_[-1][1]
    return Block(text=text, tokens=tokens, range_=range_)

class Counter:
    def __init__(self):
        self.retrieve = 0
        self.generate = 0
        self.hallucinated = 0
        self.token = 0
        self.sentence = 0

    def add_generate(self, text, tokenizer):
        self.generate += 1
        ids = tokenizer(text, return_tensors="pt")['input_ids'][0].tolist()
        self.token += len(ids)
        sentences = [sent.text for sent in nlp(text).sents]
        self.sentence += len(sentences)

    def calc(self, other_counter):
        return {
            "retrieve_count": self.retrieve - other_counter.retrieve, 
            "generate_count": self.generate - other_counter.generate,
            "hallucinated_count": self.hallucinated - other_counter.hallucinated, 
            "token_count": self.token - other_counter.token, 
            "sentence_count": self.sentence - other_counter.sentence 
        }

@dataclass
class GeneratorOutput:
    '''
返回值：
- ended: 是否检测到结束。通过eos_token来判定
- blocks: 文本分段存储的各块。在后面的使用中，将根据[demo, "Question:", question, "\\nAnswer:", text]
    ，以及新生成的new_text划分。将要在各块内做好合并词操作。约定合并词不会跨过两段的交界。
- atten: (len_words, len_new_words)。\
    已经对多头取平均。注意是在合并词之后意义下的。
- max_atten: (len_new_words,)
- entropies: (len_new_words,)

注记：源代码以字符串形式存储了合并词的结果，并且将词列转成字符串的时候用空格隔开。\
    我认为这不好，因为这给分词器的行为加了较强的预设。
- 一方面分词器编码时有很多东西是以转义的方式存储，比如" "、"\\n"、"\\t"分别变成"▁"、"<0x0A>"、"<0x09>"（还有更多我不清楚的）。\
    将它们合并成新的字符串容易让分词器搞不懂它原来的意思，转不回去。
- 另一方面，不能保证合并后词是用空格隔开的。比如"\\n"和下一个词之间就不用空格。
    '''
    ended: bool
    blocks: List[Block] = None
    merged_blocks: Block = None
    atten: Tensor = None
    max_atten: Tensor = None
    entropies: Tensor = None
    @property
    def new_text(self):
        return self.blocks[-1].text
    @property
    def len_new_words(self):
        return self.blocks[-1].len_words
    
class Generator:
    def __init__(
        self,
        model_name_or_path: str
    ):
        logger.info(f"Loading model from {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # self.model_config = AutoConfig.from_pretrained(model_name_or_path)
        # assert self.model_config.model_type == "llama"
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto") # device_map表示自动分到显卡上
        self.model: PreTrainedModel
        logger.info(f"device = {self.model.device}")
        # 总之只参考了llama
        self.space_token = "▁"
        self.tokenizer.pad_token = self.tokenizer.eos_token # 将填充标记设置为结束标记
        # 看llama的tokenizer_config可知，其bos_token为"<s>"，eos_token为"</s>"，pad_token为null。
        
        self.tokens_cannot_merged = {
            self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode("0" + ch)[-1:])[0]
            # 带上"0"是为了防止生成和"▁"的结合体
            for ch in string.whitespace + string.punctuation
        } | {self.space_token, self.tokenizer.bos_token, self.tokenizer.eos_token}

    def simply_generate(
        self,
        input_text: str,
        max_length: int
    ) -> Tuple[bool, str]:
        '''
        return ended, new_text
        '''
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.model.device) # (batch_size=1, input_length)
        input_length = input_ids.shape[1]
        output_ids = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_length,
            stop_strings = "\n",
            tokenizer=self.tokenizer
        )[0, input_length:]
        if output_ids.shape[0] == 0:
            # 应该没有这种情况吧，output_ids最少也该输出eos_token
            logger.info("generate '' in simply_generate()!")
            return True, ""
        if output_ids[0] == self.tokenizer.bos_token_id:
            output_ids = output_ids[1:]
        if output_ids[-1] == self.tokenizer.eos_token_id:
            return True, self.tokenizer.decode(output_ids[:-1])
        return False, self.tokenizer.decode(output_ids)

    def tokenize(
        self,        
        text: str,
        is_start: bool = False # 若否，则删除bos_token
    ):
        ids = self.tokenizer.encode(text) # List[int]
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        if not is_start and tokens[0] == self.tokenizer.bos_token:
            tokens = tokens[1:]
        return tokens
        
    def merge_tokens(
        self,
        tokens
    ) -> List[Tuple[int, int]]:
        range_ = []
        for i, t in enumerate(tokens):
            if i == 0 or t.startswith(self.space_token) \
                or tokens[i] in self.tokens_cannot_merged \
                or tokens[i-1] in self.tokens_cannot_merged:
                range_.append([i, i+1]) # 作为新词
            else:
                range_[-1][1] += 1
        return range_

    def build_block(
        self,        
        text: str,
        is_start: bool = False # 若否，则删除bos_token
    ) -> Block:
        tokens = self.tokenize(text, is_start=is_start)
        range_ = self.merge_tokens(tokens)
        return Block(text=text, tokens=tokens, range_=range_)

    def generate(
        self,
        input_texts: List[str], # 输入已经分段，后面实现中为：[demo, "\nQuestion:", question, "\nAnswer:", text]
        max_length: int,
    ) -> GeneratorOutput:
        blocks = []
        for text in input_texts:
            blocks.append(self.build_block(text, is_start=not blocks))

        input_tokens = sum([block.tokens for block in blocks], [])
        input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(input_tokens)], device=self.model.device)
        input_len_tokens = len(input_tokens)

        # if DEBUG:
        #     print("用于初始生成的input_text：")
        #     print(self.tokenizer.convert_tokens_to_string(input_tokens))
        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_length,
            return_dict_in_generate=True,
            output_scores=True, # scores: Tuple[Tensor(batch_size, vocab_size)] len(scores) == generated_length
            # output_attentions=True
            stop_strings="\n",
            tokenizer=self.tokenizer
        )
        outputs: GenerateDecoderOnlyOutput

        tokens = self.tokenizer.convert_ids_to_tokens(outputs.sequences[0, input_len_tokens:]) # List[str]
        ended = (tokens[-1] == self.tokenizer.eos_token)
        if ended:
            tokens = tokens[:-1]
            # 注：如果tokens中有"</s>"（"<s>"也一样），转换成字符串后仍然会保留。因此在这里删掉。
        text = self.tokenizer.convert_tokens_to_string(tokens)
        range_ = self.merge_tokens(tokens)
        new_block = Block(text=text, tokens=tokens, range_=range_)

        blocks.append(new_block)
        merged_blocks = merge_blocks(blocks)

        # 下面这些和源代码差别挺大的。
        # - 源代码是单独new_tokens另外算了一组attention。这里尝试在原attention的基础上，截取对new_tokens的部分并归一化。
        # - 源代码的求最大attention的操作在求合并意义下的attention之前，这里反之。这里一开始就求了合并意义下的attention。
        # - 源代码是先求最大attention再对多头取平均，这里反之。

        atten = self.model(outputs.sequences, output_attentions=True).attentions[-1][0][:, -new_block.len_tokens:, :] # (num_heads, new_len_tokens, len_tokens)
        # 因为generate的attention的输出格式很怪，偷懒重新算attention。
        # 理论上整理一下格式，并补上最后一个token的对前面的attention也可以。
        # outputs.attentions: Tuple[Tuple[Tensor(batch_size, num_heads, generated_length, sequence_length)]]
        # 最外层generated_length，次外层num_layers
        # 对于生成的第一个token，generated_length==input_len_tokens；其他generated_length==1
        # 对于生成的每个token，sequence_length==生成它之前输入序列的长度

        # 合并意义下的attention：
        atten = atten.mean(dim=0)
        atten = torch.stack([atten[:, l:r].sum(dim=-1) for l, r in merged_blocks.range_], dim=-1) # 同一注意者(token)，将同组被注意者(token)的attention相加
        atten = torch.stack([atten[l:r, :].mean(dim=-2) for l, r in range_], dim=-2)  # 同组的注意者(token)，对同一被注意者(word)的attention取平均
        # assert atten.shape == (new_block.len_words, merged_blocks.len_words)

        # 求被注意的最大attention
        atten_to_new = atten[:, -new_block.len_words:]
        atten_to_new /= atten.sum(dim=-1,keepdim=True) + 1e-10 # 归一化
        max_atten, _ = atten_to_new.max(dim=1)
        # assert max_atten.shape == (new_block.len_words,)

        # 熵
        probs = torch.stack(outputs.scores).softmax(dim=-1) # (new_len_tokens, batch_size, vocab_size)
        entropies = (-probs * torch.log(probs + 1e-10)).sum(dim=-1) # (new_len_tokens, batch_size)
        # if DEBUG:
        #     print("打印候选词及合并前entropies的情况：")
        #     for i, token in enumerate(tokens):
        #         print(f"第{i}个 {token}：entropy={entropies[i][0]} 候选：", end="")
        #         li = [(probs[i][0][id].item(), self.tokenizer._convert_id_to_token(id)) for id in range(probs.shape[-1]) if probs[i][0][id] > 1e-6]
        #         li.sort(reverse=True)
        #         for j, (p, t) in enumerate(li):
        #             if p < 1e-6 or j >= 10:
        #                 break
        #             print(f"({t},{p})", end=" ")
        #         print()
        entropies = torch.stack([entropies[l:r, 0].sum() for l, r in range_])
        # assert entropies.shape == (new_block.len_words,)
        # 源代码是取平均，这里用的是求和。
        
        return GeneratorOutput(
            ended=ended,
            blocks=blocks,
            merged_blocks=merged_blocks,
            atten=atten,
            max_atten=max_atten,
            entropies=entropies
        )

def join_if_nonempty(*li, sep=" "):
    return sep.join([s for s in li if len(s) > 0])

def match(word: str, real_words): # 是否存在实词作为word的子串
    for real_word in real_words:
        if real_word in word: # 判断是否为子串
            return True
    return False

def get_top_sentence(text):
    prev = ""
    for sent in nlp(text).sents:
        prev += sent.text
        sent = sent.text.strip()
        if len(sent) > 0:
            return prev
    return ""

@dataclass
class CheckerOutput:
    '''
- hallucination: 是否存在幻觉
- curr_st: 幻觉句子的起始位置
- curr_en: 幻觉句子的终止位置
- curr_thres: 幻觉句子的各词是否达到阈值
    '''
    hallucination: bool 
    curr_st: int = None
    curr_en: int = None
    curr_thres: List[bool] = None

class DRAGIN:
    def __init__(self, args):
        for k, v in args.__dict__.items():
            setattr(self, k, v)
        self.generator = Generator(self.model_name_or_path)
        self.tokenizer = self.generator.tokenizer
        self.retriever = BM25("wiki" if "es_index_name" not in args else self.es_index_name)
        self.counter = Counter()

    def hallucination_check(
        self,
        outputs: GeneratorOutput
    ) -> CheckerOutput: # 幻觉检测
        # 这里的实现有个问题是调用了scapy用来分句，但不知道scapy的分词跟LlamaTokenizer差多少。
        # 如果能在之前的分词的基础上进行分句就好了。
        if DEBUG:
            print("开始检测幻觉。")
        new_block = outputs.blocks[-1]
        sentences = [sent.text.strip() for sent in nlp(new_block.text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        if DEBUG:
            print("调用spacy得到的分句结果：")
            for i, sent in enumerate(sentences):
                print(f"句子{i}：{sent}")
        wid = 0
        for sid, sent in enumerate(sentences):
            wl, wr = wid, wid
            if wid == new_block.len_words:
                break
            while wr < new_block.len_words and sent not in self.tokenizer.convert_tokens_to_string(
                new_block.tokens[new_block.range_[wl][0]:new_block.range_[wr][1]]
            ):
                wr += 1
                # assert wr < new_block.len_words, "sent not in the remainder of new_text!"
                # 这种情况是存在的，因为LlamaTokenizer的奇怪分词和spacy的奇怪断句。
            if wr < new_block.len_words:
                wr += 1 # sent in words[wl, wr)
            wid = wr
            if wl == wr:
                continue
            if DEBUG:
                print("当前句子：", self.tokenizer.convert_tokens_to_string(new_block.tokens[new_block.range_[wl][0]:new_block.range_[wr-1][1]]), sep="\n")
            # 按句子归一化和乘句子长度，和源代码保持一致。是否合适有待探究。
            max_atten_sent = outputs.max_atten[wl: wr]
            max_atten_sent = max_atten_sent * (wr - wl) / (max_atten_sent.sum() + 1e-10)
            value = max_atten_sent * outputs.entropies[wl: wr]
            thres = (value > self.hallucination_threshold)
            if DEBUG:
                print("word|max_atten_sent|entropy|value|thres：")
                for i in range(wl, wr):
                    print(self.tokenizer.convert_tokens_to_string(new_block.tokens[new_block.range_[i][0]:new_block.range_[i][1]]), 
                          max_atten_sent[i-wl].item(),
                          outputs.entropies[i-wl].item(),
                          value[i-wl].item(),
                          thres[i-wl].item(), sep="|")
            if True in thres:
                doc = nlp(sent)
                real_words = set(token.text for token in doc if token.pos_ in 
                    ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
                for i in range(wl, wr):
                    tl, tr = new_block.range_[i]
                    word = self.tokenizer.convert_tokens_to_string(new_block.tokens[tl:tr])
                    if not match(word, real_words):
                        if DEBUG and thres[i-wl]:
                            print(f"第{i-wl}号：{self.tokenizer.convert_tokens_to_string(new_block.tokens[new_block.range_[i][0]:new_block.range_[i][1]])}曾达阈值，但是虚词")
                        thres[i-wl] = False
                if True in thres:
                    return CheckerOutput(hallucination=True, curr_st=wl, curr_en=wr, curr_thres=thres)
            if DEBUG:
                print("当前句子未检测出幻觉。准备下一个句子。")
        return CheckerOutput(hallucination=False)

    def generate_retrieve_qry(self, outputs: GeneratorOutput, check_info: CheckerOutput):
        # 回忆：input_texts=[demo, "\nQuestion:", question, "\nAnswer:", text]
        # 在意的前缀在question和text+ptext，关键词将只会在这里选取。

        ques_st = outputs.blocks[0].len_words + outputs.blocks[1].len_words
        ques_en = ques_st + outputs.blocks[2].len_words
        text_st = ques_en + outputs.blocks[3].len_words
        text_en = text_st + outputs.blocks[4].len_words + check_info.curr_st

        ques_atten = outputs.atten[check_info.curr_st:check_info.curr_en, ques_st:ques_en]
        text_atten = outputs.atten[check_info.curr_st:check_info.curr_en, text_st:text_en]
        
        ques_atten = ques_atten[check_info.curr_thres, :].sum(dim=0)
        text_atten = text_atten[check_info.curr_thres, :].sum(dim=0)

        doc = nlp(outputs.merged_blocks.text)
        real_words = set(token.text for token in doc if token.pos_ in 
            ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])

        real_pairs = []
        for i in range(ques_st, ques_en):
            a = ques_atten[i - ques_st]
            tl, tr = outputs.merged_blocks.range_[i]
            word = self.tokenizer.convert_tokens_to_string(outputs.merged_blocks.tokens[tl:tr])
            if match(word, real_words):
                real_pairs.append((a, word, i))
        for i in range(text_st, text_en):
            a = text_atten[i - text_st]
            tl, tr = outputs.merged_blocks.range_[i]
            word = self.tokenizer.convert_tokens_to_string(outputs.merged_blocks.tokens[tl:tr])
            if match(word, real_words):
                real_pairs.append((a, word, i))
        
        if "retrieve_keep_top_k" in self.__dict__:
            top_k = min(self.retrieve_keep_top_k, len(real_pairs))
        elif "retrieve_keep_ratio" in self.__dict__:
            top_k = int(len(real_pairs) * self.retrieve_keep_ratio)

        real_pairs.sort(key=lambda x: -x[0])
        real_pairs = real_pairs[:top_k]
        real_pairs.sort(key=lambda x: x[2])

        return " ".join([x[1] for x in real_pairs])

    def inference(self, question, demo, case):
        text = ""
        demo = "\n".join([d["case"] for d in demo]) # 为了避免模型生成新的问答，在这里试图终止。
        # 在WikiMultiHopQA中case = f'Question: {question}\nAnswer:'。因此后面只用question而不用case。
        if DEBUG:
            print("准备开始推理。")
        while True:
            old_len = len(text)
            # demo：说明（包括fewshot）；text：已经回答的部分
            outputs = self.generator.generate(
                input_texts=[demo, "\nQuestion:", question, "\nAnswer:", text], 
                max_length=self.generate_max_length,
            )
            if DEBUG:
                print("初始生成新文本为：", outputs.new_text, sep="\n")
            if self.use_counter == True:
                self.counter.add_generate(outputs.new_text, self.generator.tokenizer)
            if outputs.new_text.strip() == "":
                if DEBUG:
                    print("检测到只生成了空白字符，将中断生成。")
                break
            check_info = self.hallucination_check(outputs)
            if not check_info.hallucination:
                if DEBUG:
                    print("未检测出幻觉。")
                text = join_if_nonempty(text, outputs.new_text.strip())
                if DEBUG:
                    print("当前已生成文本：", text, sep="\n")
                if outputs.ended or outputs.merged_blocks.len_tokens > self.generate_max_length:
                    if DEBUG:
                        if outputs.ended:
                            print("检测到终止符。" if outputs.ended else "检测到文本已达到最大长度。")
                    break
            else:
                if DEBUG:
                    print("检测出幻觉。准备检索。")
                retrieve_qry = self.generate_retrieve_qry(outputs, check_info)
                if DEBUG:
                    print(f"retrieve_qry: {retrieve_qry}")
                # 和源代码不同。
                # 源代码并不直接拿之前的generate中的数据，而是另外搞个前缀字符串(question+text+ptext)，重新算attention。
                # 有可能是为了消除demo的影响（还有case中question以外的格式部分）。
                # 这里用的是源自之前generate部分的attention，将需要的部分提取出来。
                # 可以选择归一化和之前的方法对齐，这里暂时不用。
                # 归一化是否更好有待商榷，问题在于，幻觉在多大程度上被few-shot所影响。
                # 如果某词的注意力集中在few-shot部分，那么将归一化的后的注意力值强行安在限制的区域内，说不定会造成偏差。

                docs = self.retriever(retrieve_qry, topk=self.retrieve_topk)
                self.counter.retrieve += 1
                if DEBUG:
                    print("检索得到补充资料：", docs, sep="\n")
                # 重新生成新的文本：
                # 因为这里已经不涉及attention，所以懒得保证分词一致性了，但理论上应该也可以。
                # 可能更规范的做法是，除了预处理、输出结果和与spacy、检索器的交互之外，完全以token为基本单位存储
                prompt = demo
                prompt += "\nContext:\n"
                for i, doc in enumerate(docs):
                    prompt += f"[{i+1}] {doc}\n"
                prompt += "Answer in the same format as before.\n"
                for i in [1, 2, 3]: # "Question:", question, "\nAnswer:"
                    prompt += outputs.blocks[i].text
                text = self.tokenizer.convert_tokens_to_string(
                    outputs.blocks[-2].tokens # text
                    + outputs.blocks[-1].tokens[:outputs.blocks[-1].range_[check_info.curr_st][0]] # ptext
                )
                prompt += text
                ended, new_texts = self.generator.simply_generate(
                    prompt, 
                    max_length=self.generate_max_length,
                )
                if self.use_counter == True:
                    self.counter.add_generate(new_texts, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                new_text = get_top_sentence(new_texts)
                # text += new_text
                text = join_if_nonempty(text, new_text.strip())
                if DEBUG:
                    print("重新生成新文本：", new_text, sep="\n")
                if DEBUG:
                    print("当前已生成文本：", text, sep="\n")
                if ended and len(new_text) >= len(new_texts.strip()):
                    if DEBUG:
                        print("检测到终止符。")
                    break
                if len(self.tokenizer.encode(text)) > self.generate_max_length:
                    if DEBUG:
                        print("检测到文本已达到最大长度。")
                    break
            if old_len >= len(text): # 应该不会出现这种情况吧？
                logger.info("old_len >= len(text) !")
                break
        if DEBUG:
            print("推理结束。最终生成文本：", text, sep="\n")
        return text
            