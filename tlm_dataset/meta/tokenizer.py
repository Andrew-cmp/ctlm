from transformers import PreTrainedTokenizerFast, AutoTokenizer
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset
import os
import json
import tqdm


def train_tokenizer(files, save_path, test_length):
    tmp_file = f"{os.path.basename(os.path.abspath(__file__))}.text"
    if os.path.exists(tmp_file):
        os.remove(tmp_file)

    tmp_file_f = open(tmp_file, 'a')
    for file in tqdm.tqdm(files):
        with open(file, 'r') as f:
            lines = f.read().strip().split("\n")
        for line in lines:
            try:
                text = json.loads(line)["text"]
            except:
                text = line
            tmp_file_f.write(text)
            tmp_file_f.write("\n")
    tmp_file_f.close()
    # wordlevel的分词器
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    # [CLS]代表“分类”（Classification）的起始标记。许多预训练模型（如BERT、GPT等）
    # 使用它作为整个输入序列的表示，用于分类任务。
    # [SEP]，代表“分隔符”（Separator），用于分隔两个句子。它帮助模型区分输入序列中的不同部分。
    trainer = WordLevelTrainer(
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[EOS]", "[BOS]"])
    # 根据空格分隔word
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.train([tmp_file], trainer)
    # 后处理模板
    tokenizer.post_processor = TemplateProcessing(
        # 单个句子在句子前后加上 cls 和 sep 两个字符
        single="[CLS] $A [SEP]",
        # 句对,正对两句变化的情况
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
    # PreTrainedTokenizerFast 接受一个 tokenizer_object，这里是你之前训练好的 tokenizer
    # （基于 tokenizers 库的 Tokenizer 对象）。这样，训练好的分词器就被包装成 Hugging Face 
    # 兼容的分词器对象，便于后续与 transformers 库中的模型进行集成。
    tokenizer_fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer)
    tokenizer_fast.add_special_tokens({
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
    })
    tokenizer_fast.save_pretrained(save_path)

    if test_length:
        test_model_max_length(tmp_file, save_path)

    if os.path.exists(tmp_file):
        os.remove(tmp_file)

# 主要目的是测试和调整分词器（Tokenizer）的最大序列长度，
# 以确保它能够处理数据集中所有文本的长度，并优化其性能。下面是对该函数的详细解释：
def test_model_max_length(file, save_path):
    data_files = {}
    data_files["train"] = file
    extension = data_files["train"].split(".")[-1]
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        keep_in_memory=True
    )
    raw_datasets["train"] = load_dataset(
        extension,
        data_files=data_files,
        split=f"train",
        keep_in_memory=True
    )
    tokenizer = AutoTokenizer.from_pretrained(save_path)

    model_max_length = -1

    def tokenize_function(examples):
        encode = tokenizer(examples["text"])
        input_ids = encode["input_ids"]
        nonlocal model_max_length
        for ids in input_ids:
            model_max_length = max(model_max_length, len(ids))
        return encode

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        load_from_cache_file=True,
        desc="Running tokenizer on every text in dataset",
        keep_in_memory=True
    )

    print("model_max_length:", model_max_length)
    tokenizer.model_max_length = ((model_max_length - 1) // 32 + 1) * 32
    tokenizer.save_pretrained(save_path)