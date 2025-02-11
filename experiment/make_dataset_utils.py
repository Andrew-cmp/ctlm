from transformers import AutoTokenizer
from datasets import load_dataset


def json_dfs_without_bracket(json, text_list: list):
    if isinstance(json, dict):
        json = dict(sorted(json.items()))
        for key, val in json.items():
            text_list.append(key)
            json_dfs_without_bracket(val, text_list)
    elif isinstance(json, (list, tuple)):
        for it in json:
            json_dfs_without_bracket(it, text_list)
    elif isinstance(json, (str, int, float, bool)):
        text_list.append(str(json))
    else:
        assert (False)


def json_dfs_with_bracket(json, text_list: list):
    if isinstance(json, dict):
        text_list.append("{")
        json = dict(sorted(json.items()))
        for idx, (key, val) in enumerate(json.items()):
            if idx != 0:
                text_list.append(",")
            text_list.append(key)
            json_dfs_with_bracket(val, text_list)
        text_list.append("}")
    elif isinstance(json, list):
        text_list.append("[")
        for idx, it in enumerate(json):
            if idx != 0:
                text_list.append(",")
            json_dfs_with_bracket(it, text_list)
        text_list.append("]")
    elif isinstance(json, (str, int, float, bool)):
        text_list.append(str(json))
    else:
        assert (False)


def json_to_token(json_lines):
    token_list = []
    for json_line in json_lines:
        text_list = []
        json_dfs_without_bracket(json_line["text"], text_list)
        json_line["text"] = " ".join(text_list)
        token_list.append(json_line)
    return token_list

# 用
def make_dataset(file, dataset_path, tokenizer_path, for_clm_or_mlm, valid_percentage=5):
    data_files = {}
    data_files["train"] = file
    dataset_name = data_files["train"].split(".")[-1]
    raw_datasets = load_dataset(
        dataset_name,
        data_files=data_files,
        keep_in_memory=True
    )
    if valid_percentage > 0:
        raw_datasets["validation"] = load_dataset(
            dataset_name,
            data_files=data_files,
            split=f"train[:{valid_percentage}%]",
            keep_in_memory=True
        )
        raw_datasets["train"] = load_dataset(
            dataset_name,
            data_files=data_files,
            split=f"train[{valid_percentage}%:]",
            keep_in_memory=True
        )
    else:
        raw_datasets["train"] = load_dataset(
            dataset_name,
            data_files=data_files,
            split=f"train",
            keep_in_memory=True
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    column_names = list(raw_datasets["train"].features)
    if for_clm_or_mlm == "clm":
        ## 分词处理
        # 核心作用
        # 将原始文本转换为模型可接受的数字序列（input_ids）
        # 自动生成注意力掩码（attention_mask）
        # 统一序列长度为模型支持的最大长度
        # 复制输入ID作为标签，常见于 自回归模型（如GPT系列）
        # 适用于语言模型预训练或文本生成任务（标签实际应为输入右移一位，需后续处理）
        def tokenize_function(examples):
            output = tokenizer(examples["text"], padding="max_length", max_length=tokenizer.model_max_length)
            #print(output)
            #input("continue")
            # 标签实际应为输入右移一位，需后续处理
            output["labels"] = output["input_ids"].copy()
            del output["token_type_ids"]
            return output
    elif for_clm_or_mlm == "mlm":
        column_names.remove("labels")
        def tokenize_function(examples):
            output = tokenizer(examples["text"], padding="max_length", max_length=tokenizer.model_max_length)
            return output
    else:
        assert(False)

    # 对raw_datasets里面的数据并行使用tokenize_function函数。
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on every text in dataset",
        keep_in_memory=True
    )

    tokenized_datasets.save_to_disk(dataset_path)


def make_dataset_test(file, dataset_path, tokenizer_path, for_clm_or_mlm):
    data_files = {}
    data_files["train"] = file
    extension = data_files["train"].split(".")[-1]
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        keep_in_memory=True
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if for_clm_or_mlm == "clm":
        def tokenize_function(examples):
            output = tokenizer(examples["text"])
            input_ids = output["input_ids"]
            for inp in input_ids:
                del inp[-1]
            attention_mask = output["attention_mask"]
            for mask in attention_mask:
                del mask[-1]
            del output["token_type_ids"]
            return output
    elif for_clm_or_mlm == "mlm":
        def tokenize_function(examples):
            output = tokenizer(examples["text"], padding="max_length", max_length=tokenizer.model_max_length)
            return output
    else:
        assert(False)

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        remove_columns=["text"],
        load_from_cache_file=True,
        desc="Running tokenizer on every text in dataset",
        keep_in_memory=True
    )

    tokenized_datasets.save_to_disk(dataset_path)