from typing import Literal
import datasets
from datasets import DatasetDict
from transformers import AutoTokenizer  # type: ignore
import lightning as L
import torch
import os

from src.util import CACHE_PATH

type DatasetName = Literal[
    "xsum", "en_de", "gsm8k", "mathinstruct", "lm1b", "sharegpt", "ultrachat"
]


class DataLoader(L.LightningDataModule):
    def __init__(
        self,
        path: str,
        bsz: int = 32,
        n_train: int | None = None,
        n_val: int | None = None,
    ):
        super().__init__()
        self.path = path
        self.bsz = bsz
        self.n_train = n_train
        self.n_val = n_val

    def prepare_data(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Path {self.path} does not exist.")

    def setup(self, stage: str):
        self.ds = DatasetDict.load_from_disk(self.path).with_format("torch")
        to_keep = ["targets"]
        if "enc_input_ids" in self.ds["train"].features:
            to_keep.append("enc_input_ids")
            to_keep.append("enc_attention_mask")
        # Keep loss_mask and prompt_input_ids for chat datasets
        if "loss_mask" in self.ds["train"].features:
            to_keep.append("loss_mask")
        if "prompt" in self.ds["train"].features:
            to_keep.append("prompt")
        self.ds = self.ds.remove_columns(
            [c for c in self.ds["train"].column_names if c not in to_keep]
        )
        if self.n_train is not None:
            self.ds["train"] = self.ds["train"].select(range(self.n_train))
        if self.n_val is not None:
            self.ds["validation"] = self.ds["validation"].select(range(self.n_val))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds["train"],  # type: ignore
            batch_size=self.bsz,
            num_workers=4,
            shuffle=False,
            drop_last=True,  # type: ignore
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds["validation"],  # type: ignore
            batch_size=self.bsz,
            num_workers=4,
            shuffle=False,
            drop_last=True,  # type: ignore
        )


def get_dataset(
    dataset_name: DatasetName,
    model_name: str = "t5",
    max_input_length: int = 1024,
    max_target_length: int = 32,
    strict: bool = False,
) -> DatasetDict:
    ## LOAD TOKENIZER
    match model_name:
        case "t5":
            tok = AutoTokenizer.from_pretrained(
                "google/t5-v1_1-small",
                cache_dir=CACHE_PATH,
                use_fast=True,
                padding_side="right",
            )
        case "opt":
            tok = AutoTokenizer.from_pretrained(
                "facebook/opt-125m",
                cache_dir=CACHE_PATH,
                use_fast=True,
                padding_side="right",
            )
        case "vicuna":
            tok = AutoTokenizer.from_pretrained(
                "lmsys/vicuna-7b-v1.3",
                cache_dir=CACHE_PATH,
                padding_side="right",
            )
            tok.truncation_side = "left"
            tok.chat_template = '{%- for message in messages %} {%- if message.role == "user" %} USER: {{ message.content }} \\n {%- elif message.role == "assistant" %} ASSISTANT: {% generation %}{{ message.content }}{% endgeneration %} \\n {%- endif %} {%- endfor %} {%- if add_generation_prompt %} ASSISTANT: {% generation %}{% endgeneration %} {%- endif %}'
        case "llama":
            tok = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.1-8B-Instruct",
                cache_dir=CACHE_PATH,
                use_fast=True,
                padding_side="right",
            )
            tok.pad_token = tok.eos_token  # Llama uses EOS as pad token
            tok.truncation_side = "left"
            tok.chat_template = "{% for message in messages %}\n  {% if (message['role'] != 'assistant') %}\n {{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n' + message['content'] + '<|eot_id|>' + '\n'}}\n {% elif (message['role'] == 'assistant')%}\n {{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n'}}\n {% generation %}\n {{message['content'] + '<|eot_id|>'}}\n {% endgeneration %}\n {{'\n'}}\n {% endif %}\n {% endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n' }}\n{%- endif %}"
        case "qwen":
            tok = AutoTokenizer.from_pretrained(
                "qwen/qwen3-0.6b",
                cache_dir=CACHE_PATH,
                use_fast=True,
                padding_side="right",
            )
            tok.pad_token = tok.eos_token  # Llama uses EOS as pad token
            tok.chat_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set content = message.content %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is defined and message.reasoning_content is not none %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in message.content %}\n                {%- set content = message.content.split('</think>')[-1].lstrip('\\n') %}\n                {%- set reasoning_content = message.content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {{- '<|im_start|>' + message.role }}\n        {% generation %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if loop.last or (not loop.last and reasoning_content) %}\n                {{- '<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- content }}\n            {%- endif %}\n        {%- else %}\n            {{- content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>' }}\n        {% endgeneration %}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"
        case _:
            raise ValueError("Invalid model name")
    path = f"./data/orig/model={model_name}_ds_name={dataset_name}_tgt_len={max_target_length}"
    if os.path.exists(path):
        print("found existing version")
        ds = DatasetDict.load_from_disk(path)
        if strict:
            # Check that targets has no pad_token
            ds = ds.filter(lambda x: x["targets"][-1] != tok.pad_token_id)
            print("Filtered Train Size:", len(ds["train"]))
        return ds

    ## LOAD DATASET
    match dataset_name:
        case "en_de":
            raw_dataset: DatasetDict = datasets.load_dataset(
                "wmt19", "de-en", cache_dir=CACHE_PATH
            )  # type: ignore
            raw_dataset = DatasetDict(
                {
                    "train": raw_dataset["train"].select(range(5_000_000)),
                    "validation": raw_dataset["validation"],
                }
            )
        case "cnn_dm":
            raw_dataset: DatasetDict = datasets.load_dataset(
                "cnn_dailymail", "3.0.0", cache_dir=CACHE_PATH
            )  # type: ignore
        case "gsm8k":
            raw_dataset: DatasetDict = datasets.load_dataset(
                "openai/gsm8k", "main", cache_dir=CACHE_PATH
            )  # type: ignore
            raw_dataset = DatasetDict(
                {
                    "train": raw_dataset["train"],
                    "validation": raw_dataset["test"],
                }
            )
        case "mathinstruct":
            raw_dataset: DatasetDict = datasets.load_dataset(
                "TIGER-Lab/MathInstruct", cache_dir=CACHE_PATH
            )  # type: ignore
            # MathInstruct typically has train split, create validation from train if needed
            train_test = raw_dataset["train"].train_test_split(test_size=0.1, seed=42)
            raw_dataset = DatasetDict(
                {
                    "train": train_test["train"],
                    "validation": train_test["test"],
                }
            )
            raw_dataset = raw_dataset.filter(lambda x: "CoT" in x["source"])
        case "lm1b":
            raw_dataset: DatasetDict = datasets.load_dataset(
                "lm1b", cache_dir=CACHE_PATH, trust_remote_code=True
            )  # type: ignore
            raw_dataset = DatasetDict(
                {
                    "train": raw_dataset["train"].filter(
                        lambda x: len(x["text"]) > 256,
                    ),
                    "validation": raw_dataset["test"].filter(
                        lambda x: len(x["text"]) > 256,
                    ),
                }
            )
        case "sharegpt":
            raw_dataset: DatasetDict = datasets.load_dataset(
                "Aeala/ShareGPT_Vicuna_unfiltered", cache_dir=CACHE_PATH
            )  # type: ignore
            # Use only 'train' split, create validation from it
            train_test = raw_dataset["train"].train_test_split(test_size=0.03, seed=42)
            raw_dataset = DatasetDict(
                {
                    "train": train_test["train"],
                    "validation": train_test["test"],
                }
            )
        case "ultrachat":
            raw_dataset: DatasetDict = datasets.load_dataset(
                "HuggingFaceH4/ultrachat_200k", cache_dir=CACHE_PATH
            )  # type: ignore
            # Use train_sft and test_sft splits
            raw_dataset = DatasetDict(
                {
                    "train": raw_dataset["train_sft"],
                    "validation": raw_dataset["test_sft"],
                }
            )
        case _:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

    def enc_dec_tok(x, enc_texts, dec_texts):
        enc_out = tok(
            enc_texts,
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
        )
        x["enc_input_ids"] = enc_out["input_ids"]
        x["enc_attention_mask"] = enc_out["attention_mask"]

        # Setup the tokenizer for targets
        with tok.as_target_tokenizer():
            x["targets"] = tok(
                dec_texts,
                max_length=max_target_length,
                truncation=True,
                padding="max_length",
            )["input_ids"]
        return x

    match (model_name, dataset_name):
        case ("llama" | "qwen" | "vicuna", "ultrachat"):

            def preproc(x):
                out = tok.apply_chat_template(
                    x["messages"],
                    tokenize=True,
                    return_dict=True,
                    add_generation_prompt=False,
                    return_assistant_tokens_mask=True,
                    padding="max_length",
                    max_length=max_target_length,
                    truncation=True,
                )

                # Tokenize full texts
                x["targets"] = out["input_ids"]
                x["loss_mask"] = out["assistant_masks"]
                return x

        case ("llama", "sharegpt"):

            def preproc(x):
                texts = [
                    "<|begin_of_text|>\n"
                    + "\n".join(
                        f"<|start_header_id|>{'user' if m['from'] == 'human' else 'assistant'}"
                        f"<|end_header_id|>\n{m['value']}\n<|eot_id|>"
                        for m in conv
                    )
                    for conv in x["conversations"]
                ]
                x["targets"] = tok(
                    texts,
                    max_length=max_target_length,
                    truncation=True,
                    padding="max_length",
                )["input_ids"]
                return x

        case ("qwen", "sharegpt"):

            def preproc(x):
                texts = [
                    "\n".join(
                        f"<|im_start|>{'user' if m['from'] == 'human' else 'assistant'}\n"
                        f"{m['value']}\n<|im_end|>"
                        for m in conv
                    )
                    + "\n<|im_start|>assistant\n"
                    for conv in x["conversations"]
                ]
                x["targets"] = tok(
                    texts,
                    max_length=max_target_length,
                    truncation=True,
                    padding="max_length",
                )["input_ids"]
                return x

        case ("opt", "lm1b"):

            def preproc(x):
                x["targets"] = tok(
                    x["text"],
                    max_length=max_target_length,
                    truncation=True,
                    padding="max_length",
                )["input_ids"]
                return x

        case ("t5", "gsm8k"):

            def preproc(x):
                enc_texts = ["Solve the following problem:" + t for t in x["question"]]
                dec_texts = x["answer"]
                return enc_dec_tok(x, enc_texts, dec_texts)

        case ("t5", "mathinstruct"):

            def preproc(x):
                enc_texts = [
                    "Solve the following problem:" + t for t in x["instruction"]
                ]
                dec_texts = x["output"]
                return enc_dec_tok(x, enc_texts, dec_texts)

        case ("t5", "en_de"):

            def preproc(x):
                enc_texts = [
                    "translate English to German: " + t["en"] for t in x["translation"]
                ]
                dec_texts = [ex["de"] for ex in x["translation"]]
                return enc_dec_tok(x, enc_texts, dec_texts)

        case ("t5", "xsum"):

            def preproc(x):
                enc_texts = ["Summarize:" + t for t in x["article"]]
                dec_texts = x["highlights"]
                return enc_dec_tok(x, enc_texts, dec_texts)

        case _:
            raise ValueError("This combination of data is not implemented yet")

    ds = raw_dataset.map(
        preproc,
        batched=True,
        # remove_columns=raw_dataset["train"].column_names,
        load_from_cache_file=False,
    )
    ds.save_to_disk(path)
    print("Original Train Size:", len(ds["train"]))
    if strict:
        # Check that targets has no pad_token
        ds = ds.filter(lambda x: x["targets"][-1] != tok.pad_token_id)
        print("Filtered Train Size:", len(ds["train"]))
    return ds
