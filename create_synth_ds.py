import torch
import datasets
from transformers import AutoTokenizer

from src.util import load_model, CACHE_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_from_ultrachat_prompts(
    model_name: str,
    seq_len: int = 512,
    max_samples: int | None = None,
    bsz: int = 32,
) -> datasets.DatasetDict:
    """Generate synthetic responses from the *first* UltraChat user prompt.

    Returns a DatasetDict with `targets` (int64 ids, left-padded to `seq_len`)
    and `loss_mask` (bool, 1 → generated token) suitable for SFT.
    """
    # --------------------------------------------------------------------- #
    # 1.  Model & tokenizer ------------------------------------------------ #
    # --------------------------------------------------------------------- #

    model, model_family = load_model(model_name, float16=True)
    model = model.to(device).eval()  # <- ensure eval mode

    if model_family == "llama":
        if "vicuna" in model_name:
            tok = AutoTokenizer.from_pretrained(
                "lmsys/vicuna-13b-v1.3",
                cache_dir=CACHE_PATH,
                padding_side="right",
            )
            tok.chat_template = 'A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user\'s input.  {%- for message in messages %}{%- if message.role == "user" %}USER: {{ message.content }}\n{%- elif message.role == "assistant" %}ASSISTANT:{% generation %}{{ message.content }}{% endgeneration %}\n{%- endif %}{%- endfor %}{%- if add_generation_prompt %}ASSISTANT:{% generation %}{% endgeneration %} {%- endif %}'
        else:
            tok = AutoTokenizer.from_pretrained(
                "meta-llama/llama-3.2-1B-instruct",
                cache_dir=CACHE_PATH,
                use_fast=True,
            )
            tok.pad_token = tok.eos_token
    elif model_family == "qwen":
        tok = AutoTokenizer.from_pretrained(
            "qwen/qwen3-0.6b",
            cache_dir=CACHE_PATH,
            use_fast=True,
        )
        tok.pad_token = tok.eos_token
    else:
        raise ValueError(f"Unsupported model family: {model_family}")

    tok.truncation_side = "left"
    tok.padding_side = "left"
    # --------------------------------------------------------------------- #
    # 2.  Load dataset ----------------------------------------------------- #
    # --------------------------------------------------------------------- #
    raw = datasets.load_dataset("HuggingFaceH4/ultrachat_200k", cache_dir=CACHE_PATH)
    splits = dict(
        train=raw["train_sft"], validation=raw["test_sft"].select(range(5000))
    )

    # --------------------------------------------------------------------- #
    # 3.  Helper to process one split ------------------------------------- #
    # --------------------------------------------------------------------- #
    def process_split(data: datasets.Dataset, split_name: str):
        if max_samples is not None:
            data = data.select(range(min(max_samples, len(data))))
        print(f"↪  {split_name}: {len(data)} samples")

        def _generate(batch):
            # ---------- a) extract first user messages ------------------- #
            first_prompts = []
            for convo in batch["messages"]:
                first_user = next(
                    (m["content"] for m in convo if m["role"] == "user"), "Hello"
                )
                first_prompts.append([{"role": "user", "content": first_user}])

            # ---------- b) tokenizer ------------------------------------- #
            prompts = tok.apply_chat_template(
                first_prompts,
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                truncation=True,
                enable_thinking=False,
                max_length=seq_len // 2,  # always leave at least 1 token for gen
                return_tensors="pt",
                return_dict=True,
                tokenizer_kwargs={
                    "return_attention_mask": True,
                },
            ).to(device)

            # per-sample prompt lengths (non-pad count)
            prompt_lens = prompts["attention_mask"].sum(dim=1)
            max_prompt_len = int(prompt_lens.max())

            max_new = seq_len - max_prompt_len

            # ---------- c) sampling -------------------------------------- #
            with torch.no_grad():
                generated: torch.LongTensor = model.generate(
                    **prompts,
                    do_sample=True,
                    max_new_tokens=max_new,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                    temperature=1.0,
                )  # type: ignore

            # ---------- d) pad/trim & build outputs ---------------------- #
            B = generated.size(0)
            out = torch.full((B, seq_len), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((B, seq_len), dtype=torch.bool)
            pl_masked = prompts["input_ids"].size(1)
            gen_len = generated.size(1) - pl_masked

            for i in range(B):
                pl = int(prompt_lens[i])
                out[i, :pl] = prompts["input_ids"][i, -pl:]
                out[i, pl : pl + gen_len] = generated[i, pl_masked:]
                mask[i, pl : pl + gen_len] = (
                    generated[i, pl_masked:] != tok.pad_token_id
                )

            return dict(
                targets=out,
                loss_mask=mask,
            )

        return data.map(
            _generate,
            batched=True,
            batch_size=bsz,
            desc=f"Generating {split_name}",
        )

    # --------------------------------------------------------------------- #
    # 4.  Process both splits & return ------------------------------------ #
    # --------------------------------------------------------------------- #
    return datasets.DatasetDict(
        {name: process_split(ds, name) for name, ds in splits.items()}
    )


def generate_from_sharegpt_prompt(
    model_name: str,
    seq_len: int = 512,
    max_samples: int | None = None,
    bsz: int = 32,
) -> datasets.DatasetDict:
    """Generate synthetic responses from the *first* UltraChat user prompt.

    Returns a DatasetDict with `targets` (int64 ids, left-padded to `seq_len`)
    and `loss_mask` (bool, 1 → generated token) suitable for SFT.
    """
    # --------------------------------------------------------------------- #
    # 1.  Model & tokenizer ------------------------------------------------ #
    # --------------------------------------------------------------------- #

    model, model_family = load_model(model_name, float16=True)
    model = model.to(device).eval()  # <- ensure eval mode
    model = model.to(torch.bfloat16)

    if model_family == "llama":
        if "vicuna" in model_name:
            tok = AutoTokenizer.from_pretrained(
                "lmsys/vicuna-13b-v1.3",
                cache_dir=CACHE_PATH,
                padding_side="right",
            )
            tok.chat_template = 'A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user\'s input.  {%- for message in messages %}{%- if message.role == "user" %}USER: {{ message.content }}\n{%- elif message.role == "assistant" %}ASSISTANT:{% generation %}{{ message.content }}{% endgeneration %}\n{%- endif %}{%- endfor %}{%- if add_generation_prompt %}ASSISTANT:{% generation %}{% endgeneration %} {%- endif %}'
        else:
            tok = AutoTokenizer.from_pretrained(
                "meta-llama/llama-3.2-1B-instruct",
                cache_dir=CACHE_PATH,
                use_fast=True,
            )
            tok.pad_token = tok.eos_token
    elif model_family == "qwen":
        tok = AutoTokenizer.from_pretrained(
            "qwen/qwen3-0.6b",
            cache_dir=CACHE_PATH,
            use_fast=True,
        )
        tok.pad_token = tok.eos_token
    else:
        raise ValueError(f"Unsupported model family: {model_family}")

    tok.truncation_side = "left"
    tok.padding_side = "left"
    # --------------------------------------------------------------------- #
    # 2.  Load dataset ----------------------------------------------------- #
    # --------------------------------------------------------------------- #
    raw = datasets.load_dataset(
        "Aeala/ShareGPT_Vicuna_unfiltered", cache_dir=CACHE_PATH
    )
    raw = raw.filter(lambda x: x["conversations"][0]["from"] == "human")
    raw = raw.map(lambda x: {"prompt": x["conversations"][0]["value"]})
    splits = raw["train"].train_test_split(test_size=0.03)
    splits = {"train": splits["train"], "validation": splits["test"]}

    # --------------------------------------------------------------------- #
    # 3.  Helper to process one split ------------------------------------- #
    # --------------------------------------------------------------------- #
    def process_split(data: datasets.Dataset, split_name: str):
        if max_samples is not None:
            data = data.select(range(min(max_samples, len(data))))
        print(f"↪  {split_name}: {len(data)} samples")

        def _generate(batch):
            # ---------- a) extract first user messages ------------------- #
            first_prompts = []
            for p in batch["prompt"]:
                first_prompts.append([{"role": "user", "content": p}])

            # ---------- b) tokenizer ------------------------------------- #
            prompts = tok.apply_chat_template(
                first_prompts,
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                truncation=True,
                enable_thinking=False,
                max_length=seq_len // 2,  # always leave at least 1 token for gen
                return_tensors="pt",
                return_dict=True,
                tokenizer_kwargs={
                    "return_attention_mask": True,
                },
            ).to(device)

            # per-sample prompt lengths (non-pad count)
            prompt_lens = prompts["attention_mask"].sum(dim=1)
            max_prompt_len = int(prompt_lens.max())

            max_new = seq_len - max_prompt_len

            # ---------- c) sampling -------------------------------------- #
            with torch.no_grad():
                generated: torch.LongTensor = model.generate(
                    **prompts,
                    do_sample=True,
                    max_new_tokens=max_new,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                    temperature=1.0,
                )  # type: ignore

            # ---------- d) pad/trim & build outputs ---------------------- #
            B = generated.size(0)
            out = torch.full((B, seq_len), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((B, seq_len), dtype=torch.bool)
            pl_masked = prompts["input_ids"].size(1)
            gen_len = generated.size(1) - pl_masked

            for i in range(B):
                pl = int(prompt_lens[i])
                out[i, :pl] = prompts["input_ids"][i, -pl:]
                out[i, pl : pl + gen_len] = generated[i, pl_masked:]
                mask[i, pl : pl + gen_len] = (
                    generated[i, pl_masked:] != tok.pad_token_id
                )

            return dict(
                targets=out,
                loss_mask=mask,
            )

        return data.map(
            _generate,
            batched=True,
            batch_size=bsz,
            desc=f"Generating {split_name}",
        )

    # --------------------------------------------------------------------- #
    # 4.  Process both splits & return ------------------------------------ #
    # --------------------------------------------------------------------- #
    return datasets.DatasetDict(
        {name: process_split(ds, name) for name, ds in splits.items()}
    )


if __name__ == "__main__":
    import argparse

    torch.set_grad_enabled(False)
    torch.set_autocast_enabled("cuda", True)
    torch.set_autocast_dtype("cuda", torch.bfloat16)
    parser = argparse.ArgumentParser(description="Create a synthetic dataset")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/llama-3.1-8b-instruct",
        help="Model to use for generation",
    )
    parser.add_argument(
        "--target_len", type=int, default=32, help="Target length for generation"
    )
    parser.add_argument("--bsz", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--max_train_samples", type=int, default=None, help="Max training samples"
    )
    parser.add_argument(
        "--use_ultrachat_prompts",
        action="store_true",
        help="Generate responses from ultrachat first prompts instead of using existing dataset",
    )
    parser.add_argument(
        "--use_sharegpt_prompts",
        action="store_true",
        help="Generate responses from sharegpt first prompts instead of using existing dataset",
    )

    args = parser.parse_args()

    if args.use_ultrachat_prompts:
        # Generate synthetic dataset from ultrachat prompts
        print("Generating responses from ultrachat first prompts...")
        synthetic_ds = generate_from_ultrachat_prompts(
            model_name=args.model_name,
            seq_len=args.target_len,
            max_samples=args.max_train_samples,
            bsz=args.bsz,
        )

        # Save the dataset
        model_short = args.model_name.split("/")[-1]
        output_path = f"./data/synthetic/{model_short}-ultrachat-prompts"
        synthetic_ds.save_to_disk(output_path)
        print(f"Saved synthetic ultrachat dataset to {output_path}")
    elif args.use_sharegpt_prompts:
        # Generate synthetic dataset from sharegpt prompts
        print("Generating responses from sharegpt first prompts...")
        synthetic_ds = generate_from_sharegpt_prompt(
            model_name=args.model_name,
            seq_len=args.target_len,
            max_samples=args.max_train_samples,
            bsz=args.bsz,
        )

        # Save the dataset
        model_short = args.model_name.split("/")[-1]
        output_path = f"./data/synthetic/{model_short}-sharegpt-prompts"
        synthetic_ds.save_to_disk(output_path)
        print(f"Saved synthetic sharegpt dataset to {output_path}")

    else:
        raise ValueError("Must use ShareGPT or UltraChat")
