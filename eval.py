import gc
import json
import random
from datasets import load_dataset

from huggingface_hub import hf_hub_download
from main import TrainingModule
import torch
import argparse

torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)
torch.set_autocast_enabled("cuda", enabled=True)
torch.set_autocast_dtype("cuda", torch.bfloat16)


configs = {
    # Dict instantiates new class
    # "qwen3-8b/no-ft": {
    #     "verifier": "qwen/qwen3-8b",
    #     "drafter": "qwen/qwen3-0.6b",
    #     "method": "independent-drafter",
    # },
    # String starting with './' to load checkpoint at local path
    # "qwen3-8b/no_guide": "./ckpts/qwen8b-noguide.ckpt",
    # String to download from hf
    # "qwen3-8b/no_guide": "peerrh/qwen3-8b-steered",
}


parser = argparse.ArgumentParser()
parser.add_argument(
    "--pattern", type=str, default="", help="Pattern to filter models for evaluation"
)
parser.add_argument(
    "--out_file",
    type=str,
    default="results/eval_result.json",
    help="Pattern to filter models for evaluation",
)
args = parser.parse_args()

if args.pattern:
    configs = {k: v for k, v in configs.items() if args.pattern in k}

DATASETS = ["ultrachat", "humaneval", "xsum", "alpaca", "gsm8k"]

BSZ = 12
TOTAL_SAMPLES = 96
assert TOTAL_SAMPLES % BSZ == 0
TGT_LEN = 128
SEEDS = [0, 1, 2]


def get_prompts(dataset):
    if dataset == "humaneval":
        data = load_dataset("openai_humaneval", split="test")
        return data["prompt"][:TOTAL_SAMPLES]
    elif dataset == "gsm8k":
        data = load_dataset("gsm8k", "main", split="test")
        return data["question"][:TOTAL_SAMPLES]
    elif dataset == "ultrachat":
        data = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
        return data["prompt"][:TOTAL_SAMPLES]
    elif dataset == "alpaca":
        with open("data/eval/alpaca.json", "r") as f:
            data = json.load(f)
            random.Random(1).shuffle(data)
            print([item["instruction"] for item in data[:TOTAL_SAMPLES]])
            return [item["instruction"] for item in data[:TOTAL_SAMPLES]]
    elif dataset == "xsum":
        data = load_dataset("xsum", split="validation")
        return [
            f"Summarize the following document: {doc}"
            for doc in data["document"][:TOTAL_SAMPLES]
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def load_from_local_ckpt(f):
    st = torch.load(f, map_location=torch.device("cuda"))
    mod = TrainingModule(
        **{k: v for k, v in st["hyper_parameters"].items() if k != "_instantiator"}
    )
    mod.n_spec_dec_batches = 100
    mod.d_base.to(torch.bfloat16)
    mod.d_base.load_state_dict(
        {
            k.replace("d_base.", ""): v
            for k, v in st["state_dict"].items()
            if k.startswith("d_base")
        }
    )
    if hasattr(mod, "latent_mod_prep"):
        mod.latent_mod_prep.to(torch.bfloat16)
        mod.latent_mod_prep.load_state_dict(
            {
                k.replace("latent_mod_prep.", ""): v
                for k, v in st["state_dict"].items()
                if k.startswith("latent_mod_prep")
            }
        )
        mod.guidance_embd_layer.load_state_dict(
            {
                k.replace("guidance_embd_layer.", ""): v
                for k, v in st["state_dict"].items()
                if k.startswith("guidance_embd_layer")
            }
        )
        mod.guidance_embd_layer.to(torch.bfloat16)

    return mod


def load_from_hf(f):
    config = json.load(open(hf_hub_download(f, "config.json")))
    st = torch.load(hf_hub_download(f, "pytorch_model.bin"))
    mod = TrainingModule(**{k: v for k, v in config.items() if k != "_instantiator"})
    mod.n_spec_dec_batches = 100
    mod.d_base.to(torch.bfloat16)
    mod.d_base.load_state_dict(
        {k.replace("d_base.", ""): v for k, v in st.items() if k.startswith("d_base")}
    )
    if hasattr(mod, "latent_mod_prep"):
        mod.latent_mod_prep.to(torch.bfloat16)
        mod.latent_mod_prep.load_state_dict(
            {
                k.replace("latent_mod_prep.", ""): v
                for k, v in st.items()
                if k.startswith("latent_mod_prep")
            }
        )
        mod.guidance_embd_layer.load_state_dict(
            {
                k.replace("guidance_embd_layer.", ""): v
                for k, v in st.items()
                if k.startswith("guidance_embd_layer")
            }
        )
        mod.guidance_embd_layer.to(torch.bfloat16)

    return mod


def load_from_ckpt(f: str):
    if f.startswith("./"):
        return load_from_local_ckpt(f)
    else:
        return load_from_hf(f)


final_out = {}

torch.set_grad_enabled(False)

for name, c in configs.items():
    print("-" * 20)
    print(f"Testing {name}...")
    print("-" * 20)
    if isinstance(c, str):
        mod = load_from_ckpt(c)
    else:
        mod = TrainingModule(**c)
    mod.eval()
    mod.to("cuda")
    if hasattr(mod, "d_base"):
        mod.d_base.to(torch.bfloat16)

    # mod.configure_model()
    for g in [True, False]:
        mod.greedy_sample = g
        print(f"Greedy Sampling: {g}")
        for dataset in DATASETS:
            print(f"Dataset: {dataset}")
            this_name = f"{name}-{'greedy' if g else 'sample'}-{dataset}"
            out = []
            total_na_avg = 0.0
            total_throughput = 0.0
            total_bucket = {}
            total_bucket_count = {}
            prompts = get_prompts(dataset)
            this_seeds = SEEDS[:1] if g else SEEDS
            for seed in this_seeds:
                torch.manual_seed(seed)
                for i in range(0, len(prompts), BSZ):
                    try:
                        input_ids, attention_mask = mod.prep_for_gen(
                            prompts[i : i + BSZ]
                        )
                        sampled, extra = mod.generate(
                            input_ids, attention_mask, max_new_tokens=TGT_LEN
                        )
                        na = extra["n_accepted"].cpu().numpy()
                        throughput = (na + 1) / (extra["time_per_block"] / BSZ)
                        for i in range(BSZ):
                            out.append(
                                {
                                    "text": mod.tok.decode(
                                        sampled[
                                            i, extra["attention_mask"][i].to(torch.bool)
                                        ]
                                        .cpu()
                                        .numpy()
                                    ),
                                    "n_accepted": str(na[i]),
                                    "throughput": str(throughput[i]),
                                }
                            )
                        total_na_avg += na.sum()
                        total_throughput += throughput.sum()
                        for k, v in extra.items():
                            if k.startswith("bucket_"):
                                if k not in total_bucket:
                                    total_bucket[k] = 0
                                    total_bucket_count[k] = 0
                                v_val = (
                                    v.cpu().numpy()
                                    if isinstance(v, torch.Tensor)
                                    else v
                                )
                                if v_val > 0:
                                    total_bucket[k] += v_val
                                    total_bucket_count[k] += 1
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print("OOM error, skipping...")
                            # Clear memory after OOM
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            gc.collect()
                        else:
                            raise e
                total_na_avg /= len(prompts)
                total_throughput /= len(prompts)
                for k, v in total_bucket.items():
                    total_bucket[k] /= total_bucket_count[k] + 1e-6
                final_out[this_name + f"_seed={seed}"] = {
                    "outputs": out,
                    "avg_n_accepted": str(total_na_avg),
                    "avg_throughput": str(total_throughput),
                    "bucket_info": total_bucket,
                }
    with open(args.out_file, "w") as f:
        json.dump(final_out, f)


print("Final Outputs:")
print(json.dumps(final_out, indent=2))
