from itertools import chain
import torch
import os

from src.models.llama import LlamaForCausalLM
from src.models.qwen import Qwen3ForCausalLM


def warn_os_get(var, default):
    env = os.environ.get("TMP_PATH")
    if not env:
        print(f"WARN: ${var} not set, using '{default}'")
        return default
    return env


TMP_PATH = warn_os_get("TMP_PATH", default="./tmp")
OUT_PATH = warn_os_get("OUT_PATH", default="./out")
CACHE_PATH = warn_os_get("CACHE_PATH", default="./tmp")


def load_model(model_name: str, float16=False):
    if "llama" in model_name or "vicuna" in model_name:
        base_class = LlamaForCausalLM
        model_family = "llama"
    elif "qwen" in model_name:
        base_class = Qwen3ForCausalLM
        model_family = "qwen"
    else:
        raise ValueError(
            f"Unknown model family for model: {model_name}. Expected 'qwen', or 'llama'."
        )
    model = base_class.from_pretrained(
        model_name,
        cache_dir=CACHE_PATH,
        torch_dtype=torch.float16 if float16 else torch.float32,
    )
    return model, model_family


def setup_optim(
    *params,
    lr_start: float = 1e-4,
    lr_end: float = 1e-5,
    warmup_steps: int = 100,
    estimated_stepping_batches: int | float = 1000,
):
    optimizer = torch.optim.AdamW(
        chain(*params),
        lr=lr_start,
    )
    scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        [
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-1,
                end_factor=1,
                total_iters=warmup_steps,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(estimated_stepping_batches) - warmup_steps,
                eta_min=lr_end,
            ),
        ],
        optimizer,
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
        },
    }


def rejection_sampling(
    drafted: torch.Tensor,
    model: torch.Tensor,
    sampled_idx: torch.Tensor,
):
    """
    Args:
        drafted: [B, ND, V]
        model: [B, ND + 1, V]
        sampled_idx: [B, ND]
    Returns:
        number_accepted: [B] int index of the first rejected token (ND if all accepted)
        next_dist: [B, V] the next token distribution
    """
    B, ND, V = drafted.shape
    q_sample = drafted.gather(-1, sampled_idx[:, :, None])[:, :, 0]
    p_sample = model[:, :-1].gather(-1, sampled_idx[:, :, None])[:, :, 0]
    rejected = (p_sample / q_sample) < torch.rand(B, ND, device=drafted.device)
    has_rejection, NA = torch.max(rejected, dim=-1)
    NA[has_rejection.logical_not()] = ND
    next_dist = model[torch.arange(B), NA, :]
    next_dist[has_rejection, :] -= drafted[has_rejection, NA[has_rejection], :]
    next_dist = next_dist.clamp_min_(min=0)
    return NA, next_dist


def print_speculative_sampling_result(
    tokenizer, drafted, n_accepted: int, next_tok: int
):
    """
    Print the results of speculative sampling with color coding.

    Args:
        tokenizer: Tokenizer to decode token IDs
        drafted: [NG] int - drafted token IDs
        n_accepted: int - number of accepted tokens
        next_tok: int - the next token ID
    """
    # ANSI color codes
    GREEN = "\033[92m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    STRIKETHROUGH = "\033[9m"

    result = ""

    # Accepted tokens (first n_accepted) in green
    if n_accepted > 0:
        accepted_tokens = drafted[:n_accepted]
        for token_id in accepted_tokens:
            token_text = tokenizer.decode([token_id]).replace("\n", "\\n")
            result += f"{GREEN}{token_text}{RESET}"

    # Declined tokens (remaining) in red with strikethrough
    if n_accepted < len(drafted):
        declined_tokens = drafted[n_accepted:]
        for token_id in declined_tokens:
            token_text = tokenizer.decode([token_id]).replace("\n", "\\n")
            result += f"{RED}{STRIKETHROUGH}{token_text}{RESET}"

    # Next token in blue
    next_token_text = tokenizer.decode([next_tok]).replace("\n", "\\n")
    result += f"{BLUE}{next_token_text}{RESET}"

    print(result)
