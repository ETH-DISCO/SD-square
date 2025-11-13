"""
Upload Lightning .ckpt files to Hugging Face with only parameters + config.

- For each entry in NAME→CKPT_PATH, creates/updates hf://<namespace>/<name>
- Saves:
    - pytorch_model.bin
    - config.json       (from 'hyper_parameters' or 'hparams' inside ckpt if present)
    - README.md
- Strips Lightning metadata; only uploads weights + config.

Usage:
  export HF_TOKEN=hf_***
  python upload_lightning_ckpts.py --namespace your-username --private
"""

from __future__ import annotations
import argparse, json, os, tempfile, shutil
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
from huggingface_hub import HfApi, create_repo, upload_folder

NAME_TO_CKPT: Dict[str, str] = {
    # "llama-3.1-8b-steered": "final/llama-guide_post_ft.ckpt",
}


def load_ckpt_minimal(
    ckpt_path: Path,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Returns (state_dict_only, config_dict)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Lightning-style
    state_dict = ckpt.get("state_dict", {})
    config = ckpt.get("hyper_parameters", ckpt.get("hparams", {})) or {}

    return state_dict, config


def write_model_files(
    tmp: Path,
    state_dict: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    repo_id: str,
):
    # config.json
    (tmp / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    # weights
    torch.save(state_dict, tmp / "pytorch_model.bin")
    weight_fname = "pytorch_model.bin"

    # minimal model card
    card = f"""---
library_name: pytorch
tags:
- pytorch
---

# {repo_id}

Weights converted from a PyTorch Lightning checkpoint.
Only the raw parameters (`{weight_fname}`) and a lightweight `config.json` are included.
"""
    (tmp / "README.md").write_text(card)


def ensure_repo(api: HfApi, repo_id: str, private: bool):
    create_repo(repo_id, private=private, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--namespace", required=True, help="HF namespace (username or org)"
    )
    parser.add_argument("--private", action="store_true", help="create private repos")
    parser.add_argument(
        "--commit-message", default="Upload weights + config (from Lightning .ckpt)"
    )
    parser.add_argument(
        "--allow-empty-config",
        action="store_true",
        help="allow uploading with empty config.json if no hyper_parameters/hparams found",
    )
    args = parser.parse_args()

    if not NAME_TO_CKPT:
        raise SystemExit("Please fill NAME_TO_CKPT at the top of the script.")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit("Set your Hugging Face token in HF_TOKEN.")

    api = HfApi(token=token)

    for name, ckpt_path in NAME_TO_CKPT.items():
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_file():
            print(f"[skip] {name}: missing file {ckpt_path}")
            continue

        repo_id = f"{args.namespace}/{name}"
        print(f"Processing {name} -> {repo_id}")

        state_dict, config = load_ckpt_minimal(ckpt_path)

        if not state_dict:
            print(f"[skip] {name}: no tensors found in checkpoint")
            continue

        if not config and not args.allow_empty_config:
            print(
                f"[warn] {name}: no config found. Re-run with --allow-empty-config to proceed."
            )
            continue

        tmpdir = Path(tempfile.mkdtemp(prefix=f"hf_{name}_"))
        try:
            write_model_files(tmpdir, state_dict, config, repo_id)
            ensure_repo(api, repo_id, private=args.private)
            upload_folder(
                repo_id=repo_id,
                folder_path=str(tmpdir),
                commit_message=args.commit_message,
                allow_patterns=[
                    "pytorch_model.bin",
                    "config.json",
                    "README.md",
                ],
            )
            print(f"[ok] Uploaded {repo_id}")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
