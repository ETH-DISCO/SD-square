import copy

import json
from typing import Any, Literal
import time

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import lightning as L
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch import loggers
import wandb  # type: ignore

from src.eagle import Model
from src.models.llama import (
    LlamaBlockMerged,
    LlamaBlockAdvanced,
    LlamaBlockSimple,
)
from src.models.qwen import (
    Qwen3BlockMerged,
    Qwen3BlockSimple,
)
from src.util import (
    OUT_PATH,
    setup_optim,
    load_model,
    rejection_sampling,
    print_speculative_sampling_result,
)
from src.model import PrepLatentDeltas, WithKwargs
from src.data import DataLoader
from transformers import AutoTokenizer, DynamicCache  # type: ignore
from src.eagle import EConfig

PRETTY_PRINT = True


class TrainingModule(L.LightningModule):
    model_family: Literal["llama", "qwen"]
    method: Literal["independent-drafter", "guided-drafter", "eagle", "vanilla"]

    def __init__(
        self,
        ngram: int = 8,
        # OPTIM
        lr_start: float = 3e-4,
        lr_end: float = 3e-5,
        warmup_steps: int = 30,
        # MODEL PARAMS
        method: Literal[
            "independent-drafter",
            "guided-drafter",
            "eagle",
            "vanilla",
        ] = "guided-drafter",
        loss_method: Literal["kl", "tvd", "rkl"] = "tvd",
        verifier: str = "meta-llama/llama-3.1-8b-instruct",
        drafter: str = "meta-llama/llama-3.2-1b-instruct",
        finetune_drafter: Literal["full", "none"] = "full",
        guide_method: Literal["merged", "advanced", "simple", "elmnt_wise"] = "merged",
        v_layer: int | tuple[int, int, int] = 10,
        d_layer: int | list[int] | Literal["all"] = "all",
        # Training Hyper Parameter
        n_spec_dec_batches: int = 4,
        pos_method: Literal["regular", "blocked"] = "regular",
        # spec_dec Parameters
        greedy_sample: bool = False,
        eagle_hf_path: None | str = None,
        benchmark_latency: bool = False,
        load_from_ckpt: str | None = None,
        tgt_len: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.tgt_len = tgt_len
        self.save_hyperparameters()
        self.strict_loading = False
        self.generate_only = False

        # Save Hyperparameters
        self.benchmark_latency = benchmark_latency
        self.method = method
        self.NG = ngram
        self.loss_method = loss_method
        if self.method == "eagle":
            self.loss_method = "tvd"
        self.n_spec_dec_batches = n_spec_dec_batches
        self.pos_method = pos_method
        self.finetune_drafter = finetune_drafter == "full"
        self.greedy_sample = greedy_sample

        self._load_models()
        if self.method == "guided-drafter":
            self.extract_method = "hml"
            self._setup_guidance_extraction()
            self._setup_guidance_application()
        elif self.method == "eagle":
            self._setup_eagle()
            self.extract_method = "pass_through"
            self.hparams["v_layer"] = [2, self.V_N_LAYERS // 2, self.V_N_LAYERS - 3]
            self._setup_guidance_extraction()
        if load_from_ckpt is not None:
            self._load_from_checkpoint(load_from_ckpt)

    def _load_from_checkpoint(self, load_from_ckpt: str):
        st = torch.load(load_from_ckpt, map_location=torch.device("cuda"))
        # mod = TrainingModule(**{k: v for k, v in st['hyper_parameters'].items() if k!= '_instantiator'}, load_from_ckpt=None)
        self.d_base.load_state_dict(
            {
                k.replace("d_base.", ""): v
                for k, v in st["state_dict"].items()
                if k.startswith("d_base")
            }
        )
        if hasattr(self, "latent_mod_prep"):
            self.latent_mod_prep.load_state_dict(
                {
                    k.replace("latent_mod_prep.", ""): v
                    for k, v in st["state_dict"].items()
                    if k.startswith("latent_mod_prep")
                }
            )
            self.guidance_embd_layer.load_state_dict(
                {
                    k.replace("guidance_embd_layer.", ""): v
                    for k, v in st["state_dict"].items()
                    if k.startswith("guidance_embd_layer")
                }
            )

    def _load_models(self):
        self.v_base, self.model_family = load_model(
            self.hparams["verifier"], float16=True
        )
        match self.model_family:
            case "llama":
                if "vicuna" in self.hparams["verifier"]:
                    self.tok = AutoTokenizer.from_pretrained(self.hparams["verifier"])
                    self.tok.chat_template = 'A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user\'s input. {%- for message in messages %}{%- if message.role == "user" %}USER: {{ message.content }}\n{%- elif message.role == "assistant" %}ASSISTANT:{% generation %}{{ message.content }}{% endgeneration %}\n{%- endif %}{%- endfor %}{%- if add_generation_prompt %}ASSISTANT:{% generation %}{% endgeneration %} {%- endif %}'
                    self.tok.truncation_side = "left"
                    self.tok.padding_side = "left"
                    pad_token_key = "pad_token_id"
                else:
                    self.tok = AutoTokenizer.from_pretrained(
                        "meta-llama/llama-3.1-8b-instruct", use_fast=True
                    )
                    self.tok.truncation_side = "left"
                    self.tok.padding_side = "left"
                    self.tok.pad_token = self.tok.eos_token
                    pad_token_key = "eos_token_id"

                h_dim_key = "hidden_size"
                ff_dim_key = "intermediate_size"
                n_layers_key = "num_hidden_layers"
                self.eot_id = 128009
            case "qwen":
                self.tok = AutoTokenizer.from_pretrained("qwen/qwen3-8B", use_fast=True)
                self.tok.truncation_side = "left"
                self.tok.padding_side = "left"
                h_dim_key = "hidden_size"
                ff_dim_key = "intermediate_size"
                n_layers_key = "num_hidden_layers"
                pad_token_key = "eos_token_id"
                self.eot_id = 151645

        self.pad_token_id = getattr(self.v_base.config, pad_token_key)
        if isinstance(self.pad_token_id, list):
            self.pad_token_id = self.pad_token_id[0]

        self.V_N_LAYERS = getattr(self.v_base.config, n_layers_key)
        self.V_H_DIM = getattr(self.v_base.config, h_dim_key)
        if self.method == "guided-drafter" or self.method == "independent-drafter":
            self.d_base, _ = load_model(self.hparams["drafter"])
            self.d_base.train()
            self.D_H_DIM = getattr(self.d_base.config, h_dim_key)
            self.D_FF_DIM = getattr(self.d_base.config, ff_dim_key)
            self.D_N_LAYERS = getattr(self.d_base.config, n_layers_key)
            for i in self.d_base.parameters():
                i.requires_grad = self.finetune_drafter
        for i in self.v_base.parameters():
            i.requires_grad = False

    def _setup_guidance_extraction(self):
        def copy_module(mod):
            return copy.deepcopy(mod.cpu().to(torch.float32))

        v_layer = self.hparams["v_layer"]

        match (self.model_family, self.extract_method):
            case (_, "pass_through"):
                guide_base = WithKwargs(nn.Identity())
            case (_, "hml"):
                guide_base = WithKwargs(nn.Linear(3 * self.V_H_DIM, self.V_H_DIM))
                # Initialize with three identity maps concatenated
                with torch.no_grad():
                    identity = torch.eye(self.V_H_DIM)
                    guide_base.orig.weight.data = torch.cat(
                        [identity, identity, identity], dim=1
                    )
                    guide_base.orig.bias.data.zero_()
            case _:
                raise ValueError(
                    f"Unsupported extract method: {self.hparams['extract_method']}"
                )
        self.guidance_embd_layer = guide_base
        self.guidance_embd_layer.train()  # type: ignore
        for param in self.guidance_embd_layer.parameters():  # type: ignore
            param.requires_grad = True
        if isinstance(v_layer, int):
            self.guidance_embd_layer.in_layer = [v_layer - 1]  # type: ignore
        else:
            self.guidance_embd_layer.in_layer = v_layer  # type: ignore
        self.v_base.get_decoder().guidance_embd_layer = self.guidance_embd_layer  # type: ignore

    def _setup_guidance_application(self):
        ##### Guidance Application
        guide_method = self.hparams["guide_method"]
        d_layer = self.hparams["d_layer"]
        if isinstance(d_layer, int):
            layers = [d_layer]
        elif d_layer == "all":
            layers = range(self.D_N_LAYERS)
        else:
            layers = d_layer
        match guide_method:
            case "merged":
                self.latent_mod_prep = PrepLatentDeltas(
                    self.V_H_DIM, self.D_FF_DIM, len(layers)
                )
            case "simple":
                self.latent_mod_prep = PrepLatentDeltas(
                    self.V_H_DIM, self.D_H_DIM, len(layers)
                )
            case _:
                self.latent_mod_prep = nn.Identity()

        self.latent_mods = []
        for i, layer in enumerate(layers):
            match (self.model_family, guide_method):
                case ("llama", "merged"):
                    latent_mod = LlamaBlockMerged(
                        self.d_base.get_decoder().layers[layer],  # type: ignore
                        layer_idx=i,
                    )
                case ("llama", "advanced"):
                    latent_mod = LlamaBlockAdvanced(
                        self.d_base.get_decoder().layers[layer],  # type: ignore
                        self.d_base.config,  # type: ignore
                        self.V_H_DIM,
                    )
                case ("llama", "simple"):
                    latent_mod = LlamaBlockSimple(
                        self.d_base.get_decoder().layers[layer],  # type: ignore
                        layer_idx=i,
                    )
                case ("qwen", "simple"):
                    latent_mod = Qwen3BlockSimple(
                        self.d_base.get_decoder().layers[layer],  # type: ignore
                        layer_idx=i,
                    )
                case ("qwen", "merged"):
                    latent_mod = Qwen3BlockMerged(
                        self.d_base.get_decoder().layers[layer],  # type: ignore
                        layer_idx=i,
                    )
                case _:
                    raise ValueError(
                        "The selected combination of Guidance and Model is not implemented"
                    )
            self.d_base.get_decoder().layers[layer] = latent_mod  # type: ignore
            self.latent_mods.append(latent_mod)

    def _setup_eagle(self):
        # Currently training eagle is not fully supported
        assert self.hparams["eagle_hf_path"] is not None
        assert self.model_family == "llama" or self.model_family == "qwen"
        self.vocab_size = self.v_base.config.vocab_size
        from huggingface_hub import hf_hub_download

        eagle_config = hf_hub_download(self.hparams["eagle_hf_path"], "config.json")
        eagle_config = json.load(open(eagle_config, "r"))
        self.eagle = Model(EConfig(**eagle_config)).to(torch.float16)
        eagle_sd = torch.load(
            hf_hub_download(self.hparams["eagle_hf_path"], "pytorch_model.bin")
        )
        # self.eagle.embed_tokens = self.v_base.get_decoder().embed_tokens
        eagle_sd["embed_tokens.weight"] = self.v_base.get_decoder().embed_tokens.weight  # type: ignore
        if "vicuna" in self.hparams["verifier"]:
            del eagle_sd["t2d"]
            del eagle_sd["d2t"]
        self.eagle.load_state_dict(eagle_sd, strict=False)
        self.eagle.d2t = self.eagle.d2t + torch.arange(
            self.eagle.d2t.shape[0], device=self.eagle.d2t.device
        )  # type: ignore
        if "vicuna" in self.hparams["verifier"]:
            self.eagle.t2d[:] = 1

    def on_fit_start(self, *args, **kwargs):
        super().on_fit_start(*args, **kwargs)
        for logger in self.loggers:
            if isinstance(logger, loggers.WandbLogger):
                self.wandb = logger
                break
        if hasattr(self, "wandb"):
            if self.method == "guided-drafter":
                self.wandb.watch(self.guidance_embd_layer)  # type: ignore
                self.wandb.watch(self.latent_mod_prep)
                for latent_mod in self.latent_mods:
                    self.wandb.watch(latent_mod)
            if self.method == "eagle":
                self.wandb.watch(self.eagle)

    def training_step(self, batch, batch_idx):
        loss, metric = self.process_batch(
            batch,
            log_extras=batch_idx % 500 == 0,
            compute_tvd=batch_idx % 50 == 0,
        )
        self.log_metric("train", loss, metric)
        return loss

    def validation_step(self, batch, batch_idx):
        if not self.generate_only:
            loss, metric = self.process_batch(
                batch,
                log_extras=batch_idx % 20 == 0,
                compute_tvd=True,
            )
        else:
            loss = torch.tensor(0.0, device=self.device)
            metric = {}
        if batch_idx < self.n_spec_dec_batches:
            input_ids, attention_mask = self.prep_for_gen(batch["prompt"])
            new_tok = self.tgt_len
            _, sd_metrics = self.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=new_tok,
                batch=batch,
            )
            metric["spec_dec_na"] = sd_metrics["n_accepted"].mean().item()
            if batch_idx > 0:
                # Discard compile time
                metric["time_per_block"] = sd_metrics["time_per_block"]
                metric.update(sd_metrics["timing_metrics"])

        self.log_metric("val", loss, metric)
        return loss

    def log_metric(self, stage: str, loss: Tensor, metric: dict[str, Any]):
        self.log(f"{stage}/loss", loss)
        for k, v in metric.items():
            if k.startswith("hidden_delta") or k == "guide_embds":
                if hasattr(self, "wandb"):
                    self.wandb.experiment.log(
                        {
                            f"{stage}/{k}": wandb.Histogram(
                                v.to(torch.float).cpu().numpy()
                            )
                        },
                        commit=False,
                    )
                continue
            self.log(f"{stage}/{k}", v)

    def process_batch(self, batch, log_extras=False, compute_tvd=False):
        if self.method == "vanilla":
            return torch.tensor(0.0, device=self.device), {}
        ### 1. Get Data from Batch
        metrics = {}
        targets = batch["targets"]
        B, S = targets.shape

        # Use provided loss_mask if available (for chat datasets), otherwise create default mask
        if "loss_mask" in batch:
            loss_mask = batch["loss_mask"].float()  # (B, S)
        else:
            loss_mask = (targets != self.pad_token_id).float()  # (B, S)
            loss_mask[:, 0] = 1.0  # Ensure BOS token is not masked

        ### 2. Get Verifier Targets (and Guidance)
        v_out = self.v_base.get_decoder()(
            targets,
            compute_guidance=self.method != "independent-drafter",
            return_dict=True,
        )
        if self.method != "independent-drafter":
            guide_embds = v_out["guide_embd"]
            v_out = v_out["out"]
        else:
            guide_embds = None

        with torch.no_grad():
            v_hidden = v_out.last_hidden_state
            v_logits = self.v_base.lm_head(v_hidden)

        ### 3. Compute Drafted Out
        if self.method == "guided-drafter" or self.method == "independent-drafter":
            if self.method == "guided-drafter":
                if self.pos_method == "regular":
                    offset = torch.randint(1, self.NG + 1, (1,), device=self.device)[0]
                    guide_pos = torch.arange(0, S, device=self.device) - offset
                    guide_pos = guide_pos.clamp_min_(0)
                else:
                    offset = torch.randint(0, self.NG, (1,), device=self.device)[0]
                    guide_pos = (
                        (torch.arange(0, S, device=self.device) - offset - 1)
                        // self.NG
                        * self.NG
                    ) + offset
                    guide_pos = guide_pos.clamp_min_(0)
                guide_embds = self.latent_mod_prep(
                    guide_embds[:, guide_pos]
                ) * loss_mask[:, :, None].roll(-1, dims=1)  # type: ignore

            d_out = self.d_base.get_decoder()(
                targets,
                guidance_embeds=guide_embds,
                output_latent_delta=log_extras,
                return_dict=True,
            )
            if log_extras:
                for i, latent_mod in enumerate(d_out["latent_mods"]):
                    metrics[f"hidden_delta_{i}"] = latent_mod.to(torch.float16).detach()
                d_out = d_out["out"]
            d_logits = self.d_base.lm_head(d_out.last_hidden_state)
        elif self.method == "eagle":
            # ith hidden state + (i+1)th token -> (i+2)th token
            d_hidden = self.eagle(guide_embds[:, :-1], targets[:, 1:])
            d_logits = self.eagle.lm_head(self.eagle.norm(d_hidden))
            v_logits = v_logits[:, 1:]
            loss_mask = loss_mask[:, 1:]
        else:
            raise ValueError("Unsupported Method")

        ### 4. Compute Loss
        if self.loss_method == "tvd" or compute_tvd:
            if self.method == "eagle":
                v_probs = v_logits.flatten(0, 1)[..., self.eagle.t2d].softmax(dim=-1)
                tvd_loss = (
                    (v_probs - d_logits.flatten(0, 1).softmax(dim=-1)).abs().sum(dim=-1)
                )  # + (v_probs[:, torch.logical_not(self.eagle.t2d)]).abs().sum(dim=-1)
            else:
                v_probs = v_logits.flatten(0, 1).softmax(dim=-1)
                tvd_loss = (
                    (v_probs - d_logits.flatten(0, 1).softmax(dim=-1)).abs().sum(dim=-1)
                )

            tvd_loss = (tvd_loss * 0.5 * loss_mask.flatten()).sum() / (
                loss_mask.sum() + 1e-4
            )
            metrics["loss_tvd"] = tvd_loss
            if self.loss_method == "tvd":
                loss = tvd_loss

        if self.loss_method == "kl" or self.loss_method == "rkl":
            targets_log = v_logits.flatten(0, 1).log_softmax(dim=-1)
            if self.loss_method == "rkl":
                kl_div_loss = F.kl_div(
                    targets_log,
                    d_logits.flatten(0, 1).log_softmax(dim=-1),
                    log_target=True,
                    reduction="none",  # Change to 'none' to get per-token loss
                )
            else:
                kl_div_loss = F.kl_div(
                    d_logits.flatten(0, 1).log_softmax(dim=-1),
                    targets_log,
                    log_target=True,
                    reduction="none",  # Change to 'none' to get per-token loss
                )
            # KL divergence returns loss per distribution, sum over vocab dim to get per-token loss
            kl_div_loss = kl_div_loss.sum(dim=-1)
            # Apply mask and compute mean only over non-padding tokens
            kl_div_loss = (kl_div_loss * loss_mask.flatten()).sum() / (
                loss_mask.sum() + 1e-4
            )
            metrics["loss_kl_div"] = kl_div_loss
            loss = kl_div_loss

        return loss, metrics  # type: ignore

    def configure_optimizers(self):  # type: ignore
        params = []
        if self.method == "independent-drafter":
            params.append(self.d_base.parameters())
        elif self.method == "guided-drafter":
            # For advanced guide_method, the parameters are included in the d_based
            if self.finetune_drafter:
                params.append(self.d_base.parameters())
            params.append(self.latent_mod_prep.parameters())
            params.append(self.guidance_embd_layer.parameters())  # type: ignore
        elif self.method == "eagle":
            params.append(self.eagle.parameters())

        return setup_optim(
            *params,
            lr_start=self.hparams["lr_start"],
            lr_end=self.hparams["lr_end"],
            warmup_steps=self.hparams["warmup_steps"],
            estimated_stepping_batches=self.trainer.estimated_stepping_batches,
        )

    def state_dict(self, *args, **kwargs):
        # Exclude v_base and d_base modules from the saved state dict
        state = super().state_dict(*args, **kwargs)
        # Remove any keys belonging to v_base or d_base
        keys_to_remove = [k for k in state.keys() if k.startswith("v_base.")]
        for k in keys_to_remove:
            state.pop(k)
        return state

    # ------------------------------------------------------------------
    #  Speculative-decoding generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 32,
        batch={},
    ):
        """
        Speculative decoding using the compiled step function.
        input_ids: should be left padded
        """
        if self.method == "vanilla":
            return self._generate_vanilla(
                input_ids, attention_mask, max_new_tokens, batch
            )
        B, S = input_ids.shape
        device = input_ids.device
        # encoder context for verifier & drafter
        if PRETTY_PRINT:
            print("----------------- NEW GENERATE ----------------")
            print("Prompt: ", self.tok.decode(input_ids[0]))

        # ------ initial decoder prompt ------
        PAD_FACTOR = 4
        sampled = torch.full(
            (B, S + max_new_tokens * PAD_FACTOR),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        sampled[:, :S] = input_ids
        curr = S - 1
        position_ids = torch.arange(0, S + max_new_tokens * PAD_FACTOR, device=device)[
            None, :
        ].expand(B, -1)
        if attention_mask is not None:
            position_ids = torch.clamp_min_(
                position_ids - S + attention_mask.sum(dim=1)[:, None], 0
            )
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.zeros((B, max_new_tokens * PAD_FACTOR)).to(attention_mask),
                ),
                dim=1,
            )

        ### COMPUTE V_PKV
        v_pkv = DynamicCache()
        v_out = self.v_base.get_decoder()(
            sampled[:, : curr + 1],
            position_ids=position_ids[:, : curr + 1],
            compute_guidance=True,
            past_key_values=v_pkv,
            attention_mask=(
                attention_mask[:, : curr + 1].to(torch.bool)
                if attention_mask is not None
                else None
            ),
            use_cache=True,
        )
        v_pkv = v_out["out"].past_key_values
        v_pkv.crop(curr)
        #### COMPUTE GUIDE
        match self.method:
            case "guided-drafter":
                guide = self.latent_mod_prep(v_out["guide_embd"][:, -1, None])

            case "eagle":
                guide = v_out["guide_embd"][:, -1, None]
                guide_init = v_out["guide_embd"][:, :-2]
            case _:
                guide = None
        ### COMPUTE D_PKV
        match self.method:
            case "guided-drafter" | "independent-drafter":
                d_pkv = DynamicCache()
                d_pkv = self.d_base.get_decoder()(
                    input_ids=sampled[:, :curr],
                    position_ids=position_ids[:, :curr],
                    past_key_values=d_pkv,
                    use_cache=True,
                    attention_mask=(
                        attention_mask[:, :curr] if attention_mask is not None else None
                    ),
                ).past_key_values
            case "eagle":
                # Shift position ids by 1 as we "discard" first input_id
                position_ids = torch.clamp_min_(position_ids - 1, 0)
                d_pkv = DynamicCache()
                _, d_pkv = self.eagle(
                    input_ids=sampled[:, 1:curr],
                    hidden_states=guide_init,
                    position_ids=position_ids[:, 1:curr],
                    past_key_values=d_pkv,
                    use_cache=True,
                    attention_mask=(
                        attention_mask[:, 1:curr]
                        if attention_mask is not None
                        else None
                    ),
                )
        STEPS = torch.zeros(B, dtype=torch.long, device=device)
        NA_COUNT = torch.zeros(B, dtype=torch.long, device=device)
        total_time = 0.0
        timing_metrics = None
        has_ended = torch.zeros(B, dtype=torch.bool, device=device)
        BUCKET_STEP = 16
        buckets = torch.zeros(
            (B, max_new_tokens // BUCKET_STEP), dtype=torch.long, device=device
        )
        bucket_latencies = torch.zeros(
            (B, max_new_tokens // BUCKET_STEP), dtype=torch.float32, device=device
        )
        buckets_steps = torch.zeros(
            (B, max_new_tokens // BUCKET_STEP), dtype=torch.long, device=device
        )
        total_steps = 0
        with torch.inference_mode():
            while not (
                has_ended.all()
                or (total_steps + 1) * (self.NG + 1) >= max_new_tokens * PAD_FACTOR
            ):
                torch.cuda.synchronize()
                start = time.time()
                (
                    sampled,
                    position_ids,
                    attention_mask,
                    NA,
                    curr,
                    guide,
                    d_pkv,
                    v_pkv,
                    this_timing_metrics,
                    has_ended,
                ) = self._spec_dec_step(
                    sampled,
                    position_ids,
                    attention_mask,  # type: ignore
                    curr,
                    guide,
                    d_pkv,  # type: ignore
                    v_pkv,
                    has_ended=has_ended,  # type: ignore
                )
                torch.cuda.synchronize()
                block_time = time.time() - start
                if timing_metrics is None:
                    timing_metrics = this_timing_metrics
                else:
                    for k, v in this_timing_metrics.items():
                        timing_metrics[k] += v
                total_time += block_time
                has_ended = torch.logical_or(
                    has_ended,
                    position_ids[:, curr] - position_ids[:, S] >= max_new_tokens,
                )
                if total_steps == 0:  # If ends in first 8 tokens...
                    NA_COUNT[has_ended] += self.NG
                    STEPS[has_ended] += 1
                is_ongoing = torch.logical_not(has_ended)
                NA_COUNT[is_ongoing] += NA[is_ongoing]
                STEPS[is_ongoing] += 1
                bucket_pos = (
                    position_ids[is_ongoing, curr] - position_ids[is_ongoing, S]
                ) // BUCKET_STEP
                buckets[is_ongoing, bucket_pos] += NA[is_ongoing]
                bucket_latencies[is_ongoing, bucket_pos] += block_time
                buckets_steps[is_ongoing, bucket_pos] += 1
                total_steps += 1

        if timing_metrics is not None:
            for k, v in timing_metrics.items():
                timing_metrics[k] = v / (total_steps)

        bucket_metrics = {}
        for i in range(max_new_tokens // BUCKET_STEP):
            b_na = buckets[:, i].sum() / (buckets_steps[:, i].sum() + 1e-6)
            b_lat = bucket_latencies[:, i].sum() / (buckets_steps[:, i].sum() + 1e-6)

            bucket_metrics[f"bucket_{i}_n_accepted"] = b_na
            bucket_metrics[f"bucket_{i}_latency"] = b_lat
            bucket_metrics[f"bucket_{i}_throughput"] = ((b_na + 1) * B) / b_lat

        return sampled, {
            "n_accepted": (NA_COUNT / STEPS),
            "time_per_block": total_time / total_steps,
            "attention_mask": attention_mask,
            "timing_metrics": timing_metrics,
            **bucket_metrics,
        }

    @torch.no_grad()
    def _spec_dec_step(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        curr_pos: int,
        guides: torch.Tensor | None,
        d_pkv: DynamicCache,
        v_pkv: DynamicCache,
        has_ended: torch.Tensor,
    ):
        """
        Args:
            input_ids: (B, max_len) - current input ids - modified in place, should always have atleast NG free spaces
            position_ids: (B, max_len) - Should be in sync with input_ids & d_pkv, v_pkv
            attention_mask: (B, max_len) - Should be in sync with input_ids & d_pkv, v_pkv
            curr_pos: the last index i, where input_ids[i] was computed
            guides: (B, 1, ...) - guidance embeddings for tokens [cache_end:curr_pos]
            d_pkv: DynamicCache
            v_pkv: DynamicCache
        """
        B, max_len = input_ids.shape
        q = torch.zeros(
            B, self.NG, self.v_base.config.vocab_size, device=input_ids.device
        )
        timing_metrics = {}
        if guides is not None and self.hparams["guide_method"] == "advanced":
            guides = guides[:, None, :, :]
        start = None
        if self.benchmark_latency:
            torch.cuda.synchronize()
            start = time.time()
        for i in range(self.NG):
            d_time_start = None
            if self.benchmark_latency:
                torch.cuda.synchronize()
                d_time_start = time.time()
            if self.method == "guided-drafter" or self.method == "independent-drafter":
                d_out = self.d_base.get_decoder()(
                    input_ids[:, d_pkv.get_seq_length() : curr_pos + i + 1],
                    position_ids=position_ids[
                        :, d_pkv.get_seq_length() : curr_pos + i + 1
                    ],
                    guidance_embeds=(
                        guides[:, :, :].expand(
                            -1, -1, curr_pos + i + 1 - d_pkv.get_seq_length(), -1
                        )
                        if guides is not None
                        else None
                    ),
                    past_key_values=d_pkv,
                    attention_mask=attention_mask[:, : curr_pos + i + 1],
                    use_cache=True,
                )
                d_pkv = d_out.past_key_values
                d_logits = self.d_base.lm_head(d_out.last_hidden_state[:, -1])  # type: ignore
                d_prob = d_logits.softmax(-1)
                q[:, i] = d_prob
                if self.greedy_sample:
                    input_ids[:, curr_pos + i + 1] = d_prob.argmax(dim=-1)
                else:
                    input_ids[:, curr_pos + i + 1] = torch.multinomial(
                        d_prob, 1
                    ).squeeze(1)
            elif self.method == "eagle":
                d_hidden, d_pkv = self.eagle(
                    guides,
                    input_ids[
                        :,
                        d_pkv.get_seq_length() + 1 : curr_pos
                        + i
                        + 1,  # +1 get_seq_length as we "discard" first token
                    ],
                    position_ids=position_ids[
                        :, d_pkv.get_seq_length() + 1 : curr_pos + i + 1
                    ],
                    past_key_values=d_pkv,
                    attention_mask=attention_mask[:, 1 : curr_pos + i + 1],
                    use_cache=True,
                )
                d_hidden = d_hidden[:, -1, :]  # (B, d_model)
                guides = d_hidden.unsqueeze(1)

                d_logits = self.eagle.lm_head(self.eagle.norm(d_hidden))
                d_prob = d_logits.softmax(-1)
                q[:, i].scatter_(-1, self.eagle.d2t[None, :].expand(B, -1), d_prob)
                if self.greedy_sample:
                    next_tok = d_prob.argmax(dim=-1)
                else:
                    next_tok = torch.multinomial(d_prob, 1).squeeze(1)
                input_ids[:, curr_pos + i + 1] = self.eagle.d2t[next_tok]
            else:
                raise ValueError("Unsupported Method")

            attention_mask[:, curr_pos + i + 1] = 1.0
            position_ids[:, curr_pos + i + 1] = position_ids[:, curr_pos + i] + 1
            if d_time_start is not None:
                torch.cuda.synchronize()
                timing_metrics[f"draft_time_{i}"] = time.time() - d_time_start

        if start is not None:
            torch.cuda.synchronize()
            timing_metrics["total_drafting_time"] = time.time() - start
        # +1 to have also sample when all tokens accepted
        # Verfier
        verif_start = None
        if self.benchmark_latency:
            torch.cuda.synchronize()
            verif_start = time.time()
        v_out = self.v_base.get_decoder()(
            input_ids[:, curr_pos : curr_pos + self.NG + 1],
            position_ids=position_ids[:, curr_pos : curr_pos + self.NG + 1],
            compute_guidance=True,
            return_dict=True,
            use_cache=True,
            past_key_values=v_pkv,
            attention_mask=attention_mask[:, : curr_pos + self.NG + 1],
        )
        if verif_start is not None:
            torch.cuda.synchronize()
            timing_metrics["verifier_time"] = time.time() - verif_start
        v_hidden = v_out["out"].last_hidden_state
        v_logits = self.v_base.lm_head(v_hidden)
        if self.greedy_sample:
            v_probs = torch.zeros(
                (B, self.NG + 1, self.v_base.config.vocab_size),
                device=self.device,
                dtype=v_logits.dtype,
            )
            v_probs.scatter_(-1, v_logits.argmax(dim=-1, keepdim=True), 1.0)
        else:
            v_probs = v_logits.softmax(-1)
        v_pkv = v_out["out"].past_key_values

        # Rejective Sampling
        NA, next_dist = rejection_sampling(
            q,
            v_probs,
            input_ids[:, curr_pos + 1 : curr_pos + self.NG + 1],  # type: ignore
        )
        attention_mask[
            torch.logical_not(has_ended), curr_pos + 1 : curr_pos + self.NG + 1
        ] = (
            torch.arange(0, self.NG, device=self.device)[None, :]
            < NA[torch.logical_not(has_ended), None]
        ).long()

        position_ids[:, curr_pos + self.NG + 1] = position_ids[:, curr_pos] + NA + 1
        next_token = torch.multinomial(next_dist, 1)
        input_ids[:, curr_pos + self.NG + 1] = next_token.squeeze(1)
        attention_mask[:, curr_pos + self.NG + 1] = 1.0
        has_ended = has_ended | (
            (
                input_ids[:, curr_pos + 1 : curr_pos + self.NG + 2]
                * attention_mask[:, curr_pos + 1 : curr_pos + self.NG + 2]
            )
            == self.eot_id
        ).any(dim=-1)
        attention_mask[has_ended, curr_pos + 1 : curr_pos + self.NG + 2] = 0
        curr_pos += self.NG + 1  # type: ignore
        if PRETTY_PRINT:
            print_speculative_sampling_result(
                self.tok,
                input_ids[0, curr_pos - self.NG : curr_pos],
                NA[0],
                next_token[0, 0],
            )

        # Update guide embedding
        if self.method == "guided-drafter":
            guide_seq = v_out["guide_embd"]
            guides = self.latent_mod_prep(
                guide_seq[torch.arange(input_ids.shape[0]), NA, None]
            )
        elif self.method == "eagle":
            # This concats the guide from the last drafted token (still has to be input to drafter) and the last sampled token with corresponding hidden state
            guide_seq = v_out["guide_embd"]
            guides = torch.cat(
                (
                    guides,
                    self.eagle.fc(
                        guide_seq[torch.arange(input_ids.shape[0]), NA, None]
                    ),
                ),
                dim=1,
            )

        return (
            input_ids,
            position_ids,
            attention_mask,
            NA,
            curr_pos,
            guides,
            d_pkv,
            v_pkv,
            timing_metrics,
            has_ended,
        )

    def prep_for_gen(self, prompts):
        toks = self.tok.apply_chat_template(
            [
                [
                    {
                        "role": "system",
                        "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
                    },
                    {"role": "user", "content": msg},
                ]
                if self.method == "eagle"
                else [{"role": "user", "content": msg}]
                for msg in prompts
            ],
            padding_side="left",
            enable_thinking=False,
            return_tensors="pt",
            add_generation_prompt=True,
            padding=True,
            # truncation=True,
            # max_length=256,
            return_dict=True,
            tokenizer_kwargs={
                "return_attention_mask": True,
            },
        )
        input_ids = toks["input_ids"].to(self.device)
        attention_mask = toks["attention_mask"].to(self.device)
        # We get the first True of the loss mask as the end of prompt
        return input_ids, attention_mask

    @torch.no_grad()
    def _generate_vanilla(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 32,
        batch={},
    ):
        B, S = input_ids.shape
        device = input_ids.device

        sampled = torch.full(
            (B, S + max_new_tokens),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        sampled[:, :S] = input_ids

        position_ids = torch.arange(0, S + max_new_tokens, device=device)[
            None, :
        ].expand(B, -1)

        if attention_mask is not None:
            position_ids = torch.clamp_min_(
                position_ids - S + attention_mask.sum(dim=1)[:, None], 0
            )
            attention_mask = torch.cat(
                (attention_mask, torch.zeros((B, max_new_tokens), device=device)),
                dim=1,
            )
        else:
            attention_mask = torch.zeros((B, S + max_new_tokens), device=device)
            attention_mask[:, :S] = 1.0

        v_pkv = DynamicCache()
        v_out = self.v_base.get_decoder()(
            sampled[:, :S],
            position_ids=position_ids[:, :S],
            past_key_values=v_pkv,
            attention_mask=attention_mask[:, :S].to(torch.bool),
            use_cache=True,
        )

        v_pkv = v_out.past_key_values

        curr = S - 1
        has_ended = torch.zeros(B, dtype=torch.bool, device=device)
        eot_id = getattr(self, "eot_id", self.pad_token_id)

        total_time = 0.0

        for _ in range(max_new_tokens):
            torch.cuda.synchronize()
            start = time.time()

            out = self.v_base.get_decoder()(
                sampled[:, curr + 1 : curr + 2],
                past_key_values=v_pkv,
                attention_mask=attention_mask[:, : curr + 2].to(torch.bool),
                position_ids=position_ids[:, curr + 1 : curr + 2],
                use_cache=True,
            )

            v_pkv = out.past_key_values
            logits = self.v_base.lm_head(out.last_hidden_state[:, -1])

            probs = logits.softmax(dim=-1)

            next_token = (
                probs.argmax(dim=-1)
                if self.greedy_sample
                else torch.multinomial(probs, 1).squeeze(1)
            )

            curr += 1
            sampled[:, curr] = next_token
            attention_mask[:, curr] = 1.0
            position_ids[:, curr] = position_ids[:, curr - 1] + 1
            has_ended |= next_token == eot_id

            torch.cuda.synchronize()
            total_time += time.time() - start

        return sampled, {
            "n_accepted": torch.ones((B,), dtype=torch.float),  # vanilla accepts all
            "time_per_block": total_time / max_new_tokens,
            "attention_mask": attention_mask,
            "timing_metrics": {},
        }


if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_float32_matmul_precision("high")
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=OUT_PATH,
            every_n_epochs=1,
            enable_version_counter=True,
        ),
    ]

    cli = LightningCLI(
        TrainingModule,
        DataLoader,
        trainer_defaults={
            "callbacks": callbacks,
        },
        save_config_callback=None,
    )


def set_pretty_print(pretty_print: bool):
    """
    Set the global pretty print flag for speculative decoding.
    """
    global PRETTY_PRINT
    PRETTY_PRINT = pretty_print
