<div align="center">

# **SD²: Steering Pretrained Drafters During Speculative Decoding**  

**Frédéric Berdoz · Peer Rheinboldt · Roger Wattenhofer**

[![arXiv](https://img.shields.io/badge/arXiv-2511.09844-b31b1b.svg)](https://arxiv.org/abs/2511.09844)

Accepted at FPI @ NeurIPS 2025, SPIGM @ NeurIPS 2025 and at AAAI 2026

</div>

---

## **Overview**
SD² is a framework for **steering pretrained drafters** during **speculative decoding** to further increase alignment between drafter and verifier.
This repo contains:
- Evaluation tools  
- Training scripts  
- Synthetic dataset generation  
- Configuration files and checkpoints used in the paper can be found in this [Huggingface collection](https://hf.co/collections/peerrh/sd-square)

---
## Getting Started
1. Create a new python environment with python version 3.12
```bash
python -m venv ./.venv
source ./.venv/bin/activate
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Note that you will have to login to [Hugging Face](https://huggingface.co/docs/huggingface_hub/en/guides/cli#hf-auth-login) and have access to the Llama Models. Additionally training requires [Weights & Biases](https://wandb.ai/site/)

---
## Running Evaluation
1. Open `eval.py` and add the configurations you'd like to try out to `configs`
2. Run the script
```bash
python eval.py --out_file 'eval_results.json'
# We recommend piping output to a seperate file for later Evaluation
python eval.py --pattern "llama" --out_file "llama_results.json"  > llama_out.log
```

---
## Running Training
1. Update `configs/experiment.yaml with correct data
2. Generate a synthetic dataset
```bash
python create_synth_ds.py --bsz 128 --target_len 256  --use_ultrachat_prompts --model_name 'meta-llama/llama-3.1-8b-instruct'
```
3. Run training script
```bash
python main.py fit --config configs/experiment.yaml \
  --data.path ./data/synthetic/llama-3.1-8b-instruct-ultrachat-prompts \
  --data.bsz=12 \
  --data.n_val 3000 \
  --trainer.max_epochs 6 \
  --trainer.val_check_interval 2000 \
  --trainer.accumulate_grad_batches 2 \
  --model.method guided-drafter \
  --model.lr_start 0.00001 \
  --model.lr_end 0.000001 \
  --model.warmup_steps 1000  \
  --model.loss_method kl \
  --model.drafter=meta-llama/llama-3.2-1b-instruct \
  --model.finetune_drafter full \
  --model.guide_method merged \
  --model.d_layer all \
  --model.verifier=meta-llama/llama-3.1-8b-instruct \
  --model.v_layer '[3,16,29]' 
```
