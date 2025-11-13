# Code for SD^2
## Getting Started
1. Create a new python environment with python version 3.12
```bash
> python -m venv ./sdsd
> ./sdsd/bin/activate
```
2. Install dependencies
```bash
> pip install -r requirements.txt
```
3. Note that you will have to login to [Hugging Face](https://huggingface.co/docs/huggingface_hub/en/guides/cli#hf-auth-login) and have access to the Llama Models. Additionally training requires [Weights & Biases](https://wandb.ai/site/)
## Getting the Checkpoint Files
- The available checkpoint and config files can be found in this [Huggingface collection](https://hf.co/collections/peerrh/sd-square)

## Running Evaluation
1. Open `eval.py` and add the configurations you'd like to try out to `configs`
  - This can also be narrowed down when running by using the `--pattern` flag
2. Set the other variables as desired
3. Run the script
```bash
> python eval.py --out_file 'eval_results.json'
# We recommend piping output to a seperate file for later Evaluation
> python eval.py --pattern "llama" --out_file "llama_results.json"  1> llama_out.log
```

## Running Training
1. Update `configs/experiment.yaml with correct data
2. Generate a synthetic dataset
```bash
> python create_synth_ds.py --bsz 128 --target_len 256  --use_ultrachat_prompts --model_name 'meta-llama/llama-3.1-8b-instruct'
```
3. Run training script
```bash
> python main.py fit --config configs/experiment.yaml --trainer.max_epochs 6 --trainer.val_check_interval 2000 --data.path ./data/synthetic/llama-3.1-8b-instruct-ultrachat-prompts --data.bsz=12 --trainer.accumulate_grad_batches 2 --model.loss_method kl --model.drafter=meta-llama/llama-3.2-1b-instruct --model.finetune_drafter full --model.guide_method merged --model.d_layer all --model.verifier=meta-llama/llama-3.1-8b-instruct --model.v_layer '[3,16,29]' --model.lr_start 0.00001 --model.lr_end 0.000001 --model.warmup_steps 1000  --model.method guided-drafter --data.n_val 3000
```
