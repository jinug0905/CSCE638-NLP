import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Force to use GPU 1

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm

# ========== Gumbel Noise ========== #
def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens

@torch.no_grad()
def mdm_generate(model, prompt_ids, steps=128, gen_length=128, block_length=32,
                 temperature=0., remasking='low_confidence', mask_id=126336):
    x = torch.full((1, prompt_ids.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt_ids.shape[1]] = prompt_ids.clone()

    num_blocks = gen_length // block_length
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt_ids.shape[1] + num_block * block_length:
                                prompt_ids.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            logits = model(x).logits
            logits_with_noise = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt_ids.shape[1] + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x

# ========== BBH Evaluation ========== #
# -------- Few-shot Prompt Constructor for LLaDA -------- #
def format_3shot_prompt(dataset, test_idx):
    examples = [ex for i, ex in enumerate(dataset) if i != test_idx][:3]
    test = dataset[test_idx]

    shots = ""
    for ex in examples:
        shots += f"Q: {ex['input'].strip()}\nA: {ex['target'].strip()}\n\n"
    prompt = shots + f"Q: {test['input'].strip()}\nA:"
    return prompt, test["target"]

# -------- Modified Evaluation with 3-shot Prompting -------- #
def evaluate_bbh_task(model, tokenizer, task_name):
    dataset = load_dataset("lukaemon/bbh", name=task_name, split="test")
    correct = 0

    for i in tqdm(range(len(dataset)), desc=f"Evaluating {task_name} (3-shot)"):
        prompt, target = format_3shot_prompt(dataset, i)

        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        output_ids = mdm_generate(model, prompt_ids)
        decoded = tokenizer.batch_decode(output_ids[:, prompt_ids.shape[1]:], skip_special_tokens=True)[0]
        prediction = decoded.strip().split("\n")[0].strip()

        if prediction.lower() == target.lower():
            correct += 1

    return correct / len(dataset)


# ========== Main Execution ========== #
if __name__ == "__main__":
    model_path = "./local_models/llada-8b-base"

    # Fix corrupted tokenizer by redownloading
    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        trust_remote_code=True,
        force_download=True
    )
    tokenizer.save_pretrained(model_path)

    # Load model and tokenizer from local path
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    tasks = [
        "boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa",
        "dyck_languages", "formal_fallacies", "geometric_shapes", "hyperbaton",
        "logical_deduction_five_objects", "logical_deduction_seven_objects",
        "logical_deduction_three_objects", "movie_recommendation", "multistep_arithmetic_two",
        "navigate", "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
        "ruin_names", "salient_translation_error_detection", "snarks",
        "sports_understanding", "temporal_sequences", "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects",
        "web_of_lies", "word_sorting"
    ]

    results = []
    for task in tasks:
        try:
            acc = evaluate_bbh_task(model, tokenizer, task)
            results.append({"task": task, "accuracy": acc})
            pd.DataFrame(results).to_csv("llada_bbh_fewshot_results.csv", index=False)
        except Exception as e:
            print(f"[!] Error on task {task}: {e}")
            results.append({"task": task, "accuracy": None})
            pd.DataFrame(results).to_csv("llada_bbh_fewshot_results.csv", index=False)

    print("\n=== BBH Evaluation Summary (LLaDA-8B-Base) ===")
    for r in results:
        print(f"{r['task']:40s}: {r['accuracy'] if r['accuracy'] is not None else 'ERROR'}")

