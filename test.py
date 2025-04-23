# import os
# import torch
# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from datasets import load_dataset
# from tqdm import tqdm

# # -------- Config -------- #
# model_path = "deepseek-ai/deepseek-llm-7b-base"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# max_new_tokens = 64

# # All BBH tasks
# bbh_tasks = [
#     "boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa",
#     "dyck_languages", "formal_fallacies", "geometric_shapes", "hyperbaton",
#     "logical_deduction_five_objects", "logical_deduction_seven_objects",
#     "logical_deduction_three_objects", "movie_recommendation", "multistep_arithmetic_two",
#     "navigate", "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
#     "ruin_names", "salient_translation_error_detection", "snarks",
#     "sports_understanding", "temporal_sequences", "tracking_shuffled_objects_five_objects",
#     "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects",
#     "web_of_lies", "word_sorting"
# ]

# # -------- Load Model -------- #
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     device_map=None
# ).to(device)
# model.eval()

# # -------- Evaluation Function -------- #
# def evaluate_task(task_name):
#     dataset = load_dataset("lukaemon/bbh", name=task_name, split="test")
#     correct = 0

#     for example in tqdm(dataset, desc=f"Evaluating {task_name}"):
#         prompt = example["input"].strip() + "\nAnswer:"
#         target = example["target"].strip()

#         inputs = tokenizer(prompt, return_tensors="pt").to(device)

#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#             pad_token_id=tokenizer.eos_token_id
#         )

#         output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         predicted = output_text[len(prompt):].strip().split("\n")[0].strip()

#         if predicted.lower() == target.lower():
#             correct += 1

#     return correct / len(dataset)

# # -------- Run Evaluation -------- #
# results = []
# for task in bbh_tasks:
#     try:
#         acc = evaluate_task(task)
#         results.append({"task": task, "accuracy": acc})
#         pd.DataFrame(results).to_csv("deepseek_bbh_results.csv", index=False)
#     except Exception as e:
#         print(f"[!] Error on task {task}: {e}")
#         results.append({"task": task, "accuracy": None})
#         pd.DataFrame(results).to_csv("deepseek_bbh_results.csv", index=False)

# # -------- Final Summary -------- #
# print("\n=== BBH Evaluation Summary (DeepSeek-7B) ===")
# for r in results:
#     print(f"{r['task']:40s}: {r['accuracy'] if r['accuracy'] is not None else 'ERROR'}")


import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# -------- Config -------- #
model_path = "./local_models/llama3-8b"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_new_tokens = 64

# All 26 BBH tasks
bbh_tasks = [
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

# -------- Load Model -------- #
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
).to(device)
model.eval()

# -------- Evaluation Function -------- #
def evaluate_task(task_name):
    dataset = load_dataset("lukaemon/bbh", name=task_name, split="test")
    correct = 0

    for example in tqdm(dataset, desc=f"Evaluating {task_name}"):
        input_text = example["input"].strip()
        target = example["target"].strip()

        prompt = f"{input_text}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        prediction = generated.strip().split("\n")[0].strip()

        if prediction.lower() == target.lower():
            correct += 1

    return correct / len(dataset)

# -------- Run Evaluation and Save -------- #
results = []
for task in bbh_tasks:
    try:
        acc = evaluate_task(task)
        results.append({"task": task, "accuracy": acc})
        pd.DataFrame(results).to_csv("llama3_bbh_results.csv", index=False)
    except Exception as e:
        print(f"[!] Failed to evaluate {task}: {e}")
        results.append({"task": task, "accuracy": None})
        pd.DataFrame(results).to_csv("llama3_bbh_results.csv", index=False)

# -------- Summary -------- #
print("\n=== BBH Evaluation Summary (LLaMA3) ===")
for r in results:
    print(f"{r['task']:45s}: {r['accuracy'] if r['accuracy'] is not None else 'ERROR'}")


###LLaDa Evaluation Script###
# This script evaluates the LLaDa model on the BBH benchmark, without original generate.py
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from datasets import load_dataset
# from tqdm import tqdm

# # -------- Config -------- #
# model_path = "./local_models/llada-8b-base"  # <-- your local path
# bbh_tasks = [
#     "date_understanding"
#     # "disambiguation_qa",
#     # "logical_deduction_three_objects",
#     # "temporal_sequences",
#     # "tracking_shuffled_objects_three_objects",
#     # "causal_judgement",
#     # "geometric_shapes"
# ]
# max_new_tokens = 64
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # -------- Load Model -------- #
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True
# ).to(device)
# model.eval()

# # -------- Evaluation Function -------- #
# def evaluate_task(task_name):
#     dataset = load_dataset("lukaemon/bbh", name=task_name, split="test")
#     correct = 0

#     for example in tqdm(dataset, desc=f"Evaluating {task_name}"):
#         input_text = example["input"].strip()
#         target = example["target"].strip()

#         # Raw prompt for base model
#         prompt = f"{input_text}\nAnswer:"

#         inputs = tokenizer(prompt, return_tensors="pt").to(device)

#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#             pad_token_id=tokenizer.eos_token_id,
#             use_cache=False  # this line fixes the MDM error
#         )


#         generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
#         prediction = generated.strip().split("\n")[0].strip()

#         if prediction.lower() == target.lower():
#             correct += 1

#     accuracy = correct / len(dataset)
#     return accuracy

# # -------- Run Evaluation -------- #
# results = {}
# for task in bbh_tasks:
#     try:
#         acc = evaluate_task(task)
#         results[task] = acc
#     except Exception as e:
#         print(f"[!] Failed to evaluate {task}: {e}")
#         results[task] = None

# # -------- Summary -------- #
# print("\n=== BBH Evaluation Summary (LLaDA-Base) ===")
# for task, acc in results.items():
#     if acc is not None:
#         print(f"{task:45s}: {acc:.3f}")
#     else:
#         print(f"{task:45s}: ERROR")



# # # === EVALUATION of LLaDa (Official Generation) ===
# import torch
# import numpy as np
# import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModel
# from datasets import load_dataset
# from tqdm import tqdm

# # ========== Gumbel Noise ========== #
# def add_gumbel_noise(logits, temperature):
#     if temperature == 0:
#         return logits
#     logits = logits.to(torch.float64)
#     noise = torch.rand_like(logits, dtype=torch.float64)
#     gumbel_noise = (- torch.log(noise)) ** temperature
#     return logits.exp() / gumbel_noise

# def get_num_transfer_tokens(mask_index, steps):
#     mask_num = mask_index.sum(dim=1, keepdim=True)
#     base = mask_num // steps
#     remainder = mask_num % steps
#     num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
#     for i in range(mask_num.size(0)):
#         num_transfer_tokens[i, :remainder[i]] += 1
#     return num_transfer_tokens

# @torch.no_grad()
# def mdm_generate(model, prompt_ids, steps=128, gen_length=128, block_length=32,
#                  temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336):
#     x = torch.full((1, prompt_ids.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
#     x[:, :prompt_ids.shape[1]] = prompt_ids.clone()
#     prompt_index = (x != mask_id)

#     num_blocks = gen_length // block_length
#     steps = steps // num_blocks

#     for num_block in range(num_blocks):
#         block_mask_index = (x[:, prompt_ids.shape[1] + num_block * block_length:
#                                 prompt_ids.shape[1] + (num_block + 1) * block_length] == mask_id)
#         num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
#         for i in range(steps):
#             mask_index = (x == mask_id)
#             logits = model(x).logits if cfg_scale == 0. else None  # no guidance for now
#             logits_with_noise = add_gumbel_noise(logits, temperature)
#             x0 = torch.argmax(logits_with_noise, dim=-1)

#             if remasking == 'low_confidence':
#                 p = F.softmax(logits.to(torch.float64), dim=-1)
#                 x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
#             elif remasking == 'random':
#                 x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
#             else:
#                 raise NotImplementedError(remasking)

#             x0_p[:, prompt_ids.shape[1] + (num_block + 1) * block_length:] = -np.inf
#             x0 = torch.where(mask_index, x0, x)
#             confidence = torch.where(mask_index, x0_p, -np.inf)

#             transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
#             for j in range(confidence.shape[0]):
#                 _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
#                 transfer_index[j, select_index] = True
#             x[transfer_index] = x0[transfer_index]

#     return x

# # ========== BBH Evaluation ========== #
# def evaluate_bbh_task(model, tokenizer, task_name):
#     dataset = load_dataset("lukaemon/bbh", name=task_name, split="test")
#     correct = 0
#     for ex in tqdm(dataset, desc=f"Evaluating {task_name}"):
#         prompt = ex["input"].strip() + "\nAnswer:"
#         target = ex["target"].strip()

#         prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
#         output_ids = mdm_generate(model, prompt_ids, steps=128, gen_length=128, block_length=32, temperature=0.0)
#         decoded = tokenizer.batch_decode(output_ids[:, prompt_ids.shape[1]:], skip_special_tokens=True)[0]
#         prediction = decoded.strip().split("\n")[0].strip()

#         if prediction.lower() == target.lower():
#             correct += 1
#     return correct / len(dataset)

# # ========== Run Full Evaluation ========== #
# if __name__ == "__main__":
#     device = 'cuda'
#     model_path = "./local_models/llada-8b-base"
#     mask_token_id = 126336  # default in LLaDA tokenizer

#     model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

#     tasks = [
#         "date_understanding"
#         # "disambiguation_qa",
#         # "logical_deduction_three_objects"
#     ]

#     results = {}
#     for task in tasks:
#         try:
#             acc = evaluate_bbh_task(model, tokenizer, task)
#             results[task] = acc
#         except Exception as e:
#             print(f"[!] Error on task {task}: {e}")
#             results[task] = None

#     print("\n=== BBH Evaluation Summary (LLaDA-Base, MDM-style) ===")
#     for task, acc in results.items():
#         print(f"{task:40s}: {acc if acc is not None else 'ERROR'}")


