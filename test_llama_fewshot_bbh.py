import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# -------- Config -------- #
model_path = "./local_models/llama3-8b"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_new_tokens = 64

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
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
model.eval()

# -------- Few-shot Prompt Constructor -------- #
def format_3shot_prompt(dataset, test_idx):
    examples = [ex for i, ex in enumerate(dataset) if i != test_idx][:3]
    test = dataset[test_idx]

    shots = ""
    for ex in examples:
        shots += f"Q: {ex['input'].strip()}\nA: {ex['target'].strip()}\n\n"
    prompt = shots + f"Q: {test['input'].strip()}\nA:"
    return prompt, test["target"]

# -------- Evaluate Task -------- #
def evaluate_task(task_name):
    dataset = load_dataset("lukaemon/bbh", name=task_name, split="test")
    correct = 0
    for i in tqdm(range(len(dataset)), desc=f"Evaluating {task_name}"):
        prompt, target = format_3shot_prompt(dataset, i)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = decoded[len(prompt):].strip().split("\n")[0].strip()

        if prediction.lower() == target.lower():
            correct += 1

    return correct / len(dataset)

# -------- Run Evaluation -------- #
results = []
for task in bbh_tasks:
    try:
        acc = evaluate_task(task)
        results.append({"task": task, "accuracy": acc})
        pd.DataFrame(results).to_csv("llama3_3shot_bbh_results.csv", index=False)
    except Exception as e:
        print(f"[!] Failed on {task}: {e}")
        results.append({"task": task, "accuracy": None})
        pd.DataFrame(results).to_csv("llama3_3shot_bbh_results.csv", index=False)

# -------- Summary -------- #
print("\n=== 3-Shot BBH Evaluation Summary (LLaMA3-8B-Base) ===")
for r in results:
    print(f"{r['task']:45s}: {r['accuracy'] if r['accuracy'] is not None else 'ERROR'}")

