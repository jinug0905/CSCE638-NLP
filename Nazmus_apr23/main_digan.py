# === INSTALL DEPENDENCIES ===# === IMPORTS ===
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, 
    BitsAndBytesConfig, AutoModel
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, load_from_disk

# === DEVICE SETUP FOR 2 GPUs ===
assert torch.cuda.device_count() >= 2, "You need 2 GPUs (e.g., 2×4090s)"
device_student = torch.device("cuda:0")
device_teacher = torch.device("cuda:1")

# === LOCAL MODEL PATHS ===
local_student_dir = "./local_models/llada-8b-base"
local_teacher_dir = "./local_models/deepseek-llm-7b-base"

# === DOWNLOAD AND SAVE MODELS LOCALLY (first run only) ===
if not os.path.exists(local_student_dir):
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True)
    model = AutoModel.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer.save_pretrained(local_student_dir)
    model.save_pretrained(local_student_dir)

if not os.path.exists(local_teacher_dir):
    teacher_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-base")
    teacher_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-llm-7b-base", torch_dtype=torch.float16)
    teacher_tokenizer.save_pretrained(local_teacher_dir)
    teacher_model.save_pretrained(local_teacher_dir)

# === LOAD TOKENIZERS ===
tokenizer = AutoTokenizer.from_pretrained(local_student_dir, trust_remote_code=True)
teacher_tokenizer = AutoTokenizer.from_pretrained(local_teacher_dir)
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# === LOAD TEACHER MODEL (DeepSeek on GPU 1) ===
bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
teacher_model = AutoModelForCausalLM.from_pretrained(
    local_teacher_dir,
    quantization_config=bnb_cfg,
    device_map={"": device_teacher},
    torch_dtype=torch.float16
)

# === LOAD STUDENT MODEL (LLADA + LoRA on GPU 0) ===
student = AutoModel.from_pretrained(
    local_student_dir,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(device_student)
lora_cfg = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)
student = get_peft_model(student, lora_cfg)

# === DEFINE DISCRIMINATOR ===
class Discriminator(nn.Module):
    def __init__(self, base="bert-base-uncased"):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(base)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return torch.sigmoid(self.classifier(cls))

discriminator = Discriminator().to(device_student)

optimizer_student = torch.optim.AdamW(student.parameters(), lr=3e-5)
optimizer_disc = torch.optim.AdamW(discriminator.parameters(), lr=1e-5)

# === LOAD ALPACA DATASET FROM LOCAL ===
dataset = load_from_disk("./local_datasets/alpaca")
dataset = dataset["train"].shuffle(seed=42).select(range(10))

# === IMPROVED LLaDA-STYLE MASKED SAMPLING ===
def sample_llada(prompt, steps=5, mask_token_id=126336):
    gen_len = 64
    encoded = tokenizer(prompt, return_tensors="pt").to(device_student)
    input_ids = encoded.input_ids
    masked = torch.cat([input_ids, torch.full((1, gen_len), mask_token_id, device=device_student)], dim=1)
    attn_mask = torch.ones_like(masked)

    for _ in range(steps):
        logits = student(input_ids=masked, attention_mask=attn_mask).logits
        logits = logits[:, -gen_len:, :]  # only last masked tokens
        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=10, dim=-1)

        sampled_ids = torch.multinomial(topk_probs.squeeze(0), num_samples=1).squeeze(-1)  # (gen_len,)
        sampled = topk_indices.squeeze(0).gather(1, sampled_ids.unsqueeze(-1)).squeeze(-1)  # (gen_len,)

        max_probs, _ = torch.max(probs, dim=-1)
        remask = max_probs < 0.9  # (1, gen_len)

        masked[:, -gen_len:] = torch.where(remask, sampled.unsqueeze(0), mask_token_id)

    return tokenizer.decode(masked[0, -gen_len:], skip_special_tokens=True)

# === TRAINING LOOP ===
lambda_adv = 0.5
for sample in dataset:
    prompt = sample["instruction"]
    with torch.no_grad():
        teacher_inputs = teacher_tokenizer(prompt, return_tensors="pt").to(device_teacher)
        teacher_out = teacher_model.generate(
            **teacher_inputs,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            max_new_tokens=128,
            repetition_penalty=1.2,
            pad_token_id=teacher_tokenizer.eos_token_id
        )
        real_resp = teacher_tokenizer.decode(teacher_out[0], skip_special_tokens=True)

    fake_resp = sample_llada(prompt)

    real_pair = bert_tokenizer(f"[CLS] {prompt} {real_resp}", return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    fake_pair = bert_tokenizer(f"[CLS] {prompt} {fake_resp}", return_tensors="pt", padding="max_length", truncation=True, max_length=512)

    real_pair = {k: v.to(device_student) for k, v in real_pair.items() if k in ["input_ids", "attention_mask"]}
    fake_pair = {k: v.to(device_student) for k, v in fake_pair.items() if k in ["input_ids", "attention_mask"]}

    real_score = discriminator(input_ids=real_pair["input_ids"], attention_mask=real_pair["attention_mask"])
    fake_score = discriminator(input_ids=fake_pair["input_ids"], attention_mask=fake_pair["attention_mask"])
    d_loss = -torch.mean(torch.log(real_score + 1e-6) + torch.log(1 - fake_score + 1e-6))

    optimizer_disc.zero_grad()
    d_loss.backward()
    optimizer_disc.step()

    gan_score = discriminator(input_ids=fake_pair["input_ids"], attention_mask=fake_pair["attention_mask"])
    gan_loss = -torch.mean(torch.log(gan_score + 1e-6))

    mdm_loss = torch.tensor(0.5, requires_grad=True, device=device_student)
    total_loss = mdm_loss + lambda_adv * gan_loss

    optimizer_student.zero_grad()
    total_loss.backward()
    optimizer_student.step()

    print(f"Prompt:\n{prompt}\n\nReal Response:\n{real_resp}\n\nFake Response:\n{fake_resp}\nD_Loss: {d_loss.item():.4f} | G_Loss: {total_loss.item():.4f}\n")
