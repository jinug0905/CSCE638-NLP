

# === IMPORTS ===
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, 
    BitsAndBytesConfig, AutoModel
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from torch.utils.data import DataLoader

# === DEVICE SETUP FOR 2 GPUs ===
assert torch.cuda.device_count() >= 2, "You need 2 GPUs (e.g., 2×4090s)"
device_student = torch.device("cuda:0")
device_teacher = torch.device("cuda:1")

# === LOCAL MODEL PATHS ===
local_student_dir = "./local_models/llada-8b-base"
local_teacher_dir = "./local_models/deepseek-llm-7b-base"

# === LOAD TOKENIZERS ===
tokenizer = AutoTokenizer.from_pretrained(local_student_dir, trust_remote_code=True)
teacher_tokenizer = AutoTokenizer.from_pretrained(local_teacher_dir)
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# === LOAD TEACHER MODEL (DeepSeek) ===
bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
teacher_model = AutoModelForCausalLM.from_pretrained(
    local_teacher_dir,
    quantization_config=bnb_cfg,
    device_map={"": device_teacher},
    torch_dtype=torch.float16
)

# === LOAD STUDENT MODEL (LLADA + LoRA) ===
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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return torch.sigmoid(self.classifier(cls))



discriminator = Discriminator().to(device_student)

optimizer_student = torch.optim.AdamW(student.parameters(), lr=3e-5)
optimizer_disc = torch.optim.AdamW(discriminator.parameters(), lr=1e-5)

# === LOAD NON-INSTRUCTION DATASET ===
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
dataset = dataset["train"].shuffle(seed=42).select(range(10))

batch_size = 4
grad_accum_steps = 16
dataloader = DataLoader(dataset, batch_size=batch_size)

# === OFFICIAL LLADA GENERATION FUNCTION ===
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
def generate_llada(model, prompt_ids, steps=128, gen_length=128, block_length=128, temperature=0., mask_id=126336):
    x = torch.full((1, prompt_ids.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt_ids.shape[1]] = prompt_ids.clone()
    prompt_index = (x != mask_id)
    num_blocks = gen_length // block_length
    steps = steps // num_blocks
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt_ids.shape[1] + num_block * block_length: prompt_ids.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            logits = model(x).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            x0_p[:, prompt_ids.shape[1] + (num_block + 1) * block_length:] = -float('inf')
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -float('inf'))
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
    return x

# === EVALUATION (Official Generation) ===
prompt = "The history of deep learning begins with the idea of"
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device_student)
out = generate_llada(student, input_ids)
print("\nEvaluation Prompt:", prompt)
print("Student Output:", tokenizer.decode(out[:, input_ids.shape[1]:][0], skip_special_tokens=True))

# === TRAINING LOOP (ADDED GRADIENT ACCUMULATION) ===
log_path = "./logs/llada_gan_pretrain_log.txt"
os.makedirs("./logs", exist_ok=True)
with open(log_path, "a") as f:
    f.write("\n=== Starting Training ===\n")

for epoch in range(100):
    total_loss, disc_loss = 0, 0
    for i, batch in enumerate(dataloader):
        text_batch = batch["text"]
        prompts = [t.strip().split(".")[0] for t in text_batch]

        real_resps, fake_resps = [], []
        for prompt in prompts:
            with torch.no_grad():
                teacher_inputs = teacher_tokenizer(prompt, return_tensors="pt").to(device_teacher)
                teacher_out = teacher_model.generate(**teacher_inputs, max_new_tokens=128, temperature=0.9, top_k=50)
                real_resps.append(teacher_tokenizer.decode(teacher_out[0], skip_special_tokens=True))

                llada_input = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device_student)
                fake_out = generate_llada(student, llada_input)
                fake_resps.append(tokenizer.decode(fake_out[:, llada_input.shape[1]:][0], skip_special_tokens=True))

        real_inputs = bert_tokenizer([f"[CLS] {p} {r}" for p, r in zip(prompts, real_resps)], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device_student)
        fake_inputs = bert_tokenizer([f"[CLS] {p} {f}" for p, f in zip(prompts, fake_resps)], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device_student)

        real_score = discriminator(**real_inputs)
        fake_score = discriminator(**fake_inputs)
        d_loss = -torch.mean(torch.log(real_score + 1e-6) + torch.log(1 - fake_score + 1e-6)) / grad_accum_steps
        disc_loss += d_loss.item()

        d_loss.backward()
        if (i + 1) % grad_accum_steps == 0:
            optimizer_disc.step()
            optimizer_disc.zero_grad()

        student_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device_student)
        noisy = torch.full_like(student_inputs["input_ids"], tokenizer.mask_token_id)
        logits = student(noisy).logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), student_inputs["input_ids"].view(-1), reduction="mean") / grad_accum_steps
        total_loss += loss.item()

        loss.backward()
        if (i + 1) % grad_accum_steps == 0:
            optimizer_student.step()
            optimizer_student.zero_grad()

        if (i + 1) % (4 * grad_accum_steps) == 0:
            print(f"Epoch {epoch+1}, Step {i+1}, G_Loss: {total_loss:.4f}, D_Loss: {disc_loss:.4f}")
            with open(log_path, "a") as f:
                f.write(f"Epoch {epoch+1}, Step {i+1}, G_Loss: {total_loss:.4f}, D_Loss: {disc_loss:.4f}\n")

