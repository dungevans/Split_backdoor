import torch
import torch.nn as nn
import torch.optim as optim
import os
import math
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from datasets import load_dataset

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Chọn device (GPU nếu có)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. Định nghĩa dataset cho dữ liệu hội thoại
class ConversationDataset(Dataset):
    def __init__(self, dialogues, tokenizer, block_size=64):
        self.examples = []
        for dialogue in dialogues:
            tokenized_dialogue = tokenizer.encode(dialogue, add_special_tokens=True)
            if len(tokenized_dialogue) < block_size:
                # Bỏ qua câu quá ngắn hoặc thêm padding
                tokenized_dialogue += [tokenizer.pad_token_id] * (block_size - len(tokenized_dialogue))
            # Cắt thành các đoạn có độ dài block_size
            for i in range(0, len(tokenized_dialogue) - block_size + 1, block_size):
                self.examples.append(tokenized_dialogue[i: i + block_size])

        if not self.examples:
            raise ValueError("Dataset is empty! Check your dialogues or block_size.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        tokens = self.examples[i]
        if tokens is None:
            raise ValueError(f"Invalid tokens at index {i}")
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.tensor(tokens, dtype=torch.long)
        }


# 2. Load tokenizer và model GPT-2 đã pre-train
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 không có token pad, dùng eos_token thay thế

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to(device)

# Chia mô hình ở lớp 6
split_layer = 6


# 3. Định nghĩa ClientModel
class ClientModel(nn.Module):
    def __init__(self, model, split_layer):
        super(ClientModel, self).__init__()
        self.wte = model.transformer.wte
        self.wpe = model.transformer.wpe
        self.drop = model.transformer.drop
        self.blocks = nn.ModuleList(model.transformer.h[:split_layer])

    def forward(self, input_ids):
        batch_size, seq_length = input_ids.size()
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_length)
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        hidden_states = self.drop(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states)[0]
        return hidden_states


# 4. Định nghĩa ServerModel
class ServerModel(nn.Module):
    def __init__(self, model, split_layer):
        super(ServerModel, self).__init__()
        self.blocks = nn.ModuleList(model.transformer.h[split_layer:])
        self.ln_f = model.transformer.ln_f
        self.lm_head = model.lm_head

    def forward(self, hidden_states, labels=None):
        for block in self.blocks:
            hidden_states = block(hidden_states)[0]
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return logits, loss


# 5. Dữ liệu hội thoại mẫu
# Load dữ liệu hội thoại từ Hugging Face
dataset_name = "daily_dialog"
dataset = load_dataset(dataset_name)

dialogues = []
for conversation in dataset["train"]:
    dialogues.append(" ".join(conversation["dialog"]))

# 6. Tạo dataset và dataloader
dataset = ConversationDataset(dialogues, tokenizer, block_size=64)
print(f"Dataset size: {len(dataset)}")  # Kiểm tra dataset không rỗng

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 7. Khởi tạo mô hình Client và Server
client_model = ClientModel(model, split_layer).to(device)
server_model = ServerModel(model, split_layer).to(device)

# 8. Định nghĩa optimizer
optimizer_client = optim.Adam(client_model.parameters(), lr=5e-5)
optimizer_server = optim.Adam(server_model.parameters(), lr=5e-5)

# 9. Training loop mô phỏng split learning
num_epochs = 3
client_model.train()
server_model.train()

def evaluate_model(dataloader, client_model, server_model, tokenizer, device):
    client_model.eval()
    server_model.eval()
    rouge = Rouge()
    smooth = SmoothingFunction().method1  # Sử dụng SmoothingFunction để tránh BLEU = 0
    total_bleu, total_rouge, total_perplexity = 0, 0, 0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # --- Phía Client ---
            client_output = client_model(input_ids)

            # --- Phía Server ---
            logits, _ = server_model(client_output, labels=None)

            # Chuyển đổi thành câu
            predicted_tokens = torch.argmax(logits, dim=-1)
            generated_texts = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predicted_tokens]
            reference_texts = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]

            # Tính BLEU và ROUGE
            for gen, ref in zip(generated_texts, reference_texts):
                bleu_score = sentence_bleu([ref.split()], gen.split(), smoothing_function=smooth)
                rouge_score = rouge.get_scores(gen, ref)[0]['rouge-l']['f']

                total_bleu += bleu_score
                total_rouge += rouge_score
                count += 1

            # Tính Perplexity
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            perplexity = math.exp(loss.item() / shift_labels.numel())
            total_perplexity += perplexity

    avg_bleu = total_bleu / count
    avg_rouge = total_rouge / count
    avg_perplexity = total_perplexity / count

    print(f"Evaluation Results: BLEU: {avg_bleu:.4f}, ROUGE-L: {avg_rouge:.4f}, Perplexity: {avg_perplexity:.4f}")

    client_model.train()
    server_model.train()


for epoch in range(num_epochs):
    print(f"\n--- Starting Epoch {epoch + 1}/{num_epochs} ---")  # Xác nhận bắt đầu epoch

    for batch_idx, batch in enumerate(dataloader):
        optimizer_client.zero_grad()
        optimizer_server.zero_grad()

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # --- Phía Client ---
        client_output = client_model(input_ids)
        client_output_detached = client_output.clone().detach().requires_grad_(True)

        # --- Phía Server ---
        logits, loss = server_model(client_output_detached, labels=labels)

        # --- Backpropagation ---
        loss.backward()
        grad_from_server = client_output_detached.grad
        client_output.backward(grad_from_server)

        optimizer_client.step()
        optimizer_server.step()

        # ✅ In thông tin batch và epoch đúng cách
        print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    # ✅ Thực hiện đánh giá mô hình sau mỗi epoch
    print(f"\n--- Evaluation after Epoch {epoch + 1} ---")
    evaluate_model(dataloader, client_model, server_model, tokenizer, device)
    print("\n" + "=" * 50 + "\n")
