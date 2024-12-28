import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


# Define a multi-head self-attention layer
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert (
                self.head_dim * num_heads == embed_size
        ), "Embedding size must be divisible by number of heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.num_heads pieces
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = query.reshape(N, query_len, self.num_heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Compute attention scores
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.embed_size
        )

        out = self.fc_out(out)
        return out


# Define a feed-forward network
class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden)
        self.fc2 = nn.Linear(ff_hidden, embed_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define a single transformer block
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, ff_hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


# Define the GPT-like model
class GPT(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden, num_layers, vocab_size, max_length, dropout):
        super(GPT, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, ff_hidden, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        out = self.fc_out(out)
        return out


# Hyperparameters
embed_size = 256
num_heads = 8
ff_hidden = 1024
num_layers = 6
vocab_size = 50257  # GPT-2 tokenizer vocab size
max_length = 100
dropout = 0.1

# Instantiate the model
model = GPT(embed_size, num_heads, ff_hidden, num_layers, vocab_size, max_length, dropout)


# Hugging Face Dataset
class HuggingFaceDataset(Dataset):
    def __init__(self, dataset_path, dataset_name, split, tokenizer, max_length, subset_size= None):
        # Load dataset from Hugging Face
        self.dataset = load_dataset(dataset_path, dataset_name, split=split)
        if subset_size is not None:
            self.dataset = self.dataset.select(range(min(subset_size, len(self.dataset))))

        self.tokenizer = tokenizer
        self.max_length = max_length

        if tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]  # Adjust the column name as needed
        tokens = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length)["input_ids"]

        # Ensure x and y don't have indices exceeding vocab_size
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        x = torch.clamp(x, 0, vocab_size - 1)  # Clamp values to be within the vocabulary range

        y = torch.tensor(tokens[1:], dtype=torch.long)
        y = torch.clamp(y, 0, vocab_size - 1)  # Clamp values to be within the vocabulary range

        return x, y


# Initialize tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained("gpt2")
dataset_name = "wikitext"
split = "train"

huggingface_dataset = HuggingFaceDataset('wikitext', 'wikitext-103-raw-v1', split, tokenizer, max_length, 50000)
huggingface_dataloader = DataLoader(huggingface_dataset, batch_size=4, shuffle=True)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_model(model, dataloader, criterion, optimizer, device, epochs=10):
    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)

            # Forward pass
            outputs = model(x, mask=None)
            outputs = outputs.view(-1, outputs.size(-1))
            y = y.view(-1)

            # Compute loss
            loss = criterion(outputs, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")


def test_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            outputs = model(x, mask=None)
            outputs = outputs.view(-1, outputs.size(-1))
            y = y.view(-1)

            # Compute predictions
            loss = criterion(outputs, y)
            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == y).sum().item()
            total += y.size(0)

    print(f"Test Loss: {total_loss / len(dataloader):.4f}, Accuracy: {correct / total:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    train_model(model, huggingface_dataloader, criterion, optimizer, device, epochs=5)
    test_model(model, huggingface_dataloader, device)
