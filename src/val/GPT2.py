import torch
import torch.nn as nn
from tqdm import tqdm
from src.dataset.dataloader import dataloader
from transformers import GPT2Tokenizer
from src.model.GPT2 import GPT2

import math

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

def val_GPT2(data_name, state_dict_full, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    rouge = Rouge()
    smooth = SmoothingFunction().method1
    total_bleu, total_rouge, total_perplexity = 0, 0, 0
    count = 0

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    test_loader = dataloader(data_name=data_name, train=False)

    model = GPT2()
    model.load_state_dict(state_dict_full)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(test_loader):
            model = model.to(device)
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['input_ids'].to(device)

            logis, _ = model(input_ids, mask)

            predicted_tokens = torch.argmax(logis, dim=-1)
            generated_texts = [tokenizer.decode(predict, skip_special_tokens=True) for predict in predicted_tokens]
            reference_texts = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]

            for gen, ref in zip(generated_texts, reference_texts):
                bleu_score = sentence_bleu([ref.split()], gen.split(), smoothing_function=smooth)
                rouge_score = rouge.get_scores(gen, ref)[0]['rouge-l']['f']

                total_bleu += bleu_score
                total_rouge += rouge_score
                count += 1

            shift_logits = logis[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            perplexity = math.exp(loss.item() / shift_labels.numel())
            total_perplexity += perplexity

        avg_bleu = total_bleu / count
        avg_rouge = total_rouge / count
        avg_perplexity = total_perplexity / count

        print(f"Evaluation Results: BLEU: {avg_bleu:.4f}, ROUGE-L: {avg_rouge:.4f}, Perplexity: {avg_perplexity:.4f}")
        logger.log_info(f"Evaluation Results: BLEU: {avg_bleu:.4f}, ROUGE-L: {avg_rouge:.4f}, Perplexity: {avg_perplexity:.4f}")