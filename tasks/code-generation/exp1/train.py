"""
Task: Code Generation
Model: GPT2
Dataset: CodeXGLUE
Author: GPT-4
If tested: No
"""

# Import necessary libraries
import torch
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import os
import json

# Define the dataset class
class CodeXGLUEDataset(Dataset):
    """
    CodeXGLUE dataset for code generation.
    This dataset includes pairs of code and documentation (or comments), which is used for training a code generation model.
    """
    def __init__(self, tokenizer, file_path, block_size=512):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                code, doc = json.loads(line)
                self.examples.append(tokenizer.encode(f"{doc} {tokenizer.eos_token} {code}", add_special_tokens=True))

        if len(self.examples) > block_size:
            self.examples = [x[:block_size] for x in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Load the dataset
train_dataset = CodeXGLUEDataset(tokenizer, 'path_to_train_dataset.json')
val_dataset = CodeXGLUEDataset(tokenizer, 'path_to_val_dataset.json')

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Define the optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=-1)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        inputs, labels = batch, batch
        outputs = model(inputs, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Validation loop
    model.eval()
    total_eval_loss = 0
    for batch in val_loader:
        inputs, labels = batch, batch
        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            loss = outputs[0]
            total_eval_loss += loss.item()

    avg_val_loss = total_eval_loss / len(val_loader)
    print(f'Epoch {epoch}, Validation Loss: {avg_val_loss}')

# Save the model
model.save_pretrained('path_to_save_model')
