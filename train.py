from datasets import Dataset

from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments
)
from datasets import Dataset  # Changed from load_dataset
import pandas as pd
import torch

# 1. Load local CSV
df = pd.read_csv("clean_train.csv")  # Update path if needed
dataset = Dataset.from_pandas(df)

# 2. Initialize model
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# 3. Tokenization
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)  # Update "text" to your column name

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 4. Training (unchanged)
training_args = TrainingArguments(...)
trainer = Trainer(...)
trainer.train()
model.save_pretrained("./saved_model")


# #train.py
# from transformers import (
#     DistilBertForSequenceClassification,
#     DistilBertTokenizerFast,
#     Trainer,
#     TrainingArguments
# )
# from datasets import load_dataset
# import torch

# #Load dataset (example: IMDB reviews)
# dataset = load_dataset("clean_train.csv")

# # Initialize tokenizer and model
# tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
# model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# # Tokenize data
# def tokenize(batch):
#     return tokenizer(batch["text"], padding=True, truncation=True)

# dataset = dataset.map(tokenize, batched=True, batch_size=16)
# dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# # Training setup
# training_args = TrainingArguments(
#     output_dir="./results",
#     per_device_train_batch_size=8,
#     num_train_epochs=3,
#     logging_dir="./logs",
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["test"],
# )

# # Train and save
# trainer.train()
# model.save_pretrained("./saved_model")
# tokenizer.save_pretrained("./saved_model")