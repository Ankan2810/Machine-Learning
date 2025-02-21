from transformers import AutoTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
import pandas as pd
from datasets import load_dataset
from phobert_model import PhoBERTModel
from preprocessing import clean_text
import os
os.environ["WANDB_DISABLED"] = "true"

def fine_tune_phobert():
    model_path = "vinai/phobert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=3)

    # Load dataset
    dataset = load_dataset("csv", data_files={"train": "./data/train.csv", "test": "./data/test.csv"})

    # Tokenize dữ liệu
    def preprocess_function(examples):
        return tokenizer(examples["comment"], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Cấu hình training
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    # Train model
    trainer.train()

    # Lưu model đã train
    trainer.save_model("./sentiment_phobert")

def main():
    # Bước 1: Fine-tune PhoBERT (chỉ chạy 1 lần)
    # Nếu đã train xong, hãy comment dòng này
    fine_tune_phobert()

    # Bước 2: Load model đã train và thực hiện dự đoán cảm xúc
    model_path = "./sentiment_phobert"
    analyzer = PhoBERTModel(model_path)

    # Load dataset
    df = pd.read_csv('./data/train.csv')

    for index, row in df.iterrows():
        text = clean_text(row['comment'])  # Làm sạch văn bản
        sentiment, confidence, scores = analyzer.predict(text)

        print(f"Text: {text}")
        print(f"Predicted Sentiment: {sentiment}, Confidence: {confidence:.2f}")
        print(f"Scores: {scores}\n")

if __name__ == "__main__":
    main()
