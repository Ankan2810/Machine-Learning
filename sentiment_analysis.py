from transformers import AutoTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from phobert_model import PhoBERTModel
from preprocessing import clean_text
import os

def preprocess_dataset(file_path):
    """Load dataset, xử lý NaN, kiểm tra giá trị label và fix lỗi"""
    df = pd.read_csv(file_path)

    # 🔍 Xóa NaN
    df.dropna(inplace=True)

    # 🔍 Chuyển label về kiểu số nguyên
    df["label"] = df["label"].astype(int)

    # 🔍 Kiểm tra giá trị bất thường
    unique_labels = df["label"].unique()
    if not np.all(np.isin(unique_labels, [0, 1])):  # Đảm bảo nhãn chỉ có 0,1
        print(f"🚨 Dataset {file_path} có label không hợp lệ: {unique_labels}")
        df = df[df["label"].between(0, 1)]  # Xóa nhãn không hợp lệ
        print("✅ Fixed labels.")

    # 🔍 Lưu dataset đã xử lý
    df.to_csv(file_path, index=False)
    return file_path


def fine_tune_phobert():
    model_path = "vinai/phobert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=2, ignore_mismatched_sizes=True)
    
    # Xử lý dataset
    preprocess_dataset("./data/train.csv")
    preprocess_dataset("./data/test.csv")

    # Load dataset
    dataset = load_dataset("csv", data_files={"train": "./data/train.csv", "test": "./data/test.csv"})
    
    # Kiểm tra NaN và labels trong dataset
    labels = np.array(dataset["train"]["label"])
    print("NaN in dataset:", np.isnan(labels).any())
    print("Unique labels:", np.unique(labels))
    
    # 🔍 Kiểm tra token có vượt quá vocab không
    for sample in dataset["train"]:
        tokens = tokenizer(sample["comment"], padding="max_length", truncation=True, max_length=512)
        if max(tokens["input_ids"]) >= tokenizer.vocab_size:
            print(f"🚨 Lỗi: Input {sample['comment']} có token ngoài vocab!")

    # Tokenize dữ liệu
    def preprocess_function(examples):
        return tokenizer(examples["comment"], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Cấu hình training
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        weight_decay=0.01,
        report_to="none",
        
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
