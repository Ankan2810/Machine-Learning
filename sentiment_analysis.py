import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from phobert_model import PhoBERTModel
from preprocessing import clean_text


def preprocess_dataset(file_path):
    """Load dataset, xử lý NaN, kiểm tra giá trị label và fix lỗi"""
    df = pd.read_csv(file_path)

    # 🔍 Xóa NaN
    df.dropna(inplace=True)

    # 🔍 Chuyển label về kiểu số nguyên
    df["label"] = df["label"].astype(int)

    # 🔍 Kiểm tra giá trị bất thường
    unique_labels = df["label"].unique()
    if not np.all(np.isin(unique_labels, [0, 1, 2])):  # Đảm bảo nhãn chỉ có 0,1,2
        print(f"🚨 Dataset {file_path} có label không hợp lệ: {unique_labels}")
        df = df[df["label"].between(0, 2)]  # Xóa nhãn không hợp lệ
        print("✅ Fixed labels.")

    # 🔍 Lưu dataset đã xử lý
    df.to_csv(file_path, index=False)
    return file_path

def fine_tune_phobert():
    """Hàm train mô hình PhoBERT"""
    model_path = "vinai/phobert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)  # Fix lỗi tokenizer
    model = RobertaForSequenceClassification.from_pretrained(
        model_path, num_labels=3, ignore_mismatched_sizes=True  # 🔥 Đặt num_labels=3 để phù hợp với dataset
    )

    # Xử lý dataset trước khi load
    train_path = preprocess_dataset("./data/train.csv")
    test_path = preprocess_dataset("./data/test.csv")

    # Load dataset
    dataset = load_dataset("csv", data_files={"train": train_path, "test": test_path})

    # 🔍 Kiểm tra NaN và labels trong dataset
    labels = np.array(dataset["train"]["label"])
    print("✅ NaN in dataset:", np.isnan(labels).any())
    print("✅ Unique labels:", np.unique(labels))

    # 🔍 Kiểm tra token có vượt quá vocab không
    vocab_size = tokenizer.vocab_size
    print("📌 PhoBERT vocab size:", vocab_size)

    for sample in dataset["train"]:
        tokens = tokenizer(sample["comment"], padding="max_length", truncation=True, max_length=256)
        if max(tokens["input_ids"]) >= vocab_size:
            print(f"🚨 Lỗi: Input {sample['comment']} có token ngoài vocab!")

    # Tokenize dữ liệu
    def preprocess_function(examples):
        tokens = tokenizer(examples["comment"], padding="max_length", truncation=True, max_length=256)
        for i, token_list in enumerate(tokens["input_ids"]):
            if max(token_list) >= tokenizer.vocab_size:
                print(f"🚨 Lỗi: Input {examples['comment'][i]} có token ngoài vocab!")
        return tokens

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Cấu hình training
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,  # Giảm batch size tránh lỗi GPU
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,  # Điều chỉnh để không giảm tốc độ học
        num_train_epochs=3,
        weight_decay=0.01,
        report_to="none",
        use_cpu=True  # 🔥 Sửa `use_cpu=True` → `no_cuda=True` để chạy trên CPU nếu cần
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    print("🚀 Starting training...")
    trainer.train()

    trainer.save_model("./sentiment_phobert")
    print("✅ Model saved successfully!")

def main():
    """Hàm chạy training và dự đoán"""
    fine_tune_phobert()

    # Load model đã train xong để dự đoán
    model_path = "./sentiment_phobert"
    analyzer = PhoBERTModel(model_path)

    # Load dataset để test
    df = pd.read_csv("./data/train.csv")

    for index, row in df.iterrows():
        text = clean_text(row['comment'])
        sentiment, confidence, scores = analyzer.predict(text)

        print(f"Text: {text}")
        print(f"Predicted Sentiment: {sentiment}, Confidence: {confidence:.2f}")
        print(f"Scores: {scores}\n")

if __name__ == "__main__":
    main()
