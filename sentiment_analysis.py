import os
import pandas as pd
import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from phobert_model import PhoBERTModel
from preprocessing import clean_text
import logging

# Enable CUDA debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def preprocess_dataframe(df):
    """Preprocess DataFrame in-memory"""
    df = df.dropna().copy()
    df["label"] = df["label"].astype(int)
    valid_labels = [0, 1, 2]
    invalid_labels = set(df["label"]) - set(valid_labels)
    if invalid_labels:
        logger.warning(f"Found invalid labels: {invalid_labels}. Filtering to valid labels {valid_labels}.")
        df = df[df["label"].isin(valid_labels)]
    return df

def process_chunk(chunk, tokenizer):
    """Process a single chunk of data"""
    chunk = preprocess_dataframe(chunk)
    dataset = Dataset.from_pandas(chunk)
    dataset_dict = DatasetDict({"train": dataset})

    def preprocess_function(examples):
        return tokenizer(
            examples["comment"],
            padding="max_length",
            truncation=True,
            max_length=256,
        )

    tokenized_dataset = dataset_dict.map(
        preprocess_function,
        batched=True,
        desc="Tokenizing chunk",
        remove_columns=["comment"]
    )
    return tokenized_dataset["train"]

def fine_tune_phobert(train_path="./data/train.csv", test_path="./data/test.csv", chunk_size=1000):
    """Fine-tune PhoBERT iteratively over chunks until the entire dataset is processed"""
    model_path = "vinai/phobert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=3, ignore_mismatched_sizes=True)

    # Freeze the first 3 layers to speed up training
    for name, param in model.named_parameters():
        if "roberta.encoder.layer" in name and int(name.split(".")[3]) < 3:
            param.requires_grad = False

    # Training configuration
    batch_size = 16 if torch.cuda.is_available() else 4
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2 if torch.cuda.is_available() else 8,
        num_train_epochs=1,  # 1 epoch per chunk
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        no_cuda=not torch.cuda.is_available(),
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
    )

    # Train iteratively over chunks
    logger.info("Starting training over chunks...")
    chunk_count = 0
    for chunk in pd.read_csv(train_path, chunksize=chunk_size):
        chunk_count += 1
        logger.info(f"Processing chunk {chunk_count}...")
        tokenized_train = process_chunk(chunk, tokenizer)
        trainer.train_dataset = tokenized_train
        trainer.train()

    # Load and process test dataset for evaluation
    test_df = preprocess_dataframe(pd.read_csv(test_path))
    tokenized_test = process_chunk(test_df, tokenizer)
    
    trainer.eval_dataset = tokenized_test
    logger.info("Evaluating model on test set...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")

    # Save the model after completion
    trainer.save_model("./sentiment_phobert")
    logger.info("Model saved successfully!")

def predict_sentiment(model_path="./sentiment_phobert", test_path="./data/test.csv", num_samples=5):
    """Predict sentiment with the trained model"""
    analyzer = PhoBERTModel(model_path)
    df = pd.read_csv(test_path).sample(num_samples, random_state=42)

    for _, row in df.iterrows():
        text = clean_text(row["comment"])
        sentiment, confidence, scores = analyzer.predict(text)
        logger.info(f"Text: {text}")
        logger.info(f"Predicted Sentiment: {sentiment}, Confidence: {confidence:.2f}")
        logger.info(f"Scores: {scores}")

def main():
    """Main execution function"""
    try:
        fine_tune_phobert()
        predict_sentiment()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()