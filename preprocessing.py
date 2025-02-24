import re

def clean_text(text):
    """Clean and normalize text"""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    return text

def prepare_data_for_model(texts, tokenizer, max_length=256):
    """Prepare data for model input (batch processing)"""
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )
    return encoded["input_ids"], encoded["attention_masks"]