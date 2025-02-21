def clean_text(text):
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)  # Loại bỏ ký tự đặc biệt
    text = re.sub(r"\s+", " ", text)  # Chuẩn hóa khoảng trắng
    return text

def tokenize_text(text, tokenizer):
    return tokenizer.tokenize(text)

def prepare_data_for_model(texts, tokenizer, max_length=512):
    input_ids = []
    attention_masks = []
    
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])
    
    return input_ids, attention_masks
