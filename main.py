# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch

# # Load model đã train
# model_path = "./sentiment_phobert"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForSequenceClassification.from_pretrained(model_path)

# def predict_sentiment(text):
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#     with torch.no_grad():
#         logits = model(**inputs).logits
#     probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
#     labels = ["negative", "neutral", "positive"]
#     predicted_class = probabilities.argmax().item()
    
#     return labels[predicted_class], probabilities.tolist()

# # Ví dụ dự đoán
# text = "Sản phẩm này thật tuyệt vời!"
# sentiment, scores = predict_sentiment(text)

# print(f"Text: {text}")
# print(f"Sentiment: {sentiment}")
# print(f"Scores: {scores}")
