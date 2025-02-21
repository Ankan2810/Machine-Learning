from transformers import AutoTokenizer, RobertaForSequenceClassification
import torch

class PhoBERTModel:
    def __init__(self, model_name="vinai/phobert-base", num_labels=3):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.eval()  # Đưa model vào chế độ đánh giá

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze().tolist()

        sentiment_labels = ["negative", "positive"]
        sentiment_scores = dict(zip(sentiment_labels, probabilities))

        predicted_class = max(sentiment_scores, key=sentiment_scores.get)
        confidence = sentiment_scores[predicted_class]

        return predicted_class, confidence, sentiment_scores
