from transformers import AutoTokenizer, RobertaForSequenceClassification
import torch

class PhoBERTModel:
    def __init__(self, model_path=None, num_labels=3):
        """Initialize PhoBERT model with option to load fine-tuned model"""
        if model_path is None:
            model_path = "vinai/phobert-base"  # Default pretrained model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
        self.model.eval()  # Set model to evaluation mode

    def predict(self, text):
        """Predict sentiment for input text"""
        if not text or not isinstance(text, str):
            return "unknown", 0.0, {"negative": 0.0, "neutral": 0.0, "positive": 0.0}

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze().tolist()

        sentiment_labels = ["negative", "neutral", "positive"]  # Match num_labels=3
        sentiment_scores = dict(zip(sentiment_labels, probabilities))

        predicted_class = max(sentiment_scores, key=sentiment_scores.get)
        confidence = sentiment_scores[predicted_class]

        return predicted_class, confidence, sentiment_scores