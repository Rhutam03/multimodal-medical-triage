from transformers import AutoTokenizer


# Use a lightweight, production-friendly tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)


def preprocess_text(text):
    """
    Tokenizes symptom text for transformer input.
    """
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    return encoded
