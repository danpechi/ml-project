from transformers import AutoModelForSequenceClassification


def get_model(model_name_or_path, device):
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path).to(device)
    return model
