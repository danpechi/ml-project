from transformers import AutoTokenizer
def get_tokenizer(model_name_or_path="doyoungkim/bert-base-uncased-finetuned-sst2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return tokenizer