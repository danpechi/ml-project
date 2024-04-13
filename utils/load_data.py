from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset


class SST2Dataset(Dataset):
    def __init__(self, tokenizer, dataset, max_length=128):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Fetch the sentence from the dataset
        sentence = self.dataset[idx]['sentence']
        # Tokenize the sentence
        inputs = self.tokenizer(sentence, max_length=self.max_length, truncation=True, padding='max_length',
                                return_tensors="pt")
        # Extract and return the input IDs and attention mask as tensors
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        # Extract the label
        label = torch.tensor(self.dataset[idx]['label'])
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label}


def get_data(glue_ds, tokenizer):
    raw_datasets = load_dataset("glue", glue_ds)
    eval_dataset = raw_datasets["validation"]

    # Create an instance of the custom dataset
    eval_dataset = SST2Dataset(tokenizer, eval_dataset)

    # Create the DataLoader
    eval_dataloader = DataLoader(eval_dataset, batch_size=32)

    return eval_dataloader
