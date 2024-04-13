import evaluate
import torch
def evaluate_model(model, ds_name, ds, device):
    metric = evaluate.load('glue', ds_name)

    for batch in ds:
        # Move batch to the same device as the model
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch['labels'])

    eval_result = metric.compute()
    return eval_result