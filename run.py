from utils.load_data import get_data
from utils.load_model import get_model
from utils.load_tokenizer import get_tokenizer
from utils.logging import log_to_wandb
from utils.eval import evaluate_model
import torch


import argparse

def run_llm_on_data(llm, data, indices):


    results = []
    for ds in data:

        prompts = [i+ds for i in indices]
        response = llm.completions.create(
            prompt=prompts,
        )

        results.append(response)
    return results

def main():

    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('-q', '--quantization_strategy', type=str, help='quantization strategy', required=True)
    parser.add_argument('-mod', '--model', type=str, help='The llm to run', required=True)
    parser.add_argument('-ds', '--dataset', type=str, help='Data to run experiment on', required=True)
    # parser.add_argument('-m', '--metric', type=str, help='The metric used to evaluate', required=True)
    # parser.add_argument('-llm', '--llm', type=str, help='The llm to run', required=True)
    # parser.add_argument('-ip', '--index_prompt', type=str, help='The indices to query (rand/all/etc)', required=True)


    # Parse arguments
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #data should look something like a dictionary

    tokenizer = get_tokenizer(args.model)

    data = get_data(args.dataset, tokenizer)
    llm = get_model(args.model, device)
    llm.eval()

    # results = llm(data)
    #something to run the model on the data
    metrics = evaluate_model(llm, args.dataset, data, device)
    print(f"Evaluation result on {args.dataset}: {metrics}")
    # log_to_wandb(results, metrics)

if __name__ == "__main__":
    main()