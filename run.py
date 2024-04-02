from utils.load_data import get_data
from utils.load_model import get_model
from utils.logging import log_to_wandb


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
    parser.add_argument('-model', '--model', type=str, help='The llm to run', required=True)
    # parser.add_argument('-dl', '--data_loc', type=str, help='Data to run experiment on', required=True)
    # parser.add_argument('-m', '--metric', type=str, help='The metric used to evaluate', required=True)
    # parser.add_argument('-llm', '--llm', type=str, help='The llm to run', required=True)
    # parser.add_argument('-ip', '--index_prompt', type=str, help='The indices to query (rand/all/etc)', required=True)


    # Parse arguments
    args = parser.parse_args()
    #data should look something like a dictionary
    data = get_data()
    llm = get_model(args.model)
    results = llm(data)
    #something to run the model on the data
    metrics = evaluate_model(llm, data, results)
    log_to_wandb(results, metrics)

if __name__ == "__main__":
    main()