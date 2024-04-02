import wandb
import time

def log_to_wandb(results, metrics):
    # Configure experiment
    experiment_name = "we should change this, probably being passed in"
    quantization_strategy = "we should change this, probably being passed in"
    config = {
        "experiment_name": experiment_name,
        "quantization_type": quantization_strategy,
    }

    # Initialize experiment
    wandb.init(project="ML_Final_Project_ODDJ", name=experiment_name, config=config)

    # Log results and additional metrics
    wandb.log(metrics)
    wandb.log({"runtime_seconds": metrics.get('runtime', 0)})

    # Finish the run
    wandb.finish()