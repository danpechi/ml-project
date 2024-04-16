# Import required packages
import torch

# Quantize classification models
def quantize_models(models, model_args):
    # Return unquantized classification models
    if model_args.quantization is None:
        return models

    # Quantize classification models
    quantized_models = {}
    for task_name, model in models.items():
        if model_args.quantization == 'absmax':
            quantized_model = quantize_model(model, absmax_quantize)
        else:
            raise ValueError('Unknown quantization scheme')
        quantized_models[task_name] = quantized_model

    # Return quantized classification models
    return quantized_models

# Absmax quantize tensor 
def absmax_quantize(X):
    # Calculate scale on tensor
    scale = 127 / torch.max(torch.abs(X))

    # Quantize tensor
    X_quant = (scale * X).round()
    X_dequant = X_quant / scale

    # Return dequantized tensor
    return X_dequant

# Quantize parameters of model
def quantize_model(model, quantizer):
    # Quantize parameters of model 
    for param in model.parameters():
        quantized_param = quantizer(param.data)
        param.data = quantized_param

    # Return quantized model
    return model
