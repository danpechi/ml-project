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
            quantized_model = quantize_model(model, lambda X: absmax_quantize(X, model_args.bits))
        elif model_args.quantization == 'zeropoint':
            quantized_model = quantize_model(model, lambda X: zeropoint_quantize(X, model_args.bits))
        elif model_args.quantization == 'norm':
            quantized_model = quantize_model(model, lambda X: norm_quantize(X, model_args.threshold))
        else:
            raise ValueError('Unknown quantization scheme')
        quantized_models[task_name] = quantized_model

    # Return quantized classification models
    return quantized_models

# Absmax quantize tensor 
def absmax_quantize(X, bits):
    # Check valid number of bits
    assert isinstance(bits, int)

    # Calculate scale on tensor
    half_max = 2 ** (bits - 1) - 1
    scale = half_max  / torch.max(torch.abs(X))

    # Quantize tensor
    X_quant = (scale * X).round()
    X_dequant = X_quant / scale

    # Return dequantized tensor
    return X_dequant

# Zero-point quantize tensor
def zeropoint_quantize(X, bits):
    # Compute value range (denominator)
    x_range = torch.max(X) - torch.min(X)
    x_range = 1 if x_range == 0 else x_range

    # Compute scale
    max_ = 2 ** bits - 1
    half_max = (max_ + 1) // 2
    scale = max_ / x_range

    # Shift by zero-point
    zeropoint = (-scale * torch.min(X) - half_max).round()

    # Scale and round the inputs
    X_quant = torch.clip((X * scale + zeropoint).round(), -half_max, half_max-1)

    # Dequantize tensor
    X_dequant = (X_quant - zeropoint) / scale

    # Return dequantized tensor
    return X_dequant

# Norm threshold tensor
def norm_quantize(X, threshold):
    # Check valid quantile
    assert isinstance(threshold, float)
    
    # Compute threshold mask
    mask = X.abs() >= threshold

    # Mask under threshold values
    X_thresh = X * mask

    # Return thresholded tensor
    return X_thresh

# Quantize parameters of model
def quantize_model(model, quantizer):
    # Quantize parameters of model 
    for param in model.parameters():
        quantized_param = quantizer(param.data)
        param.data = quantized_param

    # Return quantized model
    return model
